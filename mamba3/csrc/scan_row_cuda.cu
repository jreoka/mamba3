#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int kForwardThreads = 128;
constexpr int kBackwardThreads = 64;

template <typename scalar_t>
__global__ __launch_bounds__(256) void causal_conv_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int length,
    int channels,
    int64_t stride_batch,
    int64_t stride_time,
    int64_t stride_channel) {
  constexpr int kTile = 64;
  constexpr int kVec = 16 / sizeof(scalar_t);
  constexpr int kVectors = kTile * kTile / kVec;
  __shared__ scalar_t smem[kTile][kTile + kVec];

  const int time_base = blockIdx.x * kTile;
  const int channel_base = blockIdx.y * kTile;
  const int b = blockIdx.z;

  for (int i = threadIdx.x; i < kTile * kTile; i += blockDim.x) {
    const int local_time = i / kTile;
    const int local_channel = i % kTile;
    const int time = time_base + local_time;
    const int channel = channel_base + local_channel;
    scalar_t result = static_cast<scalar_t>(0.0f);
    if (time < length && channel < channels) {
      float value = static_cast<float>(bias[channel]);
#pragma unroll
      for (int tap = 0; tap < 4; ++tap) {
        const int source_time = time + tap - 3;
        if (source_time >= 0) {
          const int64_t source_offset = (int64_t)b * stride_batch +
              (int64_t)source_time * stride_time +
              (int64_t)channel * stride_channel;
          value = fmaf(
              static_cast<float>(x[source_offset]),
              static_cast<float>(weight[(int64_t)channel * 4 + tap]),
              value);
        }
      }
      const float sigmoid = 1.0f / (1.0f + __expf(-value));
      result = static_cast<scalar_t>(value * sigmoid);
    }
    smem[local_channel][local_time] = result;
  }
  __syncthreads();

  const bool vectorized =
      time_base + kTile <= length && channel_base + kTile <= channels &&
      ((int64_t)length * sizeof(scalar_t)) % 16 == 0;
  if (vectorized) {
    for (int v = threadIdx.x; v < kVectors; v += blockDim.x) {
      const int local_channel = v / (kTile / kVec);
      const int time_group = v % (kTile / kVec);
      uint4 packed;
      unsigned char* bytes = reinterpret_cast<unsigned char*>(&packed);
#pragma unroll
      for (int e = 0; e < kVec; ++e) {
        *reinterpret_cast<scalar_t*>(bytes + e * sizeof(scalar_t)) =
            smem[local_channel][time_group * kVec + e];
      }
      *reinterpret_cast<uint4*>(
          output + ((int64_t)b * channels + channel_base + local_channel) * length +
          time_base + time_group * kVec) = packed;
    }
  } else {
    for (int i = threadIdx.x; i < kTile * kTile; i += blockDim.x) {
      const int local_channel = i / kTile;
      const int local_time = i % kTile;
      const int channel = channel_base + local_channel;
      const int time = time_base + local_time;
      if (channel < channels && time < length) {
        output[((int64_t)b * channels + channel) * length + time] =
            smem[local_channel][local_time];
      }
    }
  }
}

template <typename scalar_t>
__global__ void causal_conv_step_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ state,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ next_state,
    int channels,
    int64_t total) {
  const int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int channel = static_cast<int>(index % channels);
  const int64_t state_offset = index * 3;
  const int64_t weight_offset = (int64_t)channel * 4;
  const float current = static_cast<float>(x[index]);
  float value = static_cast<float>(bias[channel]);
  value = fmaf(static_cast<float>(state[state_offset]),
      static_cast<float>(weight[weight_offset]), value);
  value = fmaf(static_cast<float>(state[state_offset + 1]),
      static_cast<float>(weight[weight_offset + 1]), value);
  value = fmaf(static_cast<float>(state[state_offset + 2]),
      static_cast<float>(weight[weight_offset + 2]), value);
  value = fmaf(current, static_cast<float>(weight[weight_offset + 3]), value);
  next_state[state_offset] = state[state_offset + 1];
  next_state[state_offset + 1] = state[state_offset + 2];
  next_state[state_offset + 2] = static_cast<scalar_t>(current);
  const float sigmoid = 1.0f / (1.0f + __expf(-value));
  output[index] = static_cast<scalar_t>(value * sigmoid);
}

template <typename scalar_t>
__global__ void scan_step_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ dt_logits,
    const scalar_t* __restrict__ dt_bias,
    const float* __restrict__ A,
    const scalar_t* __restrict__ B,
    const scalar_t* __restrict__ C,
    const float* __restrict__ D,
    const scalar_t* __restrict__ z,
    const float* __restrict__ state,
    scalar_t* __restrict__ output,
    float* __restrict__ next_state,
    int channels,
    int d_state,
    int64_t total) {
  const int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int channel = static_cast<int>(index % channels);
  const int batch = static_cast<int>(index / channels);
  const float x_value = static_cast<float>(x[index]);
  const float q = static_cast<float>(dt_logits[index]) + static_cast<float>(dt_bias[channel]);
  const float dt = fminf(q > 20.0f ? q : log1pf(__expf(q)), 1.0f);
  float base = D[channel] * x_value;
  const int64_t state_offset = index * d_state;
  const int64_t bc_offset = (int64_t)batch * d_state;
  for (int n = 0; n < d_state; ++n) {
    const float decay = __expf(A[(int64_t)channel * d_state + n] * dt);
    const float value = decay * state[state_offset + n] +
        static_cast<float>(B[bc_offset + n]) * x_value;
    next_state[state_offset + n] = value;
    base = fmaf(value, static_cast<float>(C[bc_offset + n]), base);
  }
  const float z_value = static_cast<float>(z[index]);
  const float sigmoid = 1.0f / (1.0f + __expf(-z_value));
  output[index] = static_cast<scalar_t>(base * z_value * sigmoid);
}

template <int N>
constexpr int checkpoint_stride() {
  // Keep the backward shared-memory working set at 32 KiB for every
  // specialization: threads * state * stride * sizeof(float).
  return 128 / N;
}

__device__ __forceinline__ float warp_sum(float value) {
  const unsigned mask = __activemask();
  const int active_lanes = __popc(mask);
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other = __shfl_down_sync(mask, value, offset);
    if (lane + offset < active_lanes) value += other;
  }
  return value;
}

template <typename scalar_t, int N, bool SaveCheckpoints, bool Reverse>
__global__ __launch_bounds__(kForwardThreads) void scan_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ dt,
    const float* __restrict__ A,
    const scalar_t* __restrict__ B,
    const scalar_t* __restrict__ C,
    const float* __restrict__ D,
    const scalar_t* __restrict__ z,
    const float* __restrict__ initial_state,
    scalar_t* __restrict__ y,
    float* __restrict__ state_checkpoints,
    float* __restrict__ final_state,
    int length,
    int channels,
    int d_state,
    int channel_blocks) {
  const int batch_index = blockIdx.x / channel_blocks;
  const int channel = (blockIdx.x % channel_blocks) * blockDim.x + threadIdx.x;
  if (channel >= channels) return;

  float state[N];
#pragma unroll
  for (int n = 0; n < N; ++n) {
    state[n] = n < d_state
        ? initial_state[(batch_index * channels + channel) * d_state + n]
        : 0.0f;
  }

  for (int time = 0; time < length; ++time) {
    const int physical_time = Reverse ? length - 1 - time : time;
    const int x_offset = (batch_index * length + physical_time) * channels + channel;
    const int bc_offset = (batch_index * length + physical_time) * d_state;
    if constexpr (SaveCheckpoints) {
      constexpr int kStride = checkpoint_stride<N>();
      if (time % kStride == 0) {
        const int checkpoint = time / kStride;
#pragma unroll
        for (int n = 0; n < N; ++n) {
          if (n < d_state) {
            state_checkpoints[
                ((batch_index * ((length + kStride - 1) / kStride) + checkpoint) *
                    channels + channel) * d_state + n] = state[n];
          }
        }
      }
    }
    const float x_value = static_cast<float>(x[x_offset]);
    const float dt_value = static_cast<float>(dt[x_offset]);
    float base = D[channel] * x_value;
#pragma unroll
    for (int n = 0; n < N; ++n) {
      if (n < d_state) {
        const float decay = __expf(A[channel * d_state + n] * dt_value);
        state[n] = decay * state[n] + static_cast<float>(B[bc_offset + n]) * x_value;
        base = fmaf(state[n], static_cast<float>(C[bc_offset + n]), base);
      }
    }
    const float z_value = static_cast<float>(z[x_offset]);
    const float sigmoid = 1.0f / (1.0f + __expf(-z_value));
    y[x_offset] = static_cast<scalar_t>(base * z_value * sigmoid);
  }

#pragma unroll
  for (int n = 0; n < N; ++n) {
    if (n < d_state) {
      final_state[(batch_index * channels + channel) * d_state + n] = state[n];
    }
  }
}

template <typename scalar_t, int N, bool Reverse>
__global__ __launch_bounds__(kBackwardThreads) void scan_backward_kernel(
    const scalar_t* __restrict__ grad_y,
    const float* __restrict__ grad_final_state,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ dt,
    const float* __restrict__ A,
    const scalar_t* __restrict__ B,
    const scalar_t* __restrict__ C,
    const float* __restrict__ D,
    const scalar_t* __restrict__ z,
    const float* __restrict__ state_checkpoints,
    scalar_t* __restrict__ grad_x,
    scalar_t* __restrict__ grad_dt,
    float* __restrict__ grad_A,
    float* __restrict__ grad_B,
    float* __restrict__ grad_C,
    float* __restrict__ grad_D,
    scalar_t* __restrict__ grad_z,
    float* __restrict__ grad_initial_state,
    int length,
    int channels,
    int d_state,
    int channel_blocks) {
  const int batch_index = blockIdx.x / channel_blocks;
  const int channel = (blockIdx.x % channel_blocks) * blockDim.x + threadIdx.x;
  if (channel >= channels) return;

  extern __shared__ float chunk_states[];
  constexpr int kStride = checkpoint_stride<N>();
  const int num_checkpoints = (length + kStride - 1) / kStride;
  float grad_state[N];
  float state[N];
  float grad_a_local[N];
#pragma unroll
  for (int n = 0; n < N; ++n) {
    grad_state[n] = n < d_state
        ? grad_final_state[(batch_index * channels + channel) * d_state + n]
        : 0.0f;
    state[n] = 0.0f;
    grad_a_local[n] = 0.0f;
  }
  float grad_d_local = 0.0f;

  for (int chunk = num_checkpoints - 1; chunk >= 0; --chunk) {
    const int chunk_start = chunk * kStride;
    const int chunk_end = min(length, chunk_start + kStride);

    // Restore a sparse checkpoint and replay this short chunk in shared memory.
#pragma unroll
    for (int n = 0; n < N; ++n) {
      if (n < d_state) {
        state[n] = state_checkpoints[
            ((batch_index * num_checkpoints + chunk) * channels + channel) *
                d_state + n];
      }
    }
    for (int time = chunk_start; time < chunk_end; ++time) {
      const int physical_time = Reverse ? length - 1 - time : time;
      const int x_offset = (batch_index * length + physical_time) * channels + channel;
      const int bc_offset = (batch_index * length + physical_time) * d_state;
      const float x_value = static_cast<float>(x[x_offset]);
      const float dt_value = static_cast<float>(dt[x_offset]);
#pragma unroll
      for (int n = 0; n < N; ++n) {
        if (n < d_state) {
          const float decay = __expf(A[channel * d_state + n] * dt_value);
          state[n] =
              decay * state[n] + static_cast<float>(B[bc_offset + n]) * x_value;
          const int local_time = time - chunk_start;
          chunk_states[(local_time * d_state + n) * blockDim.x + threadIdx.x] = state[n];
        }
      }
    }

    for (int time = chunk_end - 1; time >= chunk_start; --time) {
      const int physical_time = Reverse ? length - 1 - time : time;
      const int x_offset = (batch_index * length + physical_time) * channels + channel;
      const int bc_offset = (batch_index * length + physical_time) * d_state;
      const int local_time = time - chunk_start;
      const float x_value = static_cast<float>(x[x_offset]);
      const float dt_value = static_cast<float>(dt[x_offset]);
      const float z_value = static_cast<float>(z[x_offset]);
      const float sigmoid = 1.0f / (1.0f + __expf(-z_value));
      const float gate = z_value * sigmoid;

      float base = D[channel] * x_value;
#pragma unroll
      for (int n = 0; n < N; ++n) {
        if (n < d_state) {
          const float current_state =
              chunk_states[(local_time * d_state + n) * blockDim.x + threadIdx.x];
          base = fmaf(current_state, static_cast<float>(C[bc_offset + n]), base);
        }
      }
      const float upstream = static_cast<float>(grad_y[x_offset]);
      const float grad_base = upstream * gate;
      grad_z[x_offset] = static_cast<scalar_t>(
          upstream * base * sigmoid * (1.0f + z_value * (1.0f - sigmoid)));
      grad_d_local = fmaf(grad_base, x_value, grad_d_local);
      float gx = grad_base * D[channel];
      float gdt = 0.0f;

#pragma unroll
      for (int n = 0; n < N; ++n) {
        if (n < d_state) {
          const float current_state =
              chunk_states[(local_time * d_state + n) * blockDim.x + threadIdx.x];
          const float previous_state = local_time == 0
              ? state_checkpoints[
                    ((batch_index * num_checkpoints + chunk) * channels + channel) *
                        d_state + n]
              : chunk_states[((local_time - 1) * d_state + n) * blockDim.x + threadIdx.x];
          const float gh =
              grad_state[n] + grad_base * static_cast<float>(C[bc_offset + n]);
          const float reduced_c = warp_sum(grad_base * current_state);
          const float reduced_b = warp_sum(gh * x_value);
          if ((threadIdx.x & 31) == 0) {
            atomicAdd(grad_C + bc_offset + n, reduced_c);
            atomicAdd(grad_B + bc_offset + n, reduced_b);
          }
          gx = fmaf(gh, static_cast<float>(B[bc_offset + n]), gx);

          const float decay = __expf(A[channel * d_state + n] * dt_value);
          const float grad_decay = gh * previous_state;
          grad_a_local[n] = fmaf(grad_decay * dt_value, decay, grad_a_local[n]);
          gdt = fmaf(grad_decay * A[channel * d_state + n], decay, gdt);
          grad_state[n] = gh * decay;
        }
      }
      grad_x[x_offset] = static_cast<scalar_t>(gx);
      grad_dt[x_offset] = static_cast<scalar_t>(gdt);
    }
  }

  atomicAdd(grad_D + channel, grad_d_local);
#pragma unroll
  for (int n = 0; n < N; ++n) {
    if (n < d_state) {
      atomicAdd(grad_A + channel * d_state + n, grad_a_local[n]);
      grad_initial_state[(batch_index * channels + channel) * d_state + n] = grad_state[n];
    }
  }
}

template <typename scalar_t, bool SaveCheckpoints>
void launch_forward(
    int d_state,
    dim3 blocks,
    cudaStream_t stream,
    const scalar_t* x,
    const scalar_t* dt,
    const float* A,
    const scalar_t* B,
    const scalar_t* C,
    const float* D,
    const scalar_t* z,
    const float* initial_state,
    scalar_t* y,
    float* states,
    float* final_state,
    int length,
    int channels,
    int channel_blocks,
    bool reverse) {
#define LAUNCH_FWD(N, REVERSE) scan_forward_kernel<scalar_t, N, SaveCheckpoints, REVERSE> \
    <<<blocks, kForwardThreads, 0, stream>>>( \
    x, dt, A, B, C, D, z, initial_state, y, states, final_state, \
    length, channels, d_state, channel_blocks)
  if (reverse) {
    if (d_state <= 8) LAUNCH_FWD(8, true);
    else if (d_state <= 16) LAUNCH_FWD(16, true);
    else if (d_state <= 32) LAUNCH_FWD(32, true);
    else LAUNCH_FWD(64, true);
  } else {
    if (d_state <= 8) LAUNCH_FWD(8, false);
    else if (d_state <= 16) LAUNCH_FWD(16, false);
    else if (d_state <= 32) LAUNCH_FWD(32, false);
    else LAUNCH_FWD(64, false);
  }
#undef LAUNCH_FWD
}

template <typename scalar_t>
void launch_backward(
    int d_state,
    dim3 blocks,
    cudaStream_t stream,
    const scalar_t* grad_y,
    const float* grad_final_state,
    const scalar_t* x,
    const scalar_t* dt,
    const float* A,
    const scalar_t* B,
    const scalar_t* C,
    const float* D,
    const scalar_t* z,
    const float* state_checkpoints,
    scalar_t* grad_x,
    scalar_t* grad_dt,
    float* grad_A,
    float* grad_B,
    float* grad_C,
    float* grad_D,
    scalar_t* grad_z,
    float* grad_initial_state,
    int length,
    int channels,
    int channel_blocks,
    bool reverse) {
#define LAUNCH_BWD(N, STRIDE, REVERSE) scan_backward_kernel<scalar_t, N, REVERSE> \
    <<<blocks, kBackwardThreads, \
    kBackwardThreads * d_state * STRIDE * sizeof(float), stream>>>( \
    grad_y, grad_final_state, x, dt, A, B, C, D, z, state_checkpoints, \
    grad_x, grad_dt, grad_A, grad_B, grad_C, grad_D, grad_z, grad_initial_state, \
    length, channels, d_state, channel_blocks)
  if (reverse) {
    if (d_state <= 8) LAUNCH_BWD(8, 16, true);
    else if (d_state <= 16) LAUNCH_BWD(16, 8, true);
    else if (d_state <= 32) LAUNCH_BWD(32, 4, true);
    else LAUNCH_BWD(64, 2, true);
  } else {
    if (d_state <= 8) LAUNCH_BWD(8, 16, false);
    else if (d_state <= 16) LAUNCH_BWD(16, 8, false);
    else if (d_state <= 32) LAUNCH_BWD(32, 4, false);
    else LAUNCH_BWD(64, 2, false);
  }
#undef LAUNCH_BWD
}

}  // namespace

torch::Tensor mamba3_causal_conv_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  const int batch = static_cast<int>(x.size(0));
  const int length = static_cast<int>(x.size(1));
  const int channels = static_cast<int>(x.size(2));
  auto output = torch::empty({batch, channels, length}, x.options());
  if (output.numel() == 0) return output;
  const dim3 blocks((length + 63) / 64, (channels + 63) / 64, batch);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x.scalar_type(),
      "mamba3_causal_conv_forward_cuda",
      [&] {
        causal_conv_forward_kernel<scalar_t><<<blocks, 256, 0, stream>>>(
            x.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), length, channels, x.stride(0), x.stride(1),
            x.stride(2));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> mamba3_causal_conv_step_cuda(
    torch::Tensor x,
    torch::Tensor state,
    torch::Tensor weight,
    torch::Tensor bias) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  const int batch = static_cast<int>(x.size(0));
  const int channels = static_cast<int>(x.size(2));
  auto output = torch::empty_like(x);
  auto next_state = torch::empty_like(state);
  const int64_t total = (int64_t)batch * channels;
  if (total == 0) return {output, next_state};
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x.scalar_type(),
      "mamba3_causal_conv_step_cuda",
      [&] {
        causal_conv_step_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            x.data_ptr<scalar_t>(), state.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), next_state.data_ptr<scalar_t>(), channels, total);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, next_state};
}

std::vector<torch::Tensor> mamba3_scan_step_cuda(
    torch::Tensor x,
    torch::Tensor dt_logits,
    torch::Tensor dt_bias,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor D,
    torch::Tensor z,
    torch::Tensor state) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  const int batch = static_cast<int>(x.size(0));
  const int channels = static_cast<int>(x.size(2));
  const int d_state = static_cast<int>(A.size(1));
  auto output = torch::empty_like(x);
  auto next_state = torch::empty_like(state);
  const int64_t total = (int64_t)batch * channels;
  if (total == 0) return {output, next_state};
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x.scalar_type(),
      "mamba3_scan_step_cuda",
      [&] {
        scan_step_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            x.data_ptr<scalar_t>(), dt_logits.data_ptr<scalar_t>(),
            dt_bias.data_ptr<scalar_t>(), A.data_ptr<float>(), B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(), D.data_ptr<float>(), z.data_ptr<scalar_t>(),
            state.data_ptr<float>(), output.data_ptr<scalar_t>(),
            next_state.data_ptr<float>(), channels, d_state, total);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, next_state};
}

std::vector<torch::Tensor> mamba3_row_scan_forward_cuda(
    torch::Tensor x,
    torch::Tensor dt,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor D,
    torch::Tensor z,
    torch::Tensor initial_state,
    bool save_states,
    bool reverse) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  const int batch = static_cast<int>(x.size(0));
  const int length = static_cast<int>(x.size(1));
  const int channels = static_cast<int>(x.size(2));
  const int d_state = static_cast<int>(A.size(1));
  const int channel_blocks = (channels + kForwardThreads - 1) / kForwardThreads;
  const dim3 blocks(batch * channel_blocks);
  auto y = torch::empty_like(x);
  const int stride = d_state <= 8 ? 16 : d_state <= 16 ? 8 : d_state <= 32 ? 4 : 2;
  const int num_checkpoints = (length + stride - 1) / stride;
  const auto float_options = x.options().dtype(torch::kFloat32);
  auto states = save_states
      ? torch::empty({batch, num_checkpoints, channels, d_state}, float_options)
      : torch::empty({0}, float_options);
  auto final_state = torch::empty({batch, channels, d_state}, float_options);
  float* states_ptr = save_states ? states.data_ptr<float>() : nullptr;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x.scalar_type(),
      "mamba3_row_scan_forward_cuda",
      [&] {
        if (save_states) {
          launch_forward<scalar_t, true>(
              d_state, blocks, stream, x.data_ptr<scalar_t>(), dt.data_ptr<scalar_t>(),
              A.data_ptr<float>(), B.data_ptr<scalar_t>(), C.data_ptr<scalar_t>(),
               D.data_ptr<float>(), z.data_ptr<scalar_t>(), initial_state.data_ptr<float>(),
               y.data_ptr<scalar_t>(), states_ptr, final_state.data_ptr<float>(),
               length, channels, channel_blocks, reverse);
        } else {
          launch_forward<scalar_t, false>(
              d_state, blocks, stream, x.data_ptr<scalar_t>(), dt.data_ptr<scalar_t>(),
              A.data_ptr<float>(), B.data_ptr<scalar_t>(), C.data_ptr<scalar_t>(),
               D.data_ptr<float>(), z.data_ptr<scalar_t>(), initial_state.data_ptr<float>(),
               y.data_ptr<scalar_t>(), states_ptr, final_state.data_ptr<float>(),
               length, channels, channel_blocks, reverse);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, states, final_state};
}

std::vector<torch::Tensor> mamba3_row_scan_backward_cuda(
    torch::Tensor grad_y,
    torch::Tensor grad_final_state,
    torch::Tensor x,
    torch::Tensor dt,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor D,
    torch::Tensor z,
    torch::Tensor initial_state,
    torch::Tensor state_checkpoints,
    bool reverse) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  const int batch = static_cast<int>(x.size(0));
  const int length = static_cast<int>(x.size(1));
  const int channels = static_cast<int>(x.size(2));
  const int d_state = static_cast<int>(A.size(1));
  const int channel_blocks = (channels + kBackwardThreads - 1) / kBackwardThreads;
  const dim3 blocks(batch * channel_blocks);

  auto grad_x = torch::empty_like(x);
  auto grad_dt = torch::empty_like(dt);
  auto grad_A = torch::zeros_like(A);
  auto grad_B = torch::zeros(B.sizes(), A.options());
  auto grad_C = torch::zeros(C.sizes(), A.options());
  auto grad_D = torch::zeros_like(D);
  auto grad_z = torch::empty_like(z);
  auto grad_initial_state = torch::empty_like(initial_state);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x.scalar_type(),
      "mamba3_row_scan_backward_cuda",
      [&] {
        launch_backward<scalar_t>(
            d_state, blocks, stream, grad_y.data_ptr<scalar_t>(),
            grad_final_state.data_ptr<float>(), x.data_ptr<scalar_t>(),
            dt.data_ptr<scalar_t>(), A.data_ptr<float>(), B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(), D.data_ptr<float>(), z.data_ptr<scalar_t>(),
            state_checkpoints.data_ptr<float>(), grad_x.data_ptr<scalar_t>(),
            grad_dt.data_ptr<scalar_t>(),
            grad_A.data_ptr<float>(), grad_B.data_ptr<float>(), grad_C.data_ptr<float>(),
            grad_D.data_ptr<float>(), grad_z.data_ptr<scalar_t>(),
            grad_initial_state.data_ptr<float>(), length, channels, channel_blocks,
            reverse);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_x, grad_dt, grad_A, grad_B, grad_C, grad_D, grad_z, grad_initial_state};
}
