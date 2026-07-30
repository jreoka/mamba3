#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int kThreads = 128;

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

template <int N, bool SaveStates>
__global__ __launch_bounds__(kThreads) void scan_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ dt,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    const float* __restrict__ D,
    const float* __restrict__ z,
    const float* __restrict__ initial_state,
    float* __restrict__ y,
    float* __restrict__ states,
    float* __restrict__ final_state,
    int batch,
    int length,
    int channels,
    int d_state,
    int channel_blocks) {
  const int batch_index = blockIdx.x / channel_blocks;
  const int channel = (blockIdx.x % channel_blocks) * blockDim.x + threadIdx.x;
  if (batch_index >= batch || channel >= channels) return;

  float state[N];
#pragma unroll
  for (int n = 0; n < N; ++n) {
    state[n] = n < d_state
        ? initial_state[(batch_index * channels + channel) * d_state + n]
        : 0.0f;
  }

  for (int time = 0; time < length; ++time) {
    const int x_offset = (batch_index * length + time) * channels + channel;
    const int bc_offset = (batch_index * length + time) * d_state;
    const float x_value = x[x_offset];
    const float dt_value = dt[x_offset];
    float base = D[channel] * x_value;
#pragma unroll
    for (int n = 0; n < N; ++n) {
      if (n < d_state) {
        const float decay = __expf(A[channel * d_state + n] * dt_value);
        state[n] = decay * state[n] + B[bc_offset + n] * x_value;
        base = fmaf(state[n], C[bc_offset + n], base);
        if constexpr (SaveStates) {
          states[((batch_index * length + time) * channels + channel) * d_state + n] = state[n];
        }
      }
    }
    const float z_value = z[x_offset];
    const float sigmoid = 1.0f / (1.0f + __expf(-z_value));
    y[x_offset] = base * z_value * sigmoid;
  }

#pragma unroll
  for (int n = 0; n < N; ++n) {
    if (n < d_state) {
      final_state[(batch_index * channels + channel) * d_state + n] = state[n];
    }
  }
}

template <int N>
__global__ __launch_bounds__(kThreads) void scan_backward_kernel(
    const float* __restrict__ grad_y,
    const float* __restrict__ grad_final_state,
    const float* __restrict__ x,
    const float* __restrict__ dt,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    const float* __restrict__ D,
    const float* __restrict__ z,
    const float* __restrict__ initial_state,
    const float* __restrict__ states,
    float* __restrict__ grad_x,
    float* __restrict__ grad_dt,
    float* __restrict__ grad_A,
    float* __restrict__ grad_B,
    float* __restrict__ grad_C,
    float* __restrict__ grad_D,
    float* __restrict__ grad_z,
    float* __restrict__ grad_initial_state,
    int batch,
    int length,
    int channels,
    int d_state,
    int channel_blocks) {
  const int batch_index = blockIdx.x / channel_blocks;
  const int channel = (blockIdx.x % channel_blocks) * blockDim.x + threadIdx.x;
  if (batch_index >= batch || channel >= channels) return;

  float grad_state[N];
  float grad_a_local[N];
#pragma unroll
  for (int n = 0; n < N; ++n) {
    grad_state[n] = n < d_state
        ? grad_final_state[(batch_index * channels + channel) * d_state + n]
        : 0.0f;
    grad_a_local[n] = 0.0f;
  }
  float grad_d_local = 0.0f;

  for (int time = length - 1; time >= 0; --time) {
    const int x_offset = (batch_index * length + time) * channels + channel;
    const int bc_offset = (batch_index * length + time) * d_state;
    const int state_offset = x_offset * d_state;
    const float x_value = x[x_offset];
    const float dt_value = dt[x_offset];
    const float z_value = z[x_offset];
    const float sigmoid = 1.0f / (1.0f + __expf(-z_value));
    const float gate = z_value * sigmoid;

    float base = D[channel] * x_value;
#pragma unroll
    for (int n = 0; n < N; ++n) {
      if (n < d_state) base = fmaf(states[state_offset + n], C[bc_offset + n], base);
    }
    const float upstream = grad_y[x_offset];
    const float grad_base = upstream * gate;
    grad_z[x_offset] = upstream * base * sigmoid * (1.0f + z_value * (1.0f - sigmoid));
    grad_d_local = fmaf(grad_base, x_value, grad_d_local);
    float gx = grad_base * D[channel];
    float gdt = 0.0f;

#pragma unroll
    for (int n = 0; n < N; ++n) {
      if (n < d_state) {
        const float current_state = states[state_offset + n];
        const float previous_state = time == 0
            ? initial_state[(batch_index * channels + channel) * d_state + n]
            : states[(x_offset - channels) * d_state + n];
        float gh = grad_state[n] + grad_base * C[bc_offset + n];
        const float reduced_c = warp_sum(grad_base * current_state);
        const float reduced_b = warp_sum(gh * x_value);
        if ((threadIdx.x & 31) == 0) {
          atomicAdd(grad_C + bc_offset + n, reduced_c);
          atomicAdd(grad_B + bc_offset + n, reduced_b);
        }
        gx = fmaf(gh, B[bc_offset + n], gx);

        const float decay = __expf(A[channel * d_state + n] * dt_value);
        const float grad_decay = gh * previous_state;
        grad_a_local[n] = fmaf(grad_decay * dt_value, decay, grad_a_local[n]);
        gdt = fmaf(grad_decay * A[channel * d_state + n], decay, gdt);
        grad_state[n] = gh * decay;
      }
    }
    grad_x[x_offset] = gx;
    grad_dt[x_offset] = gdt;
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

template <bool SaveStates>
void launch_forward(
    int d_state,
    dim3 blocks,
    cudaStream_t stream,
    const float* x,
    const float* dt,
    const float* A,
    const float* B,
    const float* C,
    const float* D,
    const float* z,
    const float* initial_state,
    float* y,
    float* states,
    float* final_state,
    int batch,
    int length,
    int channels,
    int channel_blocks) {
#define LAUNCH_FWD(N) scan_forward_kernel<N, SaveStates><<<blocks, kThreads, 0, stream>>>( \
    x, dt, A, B, C, D, z, initial_state, y, states, final_state, \
    batch, length, channels, d_state, channel_blocks)
  if (d_state <= 8) LAUNCH_FWD(8);
  else if (d_state <= 16) LAUNCH_FWD(16);
  else if (d_state <= 32) LAUNCH_FWD(32);
  else LAUNCH_FWD(64);
#undef LAUNCH_FWD
}

void launch_backward(
    int d_state,
    dim3 blocks,
    cudaStream_t stream,
    const std::vector<const float*>& inputs,
    const std::vector<float*>& outputs,
    int batch,
    int length,
    int channels,
    int channel_blocks) {
#define LAUNCH_BWD(N) scan_backward_kernel<N><<<blocks, kThreads, 0, stream>>>( \
    inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], \
    inputs[7], inputs[8], inputs[9], inputs[10], outputs[0], outputs[1], outputs[2], \
    outputs[3], outputs[4], outputs[5], outputs[6], outputs[7], batch, length, channels, \
    d_state, channel_blocks)
  if (d_state <= 8) LAUNCH_BWD(8);
  else if (d_state <= 16) LAUNCH_BWD(16);
  else if (d_state <= 32) LAUNCH_BWD(32);
  else LAUNCH_BWD(64);
#undef LAUNCH_BWD
}

}  // namespace

std::vector<torch::Tensor> mamba3_scan_forward_cuda(
    torch::Tensor x,
    torch::Tensor dt,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor D,
    torch::Tensor z,
    torch::Tensor initial_state,
    bool save_states) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  const int batch = static_cast<int>(x.size(0));
  const int length = static_cast<int>(x.size(1));
  const int channels = static_cast<int>(x.size(2));
  const int d_state = static_cast<int>(A.size(1));
  const int channel_blocks = (channels + kThreads - 1) / kThreads;
  const dim3 blocks(batch * channel_blocks);
  auto y = torch::empty_like(x);
  auto states = save_states
      ? torch::empty({batch, length, channels, d_state}, x.options())
      : torch::empty({0}, x.options());
  auto final_state = torch::empty({batch, channels, d_state}, x.options());
  float* states_ptr = save_states ? states.data_ptr<float>() : nullptr;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (save_states) {
    launch_forward<true>(d_state, blocks, stream, x.data_ptr<float>(), dt.data_ptr<float>(),
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), D.data_ptr<float>(),
        z.data_ptr<float>(), initial_state.data_ptr<float>(), y.data_ptr<float>(), states_ptr,
        final_state.data_ptr<float>(), batch, length, channels, channel_blocks);
  } else {
    launch_forward<false>(d_state, blocks, stream, x.data_ptr<float>(), dt.data_ptr<float>(),
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), D.data_ptr<float>(),
        z.data_ptr<float>(), initial_state.data_ptr<float>(), y.data_ptr<float>(), states_ptr,
        final_state.data_ptr<float>(), batch, length, channels, channel_blocks);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, states, final_state};
}

std::vector<torch::Tensor> mamba3_scan_backward_cuda(
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
    torch::Tensor states) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  const int batch = static_cast<int>(x.size(0));
  const int length = static_cast<int>(x.size(1));
  const int channels = static_cast<int>(x.size(2));
  const int d_state = static_cast<int>(A.size(1));
  const int channel_blocks = (channels + kThreads - 1) / kThreads;
  const dim3 blocks(batch * channel_blocks);

  auto grad_x = torch::empty_like(x);
  auto grad_dt = torch::empty_like(dt);
  auto grad_A = torch::zeros_like(A);
  auto grad_B = torch::zeros_like(B);
  auto grad_C = torch::zeros_like(C);
  auto grad_D = torch::zeros_like(D);
  auto grad_z = torch::empty_like(z);
  auto grad_initial_state = torch::empty_like(initial_state);

  std::vector<const float*> inputs = {
      grad_y.data_ptr<float>(), grad_final_state.data_ptr<float>(), x.data_ptr<float>(),
      dt.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
      D.data_ptr<float>(), z.data_ptr<float>(), initial_state.data_ptr<float>(),
      states.data_ptr<float>()};
  std::vector<float*> outputs = {
      grad_x.data_ptr<float>(), grad_dt.data_ptr<float>(), grad_A.data_ptr<float>(),
      grad_B.data_ptr<float>(), grad_C.data_ptr<float>(), grad_D.data_ptr<float>(),
      grad_z.data_ptr<float>(), grad_initial_state.data_ptr<float>()};
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_backward(d_state, blocks, stream, inputs, outputs, batch, length, channels, channel_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_x, grad_dt, grad_A, grad_B, grad_C, grad_D, grad_z, grad_initial_state};
}
