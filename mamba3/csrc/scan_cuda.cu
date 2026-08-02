#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int kThreads = 128;
constexpr int kItems = 16;
constexpr int kChunk = kThreads * kItems;  // 2048 sequence positions per chunk
constexpr int kMaxState = 64;
constexpr float kLog2E = 1.4426950408889634f;

// Shuffle helpers for float2 (no native float2 shuffle in CUDA).
__device__ __forceinline__ float2 shfl_up_f2(float2 v, unsigned off) {
  return make_float2(
      __shfl_up_sync(0xffffffffu, v.x, off),
      __shfl_up_sync(0xffffffffu, v.y, off));
}

__device__ __forceinline__ float2 shfl_down_f2(float2 v, unsigned off) {
  return make_float2(
      __shfl_down_sync(0xffffffffu, v.x, off),
      __shfl_down_sync(0xffffffffu, v.y, off));
}

// Forward scan combine: combine(a, b) -> (a.x * b.x, b.x * a.y + b.y).
// For scan elements (decay, input), this yields the accumulated state.
struct ScanCombine {
  __device__ __forceinline__ float2 operator()(const float2 &a, const float2 &b) const {
    return make_float2(a.x * b.x, b.x * a.y + b.y);
  }
};

// Reverse scan combine: a current element (decay of the next position, local
// adjoint source) combined with the suffix of later elements.
struct ReverseCombine {
  __device__ __forceinline__ float2 operator()(const float2 &element, const float2 &suffix) const {
    return make_float2(element.x * suffix.x, element.x * suffix.y + element.y);
  }
};

template <typename scalar_t>
__device__ __forceinline__ void load_items(
    const scalar_t *__restrict__ ptr,
    int64_t offset,
    bool vectorized,
    int chunk_remaining,
    float (&vals)[kItems]) {
  offset += (int64_t)threadIdx.x * kItems;
  if (vectorized) {
    const uint4 *p = reinterpret_cast<const uint4 *>(ptr + offset);
    constexpr int kVec = 16 / sizeof(scalar_t);
#pragma unroll
    for (int j = 0; j < kItems / kVec; ++j) {
      const uint4 v = p[j];
      const unsigned char *bytes = reinterpret_cast<const unsigned char *>(&v);
#pragma unroll
      for (int e = 0; e < kVec; ++e) {
        vals[j * kVec + e] =
            static_cast<float>(*reinterpret_cast<const scalar_t *>(bytes + e * sizeof(scalar_t)));
      }
    }
  } else {
    const int start = threadIdx.x * kItems;
    const int valid = min(kItems, max(0, chunk_remaining - start));
#pragma unroll
    for (int i = 0; i < kItems; ++i) {
      vals[i] = i < valid ? static_cast<float>(ptr[offset + i]) : 0.0f;
    }
  }
}

template <typename scalar_t>
__device__ __forceinline__ void store_items(
    scalar_t *__restrict__ ptr,
    int64_t offset,
    bool vectorized,
    int chunk_remaining,
    const float (&vals)[kItems]) {
  offset += (int64_t)threadIdx.x * kItems;
  if (vectorized) {
    uint4 *p = reinterpret_cast<uint4 *>(ptr + offset);
    constexpr int kVec = 16 / sizeof(scalar_t);
#pragma unroll
    for (int j = 0; j < kItems / kVec; ++j) {
      uint4 v;
      unsigned char *bytes = reinterpret_cast<unsigned char *>(&v);
#pragma unroll
      for (int e = 0; e < kVec; ++e) {
        *reinterpret_cast<scalar_t *>(bytes + e * sizeof(scalar_t)) =
            static_cast<scalar_t>(vals[j * kVec + e]);
      }
      p[j] = v;
    }
  } else {
    const int start = threadIdx.x * kItems;
    const int valid = min(kItems, max(0, chunk_remaining - start));
#pragma unroll
    for (int i = 0; i < kItems; ++i) {
      if (i < valid) ptr[offset + i] = static_cast<scalar_t>(vals[i]);
    }
  }
}

// Forward scan over the block's 16 items per thread, seeded with `prefix`.
// On return, items[i].y is the inclusive scan state; the block-wide aggregate
// (prefix applied) is returned by thread kThreads-1 through `last_item`.
template <typename Op>
__device__ __forceinline__ void block_scan(
    float2 (&items)[kItems],
    const float2 &prefix,
    float2 *smem_warp,
    Op op) {
  float2 local[kItems];
  float2 acc = items[0];
  local[0] = acc;
#pragma unroll
  for (int i = 1; i < kItems; ++i) {
    acc = op(acc, items[i]);
    local[i] = acc;
  }
  // Warp-level forward scan over thread aggregates.
  float2 wacc = acc;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const float2 other = shfl_up_f2(wacc, off);
    if ((threadIdx.x & 31) >= off) wacc = op(other, wacc);
  }
  // The inclusive warp scan ends at lane 31, which holds the full warp
  // aggregate used by the cross-warp chain.
  if ((threadIdx.x & 31) == 31) smem_warp[threadIdx.x >> 5] = wacc;
  __syncthreads();
  const int warp = threadIdx.x >> 5;
  // Prefix of the earlier threads within this warp (warp scan is inclusive).
  // All lanes must participate in the shuffle, so lane 0's result is
  // overwritten afterward.
  float2 wprefix = shfl_up_f2(wacc, 1);
  if ((threadIdx.x & 31) == 0) wprefix = make_float2(1.0f, 0.0f);
  float2 pre = prefix;
#pragma unroll
  for (int j = 0; j < warp; ++j) pre = op(pre, smem_warp[j]);
  pre = op(pre, wprefix);
#pragma unroll
  for (int i = 0; i < kItems; ++i) items[i] = op(pre, local[i]);
  // The caller reuses smem_warp for the next state's scan; make sure every
  // thread finished reading before anyone writes again.
  __syncthreads();
}

// Reverse scan over the block's 16 items per thread, seeded with `postfix`
// (the adjoint chain from later chunks). Items hold (decay, local source) pairs
// and on return items[i].y is the adjoint of the state at position i.
template <typename Op>
__device__ __forceinline__ void block_reverse_scan(
    float2 (&items)[kItems],
    const float2 &postfix,
    float2 *smem_warp,
    Op op) {
  float2 rev_local[kItems];
  float2 acc = items[kItems - 1];
  rev_local[kItems - 1] = acc;
#pragma unroll
  for (int i = kItems - 2; i >= 0; --i) {
    acc = op(items[i], acc);
    rev_local[i] = acc;
  }
  // Warp-level reverse scan over thread aggregates; lane 0 holds the warp
  // aggregate (the chunk segment combined from first to last position).
  float2 wacc = acc;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const float2 other = shfl_down_f2(wacc, off);
    if ((threadIdx.x & 31) + off < 32) wacc = op(wacc, other);
  }
  if ((threadIdx.x & 31) == 0) smem_warp[threadIdx.x >> 5] = wacc;
  __syncthreads();
  const int warp = threadIdx.x >> 5;
  // Suffix of the later threads within this warp (warp reverse scan is
  // inclusive, so the exclusive suffix is the neighbor lane's result).
  float2 wsuffix = shfl_down_f2(wacc, 1);
  if ((threadIdx.x & 31) == 31) wsuffix = make_float2(1.0f, 0.0f);
  // Suffix of later thread aggregates combined with the postfix.
  float2 suff = postfix;
#pragma unroll
  for (int j = 3; j > warp; --j) suff = op(smem_warp[j], suff);
  suff = op(wsuffix, suff);
#pragma unroll
  for (int i = 0; i < kItems; ++i) items[i] = op(rev_local[i], suff);
  // The caller reuses smem_warp for the next state's scan; make sure every
  // thread finished reading before anyone writes again.
  __syncthreads();
}

template <typename scalar_t>
__global__ __launch_bounds__(kThreads, 3) void scan_forward_kernel(
    const scalar_t *__restrict__ u,
    const scalar_t *__restrict__ delta,
    const float *__restrict__ A,
    const scalar_t *__restrict__ B,
    const scalar_t *__restrict__ C,
    const float *__restrict__ D,
    const scalar_t *__restrict__ z,
    const float *__restrict__ initial_state,
    scalar_t *__restrict__ y,
    float *__restrict__ states,
    float *__restrict__ final_state,
    int length,
    int d_state,
    int n_chunks,
    int save_states) {
  const int b = blockIdx.x;
  const int h = blockIdx.y;
  const int H = gridDim.y;

  __shared__ float2 smem_prefix[kMaxState];
  __shared__ float2 smem_warp[4];

  if (threadIdx.x == 0) {
#pragma unroll
    for (int n = 0; n < kMaxState; ++n) {
      if (n < d_state) {
        smem_prefix[n] =
            make_float2(1.0f, initial_state[((int64_t)b * H + h) * d_state + n]);
      }
    }
  }
  __syncthreads();

  const float D_val = D[h];
  const int64_t row_start = (int64_t)b * H + h;
  const bool row_aligned =
      (row_start * length * (int64_t)sizeof(scalar_t)) % 16 == 0;
  const bool bc_aligned =
      (((int64_t)b * d_state) * length * (int64_t)sizeof(scalar_t)) % 16 == 0;

  for (int chunk = 0; chunk < n_chunks; ++chunk) {
    const int base = chunk * kChunk;
    const int remaining = length - base;
    const bool full = remaining >= kChunk;
    const bool vectorized = full && row_aligned;
    const int64_t row_base = row_start * length + base;

    float u_vals[kItems], delta_vals[kItems], z_vals[kItems];
    load_items<scalar_t>(u, row_base, vectorized, remaining, u_vals);
    load_items<scalar_t>(delta, row_base, vectorized, remaining, delta_vals);
    load_items<scalar_t>(z, row_base, vectorized, remaining, z_vals);

    float out_vals[kItems];
#pragma unroll
    for (int i = 0; i < kItems; ++i) out_vals[i] = D_val * u_vals[i];

    for (int state = 0; state < d_state; ++state) {
      const float A_scaled = A[h * d_state + state] * kLog2E;
      const bool state_aligned =
          full && row_aligned && bc_aligned && (((int64_t)state * length * (int64_t)sizeof(scalar_t)) % 16 == 0);
      const int64_t bc_base =
          ((int64_t)b * d_state + state) * length + base;
      float B_vals[kItems], C_vals[kItems];
      load_items<scalar_t>(B, bc_base, state_aligned, remaining, B_vals);
      load_items<scalar_t>(C, bc_base, state_aligned, remaining, C_vals);

      float2 items[kItems];
#pragma unroll
      for (int i = 0; i < kItems; ++i) {
        const float decay = exp2f(delta_vals[i] * A_scaled);
        items[i] = make_float2(decay, u_vals[i] * B_vals[i]);
      }
      if (!full) {
#pragma unroll
        for (int i = 0; i < kItems; ++i) {
          if (base + threadIdx.x * kItems + i >= length) {
            items[i] = make_float2(1.0f, 0.0f);
          }
        }
      }
      const float2 prefix = smem_prefix[state];
      block_scan(items, prefix, smem_warp, ScanCombine());
#pragma unroll
      for (int i = 0; i < kItems; ++i) {
        out_vals[i] = fmaf(items[i].y, C_vals[i], out_vals[i]);
      }
      if (threadIdx.x == kThreads - 1) {
        smem_prefix[state] = items[kItems - 1];
        if (save_states) {
          float *ck =
              states + (((int64_t)b * H + h) * n_chunks + chunk) * (2 * d_state) + 2 * state;
          ck[0] = items[kItems - 1].x;
          ck[1] = items[kItems - 1].y;
        }
      }
    }

    float y_vals[kItems];
#pragma unroll
    for (int i = 0; i < kItems; ++i) {
      const float z_val = z_vals[i];
      const float sigmoid = 1.0f / (1.0f + __expf(-z_val));
      y_vals[i] = out_vals[i] * z_val * sigmoid;
    }
    store_items<scalar_t>(y, row_base, vectorized, remaining, y_vals);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int n = 0; n < kMaxState; ++n) {
      if (n < d_state) {
        final_state[((int64_t)b * H + h) * d_state + n] = smem_prefix[n].y;
      }
    }
  }
}


// Pass A: all gradients except grad_B/grad_C (which are computed by
// scan_backward_bc_kernel with a cross-channel reduction to avoid H-way
// atomic contention).
template <typename scalar_t>
__global__ __launch_bounds__(kThreads, 2) void scan_backward_kernel(
    const scalar_t *__restrict__ grad_y,
    const float *__restrict__ grad_final_state,
    const scalar_t *__restrict__ u,
    const scalar_t *__restrict__ delta,
    const float *__restrict__ A,
    const scalar_t *__restrict__ B,
    const scalar_t *__restrict__ C,
    const float *__restrict__ D,
    const scalar_t *__restrict__ z,
    const float *__restrict__ initial_state,
    const float *__restrict__ states,
    scalar_t *__restrict__ grad_u,
    scalar_t *__restrict__ grad_delta,
    float *__restrict__ grad_A,
    float *__restrict__ grad_D,
    scalar_t *__restrict__ grad_z,
    float *__restrict__ grad_initial_state,
    int length,
    int d_state,
    int n_chunks) {
  const int b = blockIdx.x;
  const int h = blockIdx.y;
  const int H = gridDim.y;

  __shared__ float2 smem_postfix[kMaxState];
  __shared__ float2 smem_warp[4];
  __shared__ float smem_first_alpha[kThreads];
  __shared__ float smem_boundary_alpha[kMaxState];
  __shared__ float smem_reduce[4];

  if (threadIdx.x == 0) {
#pragma unroll
    for (int n = 0; n < kMaxState; ++n) {
      if (n < d_state) {
        smem_postfix[n] =
            make_float2(1.0f, grad_final_state[((int64_t)b * H + h) * d_state + n]);
      }
    }
  }
  __syncthreads();

  const float D_val = D[h];
  const int64_t row_start = (int64_t)b * H + h;
  const bool row_aligned =
      (row_start * length * (int64_t)sizeof(scalar_t)) % 16 == 0;
  const bool bc_aligned =
      (((int64_t)b * d_state) * length * (int64_t)sizeof(scalar_t)) % 16 == 0;
  float dD_local = 0.0f;

  for (int chunk = n_chunks - 1; chunk >= 0; --chunk) {
    const int base = chunk * kChunk;
    const int remaining = length - base;
    const bool full = remaining >= kChunk;
    const bool vectorized = full && row_aligned;
    const int64_t row_base = row_start * length + base;

    float u_vals[kItems], delta_vals[kItems], z_vals[kItems], dout_vals[kItems];
    load_items<scalar_t>(u, row_base, vectorized, remaining, u_vals);
    load_items<scalar_t>(delta, row_base, vectorized, remaining, delta_vals);
    load_items<scalar_t>(z, row_base, vectorized, remaining, z_vals);
    load_items<scalar_t>(grad_y, row_base, vectorized, remaining, dout_vals);

    float dz_pre[kItems];
#pragma unroll
    for (int i = 0; i < kItems; ++i) {
      const float z_val = z_vals[i];
      const float sigmoid = 1.0f / (1.0f + __expf(-z_val));
      dz_pre[i] = dout_vals[i] * sigmoid * (1.0f + z_val * (1.0f - sigmoid));
      dout_vals[i] *= z_val * sigmoid;
    }
    float du_vals[kItems], ddelta_vals[kItems], base_vals[kItems];
#pragma unroll
    for (int i = 0; i < kItems; ++i) {
      du_vals[i] = D_val * dout_vals[i];
      ddelta_vals[i] = 0.0f;
      base_vals[i] = D_val * u_vals[i];
    }

    for (int state = 0; state < d_state; ++state) {
      const float A_raw = A[h * d_state + state];
      const float A_scaled = A_raw * kLog2E;
      const float2 prefix =
          chunk == 0
              ? make_float2(1.0f, initial_state[((int64_t)b * H + h) * d_state + state])
              : *reinterpret_cast<const float2 *>(
                    states + (((int64_t)b * H + h) * n_chunks + chunk - 1) * (2 * d_state) +
                    2 * state);
      const float2 postfix = smem_postfix[state];

      const bool state_aligned =
          full && row_aligned && bc_aligned && (((int64_t)state * length * (int64_t)sizeof(scalar_t)) % 16 == 0);
      const int64_t bc_base = ((int64_t)b * d_state + state) * length + base;
      float B_vals[kItems], C_vals[kItems];
      load_items<scalar_t>(B, bc_base, state_aligned, remaining, B_vals);
      load_items<scalar_t>(C, bc_base, state_aligned, remaining, C_vals);

      // Recompute the chunk recurrence to recover the per-position states.
      // Note: this repository's recurrence is s_t = exp(dt * A) * s_{t-1} +
      // B * u (the dt factor only scales the decay, not the input).
      float2 items[kItems];
#pragma unroll
      for (int i = 0; i < kItems; ++i) {
        const float decay = exp2f(delta_vals[i] * A_scaled);
        items[i] = make_float2(decay, u_vals[i] * B_vals[i]);
      }
      if (!full) {
#pragma unroll
        for (int i = 0; i < kItems; ++i) {
          if (base + threadIdx.x * kItems + i >= length) {
            items[i] = make_float2(1.0f, 0.0f);
          }
        }
      }
      block_scan(items, prefix, smem_warp, ScanCombine());

      // Reverse-scan elements: (decay of the next position, local adjoint
      // source dout * C). The per-position decays are recomputed rather than
      // kept in registers to hold the register footprint down.
      smem_first_alpha[threadIdx.x] = exp2f(delta_vals[0] * A_scaled);
      __syncthreads();
      float2 reverse_items[kItems];
#pragma unroll
      for (int i = 0; i < kItems - 1; ++i) {
        reverse_items[i] =
            make_float2(exp2f(delta_vals[i + 1] * A_scaled), dout_vals[i] * C_vals[i]);
      }
      reverse_items[kItems - 1] = make_float2(
          threadIdx.x == kThreads - 1
              ? (chunk == n_chunks - 1 ? 1.0f : smem_boundary_alpha[state])
              : smem_first_alpha[threadIdx.x + 1],
          dout_vals[kItems - 1] * C_vals[kItems - 1]);
      if (!full) {
#pragma unroll
        for (int i = 0; i < kItems; ++i) {
          if (base + threadIdx.x * kItems + i >= length) {
            reverse_items[i] = make_float2(1.0f, 0.0f);
          }
        }
      }
      block_reverse_scan(reverse_items, postfix, smem_warp, ReverseCombine());


      if (threadIdx.x == 0) {
        float2 agg = smem_warp[0];
#pragma unroll
        for (int j = 1; j < 4; ++j) agg = ReverseCombine()(agg, smem_warp[j]);
        smem_postfix[state] = ScanCombine()(postfix, agg);
        // Written after the reverse scan so the current chunk's read still
        // sees the previous chunk's first decay (this chunk's own value is
        // consumed by the next chunk).
        smem_boundary_alpha[state] = exp2f(delta_vals[0] * A_scaled);
      }

      // The aggregate read above races with the next state's smem_warp
      // writes, so order them before continuing.
      __syncthreads();

      float dA_state = 0.0f;
#pragma unroll
      for (int i = 0; i < kItems; ++i) {
        const float lambda = reverse_items[i].y;
        const float s_i = items[i].y;
        const float input_i = u_vals[i] * B_vals[i];
        const float a = s_i - input_i;
        du_vals[i] = fmaf(lambda, B_vals[i], du_vals[i]);
        ddelta_vals[i] = fmaf(lambda, A_raw * a, ddelta_vals[i]);
        dA_state = fmaf(lambda, delta_vals[i] * a, dA_state);
        base_vals[i] = fmaf(C_vals[i], s_i, base_vals[i]);
      }
      if (state == 0) {
#pragma unroll
        for (int i = 0; i < kItems; ++i) {
          dD_local = fmaf(dout_vals[i], u_vals[i], dD_local);
        }
      }

#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        dA_state += __shfl_xor_sync(0xffffffffu, dA_state, off);
      }
      if ((threadIdx.x & 31) == 0) smem_reduce[threadIdx.x >> 5] = dA_state;
      __syncthreads();
      if (threadIdx.x == 0) {
        float total = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) total += smem_reduce[j];
        atomicAdd(grad_A + h * d_state + state, total);
        if (chunk == 0) {
          grad_initial_state[((int64_t)b * H + h) * d_state + state] =
              items[0].x * reverse_items[0].y;
        }
      }
    }

    float dz_vals[kItems];
#pragma unroll
    for (int i = 0; i < kItems; ++i) dz_vals[i] = dz_pre[i] * base_vals[i];
    store_items<scalar_t>(grad_u, row_base, vectorized, remaining, du_vals);
    store_items<scalar_t>(grad_delta, row_base, vectorized, remaining, ddelta_vals);
    store_items<scalar_t>(grad_z, row_base, vectorized, remaining, dz_vals);
  }

#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    dD_local += __shfl_xor_sync(0xffffffffu, dD_local, off);
  }
  if ((threadIdx.x & 31) == 0) smem_reduce[threadIdx.x >> 5] = dD_local;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) total += smem_reduce[j];
    atomicAdd(grad_D + h, total);
  }
}

// Fast transposes between the [B, X, Y] and [B, Y, X] layouts used by the
// kernels. Tiles are staged through shared memory so both the loads and the
// stores are 16-byte vectorized; torch's generic transpose path is much slower
// for these 2-byte activations.
template <typename scalar_t>
__global__ __launch_bounds__(256) void transpose_copy_kernel(
    const scalar_t *__restrict__ src,
    scalar_t *__restrict__ dst,
    int batch,
    int x,
    int y) {
  constexpr int kTile = 64;
  constexpr int kVec = 16 / sizeof(scalar_t);
  constexpr int kThreads = 256;
  constexpr int kVectors = kTile * kTile / kVec;  // 1024

  // Transposed staging layout [source col][source row] so that the destination
  // vectors (consecutive source rows) read contiguously.
  __shared__ scalar_t smem[kTile][kTile + kVec];

  const int tile_x = blockIdx.x * kTile;
  const int tile_y = blockIdx.y * kTile;
  const int b = blockIdx.z;

  // Vectorized path requires 16-byte-aligned rows on both sides.
  const bool aligned = ((int64_t)y * (int64_t)sizeof(scalar_t)) % 16 == 0 &&
      ((int64_t)x * (int64_t)sizeof(scalar_t)) % 16 == 0;

  if (aligned && tile_x + kTile <= x && tile_y + kTile <= y) {
    // Load: source row r, source col group cg (kVec elements starting at 4*cg).
#pragma unroll
    for (int v = threadIdx.x; v < kVectors; v += kThreads) {
      const int r = v / (kTile / kVec);
      const int cg = v % (kTile / kVec);
      const uint4 val = *reinterpret_cast<const uint4 *>(
          src + ((int64_t)b * x + tile_x + r) * y + tile_y + cg * kVec);
      const unsigned char *bytes = reinterpret_cast<const unsigned char *>(&val);
#pragma unroll
      for (int e = 0; e < kVec; ++e) {
        smem[cg * kVec + e][r] =
            *reinterpret_cast<const scalar_t *>(bytes + e * sizeof(scalar_t));
      }
    }
    __syncthreads();
    // Store: destination row = source col c, destination col group = source
    // row group rg.
#pragma unroll
    for (int v = threadIdx.x; v < kVectors; v += kThreads) {
      const int c = v / (kTile / kVec);
      const int rg = v % (kTile / kVec);
      uint4 val;
      unsigned char *bytes = reinterpret_cast<unsigned char *>(&val);
#pragma unroll
      for (int e = 0; e < kVec; ++e) {
        *reinterpret_cast<scalar_t *>(bytes + e * sizeof(scalar_t)) =
            smem[c][rg * kVec + e];
      }
      *reinterpret_cast<uint4 *>(dst + ((int64_t)b * y + tile_y + c) * x + tile_x + rg * kVec) =
          val;
    }
  } else {
    for (int i = threadIdx.x; i < kTile * kTile; i += kThreads) {
      const int row = i / kTile;
      const int col = i % kTile;
      const int sx = tile_x + row;
      const int sy = tile_y + col;
      if (sx < x && sy < y) {
        dst[(int64_t)(b * y + sy) * x + sx] = src[(int64_t)(b * x + sx) * y + sy];
      }
    }
  }
}

template <typename scalar_t>
void launch_forward(
    int length,
    int d_state,
    int n_chunks,
    dim3 grid,
    cudaStream_t stream,
    const scalar_t *x,
    const scalar_t *dt,
    const float *A,
    const scalar_t *B,
    const scalar_t *C,
    const float *D,
    const scalar_t *z,
    const float *initial_state,
    scalar_t *y,
    float *states,
    float *final_state,
    int save_states) {
  scan_forward_kernel<scalar_t><<<grid, kThreads, 0, stream>>>(
      x, dt, A, B, C, D, z, initial_state, y, states, final_state, length, d_state,
      n_chunks, save_states);
}

// Pass B: grad_B and grad_C only. Each block owns one (batch, state) row and
// reduces contributions from kChanGroup channels in registers before a single
// atomic add per sequence position, cutting the atomic traffic by kChanGroup.
constexpr int kChanGroup = 32;

template <typename scalar_t>
__global__ __launch_bounds__(kThreads, 2) void scan_backward_bc_kernel(
    const scalar_t *__restrict__ grad_y,
    const float *__restrict__ grad_final_state,
    const scalar_t *__restrict__ u,
    const scalar_t *__restrict__ delta,
    const float *__restrict__ A,
    const scalar_t *__restrict__ B,
    const scalar_t *__restrict__ C,
    const scalar_t *__restrict__ z,
    const float *__restrict__ initial_state,
    const float *__restrict__ states,
    float *__restrict__ grad_B,
    float *__restrict__ grad_C,
    int length,
    int d_state,
    int n_chunks,
    int channels) {
  const int b = blockIdx.x;
  const int state = blockIdx.y;
  const int h_group = blockIdx.z;
  const int h_base = h_group * kChanGroup;

  __shared__ float2 smem_postfix[kChanGroup];
  __shared__ float2 smem_warp[4];
  __shared__ float smem_first_alpha[kThreads];
  __shared__ float smem_boundary_alpha[kChanGroup];

  if (threadIdx.x == 0) {
    for (int hh = 0; hh < kChanGroup; ++hh) {
      const int h = h_base + hh;
      if (h < channels) {
        smem_postfix[hh] = make_float2(
            1.0f, grad_final_state[((int64_t)b * channels + h) * d_state + state]);
      }
    }
  }
  __syncthreads();

  for (int chunk = n_chunks - 1; chunk >= 0; --chunk) {
    const int base = chunk * kChunk;
    const int remaining = length - base;
    const bool full = remaining >= kChunk;
    const bool bc_aligned =
        (((int64_t)b * d_state + state) * length * (int64_t)sizeof(scalar_t)) % 16 == 0;
    const int64_t bc_base = ((int64_t)b * d_state + state) * length + base;

    float B_vals[kItems], C_vals[kItems];
    load_items<scalar_t>(B, bc_base, full && bc_aligned, remaining, B_vals);
    load_items<scalar_t>(C, bc_base, full && bc_aligned, remaining, C_vals);

    float dB_acc[kItems], dC_acc[kItems];
#pragma unroll
    for (int i = 0; i < kItems; ++i) {
      dB_acc[i] = 0.0f;
      dC_acc[i] = 0.0f;
    }

    for (int hh = 0; hh < kChanGroup; ++hh) {
      const int h = h_base + hh;
      if (h >= channels) continue;

      const int64_t row_start = (int64_t)b * channels + h;
      const bool row_aligned =
          (row_start * length * (int64_t)sizeof(scalar_t)) % 16 == 0;
      const bool vectorized = full && row_aligned;
      const int64_t row_base = row_start * length + base;

      float u_vals[kItems], delta_vals[kItems], z_vals[kItems], dout_vals[kItems];
      load_items<scalar_t>(u, row_base, vectorized, remaining, u_vals);
      load_items<scalar_t>(delta, row_base, vectorized, remaining, delta_vals);
      load_items<scalar_t>(z, row_base, vectorized, remaining, z_vals);
      load_items<scalar_t>(grad_y, row_base, vectorized, remaining, dout_vals);
#pragma unroll
      for (int i = 0; i < kItems; ++i) {
        const float z_val = z_vals[i];
        dout_vals[i] *= z_val / (1.0f + __expf(-z_val));
      }

      const float A_scaled = A[h * d_state + state] * kLog2E;
      const float2 prefix =
          chunk == 0
              ? make_float2(1.0f, initial_state[((int64_t)b * channels + h) * d_state + state])
              : *reinterpret_cast<const float2 *>(
                    states + (((int64_t)b * channels + h) * n_chunks + chunk - 1) *
                        (2 * d_state) +
                    2 * state);
      const float2 postfix = smem_postfix[hh];

      float2 items[kItems];
#pragma unroll
      for (int i = 0; i < kItems; ++i) {
        const float decay = exp2f(delta_vals[i] * A_scaled);
        items[i] = make_float2(decay, u_vals[i] * B_vals[i]);
      }
      if (!full) {
#pragma unroll
        for (int i = 0; i < kItems; ++i) {
          if (base + threadIdx.x * kItems + i >= length) {
            items[i] = make_float2(1.0f, 0.0f);
          }
        }
      }
      block_scan(items, prefix, smem_warp, ScanCombine());

      smem_first_alpha[threadIdx.x] = exp2f(delta_vals[0] * A_scaled);
      __syncthreads();
      float2 reverse_items[kItems];
#pragma unroll
      for (int i = 0; i < kItems - 1; ++i) {
        reverse_items[i] =
            make_float2(exp2f(delta_vals[i + 1] * A_scaled), dout_vals[i] * C_vals[i]);
      }
      reverse_items[kItems - 1] = make_float2(
          threadIdx.x == kThreads - 1
              ? (chunk == n_chunks - 1 ? 1.0f : smem_boundary_alpha[hh])
              : smem_first_alpha[threadIdx.x + 1],
          dout_vals[kItems - 1] * C_vals[kItems - 1]);
      if (!full) {
#pragma unroll
        for (int i = 0; i < kItems; ++i) {
          if (base + threadIdx.x * kItems + i >= length) {
            reverse_items[i] = make_float2(1.0f, 0.0f);
          }
        }
      }
      block_reverse_scan(reverse_items, postfix, smem_warp, ReverseCombine());

      if (threadIdx.x == 0) {
        float2 agg = smem_warp[0];
#pragma unroll
        for (int j = 1; j < 4; ++j) agg = ReverseCombine()(agg, smem_warp[j]);
        smem_postfix[hh] = ScanCombine()(postfix, agg);
        // Written after the reverse scan so the current chunk's read still
        // sees the previous chunk's first decay.
        smem_boundary_alpha[hh] = exp2f(delta_vals[0] * A_scaled);
      }
      // The aggregate read above races with the next iteration's smem_warp
      // writes, so order them before continuing.
      __syncthreads();

#pragma unroll
      for (int i = 0; i < kItems; ++i) {
        if (base + threadIdx.x * kItems + i < length) {
          dB_acc[i] = fmaf(reverse_items[i].y, u_vals[i], dB_acc[i]);
          dC_acc[i] = fmaf(dout_vals[i], items[i].y, dC_acc[i]);
        }
      }
    }

#pragma unroll
    for (int i = 0; i < kItems; ++i) {
      if (base + threadIdx.x * kItems + i < length) {
        atomicAdd(grad_B + bc_base + threadIdx.x * kItems + i, dB_acc[i]);
        atomicAdd(grad_C + bc_base + threadIdx.x * kItems + i, dC_acc[i]);
      }
    }
  }
}

template <typename scalar_t>
void launch_backward(
    int length,
    int d_state,
    int n_chunks,
    dim3 grid,
    cudaStream_t stream,
    const scalar_t *grad_y,
    const float *grad_final_state,
    const scalar_t *x,
    const scalar_t *dt,
    const float *A,
    const scalar_t *B,
    const scalar_t *C,
    const float *D,
    const scalar_t *z,
    const float *initial_state,
    const float *states,
    scalar_t *grad_x,
    scalar_t *grad_dt,
    float *grad_A,
    float *grad_D,
    scalar_t *grad_z,
    float *grad_initial_state) {
  scan_backward_kernel<scalar_t><<<grid, kThreads, 0, stream>>>(
      grad_y, grad_final_state, x, dt, A, B, C, D, z, initial_state, states, grad_x,
      grad_dt, grad_A, grad_D, grad_z, grad_initial_state, length, d_state, n_chunks);
}

template <typename scalar_t>
void launch_backward_bc(
    int length,
    int d_state,
    int n_chunks,
    int channels,
    dim3 grid,
    cudaStream_t stream,
    const scalar_t *grad_y,
    const float *grad_final_state,
    const scalar_t *x,
    const scalar_t *dt,
    const float *A,
    const scalar_t *B,
    const scalar_t *C,
    const scalar_t *z,
    const float *initial_state,
    const float *states,
    float *grad_B,
    float *grad_C) {
  scan_backward_bc_kernel<scalar_t><<<grid, kThreads, 0, stream>>>(
      grad_y, grad_final_state, x, dt, A, B, C, z, initial_state, states, grad_B, grad_C,
      length, d_state, n_chunks, channels);
}

}  // namespace

// Inputs are in the canonical kernel layout: x/dt/z/y: [batch, channels, length],
// B/C: [batch, d_state, length], A: [channels, d_state], D: [channels],
// initial/final state: [batch, channels, d_state], checkpoints: [batch, channels,
// n_chunks, 2 * d_state] (interleaved (decay product, state) pairs).
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
  const int channels = static_cast<int>(x.size(1));
  const int length = static_cast<int>(x.size(2));
  const int d_state = static_cast<int>(A.size(1));
  const int n_chunks = (length + kChunk - 1) / kChunk;
  const dim3 grid(batch, channels);
  auto y = torch::empty_like(x);
  const auto float_options = x.options().dtype(torch::kFloat32);
  auto states = save_states
      ? torch::empty({batch, channels, n_chunks, 2 * d_state}, float_options)
      : torch::empty({0}, float_options);
  auto final_state = torch::empty({batch, channels, d_state}, float_options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x.scalar_type(),
      "mamba3_scan_forward_cuda",
      [&] {
        launch_forward<scalar_t>(
            length, d_state, n_chunks, grid, stream, x.data_ptr<scalar_t>(),
            dt.data_ptr<scalar_t>(), A.data_ptr<float>(), B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(), D.data_ptr<float>(), z.data_ptr<scalar_t>(),
            initial_state.data_ptr<float>(), y.data_ptr<scalar_t>(),
            save_states ? states.data_ptr<float>() : nullptr,
            final_state.data_ptr<float>(), save_states ? 1 : 0);
      });
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
    torch::Tensor state_checkpoints) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  const int batch = static_cast<int>(x.size(0));
  const int channels = static_cast<int>(x.size(1));
  const int length = static_cast<int>(x.size(2));
  const int d_state = static_cast<int>(A.size(1));
  const int n_chunks = (length + kChunk - 1) / kChunk;
  const dim3 grid(batch, channels);
  const int h_groups = (channels + kChanGroup - 1) / kChanGroup;
  const dim3 grid_bc(batch, d_state, h_groups);

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
      "mamba3_scan_backward_cuda",
      [&] {
        launch_backward<scalar_t>(
            length, d_state, n_chunks, grid, stream, grad_y.data_ptr<scalar_t>(),
            grad_final_state.data_ptr<float>(), x.data_ptr<scalar_t>(),
            dt.data_ptr<scalar_t>(), A.data_ptr<float>(), B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(), D.data_ptr<float>(), z.data_ptr<scalar_t>(),
            initial_state.data_ptr<float>(),
            state_checkpoints.numel() > 0 ? state_checkpoints.data_ptr<float>() : nullptr,
            grad_x.data_ptr<scalar_t>(), grad_dt.data_ptr<scalar_t>(),
            grad_A.data_ptr<float>(), grad_D.data_ptr<float>(),
            grad_z.data_ptr<scalar_t>(), grad_initial_state.data_ptr<float>());
        launch_backward_bc<scalar_t>(
            length, d_state, n_chunks, channels, grid_bc, stream,
            grad_y.data_ptr<scalar_t>(), grad_final_state.data_ptr<float>(),
            x.data_ptr<scalar_t>(), dt.data_ptr<scalar_t>(), A.data_ptr<float>(),
            B.data_ptr<scalar_t>(), C.data_ptr<scalar_t>(), z.data_ptr<scalar_t>(),
            initial_state.data_ptr<float>(),
            state_checkpoints.numel() > 0 ? state_checkpoints.data_ptr<float>() : nullptr,
            grad_B.data_ptr<float>(), grad_C.data_ptr<float>());
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_x, grad_dt, grad_A, grad_B, grad_C, grad_D, grad_z, grad_initial_state};
}

// Transpose the last two dimensions of a 3-D tensor ([B, X, Y] -> [B, Y, X]).
torch::Tensor mamba3_transpose_copy_cuda(torch::Tensor src) {
  const c10::cuda::CUDAGuard device_guard(src.device());
  TORCH_CHECK(src.dim() == 3, "expected a 3-D tensor");
  const int batch = static_cast<int>(src.size(0));
  const int x = static_cast<int>(src.size(1));
  const int y = static_cast<int>(src.size(2));
  auto dst = torch::empty({batch, y, x}, src.options());
  const dim3 grid((x + 63) / 64, (y + 63) / 64, batch);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      src.scalar_type(),
      "mamba3_transpose_copy_cuda",
      [&] {
        transpose_copy_kernel<scalar_t><<<grid, 256, 0, stream>>>(
            src.data_ptr<scalar_t>(), dst.data_ptr<scalar_t>(), batch, x, y);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dst;
}
