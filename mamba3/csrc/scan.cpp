#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> mamba3_scan_forward_cuda(
    torch::Tensor x,
    torch::Tensor dt,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor D,
    torch::Tensor z,
    torch::Tensor initial_state,
    bool save_states);

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
    torch::Tensor state_checkpoints);

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
    bool reverse);

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
    bool reverse);

torch::Tensor mamba3_transpose_copy_cuda(torch::Tensor src);

static void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

static void check_cuda_float_contiguous(const torch::Tensor& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
}

static void check_cuda_activation(const torch::Tensor& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  const auto type = tensor.scalar_type();
  TORCH_CHECK(
      type == torch::kFloat32 || type == torch::kFloat16 || type == torch::kBFloat16,
      name, " must be float32, float16, or bfloat16");
}

std::vector<torch::Tensor> scan_forward(
    torch::Tensor x,
    torch::Tensor dt,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor D,
    torch::Tensor z,
    torch::Tensor initial_state,
    bool save_states) {
  check_cuda_activation(x, "x");
  check_cuda_activation(dt, "dt");
  check_cuda_float_contiguous(A, "A");
  check_cuda_activation(B, "B");
  check_cuda_activation(C, "C");
  check_cuda_float_contiguous(D, "D");
  check_cuda_activation(z, "z");
  check_cuda_float_contiguous(initial_state, "initial_state");
  TORCH_CHECK(
      x.scalar_type() == dt.scalar_type() && x.scalar_type() == B.scalar_type() &&
          x.scalar_type() == C.scalar_type() && x.scalar_type() == z.scalar_type(),
      "x, dt, B, C, and z must have the same dtype");
  TORCH_CHECK(x.dim() == 3, "x must be [batch, length, channels]");
  TORCH_CHECK(A.dim() == 2, "A must be [channels, d_state]");
  TORCH_CHECK(A.size(1) >= 1, "d_state must be positive");
  return mamba3_scan_forward_cuda(x, dt, A, B, C, D, z, initial_state, save_states);
}

std::vector<torch::Tensor> scan_backward(
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
  check_cuda_activation(grad_y, "grad_y");
  check_cuda_float_contiguous(grad_final_state, "grad_final_state");
  check_cuda_float_contiguous(state_checkpoints, "state_checkpoints");
  TORCH_CHECK(grad_y.scalar_type() == x.scalar_type(), "grad_y and x must have the same dtype");
  return mamba3_scan_backward_cuda(
      grad_y, grad_final_state, x, dt, A, B, C, D, z, initial_state, state_checkpoints);
}

std::vector<torch::Tensor> row_scan_forward(
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
  check_cuda_activation(x, "x");
  check_cuda_activation(dt, "dt");
  check_cuda_float_contiguous(A, "A");
  check_cuda_activation(B, "B");
  check_cuda_activation(C, "C");
  check_cuda_float_contiguous(D, "D");
  check_cuda_activation(z, "z");
  check_cuda_float_contiguous(initial_state, "initial_state");
  TORCH_CHECK(
      x.scalar_type() == dt.scalar_type() && x.scalar_type() == B.scalar_type() &&
          x.scalar_type() == C.scalar_type() && x.scalar_type() == z.scalar_type(),
      "x, dt, B, C, and z must have the same dtype");
  TORCH_CHECK(x.dim() == 3, "x must be [batch, length, channels]");
  TORCH_CHECK(A.dim() == 2, "A must be [channels, d_state]");
  TORCH_CHECK(
      A.size(1) >= 1 && A.size(1) <= 64,
      "row kernel d_state must be in [1, 64]");
  return mamba3_row_scan_forward_cuda(
      x, dt, A, B, C, D, z, initial_state, save_states, reverse);
}

std::vector<torch::Tensor> row_scan_backward(
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
  check_cuda_activation(grad_y, "grad_y");
  check_cuda_float_contiguous(grad_final_state, "grad_final_state");
  check_cuda_float_contiguous(state_checkpoints, "state_checkpoints");
  TORCH_CHECK(grad_y.scalar_type() == x.scalar_type(), "grad_y and x must have the same dtype");
  return mamba3_row_scan_backward_cuda(
      grad_y, grad_final_state, x, dt, A, B, C, D, z, initial_state,
      state_checkpoints, reverse);
}

torch::Tensor transpose_copy(torch::Tensor src) {
  check_cuda_contiguous(src, "src");
  TORCH_CHECK(src.dim() == 3, "src must be [batch, x, y]");
  return mamba3_transpose_copy_cuda(src);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &scan_forward, "Fused Mamba3 selective scan forward (CUDA)");
  module.def("backward", &scan_backward, "Fused Mamba3 selective scan backward (CUDA)");
  module.def("row_forward", &row_scan_forward, "Row-parallel Mamba3 selective scan forward (CUDA)");
  module.def("row_backward", &row_scan_backward, "Row-parallel Mamba3 selective scan backward (CUDA)");
  module.def("transpose", &transpose_copy, "Fast [B, X, Y] <-> [B, Y, X] transpose (CUDA)");
}
