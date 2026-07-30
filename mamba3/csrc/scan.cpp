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
    torch::Tensor states);

static void check_cuda_float_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
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
  check_cuda_float_contiguous(x, "x");
  check_cuda_float_contiguous(dt, "dt");
  check_cuda_float_contiguous(A, "A");
  check_cuda_float_contiguous(B, "B");
  check_cuda_float_contiguous(C, "C");
  check_cuda_float_contiguous(D, "D");
  check_cuda_float_contiguous(z, "z");
  check_cuda_float_contiguous(initial_state, "initial_state");
  TORCH_CHECK(x.dim() == 3, "x must be [batch, length, channels]");
  TORCH_CHECK(A.dim() == 2, "A must be [channels, d_state]");
  TORCH_CHECK(A.size(1) >= 1 && A.size(1) <= 64, "d_state must be in [1, 64]");
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
    torch::Tensor states) {
  check_cuda_float_contiguous(grad_y, "grad_y");
  check_cuda_float_contiguous(grad_final_state, "grad_final_state");
  check_cuda_float_contiguous(states, "states");
  return mamba3_scan_backward_cuda(
      grad_y, grad_final_state, x, dt, A, B, C, D, z, initial_state, states);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &scan_forward, "Fused Mamba3 selective scan forward (CUDA)");
  module.def("backward", &scan_backward, "Fused Mamba3 selective scan backward (CUDA)");
}
