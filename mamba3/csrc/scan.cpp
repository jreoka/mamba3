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
std::vector<torch::Tensor> mamba3_scan_forward_dt_cuda(
    torch::Tensor x,
    torch::Tensor dt_logits,
    torch::Tensor dt_bias,
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

torch::Tensor mamba3_transpose_copy_cuda(torch::Tensor src, int mode);
std::vector<torch::Tensor> mamba3_causal_conv_step_cuda(
    torch::Tensor x,
    torch::Tensor state,
    torch::Tensor weight,
    torch::Tensor bias);
torch::Tensor mamba3_causal_conv_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias);
std::vector<torch::Tensor> mamba3_scan_step_cuda(
    torch::Tensor x,
    torch::Tensor dt_logits,
    torch::Tensor dt_bias,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor D,
    torch::Tensor z,
    torch::Tensor state);

static void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

static void check_cuda_float_contiguous(const torch::Tensor& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
}

static void check_cuda_activation_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  const auto type = tensor.scalar_type();
  TORCH_CHECK(
      type == torch::kFloat32 || type == torch::kFloat16 || type == torch::kBFloat16,
      name, " must be float32, float16, or bfloat16");
}

static void check_cuda_activation(const torch::Tensor& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  check_cuda_activation_tensor(tensor, name);
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

std::vector<torch::Tensor> scan_forward_dt(
    torch::Tensor x,
    torch::Tensor dt_logits,
    torch::Tensor dt_bias,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor D,
    torch::Tensor z,
    torch::Tensor initial_state,
    bool save_states) {
  check_cuda_activation(x, "x");
  check_cuda_activation(dt_logits, "dt_logits");
  check_cuda_activation(dt_bias, "dt_bias");
  check_cuda_float_contiguous(A, "A");
  check_cuda_activation(B, "B");
  check_cuda_activation(C, "C");
  check_cuda_float_contiguous(D, "D");
  check_cuda_activation(z, "z");
  check_cuda_float_contiguous(initial_state, "initial_state");
  TORCH_CHECK(x.scalar_type() == dt_logits.scalar_type() &&
          x.scalar_type() == dt_bias.scalar_type() && x.scalar_type() == B.scalar_type() &&
          x.scalar_type() == C.scalar_type() && x.scalar_type() == z.scalar_type(),
      "activation tensors must have the same dtype");
  TORCH_CHECK(dt_bias.dim() == 1 && dt_bias.size(0) == x.size(1),
      "dt_bias must match channels");
  return mamba3_scan_forward_dt_cuda(
      x, dt_logits, dt_bias, A, B, C, D, z, initial_state, save_states);
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
  TORCH_CHECK(src.is_cuda(), "src must be a CUDA tensor");
  TORCH_CHECK(src.dim() == 3, "src must be [batch, x, y]");
  return mamba3_transpose_copy_cuda(src, 0);
}

torch::Tensor transpose_reverse_x(torch::Tensor src) {
  TORCH_CHECK(src.is_cuda() && src.dim() == 3, "src must be a 3-D CUDA tensor");
  return mamba3_transpose_copy_cuda(src, 1);
}

torch::Tensor transpose_reverse_y(torch::Tensor src) {
  TORCH_CHECK(src.is_cuda() && src.dim() == 3, "src must be a 3-D CUDA tensor");
  return mamba3_transpose_copy_cuda(src, 2);
}

std::vector<torch::Tensor> causal_conv_step(
    torch::Tensor x,
    torch::Tensor state,
    torch::Tensor weight,
    torch::Tensor bias) {
  check_cuda_activation(x, "x");
  check_cuda_activation(state, "state");
  check_cuda_activation(weight, "weight");
  check_cuda_activation(bias, "bias");
  TORCH_CHECK(x.dim() == 3 && x.size(1) == 1, "x must be [batch, 1, channels]");
  TORCH_CHECK(state.dim() == 3 && state.size(2) == 3, "state must be [batch, channels, 3]");
  TORCH_CHECK(weight.dim() == 3 && weight.size(1) == 1 && weight.size(2) == 4,
      "weight must be [channels, 1, 4]");
  TORCH_CHECK(bias.dim() == 1, "bias must be [channels]");
  TORCH_CHECK(
      state.size(0) == x.size(0) && state.size(1) == x.size(2) &&
          weight.size(0) == x.size(2) && bias.size(0) == x.size(2),
      "state, weight, and bias channels must match x");
  TORCH_CHECK(
      x.device() == state.device() && x.device() == weight.device() &&
          x.device() == bias.device(),
      "x, state, weight, and bias must be on the same device");
  TORCH_CHECK(
      x.scalar_type() == state.scalar_type() && x.scalar_type() == weight.scalar_type() &&
          x.scalar_type() == bias.scalar_type(),
      "x, state, weight, and bias must have the same dtype");
  return mamba3_causal_conv_step_cuda(x, state, weight, bias);
}

torch::Tensor causal_conv_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias) {
  check_cuda_activation_tensor(x, "x");
  check_cuda_activation(weight, "weight");
  check_cuda_activation(bias, "bias");
  TORCH_CHECK(x.dim() == 3, "x must be [batch, length, channels]");
  TORCH_CHECK(weight.dim() == 3 && weight.size(1) == 1 && weight.size(2) == 4,
      "weight must be [channels, 1, 4]");
  TORCH_CHECK(bias.dim() == 1, "bias must be [channels]");
  TORCH_CHECK(weight.size(0) == x.size(2) && bias.size(0) == x.size(2),
      "weight and bias channels must match x");
  TORCH_CHECK(x.device() == weight.device() && x.device() == bias.device(),
      "x, weight, and bias must be on the same device");
  TORCH_CHECK(x.scalar_type() == weight.scalar_type() && x.scalar_type() == bias.scalar_type(),
      "x, weight, and bias must have the same dtype");
  return mamba3_causal_conv_forward_cuda(x, weight, bias);
}

std::vector<torch::Tensor> scan_step(
    torch::Tensor x,
    torch::Tensor dt_logits,
    torch::Tensor dt_bias,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor D,
    torch::Tensor z,
    torch::Tensor state) {
  check_cuda_activation(x, "x");
  check_cuda_activation(dt_logits, "dt_logits");
  check_cuda_activation(dt_bias, "dt_bias");
  check_cuda_float_contiguous(A, "A");
  check_cuda_activation(B, "B");
  check_cuda_activation(C, "C");
  check_cuda_float_contiguous(D, "D");
  check_cuda_activation(z, "z");
  check_cuda_float_contiguous(state, "state");
  TORCH_CHECK(x.dim() == 3 && x.size(1) == 1, "x must be [batch, 1, channels]");
  TORCH_CHECK(dt_logits.sizes() == x.sizes() && z.sizes() == x.sizes(),
      "dt_logits and z must match x");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 3 && C.sizes() == B.sizes(),
      "A, B, and C have invalid shapes");
  TORCH_CHECK(state.size(0) == x.size(0) && state.size(1) == x.size(2) &&
          state.size(2) == A.size(1) && B.size(0) == x.size(0) && B.size(1) == 1 &&
          B.size(2) == A.size(1) && A.size(0) == x.size(2) && D.size(0) == x.size(2) &&
          dt_bias.size(0) == x.size(2),
      "scan step dimensions do not match");
  TORCH_CHECK(
      x.scalar_type() == dt_logits.scalar_type() && x.scalar_type() == dt_bias.scalar_type() &&
          x.scalar_type() == B.scalar_type() && x.scalar_type() == C.scalar_type() &&
          x.scalar_type() == z.scalar_type(),
      "activation tensors must have the same dtype");
  TORCH_CHECK(x.device() == dt_logits.device() && x.device() == dt_bias.device() &&
          x.device() == A.device() && x.device() == B.device() && x.device() == C.device() &&
          x.device() == D.device() && x.device() == z.device() && x.device() == state.device(),
      "scan step tensors must be on the same device");
  return mamba3_scan_step_cuda(x, dt_logits, dt_bias, A, B, C, D, z, state);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &scan_forward, "Fused Mamba3 selective scan forward (CUDA)");
  module.def("forward_dt", &scan_forward_dt,
      "Fused Mamba3 selective scan with delta activation (CUDA)");
  module.def("backward", &scan_backward, "Fused Mamba3 selective scan backward (CUDA)");
  module.def("row_forward", &row_scan_forward, "Row-parallel Mamba3 selective scan forward (CUDA)");
  module.def("row_backward", &row_scan_backward, "Row-parallel Mamba3 selective scan backward (CUDA)");
  module.def("transpose", &transpose_copy, "Fast [B, X, Y] <-> [B, Y, X] transpose (CUDA)");
  module.def("transpose_reverse_x", &transpose_reverse_x,
      "Transpose while reversing the source X axis (CUDA)");
  module.def("transpose_reverse_y", &transpose_reverse_y,
      "Transpose while reversing the source Y axis (CUDA)");
  module.def("conv_step", &causal_conv_step, "Cached causal depthwise convolution step (CUDA)");
  module.def("conv_forward", &causal_conv_forward, "Fused causal depthwise convolution (CUDA)");
  module.def("scan_step", &scan_step, "Cached selective scan step (CUDA)");
}
