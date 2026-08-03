from __future__ import annotations

import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


# Silence warnings caused by loading and compiling Mamba3's fused CUDA kernels.
warnings.filterwarnings(
    "ignore",
    message=r"Dynamo detected a call to a `functools\.lru_cache`-wrapped function.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"Dynamo does not know how to trace the builtin `mamba3_cuda_v\d+\.[^`]*\.row_forward\.`.*",
)
warnings.filterwarnings("ignore", message=r"_get_vc_env is private.*")


@lru_cache(maxsize=1)
def load_cuda_extension(verbose: bool = False) -> Any | None:
    """Load the prebuilt extension or JIT-compile the bundled CUDA sources.

    Compilation is cached by PyTorch, so subsequent processes only load the
    resulting binary. Returns ``None`` when CUDA compilation is unavailable.
    Set ``MAMBA3_STRICT_CUDA=1`` to turn a build failure into an exception.
    """

    if os.getenv("MAMBA3_DISABLE_CUDA", "0") == "1" or not torch.cuda.is_available():
        return None

    try:
        from . import _C

        return _C
    except ImportError:
        pass

    try:
        from torch.utils.cpp_extension import load

        root = Path(__file__).resolve().parent / "csrc"
        cxx_flags = ["/O2"] if os.name == "nt" else ["-O3"]
        cuda_flags = [
            "-O3",
            "--use_fast_math",
            "--extra-device-vectorization",
        ]
        if os.name == "nt":
            cuda_flags.append("-Xcompiler=/Zc:preprocessor")
        return load(
            name="mamba3_cuda_v22",
            sources=[
                str(root / "scan.cpp"),
                str(root / "scan_cuda.cu"),
                str(root / "scan_row_cuda.cu"),
            ],
            extra_cflags=cxx_flags,
            extra_cuda_cflags=cuda_flags,
            verbose=verbose or os.getenv("MAMBA3_VERBOSE_BUILD", "0") == "1",
        )
    except Exception as exc:  # pragma: no cover - toolchain dependent
        if os.getenv("MAMBA3_STRICT_CUDA", "0") == "1":
            raise
        warnings.warn(
            f"mamba3 CUDA extension could not be loaded; using the PyTorch "
            f"reference scan instead. Build error: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def _validate_scan_inputs(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> tuple[int, int, int, int]:
    if x.ndim != 3:
        raise ValueError(f"x must have shape [batch, length, channels], got {tuple(x.shape)}")
    batch, length, channels = x.shape
    if dt.shape != x.shape or z.shape != x.shape:
        raise ValueError("dt and z must have the same shape as x")
    if A.ndim != 2 or A.shape[0] != channels:
        raise ValueError("A must have shape [channels, d_state]")
    d_state = A.shape[1]
    if d_state < 1:
        raise ValueError("d_state must be positive")
    if B.shape != (batch, length, d_state) or C.shape != B.shape:
        raise ValueError("B and C must have shape [batch, length, d_state]")
    if D.shape != (channels,):
        raise ValueError("D must have shape [channels]")
    if initial_state is not None and initial_state.shape != (batch, channels, d_state):
        raise ValueError("initial_state must have shape [batch, channels, d_state]")
    devices = {tensor.device for tensor in (x, dt, A, B, C, D, z)}
    if initial_state is not None:
        devices.add(initial_state.device)
    if len(devices) != 1:
        raise ValueError("all selective_scan tensors must be on the same device")
    return batch, length, channels, d_state


def _reference_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # State math stays in float32 under autocast; this avoids long-sequence
    # underflow and matches the CUDA kernel's accumulation behavior.
    output_dtype = x.dtype
    x_f, dt_f, A_f = x.float(), dt.float(), A.float()
    B_f, C_f, D_f, z_f = B.float(), C.float(), D.float(), z.float()
    state = initial_state.float()
    outputs: list[torch.Tensor] = []
    for index in range(x.shape[1]):
        decay = torch.exp(dt_f[:, index, :, None] * A_f[None, :, :])
        state = decay * state + x_f[:, index, :, None] * B_f[:, index, None, :]
        base = (state * C_f[:, index, None, :]).sum(dim=-1)
        base = base + D_f[None, :] * x_f[:, index, :]
        outputs.append(base * F.silu(z_f[:, index, :]))
    if outputs:
        y = torch.stack(outputs, dim=1)
    else:
        y = x_f.new_empty(x_f.shape)
    return y.to(output_dtype), state


class _SelectiveScanCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dt, A, B, C, D, z, initial_state, extension, reverse):
        # The kernels scan sequence positions in parallel, so they want the
        # channel-major layouts [B, H, L] for x/dt/z/y and [B, N, L] for B/C.
        # The fused transposes are cheap GPU copies; doing them inside this
        # function keeps the whole scan a single autograd node.
        kernel_tensors = (
            _to_kernel_layout(x, extension, reverse),
            _to_kernel_layout(dt, extension, reverse),
            A,
            _to_kernel_layout(B, extension, reverse),
            _to_kernel_layout(C, extension, reverse),
            D,
            _to_kernel_layout(z, extension, reverse),
            initial_state,
        )
        # Save sparse recurrent checkpoints (one (decay, state) pair per chunk)
        # rather than the full [B, L, H, N] history. The backward kernels
        # exactly recompute each short chunk, avoiding both the large
        # allocation and unstable reverse inversion of the recurrence.
        y_t, state_checkpoints, final_state = extension.forward(*kernel_tensors, True)
        transpose_output = extension.transpose_reverse_y if reverse else extension.transpose
        y = transpose_output(y_t)
        ctx.extension = extension
        ctx.reverse = reverse
        ctx.input_dtypes = tuple(tensor.dtype for tensor in (x, dt, A, B, C, D, z, initial_state))
        ctx.save_for_backward(*kernel_tensors, state_checkpoints)
        return y.to(x.dtype), final_state

    @staticmethod
    def backward(ctx, grad_y, grad_final_state):
        x_t, dt_t, A, B_t, C_t, D, z_t, initial_state, state_checkpoints = ctx.saved_tensors
        extension = ctx.extension
        if grad_y is None:
            grad_y = torch.zeros_like(x_t)
        else:
            grad_y = _to_kernel_layout(grad_y.contiguous(), extension, ctx.reverse)
        if grad_final_state is None:
            grad_final_state = torch.zeros_like(initial_state)
        grads = extension.backward(
            grad_y.to(dtype=x_t.dtype).contiguous(),
            grad_final_state.contiguous().float(),
            x_t,
            dt_t,
            A,
            B_t,
            C_t,
            D,
            z_t,
            initial_state,
            state_checkpoints,
        )
        grad_x, grad_dt, grad_A, grad_B, grad_C, grad_D, grad_z, grad_is = grads
        # Gradients come back in the kernel layout and are transposed to match
        # the [B, L, H] / [B, L, N] inputs of this function.
        grads = (
            (extension.transpose_reverse_y if ctx.reverse else extension.transpose)(grad_x),
            (extension.transpose_reverse_y if ctx.reverse else extension.transpose)(grad_dt),
            grad_A,
            (extension.transpose_reverse_y if ctx.reverse else extension.transpose)(grad_B),
            (extension.transpose_reverse_y if ctx.reverse else extension.transpose)(grad_C),
            grad_D,
            (extension.transpose_reverse_y if ctx.reverse else extension.transpose)(grad_z),
            grad_is,
        )
        cast_grads = tuple(grad.to(dtype) for grad, dtype in zip(grads, ctx.input_dtypes))
        return (*cast_grads, None, None)


class _SelectiveScanCudaRow(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dt, A, B, C, D, z, initial_state, extension, reverse):
        tensors = (x, dt, A, B, C, D, z, initial_state)
        y, state_checkpoints, final_state = extension.row_forward(
            *tensors, True, reverse
        )
        ctx.extension = extension
        ctx.reverse = reverse
        ctx.input_dtypes = tuple(tensor.dtype for tensor in tensors)
        ctx.save_for_backward(*tensors, state_checkpoints)
        return y.to(x.dtype), final_state

    @staticmethod
    def backward(ctx, grad_y, grad_final_state):
        x, dt, A, B, C, D, z, initial_state, state_checkpoints = ctx.saved_tensors
        if grad_y is None:
            grad_y = torch.zeros_like(x)
        if grad_final_state is None:
            grad_final_state = torch.zeros_like(initial_state)
        grads = ctx.extension.row_backward(
            grad_y.to(dtype=x.dtype).contiguous(),
            grad_final_state.contiguous().float(),
            x,
            dt,
            A,
            B,
            C,
            D,
            z,
            initial_state,
            state_checkpoints,
            ctx.reverse,
        )
        cast_grads = tuple(grad.to(dtype) for grad, dtype in zip(grads, ctx.input_dtypes))
        return (*cast_grads, None, None)


def _use_row_cuda_kernel(batch: int, length: int, d_state: int = 64) -> bool:
    """Use row parallelism when audio-style shapes provide enough independent rows."""

    return d_state <= 64 and batch >= 8 and length <= 64 * batch


def _row_checkpoint_bytes(batch: int, length: int, channels: int, d_state: int) -> int:
    bucket = 8 if d_state <= 8 else 16 if d_state <= 16 else 32 if d_state <= 32 else 64
    stride = 128 // bucket
    checkpoints = (length + stride - 1) // stride
    return batch * checkpoints * channels * d_state * 4


def _reverse_scan_inputs(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return (
        x.flip(1),
        dt.flip(1),
        A,
        B.flip(1),
        C.flip(1),
        D,
        z.flip(1),
        initial_state,
    )


def _prepare_cuda_inputs(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: torch.Tensor,
    initial_state: torch.Tensor,
    contiguous: bool,
) -> tuple[torch.Tensor, ...]:
    """Normalize CUDA kernel dtypes without expanding mixed-precision activations."""

    activation_dtype = (
        x.dtype
        if x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        else torch.float32
    )
    activations = tuple(
        tensor.to(dtype=activation_dtype) for tensor in (x, dt, B, C, z)
    )
    if contiguous:
        activations = tuple(tensor.contiguous() for tensor in activations)
    x, dt, B, C, z = activations
    return (
        x,
        dt,
        A.float().contiguous(),
        B,
        C,
        D.float().contiguous(),
        z,
        initial_state.float().contiguous(),
    )


def _to_kernel_layout(
    tensor: torch.Tensor, extension, reverse: bool = False
) -> torch.Tensor:
    if reverse:
        return extension.transpose_reverse_x(tensor)
    transposed = tensor.transpose(1, 2)
    if transposed.is_contiguous():
        return transposed
    return extension.transpose(tensor)


def _causal_conv_step(
    x: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    extension = load_cuda_extension() if x.is_cuda else None
    same_dtype = x.dtype == state.dtype == weight.dtype == bias.dtype
    same_device = x.device == state.device == weight.device == bias.device
    if (
        extension is not None
        and not torch.is_grad_enabled()
        and same_dtype
        and same_device
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        output, state = extension.conv_step(
            x.contiguous(), state.contiguous(), weight.contiguous(), bias.contiguous()
        )
        return output, state

    conv_input = torch.cat((state, x.transpose(1, 2)), dim=-1)
    output = F.silu(F.conv1d(conv_input, weight, bias=bias, groups=x.shape[-1]))
    return output.transpose(1, 2), conv_input[:, :, 1:]


def _causal_conv_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor | None:
    if not x.is_cuda:
        return None
    extension = load_cuda_extension()
    if (
        extension is None
        or x.dtype != weight.dtype
        or x.dtype != bias.dtype
        or x.device != weight.device
        or x.device != bias.device
        or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return None
    if torch.is_grad_enabled():
        return None
    return extension.conv_forward(x, weight.contiguous(), bias.contiguous())


def _selective_scan_step(
    x: torch.Tensor,
    dt_logits: torch.Tensor,
    dt_bias: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if torch.is_grad_enabled() or not x.is_cuda:
        return None
    extension = load_cuda_extension()
    activations = (x, dt_logits, dt_bias, B, C, z)
    if (
        extension is None
        or any(tensor.dtype != x.dtype for tensor in activations)
        or any(tensor.device != x.device for tensor in (*activations, A, D, state))
        or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return None
    return extension.scan_step(
        x.contiguous(),
        dt_logits.contiguous(),
        dt_bias.contiguous(),
        A.float().contiguous(),
        B.contiguous(),
        C.contiguous(),
        D.float().contiguous(),
        z.contiguous(),
        state.float().contiguous(),
    )


def selective_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    return_state: bool = False,
    use_cuda_kernel: bool = True,
    reverse: bool = False,
    _force_row: bool = False,
    _dt_bias: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run the gated selective state-space recurrence.

    Shapes are ``x/dt/z: [B, L, H]``, ``A: [H, N]``, ``B/C: [B, L, N]``,
    and ``D: [H]``. CUDA uses one fused kernel for recurrence, skip, and gate.
    ``reverse=True`` scans from the final position while returning outputs in
    the original sequence order.
    """

    batch, _, channels, d_state = _validate_scan_inputs(x, dt, A, B, C, D, z, initial_state)
    if initial_state is None:
        initial_state = torch.zeros(
            batch, channels, d_state, device=x.device, dtype=torch.float32
        )
    if batch == 0 or x.shape[1] == 0:
        return (torch.empty_like(x), initial_state) if return_state else torch.empty_like(x)

    extension = load_cuda_extension() if use_cuda_kernel and x.is_cuda else None
    if _dt_bias is not None and extension is None:
        dt = F.softplus(dt + _dt_bias).clamp(max=1.0)
        _dt_bias = None
    if extension is not None:
        needs_backward = torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (x, dt, A, B, C, D, z, initial_state)
        )
        use_row_kernel = d_state <= 64 and (
            _force_row or _use_row_cuda_kernel(batch, x.shape[1], d_state)
        )
        if use_row_kernel and needs_backward:
            free_memory, _ = torch.cuda.mem_get_info(x.device)
            checkpoint_budget = min(2 * 1024**3, free_memory // 4)
            use_row_kernel = (
                _row_checkpoint_bytes(batch, x.shape[1], channels, d_state)
                <= checkpoint_budget
            )
        fused_reverse = reverse and not use_row_kernel
        if _dt_bias is not None and (use_row_kernel or needs_backward):
            dt = F.softplus(dt + _dt_bias).clamp(max=1.0)
            _dt_bias = None
        inputs = (x, dt, A, B, C, D, z, initial_state)
        kernel_tensors = _prepare_cuda_inputs(*inputs, contiguous=use_row_kernel)
        if use_row_kernel and needs_backward:
            y, final_state = _SelectiveScanCudaRow.apply(
                *kernel_tensors, extension, reverse
            )
        elif use_row_kernel:
            y, _, final_state = extension.row_forward(
                *kernel_tensors, False, reverse
            )
        elif needs_backward:
            y, final_state = _SelectiveScanCuda.apply(
                *kernel_tensors, extension, fused_reverse
            )
        else:
            x_k, dt_k, A_k, B_k, C_k, D_k, z_k, init_k = kernel_tensors
            arguments = (
                _to_kernel_layout(x_k, extension, fused_reverse),
                _to_kernel_layout(dt_k, extension, fused_reverse),
            )
            if _dt_bias is not None:
                arguments += (_dt_bias.to(dtype=x_k.dtype).contiguous(),)
            arguments += (
                A_k,
                _to_kernel_layout(B_k, extension, fused_reverse),
                _to_kernel_layout(C_k, extension, fused_reverse),
                D_k,
                _to_kernel_layout(z_k, extension, fused_reverse),
                init_k,
                False,
            )
            forward = extension.forward_dt if _dt_bias is not None else extension.forward
            y_t, _, final_state = forward(*arguments)
            transpose_output = (
                extension.transpose_reverse_y if fused_reverse else extension.transpose
            )
            y = transpose_output(y_t).to(x.dtype)
    else:
        inputs = (x, dt, A, B, C, D, z, initial_state)
        if reverse:
            inputs = _reverse_scan_inputs(*inputs)
        y, final_state = _reference_scan(*inputs)
        if reverse:
            y = y.flip(1)
    return (y, final_state) if return_state else y
