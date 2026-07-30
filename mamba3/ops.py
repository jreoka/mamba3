from __future__ import annotations

import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


_LOAD_ERROR: Exception | None = None


@lru_cache(maxsize=1)
def load_cuda_extension(verbose: bool = False) -> Any | None:
    """Load the prebuilt extension or JIT-compile the bundled CUDA sources.

    Compilation is cached by PyTorch, so subsequent processes only load the
    resulting binary. Returns ``None`` when CUDA compilation is unavailable.
    Set ``MAMBA3_STRICT_CUDA=1`` to turn a build failure into an exception.
    """

    global _LOAD_ERROR
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
            name="mamba3_cuda_v1",
            sources=[str(root / "scan.cpp"), str(root / "scan_cuda.cu")],
            extra_cflags=cxx_flags,
            extra_cuda_cflags=cuda_flags,
            verbose=verbose or os.getenv("MAMBA3_VERBOSE_BUILD", "0") == "1",
        )
    except Exception as exc:  # pragma: no cover - toolchain dependent
        _LOAD_ERROR = exc
        if os.getenv("MAMBA3_STRICT_CUDA", "0") == "1":
            raise
        warnings.warn(
            f"mamba3 CUDA extension could not be loaded; using the PyTorch "
            f"reference scan instead. Build error: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def cuda_extension_available() -> bool:
    """Return whether the fused CUDA extension can be loaded."""

    return load_cuda_extension() is not None


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
    def forward(ctx, x, dt, A, B, C, D, z, initial_state, extension):
        tensors = [tensor.contiguous().float() for tensor in (x, dt, A, B, C, D, z, initial_state)]
        y, states, final_state = extension.forward(*tensors, True)
        ctx.extension = extension
        ctx.input_dtypes = tuple(tensor.dtype for tensor in (x, dt, A, B, C, D, z, initial_state))
        ctx.save_for_backward(*tensors, states)
        return y.to(x.dtype), final_state

    @staticmethod
    def backward(ctx, grad_y, grad_final_state):
        x, dt, A, B, C, D, z, initial_state, states = ctx.saved_tensors
        if grad_y is None:
            grad_y = torch.zeros_like(x)
        if grad_final_state is None:
            grad_final_state = torch.zeros_like(initial_state)
        grads = ctx.extension.backward(
            grad_y.contiguous().float(),
            grad_final_state.contiguous().float(),
            x,
            dt,
            A,
            B,
            C,
            D,
            z,
            initial_state,
            states,
        )
        cast_grads = tuple(grad.to(dtype) for grad, dtype in zip(grads, ctx.input_dtypes))
        return (*cast_grads, None)


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
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run the gated selective state-space recurrence.

    Shapes are ``x/dt/z: [B, L, H]``, ``A: [H, N]``, ``B/C: [B, L, N]``,
    and ``D: [H]``. CUDA uses one fused kernel for recurrence, skip, and gate.
    """

    batch, _, channels, d_state = _validate_scan_inputs(x, dt, A, B, C, D, z, initial_state)
    if initial_state is None:
        initial_state = torch.zeros(
            batch, channels, d_state, device=x.device, dtype=torch.float32
        )

    extension = load_cuda_extension() if use_cuda_kernel and x.is_cuda else None
    if extension is not None:
        needs_backward = torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (x, dt, A, B, C, D, z, initial_state)
        )
        if needs_backward:
            y, final_state = _SelectiveScanCuda.apply(
                x, dt, A, B, C, D, z, initial_state, extension
            )
        else:
            tensors = [
                tensor.contiguous().float()
                for tensor in (x, dt, A, B, C, D, z, initial_state)
            ]
            y, _, final_state = extension.forward(*tensors, False)
            y = y.to(x.dtype)
    else:
        y, final_state = _reference_scan(x, dt, A, B, C, D, z, initial_state)
    return (y, final_state) if return_state else y
