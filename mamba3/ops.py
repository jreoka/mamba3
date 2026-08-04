from __future__ import annotations

import math
import os
import warnings

import torch
from torch.nn import functional as F


_TRITON_STEP = None
_TRITON_STEP_CHECKED = False
_TRITON_FAILURES: set[tuple[object, ...]] = set()
_TRITON_LAST_DISPATCH = False
try:
    _TORCH_VERSION = tuple(
        int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2]
    )
except ValueError:  # pragma: no cover - nonstandard development version
    _TORCH_VERSION = (0, 0)
_BMM_OUT_DTYPE_SUPPORTED = _TORCH_VERSION >= (2, 8)


def heavy_tail_activation(x: torch.Tensor) -> torch.Tensor:
    """Positive heavy-tail map used by the maintained Mamba-3 implementation."""

    negative_branch = torch.reciprocal(1.0 - x.clamp_max(0))
    return torch.where(x >= 0, 1.0 + x, negative_branch)


def rotate_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    phase: torch.Tensor,
    *,
    mimo: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Mamba-3's real representation of data-dependent complex dynamics.

    ``q`` and ``k`` are ``[B, L, H, R, N]`` and ``phase`` is
    ``[B, L, H, S]``. SISO uses adjacent complex pairs. MIMO follows the
    official split-half checkpoint layout.
    """

    if q.shape != k.shape or q.ndim != 5:
        raise ValueError("q and k must have matching [B, L, H, R, N] shapes")
    if phase.ndim != 4 or phase.shape[:3] != q.shape[:3]:
        raise ValueError("phase must have shape [B, L, H, S]")

    angles = phase.unsqueeze(3)
    cosine = torch.cos(angles).to(dtype=q.dtype)
    sine = torch.sin(angles).to(dtype=q.dtype)
    angle_count = phase.shape[-1]
    required_width = (4 if mimo else 2) * angle_count
    if required_width > q.shape[-1]:
        raise ValueError("phase has too many angles for the Q/K state width")

    def rotate(tensor: torch.Tensor) -> torch.Tensor:
        if mimo:
            # For rope_fraction=0.5, pair the first and third state quarters.
            first = tensor[..., :angle_count]
            second = tensor[..., angle_count : 2 * angle_count]
            third = tensor[..., 2 * angle_count : 3 * angle_count]
            fourth = tensor[..., 3 * angle_count : 4 * angle_count]
            tail = tensor[..., 4 * angle_count :]
            first_rotated = first * cosine - third * sine
            third_rotated = first * sine + third * cosine
            return torch.cat(
                (first_rotated, second, third_rotated, fourth, tail), dim=-1
            )

        rotary_width = 2 * angle_count
        paired = tensor[..., :rotary_width].reshape(*tensor.shape[:-1], angle_count, 2)
        first, second = paired.unbind(dim=-1)
        first_rotated = first * cosine - second * sine
        second_rotated = first * sine + second * cosine
        rotated = torch.stack((first_rotated, second_rotated), dim=-1).flatten(-2)
        return torch.cat((rotated, tensor[..., rotary_width:]), dim=-1)

    return rotate(q), rotate(k)


def fused_siso_step(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    adt: torch.Tensor,
    dt: torch.Tensor,
    trap_logits: torch.Tensor,
    angle_rate: torch.Tensor,
    D: torch.Tensor,
    cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] | None:
    """Use the fused Triton decoder when the runtime and shape support it."""

    global _TRITON_STEP, _TRITON_STEP_CHECKED, _TRITON_LAST_DISPATCH
    _TRITON_LAST_DISPATCH = False
    d_state = q.shape[-1]
    head_dim = value.shape[-1]
    failure_key = (
        q.device,
        q.dtype,
        k.dtype,
        value.dtype,
        gate.dtype,
        adt.dtype,
        dt.dtype,
        trap_logits.dtype,
        angle_rate.dtype,
        D.dtype,
        tuple(tensor.dtype for tensor in cache),
        d_state,
        head_dim,
        cache[0].shape[-1],
    )
    supported_shape = (
        d_state >= 16
        and head_dim >= 16
        and d_state & (d_state - 1) == 0
        and head_dim & (head_dim - 1) == 0
        and cache[0].shape[-1] <= d_state // 2
        and cache[2].dtype == q.dtype
        and cache[3].dtype == value.dtype
    )
    if (
        torch.is_grad_enabled()
        or not q.is_cuda
        or q.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or not supported_shape
        or failure_key in _TRITON_FAILURES
        or os.getenv("MAMBA3_DISABLE_TRITON", "0") == "1"
    ):
        return None

    if not _TRITON_STEP_CHECKED:
        _TRITON_STEP_CHECKED = True
        try:
            from ._triton import siso_step

            _TRITON_STEP = siso_step
        except Exception:  # pragma: no cover - optional runtime import
            _TRITON_STEP = None
    if _TRITON_STEP is None:
        return None
    try:
        result = _TRITON_STEP(
            q,
            k,
            value,
            gate,
            adt,
            dt,
            trap_logits,
            angle_rate,
            D,
            cache,
        )
        _TRITON_LAST_DISPATCH = True
        return result
    except Exception as error:  # pragma: no cover - backend and toolchain dependent
        _TRITON_FAILURES.add(failure_key)
        warnings.warn(
            f"fused Mamba-3 decoding is unavailable; using PyTorch ({error})",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def _rank_values(
    value: torch.Tensor,
    projection: torch.Tensor | None,
) -> torch.Tensor:
    if projection is None:
        return value.unsqueeze(3)
    return value.unsqueeze(3) * projection.to(value.dtype).unsqueeze(0).unsqueeze(0)


class _BmmFloat(torch.autograd.Function):
    """CUDA low-precision GEMM with an FP32 output and explicit backward."""

    @staticmethod
    def forward(ctx, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            ctx.save_for_backward(left, right)
        return torch.bmm(left, right, out_dtype=torch.float32)

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        left, right = ctx.saved_tensors
        grad_left = grad_right = None
        low_precision_cuda = (
            left.is_cuda
            and left.dtype in (torch.float16, torch.bfloat16)
            and right.dtype == left.dtype
            and _BMM_OUT_DTYPE_SUPPORTED
        )
        with torch.autocast(left.device.type, enabled=False):
            if low_precision_cuda:
                # Mirror the forward: low-precision tensor-core GEMMs with
                # FP32 accumulation. Only the incoming FP32 gradient needs a
                # precision-preserving cast; the operands are already low
                # precision.
                if ctx.needs_input_grad[0]:
                    grad_left = torch.bmm(
                        grad_output.to(left.dtype),
                        right.transpose(1, 2),
                        out_dtype=torch.float32,
                    ).to(left.dtype)
                if ctx.needs_input_grad[1]:
                    grad_right = torch.bmm(
                        left.transpose(1, 2),
                        grad_output.to(right.dtype),
                        out_dtype=torch.float32,
                    ).to(right.dtype)
            else:
                if ctx.needs_input_grad[0]:
                    grad_left = torch.bmm(
                        grad_output.float(), right.float().transpose(1, 2)
                    ).to(left.dtype)
                if ctx.needs_input_grad[1]:
                    grad_right = torch.bmm(
                        left.float().transpose(1, 2), grad_output.float()
                    ).to(right.dtype)
        return grad_left, grad_right


def _batched_matmul(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    accumulate_float: bool = False,
) -> torch.Tensor:
    """Run one batched GEMM while preserving the leading batch/head axes."""

    leading = left.shape[:-2]
    if leading != right.shape[:-2]:
        raise ValueError("batched matrix operands must have matching leading axes")
    left = left.reshape(-1, left.shape[-2], left.shape[-1])
    right = right.reshape(-1, right.shape[-2], right.shape[-1])
    low_precision_cuda = (
        left.is_cuda
        and left.dtype in (torch.float16, torch.bfloat16)
        and right.dtype == left.dtype
    )
    if accumulate_float and low_precision_cuda and _BMM_OUT_DTYPE_SUPPORTED:
        result = _BmmFloat.apply(left, right)
    elif accumulate_float:
        with torch.autocast(left.device.type, enabled=False):
            result = torch.bmm(left.float(), right.float())
    else:
        result = torch.bmm(left, right)
    return result.reshape(*leading, left.shape[-2], right.shape[-1])


def _outer_sum(value: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Sum rank outer products into a head state with FP32 accumulation."""

    return _batched_matmul(
        value.transpose(-1, -2), key, accumulate_float=True
    )


def mamba3_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    adt: torch.Tensor,
    dt: torch.Tensor,
    trap_logits: torch.Tensor,
    D: torch.Tensor,
    *,
    mimo_x: torch.Tensor | None = None,
    mimo_z: torch.Tensor | None = None,
    mimo_out: torch.Tensor | None = None,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    initial_k: torch.Tensor | None = None,
    initial_value: torch.Tensor | None = None,
    phase: torch.Tensor | None = None,
    mimo_rotation: bool = False,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parallel chunked Mamba-3 SSD with FP32 recurrent state.

    Shapes are ``q/k: [B,L,G,R,N]``, ``value/gate: [B,L,H,P]`` and
    ``adt/dt/trap_logits: [B,L,H]``. The algorithm performs only one Python
    iteration per chunk; all token and rank interactions inside a chunk are
    batched GEMMs (tensor-core accelerated on supported low-precision CUDA).
    """

    if q.shape != k.shape or q.ndim != 5:
        raise ValueError("q and k must have matching [B, L, G, R, N] shapes")
    batch, length, qk_heads, rank, d_state = q.shape
    if value.ndim != 4 or gate.shape != value.shape:
        raise ValueError("value and gate must have matching [B, L, H, P] shapes")
    if value.shape[:2] != (batch, length):
        raise ValueError("value dimensions must match q and k")
    heads = value.shape[2]
    if heads % qk_heads != 0:
        raise ValueError("value heads must be divisible by Q/K groups")
    if adt.shape != (batch, length, heads):
        raise ValueError("adt must have shape [B, L, H]")
    if dt.shape != adt.shape or trap_logits.shape != adt.shape:
        raise ValueError("dt and trap_logits must match adt")
    if D.shape != (heads,):
        raise ValueError("D must have shape [H]")
    if (q_bias is None) != (k_bias is None):
        raise ValueError("q_bias and k_bias must be provided together")
    if q_bias is not None:
        expected_bias = (heads, rank, d_state)
        if q_bias.shape != expected_bias or k_bias.shape != expected_bias:
            raise ValueError(f"Q/K biases must have shape {expected_bias}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if phase is not None and (
        phase.ndim != 4 or phase.shape[:3] != (batch, length, heads)
    ):
        raise ValueError("phase must have shape [B, L, H, S]")

    head_dim = value.shape[-1]
    expected_state = (batch, heads, head_dim, d_state)
    if initial_state is None:
        state = torch.zeros(expected_state, device=value.device, dtype=torch.float32)
    elif initial_state.shape != expected_state:
        raise ValueError(f"initial_state must have shape {expected_state}")
    else:
        state = initial_state.float()

    projections = (mimo_x, mimo_z, mimo_out)
    if rank == 1:
        if any(projection is not None for projection in projections):
            raise ValueError("SISO inputs must not provide MIMO projections")
    else:
        expected_projection = (heads, rank, head_dim)
        if any(
            projection is None or projection.shape != expected_projection
            for projection in projections
        ):
            raise ValueError(f"MIMO projections must have shape {expected_projection}")

    endpoint_cache = initial_k is not None or initial_value is not None
    if endpoint_cache:
        expected_k = (batch, heads, rank, d_state)
        expected_value = (batch, heads, head_dim)
        if initial_k is None or initial_k.shape != expected_k:
            raise ValueError(f"initial_k must have shape {expected_k}")
        if initial_value is None or initial_value.shape != expected_value:
            raise ValueError(f"initial_value must have shape {expected_value}")
    if length == 0:
        return value.clone(), state

    trap = torch.sigmoid(trap_logits.float())
    gamma = dt * trap
    next_endpoint = torch.cat(
        (dt[:, 1:] * (1.0 - trap[:, 1:]), torch.zeros_like(dt[:, :1])),
        dim=1,
    )
    scale = gamma + next_endpoint
    # The chunk states compose only when ``scale`` carries the next token's
    # left trapezoid endpoint; same-token outputs must see only gamma. Folding
    # the ratio into the decay diagonal below makes both true at once.
    diag_decay = gamma / scale
    mimo_x_activation = mimo_x.to(value.dtype) if mimo_x is not None else None
    mimo_z_activation = mimo_z.to(gate.dtype) if mimo_z is not None else None
    mimo_out_float = mimo_out.float() if mimo_out is not None else None
    D_float = D.float()
    q_bias_activation = q_bias.to(q.dtype) if q_bias is not None else None
    k_bias_activation = k_bias.to(k.dtype) if k_bias is not None else None

    # A continued segment starts with the previous endpoint contribution. It
    # is multiplied by alpha at the first token by the inter-chunk path below.
    if endpoint_cache:
        previous_rank_value = _rank_values(initial_value.unsqueeze(1), mimo_x).squeeze(1)
        previous_outer = _outer_sum(previous_rank_value, initial_k)
        state = state + (
            dt[:, 0] * (1.0 - trap[:, 0])
        ).unsqueeze(-1).unsqueeze(-1) * previous_outer

    outputs: list[torch.Tensor] = []
    causal_mask = torch.ones(
        min(length, chunk_size),
        min(length, chunk_size),
        dtype=torch.bool,
        device=q.device,
    ).tril()
    # Expansion, bias, and rotation act per token, so they are hoisted out of
    # the chunk loop; every chunk then starts from a simple slice.
    if qk_heads == 1 and heads != 1:
        q_full = q.expand(-1, -1, heads, -1, -1)
        k_full = k.expand(-1, -1, heads, -1, -1)
    elif qk_heads != heads:
        repeats = heads // qk_heads
        q_full = q.repeat_interleave(repeats, dim=2)
        k_full = k.repeat_interleave(repeats, dim=2)
    else:
        q_full = q
        k_full = k
    if q_bias is not None:
        q_full = q_full + q_bias_activation[None, None]
        k_full = k_full + k_bias_activation[None, None]
    if phase is not None:
        q_full, k_full = rotate_qk(q_full, k_full, phase, mimo=mimo_rotation)

    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        width = end - start

        q_chunk = q_full[:, start:end].permute(0, 2, 1, 3, 4)
        k_chunk = k_full[:, start:end].permute(0, 2, 1, 3, 4)
        base_value_chunk = value[:, start:end].permute(0, 2, 1, 3)
        base_gate_chunk = gate[:, start:end].permute(0, 2, 1, 3)
        if mimo_x is None:
            value_chunk = base_value_chunk.unsqueeze(3)
            gate_chunk = base_gate_chunk.unsqueeze(3)
        else:
            value_chunk = (
                base_value_chunk.unsqueeze(3)
                * mimo_x_activation[None, :, None]
            )
            gate_chunk = (
                base_gate_chunk.unsqueeze(3)
                * mimo_z_activation[None, :, None]
            )
        adt_chunk = adt[:, start:end].permute(0, 2, 1).float()
        scale_chunk = scale[:, start:end].permute(0, 2, 1)

        cumulative = torch.cumsum(adt_chunk, dim=-1)
        prefix_decay = torch.exp(cumulative)

        q_flat = q_chunk.flatten(2, 3)
        k_scaled = k_chunk * scale_chunk[..., None, None].to(k_chunk.dtype)
        k_flat = k_scaled.flatten(2, 3)
        value_flat = value_chunk.flatten(2, 3)

        # Contribution from all earlier chunks.
        if start == 0 and initial_state is None and not endpoint_cache:
            inter = torch.zeros(
                batch, heads, width, rank, head_dim, device=q.device
            )
        else:
            inter = _batched_matmul(
                state.to(dtype=q.dtype),
                q_flat.transpose(-1, -2),
                accumulate_float=True,
            ).transpose(-1, -2)
            inter = inter.reshape(batch, heads, width, rank, head_dim)
            inter = inter * prefix_decay[..., None, None]

        # Contributions within this chunk. Rank is flattened into the token
        # axis so one GEMM handles both SISO and MIMO cross-rank interactions.
        scores = _batched_matmul(
            q_flat, k_flat.transpose(-1, -2), accumulate_float=True
        )
        causal = causal_mask[:width, :width]
        decay = torch.exp(cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2))
        decay = decay.masked_fill(~causal, 0.0)
        # ``scale`` includes the next token's left endpoint so chunk states
        # compose; the diagonal is set to gamma/scale so that the same-token
        # output sees exactly gamma without a separate correction GEMM.
        decay.diagonal(0, -2, -1).copy_(
            diag_decay[:, start:end].permute(0, 2, 1)
        )
        scores = scores.reshape(batch, heads, width, rank, width, rank)
        weighted_scores = (
            scores * decay[:, :, :, None, :, None]
        ).reshape(batch, heads, width * rank, width * rank).to(value_flat.dtype)
        intra = _batched_matmul(
            weighted_scores, value_flat, accumulate_float=True
        ).reshape(batch, heads, width, rank, head_dim)

        mixed = inter + intra
        mixed = mixed + D_float[None, :, None, None, None] * value_chunk.float()

        mixed = mixed * F.silu(gate_chunk.float())
        if mimo_out_float is not None:
            mixed = (
                mixed * mimo_out_float[None, :, None]
            ).sum(dim=3)
        else:
            mixed = mixed.squeeze(3)
        outputs.append(mixed.permute(0, 2, 1, 3).to(value.dtype))

        # Carry the pre-state for the next chunk. At the end of the full
        # sequence next_endpoint is zero, so this is the exact recurrent state.
        end_decay = torch.exp(cumulative[..., -1:] - cumulative)
        state_k = k_scaled * end_decay[..., None, None].to(k_scaled.dtype)
        state_update = _batched_matmul(
            value_flat.transpose(-1, -2),
            state_k.flatten(2, 3),
            accumulate_float=True,
        )
        state = (
            torch.exp(cumulative[..., -1]).unsqueeze(-1).unsqueeze(-1) * state
            + state_update
        )

    if outputs:
        output = torch.cat(outputs, dim=1).to(dtype=value.dtype)
    else:
        output = value.new_empty(value.shape)
    return output, state


def mamba3_step(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    adt: torch.Tensor,
    dt: torch.Tensor,
    trap_logits: torch.Tensor,
    D: torch.Tensor,
    cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    mimo_x: torch.Tensor | None = None,
    mimo_z: torch.Tensor | None = None,
    mimo_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """One exact recurrent Mamba-3 update.

    The phase has already been updated and Q/K rotated. ``cache`` contains
    phase, FP32 SSM state, previous rotated K, and previous unexpanded value.
    """

    phase, state, previous_k, previous_value = cache
    rank_value = _rank_values(value.unsqueeze(1), mimo_x).squeeze(1)
    rank_gate = _rank_values(gate.unsqueeze(1), mimo_z).squeeze(1)
    previous_rank_value = _rank_values(previous_value.unsqueeze(1), mimo_x).squeeze(1)

    trap = torch.sigmoid(trap_logits.float())
    alpha = torch.exp(adt.float())
    beta = (1.0 - trap) * dt * alpha
    gamma = trap * dt

    previous_outer = _outer_sum(previous_rank_value, previous_k)
    current_outer = _outer_sum(rank_value, k)
    next_state = (
        alpha[..., None, None] * state.float()
        + beta[..., None, None] * previous_outer
        + gamma[..., None, None] * current_outer
    )

    rank_output = _batched_matmul(
        q,
        next_state.to(dtype=q.dtype).transpose(-1, -2),
        accumulate_float=True,
    )
    rank_output = rank_output + D[None, :, None, None].float() * rank_value.float()
    rank_output = rank_output * F.silu(rank_gate.float())
    if mimo_out is not None:
        output = (
            rank_output * mimo_out.unsqueeze(0).float()
        ).sum(dim=2)
    else:
        output = rank_output.squeeze(2)

    next_cache = (phase, next_state, k, value)
    return output.to(dtype=value.dtype), next_cache
