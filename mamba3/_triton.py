# Copyright (c) 2025, Dao AI Lab, Goombalab
# Modified by the mamba3 project to fuse and simplify the decoding interface.
"""Optional fused CUDA decoding kernel.

The kernel structure is adapted from the Apache-2.0 Mamba-3 SISO step kernel
by Dao AI Lab and Goombalab, with a smaller interface and explicit fallbacks.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _tanh_approx(x):
    return tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _cos_approx(x):
    return tl.inline_asm_elementwise(
        "cos.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sin_approx(x):
    return tl.inline_asm_elementwise(
        "sin.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _siso_step_kernel(
    q,
    k,
    value,
    gate,
    adt,
    dt,
    trap_logits,
    angle_rate,
    D,
    phase,
    state,
    previous_k,
    previous_value,
    output,
    output_phase,
    output_state,
    output_k,
    q_stride_batch,
    q_stride_head,
    q_stride_state,
    k_stride_batch,
    k_stride_head,
    k_stride_state,
    value_stride_batch,
    value_stride_head,
    value_stride_dim,
    gate_stride_batch,
    gate_stride_head,
    gate_stride_dim,
    scalar_stride_batch,
    scalar_stride_head,
    angle_stride_batch,
    angle_stride_dim,
    phase_stride_batch,
    phase_stride_head,
    phase_stride_dim,
    state_stride_batch,
    state_stride_head,
    state_stride_dim,
    state_stride_state,
    previous_k_stride_batch,
    previous_k_stride_head,
    previous_k_stride_state,
    previous_value_stride_batch,
    previous_value_stride_head,
    previous_value_stride_dim,
    output_stride_batch,
    output_stride_head,
    output_stride_dim,
    output_phase_stride_batch,
    output_phase_stride_head,
    output_phase_stride_dim,
    output_state_stride_batch,
    output_state_stride_head,
    output_state_stride_dim,
    output_state_stride_state,
    output_k_stride_batch,
    output_k_stride_head,
    output_k_stride_state,
    D_STATE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_ANGLES: tl.constexpr,
    USE_BF16_DOT: tl.constexpr,
    USE_FP16_DOT: tl.constexpr,
):
    head = tl.program_id(0)
    batch = tl.program_id(1)
    state_offsets = tl.arange(0, D_STATE)
    dim_offsets = tl.arange(0, HEAD_DIM)
    pair_offsets = tl.arange(0, D_STATE // 2)

    q_pointer = q + batch * q_stride_batch + head * q_stride_head
    k_pointer = k + batch * k_stride_batch + head * k_stride_head
    value_pointer = value + batch * value_stride_batch + head * value_stride_head
    gate_pointer = gate + batch * gate_stride_batch + head * gate_stride_head
    scalar_offset = batch * scalar_stride_batch + head * scalar_stride_head
    phase_pointer = phase + batch * phase_stride_batch + head * phase_stride_head
    state_pointer = state + batch * state_stride_batch + head * state_stride_head
    previous_k_pointer = (
        previous_k
        + batch * previous_k_stride_batch
        + head * previous_k_stride_head
    )
    previous_value_pointer = (
        previous_value
        + batch * previous_value_stride_batch
        + head * previous_value_stride_head
    )

    dt_value = tl.load(dt + scalar_offset).to(tl.float32)
    raw_angle = tl.load(
        angle_rate
        + batch * angle_stride_batch
        + pair_offsets * angle_stride_dim,
        mask=pair_offsets < NUM_ANGLES,
        other=0.0,
    ).to(tl.float32)
    old_phase = tl.load(
        phase_pointer + pair_offsets * phase_stride_dim,
        mask=pair_offsets < NUM_ANGLES,
        other=0.0,
    ).to(tl.float32)
    phase_increment = math.pi * _tanh_approx(raw_angle) * dt_value
    phase_increment -= (2.0 * math.pi) * tl.where(
        phase_increment >= 0,
        tl.floor(phase_increment / (2.0 * math.pi)),
        tl.ceil(phase_increment / (2.0 * math.pi)),
    )
    new_phase = old_phase + phase_increment
    new_phase -= (2.0 * math.pi) * tl.where(
        new_phase >= 0,
        tl.floor(new_phase / (2.0 * math.pi)),
        tl.ceil(new_phase / (2.0 * math.pi)),
    )
    tl.store(
        output_phase
        + batch * output_phase_stride_batch
        + head * output_phase_stride_head
        + pair_offsets * output_phase_stride_dim,
        new_phase,
        mask=pair_offsets < NUM_ANGLES,
    )

    cosine = _cos_approx(new_phase)
    sine = _sin_approx(new_phase)
    q_raw = tl.load(q_pointer + state_offsets * q_stride_state)
    k_raw = tl.load(k_pointer + state_offsets * k_stride_state)
    q_first, q_second = tl.split(tl.reshape(q_raw, [D_STATE // 2, 2]))
    k_first, k_second = tl.split(tl.reshape(k_raw, [D_STATE // 2, 2]))
    q_first_rotated = q_first * cosine - q_second * sine
    q_second_rotated = q_first * sine + q_second * cosine
    k_first_rotated = k_first * cosine - k_second * sine
    k_second_rotated = k_first * sine + k_second * cosine
    q_rotated = tl.reshape(
        tl.join(q_first_rotated, q_second_rotated), [D_STATE]
    ).to(q_raw.dtype)
    k_rotated = tl.reshape(
        tl.join(k_first_rotated, k_second_rotated), [D_STATE]
    ).to(k_raw.dtype)
    old_k = tl.load(previous_k_pointer + state_offsets * previous_k_stride_state)
    tl.store(
        output_k
        + batch * output_k_stride_batch
        + head * output_k_stride_head
        + state_offsets * output_k_stride_state,
        k_rotated,
    )

    current_value = tl.load(value_pointer + dim_offsets * value_stride_dim)
    old_value = tl.load(
        previous_value_pointer + dim_offsets * previous_value_stride_dim
    )
    decay = tl.load(adt + scalar_offset).to(tl.float32) * 1.4426950408889634
    trap = tl.sigmoid(tl.load(trap_logits + scalar_offset).to(tl.float32))
    alpha = tl.exp2(decay)
    beta = alpha * dt_value * (1.0 - trap)
    gamma = dt_value * trap

    old_state = tl.load(
        state_pointer
        + dim_offsets[:, None] * state_stride_dim
        + state_offsets[None, :] * state_stride_state
    ).to(tl.float32)
    next_state = (
        alpha * old_state
        + (beta * old_value)[:, None] * old_k[None, :]
        + (gamma * current_value)[:, None] * k_rotated[None, :]
    )
    tl.store(
        output_state
        + batch * output_state_stride_batch
        + head * output_state_stride_head
        + dim_offsets[:, None] * output_state_stride_dim
        + state_offsets[None, :] * output_state_stride_state,
        next_state,
    )

    q_column = tl.reshape(q_rotated, [D_STATE, 1])
    if USE_BF16_DOT:
        mixed = tl.dot(next_state.to(tl.bfloat16), q_column.to(tl.bfloat16))
    elif USE_FP16_DOT:
        mixed = tl.dot(next_state.to(tl.float16), q_column.to(tl.float16))
    else:
        mixed = tl.sum(next_state * q_rotated[None, :], axis=1)[:, None]
    mixed = tl.reshape(mixed, [HEAD_DIM]).to(tl.float32)
    mixed += tl.load(D + head).to(tl.float32) * current_value
    gate_value = tl.load(gate_pointer + dim_offsets * gate_stride_dim).to(tl.float32)
    mixed *= gate_value * tl.sigmoid(gate_value)
    tl.store(
        output
        + batch * output_stride_batch
        + head * output_stride_head
        + dim_offsets * output_stride_dim,
        mixed,
    )


def siso_step(
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
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Run one fused SISO update; inputs are normalized but not yet rotated."""

    phase, state, previous_k, previous_value = cache
    batch, heads, d_state = q.shape
    head_dim = value.shape[-1]
    num_angles = angle_rate.shape[-1]
    if num_angles <= 0 or num_angles > d_state // 2:
        raise ValueError("angle width must be in [1, d_state / 2]")
    expected_shapes = {
        "k": (batch, heads, d_state),
        "value": (batch, heads, head_dim),
        "gate": (batch, heads, head_dim),
        "adt": (batch, heads),
        "dt": (batch, heads),
        "trap_logits": (batch, heads),
        "angle_rate": (batch, num_angles),
        "D": (heads,),
        "phase": (batch, heads, num_angles),
        "state": (batch, heads, head_dim, d_state),
        "previous_k": (batch, heads, 1, d_state),
        "previous_value": (batch, heads, head_dim),
    }
    tensors_by_name = {
        "k": k,
        "value": value,
        "gate": gate,
        "adt": adt,
        "dt": dt,
        "trap_logits": trap_logits,
        "angle_rate": angle_rate,
        "D": D,
        "phase": phase,
        "state": state,
        "previous_k": previous_k,
        "previous_value": previous_value,
    }
    for name, expected in expected_shapes.items():
        if tensors_by_name[name].shape != expected:
            raise ValueError(
                f"invalid {name} shape: expected {expected}, "
                f"got {tuple(tensors_by_name[name].shape)}"
            )
    if q.shape != k.shape:
        raise ValueError("q and k must have the same shape")
    if not (q.dtype == k.dtype == value.dtype == gate.dtype):
        raise ValueError("q, k, value, and gate must have the same dtype")
    if previous_k.dtype != q.dtype or previous_value.dtype != value.dtype:
        raise ValueError("K/value cache dtypes must match current activations")
    if phase.dtype != torch.float32 or state.dtype != torch.float32:
        raise ValueError("phase and state must be float32")

    tensors = (q, k, value, gate, adt, dt, trap_logits, angle_rate, D, *cache)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("fused SISO step requires CUDA tensors")
    if any(tensor.device != q.device for tensor in tensors):
        raise ValueError("fused SISO step tensors must be on the same CUDA device")

    q = q.contiguous()
    k = k.contiguous()
    value = value.contiguous()
    gate = gate.contiguous()
    adt = adt.contiguous()
    dt = dt.contiguous()
    trap_logits = trap_logits.contiguous()
    angle_rate = angle_rate.contiguous()
    previous_k_siso = previous_k.squeeze(2).contiguous()
    previous_value = previous_value.contiguous()

    output = torch.empty_like(value)
    output_phase = torch.empty_like(phase)
    output_state = torch.empty_like(state)
    output_k = torch.empty_like(q)
    num_warps = 8 if d_state * head_dim >= 4096 else 4
    kernel = _siso_step_kernel[(heads, batch)]
    with torch.cuda.device(q.device):
        kernel(
            q,
            k,
            value,
            gate,
            adt,
            dt,
            trap_logits,
            angle_rate,
            D,
            phase,
            state,
            previous_k_siso,
            previous_value,
            output,
            output_phase,
            output_state,
            output_k,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            gate.stride(0),
            gate.stride(1),
            gate.stride(2),
            adt.stride(0),
            adt.stride(1),
            angle_rate.stride(0),
            angle_rate.stride(1),
            phase.stride(0),
            phase.stride(1),
            phase.stride(2),
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            previous_k_siso.stride(0),
            previous_k_siso.stride(1),
            previous_k_siso.stride(2),
            previous_value.stride(0),
            previous_value.stride(1),
            previous_value.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output_phase.stride(0),
            output_phase.stride(1),
            output_phase.stride(2),
            output_state.stride(0),
            output_state.stride(1),
            output_state.stride(2),
            output_state.stride(3),
            output_k.stride(0),
            output_k.stride(1),
            output_k.stride(2),
            D_STATE=d_state,
            HEAD_DIM=head_dim,
            NUM_ANGLES=phase.shape[-1],
            USE_BF16_DOT=q.dtype == torch.bfloat16,
            USE_FP16_DOT=q.dtype == torch.float16,
            num_warps=num_warps,
            num_stages=1,
        )
    return output, (output_phase, output_state, output_k.unsqueeze(2), value)
