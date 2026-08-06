# Copyright (c) 2025, Dao AI Lab, Goombalab
# Modified by the mamba3 project to fuse and simplify the decoding interface.
"""Optional fused CUDA kernels.

The step kernel structure is adapted from the Apache-2.0 Mamba-3 SISO step
kernel by Dao AI Lab and Goombalab, with a smaller interface, explicit
fallbacks, and MIMO support. The chunked-scan kernels are original and
replicate the exact chunked SSD recurrence of ``mamba3.ops.mamba3_scan`` in
one launch per chunk.
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
def _advance_phase(raw_angle, old_phase, dt_value):
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
    return new_phase


@triton.jit
def _chunk_main_kernel(
    q_ptr,
    k_ptr,
    value_ptr,
    gate_ptr,
    cum_ptr,
    scale_ptr,
    diag_ptr,
    D_ptr,
    state_ptr,
    out_ptr,
    q_sb,
    q_sh,
    q_st,
    q_sr,
    q_sn,
    k_sb,
    k_sh,
    k_st,
    k_sr,
    k_sn,
    v_sb,
    v_sh,
    v_st,
    v_sr,
    v_sp,
    g_sb,
    g_sh,
    g_st,
    g_sr,
    g_sp,
    cum_sb,
    cum_st,
    cum_sh,
    scale_sb,
    scale_st,
    scale_sh,
    diag_sb,
    diag_st,
    diag_sh,
    state_sb,
    state_sh,
    state_sp,
    state_sn,
    out_sb,
    out_sh,
    out_st,
    out_sr,
    out_sp,
    D_STATE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    RANK: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    NUM_JBLOCKS: tl.constexpr,
    WIDTH: tl.constexpr,
    USE_BF16_DOT: tl.constexpr,
    USE_FP16_DOT: tl.constexpr,
):
    """Fuse one chunk of the SSD into a single launch.

    Each program handles ``ROW_BLOCK`` flattened ``(token, rank)`` rows of one
    head and accumulates the intra-chunk contributions in a causal loop over
    source row blocks, then adds the inter-chunk state contribution, the D
    term, and the SiLU gate. Rows are flattened as ``row = token * RANK +
    rank`` so token-level causality is a simple block skip. The diagonal
    carries ``gamma / scale`` (the same-token endpoint fold) and same-token
    cross-rank pairs use it too, exactly like the PyTorch path.
    """

    b = tl.program_id(0)
    h = tl.program_id(1)
    ib = tl.program_id(2)

    row_offsets = tl.arange(0, ROW_BLOCK)
    rows = ib * ROW_BLOCK + row_offsets
    row_mask = rows < WIDTH * RANK
    tokens = rows // RANK
    ranks = rows % RANK

    state_offsets = tl.arange(0, D_STATE)
    dim_offsets = tl.arange(0, HEAD_DIM)

    q_rows = b * q_sb + tokens * q_st + h * q_sh + ranks * q_sr
    q_block = tl.load(
        q_ptr + q_rows[:, None] + state_offsets[None, :] * q_sn,
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    if USE_BF16_DOT:
        q_dot = q_block.to(tl.bfloat16)
    elif USE_FP16_DOT:
        q_dot = q_block.to(tl.float16)
    else:
        q_dot = q_block

    old_state = tl.load(
        state_ptr
        + b * state_sb
        + h * state_sh
        + dim_offsets[:, None] * state_sp
        + state_offsets[None, :] * state_sn
    ).to(tl.float32)
    if USE_BF16_DOT:
        state_dot = old_state.to(tl.bfloat16)
    elif USE_FP16_DOT:
        state_dot = old_state.to(tl.float16)
    else:
        state_dot = old_state

    cum_rows = tl.load(
        cum_ptr + b * cum_sb + tokens * cum_st + h * cum_sh,
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)
    prefix = tl.exp2(cum_rows * 1.4426950408889634)
    inter = tl.dot(q_dot, tl.trans(state_dot)) * prefix[:, None]

    diag_rows = tl.load(
        diag_ptr + b * diag_sb + tokens * diag_st + h * diag_sh,
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)
    D_value = tl.load(D_ptr + h).to(tl.float32)

    intra = tl.zeros([ROW_BLOCK, HEAD_DIM], dtype=tl.float32)
    for jb in tl.static_range(NUM_JBLOCKS):
        if jb <= ib:
            j_rows = jb * ROW_BLOCK + row_offsets
            j_mask = j_rows < WIDTH * RANK
            j_tokens = j_rows // RANK
            j_ranks = j_rows % RANK

            k_rows = b * k_sb + j_tokens * k_st + h * k_sh + j_ranks * k_sr
            k_block = tl.load(
                k_ptr + k_rows[:, None] + state_offsets[None, :] * k_sn,
                mask=j_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            scale_rows = tl.load(
                scale_ptr + b * scale_sb + j_tokens * scale_st + h * scale_sh,
                mask=j_mask,
                other=0.0,
            ).to(tl.float32)
            k_scaled = k_block * scale_rows[:, None]
            if USE_BF16_DOT:
                k_dot = k_scaled.to(tl.bfloat16)
            elif USE_FP16_DOT:
                k_dot = k_scaled.to(tl.float16)
            else:
                k_dot = k_scaled
            scores = tl.dot(q_dot, tl.trans(k_dot))

            cum_j = tl.load(
                cum_ptr + b * cum_sb + j_tokens * cum_st + h * cum_sh,
                mask=j_mask,
                other=0.0,
            ).to(tl.float32)
            decay = tl.exp2(
                (cum_rows[:, None] - cum_j[None, :]) * 1.4426950408889634
            )
            decay = tl.where(tokens[:, None] >= j_tokens[None, :], decay, 0.0)
            decay = tl.where(
                tokens[:, None] == j_tokens[None, :],
                diag_rows[:, None],
                decay,
            )

            weighted = scores * decay
            if USE_BF16_DOT:
                weighted_dot = weighted.to(tl.bfloat16)
            elif USE_FP16_DOT:
                weighted_dot = weighted.to(tl.float16)
            else:
                weighted_dot = weighted
            v_rows = b * v_sb + h * v_sh + j_tokens * v_st + j_ranks * v_sr
            value_block = tl.load(
                value_ptr + v_rows[:, None] + dim_offsets[None, :] * v_sp,
                mask=j_mask[:, None],
                other=0.0,
            )
            intra += tl.dot(weighted_dot, value_block)

    value_self = tl.load(
        value_ptr
        + b * v_sb
        + h * v_sh
        + tokens[:, None] * v_st
        + ranks[:, None] * v_sr
        + dim_offsets[None, :] * v_sp,
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    gate_value = tl.load(
        gate_ptr
        + b * g_sb
        + h * g_sh
        + tokens[:, None] * g_st
        + ranks[:, None] * g_sr
        + dim_offsets[None, :] * g_sp,
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    mixed = inter + intra + D_value * value_self
    mixed = mixed * gate_value * tl.sigmoid(gate_value)

    out_rows = b * out_sb + h * out_sh + tokens * out_st + ranks * out_sr
    if USE_BF16_DOT:
        mixed_out = mixed.to(tl.bfloat16)
    elif USE_FP16_DOT:
        mixed_out = mixed.to(tl.float16)
    else:
        mixed_out = mixed
    tl.store(
        out_ptr + out_rows[:, None] + dim_offsets[None, :] * out_sp,
        mixed_out,
        mask=row_mask[:, None],
    )


@triton.jit
def _chunk_state_kernel(
    k_ptr,
    value_ptr,
    cum_ptr,
    scale_ptr,
    state_ptr,
    out_ptr,
    k_sb,
    k_sh,
    k_st,
    k_sr,
    k_sn,
    v_sb,
    v_sh,
    v_st,
    v_sr,
    v_sp,
    cum_sb,
    cum_st,
    cum_sh,
    scale_sb,
    scale_st,
    scale_sh,
    state_sb,
    state_sh,
    state_sp,
    state_sn,
    out_sb,
    out_sh,
    out_sp,
    out_sn,
    D_STATE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    RANK: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    NUM_RCHUNKS: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    WIDTH: tl.constexpr,
    USE_BF16_DOT: tl.constexpr,
    USE_FP16_DOT: tl.constexpr,
):
    """Update one head state after one chunk: ``exp(c) * S + U^T K_scaled``."""

    b = tl.program_id(0)
    h = tl.program_id(1)

    state_offsets = tl.arange(0, D_STATE)
    dim_offsets = tl.arange(0, HEAD_DIM)
    row_offsets = tl.arange(0, ROW_BLOCK)
    token_offsets = tl.arange(0, TOKEN_BLOCK)

    t_mask = token_offsets < WIDTH
    # adt is strictly negative, so cumulative sums decrease monotonically and
    # the last real token is the minimum; masked positions are +inf.
    cum_t = tl.load(
        cum_ptr + b * cum_sb + token_offsets * cum_st + h * cum_sh,
        mask=t_mask,
        other=float("inf"),
    ).to(tl.float32)
    cum_last = tl.min(cum_t, axis=0)
    final_scale = tl.exp2(cum_last * 1.4426950408889634)
    end_decay = tl.exp2((cum_last - cum_t) * 1.4426950408889634)
    end_decay = tl.where(t_mask, end_decay, 0.0)

    acc = tl.zeros([HEAD_DIM, D_STATE], dtype=tl.float32)
    for jh in tl.static_range(NUM_RCHUNKS):
        rows = jh * ROW_BLOCK + row_offsets
        r_mask = rows < WIDTH * RANK
        tokens = rows // RANK
        ranks = rows % RANK

        k_rows = b * k_sb + tokens * k_st + h * k_sh + ranks * k_sr
        k_block = tl.load(
            k_ptr + k_rows[:, None] + state_offsets[None, :] * k_sn,
            mask=r_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        scale_rows = tl.load(
            scale_ptr + b * scale_sb + tokens * scale_st + h * scale_sh,
            mask=r_mask,
            other=0.0,
        ).to(tl.float32)
        cum_rows = tl.load(
            cum_ptr + b * cum_sb + tokens * cum_st + h * cum_sh,
            mask=r_mask,
            other=0.0,
        ).to(tl.float32)
        end_rows = tl.exp2((cum_last - cum_rows) * 1.4426950408889634)
        k_scaled = k_block * scale_rows[:, None] * end_rows[:, None]
        if USE_BF16_DOT:
            k_dot = k_scaled.to(tl.bfloat16)
        elif USE_FP16_DOT:
            k_dot = k_scaled.to(tl.float16)
        else:
            k_dot = k_scaled
        v_rows = b * v_sb + h * v_sh + tokens * v_st + ranks * v_sr
        value_block = tl.load(
            value_ptr + v_rows[:, None] + dim_offsets[None, :] * v_sp,
            mask=r_mask[:, None],
            other=0.0,
        )
        acc += tl.dot(tl.trans(value_block), k_dot)

    old_state = tl.load(
        state_ptr
        + b * state_sb
        + h * state_sh
        + dim_offsets[:, None] * state_sp
        + state_offsets[None, :] * state_sn
    ).to(tl.float32)
    next_state = acc + final_scale * old_state
    tl.store(
        out_ptr
        + b * out_sb
        + h * out_sh
        + dim_offsets[:, None] * out_sp
        + state_offsets[None, :] * out_sn,
        next_state,
    )


def chunk_main(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    cumulative: torch.Tensor,
    scale: torch.Tensor,
    diag_decay: torch.Tensor,
    D: torch.Tensor,
    state: torch.Tensor,
    output: torch.Tensor,
    *,
    width: int,
    rank: int,
) -> None:
    """Run one fused SSD chunk; shapes follow ``mamba3.ops.mamba3_scan``."""

    batch, heads, head_dim, d_state = state.shape
    row_block = 64
    num_jblocks = triton.cdiv(width * rank, row_block)
    kernel = _chunk_main_kernel[(batch, heads, num_jblocks)]
    with torch.cuda.device(q.device):
        kernel(
            q,
            k,
            value,
            gate,
            cumulative,
            scale,
            diag_decay,
            D,
            state,
            output,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            q.stride(4),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            k.stride(4),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            value.stride(3),
            value.stride(4),
            gate.stride(0),
            gate.stride(1),
            gate.stride(2),
            gate.stride(3),
            gate.stride(4),
            cumulative.stride(0),
            cumulative.stride(1),
            cumulative.stride(2),
            scale.stride(0),
            scale.stride(1),
            scale.stride(2),
            diag_decay.stride(0),
            diag_decay.stride(1),
            diag_decay.stride(2),
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            output.stride(4),
            D_STATE=d_state,
            HEAD_DIM=head_dim,
            RANK=rank,
            ROW_BLOCK=row_block,
            NUM_JBLOCKS=num_jblocks,
            WIDTH=width,
            USE_BF16_DOT=q.dtype == torch.bfloat16,
            USE_FP16_DOT=q.dtype == torch.float16,
            num_warps=8,
            num_stages=3,
        )


def chunk_state(
    k: torch.Tensor,
    value: torch.Tensor,
    cumulative: torch.Tensor,
    scale: torch.Tensor,
    state: torch.Tensor,
    output: torch.Tensor,
    *,
    width: int,
    rank: int,
) -> None:
    """Run one fused chunk state update."""

    batch, heads, head_dim, d_state = state.shape
    row_block = 64
    num_rchunks = triton.cdiv(width * rank, row_block)
    token_block = triton.next_power_of_2(width)
    kernel = _chunk_state_kernel[(batch, heads)]
    with torch.cuda.device(k.device):
        kernel(
            k,
            value,
            cumulative,
            scale,
            state,
            output,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            k.stride(4),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            value.stride(3),
            value.stride(4),
            cumulative.stride(0),
            cumulative.stride(1),
            cumulative.stride(2),
            scale.stride(0),
            scale.stride(1),
            scale.stride(2),
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            D_STATE=d_state,
            HEAD_DIM=head_dim,
            RANK=rank,
            ROW_BLOCK=row_block,
            NUM_RCHUNKS=num_rchunks,
            TOKEN_BLOCK=token_block,
            WIDTH=width,
            USE_BF16_DOT=k.dtype == torch.bfloat16,
            USE_FP16_DOT=k.dtype == torch.float16,
            num_warps=8,
            num_stages=3,
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
    mimo_x,
    mimo_z,
    mimo_out,
    output,
    output_phase,
    output_state,
    output_k,
    q_stride_batch,
    q_stride_head,
    q_stride_rank,
    q_stride_state,
    k_stride_batch,
    k_stride_head,
    k_stride_rank,
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
    previous_k_stride_rank,
    previous_k_stride_state,
    previous_value_stride_batch,
    previous_value_stride_head,
    previous_value_stride_dim,
    mimo_stride_head,
    mimo_stride_rank,
    mimo_stride_dim,
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
    output_k_stride_rank,
    output_k_stride_state,
    D_STATE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_ANGLES: tl.constexpr,
    RANK: tl.constexpr,
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
    mimo_x_pointer = mimo_x + head * mimo_stride_head
    mimo_z_pointer = mimo_z + head * mimo_stride_head
    mimo_out_pointer = mimo_out + head * mimo_stride_head

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
    new_phase = _advance_phase(raw_angle, old_phase, dt_value)
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
    if RANK > 1:
        angle_offsets = tl.arange(0, NUM_ANGLES)
        raw_angle_angles = tl.load(
            angle_rate
            + batch * angle_stride_batch
            + angle_offsets * angle_stride_dim
        ).to(tl.float32)
        old_phase_angles = tl.load(
            phase_pointer + angle_offsets * phase_stride_dim
        ).to(tl.float32)
        new_phase_angles = _advance_phase(
            raw_angle_angles, old_phase_angles, dt_value
        )
        cosine_angles = _cos_approx(new_phase_angles)
        sine_angles = _sin_approx(new_phase_angles)

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
    gate_value = tl.load(gate_pointer + dim_offsets * gate_stride_dim).to(tl.float32)

    if RANK > 1:
        beta_acc = tl.zeros([HEAD_DIM, D_STATE], dtype=tl.float32)
        gamma_acc = tl.zeros([HEAD_DIM, D_STATE], dtype=tl.float32)
        output_acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for r in tl.static_range(RANK):
            r_base = r * q_stride_rank
            q_first = tl.load(
                q_pointer + r_base + angle_offsets * q_stride_state
            ).to(tl.float32)
            q_second = tl.load(
                q_pointer
                + r_base
                + (NUM_ANGLES + angle_offsets) * q_stride_state
            ).to(tl.float32)
            q_third = tl.load(
                q_pointer
                + r_base
                + (2 * NUM_ANGLES + angle_offsets) * q_stride_state
            ).to(tl.float32)
            q_fourth = tl.load(
                q_pointer
                + r_base
                + (3 * NUM_ANGLES + angle_offsets) * q_stride_state
            ).to(tl.float32)
            q_first_rotated = q_first * cosine_angles - q_third * sine_angles
            q_third_rotated = q_first * sine_angles + q_third * cosine_angles
            q_rotated = tl.cat(
                tl.cat(q_first_rotated, q_second),
                tl.cat(q_third_rotated, q_fourth),
            )

            k_first = tl.load(
                k_pointer + r * k_stride_rank + angle_offsets * k_stride_state
            ).to(tl.float32)
            k_second = tl.load(
                k_pointer
                + r * k_stride_rank
                + (NUM_ANGLES + angle_offsets) * k_stride_state
            ).to(tl.float32)
            k_third = tl.load(
                k_pointer
                + r * k_stride_rank
                + (2 * NUM_ANGLES + angle_offsets) * k_stride_state
            ).to(tl.float32)
            k_fourth = tl.load(
                k_pointer
                + r * k_stride_rank
                + (3 * NUM_ANGLES + angle_offsets) * k_stride_state
            ).to(tl.float32)
            k_first_rotated = k_first * cosine_angles - k_third * sine_angles
            k_third_rotated = k_first * sine_angles + k_third * cosine_angles
            k_rotated = tl.cat(
                tl.cat(k_first_rotated, k_second),
                tl.cat(k_third_rotated, k_fourth),
            )
            tl.store(
                output_k
                + batch * output_k_stride_batch
                + head * output_k_stride_head
                + r * output_k_stride_rank
                + state_offsets * output_k_stride_state,
                k_rotated,
            )

            old_k = tl.load(
                previous_k_pointer
                + r * previous_k_stride_rank
                + state_offsets * previous_k_stride_state
            )
            mimo_x_r = tl.load(
                mimo_x_pointer
                + r * mimo_stride_rank
                + dim_offsets * mimo_stride_dim
            )
            mimo_z_r = tl.load(
                mimo_z_pointer
                + r * mimo_stride_rank
                + dim_offsets * mimo_stride_dim
            )
            mimo_out_r = tl.load(
                mimo_out_pointer
                + r * mimo_stride_rank
                + dim_offsets * mimo_stride_dim
            )
            current_rank_value = current_value * mimo_x_r
            beta_acc += (beta * old_value * mimo_x_r)[:, None] * old_k[None, :]
            gamma_acc += (gamma * current_rank_value)[:, None] * k_rotated[None, :]

        next_state = alpha * old_state + beta_acc + gamma_acc
        tl.store(
            output_state
            + batch * output_state_stride_batch
            + head * output_state_stride_head
            + dim_offsets[:, None] * output_state_stride_dim
            + state_offsets[None, :] * output_state_stride_state,
            next_state,
        )

        output_acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for r in tl.static_range(RANK):
            r_base = r * q_stride_rank
            q_first = tl.load(
                q_pointer + r_base + angle_offsets * q_stride_state
            ).to(tl.float32)
            q_second = tl.load(
                q_pointer
                + r_base
                + (NUM_ANGLES + angle_offsets) * q_stride_state
            ).to(tl.float32)
            q_third = tl.load(
                q_pointer
                + r_base
                + (2 * NUM_ANGLES + angle_offsets) * q_stride_state
            ).to(tl.float32)
            q_fourth = tl.load(
                q_pointer
                + r_base
                + (3 * NUM_ANGLES + angle_offsets) * q_stride_state
            ).to(tl.float32)
            q_first_rotated = q_first * cosine_angles - q_third * sine_angles
            q_third_rotated = q_first * sine_angles + q_third * cosine_angles
            q_rotated = tl.cat(
                tl.cat(q_first_rotated, q_second),
                tl.cat(q_third_rotated, q_fourth),
            )
            mimo_x_r = tl.load(
                mimo_x_pointer
                + r * mimo_stride_rank
                + dim_offsets * mimo_stride_dim
            )
            mimo_z_r = tl.load(
                mimo_z_pointer
                + r * mimo_stride_rank
                + dim_offsets * mimo_stride_dim
            )
            mimo_out_r = tl.load(
                mimo_out_pointer
                + r * mimo_stride_rank
                + dim_offsets * mimo_stride_dim
            )
            current_rank_value = current_value * mimo_x_r

            q_column = tl.reshape(q_rotated, [D_STATE, 1])
            if USE_BF16_DOT:
                mixed = tl.dot(
                    next_state.to(tl.bfloat16), q_column.to(tl.bfloat16)
                )
            elif USE_FP16_DOT:
                mixed = tl.dot(
                    next_state.to(tl.float16), q_column.to(tl.float16)
                )
            else:
                mixed = tl.sum(next_state * q_rotated[None, :], axis=1)[:, None]
            mixed = tl.reshape(mixed, [HEAD_DIM]).to(tl.float32)
            mixed += tl.load(D + head).to(tl.float32) * current_rank_value
            gate_rank = gate_value * mimo_z_r
            mixed *= gate_rank * tl.sigmoid(gate_rank)
            output_acc += mixed * mimo_out_r

        tl.store(
            output
            + batch * output_stride_batch
            + head * output_stride_head
            + dim_offsets * output_stride_dim,
            output_acc,
        )
    else:
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
        mixed *= gate_value * tl.sigmoid(gate_value)
        tl.store(
            output
            + batch * output_stride_batch
            + head * output_stride_head
            + dim_offsets * output_stride_dim,
            mixed,
        )


def fused_step(
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
    *,
    mimo_x: torch.Tensor | None = None,
    mimo_z: torch.Tensor | None = None,
    mimo_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Run one fused step update for SISO or MIMO; inputs are unrotated."""

    phase, state, previous_k, previous_value = cache
    batch, heads, rank, d_state = q.shape
    head_dim = value.shape[-1]
    num_angles = angle_rate.shape[-1]
    if num_angles <= 0 or num_angles > d_state // 2:
        raise ValueError("angle width must be in [1, d_state / 2]")
    if rank > 1 and 4 * num_angles != d_state:
        raise ValueError(
            "fused MIMO step requires 4 * num_angles == d_state "
            "(rope_fraction 0.5 and d_state divisible by 4)"
        )
    expected_shapes = {
        "k": (batch, heads, rank, d_state),
        "value": (batch, heads, head_dim),
        "gate": (batch, heads, head_dim),
        "adt": (batch, heads),
        "dt": (batch, heads),
        "trap_logits": (batch, heads),
        "angle_rate": (batch, num_angles),
        "D": (heads,),
        "phase": (batch, heads, num_angles),
        "state": (batch, heads, head_dim, d_state),
        "previous_k": (batch, heads, rank, d_state),
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
    if rank > 1 and (mimo_x is None or mimo_z is None or mimo_out is None):
        raise ValueError("fused MIMO step requires all three MIMO projections")
    if rank > 1:
        expected_projection = (heads, rank, head_dim)
        for name, projection in (
            ("mimo_x", mimo_x),
            ("mimo_z", mimo_z),
            ("mimo_out", mimo_out),
        ):
            if projection.shape != expected_projection:
                raise ValueError(
                    f"invalid {name} shape: expected {expected_projection}, "
                    f"got {tuple(projection.shape)}"
                )
    if mimo_x is not None and rank == 1:
        raise ValueError("MIMO projections require rank > 1")

    tensors = (q, k, value, gate, adt, dt, trap_logits, angle_rate, D, *cache)
    if mimo_x is not None:
        tensors = tensors + (mimo_x, mimo_z, mimo_out)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("fused step requires CUDA tensors")
    if any(tensor.device != q.device for tensor in tensors):
        raise ValueError("fused step tensors must be on the same CUDA device")

    q = q.contiguous()
    k = k.contiguous()
    value = value.contiguous()
    gate = gate.contiguous()
    adt = adt.contiguous()
    dt = dt.contiguous()
    trap_logits = trap_logits.contiguous()
    angle_rate = angle_rate.contiguous()
    previous_k = previous_k.contiguous()
    previous_value = previous_value.contiguous()
    if mimo_x is not None:
        mimo_x = mimo_x.contiguous()
        mimo_z = mimo_z.contiguous()
        mimo_out = mimo_out.contiguous()
    else:
        # RANK == 1 kernels never touch the MIMO pointers, but the launch
        # still reads their strides.
        mimo_x = mimo_z = mimo_out = value.new_zeros(1, 1, 1)

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
            previous_k,
            previous_value,
            mimo_x,
            mimo_z,
            mimo_out,
            output,
            output_phase,
            output_state,
            output_k,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
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
            previous_k.stride(0),
            previous_k.stride(1),
            previous_k.stride(2),
            previous_k.stride(3),
            previous_value.stride(0),
            previous_value.stride(1),
            previous_value.stride(2),
            mimo_x.stride(0),
            mimo_x.stride(1),
            mimo_x.stride(2),
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
            output_k.stride(3),
            D_STATE=d_state,
            HEAD_DIM=head_dim,
            NUM_ANGLES=num_angles,
            RANK=rank,
            USE_BF16_DOT=q.dtype == torch.bfloat16,
            USE_FP16_DOT=q.dtype == torch.float16,
            num_warps=num_warps,
            num_stages=1,
        )
    return output, (output_phase, output_state, output_k, value)
