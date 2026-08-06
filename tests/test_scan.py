from __future__ import annotations

import importlib.util
import math

import pytest
import torch
from torch.nn import functional as F

from mamba3.ops import _batched_matmul, heavy_tail_activation, mamba3_scan, rotate_qk


def _rank_values(
    value: torch.Tensor, projection: torch.Tensor | None
) -> torch.Tensor:
    if projection is None:
        return value.unsqueeze(3)
    return value.unsqueeze(3) * projection.to(value.dtype)[None, None]


def _recurrent_reference(
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
    initial_state: torch.Tensor | None = None,
    initial_k: torch.Tensor | None = None,
    initial_value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, length, heads, rank, d_state = q.shape
    head_dim = value.shape[-1]
    rank_value = _rank_values(value, mimo_x)
    rank_gate = _rank_values(gate, mimo_z)
    state = (
        torch.zeros(
            batch,
            heads,
            head_dim,
            d_state,
            device=q.device,
            dtype=torch.float32,
        )
        if initial_state is None
        else initial_state.float()
    )
    previous_k = (
        torch.zeros(batch, heads, rank, d_state, device=q.device, dtype=q.dtype)
        if initial_k is None
        else initial_k
    )
    previous_value = (
        torch.zeros(
            batch, heads, head_dim, device=value.device, dtype=value.dtype
        )
        if initial_value is None
        else initial_value
    )
    previous_rank_value = _rank_values(previous_value.unsqueeze(1), mimo_x).squeeze(1)

    outputs = []
    for index in range(length):
        trap = torch.sigmoid(trap_logits[:, index].float())
        alpha = torch.exp(adt[:, index].float())
        beta = (1.0 - trap) * dt[:, index] * alpha
        gamma = trap * dt[:, index]
        current_value = rank_value[:, index]
        previous_outer = torch.einsum(
            "bhrp,bhrn->bhpn", previous_rank_value.float(), previous_k.float()
        )
        current_outer = torch.einsum(
            "bhrp,bhrn->bhpn", current_value.float(), k[:, index].float()
        )
        state = (
            alpha[..., None, None] * state
            + beta[..., None, None] * previous_outer
            + gamma[..., None, None] * current_outer
        )
        output = torch.einsum(
            "bhpn,bhrn->bhrp", state, q[:, index].float()
        )
        output = output + D[None, :, None, None].float() * current_value.float()
        output = output * F.silu(rank_gate[:, index].float())
        if mimo_out is not None:
            output = (output * mimo_out[None].float()).sum(dim=2)
        else:
            output = output.squeeze(2)
        outputs.append(output)
        previous_k = k[:, index]
        previous_rank_value = current_value
    return torch.stack(outputs, dim=1).to(value.dtype), state


def _make_inputs(rank: int, *, requires_grad: bool = False) -> tuple:
    torch.manual_seed(42 + rank)
    batch, length, heads, head_dim, d_state = 2, 7, 3, 4, 8
    q = torch.randn(batch, length, heads, rank, d_state)
    k = torch.randn_like(q)
    value = torch.randn(batch, length, heads, head_dim)
    gate = torch.randn_like(value)
    adt = -torch.rand(batch, length, heads) * 0.2
    dt = torch.rand(batch, length, heads) * 0.1 + 0.001
    trap = torch.randn(batch, length, heads)
    D = torch.randn(heads)
    if rank > 1:
        mimo_x = torch.randn(heads, rank, head_dim) / rank
        mimo_z = torch.randn(heads, rank, head_dim)
        mimo_out = torch.randn(heads, rank, head_dim) / rank
    else:
        mimo_x = mimo_z = mimo_out = None
    values = (q, k, value, gate, adt, dt, trap, D, mimo_x, mimo_z, mimo_out)
    if requires_grad:
        values = tuple(
            item.detach().clone().requires_grad_(True) if item is not None else None
            for item in values
        )
    return values


def _run(values: tuple, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    q, k, value, gate, adt, dt, trap, D, mimo_x, mimo_z, mimo_out = values
    return mamba3_scan(
        q,
        k,
        value,
        gate,
        adt,
        dt,
        trap,
        D,
        mimo_x=mimo_x,
        mimo_z=mimo_z,
        mimo_out=mimo_out,
        chunk_size=chunk_size,
    )


def test_heavy_tail_activation_matches_definition_and_gradient() -> None:
    x = torch.tensor([-3.0, -0.5, 0.0, 0.5, 3.0], requires_grad=True)
    expected = torch.where(x >= 0, 1.0 + x, 1.0 / (1.0 - x))
    actual = heavy_tail_activation(x)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected_gradient = torch.where(x >= 0, torch.ones_like(x), expected.square())
    torch.testing.assert_close(x.grad, expected_gradient)


def test_siso_rotation_uses_adjacent_complex_pairs() -> None:
    q = torch.tensor([[[[[1.0, 0.0, 2.0, 0.0, 5.0, 6.0, 7.0, 8.0]]]]])
    phase = torch.tensor([[[[math.pi / 2, 0.0]]]])
    actual, _ = rotate_qk(q, q, phase, mimo=False)
    expected = torch.tensor([[[[[0.0, 1.0, 2.0, 0.0, 5.0, 6.0, 7.0, 8.0]]]]])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)


def test_mimo_rotation_matches_official_split_half_layout() -> None:
    q = torch.tensor([[[[[1.0, 2.0, 3.0, 4.0, 0.0, 5.0, 6.0, 7.0]]]]])
    phase = torch.tensor([[[[math.pi / 2, 0.0]]]])
    actual, _ = rotate_qk(q, q, phase, mimo=True)
    expected = torch.tensor([[[[[0.0, 2.0, 3.0, 4.0, 1.0, 5.0, 6.0, 7.0]]]]])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("rank,chunk_size", [(1, 3), (4, 2), (4, 5)])
def test_chunked_scan_matches_independent_recurrence(
    rank: int, chunk_size: int
) -> None:
    values = _make_inputs(rank)
    actual, actual_state = _run(values, chunk_size)
    q, k, value, gate, adt, dt, trap, D, mimo_x, mimo_z, mimo_out = values
    expected, expected_state = _recurrent_reference(
        q,
        k,
        value,
        gate,
        adt,
        dt,
        trap,
        D,
        mimo_x=mimo_x,
        mimo_z=mimo_z,
        mimo_out=mimo_out,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual_state, expected_state, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("rank", [1, 4])
def test_chunked_scan_gradients_match_recurrence(rank: int) -> None:
    actual_values = _make_inputs(rank, requires_grad=True)
    reference_values = tuple(
        item.detach().clone().requires_grad_(True) if item is not None else None
        for item in actual_values
    )
    actual, actual_state = _run(actual_values, chunk_size=3)
    q, k, value, gate, adt, dt, trap, D, mimo_x, mimo_z, mimo_out = reference_values
    expected, expected_state = _recurrent_reference(
        q,
        k,
        value,
        gate,
        adt,
        dt,
        trap,
        D,
        mimo_x=mimo_x,
        mimo_z=mimo_z,
        mimo_out=mimo_out,
    )
    output_weight = torch.randn_like(actual)
    state_weight = torch.randn_like(actual_state)
    ((actual * output_weight).sum() + (actual_state * state_weight).sum()).backward()
    ((expected * output_weight).sum() + (expected_state * state_weight).sum()).backward()
    for actual_value, reference_value in zip(actual_values, reference_values):
        if actual_value is not None:
            torch.testing.assert_close(
                actual_value.grad, reference_value.grad, rtol=2e-4, atol=3e-5
            )


@pytest.mark.parametrize("rank", [1, 4])
def test_scan_state_can_continue_a_sequence(rank: int) -> None:
    values = _make_inputs(rank)
    q, k, value, gate, adt, dt, trap, D, mimo_x, mimo_z, mimo_out = values
    split = 4
    full, full_state = _run(values, chunk_size=3)

    first_values = tuple(
        item[:, :split] if index < 7 else item
        for index, item in enumerate(values)
    )
    first, first_state = _run(first_values, chunk_size=3)
    second, second_state = mamba3_scan(
        q[:, split:],
        k[:, split:],
        value[:, split:],
        gate[:, split:],
        adt[:, split:],
        dt[:, split:],
        trap[:, split:],
        D,
        mimo_x=mimo_x,
        mimo_z=mimo_z,
        mimo_out=mimo_out,
        chunk_size=3,
        initial_state=first_state,
        initial_k=k[:, split - 1].permute(0, 1, 2, 3),
        initial_value=value[:, split - 1],
    )
    torch.testing.assert_close(torch.cat((first, second), dim=1), full, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(second_state, full_state, rtol=2e-5, atol=2e-6)


def test_empty_continued_scan_preserves_state() -> None:
    batch, heads, rank, head_dim, d_state = 2, 3, 1, 4, 8
    q = torch.empty(batch, 0, heads, rank, d_state)
    value = torch.empty(batch, 0, heads, head_dim, requires_grad=True)
    state = torch.randn(batch, heads, head_dim, d_state)
    output, final_state = mamba3_scan(
        q,
        q,
        value,
        value,
        torch.empty(batch, 0, heads),
        torch.empty(batch, 0, heads),
        torch.empty(batch, 0, heads),
        torch.ones(heads),
        initial_state=state,
        initial_k=torch.randn(batch, heads, rank, d_state),
        initial_value=torch.randn(batch, heads, head_dim),
    )
    assert output.shape == value.shape
    torch.testing.assert_close(final_state, state)
    output.sum().backward()
    assert value.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("rank", [1, 4])
def test_cuda_bfloat16_scan_uses_fp32_accumulation(rank: int) -> None:
    values = list(_make_inputs(rank))
    low_precision = {0, 1, 2, 3, 6}
    values = [
        item.cuda().to(torch.bfloat16 if index in low_precision else torch.float32)
        if item is not None
        else None
        for index, item in enumerate(values)
    ]
    actual, actual_state = _run(tuple(values), chunk_size=3)
    q, k, value, gate, adt, dt, trap, D, mimo_x, mimo_z, mimo_out = values
    expected, expected_state = _recurrent_reference(
        q,
        k,
        value,
        gate,
        adt,
        dt,
        trap,
        D,
        mimo_x=mimo_x,
        mimo_z=mimo_z,
        mimo_out=mimo_out,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_state, expected_state, rtol=3e-3, atol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_fp32_bmm_accumulation_survives_autocast_and_backward() -> None:
    torch.manual_seed(50)
    left = torch.randn(3, 5, 7, device="cuda", dtype=torch.bfloat16).requires_grad_()
    right = torch.randn(3, 7, 4, device="cuda", dtype=torch.bfloat16).requires_grad_()
    reference_left = left.detach().float().requires_grad_()
    reference_right = right.detach().float().requires_grad_()
    weight = torch.randn(3, 5, 4, device="cuda")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = _batched_matmul(left, right, accumulate_float=True)
    expected = torch.bmm(reference_left, reference_right)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(
        left.grad.float(), reference_left.grad, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        right.grad.float(), reference_right.grad, rtol=2e-2, atol=2e-2
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("rank", [1, 4])
def test_cuda_bfloat16_scan_gradients_match_fp32_reference(rank: int) -> None:
    actual_values = list(_make_inputs(rank))
    low_precision = {0, 1, 2, 3, 6}
    actual_values = tuple(
        item.cuda()
        .to(torch.bfloat16 if index in low_precision else torch.float32)
        .requires_grad_(True)
        if item is not None
        else None
        for index, item in enumerate(actual_values)
    )
    reference_values = tuple(
        item.detach().clone().requires_grad_(True) if item is not None else None
        for item in actual_values
    )
    actual, actual_state = _run(actual_values, chunk_size=3)
    q, k, value, gate, adt, dt, trap, D, mimo_x, mimo_z, mimo_out = reference_values
    expected, expected_state = _recurrent_reference(
        q,
        k,
        value,
        gate,
        adt,
        dt,
        trap,
        D,
        mimo_x=mimo_x,
        mimo_z=mimo_z,
        mimo_out=mimo_out,
    )
    output_weight = torch.randn_like(actual)
    state_weight = torch.randn_like(actual_state)
    ((actual * output_weight).sum() + (actual_state * state_weight).sum()).backward()
    ((expected * output_weight).sum() + (expected_state * state_weight).sum()).backward()
    for actual_value, reference_value in zip(actual_values, reference_values):
        if actual_value is not None:
            torch.testing.assert_close(
                actual_value.grad.float(),
                reference_value.grad.float(),
                rtol=5e-2,
                atol=5e-2,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="Triton is unavailable")
@pytest.mark.parametrize("rank", [1, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_chunked_scan_matches_pytorch_path(
    monkeypatch: pytest.MonkeyPatch, rank: int, dtype: torch.dtype
) -> None:
    import mamba3.ops as ops
    from mamba3 import Mamba3

    torch.manual_seed(11 + rank)
    model = Mamba3(64, d_state=16, depth=1, mimo_rank=rank).cuda().to(dtype).eval()
    x = torch.randn(2, 37, 64, device="cuda", dtype=dtype)

    monkeypatch.setenv("MAMBA3_DISABLE_TRITON", "1")
    with torch.inference_mode():
        reference = model(x)
        prefix, reference_cache = model.prefill(x[:, :9])
        continued = [prefix]
        for index in range(9, 37):
            output, reference_cache = model.step(
                x[:, index : index + 1], reference_cache
            )
            continued.append(output)
        continued = torch.cat(continued, dim=1)
    monkeypatch.delenv("MAMBA3_DISABLE_TRITON")

    with torch.inference_mode():
        actual = model(x)
        prefix, actual_cache = model.prefill(x[:, :9])
        actual_continued = [prefix]
        for index in range(9, 37):
            output, actual_cache = model.step(
                x[:, index : index + 1], actual_cache
            )
            actual_continued.append(output)
        actual_continued = torch.cat(actual_continued, dim=1)

    assert ops._TRITON_LAST_SCAN_DISPATCH
    torch.testing.assert_close(actual, reference, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(
        actual_continued, continued, rtol=3e-2, atol=3e-2
    )
    for reference_layer, actual_layer in zip(reference_cache, actual_cache):
        for reference_tensor, actual_tensor in zip(reference_layer, actual_layer):
            torch.testing.assert_close(
                actual_tensor.float(),
                reference_tensor.float(),
                rtol=3e-2,
                atol=3e-2,
            )
