from __future__ import annotations

import pytest
import torch

from mamba3.ops import _use_row_cuda_kernel, load_cuda_extension, selective_scan


def make_inputs(device: str, requires_grad: bool = False):
    torch.manual_seed(42)
    batch, length, channels, state = 2, 13, 7, 5
    values = (
        torch.randn(batch, length, channels, device=device) * 0.2,
        torch.rand(batch, length, channels, device=device) * 0.1,
        -torch.rand(channels, state, device=device) - 0.1,
        torch.randn(batch, length, state, device=device) * 0.2,
        torch.randn(batch, length, state, device=device) * 0.2,
        torch.randn(channels, device=device) * 0.2,
        torch.randn(batch, length, channels, device=device) * 0.2,
        torch.randn(batch, channels, state, device=device) * 0.1,
    )
    return tuple(value.requires_grad_(requires_grad) for value in values)


def run(values, use_cuda_kernel: bool, reverse: bool = False):
    x, dt, A, B, C, D, z, initial = values
    return selective_scan(
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        initial_state=initial,
        return_state=True,
        use_cuda_kernel=use_cuda_kernel,
        reverse=reverse,
    )


def test_row_kernel_dispatch_targets_high_batch_short_sequences() -> None:
    assert _use_row_cuda_kernel(batch=124, length=690)
    assert _use_row_cuda_kernel(batch=690, length=124)
    assert not _use_row_cuda_kernel(batch=8, length=2048)
    assert not _use_row_cuda_kernel(batch=1, length=690)
    assert not _use_row_cuda_kernel(batch=124, length=690, d_state=65)


def test_d_state_must_be_positive() -> None:
    x = torch.empty(1, 2, 3)
    with pytest.raises(ValueError, match="d_state must be positive"):
        selective_scan(
            x,
            x,
            torch.empty(3, 0),
            torch.empty(1, 2, 0),
            torch.empty(1, 2, 0),
            torch.empty(3),
            x,
            use_cuda_kernel=False,
        )


def test_reverse_reference_matches_explicit_flip() -> None:
    values = make_inputs("cpu", requires_grad=False)
    y_reverse, state_reverse = run(values, use_cuda_kernel=False, reverse=True)
    x, dt, A, B, C, D, z, initial = values
    explicit_values = (
        x.flip(1),
        dt.flip(1),
        A,
        B.flip(1),
        C.flip(1),
        D,
        z.flip(1),
        initial,
    )
    y_explicit, state_explicit = run(explicit_values, use_cuda_kernel=False)
    torch.testing.assert_close(y_reverse, y_explicit.flip(1))
    torch.testing.assert_close(state_reverse, state_explicit)


def test_reference_scan_state_and_gradients() -> None:
    values = make_inputs("cpu", requires_grad=True)
    y, state = run(values, use_cuda_kernel=False)
    (y.square().mean() + state.square().mean()).backward()
    assert y.shape == (2, 13, 7)
    assert state.shape == (2, 7, 5)
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_matches_reference_forward_and_backward() -> None:
    if load_cuda_extension() is None:
        pytest.skip("mamba3 CUDA extension did not compile")

    cuda_values = make_inputs("cuda", requires_grad=True)
    reference_values = tuple(value.detach().clone().requires_grad_(True) for value in cuda_values)
    y_cuda, state_cuda = run(cuda_values, use_cuda_kernel=True)
    y_ref, state_ref = run(reference_values, use_cuda_kernel=False)
    torch.testing.assert_close(y_cuda, y_ref, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(state_cuda, state_ref, rtol=2e-4, atol=2e-5)

    weights = torch.randn_like(y_cuda)
    state_weights = torch.randn_like(state_cuda)
    (y_cuda * weights).sum().add((state_cuda * state_weights).sum()).backward()
    (y_ref * weights).sum().add((state_ref * state_weights).sum()).backward()
    for cuda_value, reference_value in zip(cuda_values, reference_values):
        torch.testing.assert_close(cuda_value.grad, reference_value.grad, rtol=1e-3, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_row_reverse_matches_reference() -> None:
    if load_cuda_extension() is None:
        pytest.skip("mamba3 CUDA extension did not compile")

    base_values = make_inputs("cuda")
    cuda_values = tuple(
        value if index in (2, 5) else value.repeat(4, *([1] * (value.ndim - 1)))
        for index, value in enumerate(base_values)
    )
    cuda_values = tuple(value.detach().requires_grad_(True) for value in cuda_values)
    reference_values = tuple(value.detach().clone().requires_grad_(True) for value in cuda_values)
    y_cuda, state_cuda = run(cuda_values, use_cuda_kernel=True, reverse=True)
    y_ref, state_ref = run(reference_values, use_cuda_kernel=False, reverse=True)
    torch.testing.assert_close(y_cuda, y_ref, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(state_cuda, state_ref, rtol=2e-4, atol=2e-5)

    weights = torch.randn_like(y_cuda)
    state_weights = torch.randn_like(state_cuda)
    (y_cuda * weights).sum().add((state_cuda * state_weights).sum()).backward()
    (y_ref * weights).sum().add((state_ref * state_weights).sum()).backward()
    for cuda_value, reference_value in zip(cuda_values, reference_values):
        torch.testing.assert_close(cuda_value.grad, reference_value.grad, rtol=1e-3, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    ((torch.float16, 5e-3, 5e-4), (torch.bfloat16, 2e-2, 2e-3)),
)
def test_cuda_mixed_precision_matches_reference(
    dtype: torch.dtype,
    rtol: float,
    atol: float,
) -> None:
    if load_cuda_extension() is None:
        pytest.skip("mamba3 CUDA extension did not compile")

    cuda_values = make_inputs("cuda", requires_grad=True)
    cuda_values = tuple(
        value.detach().to(dtype).requires_grad_(True)
        if index not in (2, 5, 7)
        else value
        for index, value in enumerate(cuda_values)
    )
    reference_values = tuple(value.detach().clone().requires_grad_(True) for value in cuda_values)
    y_cuda, state_cuda = run(cuda_values, use_cuda_kernel=True)
    y_ref, state_ref = run(reference_values, use_cuda_kernel=False)
    assert y_cuda.dtype == dtype
    assert state_cuda.dtype == torch.float32
    torch.testing.assert_close(y_cuda, y_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(state_cuda, state_ref, rtol=1e-3, atol=2e-4)

    weights = torch.randn_like(y_cuda)
    state_weights = torch.randn_like(state_cuda)
    (y_cuda * weights).sum().add((state_cuda * state_weights).sum()).backward()
    (y_ref * weights).sum().add((state_ref * state_weights).sum()).backward()
    for cuda_value, reference_value in zip(cuda_values, reference_values):
        torch.testing.assert_close(cuda_value.grad, reference_value.grad, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_long_sequence_gradients_are_stable() -> None:
    if load_cuda_extension() is None:
        pytest.skip("mamba3 CUDA extension did not compile")

    torch.manual_seed(123)
    batch, length, channels, state = 1, 257, 5, 16
    cuda_values = (
        torch.randn(batch, length, channels, device="cuda") * 0.2,
        torch.rand(batch, length, channels, device="cuda") * 0.099 + 0.001,
        -torch.arange(1, state + 1, device="cuda").float().repeat(channels, 1),
        torch.randn(batch, length, state, device="cuda") * 0.2,
        torch.randn(batch, length, state, device="cuda") * 0.2,
        torch.randn(channels, device="cuda") * 0.2,
        torch.randn(batch, length, channels, device="cuda") * 0.2,
        torch.zeros(batch, channels, state, device="cuda"),
    )
    cuda_values = tuple(value.requires_grad_(True) for value in cuda_values)
    reference_values = tuple(value.detach().clone().requires_grad_(True) for value in cuda_values)
    y_cuda, state_cuda = run(cuda_values, use_cuda_kernel=True)
    y_ref, state_ref = run(reference_values, use_cuda_kernel=False)
    weights = torch.randn_like(y_cuda)
    state_weights = torch.randn_like(state_cuda)
    (y_cuda * weights).sum().add((state_cuda * state_weights).sum()).backward()
    (y_ref * weights).sum().add((state_ref * state_weights).sum()).backward()

    for cuda_value, reference_value in zip(cuda_values, reference_values):
        assert torch.isfinite(cuda_value.grad).all()
        torch.testing.assert_close(cuda_value.grad, reference_value.grad, rtol=2e-3, atol=3e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_supports_d_state_above_specialized_range() -> None:
    if load_cuda_extension() is None:
        pytest.skip("mamba3 CUDA extension did not compile")

    torch.manual_seed(321)
    batch, length, channels, state = 8, 9, 3, 65
    cuda_values = (
        torch.randn(batch, length, channels, device="cuda") * 0.2,
        torch.rand(batch, length, channels, device="cuda") * 0.1,
        -torch.rand(channels, state, device="cuda") - 0.1,
        torch.randn(batch, length, state, device="cuda") * 0.2,
        torch.randn(batch, length, state, device="cuda") * 0.2,
        torch.randn(channels, device="cuda") * 0.2,
        torch.randn(batch, length, channels, device="cuda") * 0.2,
        torch.randn(batch, channels, state, device="cuda") * 0.1,
    )
    cuda_values = tuple(value.requires_grad_(True) for value in cuda_values)
    reference_values = tuple(value.detach().clone().requires_grad_(True) for value in cuda_values)
    y_cuda, state_cuda = run(cuda_values, use_cuda_kernel=True)
    y_ref, state_ref = run(reference_values, use_cuda_kernel=False)
    torch.testing.assert_close(y_cuda, y_ref, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(state_cuda, state_ref, rtol=3e-4, atol=3e-5)

    weights = torch.randn_like(y_cuda)
    state_weights = torch.randn_like(state_cuda)
    (y_cuda * weights).sum().add((state_cuda * state_weights).sum()).backward()
    (y_ref * weights).sum().add((state_ref * state_weights).sum()).backward()
    for cuda_value, reference_value in zip(cuda_values, reference_values):
        torch.testing.assert_close(cuda_value.grad, reference_value.grad, rtol=2e-3, atol=3e-4)
