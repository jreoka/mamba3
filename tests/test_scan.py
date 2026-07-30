from __future__ import annotations

import pytest
import torch

from mamba3.ops import load_cuda_extension, selective_scan


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


def run(values, use_cuda_kernel: bool):
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
    )


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
