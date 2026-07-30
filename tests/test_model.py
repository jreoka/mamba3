from __future__ import annotations

import pytest
import torch

from mamba3 import Mamba3, Mamba3Config, Mamba3LM


def test_simple_api_and_shape() -> None:
    model = Mamba3(d_model=24, d_state=8, depth=2, causal=True, d_conv=3)
    x = torch.randn(2, 17, 24)
    y = model(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_config_api() -> None:
    config = Mamba3Config(d_model=16, d_state=4, depth=1, causal=False)
    model = Mamba3(config)
    assert model(torch.randn(1, 9, 16)).shape == (1, 9, 16)


def test_causal_prefix_is_unchanged() -> None:
    torch.manual_seed(0)
    model = Mamba3(16, d_state=4, depth=2, causal=True, d_conv=3).eval()
    first = torch.randn(1, 12, 16)
    second = first.clone()
    second[:, 7:] = torch.randn_like(second[:, 7:])
    with torch.no_grad():
        y_first = model(first)
        y_second = model(second)
    torch.testing.assert_close(y_first[:, :7], y_second[:, :7], rtol=0, atol=0)


def test_noncausal_uses_future_context() -> None:
    torch.manual_seed(1)
    model = Mamba3(16, d_state=4, depth=1, causal=False, d_conv=3).eval()
    first = torch.randn(1, 12, 16)
    second = first.clone()
    second[:, 7:] = torch.randn_like(second[:, 7:])
    with torch.no_grad():
        difference = (model(first)[:, :7] - model(second)[:, :7]).abs().max()
    assert difference > 1e-7


def test_cpu_backward() -> None:
    model = Mamba3(12, d_state=4, depth=1, causal=True, d_conv=2)
    x = torch.randn(2, 8, 12, requires_grad=True)
    model(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_language_model_wrapper() -> None:
    model = Mamba3LM(vocab_size=101, d_model=16, d_state=4, depth=1)
    logits = model(torch.randint(0, 101, (2, 11)))
    assert logits.shape == (2, 11, 101)
    assert model.lm_head.weight is model.embedding.weight


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_model_smoke() -> None:
    model = Mamba3(16, d_state=8, depth=1, causal=True, d_conv=3).cuda()
    x = torch.randn(2, 23, 16, device="cuda", requires_grad=True)
    loss = model(x).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert x.grad is not None and torch.isfinite(x.grad).all()
