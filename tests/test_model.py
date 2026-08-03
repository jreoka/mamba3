from __future__ import annotations

import inspect

import pytest
import torch

import mamba3
from mamba3 import Mamba3


def test_simple_api_and_shape() -> None:
    model = Mamba3(d_model=24, d_state=8, depth=2, causal=True)
    x = torch.randn(2, 17, 24)
    y = model(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert model(torch.empty(0, 5, 24)).shape == (0, 5, 24)
    assert model(torch.empty(2, 0, 24)).shape == (2, 0, 24)
    with pytest.raises(ValueError, match="expected"):
        model(torch.empty(0, 5, 7))

    model.zero_grad(set_to_none=True)
    model(torch.empty(2, 0, 24)).sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_only_mamba3_is_public() -> None:
    assert mamba3.__all__ == ["Mamba3"]
    assert not hasattr(Mamba3, "compile_kernels")
    assert list(inspect.signature(Mamba3).parameters) == [
        "d_model",
        "d_state",
        "depth",
        "causal",
    ]


def test_accepts_large_d_state() -> None:
    model = Mamba3(d_model=16, d_state=65, depth=1)
    assert not hasattr(model, "config")
    assert model(torch.randn(1, 3, 16)).shape == (1, 3, 16)


def test_causal_prefix_is_unchanged() -> None:
    torch.manual_seed(0)
    model = Mamba3(16, d_state=4, depth=2, causal=True).eval()
    first = torch.randn(1, 12, 16)
    second = first.clone()
    second[:, 7:] = torch.randn_like(second[:, 7:])
    with torch.no_grad():
        y_first = model(first)
        y_second = model(second)
    torch.testing.assert_close(y_first[:, :7], y_second[:, :7], rtol=0, atol=0)


def test_noncausal_uses_future_context() -> None:
    torch.manual_seed(1)
    model = Mamba3(16, d_state=4, depth=1, causal=False).eval()
    first = torch.randn(1, 12, 16)
    second = first.clone()
    second[:, 7:] = torch.randn_like(second[:, 7:])
    with torch.no_grad():
        difference = (model(first)[:, :7] - model(second)[:, :7]).abs().max()
    assert difference > 1e-7


def test_cpu_backward() -> None:
    model = Mamba3(12, d_state=4, depth=1, causal=True)
    x = torch.randn(2, 8, 12, requires_grad=True)
    model(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_cached_steps_match_causal_forward() -> None:
    torch.manual_seed(2)
    model = Mamba3(16, d_state=4, depth=2, causal=True).eval()
    x = torch.randn(2, 9, 16)
    cache = None
    outputs = []
    with torch.no_grad():
        expected = model(x)
        for index in range(x.shape[1]):
            output, cache = model.step(x[:, index : index + 1], cache)
            outputs.append(output)
    torch.testing.assert_close(torch.cat(outputs, dim=1), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_model_smoke() -> None:
    model = Mamba3(16, d_state=8, depth=1, causal=True).cuda()
    x = torch.randn(2, 23, 16, device="cuda", requires_grad=True)
    loss = model(x).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert x.grad is not None and torch.isfinite(x.grad).all()

    model.eval()
    cache = None
    outputs = []
    reference = model(x.detach())
    with torch.no_grad():
        expected = model(x.detach())
        for index in range(x.shape[1]):
            output, cache = model.step(x[:, index : index + 1], cache)
            outputs.append(output)
    torch.testing.assert_close(expected, reference, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(torch.cat(outputs, dim=1), expected, rtol=1e-4, atol=1e-5)

    with torch.no_grad():
        _, cache = model.step(x[:, :1], None)
        cached_conv = cache[0][0].clone()
        model.step(x[:, 1:2], cache)
    torch.testing.assert_close(cache[0][0], cached_conv)

    mixed_model = Mamba3(16, d_state=8, depth=1, causal=True).cuda().eval()
    mixed_x = x.detach()[:, :5]
    mixed_outputs = []
    cache = None
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = mixed_model(mixed_x)
        for index in range(mixed_x.shape[1]):
            output, cache = mixed_model.step(mixed_x[:, index : index + 1], cache)
            mixed_outputs.append(output)
    torch.testing.assert_close(
        torch.cat(mixed_outputs, dim=1), expected, rtol=2e-2, atol=2e-2
    )

    odd_model = Mamba3(17, d_state=5, depth=1).cuda()
    odd_x = torch.randn(2, 7, 17, device="cuda", requires_grad=True)
    odd_model(odd_x).sum().backward()
    assert odd_x.grad is not None and torch.isfinite(odd_x.grad).all()

    graph_model = Mamba3(16, d_state=8, depth=1).cuda().eval()
    graph_x = torch.randn(2, 5, 16, device="cuda")
    decoder = graph_model.cuda_graph(batch_size=2)
    with torch.inference_mode():
        expected = graph_model(graph_x)
        actual = torch.cat(
            [decoder(graph_x[:, index : index + 1]).clone() for index in range(5)],
            dim=1,
        )
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    with torch.no_grad():
        graph_model.layers[0].A_log.add_(0.1)
    with pytest.raises(RuntimeError, match="recreate"):
        decoder(graph_x[:, :1])

    replacement_model = Mamba3(16, d_state=8, depth=1).cuda().eval()
    replacement_decoder = replacement_model.cuda_graph(batch_size=2)
    replacement_model.layers[0].in_proj.weight = torch.nn.Parameter(
        replacement_model.layers[0].in_proj.weight.detach().clone()
    )
    with pytest.raises(RuntimeError, match="recreate"):
        replacement_decoder(graph_x[:, :1])
