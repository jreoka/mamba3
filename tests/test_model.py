from __future__ import annotations

import importlib.util
import inspect
import math

import pytest
import torch
from torch.nn import functional as F

import mamba3
from mamba3 import Mamba3


def test_simple_api_shape_and_empty_inputs() -> None:
    model = Mamba3(d_model=24, d_state=8, depth=2)
    x = torch.randn(2, 17, 24)
    y = model(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert model(torch.empty(0, 5, 24)).shape == (0, 5, 24)
    assert model(torch.empty(2, 0, 24)).shape == (2, 0, 24)
    with pytest.raises(ValueError, match="expected"):
        model(torch.empty(2, 5, 7))

    model.zero_grad(set_to_none=True)
    model(torch.empty(2, 0, 24)).sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_only_mamba3_is_public_and_constructor_is_small() -> None:
    assert mamba3.__all__ == ["Mamba3"]
    assert list(inspect.signature(Mamba3).parameters) == [
        "d_model",
        "d_state",
        "depth",
        "mimo_rank",
    ]


def test_constructor_validation() -> None:
    with pytest.raises(ValueError, match="d_model"):
        Mamba3(0)
    with pytest.raises(ValueError, match="at least 4"):
        Mamba3(16, d_state=2)
    with pytest.raises(ValueError, match="divisible by 4"):
        Mamba3(16, d_state=10, mimo_rank=4)
    with pytest.raises(ValueError, match="depth"):
        Mamba3(16, depth=0)
    with pytest.raises(ValueError, match="mimo_rank"):
        Mamba3(16, mimo_rank=0)


@pytest.mark.parametrize("rank", [1, 4])
def test_has_canonical_mamba3_parameters(rank: int) -> None:
    model = Mamba3(64, depth=1, mimo_rank=rank)
    mixer = model.layers[0].mixer
    assert mixer.d_state == 128
    assert mixer.d_inner == 128
    assert mixer.headdim == 64
    assert mixer.nheads == 2
    assert mixer.num_angles == 32
    projection_size = 2 * 128 + 2 * 128 * rank + 3 * 2 + 32
    assert mixer.in_proj.weight.shape == (projection_size, 64)
    assert mixer.B_bias.shape == (2, rank, 128)
    assert mixer.C_bias.shape == (2, rank, 128)
    assert not hasattr(mixer, "conv1d")
    assert not hasattr(mixer, "A_log")
    if rank == 1:
        assert mixer.mimo_x is None
    else:
        assert mixer.mimo_x.shape == (2, rank, 64)
    q, k, *_ = mixer._project(torch.randn(1, 3, 64))
    assert q.shape == k.shape == (1, 3, 1, rank, 128)


def test_canonical_initialization() -> None:
    model = Mamba3(32, d_state=8, depth=1, mimo_rank=4)
    mixer = model.layers[0].mixer
    torch.testing.assert_close(mixer.B_bias, torch.ones_like(mixer.B_bias))
    torch.testing.assert_close(mixer.C_bias, torch.ones_like(mixer.C_bias))
    torch.testing.assert_close(mixer.D, torch.ones_like(mixer.D))
    torch.testing.assert_close(
        mixer.mimo_x, torch.full_like(mixer.mimo_x, 0.25)
    )
    torch.testing.assert_close(mixer.mimo_z, torch.ones_like(mixer.mimo_z))
    torch.testing.assert_close(
        mixer.mimo_out, torch.full_like(mixer.mimo_out, 0.25)
    )
    dt = F.softplus(mixer.dt_bias)
    assert (dt >= 0.001).all() and (dt <= 0.1).all()


@pytest.mark.parametrize("rank", [1, 4])
def test_cpu_backward_reaches_every_parameter(rank: int) -> None:
    model = Mamba3(16, d_state=8, depth=1, mimo_rank=rank)
    x = torch.randn(2, 9, 16, requires_grad=True)
    model(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_causal_prefix_is_unchanged() -> None:
    torch.manual_seed(0)
    model = Mamba3(16, d_state=8, depth=2).eval()
    first = torch.randn(1, 12, 16)
    second = first.clone()
    second[:, 7:] = torch.randn_like(second[:, 7:])
    with torch.no_grad():
        y_first = model(first)
        y_second = model(second)
    torch.testing.assert_close(y_first[:, :7], y_second[:, :7], rtol=0, atol=1e-7)


@pytest.mark.parametrize("rank", [1, 4])
def test_cached_steps_match_full_forward(rank: int) -> None:
    torch.manual_seed(2)
    model = Mamba3(16, d_state=8, depth=2, mimo_rank=rank).eval()
    x = torch.randn(2, 11, 16)
    cache = None
    outputs = []
    with torch.no_grad():
        expected = model(x)
        for index in range(x.shape[1]):
            output, cache = model.step(x[:, index : index + 1], cache)
            outputs.append(output)
    torch.testing.assert_close(torch.cat(outputs, dim=1), expected, rtol=2e-5, atol=2e-6)

    mixer = model.layers[0].mixer
    phase, state, previous_k, previous_value = cache[0]
    assert phase.shape == (2, mixer.nheads, mixer.num_angles)
    assert state.shape == (2, mixer.nheads, mixer.headdim, mixer.d_state)
    assert state.dtype == torch.float32
    assert previous_k.shape == (2, mixer.nheads, rank, mixer.d_state)
    assert previous_value.shape == (2, mixer.nheads, mixer.headdim)


@pytest.mark.parametrize("rank", [1, 4])
def test_prefill_cache_continues_without_rescanning(rank: int) -> None:
    torch.manual_seed(3)
    model = Mamba3(16, d_state=8, depth=2, mimo_rank=rank).eval()
    x = torch.randn(2, 15, 16)
    split = 9
    with torch.no_grad():
        expected = model(x)
        prefix, cache = model.prefill(x[:, :split])
        outputs = [prefix]
        for index in range(split, x.shape[1]):
            output, cache = model.step(x[:, index : index + 1], cache)
            outputs.append(output)
    torch.testing.assert_close(
        torch.cat(outputs, dim=1), expected, rtol=2e-5, atol=2e-6
    )


def test_prefill_cache_does_not_retain_prefix_storage() -> None:
    model = Mamba3(32, d_state=16, depth=1).eval()
    with torch.inference_mode():
        _, cache = model.prefill(torch.randn(2, 257, 32))
    for tensor in (cache[0][0], cache[0][2], cache[0][3]):
        assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_phase_accumulation_is_bounded_per_chunk(device: str) -> None:
    mixer = Mamba3(16, d_state=8, depth=1).to(device).layers[0].mixer
    length = 2048
    q = torch.zeros(1, length, mixer.nheads, 1, mixer.d_state, device=device)
    angles = torch.full((1, length, mixer.num_angles), 3.0, device=device)
    dt = torch.full((1, length, mixer.nheads), 100.0, device=device)
    _, _, phase = mixer._phase_and_rotate(q, q, angles, dt, None)

    expected = []
    state = torch.zeros_like(phase[:, 0])
    increment = torch.fmod(
        math.pi
        * torch.tanh(angles[:, 0]).unsqueeze(1)
        * dt[:, 0].unsqueeze(-1),
        2.0 * math.pi,
    )
    for _ in range(length):
        state = torch.fmod(state + increment, 2.0 * math.pi)
        expected.append(state)
    expected = torch.stack(expected, dim=1)
    circular_error = torch.remainder(
        phase - expected + math.pi, 2.0 * math.pi
    ) - math.pi
    assert circular_error.abs().max() < 2e-3
    assert phase.abs().max() < 2.0 * math.pi


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_negative_phase_increments_do_not_drift_at_long_context() -> None:
    mixer = Mamba3(16, d_state=8, depth=1).cuda().layers[0].mixer
    length = 131_072
    target_increment = -1e-4
    raw_angle = math.atanh(target_increment / math.pi)
    angles = torch.full(
        (1, length, mixer.num_angles), raw_angle, device="cuda"
    )
    dt = torch.ones(1, length, mixer.nheads, device="cuda")
    phase = mixer._phase(angles, dt, None)

    increment = math.pi * math.tanh(raw_angle)
    positions = torch.arange(
        1, length + 1, device="cuda", dtype=torch.float64
    )
    expected = torch.remainder(positions * increment, 2.0 * math.pi).float()
    expected = expected[None, :, None, None].expand_as(phase)
    circular_error = torch.remainder(
        phase - expected + math.pi, 2.0 * math.pi
    ) - math.pi
    assert circular_error.abs().max() < 2e-3


def test_step_validates_input_and_cache() -> None:
    model = Mamba3(16, d_state=8, depth=2)
    with pytest.raises(ValueError, match="batch, 1"):
        model.step(torch.randn(2, 3, 16))
    with pytest.raises(ValueError, match="cache for 2"):
        model.step(torch.randn(2, 1, 16), cache=[])
    _, cache = model.step(torch.randn(2, 1, 16))
    invalid = list(cache)
    invalid[0] = (
        invalid[0][0],
        invalid[0][1],
        invalid[0][2][:1],
        invalid[0][3],
    )
    with pytest.raises(ValueError, match="cache shapes"):
        model.step(torch.randn(2, 1, 16), invalid)
    with pytest.raises(ValueError, match="non-empty"):
        model.prefill(torch.empty(2, 0, 16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("rank", [1, 4])
def test_cuda_bfloat16_forward_backward_and_cache(rank: int) -> None:
    torch.manual_seed(4)
    model = Mamba3(64, d_state=16, depth=1, mimo_rank=rank).cuda()
    x = torch.randn(2, 33, 64, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(x)
    output.float().square().mean().backward()
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    mixer = model.layers[0].mixer
    assert mixer.dt_bias.dtype == torch.float32
    assert mixer.B_bias.dtype == torch.float32
    assert mixer.D.dtype == torch.float32

    model.eval()
    cache = None
    step_outputs = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        normalized = model.layers[0].norm(x.detach())
        q, k, value, gate, *_ = mixer._project(normalized)
        assert q.dtype == k.dtype == value.dtype == gate.dtype == torch.bfloat16
        expected, prefill_cache = model.prefill(x.detach())
        for index in range(x.shape[1]):
            step_output, cache = model.step(x[:, index : index + 1], cache)
            step_outputs.append(step_output)
    torch.testing.assert_close(
        torch.cat(step_outputs, dim=1), expected, rtol=3e-2, atol=3e-2
    )
    for prefill_layer, step_layer in zip(prefill_cache, cache):
        for prefill_state, step_state in zip(prefill_layer, step_layer):
            torch.testing.assert_close(
                prefill_state.float(), step_state.float(), rtol=3e-2, atol=3e-2
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_graph_decoder_matches_forward_and_resets() -> None:
    torch.manual_seed(5)
    model = Mamba3(64, d_state=128, depth=1).cuda().eval()
    x = torch.randn(2, 5, 64, device="cuda")
    reference_cache = None
    reference_outputs = []
    with torch.inference_mode():
        for index in range(x.shape[1]):
            output, reference_cache = model.step(
                x[:, index : index + 1], reference_cache
            )
            reference_outputs.append(output)
    decoder = model.cuda_graph(batch_size=2)
    with torch.inference_mode():
        expected = model(x)
        actual = torch.cat(
            [decoder(x[:, index : index + 1]).clone() for index in range(5)],
            dim=1,
        )
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(
        actual, torch.cat(reference_outputs, dim=1), rtol=2e-4, atol=2e-5
    )
    for graph_layer, reference_layer in zip(decoder.cache, reference_cache):
        for graph_state, reference_state in zip(graph_layer, reference_layer):
            torch.testing.assert_close(
                graph_state, reference_state, rtol=2e-4, atol=2e-5
            )

    decoder.reset()
    with torch.inference_mode():
        reset_output = decoder(x[:, :1]).clone()
    torch.testing.assert_close(reset_output, expected[:, :1], rtol=2e-4, atol=2e-5)

    with torch.no_grad():
        model.layers[0].mixer.D.add_(0.1)
    with pytest.raises(RuntimeError, match="recreate"):
        decoder.validate()

    with pytest.raises(ValueError, match="gradient"):
        decoder(x[:, :1].requires_grad_())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 2e-4, 2e-5),
        (torch.float16, 2e-2, 2e-2),
        (torch.bfloat16, 3e-2, 3e-2),
    ],
)
def test_fused_triton_siso_step_matches_pytorch(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    rtol: float,
    atol: float,
) -> None:
    if importlib.util.find_spec("triton") is None:
        pytest.skip("Triton is unavailable")

    torch.manual_seed(6)
    model = Mamba3(64, d_state=16, depth=1).cuda().to(dtype).eval()
    x = torch.randn(2, 7, 64, device="cuda", dtype=dtype)
    monkeypatch.setenv("MAMBA3_DISABLE_TRITON", "1")
    reference_cache = None
    reference = []
    with torch.inference_mode():
        for index in range(x.shape[1]):
            output, reference_cache = model.step(
                x[:, index : index + 1], reference_cache
            )
            reference.append(output)

    monkeypatch.delenv("MAMBA3_DISABLE_TRITON")
    fused_cache = None
    fused = []
    with torch.inference_mode():
        for index in range(x.shape[1]):
            output, fused_cache = model.step(x[:, index : index + 1], fused_cache)
            fused.append(output)

    import mamba3.ops as ops

    assert ops._TRITON_STEP is not None
    torch.testing.assert_close(
        torch.cat(fused, dim=1),
        torch.cat(reference, dim=1),
        rtol=rtol,
        atol=atol,
    )
    for reference_layer, fused_layer in zip(reference_cache, fused_cache):
        for reference_state, fused_state in zip(reference_layer, fused_layer):
            torch.testing.assert_close(
                fused_state, reference_state, rtol=rtol, atol=atol
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="Triton is unavailable")
def test_fused_triton_mimo_step_matches_pytorch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(8)
    model = Mamba3(64, d_state=16, depth=1, mimo_rank=4).cuda().eval()
    x = torch.randn(2, 7, 64, device="cuda", dtype=torch.float32)
    monkeypatch.setenv("MAMBA3_DISABLE_TRITON", "1")
    reference_cache = None
    reference = []
    with torch.inference_mode():
        for index in range(x.shape[1]):
            output, reference_cache = model.step(
                x[:, index : index + 1], reference_cache
            )
            reference.append(output)

    monkeypatch.delenv("MAMBA3_DISABLE_TRITON")
    fused_cache = None
    fused = []
    with torch.inference_mode():
        for index in range(x.shape[1]):
            output, fused_cache = model.step(x[:, index : index + 1], fused_cache)
            fused.append(output)

    import mamba3.ops as ops

    assert ops._TRITON_LAST_DISPATCH
    torch.testing.assert_close(
        torch.cat(fused, dim=1),
        torch.cat(reference, dim=1),
        rtol=2e-4,
        atol=2e-5,
    )
    for reference_layer, fused_layer in zip(reference_cache, fused_cache):
        for reference_state, fused_state in zip(reference_layer, fused_layer):
            torch.testing.assert_close(
                fused_state, reference_state, rtol=2e-4, atol=2e-5
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="Triton is unavailable")
def test_compiled_forward_matches_eager() -> None:
    torch.manual_seed(9)
    model = Mamba3(64, d_state=16, depth=2, mimo_rank=4).cuda()
    x = torch.randn(2, 9, 64, device="cuda", requires_grad=True)
    reference = model(x)
    reference.square().mean().backward()
    reference_grads = [
        parameter.grad.clone() if parameter.grad is not None else None
        for parameter in model.parameters()
    ]

    model.zero_grad(set_to_none=True)
    compiled = model.compile()
    assert compiled is model
    actual = model(x)
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
    actual.square().mean().backward()
    for actual_grad, reference_grad in zip(
        (parameter.grad for parameter in model.parameters()), reference_grads
    ):
        if reference_grad is not None:
            assert actual_grad is not None
            torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-4, atol=1e-5)
