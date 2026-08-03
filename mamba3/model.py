from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .ops import (
    _causal_conv_forward,
    _causal_conv_step,
    _selective_scan_step,
    selective_scan,
)


_EXPANSION = 2
_CONV_KERNEL_SIZE = 4


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Matching the weight dtype enables PyTorch's fused RMSNorm path. The
        # previous implementation materialized the full activation in FP32
        # twice, which was especially expensive for long BF16 audio sequences.
        return F.rms_norm(
            x,
            (x.shape[-1],),
            self.weight.to(dtype=x.dtype),
            self.eps,
        )


class _Mamba3Block(nn.Module):
    """A pre-normalized selective state-space residual block."""

    def __init__(self, d_model: int, d_state: int, causal: bool) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.causal = causal
        self.d_inner = _EXPANSION * d_model
        self.dt_rank = max(1, (d_model + 15) // 16)

        self.norm = _RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=_CONV_KERNEL_SIZE,
            groups=self.d_inner,
            padding=0,
            bias=True,
        )
        self.param_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.empty(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        if not causal:
            self.direction_mix = nn.Parameter(torch.tensor(0.0))
        self._inference_constants: tuple[tuple[object, ...], torch.Tensor, torch.Tensor] | None = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Log-spaced stable continuous-time dynamics.
        base = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        with torch.no_grad():
            self.A_log.copy_(base.log().repeat(self.d_inner, 1))
            # Initialize dt in the useful 1e-3 .. 1e-1 range.
            dt = torch.exp(
                torch.rand(self.d_inner)
                * (math.log(0.1) - math.log(0.001))
                + math.log(0.001)
            ).clamp_min(1e-4)
            inverse_softplus = dt + torch.log(-torch.expm1(-dt))
            self.dt_proj.bias.copy_(inverse_softplus)

    def _convolve(self, x: torch.Tensor) -> torch.Tensor:
        if self.causal:
            fused = _causal_conv_forward(x, self.conv1d.weight, self.conv1d.bias)
            if fused is not None:
                return fused.transpose(1, 2)
        # Conv1d expects [B, H, L]. Explicit padding keeps causality obvious.
        x = x.transpose(1, 2)
        if self.causal:
            x = F.pad(x, (_CONV_KERNEL_SIZE - 1, 0))
        else:
            left = (_CONV_KERNEL_SIZE - 1) // 2
            right = _CONV_KERNEL_SIZE - 1 - left
            x = F.pad(x, (left, right))
        return F.silu(self.conv1d(x)).transpose(1, 2)

    def _scan_direction(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        z: torch.Tensor,
        dt_bias: torch.Tensor | None = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        return selective_scan(
            x,
            dt,
            A,
            B,
            C,
            D,
            z,
            reverse=reverse,
            _dt_bias=dt_bias,
        )

    def _project_scan_parameters(
        self, x: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        if x.is_cuda and not x.is_contiguous():
            weight = self.param_proj.weight.t().unsqueeze(0).expand(x.shape[0], -1, -1)
            params = torch.bmm(x, weight)
        else:
            params = self.param_proj(x)
        dt_low_rank, B, C = torch.split(
            params, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        if not torch.is_grad_enabled() and x.is_cuda:
            dt = F.linear(dt_low_rank, self.dt_proj.weight)
            dt_bias = self.dt_proj.bias
        else:
            dt = F.softplus(self.dt_proj(dt_low_rank)).clamp(max=1.0)
            dt_bias = None
        A, D = self._scan_constants()
        return dt, A, B, C, D, dt_bias

    def _scan_constants(self) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.is_grad_enabled():
            return -torch.exp(self.A_log.float()), self.D

        key = (
            self.A_log._version,
            self.A_log.device,
            self.A_log.dtype,
            self.A_log.data_ptr(),
            self.D._version,
            self.D.device,
            self.D.dtype,
            self.D.data_ptr(),
        )
        if self._inference_constants is None or self._inference_constants[0] != key:
            self._inference_constants = (
                key,
                -torch.exp(self.A_log.float()),
                self.D.float().contiguous(),
            )
        return self._inference_constants[1], self._inference_constants[2]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.d_model:
            raise ValueError(
                f"expected [batch, length, {self.d_model}], got {tuple(hidden_states.shape)}"
            )
        residual = hidden_states
        x, z = self.in_proj(self.norm(hidden_states)).chunk(2, dim=-1)
        x = self._convolve(x)
        dt, A, B, C, D, dt_bias = self._project_scan_parameters(x)
        forward = self._scan_direction(x, dt, A, B, C, D, z, dt_bias)
        if self.causal:
            mixed = forward
        else:
            backward = self._scan_direction(
                x,
                dt,
                A,
                B,
                C,
                D,
                z,
                dt_bias,
                reverse=True,
            )
            mix = torch.sigmoid(self.direction_mix)
            mixed = mix * forward + (1.0 - mix) * backward
        return residual + self.out_proj(mixed)

    def step(
        self,
        hidden_states: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if not self.causal:
            raise ValueError("cached decoding requires causal=True")
        if hidden_states.ndim != 3 or hidden_states.shape[1:] != (1, self.d_model):
            raise ValueError(
                f"expected [batch, 1, {self.d_model}], got {tuple(hidden_states.shape)}"
            )

        batch = hidden_states.shape[0]
        if cache is None:
            conv_state = hidden_states.new_zeros(
                batch, self.d_inner, _CONV_KERNEL_SIZE - 1
            )
            ssm_state = torch.zeros(
                batch,
                self.d_inner,
                self.d_state,
                device=hidden_states.device,
                dtype=torch.float32,
            )
        else:
            conv_state, ssm_state = cache
            expected_conv = (batch, self.d_inner, _CONV_KERNEL_SIZE - 1)
            expected_ssm = (batch, self.d_inner, self.d_state)
            if conv_state.shape != expected_conv or ssm_state.shape != expected_ssm:
                raise ValueError(
                    f"expected cache shapes {expected_conv} and {expected_ssm}, got "
                    f"{tuple(conv_state.shape)} and {tuple(ssm_state.shape)}"
                )

        residual = hidden_states
        x, z = self.in_proj(self.norm(hidden_states)).chunk(2, dim=-1)
        x, conv_state = _causal_conv_step(
            x, conv_state, self.conv1d.weight, self.conv1d.bias
        )
        params = self.param_proj(x)
        dt_low_rank, B, C = torch.split(
            params, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt_logits = F.linear(dt_low_rank, self.dt_proj.weight)
        A, D = self._scan_constants()
        scan_result = _selective_scan_step(
            x, dt_logits, self.dt_proj.bias, A, B, C, D, z, ssm_state
        )
        if scan_result is None:
            dt = F.softplus(dt_logits + self.dt_proj.bias).clamp(max=1.0)
            mixed, ssm_state = selective_scan(
                x,
                dt,
                A,
                B,
                C,
                D,
                z,
                initial_state=ssm_state,
                return_state=True,
                _force_row=True,
            )
        else:
            mixed, ssm_state = scan_result
        output = residual + self.out_proj(mixed)
        return output, (conv_state, ssm_state)


class Mamba3(nn.Module):
    """Shape-preserving selective state-space sequence model.

    Example: ``Mamba3(d_model=256, d_state=16, depth=6, causal=True)``.
    Input and output both use ``[batch, length, d_model]``.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        depth: int = 4,
        causal: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if d_state <= 0:
            raise ValueError("d_state must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.layers = nn.ModuleList(
            _Mamba3Block(d_model, d_state, causal) for _ in range(depth)
        )
        self.norm_f = _RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.layers[0].d_model:
            raise ValueError(
                f"expected [batch, length, {self.layers[0].d_model}], got {tuple(x.shape)}"
            )
        if x.shape[0] == 0 or x.shape[1] == 0:
            dependency = sum(parameter.sum() * 0.0 for parameter in self.parameters())
            return x + dependency.to(dtype=x.dtype)
        for layer in self.layers:
            x = layer(x)
        return self.norm_f(x)

    def step(
        self,
        x: torch.Tensor,
        cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Process one causal position while carrying convolution and SSM state."""

        if cache is None:
            layer_caches: list[tuple[torch.Tensor, torch.Tensor] | None] = [
                None
            ] * len(self.layers)
        else:
            if len(cache) != len(self.layers):
                raise ValueError(f"expected cache for {len(self.layers)} layers")
            layer_caches = list(cache)

        next_cache = []
        for layer, layer_cache in zip(self.layers, layer_caches):
            x, updated_cache = layer.step(x, layer_cache)
            next_cache.append(updated_cache)
        return self.norm_f(x), next_cache

    def cuda_graph(self, batch_size: int = 1) -> _CudaGraphDecoder:
        """Capture a stateful causal CUDA step for minimum decoding latency."""

        return _CudaGraphDecoder(self, batch_size)


class _CudaGraphDecoder:
    def __init__(self, model: Mamba3, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        parameter = next(model.parameters())
        if parameter.device.type != "cuda":
            raise ValueError("cuda_graph requires a CUDA model")
        if model.training:
            raise ValueError("cuda_graph requires model.eval()")
        if any(not layer.causal for layer in model.layers):
            raise ValueError("cuda_graph requires causal=True")
        self.model = model
        self.parameters = tuple(model.parameters())
        self.parameter_signature = self._parameter_signature()

        self.input = torch.zeros(
            batch_size,
            1,
            model.layers[0].d_model,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        self.cache = [
            (
                torch.zeros(
                    batch_size,
                    layer.d_inner,
                    _CONV_KERNEL_SIZE - 1,
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
                torch.zeros(
                    batch_size,
                    layer.d_inner,
                    layer.d_state,
                    device=parameter.device,
                    dtype=torch.float32,
                ),
            )
            for layer in model.layers
        ]

        warmup_stream = torch.cuda.Stream(device=parameter.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(parameter.device))
        with torch.cuda.stream(warmup_stream), torch.inference_mode():
            model.step(self.input, None)
        torch.cuda.current_stream(parameter.device).wait_stream(warmup_stream)
        self.constants = tuple(
            tensor
            for layer in model.layers
            for tensor in layer._inference_constants[1:]
        )

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.inference_mode():
            self.output, next_cache = model.step(self.input, self.cache)
            for current, updated in zip(self.cache, next_cache):
                current[0].copy_(updated[0])
                current[1].copy_(updated[1])
        self.reset()

    def __call__(self, x: torch.Tensor, validate: bool = True) -> torch.Tensor:
        if x.shape != self.input.shape:
            raise ValueError(f"expected {tuple(self.input.shape)}, got {tuple(x.shape)}")
        if validate:
            self.validate()
        self.input.copy_(x)
        self.graph.replay()
        return self.output

    def validate(self) -> None:
        if self._parameter_signature() != self.parameter_signature:
            raise RuntimeError("model state changed; recreate the CUDA graph decoder")

    def reset(self) -> None:
        for conv_state, ssm_state in self.cache:
            conv_state.zero_()
            ssm_state.zero_()

    def _parameter_signature(self) -> tuple[tuple[object, ...], ...]:
        return tuple(
            (
                id(parameter),
                parameter._version,
                parameter.data_ptr(),
                parameter.device,
                parameter.dtype,
                tuple(parameter.shape),
            )
            for parameter in self.model.parameters()
        )
