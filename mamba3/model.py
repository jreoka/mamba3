from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .ops import selective_scan


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
        z: torch.Tensor,
        reverse: bool = False,
    ) -> torch.Tensor:
        return selective_scan(
            x,
            dt,
            A,
            B,
            C,
            self.D,
            z,
            reverse=reverse,
        )

    def _project_scan_parameters(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        params = self.param_proj(x)
        dt_low_rank, B, C = torch.split(
            params, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_low_rank)).clamp(max=1.0)
        A = -torch.exp(self.A_log.float())
        return dt, A, B, C

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.d_model:
            raise ValueError(
                f"expected [batch, length, {self.d_model}], got {tuple(hidden_states.shape)}"
            )
        residual = hidden_states
        x, z = self.in_proj(self.norm(hidden_states)).chunk(2, dim=-1)
        x = self._convolve(x)
        dt, A, B, C = self._project_scan_parameters(x)
        forward = self._scan_direction(x, dt, A, B, C, z)
        if self.causal:
            mixed = forward
        else:
            backward = self._scan_direction(
                x,
                dt,
                A,
                B,
                C,
                z,
                reverse=True,
            )
            mix = torch.sigmoid(self.direction_mix)
            mixed = mix * forward + (1.0 - mix) * backward
        return residual + self.out_proj(mixed)


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
        for layer in self.layers:
            x = layer(x)
        return self.norm_f(x)
