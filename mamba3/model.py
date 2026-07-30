from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import Mamba3Config
from .ops import load_cuda_extension, selective_scan


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


class Mamba3Block(nn.Module):
    """A pre-normalized selective state-space residual block."""

    def __init__(self, config: Mamba3Config, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        inner = config.d_inner
        rank = config.resolved_dt_rank

        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.in_proj = nn.Linear(config.d_model, 2 * inner, bias=config.bias)
        self.conv1d = nn.Conv1d(
            inner,
            inner,
            kernel_size=config.d_conv,
            groups=inner,
            padding=0,
            bias=config.conv_bias,
        )
        self.param_proj = nn.Linear(inner, rank + 2 * config.d_state, bias=False)
        self.dt_proj = nn.Linear(rank, inner, bias=True)
        self.A_log = nn.Parameter(torch.empty(inner, config.d_state))
        self.D = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        if not config.causal:
            self.direction_mix = nn.Parameter(torch.tensor(0.0))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Log-spaced stable continuous-time dynamics.
        base = torch.arange(1, self.config.d_state + 1, dtype=torch.float32)
        with torch.no_grad():
            self.A_log.copy_(base.log().repeat(self.config.d_inner, 1))
            # Initialize dt in the useful 1e-3 .. 1e-1 range.
            dt = torch.exp(
                torch.rand(self.config.d_inner)
                * (math.log(0.1) - math.log(0.001))
                + math.log(0.001)
            ).clamp_min(1e-4)
            inverse_softplus = dt + torch.log(-torch.expm1(-dt))
            self.dt_proj.bias.copy_(inverse_softplus)

    def _convolve(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1d expects [B, H, L]. Explicit padding keeps causality obvious.
        x = x.transpose(1, 2)
        if self.config.causal:
            x = F.pad(x, (self.config.d_conv - 1, 0))
        else:
            left = (self.config.d_conv - 1) // 2
            right = self.config.d_conv - 1 - left
            x = F.pad(x, (left, right))
        return F.silu(self.conv1d(x)).transpose(1, 2)

    def _scan_direction(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        params = self.param_proj(x)
        rank = self.config.resolved_dt_rank
        dt_low_rank, B, C = torch.split(
            params, [rank, self.config.d_state, self.config.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_low_rank)).clamp(max=1.0)
        A = -torch.exp(self.A_log.float())
        return selective_scan(
            x,
            dt,
            A,
            B,
            C,
            self.D,
            z,
            use_cuda_kernel=self.config.use_cuda_kernel,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.config.d_model:
            raise ValueError(
                f"expected [batch, length, {self.config.d_model}], got {tuple(hidden_states.shape)}"
            )
        residual = hidden_states
        x, z = self.in_proj(self.norm(hidden_states)).chunk(2, dim=-1)
        x = self._convolve(x)
        forward = self._scan_direction(x, z)
        if self.config.causal:
            mixed = forward
        else:
            backward = self._scan_direction(x.flip(1), z.flip(1)).flip(1)
            mix = torch.sigmoid(self.direction_mix)
            mixed = mix * forward + (1.0 - mix) * backward
        return residual + self.dropout(self.out_proj(mixed))


class Mamba3(nn.Module):
    """Easy-to-use sequence model with shape-preserving output.

    Example: ``Mamba3(d_model=256, d_state=16, depth=6, causal=True)``.
    Input and output both use ``[batch, length, d_model]``.
    """

    def __init__(
        self,
        d_model: int | Mamba3Config,
        d_state: int = 16,
        depth: int = 4,
        causal: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        if isinstance(d_model, Mamba3Config):
            if kwargs or d_state != 16 or depth != 4 or causal is not True:
                raise ValueError("pass either Mamba3Config or keyword configuration, not both")
            config = d_model
        else:
            config = Mamba3Config(
                d_model=d_model,
                d_state=d_state,
                depth=depth,
                causal=causal,
                **kwargs,
            )
        self.config = config
        self.layers = nn.ModuleList(
            Mamba3Block(config, layer_idx=index) for index in range(config.depth)
        )
        self.norm_f = RMSNorm(config.d_model, config.norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm_f(x)

    @staticmethod
    def compile_kernels(verbose: bool = True) -> bool:
        """Eagerly compile/load the CUDA extension; normally this is lazy."""

        return load_cuda_extension(verbose=verbose) is not None


class Mamba3LM(nn.Module):
    """Minimal language-model wrapper with tied token embeddings."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_state: int = 16,
        depth: int = 4,
        causal: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.backbone = Mamba3(d_model, d_state, depth, causal, **kwargs)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.backbone(self.embedding(input_ids)))
