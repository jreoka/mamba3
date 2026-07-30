from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Mamba3Config:
    """Configuration for :class:`mamba3.Mamba3`.

    The four most useful knobs are deliberately first: ``d_model``,
    ``d_state``, ``depth``, and ``causal``.
    """

    d_model: int
    d_state: int = 16
    depth: int = 4
    causal: bool = True
    expand: int = 2
    d_conv: int = 4
    dt_rank: int | None = None
    dropout: float = 0.0
    bias: bool = False
    conv_bias: bool = True
    norm_eps: float = 1e-5
    use_cuda_kernel: bool = True

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.d_state <= 0 or self.d_state > 64:
            raise ValueError("d_state must be in [1, 64]")
        if self.depth <= 0:
            raise ValueError("depth must be positive")
        if self.expand <= 0:
            raise ValueError("expand must be positive")
        if self.d_conv <= 0:
            raise ValueError("d_conv must be positive")
        if self.dt_rank is not None and self.dt_rank <= 0:
            raise ValueError("dt_rank must be positive when provided")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def resolved_dt_rank(self) -> int:
        return self.dt_rank or max(1, (self.d_model + 15) // 16)
