from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .ops import (
    fused_step,
    heavy_tail_activation,
    mamba3_scan,
    mamba3_step,
    rotate_qk,
)


_EXPANSION = 2
_CANONICAL_HEAD_DIM = 64
_ROPE_FRACTION = 0.5
_A_FLOOR = 1e-4
# Phase accumulation is kept in smaller chunks than the scan: the unbounded
# pre-fmod partial sums would otherwise lose FP32 precision at long context.
_PHASE_CHUNK = 64

_LayerCache = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = F.rms_norm(
            x,
            (x.shape[-1],),
            self.weight.to(dtype=x.dtype),
            self.eps,
        )
        return output.to(dtype=x.dtype)


class _Mamba3Mixer(nn.Module):
    """Canonical Mamba-3 SISO/MIMO mixer with a chunked SSD backend."""

    def __init__(self, d_model: int, d_state: int, mimo_rank: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.mimo_rank = mimo_rank
        self.d_inner = _EXPANSION * d_model
        # Preserve the canonical P=64 layout whenever possible while allowing
        # small feature sizes without adding configuration burden for users.
        self.headdim = math.gcd(self.d_inner, _CANONICAL_HEAD_DIM)
        self.nheads = self.d_inner // self.headdim
        self.ngroups = 1
        self.num_angles = int(d_state * _ROPE_FRACTION) // 2
        # The chunked SSD is chunk-size invariant (each chunk carries the
        # exact recurrence), so a large chunk is used to cut per-chunk kernel
        # and graph overhead; MIMO is capped lower to bound the rank-expanded
        # intra-chunk GEMM memory.
        self.chunk_size = 256 if mimo_rank == 1 else max(16, 128 // mimo_rank)

        projection_size = (
            2 * self.d_inner
            + 2 * d_state * self.ngroups * mimo_rank
            + 3 * self.nheads
            + self.num_angles
        )
        self.in_proj = nn.Linear(d_model, projection_size, bias=False)
        self.B_norm = _RMSNorm(d_state)
        self.C_norm = _RMSNorm(d_state)
        self.B_bias = nn.Parameter(torch.ones(self.nheads, mimo_rank, d_state))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, mimo_rank, d_state))

        dt = torch.exp(
            torch.rand(self.nheads)
            * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp_min(1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True

        if mimo_rank > 1:
            self.mimo_x = nn.Parameter(
                torch.full((self.nheads, mimo_rank, self.headdim), 1.0 / mimo_rank)
            )
            self.mimo_z = nn.Parameter(
                torch.ones(self.nheads, mimo_rank, self.headdim)
            )
            self.mimo_out = nn.Parameter(
                torch.full((self.nheads, mimo_rank, self.headdim), 1.0 / mimo_rank)
            )
        else:
            self.register_parameter("mimo_x", None)
            self.register_parameter("mimo_z", None)
            self.register_parameter("mimo_out", None)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _split_projection(
        self, projected: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        return torch.split(
            projected,
            (
                self.d_inner,
                self.d_inner,
                self.d_state * self.ngroups * self.mimo_rank,
                self.d_state * self.ngroups * self.mimo_rank,
                self.nheads,
                self.nheads,
                self.nheads,
                self.num_angles,
            ),
            dim=-1,
        )

    def _project(
        self, hidden_states: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch, length, _ = hidden_states.shape
        z, x, B, C, delta_logits, decay_logits, trap, angles = self._split_projection(
            self.in_proj(hidden_states)
        )
        z = z.reshape(batch, length, self.nheads, self.headdim)
        x = x.reshape(batch, length, self.nheads, self.headdim)
        B = B.reshape(
            batch, length, self.mimo_rank, self.ngroups, self.d_state
        )
        C = C.reshape(
            batch, length, self.mimo_rank, self.ngroups, self.d_state
        )

        B = self.B_norm(B)
        C = self.C_norm(C)
        k = B.permute(0, 1, 3, 2, 4)
        q = C.permute(0, 1, 3, 2, 4)

        dt = F.softplus(delta_logits.float() + self.dt_bias.float())
        decay = -heavy_tail_activation(decay_logits.float())
        decay = decay.clamp(max=-_A_FLOOR)
        adt = decay * dt
        return q, k, x, z, adt, dt, trap, angles

    def _expand_qk(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.shape[2] == 1:
            q = q.expand(-1, -1, self.nheads, -1, -1)
            k = k.expand(-1, -1, self.nheads, -1, -1)
        elif q.shape[2] != self.nheads:
            repeats = self.nheads // q.shape[2]
            q = q.repeat_interleave(repeats, dim=2)
            k = k.repeat_interleave(repeats, dim=2)
        q = q + self.C_bias[None, None].to(q.dtype)
        k = k + self.B_bias[None, None].to(k.dtype)
        return q, k

    def _phase(
        self,
        angles: torch.Tensor,
        dt: torch.Tensor,
        initial_phase: torch.Tensor | None,
    ) -> torch.Tensor:
        increments = (
            math.pi
            * torch.tanh(angles.float()).unsqueeze(2)
            * dt.unsqueeze(-1)
        )
        increments = torch.fmod(increments, 2.0 * math.pi)
        phase_state = (
            torch.zeros_like(increments[:, 0])
            if initial_phase is None
            else initial_phase
        )
        phase_chunks = []
        for start in range(0, increments.shape[1], _PHASE_CHUNK):
            chunk = increments[:, start : start + _PHASE_CHUNK]
            current = torch.fmod(
                torch.cumsum(chunk, dim=1) + phase_state.unsqueeze(1),
                2.0 * math.pi,
            )
            phase_chunks.append(current)
            phase_state = current[:, -1]
        return torch.cat(phase_chunks, dim=1)

    def _phase_and_rotate(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        angles: torch.Tensor,
        dt: torch.Tensor,
        initial_phase: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phase = self._phase(angles, dt, initial_phase)
        q, k = rotate_qk(q, k, phase, mimo=self.mimo_rank > 1)
        return q, k, phase

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: _LayerCache | None = None,
        *,
        return_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, _LayerCache]:
        q, k, x, z, adt, dt, trap, angles = self._project(hidden_states)
        initial_phase = cache[0] if cache is not None else None
        phase = self._phase(angles, dt, initial_phase)
        mixed, state = mamba3_scan(
            q,
            k,
            x,
            z,
            adt,
            dt,
            trap,
            self.D,
            mimo_x=self.mimo_x,
            mimo_z=self.mimo_z,
            mimo_out=self.mimo_out,
            chunk_size=self.chunk_size,
            initial_state=cache[1] if cache is not None else None,
            initial_k=cache[2] if cache is not None else None,
            initial_value=cache[3] if cache is not None else None,
            phase=phase,
            mimo_rotation=self.mimo_rank > 1,
            q_bias=self.C_bias,
            k_bias=self.B_bias,
        )
        output = self.out_proj(mixed.flatten(-2))
        if not return_cache:
            return output
        # Endpoint clones keep the cache storage constant in prefix length.
        final_q, final_k = self._expand_qk(q[:, -1:], k[:, -1:])
        _, final_k = rotate_qk(
            final_q,
            final_k,
            phase[:, -1:],
            mimo=self.mimo_rank > 1,
        )
        next_cache = (
            phase[:, -1].clone(),
            state,
            final_k[:, 0].clone(),
            x[:, -1].clone(),
        )
        return output, next_cache

    def allocate_cache(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> _LayerCache:
        return (
            torch.zeros(
                batch_size,
                self.nheads,
                self.num_angles,
                device=device,
                dtype=torch.float32,
            ),
            torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=device,
                dtype=torch.float32,
            ),
            torch.zeros(
                batch_size,
                self.nheads,
                self.mimo_rank,
                self.d_state,
                device=device,
                dtype=dtype,
            ),
            torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                device=device,
                dtype=dtype,
            ),
        )

    def _validate_cache(self, cache: _LayerCache, batch_size: int) -> None:
        if len(cache) != 4:
            raise ValueError("a Mamba-3 layer cache must contain four tensors")
        expected = (
            (batch_size, self.nheads, self.num_angles),
            (batch_size, self.nheads, self.headdim, self.d_state),
            (batch_size, self.nheads, self.mimo_rank, self.d_state),
            (batch_size, self.nheads, self.headdim),
        )
        actual = tuple(tuple(tensor.shape) for tensor in cache)
        if actual != expected:
            raise ValueError(f"expected cache shapes {expected}, got {actual}")
        if cache[0].dtype != torch.float32 or cache[1].dtype != torch.float32:
            raise ValueError("phase and SSM cache tensors must be float32")
        expected_device = self.in_proj.weight.device
        if any(tensor.device != expected_device for tensor in cache):
            raise ValueError(f"cache tensors must be on {expected_device}")

    def step(
        self,
        hidden_states: torch.Tensor,
        cache: _LayerCache | None,
    ) -> tuple[torch.Tensor, _LayerCache]:
        if hidden_states.ndim != 3 or hidden_states.shape[1:] != (1, self.d_model):
            raise ValueError(
                f"expected [batch, 1, {self.d_model}], got {tuple(hidden_states.shape)}"
            )
        q, k, x, z, adt, dt, trap, angles = self._project(hidden_states)
        q, k = self._expand_qk(q, k)
        if cache is None:
            cache = self.allocate_cache(
                hidden_states.shape[0],
                device=hidden_states.device,
                dtype=x.dtype,
            )
        else:
            self._validate_cache(cache, hidden_states.shape[0])
        fused = fused_step(
            q[:, 0],
            k[:, 0],
            x[:, 0],
            z[:, 0],
            adt[:, 0],
            dt[:, 0],
            trap[:, 0],
            angles[:, 0],
            self.D,
            cache,
            mimo_x=self.mimo_x,
            mimo_z=self.mimo_z,
            mimo_out=self.mimo_out,
        )
        if fused is not None:
            mixed, next_cache = fused
            return self.out_proj(mixed.flatten(-2)).unsqueeze(1), next_cache

        phase_increment = torch.fmod(
            math.pi
            * torch.tanh(angles[:, 0].float()).unsqueeze(1)
            * dt[:, 0].unsqueeze(-1),
            2.0 * math.pi,
        )
        phase = torch.fmod(cache[0] + phase_increment, 2.0 * math.pi)
        q, k = rotate_qk(
            q, k, phase.unsqueeze(1), mimo=self.mimo_rank > 1
        )
        mixed, next_cache = mamba3_step(
            q[:, 0],
            k[:, 0],
            x[:, 0],
            z[:, 0],
            adt[:, 0],
            dt[:, 0],
            trap[:, 0],
            self.D,
            (phase, cache[1], cache[2], cache[3]),
            mimo_x=self.mimo_x,
            mimo_z=self.mimo_z,
            mimo_out=self.mimo_out,
        )
        return self.out_proj(mixed.flatten(-2)).unsqueeze(1), next_cache


class _Mamba3Block(nn.Module):
    def __init__(self, d_model: int, d_state: int, mimo_rank: int) -> None:
        super().__init__()
        self.norm = _RMSNorm(d_model)
        self.mixer = _Mamba3Mixer(d_model, d_state, mimo_rank)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: _LayerCache | None = None,
        *,
        return_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, _LayerCache]:
        mixed = self.mixer(
            self.norm(hidden_states), cache=cache, return_cache=return_cache
        )
        if return_cache:
            output, next_cache = mixed
            return hidden_states + output, next_cache
        return hidden_states + mixed

    def step(
        self,
        hidden_states: torch.Tensor,
        cache: _LayerCache | None,
    ) -> tuple[torch.Tensor, _LayerCache]:
        output, next_cache = self.mixer.step(self.norm(hidden_states), cache)
        return hidden_states + output, next_cache


class Mamba3(nn.Module):
    """A shape-preserving stack of canonical Mamba-3 mixers.

    The minimal form is ``Mamba3(d_model)``. Inputs and outputs both use
    ``[batch, length, d_model]``. Set ``mimo_rank=4`` for the paper's stronger
    MIMO variant; the default rank of one is canonical SISO Mamba-3.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        depth: int = 4,
        mimo_rank: int = 1,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if d_state < 4:
            raise ValueError("d_state must be at least 4")
        if mimo_rank > 1 and d_state % 4 != 0:
            raise ValueError("MIMO requires d_state divisible by 4")
        if depth <= 0:
            raise ValueError("depth must be positive")
        if mimo_rank <= 0:
            raise ValueError("mimo_rank must be positive")

        self.d_model = d_model
        self.d_state = d_state
        self.depth = depth
        self.mimo_rank = mimo_rank
        self.layers = nn.ModuleList(
            _Mamba3Block(d_model, d_state, mimo_rank) for _ in range(depth)
        )
        self.norm_f = _RMSNorm(d_model)

        # GPT-2/Mamba residual scaling, with one residual branch per layer.
        with torch.no_grad():
            for layer in self.layers:
                layer.mixer.out_proj.weight.div_(math.sqrt(depth))

    def _validate_input(self, x: torch.Tensor, *, single_token: bool = False) -> None:
        expected = (1, self.d_model) if single_token else None
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"expected [batch, length, {self.d_model}], got {tuple(x.shape)}"
            )
        if expected is not None and x.shape[1:] != expected:
            raise ValueError(
                f"expected [batch, 1, {self.d_model}], got {tuple(x.shape)}"
            )

    def _empty_output(self, x: torch.Tensor) -> torch.Tensor:
        dependency = sum(parameter.sum() * 0.0 for parameter in self.parameters())
        return x + dependency.to(dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        if x.shape[0] == 0 or x.shape[1] == 0:
            return self._empty_output(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm_f(x)

    def prefill(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[_LayerCache]]:
        """Process a causal prefix and return the constant-size decode cache."""

        self._validate_input(x)
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("prefill requires a non-empty batch and sequence")
        cache: list[_LayerCache] = []
        for layer in self.layers:
            x, layer_cache = layer(x, return_cache=True)
            cache.append(layer_cache)
        return self.norm_f(x), cache

    def step(
        self,
        x: torch.Tensor,
        cache: list[_LayerCache] | None = None,
    ) -> tuple[torch.Tensor, list[_LayerCache]]:
        """Process one token while carrying Mamba-3's four recurrent states."""

        self._validate_input(x, single_token=True)
        if cache is None:
            layer_caches: list[_LayerCache | None] = [None] * len(self.layers)
        else:
            if len(cache) != len(self.layers):
                raise ValueError(f"expected cache for {len(self.layers)} layers")
            layer_caches = list(cache)

        next_cache: list[_LayerCache] = []
        for layer, layer_cache in zip(self.layers, layer_caches):
            x, updated = layer.step(x, layer_cache)
            next_cache.append(updated)
        return self.norm_f(x), next_cache

    def cuda_graph(self, batch_size: int = 1) -> _CudaGraphDecoder:
        """Capture the complete stateful decode step for minimum CUDA latency."""

        return _CudaGraphDecoder(self, batch_size)

    def compile(self, **kwargs: object) -> Mamba3:
        """Replace ``forward`` with a ``torch.compile``-optimized version.

        Inductor fuses the per-chunk elementwise chains of the scan (decay
        construction, diagonal fold, gate, D term) and the phase
        accumulation, leaving the tensor-core GEMMs to cuBLAS. The
        recurrence is unchanged; precision-sensitive FP32 state and phase
        handling are untouched. Pass ``torch.compile`` options (``dynamic``,
        ``mode``, ...) as keyword arguments. Compilation happens lazily on
        the first call; the returned module is this model.
        """

        self.forward = torch.compile(self.forward, **kwargs)  # type: ignore[method-assign]
        return self


class _CudaGraphDecoder:
    def __init__(self, model: Mamba3, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        parameter = next(model.parameters())
        if parameter.device.type != "cuda":
            raise ValueError("cuda_graph requires a CUDA model")
        if model.training:
            raise ValueError("cuda_graph requires model.eval()")

        self.model = model
        self.parameters = tuple(model.parameters())
        self.parameter_signature = self._parameter_signature()
        self.input = torch.zeros(
            batch_size,
            1,
            model.d_model,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        self.cache = [
            layer.mixer.allocate_cache(
                batch_size, device=parameter.device, dtype=parameter.dtype
            )
            for layer in model.layers
        ]

        with torch.cuda.device(parameter.device):
            warmup_stream = torch.cuda.Stream(device=parameter.device)
            warmup_stream.wait_stream(torch.cuda.current_stream(parameter.device))
            with torch.cuda.stream(warmup_stream), torch.inference_mode():
                model.step(self.input, self.cache)
            torch.cuda.current_stream(parameter.device).wait_stream(warmup_stream)

            capture_stream = torch.cuda.Stream(device=parameter.device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                self.graph, stream=capture_stream
            ), torch.inference_mode():
                self.output, next_cache = model.step(self.input, self.cache)
                for current, updated in zip(self.cache, next_cache):
                    for current_tensor, updated_tensor in zip(current, updated):
                        if current_tensor.data_ptr() != updated_tensor.data_ptr():
                            current_tensor.copy_(updated_tensor)
        self.reset()

    def __call__(self, x: torch.Tensor, validate: bool = False) -> torch.Tensor:
        if x.shape != self.input.shape:
            raise ValueError(f"expected {tuple(self.input.shape)}, got {tuple(x.shape)}")
        if x.device != self.input.device or x.dtype != self.input.dtype:
            raise ValueError(
                f"expected {self.input.dtype} on {self.input.device}, "
                f"got {x.dtype} on {x.device}"
            )
        if x.requires_grad:
            raise ValueError("CUDA graph decoding does not support gradient inputs")
        if validate:
            self.validate()
        with torch.inference_mode(), torch.cuda.device(self.input.device):
            self.input.copy_(x)
            self.graph.replay()
        return self.output

    def validate(self) -> None:
        if self._parameter_signature() != self.parameter_signature:
            raise RuntimeError("model state changed; recreate the CUDA graph decoder")

    def reset(self) -> None:
        for layer_cache in self.cache:
            for tensor in layer_cache:
                tensor.zero_()

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
