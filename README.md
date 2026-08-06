# mamba3

A compact, faithful implementation of the
[Mamba-3](https://arxiv.org/abs/2603.15569) sequence mixer for PyTorch. It
includes canonical SISO Mamba-3, rank-4 MIMO, a differentiable CPU/CUDA
backend, constant-memory decoding, and automatically fused CUDA kernels for
both the chunked SSD (training and prefill) and decoding when Triton is
available. The fused kernels implement the exact same recurrence with fewer,
larger launches.

## Install

Install PyTorch for your platform, then install `mamba3`:

```bash
pip install git+https://github.com/jreoka/mamba3.git
```

Triton is optional. Without it, the same API and equations run through the
portable PyTorch backend.

## Usage

```python
import torch
from mamba3 import Mamba3

model = Mamba3(d_model=256, depth=6)
x = torch.randn(2, 1024, 256)
y = model(x)  # [2, 1024, 256]
```

`Mamba3` is the only public class. Its options are:

- `d_model`: input and output feature size.
- `d_state=128`: SSM state width. `128` is the paper/checkpoint default.
- `depth=4`: number of residual Mamba-3 mixers.
- `mimo_rank=1`: canonical SISO. Use `4` for the stronger MIMO variant.

Input and output use `[batch, length, d_model]`. Mamba-3 is inherently causal.
The class is a shape-preserving sequence backbone, so it can sit between any
embedding and task head; it does not impose a tokenizer or vocabulary.
The paper's language-model backbone additionally interleaves SwiGLU residual
blocks and adds embedding/head layers. This mixer-only package intentionally
leaves those choices to the caller.

For efficient training or prefill on BF16-capable NVIDIA GPUs, keep FP32 master
parameters and use BF16 autocast:

```python
model = model.cuda()
x = x.cuda()
with torch.autocast("cuda", dtype=torch.bfloat16):
    y = model(x)
```

`torch.compile` fuses the per-chunk elementwise chains of the scan on top of
the cuBLAS GEMMs; the recurrence and the FP32 state/phase handling are
unchanged:

```python
model.compile()  # returns the same model with an optimized forward
```

## Generation

Build all four recurrent states from a prefix once, then process one token at
a time. Cache memory is constant in sequence length.

```python
model.eval()
with torch.inference_mode():
    prefix_output, cache = model.prefill(x)
    next_output, cache = model.step(next_x, cache)  # next_x: [B, 1, D]
```

For a fixed CUDA batch, capture the complete step to reduce Python and
per-operation launch overhead. The returned output storage is reused, so clone
outputs that must be retained.

```python
decoder = model.cuda().eval().cuda_graph(batch_size=1)
next_output = decoder(next_x.cuda())
decoder.reset()  # begin a new sequence
```

The fused Triton kernels are selected automatically for supported low-
precision CUDA shapes: the chunked SSD (one launch per chunk for both the
token outputs and the state update) and the SISO/MIMO decode step. Set
`MAMBA3_DISABLE_TRITON=1` to force the portable PyTorch path.

## What Is Implemented

This is not the older Mamba-1 selective scan under a new name. Each mixer uses
the Mamba-3 projection order

```text
[z, x, B, C, delta, A, lambda, angle]
```

and includes:

- Data-dependent negative heavy-tail decay `A`.
- Learned exponential-trapezoidal discretization.
- RMS-normalized `B/C` with head-specific positive biases.
- Cumulative data-dependent complex rotations in real coordinates.
- Per-head skip `D` and SiLU output gating.
- The paper's shared-state MIMO parameterization when `mimo_rank > 1`.
- No explicit short convolution, matching Mamba-3.

For each head, the recurrent update is

```text
alpha = exp(A * delta)
beta  = (1 - sigmoid(lambda)) * delta * alpha
gamma = sigmoid(lambda) * delta

S_t = alpha * S_(t-1) + beta * U_(t-1) + gamma * U_t
```

where `U_t` is the rank-1 SISO or rank-R MIMO outer-product input. Full
sequences use a chunked SSD formulation that uses tensor cores on supported
low-precision CUDA hardware (inference/prefill collapses each chunk into two
fused Triton launches); decoding uses the mathematically identical
recurrence with FP32 state.

## Development

```bash
pip install -e ".[dev]"
pytest
python benchmarks/benchmark_model.py
```

## License

Original project code is MIT licensed. `mamba3/_triton.py` contains a modified
decoder derived from the official Apache-2.0 Mamba implementation; see
`NOTICE` and `LICENSE-APACHE`.

This is an independent package and is not an official release from the Mamba
authors.
