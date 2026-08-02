# mamba3

`mamba3` is a compact PyTorch selective state-space sequence model with a small,
friendly API and a fused, handwritten CUDA scan. It supports causal and
bidirectional modeling, CPU fallback, training, mixed-precision inputs, and a
minimal language-model wrapper.

```python
import torch
from mamba3 import Mamba3

model = Mamba3(
    d_model=256,
    d_state=16,
    depth=6,
    causal=True,
).cuda()

x = torch.randn(2, 1024, 256, device="cuda")
y = model(x)  # [2, 1024, 256]
```

> This is an independent selective state-space implementation. The project is
> not an official implementation from the authors of the Mamba research papers.

## Install

Install PyTorch for your platform first, then install this repository:

```bash
pip install git+https://github.com/jreoka/mamba3.git
```

For development:

```bash
git clone https://github.com/jreoka/mamba3.git
cd mamba3
pip install -e ".[dev]"
pytest
```

The CUDA extension builds lazily on the first CUDA forward pass and is cached by
PyTorch. This keeps ordinary installation working on CPU-only machines. To build
the extension during installation instead:

```bash
MAMBA3_BUILD_CUDA=1 pip install --no-build-isolation .
```

On Windows, run that command from a Visual Studio developer shell. A C++17
compiler, the CUDA toolkit, and a CUDA-enabled PyTorch installation are needed.

## API

The main class preserves sequence shape and only requires `d_model`:

```python
from mamba3 import Mamba3, Mamba3Config

model = Mamba3(d_model=512)
model = Mamba3(d_model=512, d_state=32, depth=8, causal=False)

config = Mamba3Config(
    d_model=512,
    d_state=16,
    depth=12,
    causal=True,
    expand=2,
    d_conv=4,
    dropout=0.0,
)
model = Mamba3(config)
```

`causal=True` prevents future tokens from affecting earlier outputs.
`causal=False` combines forward and reverse selective scans for bidirectional
context.

For token logits, use the tied-embedding language-model wrapper:

```python
from mamba3 import Mamba3LM

lm = Mamba3LM(vocab_size=32_000, d_model=512, d_state=16, depth=8)
logits = lm(input_ids)  # [batch, length, vocab_size]
```

## CUDA design

The bundled kernels fuse the selective recurrence, skip connection, and SiLU
gate into one launch per pass. Dispatch selects between two complementary
implementations:

- Long sequences use a sequence-parallel scan. Each block owns one
  ``(batch, channel)`` row and scans a 2048-position chunk with 128 threads.
- High-batch, short-sequence shapes use a row-parallel scan. One thread owns a
  recurrent row, avoiding mostly idle 128-thread blocks when many independent
  rows already provide ample GPU parallelism. Reverse scans index the original
  tensors directly, avoiding full-tensor flips in bidirectional models.

This matters for dual-path audio models, which commonly reshape one axis into a
large effective batch while scanning sequences of only a few hundred tokens.

Key properties:

- The backward pass splits into two kernels: one computes every gradient
  except ``grad_B``/``grad_C``, and a second one computes only ``grad_B``/
  ``grad_C`` with a cross-channel group reduction (one block per batch/state
  row, 32 channels reduced in registers before a single atomic add per
  position). This cuts the atomic traffic by 32x, which dominates the naive
  design's backward cost.
- Sparse recurrent checkpoints (one (decay, state) pair per chunk) are stored
  instead of a full ``[batch, length, channels, d_state]`` history; the
  backward kernels exactly recompute each chunk. The row-parallel path uses
  shorter checkpoint intervals so each backward chunk fits in shared memory.
- Per-position decays use ``exp2f(A * log2(e) * dt)``; the recurrent state and
  shared gradient accumulation stay in FP32.
- Channel-major layouts (``[B, H, L]`` / ``[B, N, L]``) keep loads coalesced;
  a tiled shared-memory transpose kernel converts to and from the public
  ``[B, L, H]`` API cheaply, replacing PyTorch's slow generic transpose.
- The Python reference implementation is automatically used on CPU or if the
  local CUDA toolchain cannot compile the extension.

You can precompile the lazy extension explicitly:

```python
from mamba3 import Mamba3
Mamba3.compile_kernels(verbose=True)
```

Environment controls:

- `MAMBA3_DISABLE_CUDA=1`: always use the reference implementation.
- `MAMBA3_STRICT_CUDA=1`: raise instead of falling back after a build error.
- `MAMBA3_VERBOSE_BUILD=1`: print extension compiler output.

## Benchmark

```bash
python benchmarks/benchmark_scan.py --batch 8 --length 2048 --channels 512 --state 16
python benchmarks/benchmark_scan.py --training --dtype bfloat16 \
  --batch 16 --length 512 --channels 512 --state 16
```

Performance depends heavily on GPU, CUDA/PyTorch versions, shapes, dtype, and
whether gradients are enabled. Run the included benchmark on the deployment
machine instead of relying on a universal speed claim.

For reference, on an RTX 4090 with PyTorch 2.14 nightly and CUDA 13.2 the fused
inference scan measures about **0.36 ms / 46M tokens per second** and the
training scan (forward + backward) about **2.1 ms / 7.7M tokens per second**
for ``B=8, L=2048, H=512, N=16`` in BF16. These are local measurements, not
guarantees for other systems.

## Notes

- `d_state` can be 1 through 64; the kernel handles any value in that range.
- The fused kernels read and write FP16/BF16 activations directly while keeping
  the recurrent state and shared gradient accumulation in FP32 for stability.
- For the common `d_state=16`, training stores one recurrent checkpoint per
  2048-position chunk; inference stores no sequence-length state history.
- Non-causal mode runs both sequence directions and therefore costs roughly twice
  as much scan work as causal mode.

## License

MIT
