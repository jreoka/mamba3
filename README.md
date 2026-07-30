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

The bundled kernel fuses the selective recurrence, skip connection, and SiLU
gate into one launch. It uses one CUDA thread per batch/channel recurrence,
register-resident state, compile-time specializations for state sizes up to 64,
FP32 state accumulation, coalesced sequence I/O, and a separate handwritten
reverse recurrence for gradients. Training stores sparse recurrent checkpoints
and exactly recomputes short chunks in shared memory during the reverse pass
instead of retaining an `[batch, length, channels, d_state]` history. The Python
reference implementation is automatically used on CPU or if the local CUDA
toolchain cannot compile the extension.

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

For reference, the fused inference scan was validated at **4.398 ms / 3.73M
tokens per second** for `B=8, L=2048, H=512, N=16` on an RTX 4090 with PyTorch
2.14 nightly and CUDA 13.2. This number is a local measurement, not a guarantee
for other systems.

## Notes

- `d_state` can be 1 through 64; 8, 16, 32, and 64 map directly to kernel
  specializations.
- The fused kernel reads and writes FP16/BF16 activations directly while keeping
  the recurrent state and shared gradient accumulation in FP32 for stability.
- For the common `d_state=16`, training stores one recurrent checkpoint per
  eight sequence positions; inference stores no sequence-length state history.
- Non-causal mode runs both sequence directions and therefore costs roughly twice
  as much scan work as causal mode.

## License

MIT
