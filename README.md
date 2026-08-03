# mamba3

A compact PyTorch selective state-space sequence model with fused CUDA kernels
and a CPU fallback.

## Install

Install PyTorch for your platform, then install `mamba3`:

```bash
pip install git+https://github.com/jreoka/mamba3.git
```

## Usage

```python
import torch
from mamba3 import Mamba3

model = Mamba3(
    d_model=256,
    d_state=16,
    depth=6,
    causal=True,
)

x = torch.randn(2, 1024, 256)
y = model(x)  # [2, 1024, 256]
```

`Mamba3` is the only public API. It accepts four options:

- `d_model`: input and output feature size.
- `d_state=16`: state size for each layer.
- `depth=4`: number of layers.
- `causal=True`: prevent future positions from affecting earlier outputs. Set
  this to `False` for bidirectional context.

Input and output use `[batch, length, d_model]`. Move the model and input to
CUDA to use the fused kernels; the extension builds lazily on first use.
`d_state` may be any positive value; values above 64 use the generic CUDA path.

For causal decoding, carry the returned cache instead of rescanning the prefix:

```python
cache = None
model.eval()
with torch.inference_mode():
    for position in range(x.shape[1]):
        y, cache = model.step(x[:, position : position + 1], cache)
```

For fixed-batch CUDA inference, capture the complete recurrent step to minimize
launch overhead. The returned tensor is reused by each call, so clone it when
retaining multiple outputs.

```python
model = model.cuda().eval()
decoder = model.cuda_graph(batch_size=1)
y = decoder(x[:, :1].cuda())
```

> This is an independent selective state-space implementation, not an official
> implementation from the authors of the Mamba research papers.

## Development

```bash
pip install -e ".[dev]"
pytest
python benchmarks/benchmark_model.py
```

## License

MIT
