# torchpathint

Fast, GPU-friendly definite integrals on PyTorch.

```
I = ∫_{t_init}^{t_final} f(t) dt
```

`torchpathint` evaluates the integrand at many time points in a single
batched call, which makes it well-suited for GPU and for integrals where
`f` is something like a neural network sampled along a path.

> **Status:** pre-alpha. API may change. Not yet on PyPI.

## Install

```bash
git clone https://github.com/ericyuan00000/torchpathint.git
cd torchpathint
pip install -e .
```

Requires Python 3.10+ and PyTorch.

## Quickstart

```python
import math
import torch
from torchpathint import path_integral

# f returns shape [N, D]: N time points, D output dimensions.
# Here D = 1 because sin(t) is scalar — note the trailing unsqueeze(-1).
def f(t):
    return torch.sin(t).unsqueeze(-1)

out = path_integral(f, 0.0, math.pi)
print(out.integral)   # tensor([2.0000])  ≈ ∫_0^π sin(t) dt = 2
```

A guided tour with sharper integrands, vector outputs, and warm starts is
in [`examples/quickstart.py`](examples/quickstart.py).

## Documentation

Full docs at <https://ericyuan00000.github.io/torchpathint/>:

- **Theory** — how the integrator works, in plain language.
- **API reference** — every function, argument, and return value.
- **Memory probe** — bounding peak GPU memory automatically.
- **Migration guide** — porting code from `torchpathdiffeq`.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

Preview the docs site locally:

```bash
pip install -e ".[docs]"
mkdocs serve            # http://127.0.0.1:8000 with live reload
```

## License

CC-BY-4.0. See [`LICENSE.md`](LICENSE.md).
