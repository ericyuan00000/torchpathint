# torchpathint

Parallel adaptive quadrature for definite path integrals on PyTorch.

`torchpathint` computes definite integrals

```
I = ∫_{t_init}^{t_final} f(t) dt
```

of vector-valued integrands `f: ℝ → ℝᴰ` (e.g. a neural network sampled
along a path, an action integrand, a reward signal). It exploits the fact
that — unlike an ODE — every quadrature evaluation is independent: all
nodes in all subintervals can be evaluated in a single batched call to
`f`, then weighted, then refined. On a GPU this is typically two orders
of magnitude faster than a sequential ODE-based integrator at the same
accuracy.

This is a clean rebuild of [`torchpathdiffeq`](https://github.com/khegazy/torchpathdiffeq),
swapping Runge–Kutta ODE solvers for adaptive Gauss–Kronrod quadrature
and decoupling GPU memory from the quadrature order.

## Status

Pre-alpha. API may change. Not yet on PyPI.

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

# f: Tensor[N] -> Tensor[N, D]. Note the trailing D dim, even for D=1.
def f(t):
    return torch.sin(t).unsqueeze(-1)

out = path_integral(f, 0.0, math.pi, method="gk21", atol=1e-10, rtol=1e-10)
print(out.integral)        # tensor([2.0000])
print(out.n_iterations)    # 1   — sin is smooth, one Kronrod application converges
```

A self-contained tour of the API is in [`examples/quickstart.py`](examples/quickstart.py).

## Methods

| Family | Names | What it is | When to use |
|---|---|---|---|
| Adaptive Gauss–Kronrod | `gk15`, `gk21`, `gk31` | Embedded Kronrod (primary) + Gauss (error estimate); refines intervals where error > tol | Default — unknown smoothness, want guaranteed error control |
| Non-adaptive Gauss–Legendre | `gl<n>` for any positive `n` | Single `n`-point rule, no refinement | Smooth integrand of known regularity, or as a baseline |

`gk21` is the default; it's QUADPACK's go-to and a good balance of cost
vs. resolution. `gk31` uses more nodes per interval but typically takes
fewer interval splits. `gl<n>` is for users who know their integrand is
smooth enough that a fixed rule converges.

## Public API

```python
from torchpathint import (
    path_integral,         # top-level dispatch
    adaptive_quadrature,   # explicit adaptive engine
    fixed_quadrature,      # explicit non-adaptive engine
    evaluate_chunked,      # the flat chunked evaluator
    IntegralOutput,        # return type
    get_method,            # rule lookup (Method dataclass)
)
```

`path_integral(f, t_init, t_final, *, method="gk21", atol=1e-5, rtol=1e-5,
t=None, max_batch=None, total_mem_usage=None, max_iter=50, device=None,
dtype=torch.float64)` returns an `IntegralOutput` with the integral, error
estimate, mesh, and per-interval diagnostics. See the function docstring
for full details.

### Integrand contract

`f: Tensor[N] → Tensor[N, D]`. The input is a flat 1-d tensor of time
points; the output's leading axis must match the input length, and the
trailing `D` axis is the integrand's dimensionality. For scalar
integrands use `D=1` (with `unsqueeze(-1)`). `f` takes no other
arguments — close over any extra state via a lambda or `functools.partial`.

### Bounds

`t_init` and `t_final` accept either a Python `float`/`int` or a 0-d
`torch.Tensor`. 1-d tensors are rejected.

### Memory control

The integrator builds a flat `[total_points]` tensor of all pending
evaluations and passes it to `f`. Use `max_batch` to chunk that tensor
into fixed-size pieces — chunks may span interval boundaries, so even a
single `gk31` interval can be evaluated across multiple `f` calls if
needed.

For automatic sizing, pass `total_mem_usage=<frac>` instead. `f` is
benchmarked at a few input sizes to estimate per-evaluation peak bytes,
and `max_batch` is chosen so chunks fit in `frac * free_GPU_memory`.

```python
out = path_integral(my_expensive_f, 0.0, 1.0, total_mem_usage=0.9)
```

CPU is treated as unbounded — `total_mem_usage` is ignored there. If
both `max_batch` and `total_mem_usage` are set, `max_batch` wins (the
probe is skipped).

### Warm start

`out.t_optimal` is the converged interval-barrier mesh. Pass
`out.t_optimal[1:-1]` (interior barriers) as the `t=` argument on a
subsequent integration of a similar integrand to skip refinement:

```python
out1 = path_integral(f, 0.0, 1.0, method="gk21", atol=1e-9, rtol=1e-9)
out2 = path_integral(f, 0.0, 1.0, method="gk21", atol=1e-9, rtol=1e-9,
                     t=out1.t_optimal[1:-1])
# out2.n_iterations == 1 if f hasn't changed much
```

## Differences from torchpathdiffeq

- **Quadrature, not ODE.** No Runge–Kutta. The integrand is `f(t)`, not
  `f(t, y)`.
- **No autograd inside the integrator.** The graph through `f(t)` is
  preserved by default, but the integrator is not designed around
  differentiability — wrap it with `torch.autograd.grad` outside if you
  need gradients.
- **Per-evaluation chunking.** `max_batch` slices the flat evaluation
  tensor regardless of interval boundaries. A high-order interval no
  longer has to fit in memory.
- **Stripped down.** No serial solver, no torchdiffeq dependency, no
  variable-sampling subclass, no `T` time-axis, no extra `ode_args`.

## Development

```bash
pip install -e ".[dev]"
pytest                  # 66 tests, ~0.2s on CPU
ruff check . && ruff format --check .
```

## License

CC-BY-4.0. See `LICENSE.md`.
