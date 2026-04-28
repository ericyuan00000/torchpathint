# API reference

All public symbols are re-exported at the package root:

```python
from torchpathint import (
    path_integral,
    adaptive_quadrature,
    fixed_quadrature,
    evaluate_chunked,
    estimate_max_batch,
    IntegralOutput,
    Method,
    get_method,
    METHOD_NAMES_ADAPTIVE,
    normalize_bound,
    resolve_device,
)
```

The full call signatures are in the docstrings; this page lists what each
function is for and the contracts you have to satisfy.

## Integrand contract

Every integrator takes a function

```
f: Tensor[N] -> Tensor[N, D]
```

- The input is a flat 1-d tensor of time points, on the integrator's device
  and dtype.
- The output's leading axis must match the input length.
- The trailing `D` axis is the output dimension. Use `D = 1` for scalar
  integrands (typically `unsqueeze(-1)`).
- `f` takes no other positional or keyword arguments. Close over any extra
  state via a `lambda`, `functools.partial`, or `nn.Module.__call__`.

## Bounds

`t_init` and `t_final` accept either a Python `float`/`int` or a 0-d
`torch.Tensor`. 1-d (or higher) tensors are rejected — the integrator's
time axis is scalar. A 0-d tensor with `requires_grad=True` is preserved
through `normalize_bound`, so the autograd graph can flow through the
bounds when downstream operations are differentiable.

There is no requirement that `t_init < t_final`; reversed bounds give the
expected sign-flipped integral.

## `path_integral`

Top-level dispatch. Call this unless you have a specific reason to bypass
it.

```python
out = path_integral(f, t_init, t_final, *,
                    method="gk21",
                    atol=1e-5, rtol=1e-5,
                    max_batch=None, memory_fraction=None,
                    max_iter=50,
                    device=None, dtype=torch.float64)
```

- `method` starting with `"gk"` dispatches to `adaptive_quadrature`.
- `method` starting with `"gl"` dispatches to `fixed_quadrature`. For `gl*`,
  the `atol`, `rtol`, and `max_iter` arguments are unused.
- `device=None` defaults to CUDA when available, else CPU.

Returns an [`IntegralOutput`](#integraloutput).

## `adaptive_quadrature`

The adaptive Gauss–Kronrod engine. Always use this for `gk*` methods.

- Splits intervals at the midpoint until per-interval error meets
  `atol + rtol · |I_running|` (RMS over output dim).
- `max_iter` defaults to 50. On the last iteration any still-over-tolerance
  intervals are force-accepted with a `UserWarning`.

## `fixed_quadrature`

A single Gauss–Legendre rule applied to the full domain. No subintervals,
no error estimate.

- Returns `integral_error=None` and `error_ratios=None`.
- `gl<n>` for any positive integer `n` is supported. Larger `n` is more
  accurate on smooth integrands but allocates more memory per call.

## `evaluate_chunked`

```python
y = evaluate_chunked(f, t, max_batch)
```

Helper used by both engines. Splits a 1-d `t` tensor into chunks of at
most `max_batch` and concatenates results. `max_batch=None` (or
`max_batch >= t.numel()`) calls `f` once with the whole tensor.

You probably don't need to call this directly — the engines apply it
themselves — but it is exported for cases where you want to reuse the
same chunking logic outside the integrators.

## `estimate_max_batch`

```python
max_batch = estimate_max_batch(f, t_sample, device, memory_fraction)
```

The probe behind `memory_fraction`. Returns `None` on CPU (no chunking),
a positive `int` on CUDA when a per-evaluation cost was measured, or
`None` on CUDA when nothing measurable was allocated. See
[memory.md](memory.md) for details.

## `IntegralOutput`

Dataclass returned by every integrator. Notable fields:

| Field | Shape | Meaning |
| --- | --- | --- |
| `integral` | `[D]` | Computed integral. |
| `method` | str | Rule name, e.g. `"gk21"`. |
| `t_init`, `t_final` | 0-d | Bounds, normalized to tensors. |
| `t` | `[N, K]` | Quadrature points actually evaluated, grouped by accepted interval. |
| `y` | `[N, K, D]` | Integrand values at `t`. |
| `h` | `[N]` | Width of each accepted interval. |
| `interval_integrals` | `[N, D]` | Per-interval primary contributions. Sum over `N` = `integral`. |
| `interval_errors` | `[N, D]` or None | Per-interval embedded-rule differences. |
| `integral_error` | `[D]` or None | Total error estimate. |
| `error_ratios` | `[N]` or None | Per-interval `ε_i / tol`. Useful diagnostic for "where did refinement land". |
| `n_iterations` | int | Adaptive refinement iterations performed. |
| `n_evaluations` | int | Total integrand evaluations. |

For `gl*` methods, `interval_errors`, `integral_error`, and
`error_ratios` are `None`, and `N = 1` (one trivial "interval").

## `Method` and `get_method`

```python
m = get_method("gk21", device=torch.device("cpu"), dtype=torch.float64)
m.nodes        # [K]
m.weights      # [K] primary (Kronrod for gk*, Gauss for gl*)
m.weights_error  # [K] = primary - embedded; None for gl*
m.is_adaptive  # True for gk*, False for gl*
m.degree       # exactness degree
```

Use this if you want to compute a quadrature by hand without the engine
machinery (e.g. for sanity-checking a custom integrator).

`METHOD_NAMES_ADAPTIVE` is a tuple of the supported `gk*` names.
