# Theory

This page summarises the numerical methods behind `torchpathint`. None of
this is novel — it is a quick reference so you can read the code without
flipping back to QUADPACK or Stoer–Bulirsch.

## Gauss–Legendre quadrature

The `n`-point Gauss–Legendre rule on `[-1, 1]` chooses nodes `x_1, …, x_n`
and weights `w_1, …, w_n` so that

```
∫_{-1}^{1} p(x) dx = Σ_i w_i p(x_i)
```

holds exactly for every polynomial `p` of degree `2n − 1` or lower. The
nodes are the roots of the degree-`n` Legendre polynomial. To integrate
over a general interval `[a, b]`, change variables `x = (b − a)/2 · u + (b + a)/2`:

```
∫_a^b f(t) dt = (b − a)/2 · Σ_i w_i f((b − a)/2 · x_i + (b + a)/2)
```

`torchpathint` exposes Gauss–Legendre as `gl<n>` for any positive integer
`n`. Nodes and weights are computed by `numpy.polynomial.legendre.leggauss`
and cached.

## Gauss–Kronrod quadrature

A Kronrod extension of an `n`-point Gauss rule adds `n + 1` further nodes
to the original `n` so that the combined `2n + 1` nodes form a rule of
exactness `3n + 1`. Picking the new nodes optimally gives one rule whose
weights you can compute *and* an embedded `n`-point Gauss rule that uses
the same nodes (with different weights). The difference between the two
estimates is a cheap, conservative error estimate for the primary rule.

`torchpathint` uses Patterson's tables for `n = 7, 10, 15` (the `gk15`,
`gk21`, and `gk31` rules). The tables are transcribed from QUADPACK and
validated at import time against polynomial exactness so a mistyped digit
fails loudly.

For each subinterval the integrator computes

- primary estimate: `I_K = h/2 · Σ_i w_i^K · f(t_i)`
- embedded estimate: `I_G = h/2 · Σ_i w_i^G · f(t_i)`
- error estimate: `|I_K − I_G| = h/2 · |Σ_i (w_i^K − w_i^G) · f(t_i)|`

where `h` is the subinterval width. Storing `w^K − w^G` directly (the
`weights_error` field on `Method`) keeps both estimates aligned over the
same `2n + 1` evaluations.

## Adaptive refinement

`adaptive_quadrature` starts with a single subinterval `[t_init, t_final]`,
evaluates the chosen GK rule on every pending subinterval, and judges each
one by the ratio

```
ε_i = h_i/2 · |Σ_k (w_k^K − w_k^G) · f(t_{i,k})|
ratio_i = RMS_d ε_{i,d} / (atol + rtol · |I_running_d|)
```

where the RMS is taken over the output dimension `d` so a single noisy
component cannot dominate the splitting decision, and `I_running_d` is the
running primary estimate of the *full* integral, including the interval
under test, used to scale the relative tolerance. Subintervals with
`ratio_i < 1` are accepted and removed from the pending set; the rest are
split at the midpoint and re-evaluated on the next iteration.

If the loop hits `max_iter` while some intervals are still over tolerance
the integrator emits a `UserWarning`, force-accepts the remaining intervals
to return a partial estimate, and reports the partial state via
`IntegralOutput.error_ratios` and `n_iterations`.

## Per-evaluation chunking

A single subinterval has `K = 2n + 1` evaluations (`gk15` → 15, `gk21` → 21,
`gk31` → 31). With `n_pending` subintervals, the integrator builds a flat
`[n_pending · K]` tensor of evaluation points and asks `f` to evaluate
all of them in one call. When `max_batch` is set it slices the flat tensor
into chunks of at most `max_batch` points each, regardless of subinterval
boundaries, calls `f` once per chunk, and concatenates the results.

This decouples the rule's order from peak GPU memory: a single high-order
subinterval no longer has to fit. The per-evaluation chunking is the
property the old `torchpathdiffeq` couldn't offer — it always evaluated
every node of an interval together.

When `total_mem_usage` is set instead of `max_batch`, the integrator probes
`f` at four growing input sizes (8, 64, 512, 4096), measures peak per-call
allocation, and divides the budget (`total_mem_usage · free_GPU_memory`)
by per-evaluation cost to pick a `max_batch`. See [memory.md](memory.md)
for the gory details.

## What this library is not

- A general ODE solver. `torchpathint` only does definite integrals. If
  your integrand depends on the running solution `y(t)` (a true IVP), you
  want `torchdiffeq`.
- A Monte Carlo integrator. Convergence is via deterministic quadrature.
  High-dimensional `t` is out of scope; the input is always a 1-d sweep.
- A gradient-friendly integrator. The autograd graph through `f(t)` is
  preserved as a side effect of using `torch.cat`/`torch.einsum` on the
  raw outputs, but gradients with respect to `t_init`/`t_final` and
  through `f`'s parameters are not part of the API contract — wrap the
  call in `torch.autograd.grad` only if you've checked it works for your
  case.
