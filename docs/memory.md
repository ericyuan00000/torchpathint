# GPU memory and chunking

`adaptive_quadrature` and `fixed_quadrature` evaluate the integrand on a
single flat `Tensor[N]` of points each iteration. Whatever `f` allocates
internally — forward graphs, attention intermediates, vmap'd backward
state — has to fit on the device for that call to succeed. This page
explains how the integrator keeps that under control without making you
hand-tune anything in the common case.

## What gets allocated

Each iteration of `adaptive_quadrature` builds a flat `Tensor[n_pending · K]`
where `K = 2n + 1` is the rule's node count and `n_pending` is the number
of still-unconverged subintervals. `n_pending` starts at 1 and at most
doubles per iteration (each rejected interval splits in two), so the
input batch grows gradually rather than in one shot. The integrator's own
bookkeeping (running sums, mesh tensors) is small; the dominant cost is
whatever `f` allocates plus the `[n_pending · K, D]` output.

`fixed_quadrature` is a single iteration with `K` evaluations.

### Output-side memory: `full_output`

By default the integrators return only the integral plus cheap counters;
the per-interval `t`, `y`, `h`, and `interval_*` tensors are `None`. In
this default mode `adaptive_quadrature` doesn't accumulate accepted
intervals across iterations either — the running sum is `[D]` regardless
of mesh size. For high-D integrands on a finely refined mesh this saves
the largest output-side allocation (the `[N, K, D]` `y` tensor) entirely.

Pass `full_output=True` when you actually need the diagnostic fields
(`out.t`, `out.y`, `out.h`, `out.interval_integrals`,
`out.interval_errors`, `out.integral_error`, `out.error_ratios`) — for
example, when plotting the converged mesh or auditing per-interval error
distribution.

## The default: shrink on CUDA OOM

When `f(t)` raises a CUDA out-of-memory error, the integrator catches it,
drops partial state, calls `torch.cuda.empty_cache`, halves its current
chunk size, and retries the same call. The learned size persists across
adaptive iterations: a halving forced at iteration 5 means iterations
6, 7, … chunk at the smaller size from the start without re-discovering
the failing one.

This is the default — you don't have to ask for it. With no flags set,
the first `f(t)` call uses the full input, hits whatever ceiling your
integrand has on this GPU, and the integrator routes around it.

The OOM matcher accepts both the modern `torch.cuda.OutOfMemoryError`
and the older `RuntimeError("CUDA out of memory ...")` form some kernels
still raise. Other `RuntimeError`s (shape mismatches, etc.) propagate
unchanged.

If even `max_batch=1` OOMs, the chunker re-raises rather than spinning.
At that point the integrand needs less memory or a bigger GPU.

### Why not predict instead?

An earlier version of torchpathint ran a one-time probe of `f` at growing
input sizes and extrapolated peak memory linearly to pick a safe
`max_batch`. That works for forward-only integrands but breaks for
nested-autograd integrands like fairchem MLIPs (forces are computed via
`torch.autograd.grad` inside the forward, and an outer
`is_grads_batched=True` over that scales memory super-linearly in the
batch dimension). Probing under-predicts; the actual call OOMs.

The adaptive loop already provides a natural ramp: iteration 1 has
`n_pending = 1`, so the first `f()` call is just `K` nodes — small enough
to fit anywhere. Subsequent iterations grow into the GPU budget. Letting
the allocator be the oracle removes the prediction problem entirely.

## `max_batch` — optional initial cap

If you already know `f` OOMs above some size — for example, you've seen
the failure on this exact GPU before and want to skip the warm-up
halving — pass `max_batch` to start chunked from the beginning. OOM-shrink
still applies on top: a too-generous `max_batch` halves automatically.

```python
out = path_integral(f, 0.0, 1.0, method="gk21", max_batch=8)
```

`max_batch=None` (default) means "start unchunked."

## Diagnostics

The chunker logs a warning each time it shrinks:

```
evaluate_chunked: caught CUDA OOM at max_batch=None, retrying at 16.
```

Enable per-iteration debug logging with:

```python
import logging
logging.getLogger("torchpathint").setLevel(logging.DEBUG)
logging.basicConfig()
```
