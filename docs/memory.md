# Memory probe

`max_batch` and `total_mem_usage` are alternative ways to control peak GPU
memory inside the integrator. This page explains what each one does and
when to pick which.

## What gets allocated

Each iteration of `adaptive_quadrature` evaluates `K = 2n + 1` quadrature
nodes per pending subinterval. With `n_pending` subintervals it builds a
flat `Tensor[n_pending · K]` of evaluation points and calls `f` on it,
producing `Tensor[n_pending · K, D]`. The integrator's own bookkeeping is
small (a handful of `Tensor[n_pending, D]` running sums and per-iteration
mesh tensors); the dominant cost is whatever `f` allocates internally
plus the output tensor.

`fixed_quadrature` is a single iteration with `K` evaluations.

## `max_batch`

Sets a hard cap on how many points are sent to `f` per call. The flat
evaluation tensor is sliced into chunks of at most `max_batch` points,
each chunk is passed to `f`, and outputs are concatenated. Chunks span
subinterval boundaries — even a single high-order subinterval can be
spread across multiple `f` calls.

Use `max_batch` when you already know the answer (e.g. you've seen
`f` OOM at some size before).

## `total_mem_usage`

Asks the integrator to *measure* the right `max_batch` instead of you
guessing. The probe runs once per integrator call (before adaptive
iteration starts) and is implemented in
`torchpathint.memory.estimate_max_batch`:

1. `total_mem_usage ∈ (0, 1]` is interpreted as a fraction of *currently
   free* GPU memory, computed as
   `free + (reserved − allocated)` so the PyTorch allocator's cached blocks
   count as available. On a dedicated GPU this is roughly
   `total_mem_usage · total_memory`; on a shared GPU it scales with what is
   actually free right now.
2. `f` is invoked at four growing input sizes — 8, 64, 512, 4096 — with
   `torch.cuda.reset_peak_memory_stats` between calls. Doubling keeps the
   probe cheap (`O(log N)` measurements) while averaging out small-`N`
   startup noise.
3. The largest measured peak-bytes-per-evaluation is multiplied by a 2.0×
   safety factor (the integrator's own tensors aren't part of the probe)
   and divided into the budget to pick a positive `max_batch`.
4. If a probe run OOMs we stop early and use the largest size that fit.
5. If every probe size measured zero peak allocation (typical for trivial
   integrands like `sin(t)`), the probe returns `None`, falling through to
   "no chunking".

## Interaction

- If both `max_batch` and `total_mem_usage` are set, `max_batch` wins —
  the probe is skipped.
- On CPU, `total_mem_usage` is ignored. CPU memory accounting through the
  PyTorch allocator is unreliable and CPU OOMs are less catastrophic than
  GPU OOMs.
- The probe consumes one probe-sized allocation per probe size before the
  real integration starts; this is intentional and amortizes well over a
  long-running integration.
- The probe is a *one-time* calibration. If `f`'s allocation pattern
  drifts over the course of a long training loop, re-create the integrator
  call (or pass a tighter `max_batch`).

## When to override

The probe is conservative on purpose. Override it with `max_batch` when:

- Your integrand is allocation-stable but the probe under-estimates because
  small `N` measurements are dominated by allocator overhead. Pass a
  `max_batch` that is large enough that `f` is GPU-bound but small enough
  that the output tensor still fits.
- You are running on a shared GPU and want a hard cap that doesn't depend
  on the current free memory at probe time.
- You want bit-identical reproducibility: `max_batch` is deterministic;
  the probe uses live free-memory readings, which fluctuate.

## Diagnostics

`adaptive_quadrature` and `fixed_quadrature` log at `DEBUG` level when the
probe runs. Enable it with

```python
import logging
logging.getLogger("torchpathint").setLevel(logging.DEBUG)
logging.basicConfig()
```

The log line reports per-evaluation bytes, which probe size produced the
maximum, the safety factor, and the resulting `max_batch`.
