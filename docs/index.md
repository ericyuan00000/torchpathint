# torchpathint documentation

`torchpathint` is a PyTorch-native library for definite path integrals

```
I = ∫_{t_init}^{t_final} f(t) dt
```

of vector-valued integrands `f: ℝ → ℝᴰ`. It uses adaptive Gauss–Kronrod and
fixed Gauss–Legendre quadrature, evaluates every node in every pending
subinterval in a single batched call to `f`, and chunks evaluations across
multiple `f` calls to bound peak GPU memory.

## Contents

- [Theory](theory.md) — what Gauss–Kronrod is, what the embedded error
  estimate measures, why the integrator splits at the midpoint, and how the
  per-evaluation chunking interacts with subinterval boundaries.
- [API reference](api.md) — public functions, classes, and their contracts.
- [Memory probe](memory.md) — how `total_mem_usage` is measured and applied,
  and when to override it.
- [Migrating from torchpathdiffeq](migration.md) — what changed, what was
  cut, and how to rewrite calls.

A self-contained tour of the public API is in
[`examples/quickstart.py`](../examples/quickstart.py); the README has the
30-second pitch.
