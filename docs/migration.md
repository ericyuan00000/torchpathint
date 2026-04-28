# Migrating from torchpathdiffeq

`torchpathint` is a clean rebuild of [`torchpathdiffeq`](https://github.com/khegazy/torchpathdiffeq)
with a deliberately narrower scope. If you have existing code calling the
old library, this page is the diff.

## What stayed the same

- `IntegralOutput` is still a dataclass with `integral`, `t`, `y`, `h`,
  `interval_integrals`, `n_iterations`, `n_evaluations` fields. The semantics
  of each field are unchanged where they exist.
- The runtime is still pure PyTorch + numpy, no extra dependencies.
- The integrator still evaluates one batch at a time and stores per-step
  diagnostics on the output object.

## What was removed

| Removed | Why |
| --- | --- |
| `runge_kutta.py`, all RK tableau machinery | Quadrature replaces ODE solvers — see [theory.md](theory.md). |
| `serial_solver.py` and the `_VARIABLE_*` subclasses | Single parallel engine; the variable-sampling subclasses were a workaround for RK that quadrature doesn't need. |
| `torchdiffeq` dependency | Not used by quadrature. |
| `gradient_taken` field on `IntegralOutput` | The integrator no longer manages autograd; it preserves the graph as a side effect, but the field would imply a guarantee that isn't there. |
| `T` time-axis | Integrand is `f(t)` with scalar `t`. No `[N, T]` shape gymnastics. |
| `ode_args` | `f` takes no extra arguments — close over state via `lambda` or `nn.Module.__call__`. |
| `DistributedEnvironment` | Multi-GPU is deferred. |

## What changed

### Integrand signature

Old:

```python
def f(t, y=None, *args):
    while len(t.shape) < 2:
        t = t.unsqueeze(0)
    return scale * t**2  # shape [N, T, D]
```

New:

```python
def f(t):  # t: Tensor[N]
    return scale * (t**2).unsqueeze(-1)  # Tensor[N, D]
```

No `y`, no `args`, no time-axis insertion, no `T`-dim. `D = 1` is fine
but you must `unsqueeze(-1)` it explicitly.

### Method names

The RK method names (`bosh3`, `dopri5`, `adaptive_heun`, etc.) are gone.
The replacements are:

| Use case | New name |
| --- | --- |
| Default adaptive | `gk21` (analogous to `dopri5` as a sensible default) |
| Lighter adaptive | `gk15` |
| Heavier adaptive | `gk31` |
| Non-adaptive (any order) | `gl<n>`, e.g. `gl15`, `gl31`, `gl64` |

### Solver construction

Old:

```python
solver = get_parallel_RK_solver(
    sampling_type=steps.ADAPTIVE_UNIFORM,
    method="dopri5", atol=1e-8, rtol=1e-8, remove_cut=0.1,
)
out = solver.integrate(f, t_init=T_INIT, t_final=T_FINAL)
```

New:

```python
out = path_integral(
    f, T_INIT, T_FINAL, method="gk21", atol=1e-8, rtol=1e-8,
)
```

There is no persistent solver object. The integrator is functional —
construct the rule, run, return. Mesh state is not reused across calls;
each call starts from a single subinterval and adapts from scratch.

### Memory control

`remove_cut` (which pruned over-resolved subintervals between iterations)
is gone. `max_batch` and `memory_fraction` replace what the old library
called fixed-batch evaluation: the new chunking is per-evaluation, not
per-interval, so a single high-order interval can be spread across
multiple `f` calls. See [memory.md](memory.md).

### Bounds

Both libraries accept Python scalars or 0-d tensors. The old library was
permissive about 1-d shape-1 tensors; the new one rejects them. Replace
`torch.tensor([0.0])` with `torch.tensor(0.0)` or just `0.0`.

## Things to check after porting

1. The trailing `D` axis on every `f` output. The old library tolerated
   `[N]` outputs as scalar integrands; the new one raises a `ValueError`.
2. Callers passing 1-d shape-1 tensors as bounds. Same fix as above.
3. Code that branched on `IntegralOutput.gradient_taken`. That field is
   gone; if you cared about gradient flow, set up `torch.autograd.grad`
   around the call.
4. Reuse of solver objects or warm-started meshes. Replace with bare
   functional calls; each call adapts from scratch.
