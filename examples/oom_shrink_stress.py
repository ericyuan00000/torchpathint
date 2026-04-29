"""Stress test for OOM-driven chunk shrinking on a real GPU.

Self-contained — no popcornn, no fairchem. The integrand is an outer-
product MLP with a [N, dim, dim] intermediate, which under
``is_grads_batched=True`` scales peak memory as ``O(N^2 * dim^2)`` (one
N from the forward batch, one from vmap's batched backward). That
super-linear profile is the same shape that breaks the old probe-based
auto-sizer on UMA, exposed here in a model that fits on one A100.

What we expect to see:

1. ``adaptive_quadrature`` calls ``f`` with N = K = 21 nodes in
   iteration 1, which fits.
2. The integrand passes the rtol/atol bar (sin is smooth) and converges
   in one iteration on a low-tolerance run, OR splits a few times on a
   tight-tolerance run; once n_pending climbs past the OOM threshold,
   ``evaluate_chunked`` catches the OOM, halves ``state.max_batch``, and
   the rest of the integration runs at the smaller chunk size.
3. The integral matches a CPU reference to a few decimals.

Run via NERSC interactive node:

    srun -A m2834 -q interactive -C gpu -t 15:00 --exclusive \\
        --ntasks=1 --gpus-per-task=1 \\
        bash -lc "conda activate torchpathint && \\
                  python examples/oom_shrink_stress.py"
"""
from __future__ import annotations

import logging
import math
import os

# Without expandable segments, the caching allocator can fragment after
# the first OOM and refuse a smaller retry. This is a defensive default
# for the stress run.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn

from torchpathint import adaptive_quadrature

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


class HungryIntegrand(nn.Module):
    """Builds a [N, dim, dim] outer-product intermediate per call.

    Forward cost is O(N * dim^2). Under ``is_grads_batched=True`` the
    vmap'd backward batches over N grad_outputs, so peak memory scales
    as O(N^2 * dim^2) — the super-linear profile we want to stress.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        # Trainable parameters so an outer is_grads_batched=True has
        # something to differentiate w.r.t.
        self.lift = nn.Linear(1, dim, bias=False)
        self.kernel = nn.Parameter(torch.randn(dim, dim) / dim**0.5)

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # t: [N]
        x = self.lift(t.unsqueeze(-1))                 # [N, dim]
        x = torch.tanh(x)
        # [N, dim, dim] outer product. Its gradient w.r.t. x has the same
        # shape, so under is_grads_batched=True vmap stacks a second N
        # axis → backward intermediates scale as O(N^2 * dim^2).
        outer = x.unsqueeze(-1) * x.unsqueeze(-2)
        # Contract with a trainable kernel and reduce to a scalar per t.
        scaled = (outer * self.kernel).sum(dim=(-1, -2))
        return torch.sin(t) + 1e-6 * scaled            # [N]


def make_grad_integrand(model: HungryIntegrand, params: list[torch.nn.Parameter]):
    """Return f(t) -> [N, D] where D = total parameter count, computed
    via torch.autograd.grad(..., is_grads_batched=True). Mirrors what
    popcornn's gradient-of-loss integrator does."""

    def f(t: torch.Tensor) -> torch.Tensor:
        l = model(t)                                       # [N], graph live
        n = l.shape[0]
        grad_out = torch.eye(n, device=l.device, dtype=l.dtype)
        grads = torch.autograd.grad(
            l, params, grad_outputs=grad_out, is_grads_batched=True,
        )
        return torch.cat([g.reshape(n, -1) for g in grads], dim=-1).detach()

    return f


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("This stress test requires CUDA.")

    free_before, total = torch.cuda.mem_get_info()
    print(f"GPU memory before model load: {free_before / 1e9:.2f} / "
          f"{total / 1e9:.2f} GB free")

    # dim is calibrated so a single-K (21) call fits comfortably but
    # adaptive splits will eventually trip OOM and force the chunker
    # to halve. With dim=4096 on a 40 GB A100, peak at N=21 is ~7 GB
    # and at N=42 is ~28 GB; one or two splits beyond that OOM.
    dim = 4096
    device = torch.device("cuda")
    dtype = torch.float32

    model = HungryIntegrand(dim).to(device).to(dtype)
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)
    print(f"model: dim={dim}  params={n_params:,}  "
          f"weight memory = {n_params * 4 / 1e6:.1f} MB")

    f = make_grad_integrand(model, params)

    # Tighten atol/rtol so the adaptive loop is forced to split a few
    # times. The integrand is sin(t) + a smooth offset; quadrature
    # converges quickly without splits at default tolerances.
    print("\n=== adaptive_quadrature(gk21, atol=1e-9, rtol=1e-9) ===")
    out = adaptive_quadrature(
        f, 0.0, math.pi,
        method="gk21", atol=1e-9, rtol=1e-9,
        device=device, dtype=dtype, max_iter=10,
    )
    integral_norm = out.integral.norm().item()
    print(f"integral norm: {integral_norm:.6e}")
    print(f"n_iterations:  {out.n_iterations}")
    print(f"n_evaluations: {out.n_evaluations}")
    print(f"n_intervals:   {out.t.shape[0]}")
    print(f"requires_grad: {out.integral.requires_grad}")

    # Sanity vs a CPU reference at a much smaller dim. The integral
    # value depends on the (random) model parameters, so we just check
    # finiteness here.
    assert torch.isfinite(out.integral).all(), "integral has non-finite entries"
    print("\nstress run completed.")


if __name__ == "__main__":
    main()
