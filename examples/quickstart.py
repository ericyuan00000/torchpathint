"""Quickstart example for torchpathint.

Run with::

    python examples/quickstart.py
"""

from __future__ import annotations

import math

import torch

from torchpathint import path_integral


def example_smooth() -> None:
    """The default call: only the integral comes back."""
    print("\n[smooth] integral of sin(t) from 0 to pi (exact = 2)")
    out = path_integral(
        lambda t: torch.sin(t).unsqueeze(-1),
        0.0,
        math.pi,
        method="gk21",
        atol=1e-10,
        rtol=1e-10,
    )
    print(f"  result = {out.integral.item():.16f}")
    print(f"  evals  = {out.n_evaluations}")


def example_sharp_peak() -> None:
    """A localised peak forces the adaptive loop to refine.

    Pass ``full_output=True`` to see where the loop landed: the per-interval
    mesh, integrand evaluations, and error estimates are returned alongside
    the integral. By default these diagnostic fields are ``None`` to save
    memory.
    """
    print("\n[sharp peak] integral of exp(-1000*(t-0.5)^2) from 0 to 1")
    out = path_integral(
        lambda t: torch.exp(-1000.0 * (t - 0.5) ** 2).unsqueeze(-1),
        0.0,
        1.0,
        method="gk21",
        atol=1e-10,
        rtol=1e-10,
        full_output=True,
    )
    exact = math.sqrt(math.pi / 1000.0) * math.erf(math.sqrt(1000) * 0.5)
    print(f"  result   = {out.integral.item():.12e}")
    print(f"  exact    = {exact:.12e}")
    print(f"  err      = {abs(out.integral.item() - exact):.2e}")
    print(
        f"  iters    = {out.n_iterations}, intervals = {out.interval_integrals.shape[0]}"
    )


def example_vector_integrand() -> None:
    """Integrate a vector-valued integrand to get a vector integral."""
    print("\n[vector] integral of [sin(t), cos(t), sin(2t)] from 0 to pi")

    def f(t: torch.Tensor) -> torch.Tensor:
        return torch.stack([torch.sin(t), torch.cos(t), torch.sin(2 * t)], dim=-1)

    out = path_integral(f, 0.0, math.pi, method="gk21", atol=1e-10, rtol=1e-10)
    print(f"  result   = {out.integral.tolist()}")
    print("  expected = [2, 0, 0]")


def example_memory_chunking() -> None:
    """Bound peak GPU memory with max_batch when the integrand is expensive."""
    print("\n[chunked] gk31 (K=31 nodes per interval) chunked at 5 evals per call")

    call_count = 0

    def f(t: torch.Tensor) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        return torch.sin(t).unsqueeze(-1)

    out = path_integral(
        f, 0.0, math.pi, method="gk31", atol=1e-10, rtol=1e-10, max_batch=5
    )
    print(f"  result    = {out.integral.item():.16f}")
    print(f"  total f calls (chunked) = {call_count}")
    print(f"  total evaluations       = {out.n_evaluations}")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    example_smooth()
    example_sharp_peak()
    example_vector_integrand()
    example_memory_chunking()
