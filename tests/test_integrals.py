"""High-level integral accuracy and invariants.

These tests pin down the *behaviour* of the integrator on a small zoo of
analytically-known integrands: polynomials, sines, gaussians, exponentials,
and mixed-magnitude vector outputs. They complement ``test_quadrature.py``,
which exercises the engine internals (chunking, bound handling, max_iter),
by checking that the public ``path_integral`` actually computes the right
number across the supported method/dtype matrix.
"""

from __future__ import annotations

import math

import pytest
import torch

from torchpathint import METHOD_NAMES_ADAPTIVE, fixed_quadrature, path_integral

GK_METHODS = list(METHOD_NAMES_ADAPTIVE)
GL_METHODS = ["gl7", "gl15", "gl21", "gl31", "gl64"]


# ---------------------------------------------------------------------------
# Reference integrands and analytical antiderivatives
# ---------------------------------------------------------------------------


def _t(t: torch.Tensor) -> torch.Tensor:
    return t.unsqueeze(-1)


def _t_solution(a: float, b: float) -> float:
    return 0.5 * (b**2 - a**2)


def _t_squared(t: torch.Tensor) -> torch.Tensor:
    return (t**2).unsqueeze(-1)


def _t_squared_solution(a: float, b: float) -> float:
    return (b**3 - a**3) / 3.0


def _sin_squared(t: torch.Tensor) -> torch.Tensor:
    return (torch.sin(t) ** 2).unsqueeze(-1)


def _sin_squared_solution(a: float, b: float) -> float:
    # ∫ sin^2(t) dt = t/2 - sin(2t)/4
    def F(x: float) -> float:
        return x / 2.0 - math.sin(2.0 * x) / 4.0

    return F(b) - F(a)


def _exp_5t(t: torch.Tensor) -> torch.Tensor:
    return torch.exp(5.0 * t).unsqueeze(-1)


def _exp_5t_solution(a: float, b: float) -> float:
    return (math.exp(5.0 * b) - math.exp(5.0 * a)) / 5.0


def _gaussian_peak(t: torch.Tensor) -> torch.Tensor:
    return torch.exp(-1000.0 * (t - 0.5) ** 2).unsqueeze(-1)


def _gaussian_peak_solution(a: float, b: float) -> float:
    # closed-form via erf, only valid here for [0, 1]
    assert (a, b) == (0.0, 1.0)
    return math.sqrt(math.pi / 1000.0) * math.erf(math.sqrt(1000.0) * 0.5)


INTEGRANDS: dict[str, tuple[callable, callable, tuple[float, float]]] = {
    "t": (_t, _t_solution, (0.0, 1.0)),
    "t_squared": (_t_squared, _t_squared_solution, (0.0, 2.0)),
    "sin_squared": (_sin_squared, _sin_squared_solution, (0.0, math.pi)),
    "exp": (_exp_5t, _exp_5t_solution, (0.0, 1.0)),
    "gaussian_peak": (_gaussian_peak, _gaussian_peak_solution, (0.0, 1.0)),
}


# ---------------------------------------------------------------------------
# Adaptive: every method matches the analytical answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", GK_METHODS)
@pytest.mark.parametrize("integrand_name", list(INTEGRANDS.keys()))
def test_adaptive_matches_analytical(method, integrand_name, cpu_device):
    f, sol, (a, b) = INTEGRANDS[integrand_name]
    out = path_integral(
        f, a, b, method=method, atol=1e-10, rtol=1e-10, device=cpu_device
    )
    expected = sol(a, b)
    err = abs(out.integral.item() - expected)
    assert err < 1e-8, (
        f"{method}/{integrand_name}: |{out.integral.item()} - {expected}| = {err:.2e}"
    )


# ---------------------------------------------------------------------------
# Fixed Gauss-Legendre on smooth integrands
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", GL_METHODS)
@pytest.mark.parametrize("integrand_name", ["t", "t_squared", "sin_squared", "exp"])
def test_fixed_matches_analytical_smooth(method, integrand_name, cpu_device):
    f, sol, (a, b) = INTEGRANDS[integrand_name]
    out = fixed_quadrature(f, a, b, method=method, device=cpu_device)
    expected = sol(a, b)
    err = abs(out.integral.item() - expected)
    # Lower-order rules don't reach 1e-10 on exp; relax accordingly.
    tol = 1e-9 if method in ("gl21", "gl31", "gl64") else 1e-6
    assert err < tol, f"{method}/{integrand_name}: err={err:.2e} > {tol:.0e}"


# ---------------------------------------------------------------------------
# Polynomial exactness: a degree-d polynomial should hit the analytical
# answer in a single Kronrod application within ULP-level tolerance.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "degree"), [("gk15", 13), ("gk21", 19), ("gk31", 29)]
)
def test_adaptive_polynomial_exactness(method, degree, cpu_device):
    """For polynomials up to the *embedded Gauss* degree 2n-1, both the
    Kronrod and Gauss estimates are exact, so the error estimate is 0 and
    the loop converges in one iteration. (For degrees in (2n-1, 3n+1] the
    Kronrod estimate is still exact but the difference with Gauss is not 0,
    so the integrator may split despite getting the answer right.)
    """
    coeffs = torch.linspace(-1.0, 1.0, degree + 1, dtype=torch.float64)

    def f(t: torch.Tensor) -> torch.Tensor:
        # Σ c_k t^k via Horner
        out = torch.zeros_like(t)
        for c in reversed(coeffs.tolist()):
            out = out * t + c
        return out.unsqueeze(-1)

    a, b = 0.0, 1.0
    expected = sum(
        c.item() * (b ** (k + 1) - a ** (k + 1)) / (k + 1) for k, c in enumerate(coeffs)
    )
    out = path_integral(
        f, a, b, method=method, atol=1e-12, rtol=1e-12, device=cpu_device
    )
    assert out.n_iterations == 1
    assert abs(out.integral.item() - expected) < 1e-12


# ---------------------------------------------------------------------------
# Constants and degenerate intervals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", [*GK_METHODS, "gl15"])
def test_constant_integrand(method, cpu_device):
    out = path_integral(
        lambda t: torch.full_like(t, 3.5).unsqueeze(-1),
        0.0,
        2.0,
        method=method,
        device=cpu_device,
    )
    assert abs(out.integral.item() - 7.0) < 1e-12


@pytest.mark.parametrize("method", [*GK_METHODS, "gl15"])
def test_zero_width_interval(method, cpu_device):
    """t_init == t_final should give a zero integral, not blow up."""
    out = path_integral(
        lambda t: torch.sin(t).unsqueeze(-1),
        0.5,
        0.5,
        method=method,
        device=cpu_device,
    )
    assert out.integral.item() == 0.0


# ---------------------------------------------------------------------------
# Reverse bounds: ∫_b^a f = -∫_a^b f
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", [*GK_METHODS, "gl31"])
def test_reverse_bounds_negates_integral(method, cpu_device):
    def f(t):
        return torch.sin(t).unsqueeze(-1)

    fwd = path_integral(f, 0.0, math.pi, method=method, device=cpu_device)
    rev = path_integral(f, math.pi, 0.0, method=method, device=cpu_device)
    assert abs(fwd.integral.item() + rev.integral.item()) < 1e-10


def test_negative_interval_sin(cpu_device):
    """∫_{-π}^{0} sin(t) dt = -2."""
    out = path_integral(
        lambda t: torch.sin(t).unsqueeze(-1),
        -math.pi,
        0.0,
        method="gk21",
        atol=1e-10,
        rtol=1e-10,
        device=cpu_device,
    )
    assert abs(out.integral.item() - (-2.0)) < 1e-10


# ---------------------------------------------------------------------------
# Vector outputs: components of very different magnitudes must each converge
# ---------------------------------------------------------------------------


def test_vector_output_mixed_magnitudes(cpu_device):
    """A 1e6-magnitude component shouldn't drown out a 1e-3 one — RMS over D
    in the tolerance check protects the smaller component."""

    def f(t: torch.Tensor) -> torch.Tensor:
        return torch.stack([1e6 * torch.sin(t), 1e-3 * torch.cos(t)], dim=-1)

    out = path_integral(
        f,
        0.0,
        math.pi / 2,
        method="gk21",
        atol=1e-12,
        rtol=1e-10,
        device=cpu_device,
    )
    # ∫_0^{π/2} sin = 1, ∫_0^{π/2} cos = 1
    expected = torch.tensor([1e6, 1e-3], dtype=torch.float64, device=cpu_device)
    rel = (out.integral - expected).abs() / expected.abs()
    assert torch.all(rel < 1e-8)


def test_fixed_vector_output(cpu_device):
    """fixed_quadrature must handle vector integrands too."""

    def f(t: torch.Tensor) -> torch.Tensor:
        return torch.stack([torch.sin(t), torch.cos(t)], dim=-1)

    out = fixed_quadrature(f, 0.0, math.pi, method="gl31", device=cpu_device)
    expected = torch.tensor([2.0, 0.0], dtype=torch.float64, device=cpu_device)
    assert torch.allclose(out.integral, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Output invariants
# ---------------------------------------------------------------------------


def test_sum_intervals_equals_integral(cpu_device):
    """sum_intervals.sum(0) reconstructs integral bit-for-bit (modulo float
    summation order)."""
    out = path_integral(
        _gaussian_peak,
        0.0,
        1.0,
        method="gk21",
        atol=1e-9,
        rtol=1e-9,
        device=cpu_device,
    )
    diff = (out.sum_intervals.sum(dim=0) - out.integral).abs().max()
    assert diff < 1e-12


def test_h_sums_to_domain_width(cpu_device):
    """The interval widths sum to the full domain (no gaps, no overlaps)."""
    out = path_integral(
        _gaussian_peak,
        0.0,
        1.0,
        method="gk21",
        atol=1e-9,
        rtol=1e-9,
        device=cpu_device,
    )
    assert torch.all(out.h > 0)
    assert abs(out.h.sum().item() - 1.0) < 1e-12


def test_y_matches_reevaluation(cpu_device):
    """The stored y must match a fresh evaluation of f at the stored t."""
    out = path_integral(
        _sin_squared,
        0.0,
        math.pi,
        method="gk21",
        atol=1e-9,
        rtol=1e-9,
        device=cpu_device,
    )
    y_fresh = _sin_squared(out.t.reshape(-1)).reshape(out.y.shape)
    assert torch.allclose(out.y, y_fresh, atol=1e-15)


def test_n_evaluations_counts_correctly(cpu_device):
    """Total evaluations == sum over iterations of (pending intervals * K)."""
    out = path_integral(
        _gaussian_peak,
        0.0,
        1.0,
        method="gk15",
        atol=1e-9,
        rtol=1e-9,
        device=cpu_device,
    )
    # Final accepted intervals * K is a lower bound; total includes rejected
    # intervals re-evaluated as splits.
    K = 15
    assert out.n_evaluations >= out.sum_intervals.shape[0] * K
    assert out.n_evaluations % K == 0


def test_error_ratios_under_tolerance_after_convergence(cpu_device):
    """If the loop converged (no warning), every accepted interval has ratio < 1."""
    out = path_integral(
        _gaussian_peak,
        0.0,
        1.0,
        method="gk21",
        atol=1e-9,
        rtol=1e-9,
        device=cpu_device,
    )
    assert torch.all(out.error_ratios < 1.0)


# ---------------------------------------------------------------------------
# Dtype propagation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dt", [torch.float32, torch.float64])
def test_dtype_propagates_to_output(dt, cpu_device):
    out = path_integral(
        lambda t: torch.sin(t).unsqueeze(-1),
        0.0,
        math.pi,
        method="gk21",
        atol=1e-4 if dt == torch.float32 else 1e-10,
        rtol=1e-4 if dt == torch.float32 else 1e-10,
        device=cpu_device,
        dtype=dt,
    )
    assert out.integral.dtype == dt
    assert out.t_init.dtype == dt
    assert out.t_final.dtype == dt
    assert out.h.dtype == dt
