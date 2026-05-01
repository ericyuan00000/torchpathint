"""Tests for the public path_integral dispatch wrapper."""

from __future__ import annotations

import math

import pytest
import torch

from torchpathint import path_integral


def sin_integrand(t: torch.Tensor) -> torch.Tensor:
    return torch.sin(t).unsqueeze(-1)


def test_dispatches_to_adaptive_for_gk(cpu_device):
    out = path_integral(
        sin_integrand,
        0.0,
        math.pi,
        method="gk21",
        atol=1e-10,
        rtol=1e-10,
        device=cpu_device,
        full_output=True,
    )
    assert out.method == "gk21"
    assert out.integral_error is not None
    assert abs(out.integral.item() - 2.0) < 1e-10


def test_dispatches_to_fixed_for_gl(cpu_device):
    out = path_integral(sin_integrand, 0.0, math.pi, method="gl15", device=cpu_device)
    assert out.method == "gl15"
    assert out.integral_error is None
    assert abs(out.integral.item() - 2.0) < 1e-10


def test_unknown_method_raises(cpu_device):
    with pytest.raises(ValueError, match="Unknown method"):
        path_integral(sin_integrand, 0.0, 1.0, method="dopri5", device=cpu_device)


def test_uppercase_method_accepted(cpu_device):
    out = path_integral(sin_integrand, 0.0, math.pi, method="GK21", device=cpu_device)
    assert abs(out.integral.item() - 2.0) < 1e-9


def test_default_method_is_gk21(cpu_device):
    out = path_integral(sin_integrand, 0.0, math.pi, device=cpu_device)
    assert out.method == "gk21"


def test_max_batch_passes_through(cpu_device):
    a = path_integral(
        sin_integrand, 0.0, math.pi, method="gk31", device=cpu_device, max_batch=None
    )
    b = path_integral(
        sin_integrand, 0.0, math.pi, method="gk31", device=cpu_device, max_batch=3
    )
    assert a.integral.item() == b.integral.item()


# ---------------------------------------------------------------------------
# full_output flag: default is minimal, True populates diagnostics
# ---------------------------------------------------------------------------


def test_default_returns_minimal_output_for_adaptive(cpu_device):
    """Default (full_output=False) populates only integral plus cheap metadata."""
    out = path_integral(sin_integrand, 0.0, math.pi, method="gk21", device=cpu_device)
    assert abs(out.integral.item() - 2.0) < 1e-5
    assert out.method == "gk21"
    assert out.n_iterations >= 1
    assert out.n_evaluations > 0
    assert out.t is None
    assert out.y is None
    assert out.h is None
    assert out.interval_integrals is None
    assert out.interval_errors is None
    assert out.integral_error is None
    assert out.error_ratios is None


def test_default_returns_minimal_output_for_fixed(cpu_device):
    """fixed_quadrature default also yields minimal output."""
    out = path_integral(sin_integrand, 0.0, math.pi, method="gl21", device=cpu_device)
    assert abs(out.integral.item() - 2.0) < 1e-10
    assert out.method == "gl21"
    assert out.n_evaluations == 21
    assert out.t is None
    assert out.y is None
    assert out.h is None
    assert out.interval_integrals is None


def test_full_output_populates_diagnostics_for_adaptive(cpu_device):
    """full_output=True populates the per-interval diagnostic fields."""
    out = path_integral(
        sin_integrand,
        0.0,
        math.pi,
        method="gk21",
        device=cpu_device,
        full_output=True,
    )
    assert out.t is not None
    assert out.y is not None
    assert out.h is not None
    assert out.interval_integrals is not None
    assert out.interval_errors is not None
    assert out.integral_error is not None
    assert out.error_ratios is not None
    # Sanity: the diagnostics must be consistent with the integral.
    assert (out.interval_integrals.sum(dim=0) - out.integral).abs().max() < 1e-12


def test_full_output_matches_default_integral(cpu_device):
    """The integral value must be identical between minimal and full modes."""
    a = path_integral(
        sin_integrand,
        0.0,
        math.pi,
        method="gk21",
        atol=1e-10,
        rtol=1e-10,
        device=cpu_device,
    )
    b = path_integral(
        sin_integrand,
        0.0,
        math.pi,
        method="gk21",
        atol=1e-10,
        rtol=1e-10,
        device=cpu_device,
        full_output=True,
    )
    assert a.integral.item() == b.integral.item()
    assert a.n_iterations == b.n_iterations
    assert a.n_evaluations == b.n_evaluations
