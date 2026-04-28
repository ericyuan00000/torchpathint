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
