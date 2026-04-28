"""Tests for the base utilities: bound normalization, device resolution."""

from __future__ import annotations

import pytest
import torch

from torchpathint import IntegralOutput, normalize_bound, resolve_device


def test_normalize_bound_float(cpu_device):
    out = normalize_bound(1.5, cpu_device, torch.float64, "t_init")
    assert out.dim() == 0
    assert out.dtype == torch.float64
    assert out.item() == 1.5


def test_normalize_bound_int(cpu_device):
    out = normalize_bound(2, cpu_device, torch.float64, "t_final")
    assert out.dim() == 0
    assert out.item() == 2.0


def test_normalize_bound_zero_d_tensor(cpu_device):
    out = normalize_bound(torch.tensor(3.14), cpu_device, torch.float64, "t")
    assert out.dim() == 0
    assert out.dtype == torch.float64
    assert pytest.approx(out.item()) == 3.14


def test_normalize_bound_preserves_requires_grad(cpu_device):
    src = torch.tensor(2.5, requires_grad=True)
    out = normalize_bound(src, cpu_device, torch.float64, "t_init")
    assert out.requires_grad


def test_normalize_bound_rejects_one_d(cpu_device):
    with pytest.raises(ValueError, match="must be a scalar"):
        normalize_bound(torch.tensor([0.0]), cpu_device, torch.float64, "t_init")


def test_normalize_bound_rejects_two_d(cpu_device):
    with pytest.raises(ValueError, match="must be a scalar"):
        normalize_bound(torch.zeros(2, 2), cpu_device, torch.float64, "t_init")


def test_normalize_bound_rejects_string(cpu_device):
    with pytest.raises(TypeError, match="must be a float"):
        normalize_bound("0.0", cpu_device, torch.float64, "t_init")


def test_resolve_device_default():
    expected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert resolve_device(None) == expected


def test_resolve_device_explicit():
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(torch.device("cpu")) == torch.device("cpu")


def test_integral_output_minimum_required_fields():
    out = IntegralOutput(
        integral=torch.zeros(2),
        method="gk21",
        t_init=torch.tensor(0.0),
        t_final=torch.tensor(1.0),
        t=torch.zeros(0, 21),
        y=torch.zeros(0, 21, 2),
        h=torch.zeros(0),
        sum_intervals=torch.zeros(0, 2),
    )
    assert out.method == "gk21"
    assert out.n_iterations == 0
    assert out.integral_error is None
