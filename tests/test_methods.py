"""Tests for quadrature rules: shape, symmetry, polynomial exactness."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchpathint import METHOD_NAMES_ADAPTIVE, get_method
from torchpathint.methods import _GK_TABLES


@pytest.mark.parametrize("name", METHOD_NAMES_ADAPTIVE)
def test_gk_shape_and_symmetry(name, cpu_device):
    m = get_method(name, cpu_device, torch.float64)
    K = m.nodes.numel()
    assert 2 * _GK_TABLES[name][3] + 1 == K
    assert m.weights.shape == (K,)
    assert m.weights_error.shape == (K,)
    assert m.is_adaptive is True
    # Nodes sorted ascending and symmetric about 0
    assert torch.all(m.nodes[1:] > m.nodes[:-1])
    assert torch.allclose(
        m.nodes + m.nodes.flip(0), torch.zeros_like(m.nodes), atol=1e-15
    )
    # Kronrod weights are symmetric and sum to 2
    assert torch.allclose(m.weights, m.weights.flip(0), atol=1e-15)
    assert pytest.approx(m.weights.sum().item(), abs=1e-14) == 2.0


@pytest.mark.parametrize("name", METHOD_NAMES_ADAPTIVE)
def test_gk_kronrod_polynomial_exactness(name, cpu_device):
    m = get_method(name, cpu_device, torch.float64)
    n_gauss = _GK_TABLES[name][3]
    deg = 3 * n_gauss + 1  # Kronrod degree of exactness
    nodes_np = m.nodes.numpy()
    w_kronrod = m.weights.numpy()
    for k in range(deg + 1):
        truth = 0.0 if k % 2 == 1 else 2.0 / (k + 1)
        approx = float(np.sum(nodes_np**k * w_kronrod))
        assert abs(approx - truth) < 1e-12, f"{name}: x^{k}: {approx} != {truth}"


@pytest.mark.parametrize("name", METHOD_NAMES_ADAPTIVE)
def test_gk_embedded_gauss_polynomial_exactness(name, cpu_device):
    m = get_method(name, cpu_device, torch.float64)
    n_gauss = _GK_TABLES[name][3]
    deg = 2 * n_gauss - 1  # Gauss degree of exactness
    nodes_np = m.nodes.numpy()
    # weights_error = primary - embedded => embedded = primary - error
    w_embedded = (m.weights - m.weights_error).numpy()
    for k in range(deg + 1):
        truth = 0.0 if k % 2 == 1 else 2.0 / (k + 1)
        approx = float(np.sum(nodes_np**k * w_embedded))
        assert abs(approx - truth) < 1e-12, f"{name}: x^{k}: {approx} != {truth}"


@pytest.mark.parametrize("n", [1, 2, 5, 7, 10, 15, 21, 31, 64])
def test_gl_shape_and_polynomial_exactness(n, cpu_device):
    m = get_method(f"gl{n}", cpu_device, torch.float64)
    assert m.nodes.numel() == n
    assert m.weights.numel() == n
    assert m.weights_error is None
    assert m.is_adaptive is False
    assert pytest.approx(m.weights.sum().item(), abs=1e-13) == 2.0
    deg = 2 * n - 1
    nodes_np = m.nodes.numpy()
    w = m.weights.numpy()
    for k in range(deg + 1):
        truth = 0.0 if k % 2 == 1 else 2.0 / (k + 1)
        approx = float(np.sum(nodes_np**k * w))
        assert abs(approx - truth) < 1e-12, f"gl{n}: x^{k}: {approx} != {truth}"


def test_method_dtype_and_device(cpu_device):
    for dt in [torch.float32, torch.float64]:
        m = get_method("gk21", cpu_device, dt)
        assert m.nodes.dtype == dt
        assert m.weights.dtype == dt
        assert m.weights_error.dtype == dt
        assert m.nodes.device == cpu_device


def test_get_method_unknown_raises(cpu_device):
    with pytest.raises(ValueError, match="Unknown method"):
        get_method("rk5", cpu_device, torch.float64)
    with pytest.raises(ValueError, match="Unknown method"):
        get_method("dopri5", cpu_device, torch.float64)


def test_get_method_invalid_gl_order(cpu_device):
    with pytest.raises(ValueError, match=">= 1"):
        get_method("gl0", cpu_device, torch.float64)


def test_gk_gl_consistency_g7_inside_gk15(cpu_device):
    """The 7-point Gauss embedded inside gk15 must produce the same integral
    as a stand-alone gl7 rule, bit-for-bit."""
    gk15 = get_method("gk15", cpu_device, torch.float64)
    gl7 = get_method("gl7", cpu_device, torch.float64)

    # Embedded Gauss = primary - error
    w_embedded = gk15.weights - gk15.weights_error

    # Pick a smooth test function: sin(x)
    y_gk = torch.sin(gk15.nodes)
    y_gl = torch.sin(gl7.nodes)
    integral_embedded = (w_embedded * y_gk).sum().item()
    integral_gl = (gl7.weights * y_gl).sum().item()
    assert abs(integral_embedded - integral_gl) < 1e-15
