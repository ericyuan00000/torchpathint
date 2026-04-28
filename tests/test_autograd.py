"""Autograd flow through the integrator.

The integrator is *not* required to be autograd-transparent — gradient
support is not part of the API contract — but the current implementation
preserves the graph as a side effect (it uses ``torch.cat`` / ``torch.einsum``
on ``f``'s outputs without ``.detach()``). These tests pin that behaviour
down so the side effect is monitored if it ever changes.
"""

from __future__ import annotations

import math

import torch

from torchpathint import fixed_quadrature, path_integral


def test_grad_through_integrand_param_adaptive(cpu_device):
    """∂I/∂a where I = ∫_0^1 a t^2 dt = a/3. Expected gradient: 1/3."""
    a = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)

    def f(t: torch.Tensor) -> torch.Tensor:
        return (a * t**2).unsqueeze(-1)

    out = path_integral(
        f, 0.0, 1.0, method="gk21", atol=1e-10, rtol=1e-10, device=cpu_device
    )
    out.integral.sum().backward()
    assert abs(a.grad.item() - 1.0 / 3.0) < 1e-10


def test_grad_through_integrand_param_fixed(cpu_device):
    """Same gradient through the non-adaptive engine."""
    a = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)

    def f(t: torch.Tensor) -> torch.Tensor:
        return (a * t**2).unsqueeze(-1)

    out = fixed_quadrature(f, 0.0, 1.0, method="gl15", device=cpu_device)
    out.integral.sum().backward()
    assert abs(a.grad.item() - 1.0 / 3.0) < 1e-10


def test_grad_through_t_final_adaptive(cpu_device):
    """∂/∂b ∫_0^b sin(t) dt = sin(b). At b = π/2 the gradient should be 1."""
    b = torch.tensor(math.pi / 2.0, dtype=torch.float64, requires_grad=True)
    out = path_integral(
        lambda t: torch.sin(t).unsqueeze(-1),
        0.0,
        b,
        method="gk21",
        atol=1e-10,
        rtol=1e-10,
        device=cpu_device,
    )
    out.integral.sum().backward()
    assert abs(b.grad.item() - math.sin(math.pi / 2.0)) < 1e-9


def test_grad_through_both_bounds(cpu_device):
    """∂/∂a ∫_a^b sin(t) dt = -sin(a); ∂/∂b = sin(b)."""
    a = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
    out = path_integral(
        lambda t: torch.sin(t).unsqueeze(-1),
        a,
        b,
        method="gk21",
        atol=1e-10,
        rtol=1e-10,
        device=cpu_device,
    )
    out.integral.sum().backward()
    assert abs(a.grad.item() - (-math.sin(0.3))) < 1e-9
    assert abs(b.grad.item() - math.sin(1.7)) < 1e-9


def test_grad_through_nn_module_param(cpu_device):
    """Sanity check: nn.Module parameters receive a non-None gradient."""
    layer = torch.nn.Linear(1, 1, bias=False)
    layer.to(dtype=torch.float64)
    with torch.no_grad():
        layer.weight.fill_(2.0)

    def f(t: torch.Tensor) -> torch.Tensor:
        return layer(t.unsqueeze(-1))  # [N, 1]

    out = path_integral(
        f, 0.0, 1.0, method="gk21", atol=1e-10, rtol=1e-10, device=cpu_device
    )
    out.integral.sum().backward()
    # ∫_0^1 w t dt = w/2 → ∂I/∂w = 1/2
    assert layer.weight.grad is not None
    assert abs(layer.weight.grad.item() - 0.5) < 1e-10


def test_no_grad_context_disables_graph(cpu_device):
    """Inside torch.no_grad() the output should not require grad even when
    the inputs do."""
    a = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)

    def f(t: torch.Tensor) -> torch.Tensor:
        return (a * t**2).unsqueeze(-1)

    with torch.no_grad():
        out = path_integral(
            f, 0.0, 1.0, method="gk21", atol=1e-9, rtol=1e-9, device=cpu_device
        )
    assert not out.integral.requires_grad
