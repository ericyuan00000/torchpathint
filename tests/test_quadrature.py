"""Tests for the adaptive and fixed quadrature engines and the chunked evaluator."""

from __future__ import annotations

import math

import pytest
import torch

from torchpathint import (
    METHOD_NAMES_ADAPTIVE,
    adaptive_quadrature,
    evaluate_chunked,
    fixed_quadrature,
)


def sin_integrand(t: torch.Tensor) -> torch.Tensor:
    return torch.sin(t).unsqueeze(-1)


def cos_integrand(t: torch.Tensor) -> torch.Tensor:
    return torch.cos(t).unsqueeze(-1)


def vector_integrand(t: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.sin(t), torch.cos(t), torch.sin(2 * t)], dim=-1)


def gaussian_peak(t: torch.Tensor) -> torch.Tensor:
    """exp(-1000*(t-0.5)^2). Sharp peak that needs adaptive refinement."""
    return torch.exp(-1000.0 * (t - 0.5) ** 2).unsqueeze(-1)


# --- evaluate_chunked --------------------------------------------------------


def test_evaluate_chunked_no_chunking_matches_direct(cpu_device):
    from torchpathint.quadrature import _BatchState

    t = torch.linspace(0.0, 1.0, 50, dtype=torch.float64)
    out = evaluate_chunked(sin_integrand, t, _BatchState(None))
    assert torch.allclose(out, sin_integrand(t))


def test_evaluate_chunked_small_batch_matches_direct(cpu_device):
    from torchpathint.quadrature import _BatchState

    t = torch.linspace(0.0, 1.0, 50, dtype=torch.float64)
    direct = sin_integrand(t)
    for batch in [1, 3, 7, 49, 50, 100]:
        out = evaluate_chunked(sin_integrand, t, _BatchState(batch))
        assert out.shape == direct.shape
        assert torch.allclose(out, direct)


def test_evaluate_chunked_empty_raises(cpu_device):
    from torchpathint.quadrature import _BatchState

    with pytest.raises(ValueError, match="empty tensor"):
        evaluate_chunked(sin_integrand, torch.zeros(0), _BatchState(None))


# --- adaptive_quadrature: accuracy on smooth integrand -----------------------


@pytest.mark.parametrize("method", METHOD_NAMES_ADAPTIVE)
def test_adaptive_sin_pi_smooth_one_iteration(method, cpu_device):
    """sin is smooth enough that one Kronrod application converges."""
    out = adaptive_quadrature(
        sin_integrand,
        0.0,
        math.pi,
        method=method,
        atol=1e-10,
        rtol=1e-10,
        device=cpu_device,
    )
    assert abs(out.integral.item() - 2.0) < 1e-10
    assert out.n_iterations == 1
    assert out.interval_integrals.shape[0] == 1


@pytest.mark.parametrize("method", METHOD_NAMES_ADAPTIVE)
def test_adaptive_cos_zero_integral(method, cpu_device):
    out = adaptive_quadrature(
        cos_integrand,
        0.0,
        2 * math.pi,
        method=method,
        atol=1e-10,
        rtol=1e-10,
        device=cpu_device,
    )
    assert abs(out.integral.item()) < 1e-10


# --- adaptive_quadrature: refines on hard integrand --------------------------


def test_adaptive_gaussian_peak_refines(cpu_device):
    out = adaptive_quadrature(
        gaussian_peak,
        0.0,
        1.0,
        method="gk21",
        atol=1e-10,
        rtol=1e-10,
        device=cpu_device,
    )
    # Reference: ∫_0^1 exp(-1000(t-0.5)^2) dt = sqrt(pi/1000) * erf(sqrt(1000)/2)
    exact = math.sqrt(math.pi / 1000.0) * math.erf(math.sqrt(1000) * 0.5)
    assert abs(out.integral.item() - exact) < 1e-9
    assert out.n_iterations > 1
    assert out.interval_integrals.shape[0] > 1


# --- multi-dimensional integrand ---------------------------------------------


def test_adaptive_vector_integrand(cpu_device):
    out = adaptive_quadrature(
        vector_integrand,
        0.0,
        math.pi,
        method="gk21",
        atol=1e-10,
        rtol=1e-10,
        device=cpu_device,
    )
    expected = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64, device=cpu_device)
    assert torch.allclose(out.integral, expected, atol=1e-10)


# --- chunked evaluation correctness -----------------------------------------


def test_adaptive_chunked_vs_unchunked_identical(cpu_device):
    """Chunked and unchunked must produce bit-identical outputs."""
    common = {
        "method": "gk31",
        "atol": 1e-12,
        "rtol": 1e-12,
        "device": cpu_device,
    }
    a = adaptive_quadrature(sin_integrand, 0.0, math.pi, max_batch=None, **common)
    b = adaptive_quadrature(sin_integrand, 0.0, math.pi, max_batch=7, **common)
    c = adaptive_quadrature(sin_integrand, 0.0, math.pi, max_batch=1, **common)
    assert a.integral.item() == b.integral.item() == c.integral.item()


def test_adaptive_chunk_smaller_than_K(cpu_device):
    """A chunk smaller than the rule's K nodes must still produce correct results."""
    out = adaptive_quadrature(
        gaussian_peak,
        0.0,
        1.0,
        method="gk31",
        atol=1e-9,
        rtol=1e-9,
        max_batch=5,  # smaller than K=31
        device=cpu_device,
    )
    exact = math.sqrt(math.pi / 1000.0) * math.erf(math.sqrt(1000) * 0.5)
    assert abs(out.integral.item() - exact) < 1e-8


# --- fixed_quadrature --------------------------------------------------------


@pytest.mark.parametrize("n", [7, 10, 15, 21, 31])
def test_fixed_sin_pi(n, cpu_device):
    out = fixed_quadrature(
        sin_integrand,
        0.0,
        math.pi,
        method=f"gl{n}",
        device=cpu_device,
    )
    assert abs(out.integral.item() - 2.0) < 1e-10
    assert out.n_evaluations == n
    assert out.integral_error is None
    assert out.error_ratios is None


def test_fixed_chunked_matches_unchunked(cpu_device):
    a = fixed_quadrature(sin_integrand, 0.0, math.pi, method="gl31", device=cpu_device)
    b = fixed_quadrature(
        sin_integrand, 0.0, math.pi, method="gl31", max_batch=4, device=cpu_device
    )
    assert a.integral.item() == b.integral.item()


# --- input validation --------------------------------------------------------


def test_adaptive_rejects_fixed_method(cpu_device):
    with pytest.raises(ValueError, match="not adaptive"):
        adaptive_quadrature(sin_integrand, 0.0, 1.0, method="gl15", device=cpu_device)


def test_fixed_rejects_adaptive_method(cpu_device):
    with pytest.raises(ValueError, match="is adaptive"):
        fixed_quadrature(sin_integrand, 0.0, 1.0, method="gk21", device=cpu_device)


def test_adaptive_rejects_wrong_output_shape(cpu_device):
    def bad_f(t):
        return torch.sin(t)  # missing trailing D dim

    with pytest.raises(ValueError, match=r"shape \[N, D\]"):
        adaptive_quadrature(bad_f, 0.0, 1.0, method="gk21", device=cpu_device)


# --- max_iter behavior -------------------------------------------------------


def test_adaptive_max_iter_warns_and_returns(cpu_device):
    with pytest.warns(UserWarning, match="max_iter"):
        out = adaptive_quadrature(
            gaussian_peak,
            0.0,
            1.0,
            method="gk15",
            atol=1e-15,
            rtol=1e-15,
            max_iter=2,
            device=cpu_device,
        )
    # Still returns a usable estimate even when partial
    assert out.integral.numel() == 1
    assert out.n_iterations == 2


# --- bound-handling ----------------------------------------------------------


def test_adaptive_accepts_zero_d_tensor_bounds(cpu_device):
    a = torch.tensor(0.0)
    b = torch.tensor(math.pi)
    out = adaptive_quadrature(sin_integrand, a, b, method="gk21", device=cpu_device)
    assert abs(out.integral.item() - 2.0) < 1e-10


def test_adaptive_rejects_one_d_tensor_bound(cpu_device):
    with pytest.raises(ValueError, match="must be a scalar"):
        adaptive_quadrature(
            sin_integrand,
            torch.tensor([0.0]),
            math.pi,
            method="gk21",
            device=cpu_device,
        )


# --- dtype consistency -------------------------------------------------------


def test_adaptive_fp32(cpu_device):
    out = adaptive_quadrature(
        sin_integrand,
        0.0,
        math.pi,
        method="gk21",
        atol=1e-5,
        rtol=1e-5,
        device=cpu_device,
        dtype=torch.float32,
    )
    assert out.integral.dtype == torch.float32
    assert abs(out.integral.item() - 2.0) < 1e-5


# --- output structural invariants -------------------------------------------


def test_adaptive_t_eval_shape_matches_method_K(cpu_device):
    out = adaptive_quadrature(
        gaussian_peak,
        0.0,
        1.0,
        method="gk31",
        atol=1e-9,
        rtol=1e-9,
        device=cpu_device,
    )
    assert out.t.shape == (out.interval_integrals.shape[0], 31)
    assert out.y.shape == (out.interval_integrals.shape[0], 31, 1)


def test_adaptive_intervals_cover_domain(cpu_device):
    """Interval widths sum to the full domain with no gaps or overlaps."""
    out = adaptive_quadrature(
        gaussian_peak,
        0.0,
        1.0,
        method="gk21",
        atol=1e-9,
        rtol=1e-9,
        device=cpu_device,
    )
    assert torch.all(out.h > 0)
    assert abs(out.h.sum().item() - 1.0) < 1e-12


# --- fixed_quadrature input validation --------------------------------------


def test_fixed_rejects_wrong_output_shape(cpu_device):
    def bad_f(t):
        return torch.sin(t)  # missing trailing D

    with pytest.raises(ValueError, match=r"shape \[N, D\]"):
        fixed_quadrature(bad_f, 0.0, 1.0, method="gl15", device=cpu_device)


def test_fixed_n_evaluations_equals_K(cpu_device):
    out = fixed_quadrature(
        sin_integrand, 0.0, math.pi, method="gl21", device=cpu_device
    )
    assert out.n_evaluations == 21
    assert out.t.shape == (1, 21)
    assert out.h.shape == (1,)


# --- evaluate_chunked: more shapes ------------------------------------------


def test_evaluate_chunked_preserves_D(cpu_device):
    """A vector integrand keeps its trailing D under chunking."""
    from torchpathint.quadrature import _BatchState

    t = torch.linspace(0.0, 1.0, 13, dtype=torch.float64)
    out = evaluate_chunked(vector_integrand, t, _BatchState(4))
    assert out.shape == (13, 3)
    assert torch.allclose(out, vector_integrand(t))


def test_evaluate_chunked_single_call_when_max_batch_is_large(cpu_device):
    """If max_batch >= n, the helper should call f exactly once."""
    from torchpathint.quadrature import _BatchState

    calls = 0

    def counting(t):
        nonlocal calls
        calls += 1
        return sin_integrand(t)

    t = torch.linspace(0.0, 1.0, 50, dtype=torch.float64)
    evaluate_chunked(counting, t, _BatchState(100))
    assert calls == 1
    evaluate_chunked(counting, t, _BatchState(None))
    assert calls == 2  # one more, single-batched again
