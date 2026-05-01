"""Tests for the quadrature engines, the chunked evaluator, and OOM shrink.

We can't easily provoke a real CUDA OOM from a unit test (CUDA isn't
guaranteed in CI, and pushing the allocator over the cliff is flaky), so
the OOM-recovery tests simulate the failure mode by having the integrand
raise the matching exception itself. ``_is_cuda_oom`` accepts both
``torch.cuda.OutOfMemoryError`` and the legacy ``RuntimeError("CUDA out
of memory")`` form, and we exercise the legacy form so the tests run on
CPU.
"""

from __future__ import annotations

import math

import pytest
import torch

from torchpathint import (
    METHOD_NAMES_ADAPTIVE,
    adaptive_quadrature,
    fixed_quadrature,
)
from torchpathint.quadrature import (
    _expandable_segments_hint,
    _is_cuda_oom,
    evaluate_chunked,
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


def _oom_above(threshold: int):
    """Return ``(f, calls)`` where ``f`` raises a CUDA-OOM-shaped
    ``RuntimeError`` when called with more than ``threshold`` points and
    otherwise returns ``sin(t).unsqueeze(-1)``. ``calls`` records the
    per-call ``t.numel()`` for assertions."""
    calls: list[int] = []

    def f(t: torch.Tensor) -> torch.Tensor:
        calls.append(t.numel())
        if t.numel() > threshold:
            raise RuntimeError("CUDA out of memory. Tried to allocate ...")
        return torch.sin(t).unsqueeze(-1)

    return f, calls


# --- evaluate_chunked: basic correctness ------------------------------------


def test_evaluate_chunked_no_chunking_matches_direct(cpu_device):
    t = torch.linspace(0.0, 1.0, 50, dtype=torch.float64)
    out, max_batch = evaluate_chunked(sin_integrand, t, None)
    assert torch.allclose(out, sin_integrand(t))
    assert max_batch is None


def test_evaluate_chunked_small_batch_matches_direct(cpu_device):
    t = torch.linspace(0.0, 1.0, 50, dtype=torch.float64)
    direct = sin_integrand(t)
    for batch in [1, 3, 7, 49, 50, 100]:
        out, returned = evaluate_chunked(sin_integrand, t, batch)
        assert out.shape == direct.shape
        assert torch.allclose(out, direct)
        assert returned == batch


def test_evaluate_chunked_empty_raises(cpu_device):
    with pytest.raises(ValueError, match="empty tensor"):
        evaluate_chunked(sin_integrand, torch.zeros(0), None)


def test_evaluate_chunked_preserves_D(cpu_device):
    """A vector integrand keeps its trailing D under chunking."""
    t = torch.linspace(0.0, 1.0, 13, dtype=torch.float64)
    out, _ = evaluate_chunked(vector_integrand, t, 4)
    assert out.shape == (13, 3)
    assert torch.allclose(out, vector_integrand(t))


def test_evaluate_chunked_single_call_when_max_batch_is_large(cpu_device):
    """If max_batch >= n, the helper should call f exactly once."""
    calls = 0

    def counting(t):
        nonlocal calls
        calls += 1
        return sin_integrand(t)

    t = torch.linspace(0.0, 1.0, 50, dtype=torch.float64)
    evaluate_chunked(counting, t, 100)
    assert calls == 1
    evaluate_chunked(counting, t, None)
    assert calls == 2  # one more, single-batched again


# --- evaluate_chunked: OOM-driven shrink ------------------------------------


def test_is_cuda_oom_matches_modern_and_legacy():
    legacy = RuntimeError("CUDA out of memory. Tried to allocate 2 GB.")
    other = RuntimeError("expected scalar but got tensor")
    assert _is_cuda_oom(legacy)
    assert not _is_cuda_oom(other)
    assert not _is_cuda_oom(ValueError("not even a runtime error"))


def test_evaluate_chunked_halves_until_chunk_fits():
    """Integrand OOMs at >2 points; chunker halves 8 → 4 → 2 and succeeds."""
    f, calls = _oom_above(threshold=2)
    t = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)

    out, max_batch = evaluate_chunked(f, t, None)

    expected = torch.sin(t).unsqueeze(-1)
    assert torch.allclose(out, expected)
    # Final learned size is 2 (the largest power-of-two halving below 4).
    assert max_batch == 2
    # Sanity: there were OOM-triggering calls along the way (the 8 and the 4).
    assert 8 in calls
    assert 4 in calls
    # And at least one successful 2-point chunk.
    assert calls.count(2) >= 1


def test_returned_max_batch_lets_caller_skip_failed_sizes():
    """Threading the returned max_batch back into a second call lets the
    caller chunk straight at the learned size without re-discovering the
    failing sizes."""
    f, calls = _oom_above(threshold=2)
    t = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)

    _, learned = evaluate_chunked(f, t, None)
    calls_after_first = list(calls)

    evaluate_chunked(f, t, learned)

    new_calls = calls[len(calls_after_first) :]
    assert all(n == 2 for n in new_calls)
    assert len(new_calls) == 4  # 8 / 2 = 4 chunks


def test_non_oom_runtime_error_is_reraised():
    """RuntimeError that doesn't smell like OOM should propagate unchanged."""

    def f(t):
        raise RuntimeError("integrand: shape mismatch on tensor blah")

    t = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        evaluate_chunked(f, t, None)


def test_shrink_floor_is_one_then_reraises():
    """If the integrand can't even handle a single point, chunker shrinks to
    max_batch=1, fails again, and re-raises rather than spinning forever."""
    f, _calls = _oom_above(threshold=0)  # OOMs at any N >= 1
    t = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)

    with pytest.raises(RuntimeError, match="out of memory"):
        evaluate_chunked(f, t, None)


def test_user_max_batch_is_used_as_starting_size():
    """Caller-supplied max_batch becomes the initial chunk size; OOM still
    halves from there."""
    f, calls = _oom_above(threshold=2)
    t = torch.linspace(0.0, 1.0, 16, dtype=torch.float64)

    _, max_batch = evaluate_chunked(f, t, 8)

    assert calls[0] == 8  # First chunk size attempted is the user-set value.
    assert max_batch == 2


def test_expandable_segments_hint_when_unset(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    hint = _expandable_segments_hint()
    assert hint
    assert "expandable_segments:True" in hint


def test_expandable_segments_hint_silent_when_set(monkeypatch):
    monkeypatch.setenv(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,foo:bar",
    )
    assert _expandable_segments_hint() == ""


def test_oom_warning_includes_hint_when_unset(monkeypatch, caplog):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    f, _calls = _oom_above(threshold=2)
    t = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    with caplog.at_level("WARNING", logger="torchpathint.quadrature"):
        evaluate_chunked(f, t, None)
    assert any("expandable_segments" in r.getMessage() for r in caplog.records)


def test_oom_warning_omits_hint_when_set(monkeypatch, caplog):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    f, _calls = _oom_above(threshold=2)
    t = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    with caplog.at_level("WARNING", logger="torchpathint.quadrature"):
        evaluate_chunked(f, t, None)
    assert not any("expandable_segments" in r.getMessage() for r in caplog.records)


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
        full_output=True,
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
        full_output=True,
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


def test_fixed_rejects_wrong_output_shape(cpu_device):
    def bad_f(t):
        return torch.sin(t)  # missing trailing D

    with pytest.raises(ValueError, match=r"shape \[N, D\]"):
        fixed_quadrature(bad_f, 0.0, 1.0, method="gl15", device=cpu_device)


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
        full_output=True,
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
        full_output=True,
    )
    assert torch.all(out.h > 0)
    assert abs(out.h.sum().item() - 1.0) < 1e-12


def test_fixed_n_evaluations_equals_K(cpu_device):
    out = fixed_quadrature(
        sin_integrand, 0.0, math.pi, method="gl21", device=cpu_device, full_output=True
    )
    assert out.n_evaluations == 21
    assert out.t.shape == (1, 21)
    assert out.h.shape == (1,)


# --- end-to-end OOM recovery through the engines ----------------------------


def test_adaptive_quadrature_recovers_when_first_call_OOMs():
    """An integrand that OOMs at the first iteration's full K-node call
    still converges via the chunker's halving."""
    K = 21  # gk21 nodes per interval
    threshold = K // 2  # OOM above K/2 forces at least one halving

    def integrand(t):
        if t.numel() > threshold:
            raise RuntimeError("CUDA out of memory: synthetic")
        return torch.sin(t).unsqueeze(-1)

    out = adaptive_quadrature(
        integrand,
        0.0,
        torch.pi,
        method="gk21",
        atol=1e-8,
        rtol=1e-8,
        device="cpu",
        dtype=torch.float64,
    )
    assert torch.allclose(
        out.integral, torch.tensor([2.0], dtype=torch.float64), atol=1e-6
    )


def test_fixed_quadrature_recovers_when_full_K_OOMs():
    K = 7  # gl7

    def integrand(t):
        if t.numel() > K // 2:
            raise RuntimeError("CUDA out of memory: synthetic")
        return torch.sin(t).unsqueeze(-1)

    out = fixed_quadrature(
        integrand,
        0.0,
        torch.pi,
        method=f"gl{K}",
        device="cpu",
        dtype=torch.float64,
    )
    # gl7 isn't exact for sin over [0, π] but it's close enough for sanity.
    assert torch.allclose(
        out.integral, torch.tensor([2.0], dtype=torch.float64), atol=1e-2
    )
