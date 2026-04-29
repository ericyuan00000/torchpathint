"""Tests for OOM-driven chunk shrinking in evaluate_chunked.

We can't easily provoke a real CUDA OOM in a unit test (CUDA isn't
guaranteed in CI, and pushing the allocator over the cliff is flaky),
so each test simulates the failure mode by having the integrand raise
the matching exception itself. ``_is_cuda_oom`` matches both
``torch.cuda.OutOfMemoryError`` and the legacy ``RuntimeError("CUDA
out of memory")`` form, and we exercise the legacy form here so the
tests run on CPU.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from torchpathint import adaptive_quadrature, fixed_quadrature
from torchpathint.quadrature import (
    _BatchState,
    _expandable_segments_hint,
    _is_cuda_oom,
    evaluate_chunked,
)


def _oom_above(threshold: int):
    """Return an integrand that raises a CUDA-OOM-shaped RuntimeError when
    called with more than ``threshold`` points, otherwise returns sin(t)."""
    calls: list[int] = []

    def f(t: torch.Tensor) -> torch.Tensor:
        calls.append(t.numel())
        if t.numel() > threshold:
            raise RuntimeError("CUDA out of memory. Tried to allocate ...")
        return torch.sin(t).unsqueeze(-1)

    return f, calls


def test_is_cuda_oom_matches_modern_and_legacy():
    legacy = RuntimeError("CUDA out of memory. Tried to allocate 2 GB.")
    other = RuntimeError("expected scalar but got tensor")
    assert _is_cuda_oom(legacy)
    assert not _is_cuda_oom(other)
    assert not _is_cuda_oom(ValueError("not even a runtime error"))


def test_evaluate_chunked_halves_until_chunk_fits():
    """Integrand OOMs at >2 points; chunker halves 8 → 4 → 2 and succeeds."""
    f, calls = _oom_above(threshold=2)
    state = _BatchState(None)  # start unchunked
    t = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)

    out = evaluate_chunked(f, t, state)

    expected = torch.sin(t).unsqueeze(-1)
    assert torch.allclose(out, expected)
    # Final learned size is 2 (the largest power-of-two halving below 4).
    assert state.max_batch == 2
    # Sanity: there were OOM-triggering calls along the way (the 8 and the 4).
    assert 8 in calls and 4 in calls
    # And at least one successful 2-point chunk.
    assert calls.count(2) >= 1


def test_state_persists_so_second_call_skips_failed_sizes():
    """Once shrunk, the state is reused so the next evaluate_chunked call
    on the same state doesn't re-discover the failing sizes."""
    f, calls = _oom_above(threshold=2)
    state = _BatchState(None)
    t = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)

    evaluate_chunked(f, t, state)
    calls_after_first = list(calls)

    evaluate_chunked(f, t, state)

    # Second call should chunk straight at state.max_batch=2 and never
    # re-trigger an OOM. Compare the call sizes added by the second call.
    new_calls = calls[len(calls_after_first):]
    assert all(n == 2 for n in new_calls)
    assert len(new_calls) == 4  # 8 / 2 = 4 chunks


def test_non_oom_runtime_error_is_reraised():
    """RuntimeError that doesn't smell like OOM should propagate unchanged."""

    def f(t):
        raise RuntimeError("integrand: shape mismatch on tensor blah")

    state = _BatchState(None)
    t = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        evaluate_chunked(f, t, state)
    # And no shrinking happened.
    assert state.max_batch is None


def test_shrink_floor_is_one_then_reraises():
    """If the integrand can't even handle a single point, chunker shrinks to
    max_batch=1, fails again, and re-raises rather than spinning forever."""
    f, _calls = _oom_above(threshold=0)  # OOMs at any N >= 1
    state = _BatchState(None)
    t = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)

    with pytest.raises(RuntimeError, match="out of memory"):
        evaluate_chunked(f, t, state)
    assert state.max_batch == 1


def test_user_max_batch_is_used_as_starting_size():
    """Caller-supplied max_batch becomes the initial chunk size; OOM still
    halves from there."""
    f, calls = _oom_above(threshold=2)
    state = _BatchState(max_batch=8)
    t = torch.linspace(0.0, 1.0, 16, dtype=torch.float64)

    evaluate_chunked(f, t, state)

    # First chunk size attempted should be 8 (the user-set value), then 4,
    # then 2.
    assert calls[0] == 8
    assert state.max_batch == 2


def test_adaptive_quadrature_recovers_when_first_call_OOMs():
    """End-to-end: an integrand that OOMs at the first iteration's full
    K-node call still converges via the chunker's halving."""
    K = 21  # gk21 nodes per interval
    # OOM above K/2: forces at least one halving even on the smooth case.
    threshold = K // 2

    def integrand(t):
        if t.numel() > threshold:
            raise RuntimeError("CUDA out of memory: synthetic")
        return torch.sin(t).unsqueeze(-1)

    out = adaptive_quadrature(
        integrand, 0.0, torch.pi,
        method="gk21", atol=1e-8, rtol=1e-8,
        device="cpu", dtype=torch.float64,
    )
    assert torch.allclose(out.integral, torch.tensor([2.0], dtype=torch.float64),
                          atol=1e-6)


def test_fixed_quadrature_recovers_when_full_K_OOMs():
    K = 7  # we'll use gl7 below

    def integrand(t):
        if t.numel() > K // 2:
            raise RuntimeError("CUDA out of memory: synthetic")
        return torch.sin(t).unsqueeze(-1)

    out = fixed_quadrature(
        integrand, 0.0, torch.pi, method=f"gl{K}",
        device="cpu", dtype=torch.float64,
    )
    # gl7 isn't exact for sin over [0, π], but it's close enough for sanity.
    assert torch.allclose(out.integral, torch.tensor([2.0], dtype=torch.float64),
                          atol=1e-2)


def test_expandable_segments_hint_when_unset(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    hint = _expandable_segments_hint()
    assert hint
    assert "expandable_segments:True" in hint


def test_expandable_segments_hint_silent_when_set(monkeypatch):
    monkeypatch.setenv(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,foo:bar",
    )
    assert _expandable_segments_hint() == ""


def test_oom_warning_includes_hint_when_unset(monkeypatch, caplog):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    f, _calls = _oom_above(threshold=2)
    state = _BatchState(None)
    t = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    with caplog.at_level("WARNING", logger="torchpathint.quadrature"):
        evaluate_chunked(f, t, state)
    assert any("expandable_segments" in r.getMessage() for r in caplog.records)


def test_oom_warning_omits_hint_when_set(monkeypatch, caplog):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    f, _calls = _oom_above(threshold=2)
    state = _BatchState(None)
    t = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    with caplog.at_level("WARNING", logger="torchpathint.quadrature"):
        evaluate_chunked(f, t, state)
    assert not any("expandable_segments" in r.getMessage() for r in caplog.records)


def test_memory_fraction_emits_deprecation_warning():
    """The deprecated kwarg is accepted but warns and is otherwise ignored."""

    def integrand(t):
        return torch.sin(t).unsqueeze(-1)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        adaptive_quadrature(
            integrand, 0.0, torch.pi,
            method="gk21", atol=1e-8, rtol=1e-8,
            memory_fraction=0.5, device="cpu", dtype=torch.float64,
        )
    deprecation_warnings = [
        x for x in w if issubclass(x.category, DeprecationWarning)
        and "memory_fraction" in str(x.message)
    ]
    assert len(deprecation_warnings) == 1
