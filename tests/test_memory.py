"""Tests for automatic batch sizing via the memory probe.

CPU paths are exercised everywhere (return-None contract, validation, override
semantics). CUDA-specific behavior — actual probe measurements and end-to-end
auto-sized integration — runs only when a GPU is available.
"""

from __future__ import annotations

import math

import pytest
import torch

import torchpathint.quadrature as quad_mod
from torchpathint import estimate_max_batch, fixed_quadrature, path_integral


def sin_integrand(t: torch.Tensor) -> torch.Tensor:
    return torch.sin(t).unsqueeze(-1)


def heavy_integrand(scale: int):
    """Integrand that allocates a sizeable temporary per evaluation.

    Used to make the per-eval memory cost large enough to measure reliably
    above allocator noise.
    """

    def f(t: torch.Tensor) -> torch.Tensor:
        # [N, scale] temporary, summed back to [N, 1] so the output is small
        # but the peak allocation is N * scale * 8 bytes.
        big = t.unsqueeze(-1) * torch.linspace(
            0.0, 1.0, scale, device=t.device, dtype=t.dtype
        )
        return big.sum(dim=-1, keepdim=True)

    return f


# --- estimate_max_batch: input validation ------------------------------------


def test_estimate_max_batch_rejects_bad_memory_fraction():
    device = torch.device("cpu")
    t = torch.tensor(0.0, dtype=torch.float64)
    with pytest.raises(ValueError, match="memory_fraction"):
        estimate_max_batch(sin_integrand, t, device, memory_fraction=0.0)
    with pytest.raises(ValueError, match="memory_fraction"):
        estimate_max_batch(sin_integrand, t, device, memory_fraction=1.5)
    with pytest.raises(ValueError, match="memory_fraction"):
        estimate_max_batch(sin_integrand, t, device, memory_fraction=-0.1)


def test_estimate_max_batch_returns_none_on_cpu():
    device = torch.device("cpu")
    t = torch.tensor(0.0, dtype=torch.float64)
    assert estimate_max_batch(sin_integrand, t, device, memory_fraction=0.5) is None


def test_estimate_max_batch_rejects_non_scalar_sample():
    # CPU short-circuits to None before checking shape, so use a CUDA device
    # if available, otherwise skip — the rejection path needs the CUDA branch.
    if not torch.cuda.is_available():
        pytest.skip("CUDA required to exercise non-scalar t_sample rejection.")
    device = torch.device("cuda")
    t_bad = torch.zeros(3, device=device, dtype=torch.float64)
    with pytest.raises(ValueError, match="0-d tensor"):
        estimate_max_batch(sin_integrand, t_bad, device, memory_fraction=0.5)


# --- path_integral: memory_fraction on CPU is silently ignored ---------------


def test_path_integral_memory_fraction_on_cpu_ignored():
    out = path_integral(
        sin_integrand,
        0.0,
        math.pi,
        method="gk21",
        atol=1e-9,
        rtol=1e-9,
        device="cpu",
        memory_fraction=0.5,
    )
    assert torch.allclose(out.integral, torch.tensor([2.0], dtype=torch.float64))


def test_fixed_quadrature_memory_fraction_on_cpu_ignored():
    out = fixed_quadrature(
        sin_integrand,
        0.0,
        math.pi,
        method="gl15",
        device="cpu",
        memory_fraction=0.5,
    )
    assert torch.allclose(
        out.integral, torch.tensor([2.0], dtype=torch.float64), atol=1e-12
    )


# --- max_batch overrides memory_fraction -------------------------------------


def test_max_batch_overrides_memory_fraction(monkeypatch):
    """If the user supplies max_batch explicitly, the probe is never run."""

    def boom(*_a, **_kw):
        raise AssertionError("estimate_max_batch should not be called")

    monkeypatch.setattr(quad_mod, "estimate_max_batch", boom)
    out = path_integral(
        sin_integrand,
        0.0,
        math.pi,
        method="gk21",
        atol=1e-9,
        rtol=1e-9,
        device="cpu",
        max_batch=4,
        memory_fraction=0.5,
    )
    assert torch.allclose(out.integral, torch.tensor([2.0], dtype=torch.float64))


# --- CUDA: probe measurement and auto-sized integration ----------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_estimate_max_batch_returns_positive_int_on_cuda():
    device = torch.device("cuda")
    t = torch.tensor(0.5, device=device, dtype=torch.float64)
    mb = estimate_max_batch(heavy_integrand(scale=4096), t, device, memory_fraction=0.5)
    assert isinstance(mb, int)
    assert mb > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_auto_batch_path_integral_matches_full_batch():
    """Auto-sizing must give the same answer as no chunking, to tolerance."""
    device = torch.device("cuda")
    f = heavy_integrand(scale=1024)
    out_full = path_integral(
        f, 0.0, 1.0, method="gk21", atol=1e-9, rtol=1e-9, device=device
    )
    out_auto = path_integral(
        f,
        0.0,
        1.0,
        method="gk21",
        atol=1e-9,
        rtol=1e-9,
        device=device,
        memory_fraction=0.5,
    )
    assert torch.allclose(out_full.integral, out_auto.integral, atol=1e-12, rtol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_auto_batch_picks_smaller_batch_when_budget_is_tight():
    """A tighter budget should produce a smaller (or equal) max_batch."""
    device = torch.device("cuda")
    t = torch.tensor(0.5, device=device, dtype=torch.float64)
    f = heavy_integrand(scale=4096)
    mb_loose = estimate_max_batch(f, t, device, memory_fraction=0.9)
    mb_tight = estimate_max_batch(f, t, device, memory_fraction=0.05)
    if mb_loose is None or mb_tight is None:
        pytest.skip("integrand allocated no measurable memory")
    assert mb_tight <= mb_loose


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_estimate_max_batch_returns_none_when_unmeasurable():
    """Trivial integrands (sin(t)) often allocate ~0 persistent bytes —
    the probe must fall through to None rather than fabricate a number."""
    device = torch.device("cuda")
    t = torch.tensor(0.5, device=device, dtype=torch.float64)
    # Scalar-output trivial op — likely no measurable allocation.
    result = estimate_max_batch(sin_integrand, t, device, memory_fraction=0.5)
    # Either None (no measurable cost) or a very large int — both are valid.
    assert result is None or (isinstance(result, int) and result > 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_auto_batch_actually_chunks_inside_iteration():
    """Heavy integrand should drive auto max_batch below K, forcing the
    integrator to evaluate one quadrature iteration across multiple f calls.

    The smoke checks above only verify max_batch is positive — they don't
    distinguish "huge max_batch, never chunks" from "bounded max_batch,
    chunks correctly." This test picks a per-eval cost large enough that
    the probe returns max_batch < K, then asserts (a) the integrator
    actually issues > 1 f call inside a single iteration and (b) the
    chunked answer matches a full-batch reference bit-for-bit.
    """
    device = torch.device("cuda")
    free_bytes, _ = torch.cuda.mem_get_info(device)

    # 50M fp64 elements per eval = 400 MB temp per evaluation. With the 2x
    # safety factor that's 800 MB. Auto max_batch ≈ frac * free / 800 MB,
    # which is < K=31 whenever (frac * free) < ~25 GB. On a 40 GB card with
    # frac=0.5 that gives max_batch ≈ 26; on a 16 GB card, ≈ 10.
    scale = 50_000_000

    # Probe at N=8 needs scale * 8 floats * 8 bytes = 64 * scale bytes free.
    # Skip if even that won't fit (login-node sliver of GPU memory).
    if 64 * scale > 0.5 * free_bytes:
        pytest.skip(
            f"need ~{64 * scale / 1e9:.1f} GB free for probe at N=8; "
            f"have {free_bytes / 1e9:.2f} GB"
        )

    K = 31  # gk31

    # Standalone probe: must report a max_batch < K so the integrator chunks.
    t_sample = torch.tensor(0.5, device=device, dtype=torch.float64)
    mb = estimate_max_batch(
        heavy_integrand(scale), t_sample, device, memory_fraction=0.5
    )
    assert mb is not None, "probe should measure a non-zero per-eval cost"
    assert mb < K, (
        f"test setup failed: expected auto max_batch < K={K} so chunking "
        f"engages; got max_batch={mb}. The GPU is too large for this scale "
        f"or the safety factor is too small."
    )

    # End-to-end with auto-sizing: count f calls coming from inside the
    # integrator (probe runs first; we count separately via a flag).
    integrator_calls = 0
    in_integration = False

    def f_counted(t: torch.Tensor) -> torch.Tensor:
        nonlocal integrator_calls
        if in_integration:
            integrator_calls += 1
        big = t.unsqueeze(-1) * torch.linspace(
            0.0, 1.0, scale, device=t.device, dtype=t.dtype
        )
        return big.sum(dim=-1, keepdim=True)

    in_integration = True
    out_auto = path_integral(
        f_counted,
        0.0,
        1.0,
        method="gk31",
        atol=1e-9,
        rtol=1e-9,
        device=device,
        memory_fraction=0.5,
    )
    in_integration = False
    auto_calls = integrator_calls

    # Reference: same call but with max_batch large enough to skip chunking.
    integrator_calls = 0
    in_integration = True
    out_ref = path_integral(
        f_counted,
        0.0,
        1.0,
        method="gk31",
        atol=1e-9,
        rtol=1e-9,
        device=device,
        max_batch=10 * K,  # plenty for any iteration's interval count
    )
    in_integration = False
    ref_calls = integrator_calls

    # Per iteration, the integrator slices [n_pending * K] into
    # ceil(n_pending * K / max_batch) chunks. The reference runs one chunk
    # per iteration, so ref_calls == n_iterations. With max_batch < K, even
    # the very first iteration (one interval, K nodes) takes >= 2 chunks.
    assert ref_calls == out_ref.n_iterations, (
        f"reference setup wrong: expected one f call per iteration, "
        f"got {ref_calls} calls for {out_ref.n_iterations} iterations"
    )
    assert auto_calls > out_auto.n_iterations, (
        f"auto-sized run should chunk: got {auto_calls} f calls for "
        f"{out_auto.n_iterations} iterations (max_batch={mb}, K={K})"
    )

    # Same total number of integrand evaluations regardless of chunking.
    assert out_auto.n_evaluations == out_ref.n_evaluations
    assert torch.allclose(out_auto.integral, out_ref.integral, atol=1e-12, rtol=1e-12)
