"""GPU memory probing for automatic batch sizing.

The adaptive integrator builds a flat ``[total_points]`` tensor of all pending
quadrature evaluations and passes it to ``f``. ``max_batch`` chunks that
tensor so peak memory is bounded by what one chunk costs. Picking a good
``max_batch`` by hand is annoying — it depends on ``f`` (its intermediates,
its output dim ``D``), the device, and what else is on the GPU right now.

``estimate_max_batch`` measures ``f`` at a few growing input sizes, infers
per-evaluation peak bytes, and divides the budget by it. The budget is
``total_mem_usage * free_memory``: a fraction of what's currently available
on the device. This matches the user's intent on shared GPUs (where
"reserve a fraction of total" is meaningless because total is owned by
other processes) and reduces to the dedicated-GPU case when free ≈ total.

CPU is treated as unbounded: the function returns ``None`` so the caller
falls through to a single big batch. CPU memory is harder to attribute
correctly across PyTorch and the OS, and CPU OOMs hurt less than GPU ones.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Multiplier on measured peak bytes per evaluation. The integrator allocates
# extra working tensors during quadrature (running totals, error tensors,
# concatenations across iterations) that the probe doesn't see, so leave a
# margin. 2.0x matches the old torchpathdiffeq probe's safety factor.
_SAFETY_FACTOR = 2.0

# Probe sizes. Doubling keeps the probe cheap (O(log N) measurements) while
# averaging out small-N startup noise — small N is dominated by fixed
# allocator overhead, so we want the largest size that still fits.
_PROBE_SIZES = (8, 64, 512, 4096)


def _cuda_budget_bytes(device: torch.device, total_mem_usage: float) -> int:
    """Bytes the integrator may consume under the user's memory cap.

    Budget = ``total_mem_usage * available``, where ``available`` is the
    currently free memory plus PyTorch's cached-but-unused memory (the
    allocator will hand the latter back to us). On a dedicated GPU this is
    essentially ``total_mem_usage * total``; on a shared GPU it scales with
    what's actually free right now, which is what the user almost always
    means by "use up to X% of memory."
    """
    free, _total = torch.cuda.mem_get_info(device)
    cached_unused = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(
        device
    )
    available = free + cached_unused
    return max(0, int(total_mem_usage * available))


def estimate_max_batch(
    f: Callable[[torch.Tensor], torch.Tensor],
    t_sample: torch.Tensor,
    device: torch.device,
    total_mem_usage: float,
) -> int | None:
    """Pick a ``max_batch`` value that fits ``f`` into the memory budget.

    Args:
        f: The integrand. Must satisfy ``f: Tensor[N] -> Tensor[N, D]``.
        t_sample: A 0-d tensor used to construct probe inputs. Use a
            representative time point — typically ``t_init`` or the midpoint
            of the integration domain.
        device: Device to probe. Non-CUDA devices return ``None`` (no
            chunking).
        total_mem_usage: Fraction of currently-available device memory the
            integrator may consume, in ``(0, 1]``. ``0.9`` leaves a 10%
            headroom of free memory for kernel workspaces and concurrent
            allocations.

    Returns:
        A positive ``int`` ``max_batch`` for CUDA, or ``None`` if probing
        is not applicable (CPU device, or per-eval cost couldn't be
        measured — e.g. an integrand that allocates nothing persistent).

    Raises:
        ValueError: if ``total_mem_usage`` is outside ``(0, 1]``.
        RuntimeError: if the budget computes to zero — caller must reduce
            memory pressure or raise the cap before integrating.
    """
    if not 0.0 < total_mem_usage <= 1.0:
        raise ValueError(f"total_mem_usage must be in (0, 1]; got {total_mem_usage!r}.")

    if device.type != "cuda":
        return None

    if t_sample.dim() != 0:
        raise ValueError(
            f"t_sample must be a 0-d tensor; got shape {tuple(t_sample.shape)}."
        )

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    budget = _cuda_budget_bytes(device, total_mem_usage)
    if budget <= 0:
        free, total = torch.cuda.mem_get_info(device)
        raise RuntimeError(
            f"GPU memory budget is empty: only {free / 1e9:.2f} GB free of "
            f"{total / 1e9:.2f} GB total. Free up memory before integrating."
        )

    per_eval_bytes = 0.0
    measured_at: int | None = None
    for n in _PROBE_SIZES:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        t_probe = t_sample.detach().expand(n).contiguous()
        try:
            y = f(t_probe)
            torch.cuda.synchronize(device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
        peak = torch.cuda.max_memory_allocated(device)
        cost = max(0, peak - baseline) / n
        if cost > per_eval_bytes:
            per_eval_bytes = cost
            measured_at = n
        del y, t_probe
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    if per_eval_bytes <= 0:
        # Integrand allocates nothing the allocator notices (e.g. closed-form
        # operations whose intermediates fit in registers/cache). No chunking
        # needed; let the caller pass the full tensor.
        logger.debug(
            "estimate_max_batch: measured 0 B/eval at all probe sizes; "
            "skipping chunking."
        )
        return None

    per_eval_bytes *= _SAFETY_FACTOR
    max_batch = max(1, int(budget // per_eval_bytes))
    logger.debug(
        "estimate_max_batch: %.1f B/eval (probed at N=%d, %.1fx safety), "
        "budget %.2f GB -> max_batch=%d",
        per_eval_bytes,
        measured_at,
        _SAFETY_FACTOR,
        budget / 1e9,
        max_batch,
    )
    return max_batch
