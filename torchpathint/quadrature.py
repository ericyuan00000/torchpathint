"""Adaptive and fixed quadrature engines.

The adaptive engine ``adaptive_quadrature`` is the heart of the library:

1. Start with one interval ``[t_init, t_final]`` (or a user-provided mesh).
2. Evaluate the integrand at the rule's quadrature nodes inside every
   pending interval, *in parallel*. Evaluations are flattened into a single
   ``[total_points]`` tensor and chunked by ``max_batch`` if the user wants
   bounded GPU memory — chunks may span interval boundaries, so a single
   high-order interval no longer has to fit in memory.
3. For each interval, compute the primary (Kronrod) integral estimate and
   the embedded error from the same K evaluations. Compare the per-interval
   error against ``atol + rtol * |total_integral_so_far|``.
4. Accept intervals under tolerance — store their contributions and stop
   refining them. Split rejected intervals at the midpoint and re-evaluate
   the two halves on the next iteration.
5. Repeat until all intervals are accepted or ``max_iter`` is hit.

The autograd graph is preserved end-to-end: ``f(t)`` is never detached, and
all interval bookkeeping uses concatenation/indexing that flows gradients
back through the integration bounds and the integrand parameters.

For non-adaptive use, ``fixed_quadrature`` applies a single Gauss-Legendre
rule on the full domain — useful as a baseline or when smoothness is known.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import torch

from .base import IntegralOutput, normalize_bound, resolve_device
from .memory import estimate_max_batch
from .methods import get_method

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def evaluate_chunked(
    f: Callable[[torch.Tensor], torch.Tensor],
    t: torch.Tensor,
    max_batch: int | None,
) -> torch.Tensor:
    """Evaluate ``f`` on a flat 1-d tensor of points, optionally in chunks.

    Decouples GPU memory from the quadrature rule's order: even a single
    interval's K nodes can span multiple chunks if K is large enough.

    Args:
        f: Integrand. Takes shape ``[N]``, returns ``[N, D]``.
        t: 1-d tensor of evaluation points.
        max_batch: Maximum chunk size. If ``None`` or ``>= t.numel()``,
            evaluates all points in a single call.

    Returns:
        Tensor of shape ``[N, D]``.
    """
    n = t.numel()
    if n == 0:
        raise ValueError("evaluate_chunked received an empty tensor.")
    if max_batch is None or max_batch >= n:
        return f(t)
    parts = [f(t[start : start + max_batch]) for start in range(0, n, max_batch)]
    return torch.cat(parts, dim=0)


def adaptive_quadrature(
    f: Callable[[torch.Tensor], torch.Tensor],
    t_init: float | torch.Tensor,
    t_final: float | torch.Tensor,
    *,
    method: str = "gk21",
    atol: float = 1e-5,
    rtol: float = 1e-5,
    t: torch.Tensor | None = None,
    max_batch: int | None = None,
    total_mem_usage: float | None = None,
    max_iter: int = 50,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> IntegralOutput:
    """Adaptive Gauss-Kronrod quadrature on ``[t_init, t_final]``.

    Args:
        f: Integrand ``f: Tensor[N] -> Tensor[N, D]``. Receives a flat tensor
            of time points and returns a 2-d tensor where the leading axis
            matches the input length.
        t_init: Lower integration bound (Python scalar or 0-d tensor).
        t_final: Upper integration bound (same).
        method: Adaptive Gauss-Kronrod rule name (``"gk15"``, ``"gk21"``, or
            ``"gk31"``).
        atol: Absolute tolerance for per-interval error.
        rtol: Relative tolerance, scaled by the running integral magnitude.
        t: Optional initial interior mesh of barrier positions (1-d). The
            integrator inserts ``t_init`` and ``t_final`` and sorts. Useful
            for warm-starting from a previous ``IntegralOutput.t_optimal``.
        max_batch: Maximum integrand evaluations per ``f`` call. ``None``
            means no chunking (one big batch) unless ``total_mem_usage`` is
            set, in which case chunking is sized automatically. Chunks span
            interval boundaries.
        total_mem_usage: Fraction of currently-free GPU memory
            (``(0, 1]``) the integrator may consume. When set and
            ``max_batch is None``, ``f`` is benchmarked at a few sizes to
            estimate per-evaluation cost, and ``max_batch`` is chosen so
            chunks fit in ``total_mem_usage * free_memory``. Ignored on CPU.
            ``max_batch`` overrides this if both are set.
        max_iter: Maximum refinement iterations. On the last iteration any
            still-over-tolerance intervals are force-accepted with a warning.
        device: Device for internal tensors. Defaults to CUDA if available.
        dtype: Floating-point dtype. Defaults to ``torch.float64``.

    Returns:
        :class:`IntegralOutput` with the integral, error estimate, mesh, and
        per-interval diagnostics. ``t_optimal`` is the converged mesh suitable
        for passing back as ``t`` on a subsequent call.
    """
    device = resolve_device(device)
    t_init_t = normalize_bound(t_init, device, dtype, "t_init")
    t_final_t = normalize_bound(t_final, device, dtype, "t_final")
    method_obj = get_method(method, device, dtype)
    if not method_obj.is_adaptive:
        raise ValueError(
            f"Method {method!r} is not adaptive. Use a 'gk*' method here, "
            "or call fixed_quadrature for non-adaptive 'gl*' rules."
        )
    nodes = method_obj.nodes
    weights = method_obj.weights
    weights_error = method_obj.weights_error
    K = nodes.numel()

    if max_batch is None and total_mem_usage is not None:
        max_batch = estimate_max_batch(f, t_init_t, device, total_mem_usage)

    # Initial pending intervals
    if t is None:
        pending_left = t_init_t.unsqueeze(0)
        pending_right = t_final_t.unsqueeze(0)
    else:
        if t.dim() != 1:
            raise ValueError(f"t must be 1-d; got shape {tuple(t.shape)}.")
        barriers = torch.cat(
            [
                t_init_t.unsqueeze(0),
                t.to(device=device, dtype=dtype),
                t_final_t.unsqueeze(0),
            ]
        )
        barriers, _ = barriers.sort()
        pending_left = barriers[:-1]
        pending_right = barriers[1:]

    accepted_t_left: list[torch.Tensor] = []
    accepted_t_right: list[torch.Tensor] = []
    accepted_contrib: list[torch.Tensor] = []
    accepted_err: list[torch.Tensor] = []
    accepted_y: list[torch.Tensor] = []
    accepted_t_eval: list[torch.Tensor] = []

    # Running sum over accepted intervals, shape [D]. Lazily allocated on the
    # first iteration once we learn D from f's output.
    accepted_integral: torch.Tensor | None = None
    n_evaluations = 0
    n_iter = 0
    forced_partial = False

    while pending_left.numel() > 0:
        n_iter += 1
        n_pending = pending_left.numel()

        # Map the rule's [-1, 1] nodes into each pending interval's t range.
        h_half = (pending_right - pending_left) / 2
        t_mid = (pending_right + pending_left) / 2
        # [n_pending, K] node positions in original t space.
        t_eval_pending = h_half.unsqueeze(-1) * nodes.unsqueeze(0) + t_mid.unsqueeze(-1)

        # Single batched call: every pending interval's K nodes are evaluated
        # together, optionally chunked across f calls when max_batch is set.
        t_flat = t_eval_pending.reshape(-1)
        y_flat = evaluate_chunked(f, t_flat, max_batch)
        if y_flat.dim() != 2 or y_flat.shape[0] != t_flat.numel():
            raise ValueError(
                "Integrand f must return shape [N, D] matching the input length; "
                f"got input shape [{t_flat.numel()}] and output shape "
                f"{tuple(y_flat.shape)}."
            )
        D = y_flat.shape[-1]
        y_pending = y_flat.reshape(n_pending, K, D)
        n_evaluations += t_flat.numel()
        if accepted_integral is None:
            accepted_integral = y_flat.new_zeros(D)

        # Per-interval contributions (Kronrod = primary) and embedded-rule
        # difference (= primary - Gauss). The Jacobian h/2 maps [-1, 1] -> [t_l, t_r].
        contrib_pending = h_half.unsqueeze(-1) * torch.einsum(
            "k,ikd->id", weights, y_pending
        )
        err_pending = h_half.unsqueeze(-1) * torch.einsum(
            "k,ikd->id", weights_error, y_pending
        )

        # Accept / reject by per-interval error vs the *running* total: the
        # rtol scale uses our best estimate of the integral so far so early
        # intervals don't get judged against a near-zero magnitude.
        running_total = accepted_integral + contrib_pending.sum(dim=0)
        ratio_denom = atol + rtol * running_total.abs()  # [D]
        ratio_per_d = err_pending.abs() / ratio_denom
        # RMS over D so a single noisy output dim doesn't dominate splitting.
        ratio = torch.sqrt((ratio_per_d**2).mean(dim=-1))  # [n_pending]

        if n_iter >= max_iter:
            n_over = int((ratio >= 1.0).sum())
            if n_over > 0:
                forced_partial = True
                warnings.warn(
                    f"Adaptive quadrature: max_iter={max_iter} reached with "
                    f"{n_over}/{n_pending} intervals over tolerance. Returning "
                    "partial estimate; tighten max_iter, atol/rtol, or pass an "
                    "initial mesh `t` to refine further.",
                    stacklevel=2,
                )
            accept_mask = torch.ones_like(ratio, dtype=torch.bool)
        else:
            accept_mask = ratio < 1.0

        if accept_mask.any():
            accepted_t_left.append(pending_left[accept_mask])
            accepted_t_right.append(pending_right[accept_mask])
            accepted_contrib.append(contrib_pending[accept_mask])
            accepted_err.append(err_pending[accept_mask])
            accepted_y.append(y_pending[accept_mask])
            accepted_t_eval.append(t_eval_pending[accept_mask])
            accepted_integral = accepted_integral + contrib_pending[accept_mask].sum(
                dim=0
            )

        # Rejected intervals are split at their midpoint; the two halves go
        # into the next iteration. Accepted intervals are dropped from pending.
        reject_mask = ~accept_mask
        rej_left = pending_left[reject_mask]
        rej_right = pending_right[reject_mask]
        rej_mid = (rej_left + rej_right) / 2
        pending_left = torch.cat([rej_left, rej_mid])
        pending_right = torch.cat([rej_mid, rej_right])

    if not accepted_contrib:
        # All initial intervals were forced-rejected with no acceptance path —
        # only possible if pending was empty from the start, but defensive.
        raise RuntimeError("No intervals were accepted by adaptive_quadrature.")

    all_t_left = torch.cat(accepted_t_left)
    all_t_right = torch.cat(accepted_t_right)
    all_contrib = torch.cat(accepted_contrib, dim=0)
    all_err = torch.cat(accepted_err, dim=0)
    all_y = torch.cat(accepted_y, dim=0)
    all_t_eval = torch.cat(accepted_t_eval, dim=0)

    sort_idx = all_t_left.argsort()
    all_t_left = all_t_left[sort_idx]
    all_t_right = all_t_right[sort_idx]
    all_contrib = all_contrib[sort_idx]
    all_err = all_err[sort_idx]
    all_y = all_y[sort_idx]
    all_t_eval = all_t_eval[sort_idx]

    integral = all_contrib.sum(dim=0)
    integral_error_total = all_err.sum(dim=0)
    h_out = all_t_right - all_t_left
    t_optimal = torch.cat([all_t_left, all_t_right[-1:]])

    final_denom = atol + rtol * integral.abs()
    final_ratio_per_d = all_err.abs() / final_denom
    final_error_ratios = torch.sqrt((final_ratio_per_d**2).mean(dim=-1))

    logger.debug(
        "adaptive_quadrature: %d iterations, %d evaluations, %d intervals%s",
        n_iter,
        n_evaluations,
        all_t_left.numel(),
        " (partial)" if forced_partial else "",
    )

    return IntegralOutput(
        integral=integral,
        method=method_obj.name,
        t_init=t_init_t,
        t_final=t_final_t,
        t=all_t_eval,
        y=all_y,
        h=h_out,
        sum_intervals=all_contrib,
        sum_interval_errors=all_err,
        integral_error=integral_error_total,
        error_ratios=final_error_ratios,
        t_optimal=t_optimal,
        n_iterations=n_iter,
        n_evaluations=n_evaluations,
    )


def fixed_quadrature(
    f: Callable[[torch.Tensor], torch.Tensor],
    t_init: float | torch.Tensor,
    t_final: float | torch.Tensor,
    *,
    method: str = "gl15",
    max_batch: int | None = None,
    total_mem_usage: float | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> IntegralOutput:
    """Non-adaptive Gauss-Legendre quadrature on ``[t_init, t_final]``.

    Applies a single ``n``-point Gauss-Legendre rule to the full domain
    (no subintervals, no refinement). Useful as a baseline against
    :func:`adaptive_quadrature`, and when the integrand is known to be
    smooth enough that a fixed rule of high enough degree converges.

    Args:
        f: Integrand ``f: Tensor[N] -> Tensor[N, D]``.
        t_init: Lower integration bound.
        t_final: Upper integration bound.
        method: Non-adaptive rule name ``"gl<n>"`` for any positive ``n``.
        max_batch: Maximum evaluations per ``f`` call. The rule's K nodes
            are chunked across multiple ``f`` calls if K exceeds this.
            ``None`` means no chunking unless ``total_mem_usage`` is set.
        total_mem_usage: Fraction of currently-free GPU memory
            (``(0, 1]``) the integrator may consume. Triggers automatic
            ``max_batch`` sizing via a probe of ``f``'s per-evaluation
            cost. Ignored on CPU. ``max_batch`` overrides this if both
            are set.
        device: Device for internal tensors.
        dtype: Floating-point dtype.

    Returns:
        :class:`IntegralOutput`. ``integral_error``, ``sum_interval_errors``,
        ``error_ratios``, and ``t_optimal`` are ``None`` (no error estimate
        without an embedded rule).
    """
    device = resolve_device(device)
    t_init_t = normalize_bound(t_init, device, dtype, "t_init")
    t_final_t = normalize_bound(t_final, device, dtype, "t_final")
    method_obj = get_method(method, device, dtype)
    if method_obj.is_adaptive:
        raise ValueError(
            f"Method {method!r} is adaptive. Use a 'gl*' method here, or call "
            "adaptive_quadrature for adaptive 'gk*' rules."
        )
    nodes = method_obj.nodes
    weights = method_obj.weights

    if max_batch is None and total_mem_usage is not None:
        max_batch = estimate_max_batch(f, t_init_t, device, total_mem_usage)

    h_half = (t_final_t - t_init_t) / 2
    t_mid = (t_final_t + t_init_t) / 2
    t_eval = h_half * nodes + t_mid  # [K]

    y = evaluate_chunked(f, t_eval, max_batch)  # [K, D]
    if y.dim() != 2 or y.shape[0] != t_eval.numel():
        raise ValueError(
            "Integrand f must return shape [N, D] matching the input length; "
            f"got input shape [{t_eval.numel()}] and output shape {tuple(y.shape)}."
        )
    integral = h_half * (weights.unsqueeze(-1) * y).sum(dim=0)  # [D]
    h_full = t_final_t - t_init_t

    return IntegralOutput(
        integral=integral,
        method=method_obj.name,
        t_init=t_init_t,
        t_final=t_final_t,
        t=t_eval.unsqueeze(0),
        y=y.unsqueeze(0),
        h=h_full.unsqueeze(0),
        sum_intervals=integral.unsqueeze(0),
        n_iterations=0,
        n_evaluations=t_eval.numel(),
    )
