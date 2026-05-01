"""Core data structures and utilities for torchpathint."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class IntegralOutput:
    """Result of a definite-integral computation.

    The integrators always populate the cheap fields (``integral``, ``method``,
    ``t_init``, ``t_final``, ``n_iterations``, ``n_evaluations``). The
    per-interval diagnostic fields (``t``, ``y``, ``h``, ``interval_integrals``,
    ``interval_errors``, ``integral_error``, ``error_ratios``) are populated
    only when the integrator is called with ``full_output=True``; otherwise
    they are ``None``.

    Attributes:
        integral: Computed integral value. Shape: [D].
        method: Name of the quadrature rule used (e.g. 'gk21', 'gl15').
        t_init: Lower integration bound, as a 0-d tensor.
        t_final: Upper integration bound, as a 0-d tensor.
        t: Quadrature points actually evaluated, grouped by interval.
            Shape: [N, K] where N = number of (sub)intervals, K = nodes per rule.
            None unless ``full_output=True``.
        y: Integrand values at t. Shape: [N, K, D].
            None unless ``full_output=True``.
        h: Width of each (sub)interval, t_right - t_left. Shape: [N].
            None unless ``full_output=True``.
        interval_integrals: Per-interval integral contributions. Shape: [N, D].
            None unless ``full_output=True``.
        interval_errors: Per-interval error estimates from the embedded rule.
            Shape: [N, D]. None for non-adaptive methods or when
            ``full_output=False``.
        integral_error: Estimated total error. Shape: [D]. None for
            non-adaptive methods or when ``full_output=False``.
        error_ratios: Per-interval error / tolerance. Shape: [N]. None for
            non-adaptive methods or when ``full_output=False``.
        n_iterations: Adaptive refinement iterations performed (0 for non-adaptive).
        n_evaluations: Total number of integrand evaluations.
    """

    integral: torch.Tensor
    method: str
    t_init: torch.Tensor
    t_final: torch.Tensor
    t: torch.Tensor | None = None
    y: torch.Tensor | None = None
    h: torch.Tensor | None = None
    interval_integrals: torch.Tensor | None = None
    interval_errors: torch.Tensor | None = None
    integral_error: torch.Tensor | None = None
    error_ratios: torch.Tensor | None = None
    n_iterations: int = 0
    n_evaluations: int = 0


def normalize_bound(
    bound: float | torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """Coerce an integration bound to a 0-d tensor on the target device/dtype.

    Accepts a Python ``float``/``int`` or a 0-d ``torch.Tensor``. A 0-d tensor
    that already has ``requires_grad=True`` is preserved (the result is on the
    same autograd graph), enabling gradients with respect to integration limits.

    Rejects 1-d or higher tensors — the integrator's time axis is scalar, so
    multi-element bound tensors are almost always a shape bug carried over from
    older APIs.
    """
    if isinstance(bound, torch.Tensor):
        if bound.dim() != 0:
            raise ValueError(
                f"{name} must be a scalar (Python number or 0-d tensor); "
                f"got tensor of shape {tuple(bound.shape)}."
            )
        return bound.to(device=device, dtype=dtype)
    if isinstance(bound, (int, float)):
        return torch.tensor(bound, device=device, dtype=dtype)
    raise TypeError(
        f"{name} must be a float, int, or 0-d torch.Tensor; got {type(bound).__name__}."
    )


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Default to CUDA if available, else CPU."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
