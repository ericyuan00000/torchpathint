"""Public top-level API: ``path_integral`` dispatches to the right engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .quadrature import adaptive_quadrature, fixed_quadrature

if TYPE_CHECKING:
    from collections.abc import Callable

    from .base import IntegralOutput


def path_integral(
    f: Callable[[torch.Tensor], torch.Tensor],
    t_init: float | torch.Tensor,
    t_final: float | torch.Tensor,
    *,
    method: str = "gk21",
    atol: float = 1e-5,
    rtol: float = 1e-5,
    max_batch: int | None = None,
    memory_fraction: float | None = None,
    max_iter: int = 50,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
    full_output: bool = False,
) -> IntegralOutput:
    """Definite integral ``∫_{t_init}^{t_final} f(t) dt`` via quadrature.

    Dispatches to :func:`adaptive_quadrature` for ``"gk*"`` (adaptive
    Gauss-Kronrod) methods and :func:`fixed_quadrature` for ``"gl*"``
    (non-adaptive Gauss-Legendre) methods.

    Args:
        f: Integrand ``f: Tensor[N] -> Tensor[N, D]``.
        t_init: Lower integration bound (Python scalar or 0-d tensor).
        t_final: Upper integration bound (same).
        method: Rule name. ``"gk15"`` / ``"gk21"`` / ``"gk31"`` select
            adaptive Gauss-Kronrod; ``"gl<n>"`` selects ``n``-point
            non-adaptive Gauss-Legendre.
        atol: Absolute tolerance (adaptive only; ignored for ``gl*``).
        rtol: Relative tolerance (adaptive only; ignored for ``gl*``).
        max_batch: Initial cap on integrand evaluations per ``f`` call.
            Applies to both adaptive and fixed. ``None`` (default) starts
            unchunked. CUDA OOM halves this cap automatically and the
            learned size persists across iterations.
        memory_fraction: Deprecated. Previously triggered an upfront
            memory probe; now ignored — chunk sizing is OOM-driven, so
            no probe is needed.
        max_iter: Maximum refinement iterations (adaptive only).
        device: Device for internal tensors. Defaults to CUDA if available.
        dtype: Floating-point dtype. Defaults to ``torch.float64``.
        full_output: If ``True``, populate the per-interval diagnostic
            fields on the returned :class:`IntegralOutput`. Default
            ``False`` returns only ``integral`` plus cheap metadata
            (``method``, bounds, ``n_iterations``, ``n_evaluations``); the
            diagnostic fields (``t``, ``y``, ``h``, ``interval_integrals``,
            ``interval_errors``, ``integral_error``, ``error_ratios``) are
            ``None``. The default mode also avoids retaining per-interval
            evaluations across adaptive iterations.

    Returns:
        :class:`IntegralOutput`. With ``full_output=True``: full diagnostics
        (error fields are still ``None`` for ``gl*`` methods). With
        ``full_output=False`` (default): only ``integral`` and cheap
        metadata; all per-interval diagnostic fields are ``None``.

    Raises:
        ValueError: If ``method`` is neither a known ``gk*`` nor a ``gl<n>``
            rule.
    """
    name = method.lower()
    if name.startswith("gk"):
        return adaptive_quadrature(
            f,
            t_init,
            t_final,
            method=method,
            atol=atol,
            rtol=rtol,
            max_batch=max_batch,
            memory_fraction=memory_fraction,
            max_iter=max_iter,
            device=device,
            dtype=dtype,
            full_output=full_output,
        )
    if name.startswith("gl"):
        return fixed_quadrature(
            f,
            t_init,
            t_final,
            method=method,
            max_batch=max_batch,
            memory_fraction=memory_fraction,
            device=device,
            dtype=dtype,
            full_output=full_output,
        )
    raise ValueError(
        f"Unknown method {method!r}. Use 'gk15' / 'gk21' / 'gk31' for adaptive "
        "Gauss-Kronrod, or 'gl<n>' for non-adaptive n-point Gauss-Legendre."
    )
