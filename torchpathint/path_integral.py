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
        max_batch: Maximum integrand evaluations per ``f`` call. Applies
            to both adaptive and fixed. Overrides ``memory_fraction`` if
            both are set.
        memory_fraction: Fraction of currently-free GPU memory
            (``(0, 1]``) the integrator may consume. When set with
            ``max_batch=None``, ``f`` is benchmarked at probe sizes to
            pick a ``max_batch`` that fits the budget. Ignored on CPU.
        max_iter: Maximum refinement iterations (adaptive only).
        device: Device for internal tensors. Defaults to CUDA if available.
        dtype: Floating-point dtype. Defaults to ``torch.float64``.

    Returns:
        :class:`IntegralOutput`. For ``gl*`` methods, the error fields are
        ``None``.

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
        )
    raise ValueError(
        f"Unknown method {method!r}. Use 'gk15' / 'gk21' / 'gk31' for adaptive "
        "Gauss-Kronrod, or 'gl<n>' for non-adaptive n-point Gauss-Legendre."
    )
