"""torchpathint — parallel adaptive quadrature for definite path integrals."""

from __future__ import annotations

from .base import IntegralOutput, normalize_bound, resolve_device

__version__ = "0.0.1"

__all__ = [
    "IntegralOutput",
    "__version__",
    "normalize_bound",
    "resolve_device",
]
