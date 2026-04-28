"""torchpathint — parallel adaptive quadrature for definite path integrals."""

from __future__ import annotations

from .base import IntegralOutput, normalize_bound, resolve_device
from .methods import METHOD_NAMES_ADAPTIVE, Method, get_method
from .quadrature import adaptive_quadrature, evaluate_chunked, fixed_quadrature

__version__ = "0.0.1"

__all__ = [
    "METHOD_NAMES_ADAPTIVE",
    "IntegralOutput",
    "Method",
    "__version__",
    "adaptive_quadrature",
    "evaluate_chunked",
    "fixed_quadrature",
    "get_method",
    "normalize_bound",
    "resolve_device",
]
