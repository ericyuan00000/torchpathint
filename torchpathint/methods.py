"""Quadrature rules for torchpathint.

Two families of rules on ``[-1, 1]``:

- ``gk<m>`` — adaptive Gauss-Kronrod rules. Provide a primary Kronrod
  estimate (exact for polynomials of degree at least ``3n+1``) and an
  embedded ``n``-point Gauss estimate (exact for degree ``2n-1``) computed
  from the same ``2n+1`` integrand evaluations. Supported sizes: 15, 21, 31.
- ``gl<n>`` — non-adaptive ``n``-point Gauss-Legendre rules (exact for
  degree ``2n-1``). Any positive integer ``n`` is accepted; nodes/weights
  are computed via ``numpy.polynomial.legendre.leggauss``.

The :func:`get_method` accessor returns a :class:`Method` ready to use on a
caller-specified device/dtype. Mapping from ``[-1, 1]`` to ``[t_init, t_final]``
is the integrator's job, not this module's.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch


@dataclass(frozen=True)
class Method:
    """A quadrature rule on ``[-1, 1]``.

    Attributes:
        name: Rule identifier (e.g. ``"gk21"``, ``"gl15"``).
        nodes: Quadrature node positions on ``[-1, 1]``, sorted ascending.
            Shape: ``[K]``.
        weights: Primary-rule weights aligned with ``nodes``. Shape: ``[K]``.
        weights_error: ``primary - embedded`` weights for adaptive rules
            (Kronrod minus padded Gauss). ``None`` for non-adaptive rules.
            Shape: ``[K]`` when present.
        is_adaptive: ``True`` for ``gk*`` rules; ``False`` for ``gl*``.
        degree: Exactness degree — the rule integrates polynomials of degree
            up to this value exactly (modulo floating-point error).
    """

    name: str
    nodes: torch.Tensor
    weights: torch.Tensor
    weights_error: torch.Tensor | None
    is_adaptive: bool
    degree: int


# === Gauss-Kronrod tables (QUADPACK source) =================================
# Source: QUADPACK Fortran files dqk15.f / dqk21.f / dqk31.f (Piessens, de
# Doncker-Kapenga, Ueberhuber, Kahaner, 1983; public domain).
#
# QUADPACK stores positive-half nodes in *decreasing* magnitude, ending with
# the central node 0. Among those, the Gauss nodes sit at odd indices
# (1, 3, ...); the trailing 0 is also a Gauss node iff n is odd. We expand
# to a full ascending [-1, 1] array at use time.

_GK15_POS_NODES = np.array(
    [
        0.991455371120812639206854697526329,
        0.949107912342758524526189684047851,  # G
        0.864864423359769072789712788640926,
        0.741531185599394439863864773280788,  # G
        0.586087235467691130294144838258730,
        0.405845151377397166906606412076961,  # G
        0.207784955007898467600689403773245,
        0.0,  # G
    ],
    dtype=np.float64,
)
_GK15_POS_W_KRONROD = np.array(
    [
        0.022935322010529224963732008058970,
        0.063092092629978553290700663189204,
        0.104790010322250183839876322541518,
        0.140653259715525918745189590510238,
        0.169004726639267902826583426598550,
        0.190350578064785409913256402421014,
        0.204432940075298892414161999234649,
        0.209482141084727828012999174891714,
    ],
    dtype=np.float64,
)
# Listed in the same |x|-decreasing order as positive Gauss nodes appear,
# trailing entry = weight at 0 (only present when n is odd).
_GK15_POS_W_GAUSS = np.array(
    [
        0.129484966168869693270611432679082,  # @ ±0.949107... (pos-idx 1)
        0.279705391489276667901467771423780,  # @ ±0.741531... (pos-idx 3)
        0.381830050505118944950369775488975,  # @ ±0.405845... (pos-idx 5)
        0.417959183673469387755102040816327,  # @  0.0          (pos-idx 7)
    ],
    dtype=np.float64,
)


_GK21_POS_NODES = np.array(
    [
        0.995657163025808080735527280689003,
        0.973906528517171720077964012084452,  # G
        0.930157491355708226001207180059508,
        0.865063366688984510732096688423493,  # G
        0.780817726586416897063717578345042,
        0.679409568299024406234327365114874,  # G
        0.562757134668604683339000099272694,
        0.433395394129247190799265943165784,  # G
        0.294392862701460198131126603103866,
        0.148874338981631210884826001129720,  # G
        0.0,
    ],
    dtype=np.float64,
)
_GK21_POS_W_KRONROD = np.array(
    [
        0.011694638867371874278064396062192,
        0.032558162307964727478818972459390,
        0.054755896574351996031381300244582,
        0.075039674810919952767043140916190,
        0.093125454583697605535065465083366,
        0.109387158802297641899210590325805,
        0.123491976262065851077958109831074,
        0.134709217311473325928054001771707,
        0.142775938577060080797094273138717,
        0.147739104901338491374841515972068,
        0.149445554002916905664936468389821,
    ],
    dtype=np.float64,
)
_GK21_POS_W_GAUSS = np.array(
    [
        0.066671344308688137593568809893332,  # @ ±0.973906... (pos-idx 1)
        0.149451349150580593145776339657697,  # @ ±0.865063... (pos-idx 3)
        0.219086362515982043995534934228163,  # @ ±0.679409... (pos-idx 5)
        0.269266719309996355091226921569469,  # @ ±0.433395... (pos-idx 7)
        0.295524224714752870173892994651338,  # @ ±0.148874... (pos-idx 9)
    ],
    dtype=np.float64,
)


_GK31_POS_NODES = np.array(
    [
        0.998002298693397060285172840152271,
        0.987992518020485428489565718586613,  # G
        0.967739075679139134257347978784337,
        0.937273392400705904307758947710209,  # G
        0.897264532344081900882509656454496,
        0.848206583410427216200648320774217,  # G
        0.790418501442465932967649294817947,
        0.724417731360170047416186054613938,  # G
        0.650996741297416970533735895313275,
        0.570972172608538847537226737253911,  # G
        0.485081863640239680693655740232351,
        0.394151347077563369897207370981045,  # G
        0.299180007153168812166780024266389,
        0.201194093997434522300628303394596,  # G
        0.101142066918717499027074231447392,
        0.0,  # G
    ],
    dtype=np.float64,
)
_GK31_POS_W_KRONROD = np.array(
    [
        0.005377479872923348987792051430128,
        0.015007947329316122538374763075807,
        0.025460847326715320186874001019653,
        0.035346360791375846222037948478360,
        0.044589751324764876608227299373280,
        0.053481524690928087265343147239430,
        0.062009567800670640285139230960803,
        0.069854121318728258709520077099147,
        0.076849680757720378894432777482659,
        0.083080502823133021038289247286104,
        0.088564443056211770647275443693774,
        0.093126598170825321225486872747346,
        0.096642726983623678505179907627589,
        0.099173598721791959332393173484603,
        0.100769845523875595044946662617570,
        0.101330007014791549017374792767493,
    ],
    dtype=np.float64,
)
_GK31_POS_W_GAUSS = np.array(
    [
        0.030753241996117268354628393577204,  # @ ±0.987992... (pos-idx 1)
        0.070366047488108124709267416450667,  # @ ±0.937273... (pos-idx 3)
        0.107159220467171935011869546685869,  # @ ±0.848206... (pos-idx 5)
        0.139570677926154314447804794511028,  # @ ±0.724417... (pos-idx 7)
        0.166269205816993933553200860481209,  # @ ±0.570972... (pos-idx 9)
        0.186161000015562211026800561866423,  # @ ±0.394151... (pos-idx 11)
        0.198431485327111576456118326443839,  # @ ±0.201194... (pos-idx 13)
        0.202578241925561272880620199967519,  # @  0.0          (pos-idx 15)
    ],
    dtype=np.float64,
)


_GK_TABLES: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {
    "gk15": (_GK15_POS_NODES, _GK15_POS_W_KRONROD, _GK15_POS_W_GAUSS, 7),
    "gk21": (_GK21_POS_NODES, _GK21_POS_W_KRONROD, _GK21_POS_W_GAUSS, 10),
    "gk31": (_GK31_POS_NODES, _GK31_POS_W_KRONROD, _GK31_POS_W_GAUSS, 15),
}


def _expand_gk(
    pos_nodes: np.ndarray,
    pos_w_kronrod: np.ndarray,
    pos_w_gauss: np.ndarray,
    n_gauss: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand QUADPACK positive-half tables to full symmetric arrays.

    Returns:
        nodes_full: shape ``[2n+1]``, sorted ascending on ``[-1, 1]``.
        w_kronrod_full: shape ``[2n+1]``, Kronrod weights aligned with nodes.
        w_gauss_full: shape ``[2n+1]``, ``n``-point Gauss weights placed at
            the Gauss-node positions among the Kronrod nodes; zero elsewhere.
    """
    n_pos = len(pos_nodes)
    assert 2 * n_pos - 1 == 2 * n_gauss + 1

    # nodes_full = [neg-mirror (excluding center)] + [center] + [positive-asc]
    neg = -pos_nodes[:-1]  # already in ascending order: -0.99, -0.95, ...
    center = pos_nodes[-1:]  # = [0.0]
    pos_asc = pos_nodes[:-1][::-1]  # 0.21, 0.40, ..., 0.99
    nodes_full = np.concatenate([neg, center, pos_asc])

    w_kronrod_neg = pos_w_kronrod[:-1]
    w_kronrod_center = pos_w_kronrod[-1:]
    w_kronrod_pos = pos_w_kronrod[:-1][::-1]
    w_kronrod_full = np.concatenate([w_kronrod_neg, w_kronrod_center, w_kronrod_pos])

    # Gauss weights: scatter pos_w_gauss into the Gauss-node positions of
    # the full array. In QUADPACK positive-half order (decreasing |x|), the
    # j-th positive Gauss node sits at index 1 + 2*j; index n_pos-1 (the
    # 0-node) is also a Gauss node iff n_gauss is odd.
    n_positive_gauss = n_gauss // 2
    has_zero_gauss = n_gauss % 2 == 1
    expected_g_len = n_positive_gauss + (1 if has_zero_gauss else 0)
    assert len(pos_w_gauss) == expected_g_len

    w_gauss_full = np.zeros_like(nodes_full)
    for j in range(n_positive_gauss):
        idx_pos = 1 + 2 * j  # QUADPACK positive-half index
        full_neg_idx = idx_pos
        full_pos_idx = 2 * n_pos - 2 - idx_pos
        w_gauss_full[full_neg_idx] = pos_w_gauss[j]
        w_gauss_full[full_pos_idx] = pos_w_gauss[j]
    if has_zero_gauss:
        w_gauss_full[n_pos - 1] = pos_w_gauss[-1]

    return nodes_full, w_kronrod_full, w_gauss_full


def _validate_gk_table(name: str) -> None:
    """Sanity-check a hardcoded GK table.

    Verifies that the Kronrod weights integrate ``x^k`` exactly for
    ``k = 0 ... 3n+1``, and the embedded Gauss weights (placed at the Gauss
    positions among the Kronrod nodes) integrate ``x^k`` exactly for
    ``k = 0 ... 2n-1``. A transcription error in any digit will fail this.
    """
    pos_nodes, pos_w_kronrod, pos_w_gauss, n_gauss = _GK_TABLES[name]
    nodes, w_kronrod, w_gauss = _expand_gk(
        pos_nodes, pos_w_kronrod, pos_w_gauss, n_gauss
    )

    deg_K = 3 * n_gauss + 1
    deg_G = 2 * n_gauss - 1
    tol = 1e-12

    for k in range(deg_K + 1):
        true = 0.0 if (k % 2 == 1) else 2.0 / (k + 1)
        approx = float(np.sum(nodes**k * w_kronrod))
        if abs(approx - true) > tol:
            raise AssertionError(
                f"{name}: Kronrod weights fail to integrate x^{k} "
                f"(got {approx:.16e}, want {true:.16e}, "
                f"err {abs(approx - true):.2e})"
            )

    for k in range(deg_G + 1):
        true = 0.0 if (k % 2 == 1) else 2.0 / (k + 1)
        approx = float(np.sum(nodes**k * w_gauss))
        if abs(approx - true) > tol:
            raise AssertionError(
                f"{name}: Embedded Gauss weights fail to integrate x^{k} "
                f"(got {approx:.16e}, want {true:.16e}, "
                f"err {abs(approx - true):.2e})"
            )


# Validate at module import; transcription errors fail loudly here.
for _name in _GK_TABLES:
    _validate_gk_table(_name)


@lru_cache(maxsize=64)
def _gauss_legendre_arrays(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute n-point Gauss-Legendre nodes/weights on ``[-1, 1]`` (numpy)."""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return (
        np.ascontiguousarray(nodes, dtype=np.float64),
        np.ascontiguousarray(weights, dtype=np.float64),
    )


def _build_gk_method(
    name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Method:
    pos_nodes, pos_w_kronrod, pos_w_gauss, n_gauss = _GK_TABLES[name]
    nodes, w_kronrod, w_gauss = _expand_gk(
        pos_nodes, pos_w_kronrod, pos_w_gauss, n_gauss
    )
    w_error = w_kronrod - w_gauss

    return Method(
        name=name,
        nodes=torch.from_numpy(nodes).to(device=device, dtype=dtype),
        weights=torch.from_numpy(w_kronrod).to(device=device, dtype=dtype),
        weights_error=torch.from_numpy(w_error).to(device=device, dtype=dtype),
        is_adaptive=True,
        degree=3 * n_gauss + 1,
    )


def _build_gl_method(n: int, device: torch.device, dtype: torch.dtype) -> Method:
    if n < 1:
        raise ValueError(f"Gauss-Legendre order must be >= 1; got {n}.")
    nodes_np, weights_np = _gauss_legendre_arrays(n)
    return Method(
        name=f"gl{n}",
        nodes=torch.from_numpy(nodes_np).to(device=device, dtype=dtype),
        weights=torch.from_numpy(weights_np).to(device=device, dtype=dtype),
        weights_error=None,
        is_adaptive=False,
        degree=2 * n - 1,
    )


_GL_NAME_RE = re.compile(r"^gl(\d+)$")


def get_method(name: str, device: torch.device, dtype: torch.dtype) -> Method:
    """Look up or compute a quadrature rule on ``[-1, 1]``.

    Args:
        name: Rule identifier — ``"gk15"`` / ``"gk21"`` / ``"gk31"`` for
            adaptive Gauss-Kronrod, or ``"gl<n>"`` for non-adaptive
            ``n``-point Gauss-Legendre with any positive integer ``n``.
        device: Device for the returned tensors.
        dtype: Dtype for the returned tensors.

    Returns:
        A :class:`Method` with ``nodes`` and ``weights`` on the requested
        device and dtype.
    """
    name = name.lower()
    if name in _GK_TABLES:
        return _build_gk_method(name, device, dtype)
    m = _GL_NAME_RE.match(name)
    if m:
        return _build_gl_method(int(m.group(1)), device, dtype)
    raise ValueError(
        f"Unknown method {name!r}. Expected one of {sorted(_GK_TABLES.keys())} "
        "or 'gl<n>' for any positive integer n."
    )


METHOD_NAMES_ADAPTIVE: tuple[str, ...] = tuple(_GK_TABLES.keys())
