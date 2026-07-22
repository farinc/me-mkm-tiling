"""
Committor (splitting probability, p_fold) in TT format.

TT analog of me_mkm.sparse.committor. The committor solves the backward
equation on the interior states,

    (W^T q)_j = 0   (j not in A, B),   q|_A = 0,  q|_B = 1,

which as a full-space linear system is

    M q = b,   M = P_I W^T + P_A + P_B,   b = 1_B,

where P_A, P_B, P_I are the diagonal projectors onto basins A, B and the
interior, and 1_B is the indicator vector of B. Interior rows carry W^T;
basin rows are the identity, pinning q to its boundary value.
"""

import numpy as np
import torchtt
from torchtt import TT

from me_mkm.tt.convert import rank1_operator, rank1_vector


def _diag_projector(l: int, n: int, site_indicator) -> TT:
    """Rank-1 diagonal MPO projecting onto a product basin: core p is
    diag(site_indicator[p]), a length-n 0/1 vector marking which species keep
    site p in the basin. The product is 1 exactly on states where every site
    satisfies its indicator."""
    return rank1_operator(
        l, n, {p: np.diag(np.asarray(site_indicator[p], dtype=float)) for p in range(l)}
    )


def committor_tt(
    W_tt: TT,
    siteA,
    siteB,
    max_rank: int = 50,
    threshold: float = 1e-12,
    repeats: int = 30,
) -> TT:
    """Committor MPS for PRODUCT-FORM basins.

    W_tt         : generator MPO (me_mkm.tt.build_W_tt), column convention.
    siteA, siteB : per-site indicator vectors (indexable by site 0..l-1, each a
        length-n 0/1 vector). The basin is the product over sites -- a state is
        in the basin iff every site's species is marked by that site's vector.
        E.g. CO-poisoned B = all-CO: siteB[p] = e_CO for all p; reactive
        A = no-CO: siteA[p] = ones - e_CO for all p.

    Returns q_tt (committor as an MPS). Densify with convert.tt_to_dense to
    compare against the dense solve; check quality with committor_tt_residual.
    """
    l = len(W_tt.N)
    n = W_tt.N[0]
    P_A = _diag_projector(l, n, siteA)
    P_B = _diag_projector(l, n, siteB)
    identity = rank1_operator(l, n, {})  # all-identity cores
    P_I = identity + P_A * (-1.0) + P_B * (-1.0)  # interior projector I - P_A - P_B

    M = (P_I @ W_tt.t() + P_A + P_B).round(eps=threshold)
    b = rank1_vector(l, n, {p: np.asarray(siteB[p], dtype=float) for p in range(l)})

    q = torchtt.solvers.amen_solve(
        M, b, x0=b, nswp=repeats, eps=threshold, rmax=max_rank
    )
    return q


def committor_tt_residual(W_tt: TT, q_tt: TT, siteA, siteB) -> float:
    """Relative residual ||M q - b|| / ||b|| of a TT committor solution, so
    callers can check AMEn convergence without densifying."""
    l = len(W_tt.N)
    n = W_tt.N[0]
    P_A = _diag_projector(l, n, siteA)
    P_B = _diag_projector(l, n, siteB)
    identity = rank1_operator(l, n, {})
    P_I = identity + P_A * (-1.0) + P_B * (-1.0)
    M = P_I @ W_tt.t() + P_A + P_B
    b = rank1_vector(l, n, {p: np.asarray(siteB[p], dtype=float) for p in range(l)})
    return float((M @ q_tt - b).norm() / b.norm())
