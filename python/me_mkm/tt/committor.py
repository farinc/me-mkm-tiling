"""
Committor in TT format.

TT analog of me_mkm.sparse.committor. The committor solves the backward
equation on the interior states,

    (W^T q)_j = 0   (j not in A, B),   q|_A = 0,  q|_B = 1,

which as a full-space linear system is

    M q = b,   M = P_I W^T + P_A + P_B,   b = 1_B,

where P_A, P_B, P_I are the diagonal projectors onto basins A, B and the
interior, and 1_B is the indicator vector of B. Interior rows carry W^T;
basin rows are the identity, pinning q to its boundary value.

Basins A/B are diagonal projector MPOs. Per-site product basins build a
rank-1 one directly while threshold_projector_tt builds one for coverage-
fraction basins.
"""

import numpy as np
from scikit_tt.solvers import sle
from scikit_tt.tensor_train import TT

from me_mkm.tt.convert import ones_tt, rank1_operator, rank1_vector


def _diag_projector(l: int, n: int, site_indicator) -> TT:
    """Rank-1 diagonal MPO projecting onto a product basin: core p is
    diag(site_indicator[p]) with 1 being the case where the indicator
    is true for every site ."""
    return rank1_operator(
        l, n, {p: np.diag(np.asarray(site_indicator[p], dtype=float)) for p in range(l)}
    )


def threshold_projector_tt(
    l: int, n: int, species, k: int, at_least: bool = True
) -> TT:
    """Diagonal projector MPO for a coverage-fraction basin: the count of
    sites carrying `species` is >= k (at_least=True) or <= k (at_least=False).
    Pass as basin_A/basin_B to committor_tt in place of a per-site dict.

    species : an int species code, or a length-n 0/1 indicator vector.
    k       : site-count threshold, 0 <= k <= l (e.g. k = ceil(hi * l) for a
        coverage fraction hi, mirroring microstates.microstate_mask).
    """
    if not (0 <= k <= l):
        raise ValueError(f"k must be in [0, {l}], got {k}")
    if np.ndim(species) == 0:
        match = np.zeros(n)
        match[int(species)] = 1.0
    else:
        match = np.asarray(species, dtype=float)
        if match.shape != (n,):
            raise ValueError(
                f"species indicator must have shape ({n},), got {match.shape}"
            )
    hit, cap = (1.0 - match, l - k) if at_least else (match, k)
    no_hit = 1.0 - hit
    r = cap + 2
    REJECT = cap + 1

    no_hit_block, hit_block = np.diag(no_hit), np.diag(hit)
    core = np.zeros((r, n, n, r))
    for d in range(cap + 1):
        core[d, :, :, d] = no_hit_block  # no hit: count unchanged
    for d in range(cap):
        core[d, :, :, d + 1] = hit_block  # hit below cap: count++
    core[cap, :, :, REJECT] = hit_block  # hit at cap: exceed budget, reject
    core[REJECT, :, :, REJECT] = np.eye(n)  # absorbing, either way

    v_L = np.zeros(r)
    v_L[0] = 1.0
    v_R = np.zeros(r)
    v_R[: cap + 1] = 1.0  # accept every non-REJECT final state

    if l == 1:
        return TT([np.einsum("i,inmj,j->nm", v_L, core, v_R).reshape(1, n, n, 1)])
    first = np.tensordot(v_L, core, axes=(0, 0)).reshape(1, n, n, r)
    last = np.tensordot(core, v_R, axes=(-1, 0)).reshape(r, n, n, 1)
    return TT([first] + [core.copy() for _ in range(l - 2)] + [last])


def _basin_projector(l: int, n: int, basin) -> TT:
    """Normalize a basin spec into a diagonal projector MPO: a per-site
    indicator dict builds a product basin; a TT (e.g. from
    threshold_projector_tt) is used as-is."""
    return basin if isinstance(basin, TT) else _diag_projector(l, n, basin)


def _basin_indicator(l: int, n: int, basin, P_basin: TT) -> TT:
    """The basin's indicator as an MPS. b = 1_B in the RHS. A dict builds it
    directly. For a TT projector, P @ 1 recovers its diagonal exactly."""
    if isinstance(basin, TT):
        return (P_basin @ ones_tt(l, n)).ortho()
    return rank1_vector(l, n, {p: np.asarray(basin[p], dtype=float) for p in range(l)})


def committor_tt(
    W_tt: TT,
    basin_A,
    basin_B,
    max_rank: int = 50,
    threshold: float = 1e-12,
    repeats: int = 30,
) -> TT:
    """Committor MPS: basin_A absorbs at q=0, basin_B at q=1.

    W_tt             : generator MPO (me_mkm.tt.build_W_tt), column convention.
    basin_A, basin_B : each either a per-site indicator dict (product basin:
        indexable by site 0..l-1, each a length-n 0/1 vector; in the basin
        iff every site matches, e.g. all-CO = {p: e_CO for p in range(l)}),
        or a diagonal projector MPO (e.g. threshold_projector_tt, for
        coverage-fraction basins). Must be disjoint (P_A @ P_B = 0).

    Returns q_tt (committor as an MPS). Densify with convert.tt_to_dense to
    compare against the dense solve; check quality with committor_tt_residual.
    """
    l = W_tt.order
    n = W_tt.row_dims[0]
    P_A = _basin_projector(l, n, basin_A)
    P_B = _basin_projector(l, n, basin_B)
    identity = rank1_operator(l, n, {})  # all-identity cores
    P_I = identity + P_A * (-1.0) + P_B * (-1.0)  # interior projector I - P_A - P_B

    M = (P_I @ W_tt.transpose() + P_A + P_B).ortho(threshold=threshold)
    b = _basin_indicator(l, n, basin_B, P_B)

    q = sle.mals(M, b, b, repeats=repeats, threshold=threshold, max_rank=max_rank)
    return q


def committor_tt_residual(W_tt: TT, q_tt: TT, basin_A, basin_B) -> float:
    """Relative residual ||M q - b|| / ||b|| of a TT committor solution, so
    callers can check MALS convergence without densifying. basin_A, basin_B:
    same spec as committor_tt."""
    l = W_tt.order
    n = W_tt.row_dims[0]
    P_A = _basin_projector(l, n, basin_A)
    P_B = _basin_projector(l, n, basin_B)
    identity = rank1_operator(l, n, {})
    P_I = identity + P_A * (-1.0) + P_B * (-1.0)
    M = P_I @ W_tt.transpose() + P_A + P_B
    b = _basin_indicator(l, n, basin_B, P_B)
    return float((M @ q_tt - b).norm() / b.norm())
