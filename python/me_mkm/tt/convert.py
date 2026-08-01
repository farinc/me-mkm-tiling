"""
TT <-> dense bridges and elementary rank-1 TT constructors.

Conventions (must match the Rust encoding in src/memkm_rs_lib.rs so TT and
dense objects are index-aligned):

- A microstate index is idx = sum_p s_p * n**(l-1-p): site 0 is the MOST
  significant digit. So a dense Theta reshaped C-order to [n]*l has site p on
  axis p, and TT core p corresponds to lattice site p.
- scikit_tt stores each core as an order-4 array (r_left, row_dim, col_dim,
  r_right). Operators (MPOs) have row_dims = col_dims = [n]*l; state vectors
  (MPS) have col_dims = [1]*l.
- TT.matricize() contracts the cores into a full matrix (operator) or vector
  (state) with site 0 outermost -- exactly the encode_state ordering -- so it
  is the direct densify bridge, no reshaping/transposing needed.
"""

import numpy as np
from scikit_tt.tensor_train import TT

from me_mkm._me_mkm import MEMKMBuilder


def rank1_operator(l: int, n: int, factors: dict) -> TT:
    """Rank-1 MPO whose core at site p is factors[p] (an (n, n) matrix), or the
    n x n identity where p is absent. All TT ranks are 1."""
    identity = np.eye(n)
    cores = []
    for p in range(l):
        m = np.asarray(factors.get(p, identity), dtype=float)
        cores.append(m.reshape(1, n, n, 1))
    return TT(cores)


def rank1_vector(l: int, n: int, factors: dict) -> TT:
    """Rank-1 MPS whose core at site p is the length-n vector factors[p], or
    all-ones where absent (col_dims = [1]*l). All TT ranks are 1."""
    ones = np.ones(n)
    cores = []
    for p in range(l):
        v = np.asarray(factors.get(p, ones), dtype=float)
        cores.append(v.reshape(1, n, 1, 1))
    return TT(cores)


def ones_tt(l: int, n: int) -> TT:
    """The all-ones covector <1| as a rank-1 MPS. Contracting it with a state
    MPS gives the sum of all its entries (the probability normalization)."""
    return rank1_vector(l, n, {})


def unit_tt(l: int, n: int, digits) -> TT:
    """Basis state e_idx as a rank-1 MPS, where `digits` is the per-site species
    vector (site 0 first). Its dense form is 1 at encode_state(digits) else 0."""
    factors = {}
    for p, d in enumerate(digits):
        v = np.zeros(n)
        v[int(d)] = 1.0
        factors[p] = v
    return rank1_vector(l, n, factors)


def product_state_tt(builder: MEMKMBuilder, coverage) -> TT:
    """Max-entropy product (independent-site) distribution with prescribed
    marginal coverages, as a rank-1 MPS. TT analog of
    observables.independent_site_distribution: Theta0 = prod_p p[s_p], with
    p[0] the remainder 1 - sum(p[1:]). A natural warm-start initial condition."""
    p = np.array(coverage, dtype=float)
    p[0] = max(0.0, 1.0 - p[1:].sum())
    l, n = builder.l, builder.n_species
    return rank1_vector(l, n, {site: p for site in range(l)})


def vector_outer_product_tt(u_tt: TT, v_tt: TT) -> TT:
    """Diagonal-free rank-(rank(u)*rank(v)) MPO representing |u><v|:
    (u (x) v) @ x = u * <v, x>. Used for exact (Wielandt/Hotelling)
    eigenvalue deflation of non-normal operators, where the deflation term
    for a known (right, left) eigenvector pair must use the true dual, not
    the same vector twice (see tt_spectral_deflation_followup.md).

    Site p's operator core combines u's core (r_u_left, n, 1, r_u_right) and
    v's core (r_v_left, n, 1, r_v_right) into (r_u_left*r_v_left, n, n,
    r_u_right*r_v_right), whose [(a1,a2), s, t, (b1,b2)] entry is
    u_core[a1,s,b1] * v_core[a2,t,b2] -- the standard MPO-from-two-MPS
    outer-product construction."""
    cores = []
    for u_core, v_core in zip(u_tt.cores, v_tt.cores):
        ru_l, n, _, ru_r = u_core.shape
        rv_l, _, _, rv_r = v_core.shape
        u = u_core.reshape(ru_l, n, ru_r)
        v = v_core.reshape(rv_l, n, rv_r)
        block = np.einsum("asb,ctd->acstbd", u, v)
        cores.append(block.reshape(ru_l * rv_l, n, n, ru_r * rv_r))
    return TT(cores)


def tt_to_dense(theta_tt: TT) -> np.ndarray:
    """Dense state vector of length n**l, index-aligned with encode_state
    (site 0 most significant). Only feasible for small l -- validation and
    observable bridging."""
    return theta_tt.matricize()


def mpo_to_dense(W_tt: TT) -> np.ndarray:
    """Dense (n**l, n**l) matrix of a TT operator, rows/cols in encode_state
    order. Validation only."""
    return W_tt.matricize()


def tt_inner(x_tt: TT, y_tt: TT) -> float:
    """Scalar <x, y> = sum_i x_i y_i for two state MPS (via x^T @ y)."""
    return float(np.real(x_tt.transpose() @ y_tt))


def tt_normalize_prob(theta_tt: TT) -> TT:
    """theta / <1, theta>, so its entries sum to 1 (probability normalization).
    Cheap: <1, theta> is a rank-1 contraction and scaling is a scalar multiply."""
    l = theta_tt.order
    n = theta_tt.row_dims[0]
    total = tt_inner(ones_tt(l, n), theta_tt)
    return theta_tt * (1.0 / total)
