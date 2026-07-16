"""
Solve the master equation for its stationary distribution in TT format.

No time component: the stationary state Theta (W Theta = 0, <1,Theta> = 1) is
found by a single regularized linear solve, the TT analog of the dense
last-row-normalization path in me_mkm.sparse.steady_state.

The method involves solving A Theta = c*u with A = W + c*|u><1|, where u is
the uniform product state and <1| the all-ones covector. Because W is a
column-stochastic generator (<1|W = 0) and the rank-1 grounding term shifts the
zero eigenvalue to c, A is nonsingular and its unique solution is *already*
normalized (<1,Theta> = 1). W is pre-normalized by its Frobenius norm so the
grounding strength c is scale-invariant (default c = 1) regardless of how large
the bare rates are. The same A drives the analytic dTheta/dbeta solve.
"""

import dataclasses

import numpy as np
from scikit_tt.solvers import sle
from scikit_tt.tensor_train import TT

from me_mkm.tt.convert import ones_tt, rank1_operator, rank1_vector, tt_inner


@dataclasses.dataclass
class TTSolveInfo:
    """Diagnostics from a TT stationary solve."""

    residual: float  # ||W @ theta|| / ||theta||
    ranks: list  # final TT ranks of theta
    n_sweeps: int
    c: float = 1.0


def steady_state_residual(W_tt: TT, theta_tt: TT) -> float:
    """Relative stationary residual ||W @ theta|| / ||theta|| (2-norm)."""
    return float((W_tt @ theta_tt).norm() / theta_tt.norm())


def _uniform_state(l: int, n: int) -> TT:
    """Uniform product distribution u (per-site ones/n); sums to 1."""
    return rank1_vector(l, n, {p: np.ones(n) / n for p in range(l)})


def _grounding_op(l: int, n: int) -> TT:
    """Rank-1 MPO |u><1| for the uniform u: core_p = (1/n)*ones(n,n), so the
    product is (1/n^l)*ones -- exactly u outer <1|."""
    return rank1_operator(l, n, {p: np.full((n, n), 1.0 / n) for p in range(l)})


def solve_steady_state_tt(
    W_tt: TT,
    theta0: TT = None,
    c: float = 1.0,
    max_rank: int = 50,
    threshold: float = 1e-12,
    repeats: int = 20,
):
    """Stationary distribution of the generator MPO W_tt, by the solve
    A Theta = c*u with A = W_norm + c*|u><1| (see the module docstring).

    Returns (theta_tt, TTSolveInfo); check info.residual for solve quality.
    theta_tt is probability-normalized (<1,theta> = 1). Warm-start with theta0
    (e.g. a neighboring sweep point or convert.product_state_tt) to cut sweeps
    near a transition. `c` sets the grounding strength on the norm-normalized W.
    """
    l = W_tt.order
    n = W_tt.row_dims[0]

    wn = W_tt.norm()
    Wn = W_tt * (1.0 / wn) if wn > 0 else W_tt
    u = _uniform_state(l, n)

    # MALS adapts rank by SVD-splitting merged core pairs, so a rank-1 uniform
    # start is fine -- it grows as needed up to max_rank.
    A = (Wn + _grounding_op(l, n) * c).ortho(threshold=threshold)
    rhs = u * c
    theta0 = u if theta0 is None else theta0
    theta = sle.mals(
        A, theta0, rhs, repeats=repeats, threshold=threshold, max_rank=max_rank
    )

    # The grounded solution is normalized by construction; renormalize anyway to
    # absorb solver error so <1,theta> == 1 exactly.
    theta = theta * (1.0 / tt_inner(ones_tt(l, n), theta))

    info = TTSolveInfo(
        residual=steady_state_residual(W_tt, theta),
        ranks=list(theta.ranks),
        n_sweeps=repeats,
        c=c,
    )
    return theta, info


def steady_state_derivative_tt(
    W_tt: TT,
    dW_tt: TT,
    theta_tt: TT,
    c: float = 1.0,
    max_rank: int = 50,
    threshold: float = 1e-12,
    repeats: int = 20,
) -> TT:
    """Analytic dTheta/dbeta, mirroring steady_state.steady_state_derivative.

    Differentiating the grounded system A Theta = c*u (constant RHS) gives
        A (dTheta) = -(dW/dbeta) @ Theta,
    solved with the same grounded operator A. The gauge <1, dTheta> = 0 is
    automatic: <1|A = c<1| and the RHS -(dW)@Theta has zero column sums
    (<1|dW = 0), so the solve returns the correctly gauged derivative."""
    l = W_tt.order
    n = W_tt.row_dims[0]
    wn = W_tt.norm()
    Wn = W_tt * (1.0 / wn) if wn > 0 else W_tt
    A = (Wn + _grounding_op(l, n) * c).ortho(threshold=threshold)
    # RHS uses the SAME normalization as A (W was scaled by 1/wn).
    rhs = (dW_tt @ theta_tt) * (-1.0 / wn)
    rhs = rhs.ortho(threshold=threshold)
    dtheta0 = (theta_tt * 0.0 + rhs).ortho(threshold=threshold)
    return sle.mals(
        A, dtheta0, rhs, repeats=repeats, threshold=threshold, max_rank=max_rank
    )
