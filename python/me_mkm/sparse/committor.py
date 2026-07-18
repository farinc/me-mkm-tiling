"""
Committor probabilties for the ME-MKM generator.
The committor q[i] is the probability of reaching basin B before basin A,
starting from microstate i.

It solves the backward equation on the interior (states in neither basin):

    (W^T q)_j = 0   for j not in A, B,     q|_A = 0,  q|_B = 1,

which reduces to the sparse linear system (Eidelson & Peters 2012 eq. 10;
Berezhkovskii et al.; Noe et al. 2009 PNAS)

    (W^T)_II q_I = -(W^T)_IB 1.

W is the general (non-normalized) generator in this codebase's column convention
NOT the steady-state form whose last row is the normalisation condition.

Basin masks (in_A, in_B) are length-n_states boolean arrays; build them from
coverage level sets with me_mkm.microstates.microstate_mask.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def committor(W, in_A, in_B) -> np.ndarray:
    """Forward committor q over all microstates: q[i] = P(reach B before A | i).

    W         : dynamical generator
    in_A/in_B : length-n boolean microstate masks from microstate_mask; q = 0 on A, q = 1 on B.

    Returns q, a length-n array (0 on A, 1 on B, the interior solve elsewhere).
    """
    in_A = np.asarray(in_A, dtype=bool)
    in_B = np.asarray(in_B, dtype=bool)
    if np.any(in_A & in_B):
        raise ValueError("basins A and B overlap")
    if not in_A.any() or not in_B.any():
        raise ValueError("both basins must be non-empty")

    WT = sp.csc_array(W).T.tocsr()  # backward generator W^T
    interior = ~(in_A | in_B)
    q = np.zeros(WT.shape[0])
    q[in_B] = 1.0
    if interior.any():
        A_II = WT[interior][:, interior].tocsc()
        # -(W^T)_IB @ 1 = -(row sums of the interior->B block)
        rhs = -np.asarray(WT[interior][:, in_B].sum(axis=1)).ravel()
        q[interior] = spsolve(A_II, rhs)
    return q


def committor_backward(W, in_A, in_B, Theta_ss) -> np.ndarray:
    """Backward committor q^-[i] = P(the system last came from A rather than B).

    Needed for transition-path fluxes at a *driven* (non-equilibrium) steady
    state, where q^- is NOT 1 - q. It is the forward committor of the
    time-reversed chain, whose generator is

        W* = D W^T D^{-1},   D = diag(Theta_ss),

    solved with the target basin = A:  q^- = committor(W*, in_A=in_B, in_B=in_A).
    At detailed balance W* = W and q^- = 1 - q exactly.

    Theta_ss must be strictly positive (an irreducible steady state); a state
    with zero stationary mass makes the reversed generator ill-defined.
    """
    pi = np.asarray(Theta_ss, dtype=float)
    if np.any(pi <= 0.0):
        raise ValueError(
            "committor_backward needs a strictly positive Theta_ss (irreducible steady state). Found non-positive entries"
        )
    D = sp.diags(pi)
    Dinv = sp.diags(1.0 / pi)
    W_star = D @ sp.csc_array(W).T @ Dinv
    return committor(W_star, in_A=in_B, in_B=in_A)
