"""
Slow eigenpairs of the generator in TT format.

The k eigenvalues of W nearest lambda=0 (index 0 is the stationary mode,
index 1 the slowest relaxation mode lambda_2, etc.) via scikit_tt's block ALS
eigensolver (scikit_tt.solvers.evp.als).

Un general, W is non-Hermitian and can have genuine complex-conjugate eigenvalue pairs.

TODO In the case of systems with detailed balance, one should make use of symmetric postive definite
or symmetric semi-postive definite methods that are much mor efficient analogous to what can be done
for the sparse matrices. One alogorithms that seem to be particularly good at this are AMEn as there
are garentees for the rank to be bounded. Unfortunately, AMEn is not in scikit_tt...
"""

import numpy as np
from scikit_tt.solvers import evp
from scikit_tt.tensor_train import TT
from scikit_tt.tensor_train import rand as tt_rand


def slow_eigenpair_residual(W_tt: TT, lam: complex, phi_tt: TT) -> float:
    """||W_tt @ phi - lam*phi|| / ||phi||, TT-native (no densification),
    mirrors committor_tt_residual/steady_state_residual."""
    return float((W_tt @ phi_tt - phi_tt * lam).norm() / phi_tt.norm())


def slow_eigenpairs_tt(
    W_tt: TT,
    k: int,
    sigma: float = None,
    initial_guess: TT = None,
    max_rank: int = 50,
    repeats: int = 20,
    solver: str = "eig",
):
    """The k eigenpairs of W_tt nearest lambda=sigma.

    sigma : target eigenvalue; defaults to a small shift off zero
        scaled to W_tt's own norm (1e-3 * W_tt.norm()), since W has no
        natural unit scale.
    initial_guess : warm-start MPS (e.g. a converged theta_tt, or a
        perturbed product state); defaults to a random rank-(k+2) MPS. A
        rank-1 guess isn't enough for k > 1: the block solver needs local
        rank >= k at every core to extract k distinct local eigenpairs per
        sweep step.

    Returns (eigenvalues, eigentensors, residuals): eigenvalues a length-k
    complex array sorted by descending real part; residuals[i] =
    slow_eigenpair_residual(W_tt, eigenvalues[i], eigentensors[i]).
    """
    l = W_tt.order
    n = W_tt.row_dims[0]
    if sigma is None:
        sigma = 1e-3 * W_tt.norm()
    if initial_guess is None:
        initial_guess = tt_rand([n] * l, [1] * l, ranks=max(k + 2, 2))

    eigenvalues, eigentensors, _ = evp.als(
        W_tt,
        initial_guess,
        number_ev=k,
        repeats=repeats,
        sigma=sigma,
        solver=solver,
        real=False,
    )
    if k == 1:  # evp.als returns a bare scalar/TT (not length-1 containers) for k=1
        eigenvalues, eigentensors = [eigenvalues], [eigentensors]
    eigenvalues = np.asarray(eigenvalues)
    order = np.argsort(-eigenvalues.real)
    eigenvalues = eigenvalues[order]
    eigentensors = [eigentensors[i] for i in order]
    for i, phi in enumerate(eigentensors):
        if max(phi.ranks) > max_rank:
            eigentensors[i] = phi.ortho(max_rank=max_rank)
    residuals = [
        slow_eigenpair_residual(W_tt, eigenvalues[i], eigentensors[i]) for i in range(k)
    ]
    return eigenvalues, eigentensors, residuals


def left_slow_eigenpairs_tt(W_tt: TT, k: int, sigma: float = None, **kwargs):
    """Left eigenpairs of W_tt (eigenpairs of its transpose); a thin wrapper
    since W_tt.transpose() is already used in committor.py."""
    return slow_eigenpairs_tt(W_tt.transpose(), k, sigma, **kwargs)
