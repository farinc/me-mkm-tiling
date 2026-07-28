"""
Slow eigenpairs of the generator in TT format.

The k eigenvalues of W nearest lambda=0 (index 0 is the stationary mode,
index 1 the slowest relaxation mode lambda_2, etc.), in TT format.

In general, W is non-Hermitian and can have genuine complex-conjugate
eigenvalue pairs.

Two solvers are provided:

- slow_eigenpairs_tt / left_slow_eigenpairs_tt (default): sequential
  Wielandt/Hotelling-deflated shift-invert, via repeated sle.mals linear
  solves (inverse iteration). Handles the stiff-timescale-separation
  regime (fast hopping rates orders of magnitude above the slow relaxation
  modes) that evp.als empirically fails to converge on. Real eigenvalues only.

- slow_eigenpairs_tt_als / left_slow_eigenpairs_tt_als: the original
  scikit_tt.solvers.evp.als block eigensolver. Can chase complex
  eigenvalue pairs, but its local per-core micro-eigensolves have no
  global spectral transform, so it does not converge when the operator's
  timescales are separated by many orders of magnitude. Used for for
  milder timescale-separation regimes, where its complex-eigenvalue support
  may still be useful.

TODO In the case of systems with detailed balance, one should make use of symmetric postive definite
or symmetric semi-postive definite methods that are much mor efficient analogous to what can be done
for the sparse matrices. One alogorithm that seem to be particularly good at this is AMEn as there
is a garentee for the rank to be bounded. Unfortunately, AMEn is not in scikit_tt...
"""

import numpy as np
from scikit_tt.solvers import evp, sle
from scikit_tt.tensor_train import TT
from scikit_tt.tensor_train import eye as tt_eye
from scikit_tt.tensor_train import rand as tt_rand

from me_mkm.tt.convert import ones_tt, vector_outer_product_tt


def slow_eigenpair_residual(W_tt: TT, lam: complex, phi_tt: TT) -> float:
    """||W_tt @ phi - lam*phi|| / ||phi||, TT-native (no densification),
    mirrors committor_tt_residual/steady_state_residual."""
    return float((W_tt @ phi_tt - phi_tt * lam).norm() / phi_tt.norm())


def _shift_deflate(A: TT, sigma: float, pairs, c: float, threshold: float) -> TT:
    """A - sigma*I + c * sum_i |phi_i><psi_i|, the operator whose dominant
    inverse-iteration mode is A's eigenvalue nearest sigma among the
    eigenspace NOT spanned by the (already found) (phi_i, psi_i) pairs.
    Each pair must be the true biorthogonal (right, left) eigenvector pair
    for its eigenvalue -- not the same vector twice -- since A is
    non-normal (see tt_spectral_deflation_followup.md)."""
    op = A + tt_eye(A.row_dims) * (-sigma)
    for phi, psi in pairs:
        op = op + vector_outer_product_tt(phi, psi) * c
    return op.ortho(threshold=threshold)


def _inverse_iterate(
    A_shifted: TT,
    A: TT,
    x0: TT,
    repeats: int,
    mals_repeats: int,
    max_rank: int,
    threshold: float,
):
    """sle.mals inverse power iteration on A_shifted, `repeats` outer steps
    (mirrors scikit_tt.solvers.evp.power_method's sle.als/bare-shift
    iteration, but with sle.mals for rank-adaptivity and a deflated
    A_shifted). Eigenvalue is the Rayleigh quotient on the ORIGINAL
    (undeflated, unshifted) A, evaluated at the converged vector."""
    x = x0 * (1.0 / x0.norm())
    for _ in range(repeats):
        x = sle.mals(
            A_shifted,
            x,
            x,
            repeats=mals_repeats,
            threshold=threshold,
            max_rank=max_rank,
        )
        x = x * (1.0 / x.norm())
    lam = float(np.real(x.transpose() @ A @ x))
    return lam, x


def slow_eigenpairs_mals(
    W_tt: TT,
    k: int,
    theta_tt: TT,
    sigma: float,
    c: float,
    max_rank: int,
    repeats: int,
    mals_repeats: int,
    threshold: float,
    initial_guess: TT,
):
    """The shared two-sided deflation sweep behind slow_eigenpairs_tt and
    left_slow_eigenpairs_tt: modes 1..k-1 each need BOTH the right
    eigenvector of W_tt (from deflating W_tt) and the left eigenvector of
    W_tt (from deflating W_tt.transpose()) before the NEXT mode can be
    deflated on either side (tt_spectral_deflation_followup.md, step 5),
    so finding k eigenpairs on either side always does both solves.

    Returns (eigenvalues, right_eigentensors, left_eigentensors). Mode 0 is
    exact and free (eigenvalue 0, right=theta_tt, left=ones); modes 1..k-1
    come from sequential deflated shift-invert, `repeats` outer sle.mals
    inverse iterations per mode per side, each restarted from
    initial_guess (or theta_tt/ones) rather than the previous mode's
    vector, since the previous mode's own direction is exactly what the
    next deflation step removes.

    W_tt is internally rescaled by its own norm (mirroring
    solve_steady_state_tt's grounding-operator normalization) so that
    sigma/c are meaningful regardless of the model's raw rate scale.
    """
    l = W_tt.order
    n = W_tt.row_dims[0]
    ones = ones_tt(l, n)
    wn = W_tt.norm()
    Wn = W_tt * (1.0 / wn) if wn > 0 else W_tt
    WnT = Wn.transpose()

    eigenvalues = [0.0]
    right_modes = [theta_tt]
    left_modes = [ones]

    x0 = theta_tt if initial_guess is None else initial_guess
    y0 = ones if initial_guess is None else initial_guess

    for _ in range(1, k):
        pairs_W = list(zip(right_modes, left_modes))
        pairs_WT = list(zip(left_modes, right_modes))
        A_shift_W = _shift_deflate(Wn, sigma, pairs_W, c, threshold)
        A_shift_WT = _shift_deflate(WnT, sigma, pairs_WT, c, threshold)

        lam_r, phi_R = _inverse_iterate(
            A_shift_W, Wn, x0, repeats, mals_repeats, max_rank, threshold
        )
        _, phi_L = _inverse_iterate(
            A_shift_WT, WnT, y0, repeats, mals_repeats, max_rank, threshold
        )

        eigenvalues.append(lam_r * wn)
        right_modes.append(phi_R)
        left_modes.append(phi_L)

    return eigenvalues, right_modes, left_modes


def slow_eigenpairs_tt_als(
    W_tt: TT,
    k: int,
    sigma: float | None = None,
    initial_guess: TT = None,
    max_rank: int = 50,
    repeats: int = 20,
    solver: str = "eig",
):
    """The k eigenpairs of W_tt nearest lambda=sigma, via scikit_tt's block
    ALS eigensolver (scikit_tt.solvers.evp.als). Can return complex
    eigenvalues. Generally struggles to converge when the spectral gap between
    eigenvalues are small.

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


def left_slow_eigenpairs_tt_als(W_tt: TT, k: int, sigma: float = None, **kwargs):
    """Left eigenpairs of W_tt (eigenpairs of its transpose) via evp.als; a
    thin wrapper since W_tt.transpose() is already used in committor.py."""
    return slow_eigenpairs_tt_als(W_tt.transpose(), k, sigma, **kwargs)
