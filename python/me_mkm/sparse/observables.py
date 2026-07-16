"""
Production-rate observables from a solved distribution.

These consume the sparse per-reaction W components (their diagonals carry the
per-state event counts), so they live in the scipy-backed subpackage. The
purely combinatorial observables (coverages, class averages) are scipy-free
and stay in me_mkm.observables.

All production sums are linear in Theta, so passing a derivative dTheta/dx in
place of Theta yields the derivative of the observable directly.
"""

import numpy as np

from me_mkm._me_mkm import MEMKMBuilder
from me_mkm.sparse.generator import build_W_components, build_dW_dbeta_components


def _event_flux(builder: MEMKMBuilder) -> list:
    """Per-reaction per-state total event flux at unit base rate,
    -components[i].diagonal() (builder.get_reactions() order) -- the per-state
    reaction count already sitting on each component's diagonal."""
    return [-comp.diagonal() for comp in build_W_components(builder)]


def production_rate_vector(builder: MEMKMBuilder, stoich) -> np.ndarray:
    """
    Per-microstate production rate r_P[state] (paper eq. 4):
        r_P = sum(stoich[i] * rate_i * event_flux_i).

    stoich : array indexed by reaction, net product count per event (0 = no
        contribution, e.g. only the desorption entry set to track desorption).
    """
    r_P = np.zeros(builder.n_states)
    for rxn, nu, flux in zip(builder.get_reactions(), stoich, _event_flux(builder)):
        if nu != 0.0:
            r_P += nu * rxn.rate * flux
    return r_P


def production_rate_dbeta_vector(builder: MEMKMBuilder, stoich, dk_dbeta) -> np.ndarray:
    """d(r_P[state])/dbeta (paper eq. 6's per-state rate term), product rule analog
    of assemble_dW_dbeta. stoich and dk_dbeta are arrays indexed by reaction."""
    flux = _event_flux(builder)
    dflux = [-dcomp.diagonal() for dcomp in build_dW_dbeta_components(builder)]
    dr_P = np.zeros(builder.n_states)
    for rxn, nu, dk, f, df in zip(
        builder.get_reactions(), stoich, dk_dbeta, flux, dflux
    ):
        if nu != 0.0:
            dr_P += nu * (dk * f + rxn.rate * df)
    return dr_P


def production_rate_dlnC_vector(builder: MEMKMBuilder, stoich, conc_mask) -> np.ndarray:
    """d(r_P[state])/d(ln C) (paper eq. 6's per-state rate term), restricted to the
    concentration-proportional steps marked by conc_mask. stoich and conc_mask are
    arrays indexed by reaction."""
    dr_P = np.zeros(builder.n_states)
    for rxn, nu, m, flux in zip(
        builder.get_reactions(), stoich, conc_mask, _event_flux(builder)
    ):
        if m and nu != 0.0:
            dr_P += nu * rxn.rate * flux
    return dr_P


def production_rate(builder: MEMKMBuilder, Theta_ss, stoich) -> float:
    """Scalar steady-state production rate (paper eq. 4): (1/L) * sum(Theta_ss *
    r_P[state]). stoich is an array indexed by reaction."""
    return float(Theta_ss @ production_rate_vector(builder, stoich)) / builder.l


def production_rate_derivative(
    builder: MEMKMBuilder, Theta_ss, dTheta_dx, stoich, dr_P_dx_vector: np.ndarray
) -> float:
    """
    Scalar steady-state production-rate derivative (paper eq. 6):
        (1/L) * sum(dTheta_ss/dx * r_P[state] + Theta_ss * dr_P[state]/dx)

    dr_P_dx_vector : the per-state rate derivative, from
        production_rate_dbeta_vector or production_rate_dlnC_vector.
    """
    r_P = production_rate_vector(builder, stoich)
    return float(dTheta_dx @ r_P + Theta_ss @ dr_P_dx_vector) / builder.l
