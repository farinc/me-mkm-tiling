"""
Physical quantities from a solved distribution.

Reduce a stationary (or time-resolved) distribution Theta over microstates to
the numbers you actually report: per-species coverages, coverage histograms, and
stoichiometric production rates, plus their parameter derivatives. Also builds
independent-site initial conditions (the inverse direction: coverage -> Theta).

All coverage/production sums are linear in Theta, so passing a derivative
dTheta/dx in place of Theta yields the derivative of the observable directly.
"""

import numpy as np

from me_mkm._me_mkm import MEMKMBuilder
from me_mkm.generator import build_W_components, build_dW_dbeta_components
from me_mkm.microstates import _decode_all, coverage_classes


def _specie_or_code(builder, target_species: str | int):
    return (
        list(builder.species_names).index(target_species)
        if isinstance(target_species, str)
        else int(target_species)
    )


def coverage_mean(builder: MEMKMBuilder, Theta) -> np.ndarray:
    """
    Per-species mean coverage or coverage derivative dTheta_ss/dx from Theta, the
    distribution over all microstates, as an array indexed by species code as given
    from builder.

    Theta is either (n,) or (n, n_t) (from a time series); the
    result gains a matching trailing axis.
    """
    Theta = np.asarray(Theta)
    states = _decode_all(builder)  # (n_states, l)
    counts = np.stack([(states == s).sum(axis=1) for s in range(builder.n_species)])
    return (counts @ Theta) / builder.l  # (base,) or (base, n_t)


def coverage_distribution(builder: MEMKMBuilder, Theta):
    """
    Histogram P(n) = total Theta over microstates with exactly n sites of a
    species, for n = 0..l.

    Returns an array indexed [species, n] (plus a trailing time axis if Theta
    is 2-D): entry [s, n] is the total Theta over microstates with exactly n sites
    of species s.
    """
    Theta = np.asarray(Theta)
    l = builder.l

    # One pass over the classes fills every species' histogram at once; each
    # class contributes its total mass to bin n0 of species 0 (the remainder)
    # and to bin counts[code-1] of every other species.
    P = np.zeros((builder.n_species, l + 1, *Theta.shape[1:]))
    for counts, idxs in coverage_classes(builder):
        mass = Theta[idxs].sum(axis=0)
        P[0, l - sum(counts)] += mass
        for code, n in enumerate(counts, start=1):
            P[code, n] += mass

    return P


def independent_site_distribution(builder: MEMKMBuilder, coverage) -> np.ndarray:
    """
    Maximum-entropy microstate distribution with prescribed marginal coverages:
    sites are independent, so Theta0[s] = prod_j p_j^n_j(s). The natural IC for a
    known coverage with no spatial correlation.

    coverage : array indexed by species code, coverage[s] = fraction of sites in
        species s. Entry 0 is replaced by the remainder 1 - sum(coverage[1:]),
        which must be >= 0.
    """
    p = np.array(coverage, dtype=float)
    p[0] = max(0.0, 1.0 - p[1:].sum())  # species 0's fraction is the remainder

    # Site-independent product Theta0[s] = prod_j p[site_j]; p[states] maps each
    # site to its marginal probability, then the row product gives the state's.
    return np.prod(p[_decode_all(builder)], axis=1)


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
