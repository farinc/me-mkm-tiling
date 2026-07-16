"""
Build the ME-MKM generator matrix W.

If the rate constants in the reactions change over time (like in the case of forced
systems in pressure/concentration) then one can make use of the fact that
     W(t) = sum(k_i(t) * components[i])
where k_i is the reaction constant given in the Reaction(rate=k_i) set you gave the builder.
The rate value defined in Reaction becomes unused.
"""

import numpy as np
import scipy.sparse as sp

from me_mkm._me_mkm import MEMKMBuilder


def build_W(builder: MEMKMBuilder, steady_state: bool = True) -> sp.csc_array:
    """
    Build the ME-MKM transition matrix as a scipy sparse array. If steady-state is True,
    the last row is overwritten with the normalisation condition, providing the reduced
    transition matrix for solving the steady-state distribution. If steady_state is False,
    the full transition matrix is returned for debugging purposes.
    """
    rows, cols, vals = (
        builder.build_w_ss_coo() if steady_state else builder.build_w_coo()
    )
    n = builder.n_states
    return sp.csc_array((vals, (rows, cols)), shape=(n, n))


def build_W_components(builder: MEMKMBuilder) -> list:
    """
    Per-reaction dynamical W = W(t) matrices, each at unit base rate
    (builder.get_reactions() order).

    W is linear in each rate constant, so for time-dependent rates k_i(t):
        W(t) = sum(k_i(t) * components[i])
    Evaluating this sum is much cheaper per ODE step than rebuilding W from
    scratch.
    """
    n = builder.n_states
    return [
        sp.csc_array((vals, (rows, cols)), shape=(n, n))
        for rows, cols, vals in builder.build_w_components_coo()
    ]


def build_dW_dbeta_components(builder: MEMKMBuilder) -> list:
    """
    Per-reaction d(dynamical-form W)/dbeta at unit base rate, from the
    interaction correction only (beta = 1/kbt).

    Base rates are held fixed here; if they carry their own beta-dependence
    (e.g. Arrhenius), the caller adds that dk_i/dbeta term at assembly.
    """
    n = builder.n_states
    return [
        sp.csc_array((vals, (rows, cols)), shape=(n, n))
        for rows, cols, vals in builder.build_dw_dbeta_components_coo()
    ]


def _base_rates(builder: MEMKMBuilder) -> np.ndarray:
    """Each reaction's base rate constant, in builder.get_reactions() order."""
    return np.array([rxn.rate for rxn in builder.get_reactions()], dtype=float)


def _combine(builder: MEMKMBuilder, weights, components) -> sp.csc_array:
    """sum(weights[i] * components[i]), skipping zero weights. Returns an
    explicit zero matrix if every weight is zero, so callers always get a
    sparse array of the right shape back (never the int 0 from sum())."""
    n = builder.n_states
    total = sp.csc_array((n, n))
    for w, comp in zip(weights, components):
        if w != 0.0:
            total = total + w * comp
    return sp.csc_array(total)


def assemble_W(builder: MEMKMBuilder, rates=None) -> sp.csc_array:
    """
    Dynamical-form W assembled from the per-reaction components:
        W = sum(k_i * components[i]).

    Equivalent to build_W(builder, steady_state=False), but the rates are
    supplied here rather than baked in, so `rates` (an array indexed by
    reaction, defaulting to each Reaction's own rate) can override them without
    rebuilding the components.
    """
    if rates is None:
        rates = _base_rates(builder)
    return _combine(builder, rates, build_W_components(builder))


def assemble_dW_dbeta(builder: MEMKMBuilder, dk_dbeta) -> sp.csc_array:
    """
    d(dynamical-form W)/dbeta (beta = 1/kbt), by the product rule over both
    beta-dependent factors of each reaction's contribution k_i * component_i:

        dW/dbeta = sum(dk_i/dbeta * components[i] + k_i * dcomponents[i]/dbeta)

    The first term is the bare rate's own beta-dependence (e.g. Arrhenius
    k(beta) = k0*exp(-beta*Ea) => dk/dbeta = -Ea*k), which only the caller
    knows, hence `dk_dbeta` (an array indexed by reaction; pass zeros for rates
    that don't depend on beta). The second is the lateral-interaction
    correction's, which comes from build_dW_dbeta_components.
    """
    dk_dbeta = np.asarray(dk_dbeta, dtype=float)
    rates = _base_rates(builder)
    return _combine(builder, dk_dbeta, build_W_components(builder)) + _combine(
        builder, rates, build_dW_dbeta_components(builder)
    )


def assemble_dW_dlnC(builder: MEMKMBuilder, conc_mask) -> sp.csc_array:
    """
    d(dynamical-form W)/d(ln C), restricted to the concentration-proportional
    steps marked by `conc_mask` (an array indexed by reaction).

    Mass action makes such a step's rate linear in the concentration,
    rate = k0 * C, so d(rate)/d(lnC) = C * d(rate)/dC = rate -- the derivative
    reuses the very same component at the very same rate:
        dW/dlnC = sum(k_i * components[i]) over the marked reactions only.
    """
    rates = _base_rates(builder) * np.asarray(conc_mask, dtype=float)
    return _combine(builder, rates, build_W_components(builder))


def to_steady_state_form(W: sp.csc_array) -> sp.csc_array:
    """
    Applies the normalisation condition sum(Theta) = 1.
    Idenitical to build_W(builder, steady_state=True).
    """
    Wbar = W.tolil()
    Wbar[-1, :] = 1.0
    return sp.csc_array(Wbar.tocsc())


def to_steady_state_derivative_form(dW: sp.csc_array) -> sp.csc_array:
    """
    d(dynamical W)/dx -> d(Wbar)/dx, by zeroing the last row.

    to_steady_state_form overwrites that row with constant 1s (the
    normalisation condition), and a constant's derivative is zero -- so the
    parameter-dependence there vanishes no matter what dW/dx held.
    """
    dWbar = dW.tolil()
    dWbar[-1, :] = 0.0
    return sp.csc_array(dWbar.tocsc())
