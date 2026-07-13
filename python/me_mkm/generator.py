"""
Build the ME-MKM generator matrix W.

If the rate constants in the reactions change over time (like in the case of forced
systems in pressure/concentration) then one can make use of the fact that
     W(t) = sum(k_i(t) * components[i])
where k_i is the reaction constant given in the Reaction(rate=k_i) set you gave the builder.
The rate value defined in Reaction becomes unused.
"""

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
