"""
W components: W (the transition-rate matrix) is linear in each reaction's
bare rate constant. build_W_components / build_dW_dbeta_components ask Rust
for each reaction's contribution. This allows building W 
rates, dW/dbeta, dW/dlnC, steady-state forms, derivative forms -- is then
just a weighted sum of those fixed sparse components computed in Python, with
no further calls back into Rust. That's what makes rate sweeps, dynamic
k_i(t), and the derivative machinery below cheap: rebuild the components
once, then recombine.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

from me_mkm._me_mkm import (
    MEMKMBuilder,
    decode_state,
)  # noqa: F401


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
    Per-reaction d(dynamical-form W)/dbeta matrices, each at unit base rate
    (builder.get_reactions() order), where beta = 1/kbt.

    Mirrors build_W_components: rates themselves don't depend on beta, only
    the interaction-correction factor baked into each component does, so
    d(rate_i * component_i)/dbeta = rate_i * dbeta_components[i].
    """
    n = builder.n_states
    return [
        sp.csc_array((vals, (rows, cols)), shape=(n, n))
        for rows, cols, vals in builder.build_dw_dbeta_components_coo()
    ]


def assemble_W(builder: MEMKMBuilder, rates: dict = None) -> sp.csc_array:
    """
    Assemble the full dynamical-form W from per-reaction components:
    W = sum(rate_i * components[i]).

    rates : dict, optional
        Reaction.name -> rate override. Reactions not present default to
        their builder-assigned rxn.rate. With no overrides this reproduces
        build_W(builder, steady_state=False) exactly.
    """
    n = builder.n_states
    components = build_W_components(builder)
    reactions = builder.get_reactions()
    W = sp.csc_array((n, n))
    for rxn, comp in zip(reactions, components):
        rate = rates.get(rxn.name, rxn.rate) if rates is not None else rxn.rate
        W = W + rate * comp
    return W


def assemble_dW_dbeta(builder: MEMKMBuilder, dk_dbeta: dict) -> sp.csc_array:
    """
    Assemble dW/dbeta via the product rule:
        dW/dbeta = sum(dk_i/dbeta * components[i] + k_i * dbeta_components[i])

    dk_dbeta : dict
        Reaction.name -> dk_i/dbeta (the user's own derivative of their
        Arrhenius expression, e.g. for k(beta) = k0*exp(-beta*Ea),
        dk_dbeta = -Ea*k0*exp(-beta*Ea)). Reactions absent from the dict are
        treated as beta-independent (dk_i/dbeta = 0).
    """
    n = builder.n_states
    components = build_W_components(builder)
    dbeta_components = build_dW_dbeta_components(builder)
    reactions = builder.get_reactions()
    dW = sp.csc_array((n, n))
    for rxn, comp, dcomp in zip(reactions, components, dbeta_components):
        dk = dk_dbeta.get(rxn.name, 0.0)
        dW = dW + dk * comp + rxn.rate * dcomp
    return dW


def assemble_dW_dlnC(builder: MEMKMBuilder, conc_reaction_names: set) -> sp.csc_array:
    """
    Assemble dW/d(ln C) for mass-action steps whose rate is proportional to a
    concentration C (rate = k*C => d(rate)/d(ln C) = rate):
        dW/d(ln C) = sum(rate_i * components[i])  for i in conc_reaction_names
    """
    n = builder.n_states
    components = build_W_components(builder)
    reactions = builder.get_reactions()
    dW = sp.csc_array((n, n))
    for rxn, comp in zip(reactions, components):
        if rxn.name in conc_reaction_names:
            dW = dW + rxn.rate * comp
    return dW


def to_steady_state_form(W: sp.csc_array) -> sp.csc_array:
    """
    Replace the last row of W with the normalisation condition (all 1s),
    giving the reduced matrix Wbar used to solve for the steady-state
    distribution. to_steady_state_form(assemble_W(builder)) is mathematically
    identical to build_W(builder, steady_state=True).
    """
    Wbar = W.tolil()
    Wbar[-1, :] = 1.0
    return Wbar.tocsc()


def to_steady_state_derivative_form(dW: sp.csc_array) -> sp.csc_array:
    """
    Replace the last row of dW/dx with zeros: the normalisation row is
    constant in x, so its derivative is the zero row. The derivative-side
    analog of to_steady_state_form, used for dWbar/dbeta and dWbar/dlnC alike.
    """
    dWbar = dW.tolil()
    dWbar[-1, :] = 0.0
    return dWbar.tocsc()


def ergodic_structure(W: sp.csc_array) -> dict:
    """
    Analyse the ergodic structure of the CTMC defined by W.

    Uses strongly connected components (SCCs) on the directed graph of W.
    An SCC with no outgoing edges to any other SCC is an absorbing class —
    once probability enters it never leaves. All other states are transient
    (they eventually drain into an absorbing class).

    Returns a dict with keys:
      'absorbing_classes' : list of 1-D int arrays, one per absorbing class
      'transient_states'  : 1-D int array of all transient state indices
      'is_ergodic'        : True iff there is exactly one absorbing class
                            covering all n states (no transient states)
      'n_classes'         : number of absorbing classes
    """
    n_comp, labels = connected_components(W, directed=True, connection='strong')

    # Component c has an outgoing edge if W[i,j] != 0 with labels[j]==c, labels[i]!=c
    # (W[i,j] > 0 means transition j→i, so j's SCC sends probability to i's SCC).
    W_coo = W.tocoo()
    has_outgoing = set()
    for i, j in zip(W_coo.row, W_coo.col):
        if labels[i] != labels[j]:
            has_outgoing.add(labels[j])

    absorbing = [np.where(labels == c)[0]
                 for c in range(n_comp) if c not in has_outgoing]
    absorbing_flat = np.concatenate(absorbing) if absorbing else np.array([], int)
    transient = np.array([s for s in range(W.shape[0])
                          if s not in set(absorbing_flat.tolist())], dtype=int)
    is_ergodic = len(absorbing) == 1 and len(transient) == 0

    return {
        'absorbing_classes': absorbing,
        'transient_states':  transient,
        'is_ergodic':        is_ergodic,
        'n_classes':         len(absorbing),
    }


def solve_steady_state_components(W: sp.csc_array, theta0: np.ndarray, struct: dict = None):
    """
    Steady-state distribution for W, handling non-ergodic systems.

    If struct (from ergodic_structure) is not provided it is computed here.
    For a fully ergodic system (single absorbing class) this is identical to
    solve_steady_state(to_steady_state_form(W)).

    For multiple absorbing classes each class's W slice is solved independently
    and the results are weighted by theta0's probability mass inside that class.
    The weight reflects how much of the initial distribution ends up in each
    absorbing class — exact when theta0 starts inside the absorbing classes,
    an approximation otherwise (transient-state mass is distributed
    proportionally to the initial absorbing-class weights).
    """
    if struct is None:
        struct = ergodic_structure(W)

    if struct['is_ergodic']:
        return solve_steady_state(to_steady_state_form(W))

    classes = struct['absorbing_classes']
    weights = np.array([theta0[idx].sum() for idx in classes])
    total = weights.sum()
    if total <= 0:
        raise ValueError(
            "theta0 has no mass in any absorbing class — cannot determine weights."
        )
    weights /= total

    n = W.shape[0]
    Theta_ss = np.zeros(n)
    for w, idx in zip(weights, classes):
        if w == 0.0:
            continue
        W_sub = sp.csc_array(W[np.ix_(idx, idx)])
        theta_sub, _ = solve_steady_state(to_steady_state_form(W_sub))
        Theta_ss[idx] += w * theta_sub

    return Theta_ss, None


def restrict_to_ergodic_core(components: list, struct: dict):
    """
    Slice per-reaction W components to the ergodic core (union of absorbing classes).

    components : list of csc_array from build_W_components(builder)
    struct     : dict from ergodic_structure(W)

    Returns (restricted_components, core_idx) where core_idx is the sorted
    array of state indices in the ergodic core. Each restricted component is
    comp[core_idx, :][:, core_idx].

    Typical dynamic usage:
        W = build_W(builder, steady_state=False)
        struct = ergodic_structure(W)
        comps, core_idx = restrict_to_ergodic_core(build_W_components(builder), struct)
        theta0_core = theta0[core_idx]
        # ... solve ODE on len(core_idx)-dimensional system ...
        Theta_full = np.zeros((builder.n_states, n_t))
        Theta_full[core_idx] = sol.y
        coverages(builder, Theta_full, species_names)
    """
    core_idx = np.sort(np.concatenate(struct['absorbing_classes']))
    restricted = [sp.csc_array(comp[np.ix_(core_idx, core_idx)])
                  for comp in components]
    return restricted, core_idx


def check_ergodicity(builder) -> dict:
    """
    Builder-level ergodicity precheck.

    Builds the full dynamical W, runs ergodic_structure, and returns the
    structure dict augmented with a human-readable 'message'. Does not raise
    — the caller decides what to do with the result.

    Typical usage:
        info = check_ergodicity(builder)
        if not info['is_ergodic']:
            print(info['message'])
    """
    W = build_W(builder, steady_state=False)
    struct = ergodic_structure(W)
    n_states = W.shape[0]
    n_trans  = len(struct['transient_states'])
    sizes    = [len(c) for c in struct['absorbing_classes']]

    if struct['is_ergodic']:
        msg = f"Ergodic: single absorbing class covering all {n_states} states."
    else:
        msg = (
            f"Non-ergodic: {struct['n_classes']} absorbing classes "
            f"(sizes {sizes}), {n_trans} transient states. "
            "Consider adding diffusion steps or call restrict_to_ergodic_core()."
        )

    return {**struct, 'message': msg}


def solve_steady_state(Wbar: sp.csc_array):
    """
    Solve Wbar @ Theta_ss = e_n (e_n = [0,...,0,1]) for the steady-state
    distribution, returning (Theta_ss, lu) so the same LU factorization can
    be reused by steady_state_derivative for any number of derivatives
    without re-factorizing.
    """
    n = Wbar.shape[0]
    lu = splu(Wbar.tocsc())
    e_n = np.zeros(n)
    e_n[-1] = 1.0
    Theta_ss = lu.solve(e_n)
    return Theta_ss, lu


def steady_state_derivative(lu, dWbar_dx: sp.csc_array, Theta_ss) -> np.ndarray:
    """
    dTheta_ss/dx from the already-factorized Wbar (via solve_steady_state),
    by reusing the same factorization on the right-hand side
    -(dWbar/dx) @ Theta_ss. Derived by differentiating Wbar @ Theta_ss = e_n
    (a constant RHS): Wbar @ (dTheta_ss/dx) = -(dWbar/dx) @ Theta_ss.
    """
    return lu.solve(-(dWbar_dx @ Theta_ss))


def production_rate_vector(builder: MEMKMBuilder, stoich: dict) -> np.ndarray:
    """
    Per-microstate production rate r_P[state_k] (paper eq. 4), built from
    stoichiometrically-weighted reaction event fluxes:
        r_P[state] = sum(stoich[name_i] * rate_i * (-component_i.diagonal()[state]))

    -component_i.diagonal() (at unit base rate) is exactly reaction i's total
    per-state event flux already sitting inside the existing component
    matrices.

    stoich : dict
        Reaction.name -> net stoichiometric count of the tracked product
        species produced per event (e.g. {"des": 1.0} for desorption).
        Reactions absent from the dict don't contribute.
    """
    n = builder.n_states
    components = build_W_components(builder)
    reactions = builder.get_reactions()
    r_P = np.zeros(n)
    for rxn, comp in zip(reactions, components):
        nu = stoich.get(rxn.name, 0.0)
        if nu != 0.0:
            r_P += nu * rxn.rate * (-comp.diagonal())
    return r_P


def production_rate_dbeta_vector(builder: MEMKMBuilder, stoich: dict, dk_dbeta: dict) -> np.ndarray:
    """d(r_P[state])/dbeta (paper eq. 6's per-state rate term), product rule analog of assemble_dW_dbeta."""
    n = builder.n_states
    components = build_W_components(builder)
    dbeta_components = build_dW_dbeta_components(builder)
    reactions = builder.get_reactions()
    dr_P = np.zeros(n)
    for rxn, comp, dcomp in zip(reactions, components, dbeta_components):
        nu = stoich.get(rxn.name, 0.0)
        if nu == 0.0:
            continue
        dk = dk_dbeta.get(rxn.name, 0.0)
        dr_P += nu * (dk * (-comp.diagonal()) + rxn.rate * (-dcomp.diagonal()))
    return dr_P


def production_rate_dlnC_vector(builder: MEMKMBuilder, stoich: dict, conc_reaction_names: set) -> np.ndarray:
    """d(r_P[state])/d(ln C) (paper eq. 6's per-state rate term), restricted to mass-action steps in conc_reaction_names."""
    n = builder.n_states
    components = build_W_components(builder)
    reactions = builder.get_reactions()
    dr_P = np.zeros(n)
    for rxn, comp in zip(reactions, components):
        if rxn.name not in conc_reaction_names:
            continue
        nu = stoich.get(rxn.name, 0.0)
        if nu == 0.0:
            continue
        dr_P += nu * rxn.rate * (-comp.diagonal())
    return dr_P


def production_rate(builder: MEMKMBuilder, Theta_ss, stoich: dict) -> float:
    """Scalar steady-state production rate (paper eq. 4): (1/L) * sum(Theta_ss * r_P[state])."""
    r_P = production_rate_vector(builder, stoich)
    return float(Theta_ss @ r_P) / builder.l


def production_rate_derivative(
    builder: MEMKMBuilder, Theta_ss, dTheta_dx, stoich: dict, dr_P_dx_vector: np.ndarray
) -> float:
    """
    Scalar steady-state production-rate derivative (paper eq. 6):
        (1/L) * sum(dTheta_ss/dx * r_P[state] + Theta_ss * dr_P[state]/dx)

    dr_P_dx_vector : the per-state rate derivative, from
        production_rate_dbeta_vector or production_rate_dlnC_vector.
    """
    r_P = production_rate_vector(builder, stoich)
    return float(dTheta_dx @ r_P + Theta_ss @ dr_P_dx_vector) / builder.l


def coverage_ic(builder: MEMKMBuilder, theta: dict, species_names: list = None) -> np.ndarray:
    """
    Maximum-entropy microstate distribution with prescribed marginal coverages.

    Each site is treated as statistically independent with the given marginal
    occupancy probabilities, so:
        Theta0[s] = prod_j(p_j ^ n_j(s))
    where p_j is the probability of species j on any one site and n_j(s) is
    how many sites in state s carry species j.

    This is the natural IC when you want to initialise at a known coverage
    without imposing any spatial correlation (checkerboard, clusters, etc.).

    theta : dict of species_name -> coverage, e.g. {"A": 0.3}.
        Absent species default to 0. Species 0's fraction is inferred as
        1 - sum(theta.values()) and must be >= 0.
    species_names : optional override of the species list; defaults to
        builder.species_names (index 0 first).
    """
    base = builder.n_species
    l    = builder.l
    names = species_names or list(builder.species_names)
    name_to_code = {name: i for i, name in enumerate(names)}

    p = np.zeros(base)
    for name, cov in theta.items():
        code = name_to_code.get(name)
        if code is not None:
            p[code] = float(cov)
    p[0] = max(0.0, 1.0 - p[1:].sum())  # species 0's fraction is the remainder

    Theta0 = np.zeros(builder.n_states)
    for s in range(builder.n_states):
        state = decode_state(s, l, base)
        prob = 1.0
        for sp in state:
            prob *= p[sp]
        Theta0[s] = prob
    return Theta0


def coverages(builder: MEMKMBuilder, Theta, species_names: list = None) -> dict:
    """
    Recover per-species coverage from a distribution over states.

    Theta : array, shape (n,) or (n, n_t)
        Probability mass for each state index. Pass a 1-D array for a
        steady-state distribution, or a 2-D array (one column per timestep,
        e.g. solve_ivp's sol.y) for a coverage time series. Since this sum is
        linear in Theta and the per-state species-count weights don't depend
        on any external variable x, passing a steady-state derivative vector
        dTheta_ss/dx (e.g. from steady_state_derivative) instead of Theta_ss
        directly computes the coverage derivative dtheta/dx (paper eq. 5) --
        no separate code path needed.
    species_names : list of str, optional
        Species list to key the result by; defaults to builder.species_names.
    """
    base = builder.n_species
    names = species_names or list(builder.species_names)
    Theta = np.asarray(Theta)
    shape = (base,) if Theta.ndim == 1 else (base, Theta.shape[1])
    thetas = np.zeros(shape)
    for oi in range(builder.n_states):
        state = decode_state(oi, builder.l, base)
        for s in range(base):
            thetas[s] += Theta[oi] * state.count(s)
    thetas /= builder.l
    return {names[s]: thetas[s] for s in range(base)}
