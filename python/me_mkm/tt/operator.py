"""
Build the ME-MKM generator W as a tensor-train operator (MPO), directly from
the reaction list and tile geometry -- no microstate enumeration.

Every elementary event (one reaction placed at one site or one bond, together
with its lateral-interaction correction) is an EXACT rank-1 MPO term: a product
of single-site matrices. This is because the correction

    corr = exp(-beta * sum_m epsilon[sp_reacting][s_m])      (beta = 1/kbt)

factorizes over the neighbor sites m as a product of per-site diagonal factors.
W is the sum of all such terms; we accumulate them with intermediate SVD
rounding (TT.ortho) so the running rank collapses to W's true (small) MPO rank.

This mirrors, term for term, the Rust COO builder in src/memkm_rs_lib.rs
(compute_offdiag / compute_w_components_coo): the order-2 orientation dedup and
the "correction over non-reacting neighbors" rule are reproduced exactly, so
mpo_to_dense(build_W_tt(b)) == build_W(b, steady_state=False) to round-off.
"""

import numpy as np
from scikit_tt.tensor_train import TT

from me_mkm._me_mkm import MEMKMBuilder


def _neighbors(builder: MEMKMBuilder) -> list:
    """Undirected adjacency neighbors[i] = sorted neighbor sites of i, rebuilt
    from builder.neighbor_pairs() (the same bond set the Rust W builder uses)."""
    adj = [set() for _ in range(builder.l)]
    for i, j in builder.neighbor_pairs():
        adj[i].add(j)
        adj[j].add(i)
    return [sorted(s) for s in adj]


def _correction(im):
    """Per-species diagonal correction factor rows and their energy rows.

    Returns (D, Emag, trivial) where D[a] = exp(-epsilon[a]/kbt) is the (n,)
    factor a reacting site of species a imposes on one neighbor, Emag[a] =
    epsilon[a] is the matching energy row (for d/dbeta), and trivial flags an
    all-zero interaction (corr == 1, so no neighbor factors are needed)."""
    eps = np.asarray(im.epsilon, dtype=float)
    trivial = not np.any(eps)
    D = np.exp(-eps / im.kbt)
    return D, eps, trivial


def _E(n: int, to: int, frm: int) -> np.ndarray:
    """Single-entry transition matrix: 1 at [to, frm], else 0. As an MPO core it
    maps site species `frm` -> `to` (W[to_idx, from_idx] convention)."""
    m = np.zeros((n, n))
    m[to, frm] = 1.0
    return m


def _event_terms(builder: MEMKMBuilder, rates=None):
    """Yield every elementary rank-1 term of W as
        (rxn_index, weight, factors, energies)
    where `factors` maps site -> (n, n) matrix (identity where absent) and
    `energies` maps each correction site -> the summed epsilon energy row Emag
    (an (n,) vector), used by build_dW_dbeta_tt. `weight` is the signed rate.

    rates: optional per-reaction base rates overriding rxn.rate (builder
    reaction order), so W(k) = sum_i k_i * component_i can be assembled cheaply.
    """
    n = builder.n_species
    neigh = _neighbors(builder)
    pairs = builder.neighbor_pairs()
    global_im = builder.get_interaction()
    reactions = builder.get_reactions()

    for ri, rxn in enumerate(reactions):
        k = rxn.rate if rates is None else rates[ri]
        pin, pout = list(rxn.pattern_in), list(rxn.pattern_out)
        im = rxn.get_interaction() or global_im
        D, Emag, trivial = _correction(im)

        if len(pin) == 1:
            a, b = pin[0], pout[0]
            if a == b:
                continue  # no state change -> no off-diagonal, no diagonal loss
            for i in range(builder.l):
                factors = {i: _E(n, b, a) - _E(n, a, a)}
                energies = {}
                if not trivial:
                    for m in neigh[i]:
                        factors[m] = np.diag(D[a])
                        energies[m] = Emag[a].copy()
                yield ri, k, factors, energies

        elif len(pin) == 2:
            a1, a2 = pin
            b1, b2 = pout
            if (a1, a2) == (b1, b2):
                continue
            symmetric = a1 == a2 and b1 == b2  # Rust fired_to dedup condition
            for si, sj in pairs:
                orientations = [(si, sj)] if symmetric else [(si, sj), (sj, si)]
                for i, j in orientations:
                    # Shared correction: non-reacting neighbors of both reacting
                    # sites. A site adjacent to both (complete-graph tiles)
                    # accumulates both diagonal factors (they commute).
                    corr_factors = {}
                    corr_energies = {}
                    if not trivial:
                        for site, sp in ((i, a1), (j, a2)):
                            for m in neigh[site]:
                                if m == i or m == j:
                                    continue
                                d = np.diag(D[sp])
                                corr_factors[m] = (
                                    corr_factors.get(m, np.eye(n)) @ d
                                )
                                corr_energies[m] = corr_energies.get(
                                    m, np.zeros(n)
                                ) + Emag[sp]
                    gain = {**corr_factors, i: _E(n, b1, a1), j: _E(n, b2, a2)}
                    loss = {**corr_factors, i: _E(n, a1, a1), j: _E(n, a2, a2)}
                    yield ri, +k, gain, dict(corr_energies)
                    yield ri, -k, loss, dict(corr_energies)
        else:
            raise ValueError(
                f"reaction {ri} has pattern length {len(pin)}; only 1 or 2 supported"
            )


def _sum_rank1(terms, l, n, threshold, group_size):
    """Sum an iterable of (weight, factors) rank-1 MPOs with intermediate SVD
    rounding every `group_size` terms so the running rank never blows up."""
    from me_mkm.tt.convert import rank1_operator

    acc = None
    pending = []

    def flush(acc):
        if not pending:
            return acc
        block = pending[0]
        for t in pending[1:]:
            block = block + t
        pending.clear()
        block = block.ortho(threshold=threshold)
        return block if acc is None else (acc + block).ortho(threshold=threshold)

    for weight, factors in terms:
        pending.append(rank1_operator(l, n, factors) * float(weight))
        if len(pending) >= group_size:
            acc = flush(acc)
    acc = flush(acc)
    if acc is None:
        # No terms at all (e.g. every reaction is a no-op): return the zero MPO.
        return rank1_operator(l, n, {0: np.zeros((n, n))})
    return acc


def build_W_tt(
    builder: MEMKMBuilder, rates=None, threshold: float = 1e-14, group_size: int = 16
) -> TT:
    """Generator W as a TT operator (MPO), assembled directly from the reactions.

    Dense-equivalent to build_W(builder, steady_state=False): mpo_to_dense of
    the result matches the Rust sparse W to round-off. `rates` overrides the
    per-reaction base rates (builder order). `threshold` is the SVD rounding
    tolerance during accumulation (1e-14 keeps it numerically exact)."""
    l, n = builder.l, builder.n_species
    terms = ((w, f) for _, w, f, _ in _event_terms(builder, rates=rates))
    return _sum_rank1(terms, l, n, threshold, group_size)


def build_W_tt_components(builder: MEMKMBuilder, threshold: float = 1e-14) -> list:
    """Per-reaction unit-rate W MPOs (builder reaction order), the TT analog of
    generator.build_W_components. W is linear in each rate, so for rates k:
        W(k) = sum_i k_i * components[i].
    """
    l, n = builder.l, builder.n_species
    n_rxn = len(builder.get_reactions())
    by_rxn = [[] for _ in range(n_rxn)]
    # rates=ones gives each term weight +/-1, i.e. the component at unit base
    # rate; the sign (gain vs loss) is still carried by the weight.
    ones = [1.0] * n_rxn
    for ri, w, f, _ in _event_terms(builder, rates=ones):
        by_rxn[ri].append((w, f))
    return [_sum_rank1(iter(terms), l, n, threshold, 16) for terms in by_rxn]


def build_dW_dbeta_tt(builder: MEMKMBuilder, threshold: float = 1e-14) -> TT:
    """d(W)/d(beta) as a TT operator (beta = 1/kbt), from the interaction
    correction only. Dense-equivalent to build_dW_dbeta_components recombined at
    the reactions' base rates.

    corr = prod_m exp(-beta * Emag_m[s_m]); by the product rule each event term
    contributes, per correction site m, one rank-1 term with that site's factor
    diag(exp(-beta*Emag_m)) replaced by diag(-Emag_m * exp(-beta*Emag_m)).
    With no interaction every Emag is zero, so dW/dbeta is the zero operator."""
    l, n = builder.l, builder.n_species
    global_im = builder.get_interaction()
    # beta = 1/kbt of each event's interaction model; needed to weight exp().
    reactions = builder.get_reactions()

    def terms():
        for ri, weight, factors, energies in _event_terms(builder):
            if not energies:
                continue
            im = reactions[ri].get_interaction() or global_im
            beta = 1.0 / im.kbt
            for m, Emag in energies.items():
                dfactor = np.diag(-Emag * np.exp(-beta * Emag))
                new_factors = dict(factors)
                new_factors[m] = dfactor
                yield weight, new_factors

    return _sum_rank1(terms(), l, n, threshold, 16)
