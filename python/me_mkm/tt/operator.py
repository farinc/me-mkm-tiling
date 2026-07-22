"""
Build the ME-MKM generator W as a tensor-train operator (MPO), directly from
the reaction list and tile geometry -- no microstate enumeration.

Every elementary event (one reaction placed at one site or one bond, together
with its lateral-interaction correction) is an EXACT rank-1 MPO term: a product
of single-site matrices. This holds because the multiplicative rate correction
    corr = exp(-delta_e / kbt)
has a delta_e that is a SUM over the reacting sites' non-reacting neighbors, so
corr factorizes over those neighbor sites into per-site diagonal factors.

We do NOT reimplement any interaction formula here. The interaction model
(src/memkm_rs_lib.rs, the InteractionModel trait) exposes its factorization
directly: `im.site_energy_row(a, b)` gives the per-neighbor energy row a
reacting site changing a -> b contributes (the diagonal factor is its
exp(-row/kbt)), and `im.pair_energy(a1, b1, a2, b2)` gives the reacting pair's
scalar mutual-bond energy. Any interaction whose delta_e is additive over
neighbors -- both current schemes, and any future model with the same property
-- is handled with no change here; it is exactly the set that has a rank-1 MPO
form. The pair mutual bond depends on no spectator site, so it is a scalar
prefactor folded into the term weight (kept separately as `mutual_e` for the
d/dbeta product rule).

W is the sum of all such terms; we accumulate them with intermediate SVD
rounding (TT.ortho) so the running rank collapses to W's true (small) MPO rank.
The result matches the Rust COO builder (which aggregates the SAME factorized
energies into its scalar correction) so mpo_to_dense(build_W_tt(b)) ==
build_W(b, steady_state=False) to round-off, including the order-2 orientation
dedup and the mutual-bond term.
"""

import numpy as np
from torchtt import TT

from me_mkm._me_mkm import MEMKMBuilder


def _neighbors(builder: MEMKMBuilder) -> list:
    """Undirected adjacency neighbors[i] = sorted neighbor sites of i, rebuilt
    from builder.neighbor_pairs() (the same bond set the Rust W builder uses)."""
    adj = [set() for _ in range(builder.l)]
    for i, j in builder.neighbor_pairs():
        adj[i].add(j)
        adj[j].add(i)
    return [sorted(s) for s in adj]


def _E(n: int, to: int, frm: int) -> np.ndarray:
    """Single-entry transition matrix: 1 at [to, frm], else 0. As an MPO core it
    maps site species `frm` -> `to` (W[to_idx, from_idx] convention)."""
    m = np.zeros((n, n))
    m[to, frm] = 1.0
    return m


def _event_terms(builder: MEMKMBuilder, rates=None):
    """Yield every elementary rank-1 term of W as
        (rxn_index, weight, factors, energies, mutual_e)
    where `factors` maps site -> (n, n) matrix (identity where absent),
    `energies` maps each spectator (correction) site -> the summed per-site
    energy row (an (n,) vector), and `mutual_e` is the reacting pair's scalar
    mutual-bond energy (0.0 for order-1 events). `weight` is the signed rate,
    with the mutual-bond factor exp(-mutual_e/kbt) already folded in. `energies`
    and `mutual_e` drive build_dW_dbeta_tt's product rule.

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
        kbt = im.kbt

        if len(pin) == 1:
            a, b = pin[0], pout[0]
            if a == b:
                continue  # no state change -> no off-diagonal, no diagonal loss
            g = np.asarray(im.site_energy_row(a, b), dtype=float)
            has_g = bool(np.any(g))  # a per-event trivial check (no eps needed)
            fac = np.diag(np.exp(-g / kbt)) if has_g else None
            for i in range(builder.l):
                factors = {i: _E(n, b, a) - _E(n, a, a)}
                energies = {}
                if has_g:
                    for m in neigh[i]:
                        factors[m] = fac
                        energies[m] = g
                yield ri, k, factors, energies, 0.0

        elif len(pin) == 2:
            a1, a2 = pin
            b1, b2 = pout
            if (a1, a2) == (b1, b2):
                continue
            symmetric = a1 == a2 and b1 == b2  # Rust fired_to dedup condition
            # Site energies / mutual bond depend only on the species, not the
            # placement, so query the interaction model once per reaction.
            g_i = np.asarray(im.site_energy_row(a1, b1), dtype=float)
            g_j = np.asarray(im.site_energy_row(a2, b2), dtype=float)
            mutual_e = float(im.pair_energy(a1, b1, a2, b2))
            w_scale = np.exp(-mutual_e / kbt)  # mutual bond -> scalar prefactor
            for si, sj in pairs:
                orientations = [(si, sj)] if symmetric else [(si, sj), (sj, si)]
                for i, j in orientations:
                    # Shared correction over non-reacting neighbors of both
                    # reacting sites; a site adjacent to both (complete-graph
                    # tiles) sums both energies (its factor is the product).
                    energies = {}
                    for site, g in ((i, g_i), (j, g_j)):
                        if not np.any(g):
                            continue
                        for m in neigh[site]:
                            if m == i or m == j:
                                continue
                            energies[m] = energies.get(m, np.zeros(n)) + g
                    corr_factors = {
                        m: np.diag(np.exp(-g / kbt)) for m, g in energies.items()
                    }
                    w = k * w_scale
                    gain = {**corr_factors, i: _E(n, b1, a1), j: _E(n, b2, a2)}
                    loss = {**corr_factors, i: _E(n, a1, a1), j: _E(n, a2, a2)}
                    yield ri, +w, gain, dict(energies), mutual_e
                    yield ri, -w, loss, dict(energies), mutual_e
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
        block = block.round(eps=threshold)
        return block if acc is None else (acc + block).round(eps=threshold)

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
    terms = ((w, f) for _, w, f, _, _ in _event_terms(builder, rates=rates))
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
    for ri, w, f, _, _ in _event_terms(builder, rates=ones):
        by_rxn[ri].append((w, f))
    return [_sum_rank1(iter(terms), l, n, threshold, 16) for terms in by_rxn]


def build_dW_dbeta_tt(builder: MEMKMBuilder, threshold: float = 1e-14) -> TT:
    """d(W)/d(beta) as a TT operator (beta = 1/kbt), from the interaction
    correction only. Dense-equivalent to build_dW_dbeta_components recombined at
    the reactions' base rates.

    Each event's correction is corr = exp(-beta*mutual_e) * prod_m exp(-beta*
    g_m[s_m]) (delta_e = mutual_e + sum_m g_m[s_m], beta-independent), so by the
    product rule d(corr)/dbeta splits into:
      - one rank-1 term per spectator site m, with that site's stored factor
        exp(-beta*g_m) left-multiplied by diag(-g_m) (= its d/dbeta), and
      - one rank-1 term for the mutual bond, the whole event scaled by -mutual_e.
    Since the stored factors already hold exp(-beta*g_m), diag(-energies[m]) @
    factors[m] is exactly that factor's derivative -- no kbt needed here. With no
    interaction there are no such terms, so dW/dbeta is the zero operator."""
    l, n = builder.l, builder.n_species

    def terms():
        for _, weight, factors, energies, mutual_e in _event_terms(builder):
            for m, g in energies.items():
                new_factors = dict(factors)
                new_factors[m] = np.diag(-g) @ factors[m]
                yield weight, new_factors
            if mutual_e != 0.0:
                yield weight * (-mutual_e), factors

    return _sum_rank1(terms(), l, n, threshold, 16)
