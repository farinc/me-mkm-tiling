"""
TT-native observables: physical quantities contracted directly from a TT
stationary distribution, without ever densifying it.

Every quantity here is a linear functional of Theta that the dense
me_mkm.observables computes as a sum over microstates; in TT format each is a
cheap contraction against a rank-1 (product) covector, so it stays polynomial
in l. Results match the dense functions to solver tolerance.
"""

import numpy as np
from torchtt import TT

from me_mkm._me_mkm import MEMKMBuilder
from me_mkm.tt.operator import _event_terms


def _product_contract(theta_tt: TT, factors: dict, default=None):
    """<probe, theta> for a rank-1 (product) probe covector whose site-p vector
    is factors[p] (or `default`, all-ones, where absent). Walks theta's cores
    directly so complex probe vectors (used by the generating function) work.
    theta_tt is a TT tensor: core p has shape (r_left, n, r_right)."""
    l = len(theta_tt.N)
    n = theta_tt.N[0]
    if default is None:
        default = np.ones(n)
    acc = np.ones((1, 1), dtype=complex)
    for p in range(l):
        v = np.asarray(factors.get(p, default))
        core = theta_tt.cores[p].detach().numpy()  # (r_left, n, r_right)
        m = np.tensordot(v, core, axes=(0, 1))  # (r_left, r_right)
        acc = acc @ m
    val = acc[0, 0]
    return val if np.iscomplexobj(val) or isinstance(val, complex) else float(val)


def site_marginals(theta_tt: TT) -> np.ndarray:
    """Per-site species marginals P[p, s] = prob site p is species s, for a
    probability-normalized theta. One rank-1 contraction per (site, species):
    <1| with site p pinned to basis vector e_s."""
    l = len(theta_tt.N)
    n = theta_tt.N[0]
    P = np.zeros((l, n))
    for p in range(l):
        for s in range(n):
            e = np.zeros(n)
            e[s] = 1.0
            P[p, s] = np.real(_product_contract(theta_tt, {p: e}))
    return P


def coverage_mean_tt(builder: MEMKMBuilder, theta_tt: TT) -> np.ndarray:
    """Per-species mean coverage, TT analog of observables.coverage_mean:
    average of the site marginals over sites."""
    return site_marginals(theta_tt).mean(axis=0)


def _term_diagonal(factors: dict) -> dict:
    """Per-site diagonals of an event term's factors. The term's contribution to
    W's diagonal is weight * prod_p diag(factors[p]), so this is the per-site
    probe of that rank-1 diagonal.

    Gain terms drop out automatically: every emitted event changes at least one
    site, and a changing site carries E(b, a) with b != a, whose diagonal is
    zero -- so only the loss part of each event survives the contraction."""
    return {site: np.diag(m) for site, m in factors.items()}


def production_rate_tt(builder: MEMKMBuilder, theta_tt: TT, stoich) -> float:
    """Steady-state production rate, TT analog of observables.production_rate:
        (1/l) * sum_r nu_r * <flux_r, theta>,   flux_r = -diagonal(W_r).

    Each event term contributes weight * prod diag(factors) to W's diagonal, so
    the flux is -sum_terms weight * <prod diag(factors), theta> -- one rank-1
    contraction per term. This covers BOTH orders uniformly: an order-1 event's
    single combined term E(b,a) - E(a,a) has diagonal -e_a (the loss), and an
    order-2 event's gain term has zero diagonal while its loss term is fully
    diagonal."""
    stoich = np.asarray(stoich, dtype=float)
    total = 0.0
    for ri, weight, factors, _, _ in _event_terms(builder):
        nu = stoich[ri]
        if nu == 0.0:
            continue
        probe = _term_diagonal(factors)
        total += -nu * weight * np.real(_product_contract(theta_tt, probe))
    return total / builder.l


def coverage_distribution_tt(builder: MEMKMBuilder, theta_tt: TT) -> np.ndarray:
    """P[s, k] = probability of exactly k (k = 0..l) sites of species s, TT
    analog of observables.coverage_distribution.

    Uses a generating function: for species s, G_s(z) = sum_k P[s,k] z^k =
    <prod_p (1 + (z-1) e_s)| theta> is a rank-1 contraction. Evaluating it at
    the (l+1) roots of unity and inverse-DFT-ing recovers the exact counts
    histogram (l+1 bins including zero occupation)."""
    l, n = builder.l, builder.n_species
    zs = np.exp(2j * np.pi * np.arange(l + 1) / (l + 1))
    P = np.zeros((n, l + 1))
    for s in range(n):
        e_s = np.zeros(n)
        e_s[s] = 1.0
        G = np.empty(l + 1, dtype=complex)
        for m, z in enumerate(zs):
            factor = np.ones(n) + (z - 1.0) * e_s
            G[m] = _product_contract(theta_tt, {p: factor for p in range(l)})
        # G(z_m) = sum_k P[s,k] z_m^k with z_m = exp(+2pi i m/(l+1)), so
        # P[s,k] = (1/(l+1)) sum_m G(z_m) exp(-2pi i m k/(l+1)) = fft(G)[k]/(l+1).
        P[s] = np.fft.fft(G).real / (l + 1)
    return P
