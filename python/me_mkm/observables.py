"""
Physical quantities from a solved distribution.
"""

import numpy as np

from me_mkm._me_mkm import MEMKMBuilder
from me_mkm.microstates import (
    class_match_counts,
    coverage_classes,
)


def coverage_distribution(builder: MEMKMBuilder, Theta):
    """
    P[s, n] = total Theta over microstates with exactly n (n = 0..l) sites of species s.
    If Theta is (Theta, t), then P[s, n, t] is returned.
    """
    Theta = np.asarray(Theta)
    l = builder.l
    # One pass over the classes fills every species' histogram at once; each
    # class contributes its total mass to bin n0 of species 0 (the remainder)
    # and to bin counts[code-1] of every other species.
    P = np.zeros((builder.n_species, l + 1, *Theta.shape[1:]))
    for counts, idxs in coverage_classes(builder):
        mass = Theta[idxs].sum(axis=0)
        P[0, l - counts.sum()] += mass
        for code, n in enumerate(counts, start=1):
            P[code, n] += mass

    return P


def class_average_matches(builder: MEMKMBuilder, pattern_in, Theta=None):
    """
    Per-coverage-class average reactive-match count of a reaction pattern (the
    per-state event multiplicity M_r, -diagonal(W_r) at unit rate).

    Returns (averages, nonuniform): averages maps each class's counts tuple to
    its average match count. nonuniform lists the counts tuples whose members
    disagree (empty means every average is an exact per-microstate count).

    Theta : distribution over microstates, optional. Without it every class
        member is weighted equally (combinatorial average). With it, members
        are weighted by their conditional probability within the class,
        sum(matches * Theta[idxs]) / sum(Theta[idxs])
    """
    if Theta is not None:
        Theta = np.asarray(Theta, dtype=float)
    averages, nonuniform = {}, []
    for counts, idxs, matches in class_match_counts(builder, pattern_in):
        key = tuple(int(c) for c in counts)
        if matches.size and (matches != matches[0]).any():
            nonuniform.append(key)
        mean = matches.mean() if matches.size else 0.0
        if Theta is not None:
            mass = Theta[idxs].sum()
            if mass > 0.0:
                mean = (matches @ Theta[idxs]) / mass
        averages[key] = float(mean)
    return averages, nonuniform


def committor_class_profile(builder: MEMKMBuilder, q, Theta_ss=None):
    """Committor probabilities resolved for all coverage classes. Computes the
    average committor probability of each class followed by the weighted varience
    of the commitor probability.
    q     : committor over all microstates, index-aligned with Theta.
    Theta : stationary distribution, for probability weighting within each class
        (P[q | class]); if None, class members are weighted equally.

    Returns (profile, spread), each a dict keyed by the class counts tuple:
    - profile[counts] = mean committor of the class, E[q | class] -- the
      p_fold value of that coverage.
    - spread[counts]  = (weighted) variance of q within the class. A near-zero
      spread means coverage locates the reaction well there; a large spread
      means microstates of the same coverage have different fates, so coverage
      is a poor reaction coordinate for that class (the Berezhkovskii-Szabo
      test).
    """
    q = np.asarray(q, dtype=float)
    if Theta_ss is not None:
        Theta_ss = np.asarray(Theta_ss, dtype=float)
    profile, spread = {}, {}
    for counts, idxs in coverage_classes(builder):
        key = tuple(int(c) for c in counts)
        qi = q[idxs]
        if qi.size == 0:
            profile[key], spread[key] = 0.0, 0.0
            continue
        w = Theta_ss[idxs] if Theta_ss is not None else None
        if w is not None and w.sum() > 0.0:
            wsum = w.sum()
            mean = float((qi @ w) / wsum)
            var = float((w @ (qi - mean) ** 2) / wsum)
        else:
            mean = float(qi.mean())
            var = float(qi.var())
        profile[key], spread[key] = mean, var
    return profile, spread


def coverage_mean(builder: MEMKMBuilder, Theta) -> np.ndarray:
    """
    Per-species mean coverage or mean coverage derivative (dTheta_ss/dx from Theta), the
    distribution over all microstates, as an array indexed by species code as given
    from builder.

    Theta is either (n,) or (n, n_t) (from a time series); the
    result gains a matching trailing axis.
    """
    Theta = np.asarray(Theta)
    l = builder.l
    # One pass over the classes (same walk as coverage_distribution) accumulates
    # each species' total site-count directly, without ever materializing the
    # dense (n_states, l) microstate array or looping (states == s) over it.
    total = np.zeros((builder.n_species, *Theta.shape[1:]))
    for counts, idxs in coverage_classes(builder):
        mass = Theta[idxs].sum(axis=0)
        total[0] += (l - counts.sum()) * mass
        for code, n in enumerate(counts, start=1):
            total[code] += n * mass
    return total / l  # (base,) or (base, n_t)


def independent_site_distribution(builder: MEMKMBuilder, coverage) -> np.ndarray:
    """
    Maximum-entropy microstate distribution with prescribed marginal coverages:
    sites are independent, so Theta0[s] = prod_j p_j^n_j(s). This function provides
    a simple IC for the dynamic ME-MKM given a initial coverage with no spatial
    correlation. Note that coverages are limited by the tile size
    (coverages = 0,...,l / l)

    coverage : array indexed by species code, coverage[s] = fraction of sites in
        species s. Entry 0 is replaced by the remainder 1 - sum(coverage[1:]),
        which must be >= 0.
    """
    p = np.array(coverage, dtype=float)
    p[0] = max(0.0, 1.0 - p[1:].sum())  # species 0's fraction is the remainder

    # Site-independent product Theta0[s] = prod_j p[site_j]; p[states] maps each
    # site to its marginal probability, then the row product gives the state's.
    states = np.array([list(row) for row in builder.get_all_microstates()], dtype=int)
    return np.prod(p[states], axis=1)
