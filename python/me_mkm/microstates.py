"""
Enumerate and query the microstate space.

Pure combinatorics over microstates. Provide functions to decode a microstate
index into its site vector, group states into coverage classes, and select states
by a coverage condition. Nothing here needs a solved distribution; it describes
which states exist and what they look like. Quantities computed *from* a solved
distribution (mean coverage, coverage histograms, production rates) live in
me_mkm.observables.
"""

import math

import numpy as np

from me_mkm._me_mkm import MEMKMBuilder, decode_state, state_counts  # noqa: F401


def microstate_as_coverage(builder: MEMKMBuilder, idx: int) -> np.ndarray:
    """Coverages of microstates coverage[s] = n_s / l (the fractions sum to 1)."""
    return np.array(state_counts(idx, builder.l, builder.n_species)) / builder.l


def coverage_classes(builder: MEMKMBuilder) -> list:
    """Coverage-class partition as a list of (counts, indices) pairs, sorted by
    counts. counts is the site-count array of species 1..n_species-1 (species
    0's count is l - counts.sum()); indices are that class's microstate indices.
    Degeneracy is len(indices)."""
    return [
        (np.asarray(counts, dtype=int), np.asarray(idxs, dtype=int))
        for counts, idxs in builder.coverage_classes()
    ]


def pattern_delta_counts(pattern_in, pattern_out, n_ads) -> list:
    """Net change a reaction makes to the coverage signature: delta[s-1] is the
    change in species s's site count when pattern_in becomes pattern_out. Fixed
    per reaction, so a firing from class counts always lands in class
    counts + delta -- the coverage-class transition target."""
    delta = [0] * n_ads
    for s_in, s_out in zip(pattern_in, pattern_out):
        if s_in != s_out:
            if s_in > 0:
                delta[s_in - 1] -= 1
            if s_out > 0:
                delta[s_out - 1] += 1
    return delta


def class_match_counts(builder: MEMKMBuilder, pattern_in) -> list:
    """Reactive-match counts of a reaction pattern across the coverage-class
    partition, as (counts, indices, matches) triples aligned with
    coverage_classes; matches[j] is builder.count_reactive of the class's j-th
    microstate (indices order) -- the per-state event multiplicity, equal to
    -diagonal(W_r) at unit rate. An order-1 pattern's count is a species count
    and therefore constant within a class; an order-2 (neighbor-pair) count
    depends on the arrangement over the tile's bonds and need not be."""
    base = builder.n_species
    return [
        (
            counts,
            idxs,
            np.array(
                [
                    builder.count_reactive(
                        decode_state(int(i), builder.l, base), pattern_in
                    )
                    for i in idxs
                ]
            ),
        )
        for counts, idxs in coverage_classes(builder)
    ]


def microstate_mask(
    builder: MEMKMBuilder, predicate=None, eps: float = 1e-9, **bounds
) -> np.ndarray:
    """Length-n_states boolean mask selecting a coverage level-set basin, for use
    as an A or B basin in the committor (or any other microstate-set query).

    Per-species coverage bounds (lo, hi) using keyword from species name in builder.
    Note the bounds may be open by using None for one of the bounds.
        microstate_mask(b, A=(None, 0.2))  # coverage_A <= 0.2
        microstate_mask(b, A=(0.3, None))  # coverage_A >= 0.3

    predicate: coverage_array -> bool, for conditions the bounds can't express
    (ratios, sums, differences); coverage_array is microstate_as_coverage()'s
    output, indexed by species code, e.g. predicate=lambda c: c[1] + c[2] <= 0.5.
    Applied as a Python filter over the bound-reduced candidates.

    eps: tolerance absorbed into the fraction->integer-count rounding
    (lo*l - eps / hi*l + eps) so bounds landing exactly on an integer count
    aren't excluded by float error.

    All constraints (bounds and predicate) are ANDed. Returns a boolean array
    index-aligned with a solved distribution Theta.

    Example (CO oxidation, species ["*", "CO", "O"]):
        in_B = microstate_mask(builder, CO=(1.0, 1.0))        # CO-poisoned (full CO)
        in_A = microstate_mask(builder, CO=(None, 0.0))       # reactive (no CO)

    To recover the matching microstates' site vectors rather than a mask,
    decode the indices the mask selects:
        idxs = np.where(mask)[0]
        states = np.array([decode_state(i, builder.l, builder.n_species) for i in idxs])
    """
    name_to_code = {name: i for i, name in enumerate(builder.species_names)}
    l = builder.l
    n_species = builder.n_species
    min_counts = [0] * n_species
    max_counts = [l] * n_species
    for name, bound in bounds.items():
        code = name_to_code.get(name)
        if code is None:
            raise ValueError(
                f"unknown species {name!r}; known: {builder.species_names}"
            )
        lo, hi = bound
        if lo is not None:
            min_counts[code] = max(min_counts[code], math.ceil(lo * l - eps))
        if hi is not None:
            max_counts[code] = min(max_counts[code], math.floor(hi * l + eps))
    idxs = list(builder.select_states(min_counts, max_counts))
    if predicate is not None:
        idxs = [i for i in idxs if predicate(microstate_as_coverage(builder, i))]
    mask = np.zeros(builder.n_states, dtype=bool)
    if idxs:
        mask[np.asarray(idxs, dtype=int)] = True
    return mask
