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

# Tolerance for turning coverage fractions into integer counts, so an exact
# fraction (e.g. 0.2 on l=5) lands on the intended count instead of its floor.
_EPS = 1e-9


def _decode_all(builder: MEMKMBuilder) -> np.ndarray:
    """Every microstate's site vector as an (n_states, l) int array; row oi is
    microstate oi. decode_state returns bytes, so each row is
    unpacked with list() before np.array can build an int array."""
    base = builder.n_species
    return np.array(
        [list(decode_state(oi, builder.l, base)) for oi in range(builder.n_states)],
        dtype=int,
    )


def microstate_vectors(builder: MEMKMBuilder, Theta) -> np.ndarray:
    """
    Every microstate's site vector (n_states, l), index-aligned with a solved
    distribution: states[oi] <-> Theta[oi]. Theta ((n,) or (n, n_t)) is only
    checked for a matching state axis -- the decode depends solely on the index,
    builder.l, and builder.n_species.
    """
    Theta = np.asarray(Theta)
    if Theta.shape[0] != builder.n_states:
        raise ValueError(
            f"Theta has {Theta.shape[0]} states along axis 0, expected {builder.n_states}"
        )
    return _decode_all(builder)


def microstate_coverage(builder: MEMKMBuilder, idx: int) -> np.ndarray:
    """Coverage of one microstate as an array indexed by species code,
    coverage[s] = n_s / l (the fractions sum to 1)."""
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


def _bound_pair(bound):
    """Normalise a bound to (lo, hi): a scalar is an upper bound, None is open."""
    if np.isscalar(bound) or bound is None:
        return None, bound
    return bound  # already (lo, hi)


def coverage_basin_mask(builder: MEMKMBuilder, **bounds) -> np.ndarray:
    """Length-n_states boolean mask selecting a coverage level-set basin, for use
    as an A or B basin in the committor.

    Same per-species coverage-bound semantics as microstate_coverage_query (a
    scalar is an upper bound, a (lo, hi) pair bounds both sides with None open,
    one keyword per species name); all constraints are ANDed. Unlike
    microstate_coverage_query (which returns decoded state vectors), this returns
    a boolean array index-aligned with a solved distribution Theta.

    Example (CO oxidation, species ["*", "CO", "O"]):
        in_B = coverage_basin_mask(builder, CO=(1.0, 1.0))  # CO-poisoned (full CO)
        in_A = coverage_basin_mask(builder, CO=0.0)         # reactive (no CO)
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
        lo, hi = _bound_pair(bound)
        if lo is not None:
            min_counts[code] = max(min_counts[code], math.ceil(lo * l - _EPS))
        if hi is not None:
            max_counts[code] = min(max_counts[code], math.floor(hi * l + _EPS))
    idxs = list(builder.select_states(min_counts, max_counts))
    mask = np.zeros(builder.n_states, dtype=bool)
    if idxs:
        mask[np.asarray(idxs, dtype=int)] = True
    return mask


def microstate_coverage_query(
    builder: MEMKMBuilder, predicate=None, **bounds
) -> np.ndarray:
    """
    Microstate indices matching a coverage condition. Combine any of:

    - Per-species coverage bounds (fractions n_s/l, inclusive), one keyword per
      species name in builder.species_names; a scalar is an upper bound, a
      (lo, hi) pair bounds both sides (None = open). Run in Rust via
      builder.select_states:
          microstate_coverage_query(b, A=0.2)          # coverage_A <= 0.2
          microstate_coverage_query(b, A=(0.3, None))  # coverage_A >= 0.3
      Any species may be bounded, including species 0.
    - predicate: coverage_array -> bool, for conditions the bounds can't express
      (ratios, sums, differences); coverage_array is microstate_coverage()'s
      output, indexed by species code, e.g. predicate=lambda c: c[1] + c[2] <= 0.5.

    All constraints are ANDed. Returns a sorted np.ndarray of indices.
    """
    name_to_code = {name: i for i, name in enumerate(builder.species_names)}
    l = builder.l
    n_species = builder.n_species

    # Per-species coverage fractions -> inclusive integer count windows over all
    # species: n_s in [ceil(lo*l), floor(hi*l)].
    min_counts = [0] * n_species
    max_counts = [l] * n_species
    for name, bound in bounds.items():
        code = name_to_code.get(name)
        if code is None:
            raise ValueError(
                f"unknown species {name!r}; known: {builder.species_names}"
            )
        lo, hi = _bound_pair(bound)
        if lo is not None:
            min_counts[code] = max(min_counts[code], math.ceil(lo * l - _EPS))
        if hi is not None:
            max_counts[code] = min(max_counts[code], math.floor(hi * l + _EPS))

    idxs = builder.select_states(min_counts, max_counts)

    # A predicate can express what the per-species window can't, so apply it as a
    # Python filter over the already-reduced candidates.
    if predicate is not None:
        idxs = [i for i in idxs if predicate(microstate_coverage(builder, i))]

    # decode_state returns bytes (pyo3's mapping for Vec<u8>), so each row
    # must be unpacked with list() before np.asarray can build an int array.
    vec_states = [list(decode_state(i, l, n_species)) for i in sorted(idxs)]

    return np.asarray(vec_states, dtype=int)
