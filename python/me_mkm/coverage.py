"""
Partition the microstate space by species occupancy and collect microstates
matching a coverage condition (e.g. species A at coverage <= 0.2).

A coverage class groups microstates sharing the same per-species site-count
signature; species s's coverage in a microstate is n_s / l. Species names and
their order come from the builder (builder.species_names), so callers never pass
a name list. Index 0 is not privileged -- it is just species 0 (default "*").
"""

import math

import numpy as np

from me_mkm._me_mkm import MEMKMBuilder, state_counts  # noqa: F401

# Tolerance for turning coverage fractions into integer counts, so an exact
# fraction (e.g. 0.2 on l=5) lands on the intended count instead of its floor.
_EPS = 1e-9


def microstate_coverage(builder: MEMKMBuilder, idx: int) -> dict:
    """Coverage of one microstate as {species_name: n_species / l}, over every
    species in builder.species_names (the fractions sum to 1)."""
    counts = state_counts(idx, builder.l, builder.n_species)
    return {name: counts[i] / builder.l for i, name in enumerate(builder.species_names)}


def coverage_classes(builder: MEMKMBuilder) -> dict:
    """Map each coverage class to its microstate indices as
    {counts_tuple: np.ndarray[int]}. counts_tuple holds the site counts of
    species 1..n_species-1 (species 0's count is l minus their sum). Degeneracy
    is len(indices)."""
    return {
        tuple(counts): np.asarray(idxs, dtype=int)
        for counts, idxs in builder.coverage_classes()
    }


def _bound_pair(bound):
    """Normalise a bound to (lo, hi): a scalar is an upper bound, None is open."""
    if np.isscalar(bound) or bound is None:
        return None, bound
    return bound  # already (lo, hi)


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
    - predicate: coverage_dict -> bool, for conditions the bounds can't express
      (ratios, sums, differences); coverage_dict is microstate_coverage()'s
      output, e.g. predicate=lambda c: c["A"] + c["B"] <= 0.5.

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
            raise ValueError(f"unknown species {name!r}; known: {builder.species_names}")
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

    return np.asarray(sorted(idxs), dtype=int)
