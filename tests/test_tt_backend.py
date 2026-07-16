"""
Phase 1 validation of the tensor-train backend (me_mkm.tt) against the dense
LU/sparse path. Skipped entirely when the optional scikit_tt dependency (the
`tt` uv group) is absent.

Two things are checked: the generator MPO built term-by-term from the reaction
list matches the Rust-assembled sparse W exactly (mpo_to_dense == build_W), and
the stationary solve matches solve_steady_state on the test tiles,
including a hard metastable point on the checkerboard plateau.
"""

import numpy as np
import pytest

from me_mkm import (
    InitialStateInteraction,
    MEMKMBuilder,
    Reaction,
    TileSettings,
)
from me_mkm.sparse import (
    build_W,
    build_W_components,
    solve_steady_state,
)

tt = pytest.importorskip("me_mkm.tt", exc_type=ImportError)

# Reuse the Adams Fig. 3 dimer system (ads/des with strong A-A repulsion +
# noninteracting dimerization) from the steady-state superlattice test.
from test_checkboard_superlattice import build_system  # noqa: E402

GREEK = TileSettings.square(sites=5, d=2)  # K_5, 32 states
FISH = TileSettings.square(sites=8, d=3)  # checkerboard-capable, 256 states
HEX = TileSettings.hex(2)  # creamcups K_7, 128 states


def langmuir(tile, eps=0.0):
    """Single-adsorbate Langmuir builder with optional A-A interaction."""
    im = InitialStateInteraction([[0.0, 0.0], [0.0, eps]])
    reactions = [
        Reaction([0], [1], rate=1.3, name="ads"),
        Reaction([1], [0], rate=0.7, name="des"),
    ]
    return MEMKMBuilder(tile, reactions, ["*", "A"], im)


# ===========================================================================
# 1. Convention: the MPO densifies in encode_state order on a tiny model
# ===========================================================================


def test_mpo_ordering_convention():
    """l=3, deltas=[1]: pins that TT core p == lattice site p (site 0 = most
    significant digit), so mpo_to_dense aligns with build_W's index space."""
    builder = langmuir(TileSettings(3, [1]), eps=0.0)
    W_dense = build_W(builder, steady_state=False).toarray()
    W_from_tt = tt.mpo_to_dense(tt.build_W_tt(builder))
    assert np.abs(W_dense - W_from_tt).max() < 1e-12


# ===========================================================================
# 2. MPO exactness: build_W_tt == build_W across tiles / interactions / models
# ===========================================================================

MPO_CASES = []
for tile, tname in [(GREEK, "greek"), (FISH, "fish"), (HEX, "hex")]:
    for eps, ename in [(0.0, "eps0"), (0.6, "attractive"), (-0.9, "repulsive")]:
        MPO_CASES.append((f"langmuir-{tname}-{ename}", langmuir(tile, eps)))
    # dimer-with-per-reaction-override on a couple of tiles / rates
for tile, tname in [(GREEK, "greek"), (FISH, "fish")]:
    for K in (10.0, 1000.0):
        MPO_CASES.append((f"dimer-{tname}-K{K:g}", build_system(K, tile)))


@pytest.mark.parametrize("name,builder", MPO_CASES, ids=[c[0] for c in MPO_CASES])
def test_build_W_tt_matches_build_W(name, builder):
    """The term-by-term MPO reproduces the Rust sparse generator to round-off
    (relative to the operator's own magnitude, since large base rates set the
    absolute scale)."""
    W_dense = build_W(builder, steady_state=False).toarray()
    W_from_tt = tt.mpo_to_dense(tt.build_W_tt(builder))
    scale = max(np.abs(W_dense).max(), 1.0)
    assert np.abs(W_dense - W_from_tt).max() / scale < 1e-10


# ===========================================================================
# 3. Grounded stationary solve matches the dense steady state
# ===========================================================================

SOLVE_CASES = [
    ("langmuir-greek-repulsive", langmuir(GREEK, -0.9)),
    ("langmuir-hex-attractive", langmuir(HEX, 0.6)),
    ("dimer-greek-K100", build_system(100.0, GREEK)),
    # hard metastable point: checkerboard plateau, strong repulsion, small gap
    ("dimer-fish-plateau-K1000", build_system(1000.0, FISH)),
]


@pytest.mark.parametrize("name,builder", SOLVE_CASES, ids=[c[0] for c in SOLVE_CASES])
def test_solve_steady_state_tt_matches_dense(name, builder):
    theta_dense, _ = solve_steady_state(build_W(builder, steady_state=True))
    theta_tt, info = tt.solve_steady_state_tt(tt.build_W_tt(builder))
    theta_from_tt = tt.tt_to_dense(theta_tt)
    assert np.abs(theta_dense - theta_from_tt).sum() < 1e-6
    assert info.residual < 1e-6
    # a probability distribution: sums to 1, (near-)nonnegative
    assert abs(theta_from_tt.sum() - 1.0) < 1e-8
    assert theta_from_tt.min() > -1e-8


def test_warm_start_matches_cold_start():
    """A warm start (the natural sweep usage) must reach the same solution as a
    cold start, not just converge faster."""
    builder = build_system(1000.0, FISH)
    W_tt = tt.build_W_tt(builder)
    theta_cold, _ = tt.solve_steady_state_tt(W_tt)
    nearby, _ = tt.solve_steady_state_tt(tt.build_W_tt(build_system(900.0, FISH)))
    theta_warm, info = tt.solve_steady_state_tt(W_tt, theta0=nearby)
    assert np.abs(tt.tt_to_dense(theta_cold) - tt.tt_to_dense(theta_warm)).sum() < 1e-6
    assert info.residual < 1e-6


# ===========================================================================
# 4. Rate override / linear component decomposition
# ===========================================================================


def test_rate_override_and_components():
    """build_W_tt(rates=...) == sum_i k_i * component_i == dense build_W(rates)."""
    builder = build_system(50.0, FISH)
    rates = [3.0, 0.5, 2.0]  # ads, des, dimer

    # dense reference: linear recombination of Rust unit-rate components
    dense_comps = build_W_components(builder)
    W_dense = sum(k * c for k, c in zip(rates, dense_comps)).toarray()

    W_tt_direct = tt.mpo_to_dense(tt.build_W_tt(builder, rates=rates))
    comps_tt = tt.build_W_tt_components(builder)
    W_tt_comb = tt.mpo_to_dense(
        sum((c * k for k, c in zip(rates[1:], comps_tt[1:])), comps_tt[0] * rates[0])
    )

    scale = max(np.abs(W_dense).max(), 1.0)
    assert np.abs(W_dense - W_tt_direct).max() / scale < 1e-10
    assert np.abs(W_dense - W_tt_comb).max() / scale < 1e-10
