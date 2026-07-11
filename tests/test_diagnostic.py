"""
Tests for the systems from Ge et al. 2023 and Adams et al. 2025.

Greek cross  (Ge 2023)   — l=5,  Topology.square(d=2) — K_5,  10 bond pairs
Creamcups    (Ge 2023)   — l=7,  Topology.hex(d=2)    — K_7,  21 bond pairs
Fish-scale   (Adams 2025) — l=8,  Topology.square(d=3) — 4-reg, 16 bond pairs
"""

import numpy as np
import pytest
from me_mkm import (
    InteractionModel,
    MEMKMBuilder,
    Reaction,
    TileSettings,
    build_W,
    coverages,
    decode_state,
)
from scipy.sparse.linalg import spsolve


def steady_state(W):
    """Solve W @ Theta = 0, sum(Theta) = 1 by replacing the last row."""
    n = W.shape[0]
    Wb = W.tolil()
    Wb[-1, :] = 1.0
    rhs = np.zeros(n)
    rhs[-1] = 1.0
    return spsolve(Wb.tocsc(), rhs)


TILES = {
    "greek_cross": TileSettings.square(sites=5, d=2),
    "creamcups": TileSettings.hex(d=2),  # 7-site K_7
    "fish_scale": TileSettings.square(sites=8, d=3),
}

EXPECTED_N_PAIRS = {
    "greek_cross": 10,  # K_5: 5*4/2
    "creamcups": 21,  # K_7: 7*6/2
    "fish_scale": 16,  # 4-regular: 8*4/2
}

EXPECTED_N_STATES = {
    "greek_cross": 2**5,  # 32
    "creamcups": 2**7,  # 128
    "fish_scale": 2**8,  # 256
}


def simple_builder(tile_name, k_ads=1.0, k_des=1.0, interaction=None):
    """Single-adsorbate Langmuir builder"""
    reactions = [
        Reaction([0], [1], rate=k_ads, name="ads"),
        Reaction([1], [0], rate=k_des, name="des"),
    ]
    kwargs = dict(
        tile_settings=TILES[tile_name], reactions=reactions, species_names=["*", "A"]
    )
    if interaction is not None:
        kwargs["interaction"] = interaction
    return MEMKMBuilder(**kwargs)


def run_ss(builder):
    """Solve on the full W directly and return Theta_ss."""
    W = build_W(builder)
    return steady_state(W)


# ===========================================================================
# 1. Geometry tests
# ===========================================================================


class TestGeometry:
    @pytest.mark.parametrize("tile_name", TILES)
    def test_n_pairs(self, tile_name):
        builder = simple_builder(tile_name)
        assert builder.n_pairs == EXPECTED_N_PAIRS[tile_name], (
            f"{tile_name}: expected {EXPECTED_N_PAIRS[tile_name]} pairs, got {builder.n_pairs}"
        )

    @pytest.mark.parametrize("tile_name", TILES)
    def test_n_states(self, tile_name):
        builder = simple_builder(tile_name)
        assert builder.n_states == EXPECTED_N_STATES[tile_name]

    @pytest.mark.parametrize("tile_name", TILES)
    def test_l(self, tile_name):
        builder = simple_builder(tile_name)
        assert builder.l == TILES[tile_name].l()

    def test_greek_cross_topology_deltas(self):
        assert sorted(TileSettings.square(sites=5, d=2).deltas) == sorted([1, 2])

    def test_creamcups_topology_deltas(self):
        assert sorted(TileSettings.hex(d=2).deltas) == sorted([1, 2, 3])

    def test_fish_scale_topology_deltas(self):
        assert sorted(TileSettings.square(sites=8, d=3).deltas) == sorted([1, 3])


# ===========================================================================
# 2. Langmuir isotherm (no interactions)
# ===========================================================================


class TestLangmuir:
    """
    Without lateral interactions the steady-state coverage is exactly
    theta = k_ads / (k_ads + k_des), independent of topology or tile size.
    """

    @pytest.mark.parametrize("tile_name", TILES)
    @pytest.mark.parametrize(
        "k_ads,k_des",
        [
            (1.0, 1.0),
            (2.0, 1.0),
            (1.0, 3.0),
            (0.1, 0.9),
            (10.0, 1.0),
        ],
    )
    def test_langmuir_coverage(self, tile_name, k_ads, k_des):
        builder = simple_builder(tile_name, k_ads=k_ads, k_des=k_des)
        Theta_ss = run_ss(builder)
        theta = coverages(builder, Theta_ss)
        expected = k_ads / (k_ads + k_des)
        assert abs(theta["A"] - expected) < 1e-10, (
            f"{tile_name} k_ads={k_ads} k_des={k_des}: "
            f"got {theta['A']:.12f}, expected {expected:.12f}"
        )


class TestSteadyStateDistribution:
    """
    Without interactions Theta[s] ∝ r^{popcount(s)} where r = k_ads/k_des.
    All microstates with the same occupation number must have equal weight.
    """

    @pytest.mark.parametrize("tile_name", TILES)
    @pytest.mark.parametrize("r", [0.5, 1.0, 2.0])
    def test_equal_weight_same_occupation(self, tile_name, r):
        l = TILES[tile_name].l()
        builder = simple_builder(tile_name, k_ads=r, k_des=1.0)
        W = build_W(builder)
        Theta_ss = steady_state(W)

        base = builder.n_species
        occ_to_weights = {}
        for s in range(builder.n_states):
            state = decode_state(s, l, base)
            n_ones = sum(1 for x in state if x == 1)
            occ_to_weights.setdefault(n_ones, []).append(Theta_ss[s])

        for n_ones, weights in occ_to_weights.items():
            w_arr = np.array(weights)
            assert np.allclose(w_arr, w_arr[0], rtol=1e-8), (
                f"{tile_name} r={r} n={n_ones}: weights not equal: {w_arr}"
            )

    @pytest.mark.parametrize("tile_name", TILES)
    def test_ratio_between_occupation_classes(self, tile_name):
        """Mean Theta per microstate in class n+1 / class n = r."""
        r = 2.0
        l = TILES[tile_name].l()
        builder = simple_builder(tile_name, k_ads=r, k_des=1.0)
        W = build_W(builder)
        Theta_ss = steady_state(W)

        base = builder.n_species
        occ_to_mean = {}
        for s in range(builder.n_states):
            state = decode_state(s, l, base)
            n_ones = sum(1 for x in state if x == 1)
            occ_to_mean.setdefault(n_ones, []).append(Theta_ss[s])
        occ_to_mean = {k: np.mean(v) for k, v in occ_to_mean.items()}

        ns = sorted(occ_to_mean)
        for n in ns[:-1]:
            if occ_to_mean[n] > 1e-15 and occ_to_mean[n + 1] > 1e-15:
                ratio = occ_to_mean[n + 1] / occ_to_mean[n]
                # Theta[s] ∝ r^n for all s with n ones, so mean per class
                # scales as r^n / Z. Ratio of consecutive means is r.
                assert abs(ratio - r) < 1e-8, (
                    f"{tile_name} n={n}: got ratio {ratio:.8f}, expected {r:.8f}"
                )


class TestInteractions:
    """
    Attractive interactions (eps > 0) stabilise the adsorbed state → higher coverage.
    Repulsive interactions (eps < 0) destabilise it → lower coverage.
    """

    @pytest.mark.parametrize("tile_name", TILES)
    def test_attractive_raises_coverage(self, tile_name):
        eps = 0.5  # kBT units
        interaction = InteractionModel([[0.0, 0.0], [0.0, eps]])
        builder_ni = simple_builder(tile_name, k_ads=1.0, k_des=1.0)
        builder_int = simple_builder(
            tile_name, k_ads=1.0, k_des=1.0, interaction=interaction
        )

        Theta_ni = run_ss(builder_ni)
        Theta_int = run_ss(builder_int)

        theta_ni = coverages(builder_ni, Theta_ni)["A"]
        theta_int = coverages(builder_int, Theta_int)["A"]

        assert theta_int > theta_ni, (
            f"{tile_name}: attractive eps={eps} should raise coverage "
            f"({theta_int:.4f} vs {theta_ni:.4f})"
        )

    @pytest.mark.parametrize("tile_name", TILES)
    def test_repulsive_lowers_coverage(self, tile_name):
        eps = -0.5
        interaction = InteractionModel([[0.0, 0.0], [0.0, eps]])
        builder_ni = simple_builder(tile_name, k_ads=1.0, k_des=1.0)
        builder_int = simple_builder(
            tile_name, k_ads=1.0, k_des=1.0, interaction=interaction
        )

        Theta_ni = run_ss(builder_ni)
        Theta_int = run_ss(builder_int)

        theta_ni = coverages(builder_ni, Theta_ni)["A"]
        theta_int = coverages(builder_int, Theta_int)["A"]

        assert theta_int < theta_ni, (
            f"{tile_name}: repulsive eps={eps} should lower coverage "
            f"({theta_int:.4f} vs {theta_ni:.4f})"
        )


# ===========================================================================
# 5. W-matrix structural properties
# ===========================================================================


class TestWMatrix:
    @pytest.mark.parametrize("tile_name", TILES)
    def test_columns_sum_to_zero(self, tile_name):
        W = build_W(simple_builder(tile_name), steady_state=False)
        col_sums = np.array(W.sum(axis=0)).flatten()
        assert np.allclose(col_sums, 0.0, atol=1e-12), (
            f"{tile_name}: max |col_sum| = {np.abs(col_sums).max():.2e}"
        )

    @pytest.mark.parametrize("tile_name", TILES)
    def test_diagonal_nonpositive(self, tile_name):
        W = build_W(simple_builder(tile_name), steady_state=False)
        diag = W.diagonal()
        assert np.all(diag <= 1e-14), (
            f"{tile_name}: positive diagonal entries: {diag[diag > 0]}"
        )

    @pytest.mark.parametrize("tile_name", TILES)
    def test_steady_state_is_null_vector(self, tile_name):
        builder = simple_builder(tile_name)
        W = build_W(builder, steady_state=False)
        Theta_ss = steady_state(W)
        residual = W @ Theta_ss
        assert np.linalg.norm(residual) < 1e-10, (
            f"{tile_name}: ||W Theta_ss|| = {np.linalg.norm(residual):.2e}"
        )

    @pytest.mark.parametrize("tile_name", TILES)
    def test_steady_state_normalised(self, tile_name):
        builder = simple_builder(tile_name)
        W = build_W(builder)
        Theta_ss = steady_state(W)
        assert abs(Theta_ss.sum() - 1.0) < 1e-12
        assert np.all(Theta_ss >= -1e-14)
