"""
Phase 2 validation of TT-native observables, the dW/dbeta MPO, the analytic
dTheta/dbeta solve, and a warm-started parameter sweep -- all against the dense
me_mkm path. Skipped without the optional scikit_tt (`tt` uv group).
"""

import numpy as np
import pytest
import scipy.sparse as sp

from me_mkm import (
    BepInteraction,
    InitialStateInteraction,
    MEMKMBuilder,
    Reaction,
    TileSettings,
    coverage_distribution,
    coverage_mean,
)
from me_mkm.sparse import (
    build_W,
    build_dW_dbeta_components,
    production_rate,
    solve_steady_state,
    steady_state_derivative,
)

tt = pytest.importorskip("me_mkm.tt", exc_type=ImportError)

from test_checkboard_superlattice import build_system  # noqa: E402

GREEK = TileSettings.square(sites=5, d=2)
FISH = TileSettings.square(sites=8, d=3)


def langmuir(tile, eps=0.0, k_ads=1.3, k_des=0.7, omega=None):
    m = [[0.0, 0.0], [0.0, eps]]
    im = InitialStateInteraction(m) if omega is None else BepInteraction(m, omega)
    reactions = [
        Reaction([0], [1], rate=k_ads, name="ads"),
        Reaction([1], [0], rate=k_des, name="des"),
    ]
    return MEMKMBuilder(tile, reactions, ["*", "A"], im)


def interacting_dimer(tile, eps, omega=None):
    """ads/des + dimerization under one interaction model (interacting pair)."""
    m = [[0.0, 0.0], [0.0, eps]]
    im = InitialStateInteraction(m) if omega is None else BepInteraction(m, omega)
    reactions = [
        Reaction([0], [1], rate=100.0, name="ads"),
        Reaction([1], [0], rate=1.0, name="des"),
        Reaction([1, 1], [0, 0], rate=1.0, name="dimer"),
    ]
    return MEMKMBuilder(tile, reactions, ["*", "A"], im)


# ===========================================================================
# 1. dW/dbeta MPO vs Rust components
# ===========================================================================

# (eps, omega): omega=None is the initial-state scheme; a float is BEP, whose
# delta_e = omega*(S_in - S_out) makes dW/dbeta depend on the final state too.
# The interacting-dimer cases also exercise the mutual-bond derivative term.
DW_DBETA_CASES = [
    langmuir(GREEK, 0.6),
    langmuir(GREEK, -0.9),
    langmuir(GREEK, 0.6, omega=0.35),
    langmuir(GREEK, -0.9, omega=0.7),
    interacting_dimer(GREEK, -0.9),
    interacting_dimer(GREEK, 0.6, omega=0.4),
]


@pytest.mark.parametrize("builder", DW_DBETA_CASES)
def test_dW_dbeta_tt_matches_components(builder):
    comps = build_dW_dbeta_components(builder)
    rates = [r.rate for r in builder.get_reactions()]
    dW_dense = sum(k * c for k, c in zip(rates, comps)).toarray()
    dW_from_tt = tt.mpo_to_dense(tt.build_dW_dbeta_tt(builder))
    scale = max(np.abs(dW_dense).max(), 1.0)
    assert np.abs(dW_dense - dW_from_tt).max() / scale < 1e-10


def test_dW_dbeta_tt_zero_without_interaction():
    """No lateral interaction => corr == 1 for all events => dW/dbeta == 0."""
    dW = tt.mpo_to_dense(tt.build_dW_dbeta_tt(langmuir(GREEK, 0.0)))
    assert np.abs(dW).max() < 1e-12


# ===========================================================================
# 2. TT observables vs dense observables on the solved state
# ===========================================================================

OBS_CASES = [
    ("langmuir-greek-repulsive", langmuir(GREEK, -0.9)),
    ("dimer-fish-plateau", build_system(1000.0, FISH)),
]


@pytest.mark.parametrize("name,builder", OBS_CASES, ids=[c[0] for c in OBS_CASES])
def test_tt_observables_match_dense(name, builder):
    theta_dense, _ = solve_steady_state(build_W(builder, steady_state=True))
    theta_tt, _ = tt.solve_steady_state_tt(tt.build_W_tt(builder))

    cov_d = coverage_mean(builder, theta_dense)
    cov_tt = tt.coverage_mean_tt(builder, theta_tt)
    assert np.abs(cov_d - cov_tt).max() < 1e-7

    cd_d = coverage_distribution(builder, theta_dense)
    cd_tt = tt.coverage_distribution_tt(builder, theta_tt)
    assert np.abs(cd_d - cd_tt).max() < 1e-7


def test_production_rate_tt_matches_dense_and_langmuir_closed_form():
    """Dimerization rate through the TT path matches the dense production_rate,
    and a plain Langmuir desorption rate matches the closed form k_des*theta_A."""
    builder = build_system(100.0, FISH)
    theta_dense, _ = solve_steady_state(build_W(builder, steady_state=True))
    theta_tt, _ = tt.solve_steady_state_tt(tt.build_W_tt(builder))
    stoich = np.array([0.0, 0.0, 1.0])  # count dimerization events
    r_d = production_rate(builder, theta_dense, stoich)
    r_tt = tt.production_rate_tt(builder, theta_tt, stoich)
    assert abs(r_d - r_tt) < 1e-8

    # Langmuir: desorption rate = k_des * theta_A (closed form)
    lang = langmuir(GREEK, 0.0, k_ads=2.0, k_des=0.7)
    theta_l_tt, _ = tt.solve_steady_state_tt(tt.build_W_tt(lang))
    r_des = tt.production_rate_tt(lang, theta_l_tt, np.array([0.0, 1.0]))
    theta_A = tt.coverage_mean_tt(lang, theta_l_tt)[1]
    assert abs(r_des - 0.7 * theta_A) < 1e-7


# ===========================================================================
# 3. Analytic dTheta/dbeta vs dense steady_state_derivative
# ===========================================================================


def test_steady_state_derivative_tt_matches_dense():
    builder = langmuir(GREEK, -0.9)

    # dense analytic dTheta/dbeta (reuse the LU of the steady-state W)
    W_bar = build_W(builder, steady_state=True)
    theta_dense, lu = solve_steady_state(W_bar)
    rates = [r.rate for r in builder.get_reactions()]
    dWbar = sum(k * c for k, c in zip(rates, build_dW_dbeta_components(builder)))
    dWbar_ss = dWbar.tolil()
    dWbar_ss[-1, :] = 0.0  # normalization row is constant -> zero derivative
    dtheta_dense = steady_state_derivative(lu, sp.csc_array(dWbar_ss), theta_dense)

    # TT analytic dTheta/dbeta
    W_tt = tt.build_W_tt(builder)
    dW_tt = tt.build_dW_dbeta_tt(builder)
    theta_tt, _ = tt.solve_steady_state_tt(W_tt)
    dtheta_tt = tt.steady_state_derivative_tt(W_tt, dW_tt, theta_tt)
    dtheta_from_tt = tt.tt_to_dense(dtheta_tt)

    assert np.abs(dtheta_dense - dtheta_from_tt).sum() < 1e-6
    assert abs(dtheta_from_tt.sum()) < 1e-8  # gauge <1, dTheta> = 0


# ===========================================================================
# 4. Warm-started sweep: matches dense pointwise, bounded ranks
# ===========================================================================


def test_sweep_matches_dense_and_ranks_bounded():
    K_values = np.geomspace(50.0, 2000.0, 6)  # across the checkerboard plateau
    results = tt.sweep_steady_state_tt(
        lambda K: build_system(K, FISH), K_values, max_rank=40
    )
    for (K, theta_tt, info), K2 in zip(results, K_values):
        builder = build_system(K2, FISH)
        theta_dense, _ = solve_steady_state(build_W(builder, steady_state=True))
        cov_d = coverage_mean(builder, theta_dense)[1]
        cov_tt = tt.coverage_mean_tt(builder, theta_tt)[1]
        assert abs(cov_d - cov_tt) < 1e-6
        assert max(info.ranks) <= 40  # theta rank stays bounded on the plateau
