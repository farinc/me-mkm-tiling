"""
Committor / p_fold tests for the sparse and TT backends.

The forward committor is checked against an analytically solvable birth-death
chain (linear committor q_k = k/N), the backward committor against the
equilibrium identity q^- = 1 - q, the coverage-class profile against the known
basin values, and the TT solve against the dense solve on a small tile.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from me_mkm import (
    MEMKMBuilder,
    Reaction,
    TileSettings,
    committor_class_profile,
    microstate_mask,
)
from me_mkm.sparse import (
    build_W,
    committor,
    committor_backward,
    quasi_stationary_distribution,
)


def _langmuir(sites=5, d=2, k_ads=1.3, k_des=0.7):
    tile = TileSettings.square(sites=sites, d=d)
    rxns = [
        Reaction([0], [1], rate=k_ads, name="ads"),
        Reaction([1], [0], rate=k_des, name="des"),
    ]
    return MEMKMBuilder(tile_settings=tile, reactions=rxns, species_names=["*", "A"])


# ---------------------------------------------------------------------------
# 1. Sparse forward committor vs. the exact birth-death solution
# ---------------------------------------------------------------------------


def _birth_death(n_interior, reflecting):
    """(N+1)-state nearest-neighbor chain, symmetric unit rates. If reflecting,
    the ends step inward (ergodic); otherwise 0 and N have no exits (absorbing).
    Committor with A={0}, B={N} is linear, q_k = k/N, either way."""
    N = n_interior + 1
    n = N + 1
    W = sp.lil_array((n, n))
    rng = range(n) if reflecting else range(1, N)
    for k in rng:
        if k + 1 < n:
            W[k + 1, k] += 1.0
            W[k, k] -= 1.0
        if k - 1 >= 0:
            W[k - 1, k] += 1.0
            W[k, k] -= 1.0
    in_A = np.zeros(n, bool)
    in_A[0] = True
    in_B = np.zeros(n, bool)
    in_B[N] = True
    return sp.csc_array(W), in_A, in_B, N


@pytest.mark.parametrize("reflecting", [False, True])
def test_committor_birth_death_linear(reflecting):
    W, in_A, in_B, N = _birth_death(9, reflecting)
    q = committor(W, in_A, in_B)
    assert np.allclose(q, np.arange(N + 1) / N, atol=1e-12)


def test_committor_backward_equals_one_minus_forward_at_equilibrium():
    # Reflecting symmetric chain is reversible with uniform stationary pi, so
    # the backward committor must be 1 - forward.
    W, in_A, in_B, N = _birth_death(9, reflecting=True)
    pi = np.ones(N + 1) / (N + 1)
    qf = committor(W, in_A, in_B)
    qb = committor_backward(W, in_A, in_B, pi)
    assert np.allclose(qb, 1.0 - qf, atol=1e-12)


def test_committor_rejects_bad_basins():
    W, in_A, in_B, _ = _birth_death(4, reflecting=True)
    with pytest.raises(ValueError):
        committor(W, in_A, in_A)  # A and B overlap
    empty = np.zeros_like(in_A)
    with pytest.raises(ValueError):
        committor(W, empty, in_B)  # empty basin


# ---------------------------------------------------------------------------
# 2. Basin masks and the coverage-class profile
# ---------------------------------------------------------------------------


def test_microstate_mask_matches_manual():
    builder = _langmuir()
    in_A = microstate_mask(builder, A=(None, 0.0))  # no A -> all-empty state
    in_B = microstate_mask(builder, A=(1.0, 1.0))  # full A
    manual_A = np.zeros(builder.n_states, bool)
    manual_A[0] = True
    manual_B = np.zeros(builder.n_states, bool)
    manual_B[builder.n_states - 1] = True
    assert np.array_equal(in_A, manual_A)
    assert np.array_equal(in_B, manual_B)


def test_committor_class_profile_basin_values_and_monotone():
    builder = _langmuir()
    in_A = microstate_mask(builder, A=(None, 0.0))
    in_B = microstate_mask(builder, A=(1.0, 1.0))
    W = build_W(builder, steady_state=False)
    q = committor(W, in_A, in_B)

    profile, spread = committor_class_profile(builder, q)
    l = builder.l
    # basin classes pin to 0 and 1 with zero spread
    assert profile[(0,)] == pytest.approx(0.0)
    assert profile[(l,)] == pytest.approx(1.0)
    assert spread[(0,)] == pytest.approx(0.0)
    assert spread[(l,)] == pytest.approx(0.0)
    # p_fold increases with the CO-like coverage coordinate
    means = [profile[(n,)] for n in range(l + 1)]
    assert all(b >= a - 1e-12 for a, b in zip(means, means[1:]))


# ---------------------------------------------------------------------------
# 3. Quasi-stationary distribution vs. an analytic metastable chain
# ---------------------------------------------------------------------------


def test_qsd_matches_analytic_two_state():
    # Reactive states {1,2} with internal rate a, escape 1->0 (sink) rate e.
    # The QSD is the leading eigenpair of Wtt = [[-(a+e), a], [a, -a]].
    a, e = 1.5, 0.02
    W = sp.lil_array((3, 3))
    W[2, 1] = a
    W[1, 1] -= a
    W[1, 2] = a
    W[2, 2] -= a
    W[0, 1] = e
    W[1, 1] -= e
    W = sp.csc_array(W)
    sink = np.array([True, False, False])

    nu, lam = quasi_stationary_distribution(W, sink)

    Wtt = np.array([[-(a + e), a], [a, -a]])
    vals, vecs = np.linalg.eig(Wtt)
    i = int(np.argmax(vals.real))
    va = vecs[:, i].real
    va = va * np.sign(va.sum())
    va /= va.sum()

    assert nu[0] == 0.0  # no mass on the sink
    assert np.allclose([nu[1], nu[2]], va, atol=1e-12)
    assert lam == pytest.approx(-vals[i].real, abs=1e-12)  # escape rate


def test_qsd_recovers_reactive_branch_over_stationary():
    # Two reactive states feeding a near-absorbing sink: the stationary
    # distribution collapses onto the sink, the QSD keeps the reactive mass.
    a, e = 1.0, 1e-4  # slow escape -> strong metastability
    n = 3
    W = sp.lil_array((n, n))
    W[2, 1] = a
    W[1, 1] -= a
    W[1, 2] = a
    W[2, 2] -= a
    W[0, 1] = e
    W[1, 1] -= e
    W = sp.csc_array(W)
    sink = np.array([True, False, False])
    nu, lam = quasi_stationary_distribution(W, sink)
    assert nu[1] > 0.4 and nu[2] > 0.4  # reactive mass retained
    assert lam == pytest.approx(e / 2, rel=0.2)  # escape ~ e/2 for symmetric pair


# ---------------------------------------------------------------------------
# 4. TT committor vs. dense (optional scikit_tt dependency)
# ---------------------------------------------------------------------------

tt = pytest.importorskip("me_mkm.tt", exc_type=ImportError)


def test_committor_tt_matches_dense():
    builder = _langmuir()
    l, n = builder.l, builder.n_species

    e_star = np.zeros(n)
    e_star[0] = 1.0
    e_full = np.zeros(n)
    e_full[1] = 1.0
    siteA = {p: e_star for p in range(l)}  # every site empty
    siteB = {p: e_full for p in range(l)}  # every site A

    in_A = microstate_mask(builder, A=(None, 0.0))
    in_B = microstate_mask(builder, A=(1.0, 1.0))
    q_dense = committor(build_W(builder, steady_state=False), in_A, in_B)

    W_tt = tt.build_W_tt(builder)
    q_tt = tt.committor_tt(W_tt, siteA, siteB, max_rank=40, repeats=40)
    assert tt.committor_tt_residual(W_tt, q_tt, siteA, siteB) < 1e-8
    assert np.allclose(tt.tt_to_dense(q_tt), q_dense, atol=1e-8)
