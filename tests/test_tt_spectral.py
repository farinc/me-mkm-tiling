"""
Slow eigenpair tests for the TT spectral eigensolvers (optional scikit_tt
dependency), checked against a dense/sparse reference.

Two solvers are covered (see me_mkm/tt/spectral.py and
tt_spectral_deflation_followup.md):
  - slow_eigenpairs_tt / left_slow_eigenpairs_tt: the default deflated
    shift-invert scheme, including the stiff-timescale-separation case
    (evp.als's documented failure mode) and the two-sided k=3 case.
  - slow_eigenpairs_tt_als / left_slow_eigenpairs_tt_als: the original
    evp.als block eigensolver, kept for milder regimes.
"""

import numpy as np
import pytest
from me_mkm import MEMKMBuilder, Reaction, TileSettings
from me_mkm.sparse import build_W

tt = pytest.importorskip("me_mkm.tt", exc_type=ImportError)

from test_checkboard_superlattice import build_system  # noqa: E402


def _langmuir(sites=5, d=2, k_ads=1.3, k_des=0.7):
    tile = TileSettings.square(sites=sites, d=d)
    rxns = [
        Reaction([0], [1], rate=k_ads, name="ads"),
        Reaction([1], [0], rate=k_des, name="des"),
    ]
    return MEMKMBuilder(tile_settings=tile, reactions=rxns, species_names=["*", "A"])


def _dense_eigs_sorted(M: np.ndarray) -> np.ndarray:
    """Eigenvalues of M sorted by descending real part (index 0 = nearest
    zero for a generator's own spectrum, or its transpose's)."""
    vals = np.linalg.eigvals(M)
    return vals[np.argsort(-vals.real)]


# ===========================================================================
# evp.als-based solver (slow_eigenpairs_tt_als / left_slow_eigenpairs_tt_als)
# ===========================================================================


def test_slow_eigenpairs_tt_als_stationary_mode_matches_dense():
    # The stationary mode (eigenvalue 0) is the one mode this local ALS
    # eigensolver reliably nails; higher modes can be arbitrarily close
    # together (translational symmetry of a periodic tile clusters
    # eigenvalues into degenerate groups) and are the genuinely hard case
    # tt_spectral_deflation_followup.md documents -- that's what
    # slow_eigenpairs_tt (deflated shift-invert) below is for.
    builder = _langmuir()
    W = build_W(builder, steady_state=False)
    vals = np.linalg.eigvals(W.toarray())
    lam1_dense = vals[np.argmax(vals.real)].real
    assert lam1_dense == pytest.approx(0.0, abs=1e-8)

    W_tt = tt.build_W_tt(builder)
    eigenvalues, eigentensors, residuals = tt.slow_eigenpairs_tt_als(
        W_tt, k=2, repeats=40
    )

    assert len(eigenvalues) == 2
    assert len(eigentensors) == 2
    assert len(residuals) == 2
    assert eigenvalues[0].real == pytest.approx(0.0, abs=1e-6)
    assert residuals[0] < 1e-6


def test_left_slow_eigenpairs_tt_als_is_transpose_wrapper():
    builder = _langmuir()
    W_tt = tt.build_W_tt(builder)
    left_vals, _, left_residuals = tt.left_slow_eigenpairs_tt_als(W_tt, k=1, repeats=40)
    direct_vals, _, direct_residuals = tt.slow_eigenpairs_tt_als(
        W_tt.transpose(), k=1, repeats=40
    )
    assert left_vals[0].real == pytest.approx(direct_vals[0].real, abs=1e-6)
    assert left_residuals[0] < 1e-6
    assert direct_residuals[0] < 1e-6


# ===========================================================================
# vector_outer_product_tt: exact rank-(r1*r2) MPO for |u><v|
# ===========================================================================


def test_vector_outer_product_tt_matches_dense_outer():
    builder = _langmuir()
    l, n = builder.l, builder.n_species
    W_tt = tt.build_W_tt(builder)
    theta_tt, _ = tt.solve_steady_state_tt(W_tt)
    ones_tt = tt.ones_tt(l, n)

    outer_dense = tt.mpo_to_dense(tt.vector_outer_product_tt(theta_tt, ones_tt))
    theta_dense = tt.tt_to_dense(theta_tt)
    ones_dense = tt.tt_to_dense(ones_tt)
    expected = np.outer(theta_dense, ones_dense)
    assert np.abs(outer_dense - expected).max() < 1e-10

    # the defining action (u (x) v) @ x = u * <v, x>, on a generic vector
    x_dense = np.random.default_rng(0).normal(size=theta_dense.shape)
    assert (
        np.abs(outer_dense @ x_dense - theta_dense * (ones_dense @ x_dense)).max()
        < 1e-10
    )


# ===========================================================================
# Default: deflated shift-invert (slow_eigenpairs_tt / left_slow_eigenpairs_tt)
# ===========================================================================


def test_slow_eigenpairs_tt_mode0_is_exact_and_free():
    builder = _langmuir()
    W_tt = tt.build_W_tt(builder)
    theta_tt, _ = tt.solve_steady_state_tt(W_tt)

    eigenvalues, eigentensors, residuals = tt.slow_eigenpairs_tt(
        W_tt, k=1, theta_tt=theta_tt
    )

    assert eigenvalues[0] == 0.0
    assert eigentensors[0] is theta_tt
    assert residuals[0] < 1e-6


def test_slow_eigenpairs_tt_lambda2_matches_dense():
    builder = _langmuir()
    dense_vals = _dense_eigs_sorted(build_W(builder, steady_state=False).toarray())

    W_tt = tt.build_W_tt(builder)
    theta_tt, _ = tt.solve_steady_state_tt(W_tt)
    eigenvalues, _, residuals = tt.slow_eigenpairs_tt(
        W_tt, k=2, theta_tt=theta_tt, sigma=0.0, c=1e3, repeats=15
    )

    assert eigenvalues[0] == pytest.approx(0.0, abs=1e-8)
    assert eigenvalues[1] == pytest.approx(dense_vals[1].real, rel=1e-3, abs=1e-6)
    assert residuals[1] < 1e-4


def test_left_slow_eigenpairs_tt_lambda2_matches_dense():
    builder = _langmuir()
    W = build_W(builder, steady_state=False).toarray()
    dense_left_vals = _dense_eigs_sorted(W.T)  # eigvals of W^T = left eigvals of W

    W_tt = tt.build_W_tt(builder)
    theta_tt, _ = tt.solve_steady_state_tt(W_tt)
    eigenvalues, eigentensors, residuals = tt.left_slow_eigenpairs_tt(
        W_tt, k=2, theta_tt=theta_tt, sigma=0.0, c=1e3, repeats=15
    )

    assert eigenvalues[0] == pytest.approx(0.0, abs=1e-8)
    assert eigenvalues[1] == pytest.approx(dense_left_vals[1].real, rel=1e-3, abs=1e-6)
    assert residuals[1] < 1e-4

    # eigentensors are left eigenvectors of W_tt, i.e. right eigenvectors of
    # its transpose -- check that contract directly against WT, not W.
    WT_tt = W_tt.transpose()
    resid_direct = tt.slow_eigenpair_residual(WT_tt, eigenvalues[1], eigentensors[1])
    assert resid_direct == pytest.approx(residuals[1], abs=1e-8)


def test_slow_eigenpairs_tt_stiff_timescale_separation():
    """The regression case from tt_spectral_deflation_followup.md: fast
    ads/des at rate K=1000*k_des vs. a slow dimerization reaction at
    k_des-scale, several orders of magnitude apart. evp.als
    (slow_eigenpairs_tt_als) is documented to fail in this regime; the
    deflated shift-invert scheme should not."""
    tile = TileSettings.square(sites=5, d=2)
    builder = build_system(1000.0, tile)
    dense_vals = _dense_eigs_sorted(build_W(builder, steady_state=False).toarray())

    W_tt = tt.build_W_tt(builder)
    theta_tt, info = tt.solve_steady_state_tt(W_tt)
    assert info.residual < 1e-6

    eigenvalues, _, residuals = tt.slow_eigenpairs_tt(
        W_tt, k=2, theta_tt=theta_tt, sigma=0.0, c=1e3, repeats=25, mals_repeats=3
    )

    assert eigenvalues[1] == pytest.approx(dense_vals[1].real, rel=1e-2, abs=1e-6)
    assert residuals[1] < 3e-3


def test_slow_eigenpairs_tt_lambda3_two_sided():
    """k=3 requires the two-sided scheme (tt_spectral_deflation_followup.md,
    step 5): deflating mode 1 out of W_tt needs its LEFT eigenvector too
    (found by simultaneously deflating W_tt.transpose()), not just its
    right one. Checks the result is a genuine eigenpair of W (small
    residual, eigenvalue present in the dense spectrum) -- translational
    symmetry on this tile clusters modes 1-4 at the same eigenvalue, so
    lambda_3 need not differ numerically from lambda_2, only be a valid
    eigenpair (see the _als test above for the same degeneracy caveat)."""
    builder = _langmuir()
    dense_vals = _dense_eigs_sorted(build_W(builder, steady_state=False).toarray())

    W_tt = tt.build_W_tt(builder)
    theta_tt, _ = tt.solve_steady_state_tt(W_tt)
    eigenvalues, _, residuals = tt.slow_eigenpairs_tt(
        W_tt, k=3, theta_tt=theta_tt, sigma=0.0, c=1e3, repeats=20
    )

    assert residuals[2] < 1e-3
    closest = dense_vals[np.argmin(np.abs(dense_vals.real - eigenvalues[2]))]
    assert eigenvalues[2] == pytest.approx(closest.real, rel=1e-2, abs=1e-6)
