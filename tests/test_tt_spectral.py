"""
Slow eigenpair tests for the TT spectral eigensolver (optional scikit_tt
dependency), checked against a dense/sparse reference.
"""

import numpy as np
import pytest

from me_mkm import MEMKMBuilder, Reaction, TileSettings
from me_mkm.sparse import build_W

tt = pytest.importorskip("me_mkm.tt", exc_type=ImportError)


def _langmuir(sites=5, d=2, k_ads=1.3, k_des=0.7):
    tile = TileSettings.square(sites=sites, d=d)
    rxns = [
        Reaction([0], [1], rate=k_ads, name="ads"),
        Reaction([1], [0], rate=k_des, name="des"),
    ]
    return MEMKMBuilder(tile_settings=tile, reactions=rxns, species_names=["*", "A"])


def test_slow_eigenpairs_tt_stationary_mode_matches_dense():
    # The stationary mode (eigenvalue 0) is the one mode this local ALS
    # eigensolver reliably nails; higher modes can be arbitrarily close
    # together (translational symmetry of a periodic tile clusters
    # eigenvalues into degenerate groups) and are the genuinely hard case
    # documented in tt_spectral_eigendecomposition.md -- that's what the
    # per-eigenpair residual is for, not something a single seeded test
    # should assert tight convergence on.
    builder = _langmuir()
    W = build_W(builder, steady_state=False)
    vals = np.linalg.eigvals(W.toarray())
    lam1_dense = vals[np.argmax(vals.real)].real
    assert lam1_dense == pytest.approx(0.0, abs=1e-8)

    W_tt = tt.build_W_tt(builder)
    eigenvalues, eigentensors, residuals = tt.slow_eigenpairs_tt(W_tt, k=2, repeats=40)

    assert len(eigenvalues) == 2
    assert len(eigentensors) == 2
    assert len(residuals) == 2
    assert eigenvalues[0].real == pytest.approx(0.0, abs=1e-6)
    assert residuals[0] < 1e-6


def test_left_slow_eigenpairs_tt_is_transpose_wrapper():
    builder = _langmuir()
    W_tt = tt.build_W_tt(builder)
    left_vals, _, left_residuals = tt.left_slow_eigenpairs_tt(W_tt, k=1, repeats=40)
    direct_vals, _, direct_residuals = tt.slow_eigenpairs_tt(
        W_tt.transpose(), k=1, repeats=40
    )
    assert left_vals[0].real == pytest.approx(direct_vals[0].real, abs=1e-6)
    assert left_residuals[0] < 1e-6
    assert direct_residuals[0] < 1e-6
