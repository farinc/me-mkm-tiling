"""
Tensor-train (TT / MPS-MPO) backend for ME-MKM.

An *optional* alternative to the dense/sparse steady-state path
(me_mkm.sparse). Instead of enumerating all
n_species**l microstates and factorizing an s x s generator, this backend

- builds the generator W directly as a tensor-train operator (MPO) from the
  reaction list and tile geometry, with no state enumeration, and
- solves W Theta = 0 for the stationary distribution entirely in TT format,

so cost is polynomial in the tile length l instead of exponential. This is the
approach of Gelss et al. 2016 (see tt_method_notes.md); it targets tile sizes
and species counts the dense path cannot reach.

Nothing here is imported by the base package. It depends on `scikit_tt`, which
lives in the optional `tt` uv dependency group:

    uv sync --group tt

then

    from me_mkm import tt
    W_tt = tt.build_W_tt(builder)
    theta_tt, info = tt.solve_steady_state_tt(W_tt)
"""

try:
    import scikit_tt.tensor_train  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "me_mkm.tt requires the optional 'scikit_tt' dependency, which is not "
        "installed. Install the tt dependency group:\n\n"
        "    uv sync --group tt\n\n"
        "(scikit_tt has no PyPI release, so it is pinned to a git commit in "
        "pyproject.toml's [dependency-groups].)"
    ) from exc

from me_mkm.tt.convert import (
    mpo_to_dense,
    ones_tt,
    product_state_tt,
    rank1_operator,
    rank1_vector,
    tt_inner,
    tt_normalize_prob,
    tt_to_dense,
    unit_tt,
)
from me_mkm.tt.observables import (
    coverage_distribution_tt,
    coverage_mean_tt,
    production_rate_tt,
    site_marginals,
)
from me_mkm.tt.operator import (
    build_dW_dbeta_tt,
    build_W_tt,
    build_W_tt_components,
)
from me_mkm.tt.solve import (
    TTSolveInfo,
    solve_steady_state_tt,
    steady_state_derivative_tt,
    steady_state_residual,
    sweep_steady_state_tt,
)

__all__ = [
    # convert
    "rank1_operator",
    "rank1_vector",
    "ones_tt",
    "unit_tt",
    "product_state_tt",
    "tt_to_dense",
    "mpo_to_dense",
    "tt_inner",
    "tt_normalize_prob",
    # operator
    "build_W_tt",
    "build_W_tt_components",
    "build_dW_dbeta_tt",
    # solve
    "TTSolveInfo",
    "solve_steady_state_tt",
    "steady_state_residual",
    "steady_state_derivative_tt",
    "sweep_steady_state_tt",
    # observables
    "site_marginals",
    "coverage_mean_tt",
    "production_rate_tt",
    "coverage_distribution_tt",
]
