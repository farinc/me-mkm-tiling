"""
Tensor-train (TT / MPS-MPO) backend for ME-MKM.

An *optional* alternative to the dense/sparse steady-state path
(me_mkm.sparse). Instead of enumerating all s = n_species**l microstates
and factorizing an s x s generator, this backend

- builds the generator W directly as a tensor-train operator (MPO) from the
  reaction list and tile geometry, with no state enumeration, and
- solves W Theta = 0 for the stationary distribution entirely in TT format,
- Provides committor functionality for a 2-basin system.

so cost is polynomial in the tile length l instead of exponential. This is
inspired from the approach of Gelss et al. 2016. It does have downsides,
particularly for cases where interactions become longer ranged.
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
from me_mkm.tt.committor import (
    committor_tt,
    committor_tt_residual,
    threshold_projector_tt,
)
from me_mkm.tt.observables import (
    committor_class_profile_tt,
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
    # committor
    "committor_tt",
    "committor_tt_residual",
    "threshold_projector_tt",
    "committor_class_profile_tt",
]
