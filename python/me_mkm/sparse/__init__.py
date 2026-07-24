"""
Dense/sparse (scipy) backend for ME-MKM.

Enumerates all n_species**l microstates and build the generator W as a scipy sparse
matrix (me_mkm.sparse.generator). Provides functions for observables all using sparse
functions.
"""

try:
    import scipy.sparse  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "me_mkm.sparse requires the optional 'scipy' dependency, which is not "
        "installed. Install the scipy dependency group:\n\n"
        "    uv sync --group scipy\n"
    ) from exc

from me_mkm.sparse.generator import (
    assemble_dW_dbeta,
    assemble_dW_dlnC,
    assemble_W,
    build_dW_dbeta_components,
    build_W,
    build_W_components,
    build_W_operator,
    to_steady_state_derivative_form,
    to_steady_state_form,
)
from me_mkm.sparse.observables import (
    production_rate,
    production_rate_dbeta_vector,
    production_rate_derivative,
    production_rate_dlnC_vector,
    production_rate_vector,
)
from me_mkm.sparse.committor import (
    committor,
    committor_backward,
)
from me_mkm.sparse.metastable import (
    quasi_stationary_distribution,
)
from me_mkm.sparse.steady_state import (
    solve_steady_state,
    steady_state_derivative,
)

__all__ = [
    # generator
    "build_W",
    "build_W_components",
    "build_W_operator",
    "build_dW_dbeta_components",
    "assemble_W",
    "assemble_dW_dbeta",
    "assemble_dW_dlnC",
    "to_steady_state_form",
    "to_steady_state_derivative_form",
    # steady_state
    "solve_steady_state",
    "steady_state_derivative",
    # committor
    "committor",
    "committor_backward",
    # metastable
    "quasi_stationary_distribution",
    # observables
    "production_rate_vector",
    "production_rate_dbeta_vector",
    "production_rate_dlnC_vector",
    "production_rate",
    "production_rate_derivative",
]
