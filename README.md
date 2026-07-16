# Master-Equation Microkinetics Toolkit
This is a Rust/Python package that helps generate the transition matrix $W$ for ME-MKM provide surface transition events and tools to explore the microstates of the system. 

## Dense/sparse backend (optional)

The exact steady-state path lives in `me_mkm.sparse`: it enumerates all $(m+1)^l$ microstates, assembles $W$ as a scipy sparse matrix (`sparse.generator`), factorizes $\bar{W}$ once with SuperLU, and reuses that factorization for $\Theta_{ss}$ and every parameter derivative (`sparse.steady_state`), plus the production-rate observables that consume the sparse components (`sparse.observables`). Its `scipy` dependency is optional, in its own dependency group (the base package needs only numpy — the Rust builder and the combinatorial observables have no scipy dependency):

```bash
uv sync --group scipy
```

```python
from me_mkm import sparse

Wbar = sparse.build_W(builder)                 # scipy sparse, normalization row applied
theta_ss, lu = sparse.solve_steady_state(Wbar) # SuperLU; lu reused for derivatives
```

## Tensor-train backend (optional)

The dense/sparse path above caps the tile length $l$. `me_mkm.tt` is an *optional* alternative that never enumerates the state space: it builds $W$ directly as a tensor-train operator (MPO) from the reaction list and tile geometry, and solves $W\Theta = 0$ entirely in TT format, so cost is polynomial in $l$. Following [Gelß et al. 2016](https://doi.org/10.1016/j.jcp.2016.03.025); see `tt_method_notes.md`.

It depends on `scikit_tt`, which has no PyPI release and so lives in a pinned optional dependency group. A plain `uv sync` never installs it, and nothing in the base package imports `me_mkm.tt`:

```bash
uv sync --group tt
```

```python
from me_mkm import tt

W_tt = tt.build_W_tt(builder)                      # generator as an MPO
theta_tt, info = tt.solve_steady_state_tt(W_tt)    # stationary state, no time-stepping
info.residual, info.ranks                          # diagnostics

tt.coverage_mean_tt(builder, theta_tt)             # TT-native observables
tt.production_rate_tt(builder, theta_tt, stoich)
tt.coverage_distribution_tt(builder, theta_tt)

dW_tt = tt.build_dW_dbeta_tt(builder)              # analytic dTheta/dbeta
dtheta_tt = tt.steady_state_derivative_tt(W_tt, dW_tt, theta_tt)

tt.tt_to_dense(theta_tt)                           # bridge to the dense observables (small l)
```

Solver notes: `solve_steady_state_tt` is a rank-1-regularized ("grounded") linear solve, $A = W + c\,|u\rangle\langle 1|$ solved with MALS — the TT analog of the dense last-row normalization. It is rank-adaptive and returns an already-normalized $\Theta$; check `info.residual` for solve quality. Use `sweep_steady_state_tt` to warm-start each point of a parameter sweep from the previous solution. (An eigensolver route was tried and removed: ALS cannot grow rank past its initial guess and returned wrong vectors with small residuals on the checkerboard plateau.)

Is it worth it? TT cost is $O(l\,n\,r^2)$ where $r$ is the TT rank, so it only pays off when $r$ stays bounded. `scripts/tt_rank_probe.py` measures the exact rank of the dense stationary state (via TT-SVD) across a sweep, and shows the rank *saturates* with $l$ on these tiles rather than growing exponentially. Because dense storage is $(m+1)^l$, the crossover comes much earlier for 3+ species than for a single adsorbate — so the backend is aimed at multi-species models and tile-size convergence studies, not at the small tiles the dense path already handles instantly.

## Contributing
This project uses the `pyo3` package and associated `marturin` package builder for Python bindings using the `uv` enviroment. To get setup, make sure you got Rust sdk installed on your machine and any compiler requirements (GCC or MSVC), then clone the repository and run `uv install` to get a working Python enivroment. There exists a  Bindings are generated at commit time