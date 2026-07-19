# Master-Equation Microkinetics Toolkit
A Rust/Python package for ME-MKM. Build the transition matrix $W$ from surface reaction events, solve for the steady state, and explore the microstates of the system.

## Sparse backend

```bash
uv sync --group scipy
```

As most ME-MKM master-equations are extremely sparse, an efficient approach is to utilize the sparse linear system and eigenvalue solvers provided by `scipy`. The main use is the construction of COO format data which can be directly used to make a CSC matrix. 

```python
rows, cols, data = builder.build_w_coo()
W = spr.csc_array((data, (rows, cols)), shape=(builder.n_states, builder.n_states))
```

There is also a linear operator option `build_W_operator` for matrix-free methods for use in numerical solvers. 

## Tensor-train backend

```bash
uv sync --group tt
```

`me_mkm.tt` builds $W$ as a tensor-train operator (MPO) directly from the reactions and solves $W\Theta=0$ in TT format. Use the TT solv using the uv dependency group `tt`.

Everything scales in $r$, the SVD rank of $\Theta_{ss}$ (below), and $n = (m+1)$ as the number of species ($m$ are adsorbates).

| | cost |
| --- | --- |
| storage | $O(l\,n\,r^2)$ |
| solve, intrinsic | $O(l\,r^3\,n^2)$ per sweep |
| MALS solve | $O(l\,r^6\,n^6)$ |

In general TT only pays off when $r$ is bounded, which here is determined by the following parameters.

$$r \,\lesssim\, n^{\,2d}\,\cdot\,e^{\,c\,\min(d,\,\xi)}\,,\quad \xi\sim\Delta^{-1/z}$$

Here $c$ is an order-one constant, $d$ is the tile connectivity range, and $\xi$ is the spatial correlation length (the decay length of the two-point site-occupation correlation $\langle n_i n_{i+x}\rangle_c\sim e^{-x/\xi}$. $\Delta$ is the spectral gap of $W$.

- **Geometry, $n^{2d}$, dominant.** A cut severs bonds of offset up to $d$. The halves communicate only through the boundary sites within range $d$, so $r\lesssim n^{|\partial|}$. A periodic ring cuts in two places (seam and wrap), giving $|\partial|=2d$ (an open chain would give $d$). For example `deltas=[1,3]`, $n=2$ gives $2^6=64$, and the measured $r$ is about 56 to 71. $W$ itself stays low-rank, with MPO rank $O(d\cdot|\text{reactions}|)$, about 20, constant in $l$.
- **Tile length $l$.** $l$ is absent from the exponent, so $r$ saturates as the tile grows. This is the entire value of the TTF method as it trades the dense $n^l$ for $l\cdot n^{\Theta(d)}$, exponent in $d$.
- **Correlation length, $e^{c\,\xi}$.** $r$ tracks $\xi$, equivalently $1/\Delta$ where $\Delta$ is the spectral gap of $W$. It is large wherever correlations are long-ranged, which includes cases such as ordered phases, critical points, and bistable coexistence.

## Contributing
Python bindings are built with `pyo3` and `maturin` under `uv`. Install the Rust toolchain and a C compiler (MSVC or GCC), clone, and run `uv sync` for a working environment. The `.pyi` stubs should be regenerated from the Rust sources at commit time using `scripts/regen_stubs.py`.
