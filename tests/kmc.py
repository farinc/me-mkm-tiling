"""
Gillespie KMC for single-adsorbate lattice-gas systems.

Uses the same conventions as the Rust MEMKMBuilder:
  - Directed neighbor list: neighbors[i] = [(i+d) % l for d in topo.deltas]
  - No interaction correction on adsorption (eps[0][*] = 0)
  - Desorption correction: rate = k_des * exp(-n_fwd_A * eps / kBT), kBT = 1
  - eps > 0: attractive (suppresses desorption)
  - eps < 0: repulsive  (enhances desorption)
"""

import numpy as np


def _neighbor_table(topo, l):
    """Undirected neighbor list padded for masked indexing."""
    nbr_sets = [set() for _ in range(l)]
    for i in range(l):
        for d in topo.deltas:
            j = (i + d) % l
            if i != j:
                nbr_sets[i].add(j)
                nbr_sets[j].add(i)
    max_nbrs = max(len(s) for s in nbr_sets)
    nbrs = np.full((l, max_nbrs), -1, dtype=np.intp)
    for i, s in enumerate(nbr_sets):
        row = sorted(s)
        nbrs[i, :len(row)] = row
    valid = nbrs >= 0
    return nbrs, valid


def _occupied_neighbor_counts(state, nbrs, valid, l):
    occ_extended = np.zeros(l + 1, dtype=np.int8)
    occ_extended[:l] = state
    nbr_idx = np.where(valid, nbrs, l)
    return (occ_extended[nbr_idx] * valid).sum(axis=1).astype(float)


def run_kmc(
    topo,
    l: int,
    K: float,
    eps: float = 0.0,
    k_des: float = 1.0,
    n_steps: int = 500_000,
    seed: int = 42,
) -> float:
    """
    Gillespie KMC for single-adsorbate adsorption/desorption.

    Parameters
    ----------
    topo     : Topology object (provides .deltas)
    l        : number of sites
    K        : equilibrium constant K[A] = k_ads / k_des
    eps      : A-A interaction energy in kBT units (eps[1][1] in InteractionModel)
    k_des    : bare desorption rate (default 1.0)
    n_steps  : number of KMC events
    seed     : RNG seed

    Returns
    -------
    theta : time-averaged fractional coverage
    """
    rng = np.random.default_rng(seed)
    k_ads = K * k_des
    nbrs, valid = _neighbor_table(topo, l)

    state = np.zeros(l, dtype=np.int8)
    occ_time = 0.0
    total_time = 0.0

    for _ in range(n_steps):
        n_occ_nbrs = _occupied_neighbor_counts(state, nbrs, valid, l)
        rates = np.where(
            state == 0,
            k_ads,
            k_des * np.exp(-n_occ_nbrs * eps),
        )

        total_rate = rates.sum()
        dt = rng.exponential(1.0 / total_rate)
        occ_time += float(state.sum()) * dt
        total_time += dt

        site = rng.choice(l, p=rates / total_rate)
        state[site] ^= 1  # flip occupied ↔ empty

    return occ_time / (total_time * l)


def run_kmc_curve(topo, l, K_values, eps=0.0, k_des=1.0, n_steps=500_000, seed=42):
    """Return KMC coverage at each K in K_values (independent seeds)."""
    return np.array([
        run_kmc(topo, l, K, eps=eps, k_des=k_des, n_steps=n_steps, seed=seed + i)
        for i, K in enumerate(K_values)
    ])


def run_kmc_dynamic_trajectory(topo, l, k_ads_func, k_des, t_eval, eps=0.0, seed=42, theta0=0.0):
    """
    Gillespie trajectory for single-adsorbate ads/des with a time-dependent
    adsorption rate k_ads_func(t) (e.g. driven by an oscillating gas pressure).

    Rates are recomputed from the current time at the start of each waiting-time
    draw — exact for piecewise-constant rates, and an accurate approximation
    for smoothly varying ones provided the oscillation period is long compared
    to the mean time between events (many events occur per period).

    Parameters
    ----------
    t_eval : array-like, sorted, non-negative
        Times at which to sample the (piecewise-constant) state.
    theta0 : float
        Initial fractional coverage (0–1). Exactly round(theta0 * l) randomly
        chosen sites are set to occupied; remaining sites are empty.

    Returns
    -------
    theta_t : ndarray, shape (len(t_eval),)
    """
    rng = np.random.default_rng(seed)
    nbrs, valid = _neighbor_table(topo, l)
    t_eval = np.asarray(t_eval, dtype=float)
    t_max = t_eval[-1]

    state = np.zeros(l, dtype=np.int8)
    n_occ = int(round(theta0 * l))
    if n_occ > 0:
        state[rng.choice(l, size=n_occ, replace=False)] = 1
    t = 0.0
    theta_t = np.empty(len(t_eval))
    ei = 0

    while t < t_max:
        k_ads = k_ads_func(t)
        n_occ_nbrs = _occupied_neighbor_counts(state, nbrs, valid, l)
        rates = np.where(state == 0, k_ads, k_des * np.exp(-n_occ_nbrs * eps))
        total_rate = rates.sum()
        dt = rng.exponential(1.0 / total_rate)

        while ei < len(t_eval) and t_eval[ei] < t + dt:
            theta_t[ei] = state.sum() / l
            ei += 1

        t += dt
        site = rng.choice(l, p=rates / total_rate)
        state[site] ^= 1

    while ei < len(t_eval):
        theta_t[ei] = state.sum() / l
        ei += 1

    return theta_t


def run_kmc_dynamic_ensemble(topo, l, k_ads_func, k_des, t_eval, n_trials, eps=0.0, seed=42, theta0=0.0):
    """Stack n_trials independent dynamic-kMC trajectories: shape (n_trials, len(t_eval))."""
    return np.array([
        run_kmc_dynamic_trajectory(topo, l, k_ads_func, k_des, t_eval, eps=eps, seed=seed + i, theta0=theta0)
        for i in range(n_trials)
    ])
