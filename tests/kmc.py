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


def _neighbor_bond_list(topo, l):
    """Each undirected neighbor bond (i, j), i<j, once -- for pair reactions."""
    bonds = sorted({
        (min(i, (i + d) % l), max(i, (i + d) % l))
        for i in range(l) for d in topo.deltas if i != (i + d) % l
    })
    return np.array(bonds, dtype=np.intp)


def run_kmc_dimer_steady_state(
    topo, l, K, krxn, eps=0.0, k_des=1.0, n_steps=300_000, burn_in_frac=0.2, seed=42,
):
    """
    Gillespie kMC steady state for adsorption/desorption of A plus the
    Langmuir-Hinshelwood dimerization 2A* -> A2 + 2* (Adams et al. 2025,
    Schemes 1+2 / Figure 3): reproduces the paper's coverage AND
    dimerization-rate benchmark, not just coverage.

    Matches the Rust MEMKMBuilder convention used by the exact tile ME-MKM:
      - adsorption: rate = k_ads (uncorrected)
      - desorption: rate = k_des * exp(-n_occ_nbrs * eps)  (lateral correction)
      - dimerization: rate = krxn per occupied neighbor bond (uncorrected --
        matches build_system's Reaction(...).with_interaction(noninteracting))

    Returns
    -------
    theta : time-averaged fractional coverage (post burn-in)
    rate  : time-averaged dimerization EVENTS per site per unit time
        (matches me_mkm.tile.production_rate's 1/L normalisation, i.e.
        paper eq. 4/5 -- NOT per-bond or per-A2-molecule-times-two).
    """
    rng = np.random.default_rng(seed)
    k_ads = K * k_des
    nbrs, valid = _neighbor_table(topo, l)
    bonds = _neighbor_bond_list(topo, l)

    state = np.zeros(l, dtype=np.int8)
    burn_in_steps = int(n_steps * burn_in_frac)
    occ_time = 0.0
    total_time = 0.0
    rxn_events = 0

    for step in range(n_steps):
        n_occ_nbrs = _occupied_neighbor_counts(state, nbrs, valid, l)
        site_rates = np.where(
            state == 0, k_ads, k_des * np.exp(-n_occ_nbrs * eps),
        )
        pair_occupied = (state[bonds[:, 0]] == 1) & (state[bonds[:, 1]] == 1)
        pair_rates = np.where(pair_occupied, krxn, 0.0)

        total_site = site_rates.sum()
        total_pair = pair_rates.sum()
        total_rate = total_site + total_pair
        dt = rng.exponential(1.0 / total_rate)

        if step >= burn_in_steps:
            occ_time += float(state.sum()) * dt
            total_time += dt

        u = rng.random() * total_rate
        if u < total_site:
            site = np.searchsorted(np.cumsum(site_rates), u)
            state[site] ^= 1
        else:
            bond = np.searchsorted(np.cumsum(pair_rates), u - total_site)
            i, j = bonds[bond]
            state[i] = 0
            state[j] = 0
            if step >= burn_in_steps:
                rxn_events += 1

    theta = occ_time / (total_time * l)
    rate = rxn_events / (total_time * l)
    return theta, rate


def run_kmc_cyclic_dominance_trajectory(
    topo, l, k1, k2, k3, kA, kB, kAB, t_eval, seed=42, theta0=(0.0, 0.0),
):
    """
    Gillespie kMC for the cyclic (rock-paper-scissors-style) surface reaction
    network, matching the Rust MEMKMBuilder's pair-reaction convention
    (species 0=*, 1=A*, 2=B*; each undirected neighbor bond checked in both
    orientations):

        A* + *  -k1-> 2A*        pattern_in=[1,0] -> pattern_out=[1,1]
        A* + B* -k2-> 2B*        pattern_in=[1,2] -> pattern_out=[2,2]
        B* + *  -k3-> B(g) + 2*  pattern_in=[2,0] -> pattern_out=[0,0]

        A* <=> *   both directions rate kA (single-site: 1<->0)
        B* <=> *   both directions rate kB (single-site: 2<->0)
        A* <=> B*  both directions rate kAB (single-site: 1<->2)

    All three pair reactions have symmetric outputs (both reacting sites end
    up the same species), so unlike a disproportionation reaction, which of
    the two bond sites "started" as which reactant doesn't affect the
    outcome -- only whether the bond matches the pattern in either orientation.

    theta0 : (theta_A, theta_B) initial coverages; round(theta_X * l) sites
        of each species are placed at random, remainder vacant (matches
        coverage_ic's max-entropy/independent-site convention on the Python
        ME-MKM side).

    Returns
    -------
    cov_t : ndarray, shape (len(t_eval), 3)
        Instantaneous (piecewise-constant) fractional coverage of [*, A*, B*]
        sampled at each t_eval.
    """
    rng = np.random.default_rng(seed)
    bonds = _neighbor_bond_list(topo, l)
    bi, bj = bonds[:, 0], bonds[:, 1]

    theta_A, theta_B = theta0
    n_A = int(round(theta_A * l))
    n_B = int(round(theta_B * l))
    order = rng.permutation(l)
    state = np.zeros(l, dtype=np.int8)
    state[order[:n_A]] = 1
    state[order[n_A:n_A + n_B]] = 2

    t_eval = np.asarray(t_eval, dtype=float)
    t_max = t_eval[-1]
    cov_t = np.empty((len(t_eval), 3))
    t = 0.0
    ei = 0

    # Single-site exchange: current species -> [(target, rate), (target, rate)].
    site_opts = {
        0: ((1, kA), (2, kB)),
        1: ((0, kA), (2, kAB)),
        2: ((0, kB), (1, kAB)),
    }
    # Pair growth/invasion rules: unordered {a, b} -> both sites become `out`.
    pair_rules = ((1, 0, 1, k1), (1, 2, 2, k2), (2, 0, 0, k3))

    def record(idx):
        cov_t[idx] = np.bincount(state, minlength=3) / l

    while t < t_max:
        site_rate = np.where(
            state == 0, kA + kB, np.where(state == 1, kA + kAB, kB + kAB)
        )
        si, sj = state[bi], state[bj]
        pair_rate = np.zeros(len(bi))
        for a, b, out, rate in pair_rules:
            m = ((si == a) & (sj == b)) | ((si == b) & (sj == a))
            pair_rate = np.where(m, rate, pair_rate)

        total_site, total_pair = site_rate.sum(), pair_rate.sum()
        total_rate = total_site + total_pair
        dt = rng.exponential(1.0 / total_rate)

        while ei < len(t_eval) and t_eval[ei] < t + dt:
            record(ei)
            ei += 1

        t += dt
        u = rng.random() * total_rate
        if u < total_site:
            site = np.searchsorted(np.cumsum(site_rate), u)
            (tgt0, r0), (tgt1, r1) = site_opts[int(state[site])]
            state[site] = tgt0 if rng.random() < r0 / (r0 + r1) else tgt1
        else:
            bond = np.searchsorted(np.cumsum(pair_rate), u - total_site)
            i, j = bi[bond], bj[bond]
            a, b = int(state[i]), int(state[j])
            for pa, pb, out, _ in pair_rules:
                if {a, b} == {pa, pb}:
                    state[i] = out
                    state[j] = out
                    break

    while ei < len(t_eval):
        record(ei)
        ei += 1

    return cov_t


def run_kmc_cyclic_dominance_ensemble(
    topo, l, k1, k2, k3, kA, kB, kAB, t_eval, n_trials, seed=42, theta0=(0.0, 0.0),
):
    """Stack n_trials independent cyclic-dominance trajectories: shape (n_trials, len(t_eval), 3)."""
    return np.array([
        run_kmc_cyclic_dominance_trajectory(
            topo, l, k1, k2, k3, kA, kB, kAB, t_eval, seed=seed + i, theta0=theta0
        )
        for i in range(n_trials)
    ])
