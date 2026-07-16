"""
Gillespie KMC for single-adsorbate lattice-gas systems.

Uses the same conventions as the Rust MEMKMBuilder:
  - Directed neighbor list: neighbors[i] = [(i+d) % l for d in topo.deltas]
  - No interaction correction on adsorption (eps[0][*] = 0)
  - Desorption correction: rate = k_des * exp(-n_fwd_A * eps / kBT), kBT = 1
  - eps > 0: attractive (suppresses desorption)
  - eps < 0: repulsive  (enhances desorption)

Event selection uses the n-fold method with local update (Bortz, Kalos &
Lebowitz 1975; Gibson & Bruck 2000; Schulze 2002): every possible event
instance (a site process or a bond/pair reaction) belongs to a class of
instances sharing an identical rate. Each step draws a class then an
instance within it (exact -- see BKL Eqs. 4/10/39/40), and firing an event
only reclassifies the bounded set of instances whose rate its local
neighborhood invalidates, never a full-array rescan. See
`_ClassBuckets` below for the partition-array/address-array bookkeeping
that makes reclassification O(1).
"""

import math

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


def _adjacency_lists(nbrs, valid):
    """Plain Python adjacency lists, one per site.

    Never loop the raw -1-padded `nbrs` array directly: the -1 sentinel
    reads as a numpy negative index and silently wires in a phantom
    neighbor at site l-1 for any under-degree site.
    """
    return [nbrs[i, valid[i]].tolist() for i in range(nbrs.shape[0])]


def _neighbor_bond_list(topo, l):
    """Each undirected neighbor bond (i, j), i<j, once -- for pair reactions."""
    bonds = sorted({
        (min(i, (i + d) % l), max(i, (i + d) % l))
        for i in range(l) for d in topo.deltas if i != (i + d) % l
    })
    return np.array(bonds, dtype=np.intp)


def _site_to_bonds(bonds, l):
    """Reverse map: site -> list of incident bond indices."""
    out = [[] for _ in range(l)]
    for b, (i, j) in enumerate(bonds.tolist()):
        out[int(i)].append(b)
        out[int(j)].append(b)
    return out


_RESYNC_EVERY = 20_000  # bound float drift in Q_tot over long (500k-step) runs


class _ClassBuckets:
    """
    n-fold class/instance selection with O(1) local update: the
    partition-array (BKL's LOC) + address-array (BKL's LOOK) design of
    Bortz, Kalos & Lebowitz (1975), generalized per Schulze (2002).

    Every instance belongs to exactly one class; all instances in a class
    share an identical rate. Class keys must be stable symbolic labels,
    never raw rate values -- two physically distinct classes can share a
    numeric rate (e.g. symmetric kA == kB == kAB), which would collide if
    the rate value itself were used as the dict key. Each instance also
    carries an "action" payload in a side-table, looked up at selection
    time -- this lets several instances in one class fire different
    outcomes (e.g. different exchange targets) despite sharing a rate.
    """

    __slots__ = (
        "_bucket", "_pos", "_class_of", "_action", "_rate", "R", "Q_tot",
        "_since_resync",
    )

    def __init__(self):
        self._bucket = {}    # class_key -> list[instance_id]
        self._pos = {}       # instance_id -> index within its bucket
        self._class_of = {}  # instance_id -> class_key
        self._action = {}    # instance_id -> action payload
        self._rate = {}      # class_key -> rate value
        self.R = {}          # class_key -> n_j * rate_j
        self.Q_tot = 0.0
        self._since_resync = 0

    def set_rate(self, class_key, rate):
        """Set (or replace) the shared rate for a class in O(1)."""
        old_R = self.R.get(class_key, 0.0)
        self._rate[class_key] = rate
        n_j = len(self._bucket.get(class_key, ()))
        new_R = n_j * rate
        self.R[class_key] = new_R
        self.Q_tot += new_R - old_R

    def add(self, instance_id, class_key, action=None):
        bucket = self._bucket.setdefault(class_key, [])
        self._pos[instance_id] = len(bucket)
        bucket.append(instance_id)
        self._class_of[instance_id] = class_key
        self._action[instance_id] = action
        rate = self._rate.get(class_key, 0.0)
        self.R[class_key] = self.R.get(class_key, 0.0) + rate
        self.Q_tot += rate

    def remove(self, instance_id):
        class_key = self._class_of.pop(instance_id)
        self._action.pop(instance_id, None)
        bucket = self._bucket[class_key]
        pos = self._pos.pop(instance_id)
        last_id = bucket[-1]
        bucket[pos] = last_id
        self._pos[last_id] = pos
        bucket.pop()
        rate = self._rate.get(class_key, 0.0)
        self.R[class_key] -= rate
        self.Q_tot -= rate

    def reclassify(self, instance_id, new_class_key, action=None):
        if self._class_of.get(instance_id) == new_class_key:
            self._action[instance_id] = action
            return
        self.remove(instance_id)
        self.add(instance_id, new_class_key, action)

    def resync(self):
        """Recompute Q_tot from scratch; bounds float drift over long runs."""
        self.Q_tot = sum(self.R.values())
        self._since_resync = 0

    def maybe_resync(self):
        self._since_resync += 1
        if self._since_resync >= _RESYNC_EVERY:
            self.resync()

    def select(self, zeta1, zeta2):
        """Two-step class-then-instance draw (BKL Eqs. 39-40): exact,
        zero-bias selection with probability rate_e/Q_tot per instance."""
        target = zeta1 * self.Q_tot
        cum = 0.0
        chosen_key = None
        for class_key, R_j in self.R.items():
            if R_j <= 0.0:
                continue
            cum += R_j
            chosen_key = class_key
            if cum >= target:
                break
        bucket = self._bucket[chosen_key]
        n_j = len(bucket)
        m = int(zeta2 * n_j)
        if m >= n_j:  # float-rounding guard: zeta2 arbitrarily close to 1.0
            m = n_j - 1
        instance_id = bucket[m]
        return instance_id, self._action[instance_id]


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
    eps      : A-A interaction energy in kBT units (eps[1][1] in InitialStateInteraction)
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
    adj = _adjacency_lists(nbrs, valid)
    max_coord = max(len(a) for a in adj)

    state = np.zeros(l, dtype=np.int8)
    n_occ_nbrs = np.zeros(l, dtype=np.intp)

    buckets = _ClassBuckets()
    buckets.set_rate(("A",), k_ads)
    for n in range(max_coord + 1):
        buckets.set_rate(("D", n), k_des * math.exp(-n * eps))
    for i in range(l):
        buckets.add(i, ("A",))  # all sites start vacant

    n_occ = 0
    occ_time = 0.0
    total_time = 0.0

    zetas = rng.random((n_steps, 3))
    for step in range(n_steps):
        zeta1, zeta2, zeta3 = zetas[step]
        site, _ = buckets.select(zeta1, zeta2)
        dt = -math.log(zeta3) / buckets.Q_tot
        occ_time += n_occ * dt
        total_time += dt

        if state[site] == 0:
            state[site] = 1
            n_occ += 1
            delta = 1
        else:
            state[site] = 0
            n_occ -= 1
            delta = -1

        for j in adj[site]:
            n_occ_nbrs[j] += delta
            if state[j] == 1:
                buckets.reclassify(j, ("D", int(n_occ_nbrs[j])))
        buckets.reclassify(
            site, ("A",) if state[site] == 0 else ("D", int(n_occ_nbrs[site]))
        )
        buckets.maybe_resync()

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
    to the mean time between events (many events occur per period). Since all
    vacant sites always share one n-fold class, refreshing k_ads(t) each step
    is an O(1) class-rate update (`_ClassBuckets.set_rate`), not a per-site
    rescan.

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
    adj = _adjacency_lists(nbrs, valid)
    max_coord = max(len(a) for a in adj)
    t_eval = np.asarray(t_eval, dtype=float)
    t_max = t_eval[-1]

    state = np.zeros(l, dtype=np.int8)
    n_occ = int(round(theta0 * l))
    if n_occ > 0:
        state[rng.choice(l, size=n_occ, replace=False)] = 1
    n_occ_nbrs = _occupied_neighbor_counts(state, nbrs, valid, l).astype(np.intp)

    buckets = _ClassBuckets()
    for n in range(max_coord + 1):
        buckets.set_rate(("D", n), k_des * math.exp(-n * eps))
    buckets.set_rate(("A",), k_ads_func(0.0))
    for i in range(l):
        if state[i] == 0:
            buckets.add(i, ("A",))
        else:
            buckets.add(i, ("D", int(n_occ_nbrs[i])))

    t = 0.0
    theta_t = np.empty(len(t_eval))
    ei = 0

    while t < t_max:
        buckets.set_rate(("A",), k_ads_func(t))

        zeta1, zeta2, zeta3 = rng.random(3)
        site, _ = buckets.select(zeta1, zeta2)
        dt = -math.log(zeta3) / buckets.Q_tot

        while ei < len(t_eval) and t_eval[ei] < t + dt:
            theta_t[ei] = n_occ / l
            ei += 1

        t += dt
        if state[site] == 0:
            state[site] = 1
            n_occ += 1
            delta = 1
        else:
            state[site] = 0
            n_occ -= 1
            delta = -1

        for j in adj[site]:
            n_occ_nbrs[j] += delta
            if state[j] == 1:
                buckets.reclassify(j, ("D", int(n_occ_nbrs[j])))
        buckets.reclassify(
            site, ("A",) if state[site] == 0 else ("D", int(n_occ_nbrs[site]))
        )
        buckets.maybe_resync()

    while ei < len(t_eval):
        theta_t[ei] = n_occ / l
        ei += 1

    return theta_t


def run_kmc_dynamic_ensemble(topo, l, k_ads_func, k_des, t_eval, n_trials, eps=0.0, seed=42, theta0=0.0):
    """Stack n_trials independent dynamic-kMC trajectories: shape (n_trials, len(t_eval))."""
    return np.array([
        run_kmc_dynamic_trajectory(topo, l, k_ads_func, k_des, t_eval, eps=eps, seed=seed + i, theta0=theta0)
        for i in range(n_trials)
    ])


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
        (matches me_mkm.observables.production_rate's 1/L normalisation, i.e.
        paper eq. 4/5 -- NOT per-bond or per-A2-molecule-times-two).
    """
    rng = np.random.default_rng(seed)
    k_ads = K * k_des
    nbrs, valid = _neighbor_table(topo, l)
    adj = _adjacency_lists(nbrs, valid)
    max_coord = max(len(a) for a in adj)
    bonds = _neighbor_bond_list(topo, l)
    site_bonds = _site_to_bonds(bonds, l)
    bond_i = bonds[:, 0]
    bond_j = bonds[:, 1]
    n_bonds = len(bonds)

    state = np.zeros(l, dtype=np.int8)
    n_occ_nbrs = np.zeros(l, dtype=np.intp)

    buckets = _ClassBuckets()
    buckets.set_rate(("A",), k_ads)
    for n in range(max_coord + 1):
        buckets.set_rate(("D", n), k_des * math.exp(-n * eps))
    buckets.set_rate(("B", 1), krxn)
    buckets.set_rate(("B", 0), 0.0)

    for i in range(l):
        buckets.add(("site", i), ("A",))
    for b in range(n_bonds):
        buckets.add(("bond", b), ("B", 0))  # all inactive: all sites start vacant

    def refresh_bond(b):
        i, j = int(bond_i[b]), int(bond_j[b])
        active = state[i] == 1 and state[j] == 1
        buckets.reclassify(("bond", b), ("B", 1) if active else ("B", 0))

    burn_in_steps = int(n_steps * burn_in_frac)
    n_occ = 0
    occ_time = 0.0
    total_time = 0.0
    rxn_events = 0

    zetas = rng.random((n_steps, 3))
    for step in range(n_steps):
        zeta1, zeta2, zeta3 = zetas[step]
        instance_id, _ = buckets.select(zeta1, zeta2)
        dt = -math.log(zeta3) / buckets.Q_tot

        if step >= burn_in_steps:
            occ_time += n_occ * dt
            total_time += dt

        kind, idx = instance_id
        if kind == "bond":
            b = idx
            i, j = int(bond_i[b]), int(bond_j[b])
            state[i] = 0
            state[j] = 0
            n_occ -= 2  # both endpoints were occupied (bond was class ("B",1))

            for k in adj[i]:
                n_occ_nbrs[k] -= 1
            for k in adj[j]:
                n_occ_nbrs[k] -= 1
            for s in (i, j):
                buckets.reclassify(("site", s), ("A",))
            for k in set(adj[i]) | set(adj[j]):
                if state[k] == 1:
                    buckets.reclassify(("site", k), ("D", int(n_occ_nbrs[k])))
            for b2 in set(site_bonds[i]) | set(site_bonds[j]):
                refresh_bond(b2)

            if step >= burn_in_steps:
                rxn_events += 1
        else:
            site = idx
            if state[site] == 0:
                state[site] = 1
                n_occ += 1
                delta = 1
            else:
                state[site] = 0
                n_occ -= 1
                delta = -1

            for k in adj[site]:
                n_occ_nbrs[k] += delta
                if state[k] == 1:
                    buckets.reclassify(("site", k), ("D", int(n_occ_nbrs[k])))
            buckets.reclassify(
                ("site", site),
                ("A",) if state[site] == 0 else ("D", int(n_occ_nbrs[site])),
            )
            for b2 in site_bonds[site]:
                refresh_bond(b2)

        buckets.maybe_resync()

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

    No lateral (neighbor-count-dependent) coupling exists in this model:
    single-site exchange rates depend only on the site's own current state,
    and pair-reaction rates depend only on the bond's two endpoint states.
    So the n-fold dependency graph for any fired event is just "that
    event's own site(s)' exchange instances" plus "bonds incident to
    those site(s)" -- no separate occupied-neighbor-count bookkeeping is
    needed here (contrast run_kmc_dimer_steady_state).

    theta0 : (theta_A, theta_B) initial coverages; round(theta_X * l) sites
        of each species are placed at random, remainder vacant (matches
        independent_site_distribution's max-entropy convention on the Python
        ME-MKM side).

    Returns
    -------
    cov_t : ndarray, shape (len(t_eval), 3)
        Instantaneous (piecewise-constant) fractional coverage of [*, A*, B*]
        sampled at each t_eval.
    """
    rng = np.random.default_rng(seed)
    bonds = _neighbor_bond_list(topo, l)
    bond_i, bond_j = bonds[:, 0], bonds[:, 1]
    site_bonds = _site_to_bonds(bonds, l)
    n_bonds = len(bonds)

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

    # Single-site exchange: current species -> [(target, rate, rate_label), ...].
    site_opts = {
        0: ((1, kA, "kA"), (2, kB, "kB")),
        1: ((0, kA, "kA"), (2, kAB, "kAB")),
        2: ((0, kB, "kB"), (1, kAB, "kAB")),
    }
    # Pair growth/invasion rules: unordered {a, b} -> both sites become `out`.
    pair_rules = ((1, 0, 1, k1), (1, 2, 2, k2), (2, 0, 0, k3))

    def match_pair(a, b):
        for idx, (ra, rb, out, _rate) in enumerate(pair_rules):
            if (a, b) == (ra, rb) or (a, b) == (rb, ra):
                return idx, out
        return None, None

    buckets = _ClassBuckets()
    buckets.set_rate(("SX", "kA"), kA)
    buckets.set_rate(("SX", "kB"), kB)
    buckets.set_rate(("SX", "kAB"), kAB)
    buckets.set_rate(("PAIR", 0), k1)
    buckets.set_rate(("PAIR", 1), k2)
    buckets.set_rate(("PAIR", 2), k3)
    buckets.set_rate(("PAIR", None), 0.0)

    def refresh_site_sx(i):
        s = int(state[i])
        for k in (0, 1):
            tgt, _rate, label = site_opts[s][k]
            buckets.reclassify(("sx", i, k), ("SX", label), action=tgt)

    def refresh_bond(b):
        i, j = int(bond_i[b]), int(bond_j[b])
        idx, out = match_pair(int(state[i]), int(state[j]))
        buckets.reclassify(("bond", b), ("PAIR", idx), action=out)

    for i in range(l):
        s = int(state[i])
        for k in (0, 1):
            tgt, _rate, label = site_opts[s][k]
            buckets.add(("sx", i, k), ("SX", label), action=tgt)
    for b in range(n_bonds):
        i, j = int(bond_i[b]), int(bond_j[b])
        idx, out = match_pair(int(state[i]), int(state[j]))
        buckets.add(("bond", b), ("PAIR", idx), action=out)

    def record(idx):
        cov_t[idx] = np.bincount(state, minlength=3) / l

    while t < t_max:
        zeta1, zeta2, zeta3 = rng.random(3)
        instance_id, action = buckets.select(zeta1, zeta2)
        dt = -math.log(zeta3) / buckets.Q_tot

        while ei < len(t_eval) and t_eval[ei] < t + dt:
            record(ei)
            ei += 1

        t += dt
        kind = instance_id[0]
        if kind == "sx":
            _, site, _k = instance_id
            state[site] = action
            refresh_site_sx(site)
            for b in site_bonds[site]:
                refresh_bond(b)
        else:  # "bond"
            _, b = instance_id
            i, j = int(bond_i[b]), int(bond_j[b])
            state[i] = action
            state[j] = action
            refresh_site_sx(i)
            refresh_site_sx(j)
            for b2 in set(site_bonds[i]) | set(site_bonds[j]):
                refresh_bond(b2)

        buckets.maybe_resync()

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
