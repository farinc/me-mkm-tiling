use pyo3::prelude::*;

// A microstate is just a base-`base` number with `l` digits, one per site.
// decode/encode convert between that number (the row/col index used
// everywhere below) and the actual per-site species vector.

#[inline]
fn decode(mut idx: usize, l: usize, base: usize) -> Vec<u8> {
    let mut digits = vec![0u8; l];
    for d in digits.iter_mut().rev() {
        *d = (idx % base) as u8;
        idx /= base;
    }
    digits
}

#[inline]
fn encode(state: &[u8], base: usize) -> usize {
    state.iter().fold(0usize, |acc, &d| acc * base + d as usize)
}

// ─── InteractionModel ─────────────────────────────────────────────────────────

/// Pairwise nearest-neighbor interaction energies ε[s1][s2].
///
/// Rate correction for a reaction event:
///     correction = exp( -Sum_{non-reacting neighbors} ε[sp_reacting][sp_neighbor] / kBT )
///
/// The sum runs over non-reacting neighbors of each reacting site.
#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct InteractionModel {
    #[pyo3(get)] pub epsilon: Vec<Vec<f64>>,
    #[pyo3(get)] pub kbt:     f64,
    trivial: bool,
}

#[pymethods]
impl InteractionModel {
    #[new]
    #[pyo3(signature = (epsilon, kbt=1.0))]
    pub fn new(epsilon: Vec<Vec<f64>>, kbt: f64) -> Self {
        let trivial = epsilon.iter().all(|row| row.iter().all(|&e| e == 0.0));
        Self { epsilon, kbt, trivial }
    }

    #[staticmethod]
    pub fn noninteracting(n_species: usize, kbt: f64) -> Self {
        Self::new(vec![vec![0.0; n_species]; n_species], kbt)
    }

    fn __repr__(&self) -> String {
        if self.trivial {
            format!("InteractionModel(noninteracting, kBT={})", self.kbt)
        } else {
            format!("InteractionModel(epsilon={:?}, kBT={})", self.epsilon, self.kbt)
        }
    }
}

impl InteractionModel {
    /// Compute multiplicative rate correction.
    /// reacting_sites: indices into state that are changing.
    /// The correction sums epsilon[sp_reacting][sp_neighbor] over
    /// non-reacting neighbors of each reacting site.
    #[inline]
    fn rate_correction(
        &self,
        state:          &[u8],
        reacting_sites: &[usize],
        neighbors:      &[Vec<usize>],
    ) -> f64 {
        self.rate_correction_and_delta_e(state, reacting_sites, neighbors).0
    }

    /// Same as `rate_correction`, but also returns the underlying ΔE (the
    /// summed interaction energy), since `corr = exp(-β·ΔE)` (with
    /// β = 1/kbt) gives `∂corr/∂β = -ΔE·corr`; needed by the steady-state
    /// β-derivative path.
    #[inline]
    fn rate_correction_and_delta_e(
        &self,
        state:          &[u8],
        reacting_sites: &[usize],
        neighbors:      &[Vec<usize>],
    ) -> (f64, f64) {
        if self.trivial { return (1.0, 0.0); }
        let mut in_rxn = [false; 64];
        for &s in reacting_sites { in_rxn[s] = true; }
        let mut delta_e = 0.0f64;
        for &site in reacting_sites {
            let sp = state[site] as usize;
            for &nbr in &neighbors[site] {
                if !in_rxn[nbr] {
                    delta_e += self.epsilon[sp][state[nbr] as usize];
                }
            }
        }
        ((-delta_e / self.kbt).exp(), delta_e)
    }
}

// ─── Reaction ─────────────────────────────────────────────────────────────────

/// A local reaction rule: pattern_in → pattern_out at base rate `rate`.
///
/// pattern_in   : species codes of reacting sites (length 1 or 2)
/// pattern_out  : species codes after reaction     (same length)
/// rate         : base rate constant (before interaction correction)
/// name         : human-readable label (e.g. "Adsorption of A")
/// rate_symbol  : Simpl string symbol for the rate constant
///                used as the label above the arrow in the graph viewer.
///                Defaults to name if empty.
/// rate_symbol_latex : Optional LaTeX string for the rate constant symbol (e.g. r"k_{\mathrm{ads}}")
/// interaction  : optional per-reaction InteractionModel; if None the builder's
///                global InteractionModel (default noninteracting) is used.
#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct Reaction {
    #[pyo3(get, set)] pub pattern_in:     Vec<u8>,
    #[pyo3(get, set)] pub pattern_out:    Vec<u8>,
    #[pyo3(get, set)] pub rate:           f64,
    #[pyo3(get, set)] pub name:           String,
    #[pyo3(get, set)] pub rate_symbol:    String,
    #[pyo3(get, set)] pub rate_symbol_latex:   Option<String>,
    interaction:                          Option<InteractionModel>,
}

#[pymethods]
impl Reaction {
    #[new]
    #[pyo3(signature = (pattern_in, pattern_out, rate, name=String::new(), rate_symbol=String::new(), rate_symbol_latex=None, interaction=None))]
    pub fn new(
        pattern_in:     Vec<u8>,
        pattern_out:    Vec<u8>,
        rate:           f64,
        name:           String,
        rate_symbol:    String,
        rate_symbol_latex:   Option<String>,
        interaction:    Option<InteractionModel>,
    ) -> PyResult<Self> {
        if pattern_in.len() != pattern_out.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pattern_in and pattern_out must have the same length"));
        }
        if pattern_in.is_empty() || pattern_in.len() > 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Only single-site (len=1) and pair (len=2) reactions supported"));
        }
        Ok(Self { pattern_in, pattern_out, rate, name, rate_symbol, rate_symbol_latex, interaction })
    }

    /// Get the per-reaction InteractionModel (None if not including interactions (builder default)).
    pub fn get_interaction(&self) -> Option<InteractionModel> {
        self.interaction.clone()
    }

    /// Set a per-reaction InteractionModel.
    pub fn set_interaction(&mut self, interaction: Option<InteractionModel>) {
        self.interaction = interaction;
    }

    // with_* return a modified copy instead of mutating in place; handy from
    // Python for sweeping one field (e.g. rate) without rebuilding a Reaction.
    pub fn with_rate(&self, rate: f64) -> Self {
        Self { rate, ..self.clone() }
    }

    pub fn with_rate_symbol(&self, rate_symbol: String) -> Self {
        Self { rate_symbol, ..self.clone() }
    }

    pub fn with_interaction(&self, interaction: Option<InteractionModel>) -> Self {
        Self { interaction, ..self.clone() }
    }

    fn __repr__(&self) -> String {
        let rate_symbol = if self.rate_symbol.is_empty() { String::new() }
                          else { format!(", rate_symbol={:?}", self.rate_symbol) };
        let im  = match &self.interaction {
            Some(m) => format!(", {}", m.__repr__()),
            None    => String::new(),
        };
        format!(
            "Reaction({:?}: {:?} -> {:?}, rate={}{}{})",
            self.name, self.pattern_in, self.pattern_out,
            self.rate, rate_symbol, im
        )
    }
}

impl Reaction {
    /// Resolve which InteractionModel to use: per-reaction if set, else fallback.
    #[inline]
    fn effective_interaction<'a>(&'a self, fallback: &'a InteractionModel) -> &'a InteractionModel {
        self.interaction.as_ref().unwrap_or(fallback)
    }
}

// ─── Topology ─────────────────────────────────────────────────────────────────

/// Bond set defining which site-offset pairs are neighbors.
///
/// For a ring of l sites, two sites i and j are neighbors if
/// (i - j) % l ∈ deltas or (j - i) % l ∈ deltas.
///
#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct Topology {
    #[pyo3(get)] pub deltas: Vec<usize>,
}

#[pymethods]
impl Topology {
    /// Here deltas are the site offsets that define neighbors. 
    /// For a ring of l sites, two sites i and j are neighbors 
    /// if (i - j) % l ∈ deltas or (j - i) % l ∈ deltas.
    #[new]
    pub fn new(deltas: Vec<usize>) -> PyResult<Self> {
        if deltas.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("deltas must be non-empty"));
        }
        Ok(Self { deltas })
    }

    #[staticmethod]
    /// Corresponds to the greek cross tiles
    pub fn greek_cross() -> Self { Self { deltas: vec![1, 2] } }

    #[staticmethod]
    /// Corresponds to the brickwork tiles
    pub fn square(d: usize) -> Self { Self { deltas: vec![1, d] } }

    #[staticmethod]
    /// Corresponds to the hexagonal tiles
    pub fn hex(d: usize) -> Self { Self { deltas: vec![1, d, d + 1] } }

    #[staticmethod]
    pub fn custom(deltas: Vec<usize>) -> PyResult<Self> { Self::new(deltas) }

    /// The primary long-range offset (largest delta), used for Adams 2025 validation.
    pub fn d(&self) -> usize { *self.deltas.iter().max().unwrap_or(&0) }

    fn __repr__(&self) -> String {
        format!("Topology(deltas={:?})", self.deltas)
    }
}

// ─── Tile geometry ────────────────────────────────────────────────────────────

// Precomputed geometry for an l-site ring: who's next to whom, and how big
// the resulting state space is. Built once per MEMKMBuilder and reused for
// every matrix assembly pass.
struct Tile {
    l:              usize,
    base:           usize,             // n_ads + 1 (empty site + adsorbates)
    n_states:       usize,             // base^l
    neighbors:      Vec<Vec<usize>>,   // neighbors[i] = sites adjacent to i
    neighbor_pairs: Vec<(usize, usize)>, // each undirected bond once, i < j
}

impl Tile {
    // Walk every site i and every offset in topology.deltas, wrap around the
    // ring with `(i + delta) % l`, and record the edge both ways (neighbors
    // is undirected, so i sees j and j sees i). `i != j` guards against a
    // delta that wraps a site onto itself on a tiny ring (e.g. l=2, delta=1
    // hitting itself isn't possible, but l=1 or degenerate deltas could).
    // neighbor_pairs collects each bond once (as a HashSet, so duplicate
    // deltas or deltas that produce the same edge from both directions
    // don't double-count it) for the 2nd-order reaction loop, which needs
    // one canonical (i, j) per bond rather than every directed (i, j)/(j, i).
    fn new(l: usize, topology: &Topology, n_ads: usize) -> Self {
        let base     = n_ads + 1;
        let n_states = base.pow(l as u32);
        let mut neighbors = vec![vec![]; l];
        let mut pair_set  = std::collections::HashSet::new();
        for i in 0..l {
            for &delta in &topology.deltas {
                let j = (i + delta) % l;
                if i != j {
                    neighbors[i].push(j);
                    neighbors[j].push(i);
                    pair_set.insert((i.min(j), i.max(j)));
                }
            }
        }
        for nbrs in &mut neighbors { nbrs.sort_unstable(); nbrs.dedup(); }
        let mut neighbor_pairs: Vec<_> = pair_set.into_iter().collect();
        neighbor_pairs.sort_unstable();
        Self { l, base, n_states, neighbors, neighbor_pairs }
    }

    /// Validate primary (l,d) brickwork pair against Adams et al. 2025 SI Figure S3.
    ///
    /// This is purely a geometric sanity check on the (ring length, offset)
    /// pair, independent of whatever reactions get attached later:
    /// - rule1: d=0 or d=l would make a site its own periodic neighbor.
    /// - rule2: catches offsets that double-count a bond; d=1 is already
    ///   covered by the brickwork's nearest-neighbor offset, d=l-1 is just
    ///   d=1 read the other way around the ring, and d=l/2 (even l) is its
    ///   own mirror image (i+d and i-d land on the same site).
    /// - checkerboard: whether this (l, d) can host a strict alternating
    ///   2-coloring at all (needs l even and d odd, so stepping by d always
    ///   flips parity).
    fn validate(l: usize, d: usize) -> (bool, bool, bool) {
        let rule1        = d != 0 && d != l;
        let rule2        = d != 1 && (l % 2 != 0 || d != l / 2) && d != l - 1;
        let checkerboard = l % 2 == 0 && d % 2 == 1;
        (rule1, rule2, checkerboard)
    }
}

// ─── MEMKMBuilder ─────────────────────────────────────────────────────────────

/// ME-MKM transition matrix builder.
///
/// Tile geometry and reactions for ME-MKM.
///
/// topology defines which site-offset pairs are neighbors (see Topology).
/// Defaults to Topology.square(d) for backwards compatibility when d is given.
///
/// Each Reaction can carry its own InteractionModel for its λ correction.
/// If a Reaction has none, the builder's global InteractionModel is used
/// (default: noninteracting → all corrections = 1).
#[pyclass]
pub struct MEMKMBuilder {
    tile:           Tile,
    reactions:      Vec<Reaction>,
    interaction:    InteractionModel,
    #[pyo3(get)] pub topology:  Topology,
    #[pyo3(get)] pub l:        usize,
    #[pyo3(get)] pub n_ads:    usize,
    #[pyo3(get)] pub n_states: usize,
    #[pyo3(get)] pub n_pairs:  usize,
}

#[pymethods]
impl MEMKMBuilder {
    #[new]
    #[pyo3(signature = (l, topology, reactions, interaction=None))]
    pub fn new(
        l:           usize,
        topology:    Topology,
        reactions:   Vec<Reaction>,
        interaction: Option<InteractionModel>,
    ) -> Self {
        // n_ads isn't passed in explicitly -- it's inferred from the
        // reactions themselves. Species are coded as 0 (empty), 1, 2, ...,
        // so the largest species code that appears anywhere in any
        // reaction's pattern_in/pattern_out tells us how many adsorbate
        // species there are. unwrap_or(0) covers the no-reactions-yet case.
        let n_ads = reactions.iter()
            .flat_map(|rxn| rxn.pattern_in.iter().chain(rxn.pattern_out.iter()))
            .copied()
            .max()
            .unwrap_or(0) as usize;
        // The tile only needs to know l, the topology, and how many species
        // occupy a site (base = n_ads + 1) -- it doesn't care about
        // reactions, so it's built once here and reused for every matrix
        // assembly call later.
        let tile        = Tile::new(l, &topology, n_ads);
        let n_states    = tile.n_states;
        let n_pairs     = tile.neighbor_pairs.len();
        // No interaction model given -> fall back to a noninteracting one
        // (all corrections = 1), sized for n_ads+1 species.
        let interaction = interaction
            .unwrap_or_else(|| InteractionModel::noninteracting(n_ads + 1, 1.0));
        Self {
            reactions,
            tile,
            interaction,
            topology,
            l, n_ads, n_states, n_pairs,
        }
    }

    /// Primary long-range offset; largest delta in the topology.
    /// Kept for convenience and Adams 2025 validate().
    #[getter]
    pub fn d(&self) -> usize { self.topology.d() }

    pub fn add_reaction(&mut self, rxn: Reaction) { self.reactions.push(rxn); }
    pub fn set_reactions(&mut self, reactions: Vec<Reaction>) { self.reactions = reactions; }
    pub fn get_reactions(&self) -> Vec<Reaction> { self.reactions.clone() }
    pub fn clear_reactions(&mut self) { self.reactions.clear(); }

    #[getter]
    pub fn n_rxns(&self) -> usize { self.reactions.len() }

    pub fn set_interaction(&mut self, interaction: InteractionModel) {
        self.interaction = interaction;
    }
    pub fn get_interaction(&self) -> InteractionModel { self.interaction.clone() }

    // Thin wrapper around Tile::validate that turns the (bool, bool, bool)
    // tuple into a Python dict with a human-readable "note" explaining
    // which rule failed first (rule1 takes priority over rule2, which
    // takes priority over checkerboard), so callers in Python don't have
    // to remember the rule ordering themselves.
    pub fn validate<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyDict> {
        let (r1, r2, cb) = Tile::validate(self.l, self.topology.d());
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("rule1_ok",     r1).unwrap();
        dict.set_item("rule2_ok",     r2).unwrap();
        dict.set_item("checkerboard", cb).unwrap();
        dict.set_item("note", if !r1 {
            "Violates rule 1: site abuts its own periodic image"
        } else if !r2 {
            "Violates rule 2: some neighbors counted multiple times"
        } else if !cb {
            "Valid Tile but cannot represent checkerboard superlattice"
        } else {
            "Fully valid: satisfies both rules and supports checkerboard"
        }).unwrap();
        dict
    }

    /// Full dynamical-form W as COO triples (rows, cols, vals); hand these
    /// straight to scipy.sparse on the Python side.
    pub fn build_w_coo(&self) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
        self.compute_w_coo()
    }

    /// Same as build_w_coo, but with the last row swapped for the
    /// normalisation condition (all 1s) so it's ready to solve for the
    /// steady-state distribution directly.
    pub fn build_w_ss_coo(&self) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
        self.compute_w_ss_coo()
    }

    /// One dynamical-form W matrix per reaction, each built at rate=1.
    ///
    /// W is linear in each reaction's rate constant, so for time-dependent
    /// rates k_i(t):  W(t) = Σ k_i(t) · components[i]. Useful for forced
    /// oscillations / Floquet-type driving, where rebuilding the full W from
    /// scratch at every ODE step would be wasteful.
    pub fn build_w_components_coo(&self) -> Vec<(Vec<i32>, Vec<i32>, Vec<f64>)> {
        self.compute_w_components_coo()
    }

    /// ∂(dynamical-form W)/∂β per reaction, each at unit base rate (β = 1/kbt).
    ///
    /// Mirrors build_w_components_coo's COO convention exactly, so Python can
    /// recombine both the same way: W = Σ k_i·components[i],
    /// ∂W/∂β = Σ k_i·dbeta_components[i] (rates are β-independent themselves;
    /// only the interaction correction factor depends on β).
    pub fn build_dw_dbeta_components_coo(&self) -> Vec<(Vec<i32>, Vec<i32>, Vec<f64>)> {
        self.compute_dw_dbeta_components_coo()
    }

    fn __repr__(&self) -> String {
        format!(
            "MEMKMBuilder(l={}, topology={}, n_ads={}, n_states={}, n_rxns={}, n_pairs={})",
            self.l, self.topology.__repr__(), self.n_ads, self.n_states,
            self.reactions.len(), self.tile.neighbor_pairs.len()
        )
    }
}

// ─── MEMKMBuilder ────────────────────────────────────

impl MEMKMBuilder {
    /// Each Reaction uses its own InteractionModel if set, else the builder's (no interactions).
    /// Neighbor species for the correction are derived from pattern_in:
    /// non-reacting neighbors of the reacting site(s) in the reactant state.
    ///
    /// Shared core loop behind every W variant in this file. For each
    /// microstate (from_idx, decoded to a per-site Vec<u8>) and each
    /// reaction, find every site (order-1) or neighbor pair (order-2) whose
    /// occupancy matches rxn.pattern_in. For each match: get the rate
    /// correction for that local environment, multiply by the bare rate,
    /// write pattern_out into the matched site(s) to get the new state
    /// to_idx, and emit a (to_idx, from_idx, rate) COO entry. diag[from_idx]
    /// accumulates -rate for every match, so the column sums to zero (W is a
    /// generator matrix: outflow from a state always equals total inflow
    /// elsewhere). to_idx == from_idx is skipped; a transition to the same
    /// state doesn't move probability and isn't worth a zero entry.
    ///
    /// Order-2 reactions are checked in both directions over each unordered
    /// pair (si, sj), since pattern_in = [A, B] and [B, A] are different
    /// matches.
    fn compute_offdiag(&self) -> (Vec<i32>, Vec<i32>, Vec<f64>, Vec<f64>) {
        let n = self.tile.n_states;
        // Upper bound on off-diagonal entries per state: each reaction can fire at
        // most once per matching site (order 1) or directed neighbor pair (order 2).
        let per_state_cap: usize = self.reactions.iter().map(|rxn| match rxn.pattern_in.len() {
            1 => self.tile.l,
            2 => 2 * self.tile.neighbor_pairs.len(),
            _ => 0,
        }).sum();
        let cap = n * per_state_cap;
        let mut rows = Vec::with_capacity(cap);
        let mut cols = Vec::with_capacity(cap);
        let mut vals = Vec::with_capacity(cap);
        let mut diag = vec![0.0f64; n];

        for from_idx in 0..n {
            let state = decode(from_idx, self.tile.l, self.tile.base);

            for rxn in &self.reactions {
                let im = rxn.effective_interaction(&self.interaction);

                match rxn.pattern_in.len() {

                    // ── 1st order reaction ───────────────────────────────────────────
                    1 => {
                        for site in 0..self.tile.l {
                            if state[site] == rxn.pattern_in[0] {
                                let corr = im.rate_correction(
                                    &state, &[site], &self.tile.neighbors);
                                let rate = rxn.rate * corr;
                                let mut ns = state.clone();
                                ns[site] = rxn.pattern_out[0];
                                let to_idx = encode(&ns, self.tile.base);
                                if to_idx != from_idx {
                                    rows.push(to_idx as i32);
                                    cols.push(from_idx as i32);
                                    vals.push(rate);
                                    diag[from_idx] -= rate;
                                }
                            }
                        }
                    }

                    // ── 2nd order reaction ─────────────────────────────────────────
                    2 => {
                        for &(si, sj) in &self.tile.neighbor_pairs {
                            // For a symmetric reaction (pattern_in[0] ==
                            // pattern_in[1]) with a symmetric product, both
                            // orderings of this bond match the same physical
                            // event and land on the same to_idx -- firing
                            // both would double-count a single elementary
                            // event. fired_to dedups against that. For
                            // heterogeneous pattern_in, at most one ordering
                            // ever matches, so this is a no-op there; for a
                            // symmetric input with an asymmetric product
                            // (disproportionation), the two orderings land on
                            // different to_idx and both are genuinely kept.
                            let mut fired_to: Option<usize> = None;
                            for (s0, s1) in [(si, sj), (sj, si)] {
                                if state[s0] == rxn.pattern_in[0] && state[s1] == rxn.pattern_in[1] {
                                    let mut ns = state.clone();
                                    ns[s0] = rxn.pattern_out[0];
                                    ns[s1] = rxn.pattern_out[1];
                                    let to_idx = encode(&ns, self.tile.base);
                                    if to_idx == from_idx || fired_to == Some(to_idx) {
                                        continue;
                                    }
                                    fired_to = Some(to_idx);
                                    // Correction: non-reacting neighbors of both
                                    // reacting sites, derived from pattern_in.
                                    let corr = im.rate_correction(
                                        &state, &[s0, s1], &self.tile.neighbors);
                                    let rate = rxn.rate * corr;
                                    rows.push(to_idx as i32);
                                    cols.push(from_idx as i32);
                                    vals.push(rate);
                                    diag[from_idx] -= rate;
                                }
                            }
                        }
                    }

                    _ => unreachable!(),
                }
            }
        }

        (rows, cols, vals, diag)
    }

    /// Dynamical W: diagonal = -sum of outgoing rates.
    ///
    /// compute_offdiag's output plus the diag vector appended as the matrix
    /// diagonal; the full W at each reaction's current rate.
    fn compute_w_coo(&self) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
        let n = self.tile.n_states;
        let (mut rows, mut cols, mut vals, diag) = self.compute_offdiag();
        for i in 0..n {
            rows.push(i as i32);
            cols.push(i as i32);
            vals.push(diag[i]);
        }
        (rows, cols, vals)
    }

    /// Per-reaction dynamical-form W at unit rate, one pass over all states.
    ///
    /// W is linear in the reactions' bare rate constants: each reaction i
    /// contributes its own set of off-diagonal entries and diagonal
    /// decrements, scaled by k_i and nothing else. So instead of building
    /// one W = Σ_i(k_i * Component_i), this builds each Component_i
    /// separately, evaluated at k_i = 1 (i.e. corr instead of rxn.rate *
    /// corr, since rxn.rate is multiplied in on the Python side). Given the
    /// components, recovering W for any rate vector is just a weighted sum
    /// of sparse matrices; no re-walk of states/reactions/sites needed.
    /// That's what lets the Python side change rates (sweep a concentration,
    /// drive k_i(t), or build derivative matrices like dW/dbeta) without
    /// ever calling back into Rust to rebuild the matrix from scratch.
    ///
    /// Loop body is otherwise identical to compute_offdiag, just keeping
    /// per-reaction rows/cols/vals/diag instead of one shared set.
    fn compute_w_components_coo(&self) -> Vec<(Vec<i32>, Vec<i32>, Vec<f64>)> {
        let n       = self.tile.n_states;
        let n_rxns  = self.reactions.len();
        let mut rows = vec![Vec::<i32>::new(); n_rxns];
        let mut cols = vec![Vec::<i32>::new(); n_rxns];
        let mut vals = vec![Vec::<f64>::new(); n_rxns];
        let mut diag = vec![vec![0.0f64; n]; n_rxns];

        for from_idx in 0..n {
            let state = decode(from_idx, self.tile.l, self.tile.base);

            for (ri, rxn) in self.reactions.iter().enumerate() {
                let im = rxn.effective_interaction(&self.interaction);

                match rxn.pattern_in.len() {
                    1 => {
                        for site in 0..self.tile.l {
                            if state[site] == rxn.pattern_in[0] {
                                let corr = im.rate_correction(
                                    &state, &[site], &self.tile.neighbors);
                                let mut ns = state.clone();
                                ns[site] = rxn.pattern_out[0];
                                let to_idx = encode(&ns, self.tile.base);
                                if to_idx != from_idx {
                                    rows[ri].push(to_idx as i32);
                                    cols[ri].push(from_idx as i32);
                                    vals[ri].push(corr);
                                    diag[ri][from_idx] -= corr;
                                }
                            }
                        }
                    }
                    2 => {
                        for &(si, sj) in &self.tile.neighbor_pairs {
                            // See compute_offdiag's 2nd-order branch for why
                            // this dedup is needed (symmetric reactions would
                            // otherwise double-fire the same bond).
                            let mut fired_to: Option<usize> = None;
                            for (s0, s1) in [(si, sj), (sj, si)] {
                                if state[s0] == rxn.pattern_in[0] && state[s1] == rxn.pattern_in[1] {
                                    let mut ns = state.clone();
                                    ns[s0] = rxn.pattern_out[0];
                                    ns[s1] = rxn.pattern_out[1];
                                    let to_idx = encode(&ns, self.tile.base);
                                    if to_idx == from_idx || fired_to == Some(to_idx) {
                                        continue;
                                    }
                                    fired_to = Some(to_idx);
                                    let corr = im.rate_correction(
                                        &state, &[s0, s1], &self.tile.neighbors);
                                    rows[ri].push(to_idx as i32);
                                    cols[ri].push(from_idx as i32);
                                    vals[ri].push(corr);
                                    diag[ri][from_idx] -= corr;
                                }
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }

        (0..n_rxns).map(|ri| {
            let mut r = std::mem::take(&mut rows[ri]);
            let mut c = std::mem::take(&mut cols[ri]);
            let mut v = std::mem::take(&mut vals[ri]);
            for i in 0..n {
                r.push(i as i32);
                c.push(i as i32);
                v.push(diag[ri][i]);
            }
            (r, c, v)
        }).collect()
    }

    /// Per-reaction ∂(dynamical-form W)/∂β at unit base rate, one pass over all
    /// states. Structurally identical to compute_w_components_coo, but each
    /// off-diagonal/diagonal entry is -ΔE·corr instead of corr, since
    /// corr = exp(-β·ΔE) ⇒ ∂corr/∂β = -ΔE·corr.
    fn compute_dw_dbeta_components_coo(&self) -> Vec<(Vec<i32>, Vec<i32>, Vec<f64>)> {
        let n       = self.tile.n_states;
        let n_rxns  = self.reactions.len();
        // Same per-reaction COO buffers as compute_w_components_coo, but
        // holding dcorr (the beta-derivative of the rate correction) instead
        // of corr itself.
        let mut rows = vec![Vec::<i32>::new(); n_rxns];
        let mut cols = vec![Vec::<i32>::new(); n_rxns];
        let mut vals = vec![Vec::<f64>::new(); n_rxns];
        let mut diag = vec![vec![0.0f64; n]; n_rxns];

        for from_idx in 0..n {
            let state = decode(from_idx, self.tile.l, self.tile.base);

            for (ri, rxn) in self.reactions.iter().enumerate() {
                let im = rxn.effective_interaction(&self.interaction);

                match rxn.pattern_in.len() {
                    // This is the site-matching as compute_w_components_coo (order-1
                    // reaction: single reacting site), but here we need both
                    // corr and delta_e, so call rate_correction_and_delta_e
                    // instead of rate_correction. dcorr = -delta_e * corr is
                    // d(corr)/d(beta), since corr = exp(-beta * delta_e).
                    1 => {
                        for site in 0..self.tile.l {
                            if state[site] == rxn.pattern_in[0] {
                                let (corr, delta_e) = im.rate_correction_and_delta_e(
                                    &state, &[site], &self.tile.neighbors);
                                let dcorr = -delta_e * corr;
                                let mut ns = state.clone();
                                ns[site] = rxn.pattern_out[0];
                                let to_idx = encode(&ns, self.tile.base);
                                // Same skip-self-transition + "record entry,
                                // decrement diagonal" pattern as
                                // compute_offdiag, just storing dcorr instead
                                // of a rate.
                                if to_idx != from_idx {
                                    rows[ri].push(to_idx as i32);
                                    cols[ri].push(from_idx as i32);
                                    vals[ri].push(dcorr);
                                    diag[ri][from_idx] -= dcorr;
                                }
                            }
                        }
                    }
                    // Order-2 reaction (reacting pair): identical structure,
                    // over both orderings of each neighbor pair, with
                    // the same dedup as compute_offdiag's 2nd-order branch
                    // (symmetric reactions would otherwise double-fire).
                    2 => {
                        for &(si, sj) in &self.tile.neighbor_pairs {
                            let mut fired_to: Option<usize> = None;
                            for (s0, s1) in [(si, sj), (sj, si)] {
                                if state[s0] == rxn.pattern_in[0] && state[s1] == rxn.pattern_in[1] {
                                    let mut ns = state.clone();
                                    ns[s0] = rxn.pattern_out[0];
                                    ns[s1] = rxn.pattern_out[1];
                                    let to_idx = encode(&ns, self.tile.base);
                                    if to_idx == from_idx || fired_to == Some(to_idx) {
                                        continue;
                                    }
                                    fired_to = Some(to_idx);
                                    let (corr, delta_e) = im.rate_correction_and_delta_e(
                                        &state, &[s0, s1], &self.tile.neighbors);
                                    let dcorr = -delta_e * corr;
                                    rows[ri].push(to_idx as i32);
                                    cols[ri].push(from_idx as i32);
                                    vals[ri].push(dcorr);
                                    diag[ri][from_idx] -= dcorr;
                                }
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }

        // Append each reaction's diagonal as the last entries of its own
        // COO triple, same as compute_w_components_coo's tail step.
        (0..n_rxns).map(|ri| {
            let mut r = std::mem::take(&mut rows[ri]);
            let mut c = std::mem::take(&mut cols[ri]);
            let mut v = std::mem::take(&mut vals[ri]);
            for i in 0..n {
                r.push(i as i32);
                c.push(i as i32);
                v.push(diag[ri][i]);
            }
            (r, c, v)
        }).collect()
    }

    /// Steady-state W: last row replaced by normalisation (all 1s).
    ///
    /// W's rows are linearly dependent (each column sums to zero), so
    /// W @ Theta = 0 alone has a 1-dimensional null space and no unique
    /// solution. Dropping the last row and replacing it with all-1s plus a
    /// matching rhs of 1 in that slot turns it into Σ(Theta) = 1, pinning
    /// down the one degree of freedom that "outflow = inflow" leaves open.
    fn compute_w_ss_coo(&self) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
        let n        = self.tile.n_states;
        let last_row = (n - 1) as i32;
        let (rows, cols, vals, diag) = self.compute_offdiag();

        let mut out_rows = Vec::with_capacity(rows.len() + n);
        let mut out_cols = Vec::with_capacity(cols.len() + n);
        let mut out_vals = Vec::with_capacity(vals.len() + n);

        // Off-diagonal entries not in last row + diagonal for all but last row
        for i in 0..n - 1 {
            out_rows.push(i as i32);
            out_cols.push(i as i32);
            out_vals.push(diag[i]);
        }
        for ((r, c), v) in rows.iter().zip(cols.iter()).zip(vals.iter()) {
            if *r != last_row {
                out_rows.push(*r);
                out_cols.push(*c);
                out_vals.push(*v);
            }
        }
        // Normalisation row
        for j in 0..n {
            out_rows.push(last_row);
            out_cols.push(j as i32);
            out_vals.push(1.0);
        }
        (out_rows, out_cols, out_vals)
    }
}

// Thin pyo3 wrappers so Python can decode/encode state indices too, mostly
// for debugging and for turning a Theta vector back into readable states.
#[pyfunction]
fn decode_state(idx: usize, l: usize, base: usize) -> Vec<u8> {
    decode(idx, l, base)
}

#[pyfunction]
fn encode_state(state: Vec<u8>, base: usize) -> usize {
    encode(&state, base)
}

// Everything pyo3-exposed gets registered here; this is what `import
// me_mkm._me_mkm` actually loads.
#[pymodule]
fn _me_mkm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Topology>()?;
    m.add_class::<InteractionModel>()?;
    m.add_class::<Reaction>()?;
    m.add_class::<MEMKMBuilder>()?;
    m.add_function(wrap_pyfunction!(decode_state, m)?)?;
    m.add_function(wrap_pyfunction!(encode_state, m)?)?;
    Ok(())
}
