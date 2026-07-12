use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

#[inline]
fn decode(mut microstate: usize, l: usize, base: usize) -> Vec<u8> {
    let mut digits = vec![0u8; l];
    for d in digits.iter_mut().rev() {
        *d = (microstate % base) as u8;
        microstate /= base;
    }
    digits
}

#[inline]
fn encode(state: &[u8], base: usize) -> usize {
    state.iter().fold(0usize, |acc, &d| acc * base + d as usize)
}
/// Given a microstate "number", you can decode it given the known base and tile length.
#[gen_stub_pyfunction]
#[pyfunction]
fn decode_state(microstate: usize, l: usize, base: usize) -> Vec<u8> {
    decode(microstate, l, base)
}

/// Given a state vector for a tile of `l` sites, you can encode it into a
/// microstate "number" given the known base (number of species).
#[gen_stub_pyfunction]
#[pyfunction]
fn encode_state(microstate: Vec<u8>, base: usize) -> usize {
    encode(&microstate, base)
}

#[inline]
fn count_species(state: &[u8], base: usize) -> Vec<usize> {
    let mut counts = vec![0usize; base];
    for &s in state {
        counts[s as usize] += 1;
    }
    counts
}

// Counts the number of species in a decoded microstate (given as a vector)
#[gen_stub_pyfunction]
#[pyfunction]
fn state_counts(idx: usize, l: usize, base: usize) -> Vec<usize> {
    count_species(&decode(idx, l, base), base)
}

/// Pairwise nearest-neighbor interaction energies ε[s1][s2].
///
/// Rate correction for a reaction event:
///     correction = exp( -Sum_{non-reacting neighbors} ε[sp_reacting][sp_neighbor] / kBT )
///
/// The sum runs over non-reacting neighbors of each reacting site.
#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct InteractionModel {
    #[pyo3(get)]
    pub epsilon: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub kbt: f64,
    trivial: bool,
}

#[gen_stub_pymethods]
#[pymethods]
impl InteractionModel {
    #[new]
    #[pyo3(signature = (epsilon, kbt=1.0))]
    pub fn new(epsilon: Vec<Vec<f64>>, kbt: f64) -> Self {
        let trivial = epsilon.iter().all(|row| row.iter().all(|&e| e == 0.0));
        Self {
            epsilon,
            kbt,
            trivial,
        }
    }

    #[staticmethod]
    pub fn noninteracting(n_species: usize, kbt: f64) -> Self {
        Self::new(vec![vec![0.0; n_species]; n_species], kbt)
    }

    fn __repr__(&self) -> String {
        if self.trivial {
            format!("InteractionModel(noninteracting, kBT={})", self.kbt)
        } else {
            format!(
                "InteractionModel(epsilon={:?}, kBT={})",
                self.epsilon, self.kbt
            )
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
        state: &[u8],
        reacting_sites: &[usize],
        neighbors: &[Vec<usize>],
    ) -> f64 {
        self.rate_correction_and_delta_e(state, reacting_sites, neighbors)
            .0
    }

    /// Same as `rate_correction`, but also returns the underlying ΔE (the
    /// summed interaction energy), since `corr = exp(-β·ΔE)` (with
    /// β = 1/kbt) gives `∂corr/∂β = -ΔE·corr`; needed by the steady-state
    /// β-derivative path.
    #[inline]
    fn rate_correction_and_delta_e(
        &self,
        state: &[u8],
        reacting_sites: &[usize],
        neighbors: &[Vec<usize>],
    ) -> (f64, f64) {
        if self.trivial {
            return (1.0, 0.0);
        }
        let mut in_rxn = [false; 64];
        for &s in reacting_sites {
            in_rxn[s] = true;
        }
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
#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct Reaction {
    // Reacting-site species codes (0-based indices into the builder's
    // species_names). Positional, one entry per reacting site, and may repeat a
    // code (e.g. [1, 1]).
    #[pyo3(get, set)]
    pub pattern_in: Vec<u8>,
    #[pyo3(get, set)]
    pub pattern_out: Vec<u8>,
    #[pyo3(get, set)]
    pub rate: f64,
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub rate_symbol: String,
    #[pyo3(get, set)]
    pub rate_symbol_latex: Option<String>,
    interaction: Option<InteractionModel>,
}

#[gen_stub_pymethods]
#[pymethods]
impl Reaction {
    #[new]
    #[pyo3(signature = (pattern_in, pattern_out, rate, name=String::new(), rate_symbol=String::new(), rate_symbol_latex=None, interaction=None))]
    pub fn new(
        pattern_in: Vec<u8>,
        pattern_out: Vec<u8>,
        rate: f64,
        name: String,
        rate_symbol: String,
        rate_symbol_latex: Option<String>,
        interaction: Option<InteractionModel>,
    ) -> PyResult<Self> {
        if pattern_in.len() != pattern_out.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pattern_in and pattern_out must have the same length",
            ));
        }
        if pattern_in.is_empty() || pattern_in.len() > 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Only single-site (len=1) and pair (len=2) reactions supported",
            ));
        }
        Ok(Self {
            pattern_in,
            pattern_out,
            rate,
            name,
            rate_symbol,
            rate_symbol_latex,
            interaction,
        })
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
        Self {
            rate,
            ..self.clone()
        }
    }

    pub fn with_rate_symbol(&self, rate_symbol: String) -> Self {
        Self {
            rate_symbol,
            ..self.clone()
        }
    }

    pub fn with_interaction(&self, interaction: Option<InteractionModel>) -> Self {
        Self {
            interaction,
            ..self.clone()
        }
    }

    fn __repr__(&self) -> String {
        let rate_symbol = if self.rate_symbol.is_empty() {
            String::new()
        } else {
            format!(", rate_symbol={:?}", self.rate_symbol)
        };
        let im = match &self.interaction {
            Some(m) => format!(", {}", m.__repr__()),
            None => String::new(),
        };
        format!(
            "Reaction({:?}: {:?} -> {:?}, rate={}{}{})",
            self.name, self.pattern_in, self.pattern_out, self.rate, rate_symbol, im
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

// ─── Tile geometry ────────────────────────────────────────────────────────────

/// Bond set defining which site-offset pairs are neighbors.
///
/// For a ring of l sites, two sites i and j are neighbors if
/// (i - j) % l ∈ deltas or (j - i) % l ∈ deltas.
///
#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct TileSettings {
    #[pyo3(get)]
    pub sites: usize,
    #[pyo3(get)]
    pub deltas: Vec<usize>,
}

#[gen_stub_pymethods]
#[pymethods]
impl TileSettings {
    /// Here deltas are the site offsets that define neighbors.
    /// For a ring of l sites, two sites i and j are neighbors
    /// if (i - j) % l ∈ deltas or (j - i) % l ∈ deltas.
    #[new]
    pub fn new(sites: usize, deltas: Vec<usize>) -> PyResult<Self> {
        if deltas.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "deltas must be non-empty",
            ));
        }
        Ok(Self { sites, deltas })
    }

    #[staticmethod]
    /// Corresponds to the greek cross tiles
    pub fn greek_cross() -> Self {
        Self {
            sites: 5,
            deltas: vec![1, 2],
        }
    }

    #[staticmethod]
    /// Corresponds to the brickwork tiles
    pub fn square(sites: usize, d: usize) -> Self {
        Self {
            sites,
            deltas: vec![1, d],
        }
    }

    #[staticmethod]
    /// Corresponds to the hexagonal "creamcups" tile: 7 sites (six around one
    /// centre), a complete K_7 under deltas [1, d, d+1].
    pub fn hex(d: usize) -> Self {
        Self {
            sites: 7,
            deltas: vec![1, d, d + 1],
        }
    }

    #[staticmethod]
    pub fn custom(sites: usize, deltas: Vec<usize>) -> PyResult<Self> {
        Self::new(sites, deltas)
    }

    #[staticmethod]
    /// Smallest brickwork offset d (searched from d=1 up) for a ring of
    /// `sites` sites that is fully valid per Tile::validate: rule1, rule2,
    /// AND checkerboard-capable (Adams et al. 2025 SI Figure S3) if requested.
    /// Returns None if no such d exists for this l (e.g. sites odd, or sites=4).
    pub fn smallest_valid_square(sites: usize, checkerboard: bool) -> Option<Self> {
        (1..sites)
            .find(|&d| {
                let (rule1, rule2, chkbrd) = Tile::validate(sites, d);
                rule1 && rule2 && chkbrd == checkerboard
            })
            .map(|d| Self::square(sites, d))
    }

    /// The number of sites in the tile (ring length). Same as sites.
    pub fn l(&self) -> usize {
        self.sites
    }

    /// The primary long-range offset (largest delta), used for Adams 2025 validation.
    pub fn d(&self) -> usize {
        *self.deltas.iter().max().unwrap_or(&0)
    }

    fn __repr__(&self) -> String {
        format!(
            "tile_settings(sites={:?},deltas={:?})",
            self.sites, self.deltas
        )
    }
}

// Precomputed geometry for an l-site ring: who's next to whom, and how big
// the resulting state space is. Built once per MEMKMBuilder and reused for
// every matrix assembly pass.
struct Tile {
    l: usize,
    base: usize,                         // = n_species (one code per species)
    n_states: usize,                     // base^l
    neighbors: Vec<Vec<usize>>,          // neighbors[i] = sites adjacent to i
    neighbor_pairs: Vec<(usize, usize)>, // each undirected bond once, i < j
}

impl Tile {
    // Walk every site i and every offset in tilesettings.deltas, wrap around the
    // ring with `(i + delta) % l`, and record the edge both ways (neighbors
    // is undirected, so i sees j and j sees i). `i != j` guards against a
    // delta that wraps a site onto itself on a tiny ring (e.g. l=2, delta=1
    // hitting itself isn't possible, but l=1 or degenerate deltas could).
    // neighbor_pairs collects each bond once (as a HashSet, so duplicate
    // deltas or deltas that produce the same edge from both directions
    // don't double-count it) for the 2nd-order reaction loop, which needs
    // one canonical (i, j) per bond rather than every directed (i, j)/(j, i).
    fn new(tilesettings: &TileSettings, base: usize) -> Self {
        let l = tilesettings.sites;
        let n_states = base.pow(l as u32);
        let mut neighbors = vec![vec![]; l];
        let mut pair_set = std::collections::HashSet::new();
        for i in 0..l {
            for &delta in &tilesettings.deltas {
                let j = (i + delta) % l;
                if i != j {
                    neighbors[i].push(j);
                    neighbors[j].push(i);
                    pair_set.insert((i.min(j), i.max(j)));
                }
            }
        }
        for nbrs in &mut neighbors {
            nbrs.sort_unstable();
            nbrs.dedup();
        }
        let mut neighbor_pairs: Vec<_> = pair_set.into_iter().collect();
        neighbor_pairs.sort_unstable();
        Self {
            l,
            base,
            n_states,
            neighbors,
            neighbor_pairs,
        }
    }

    /// Validate primary (l,d) brickwork pair against Adams et al. 2025 SI Figure S3.
    ///
    /// This is purely a geometric sanity check on the (ring length, offset)
    /// pair, independent of whatever reactions get attached later:
    /// - rule1: d=0 or d=l would make a site its own periodic neighbor.
    /// - rule2: a valid square-lattice fold needs l even and d odd, so
    ///   stepping by d always flips parity.
    /// - checkerboard: whether this (l, d) can additionally host a strict
    ///   alternating 2-coloring. This is stricter than rule2: d=1 is just
    ///   the brickwork's nearest-neighbor offset and adds no new bond
    ///   type, d=l-1 is d=1 read the other way around the ring, and
    ///   d=l/2 (even l) is its own mirror image (i+d and i-d land on the
    ///   same site) -- all three are valid square-lattice tiles (rule1 and
    ///   rule2 pass) but cannot represent the checkerboard superlattice.
    fn validate(l: usize, d: usize) -> (bool, bool, bool) {
        let rule1 = d != 0 && d != l;
        let rule2 = l % 2 == 0 && d % 2 == 1;
        let checkerboard = rule1 && rule2 && d != 1 && d != l - 1 && d != l / 2;
        (rule1, rule2, checkerboard)
    }
}

// ─── MEMKMBuilder ─────────────────────────────────────────────────────────────

/// ME-MKM transition matrix builder.
///
/// Tile geometry and reactions for ME-MKM.
///
/// tile_settings defines the ring length and which site-offset pairs are
/// neighbors (see TileSettings).
///
/// species_names names every species; a reaction's integer codes index into it
/// (0 = species_names[0], the conventional default "*"). It is the single source
/// of truth for both the names and the species count: `n_species == len(species_names)`
/// and the state space is `n_species ** l`. NOTE: n_species counts ALL species,
/// index 0 included -- there is no implicit extra "empty" to add.
/// InteractionModel.epsilon is indexed in species_names order.
///
/// Each Reaction can carry its own InteractionModel for its λ correction.
/// If a Reaction has none, the builder's global InteractionModel is used
/// (default: noninteracting → all corrections = 1).
#[gen_stub_pyclass]
#[pyclass]
pub struct MEMKMBuilder {
    tile: Tile,
    reactions: Vec<Reaction>,
    interaction: InteractionModel,
    #[pyo3(get)]
    pub tile_settings: TileSettings,
    #[pyo3(get)]
    pub species_names: Vec<String>,
    #[pyo3(get)]
    pub n_states: usize,
    #[pyo3(get)]
    pub n_pairs: usize,
}

#[gen_stub_pymethods]
#[pymethods]
impl MEMKMBuilder {
    #[new]
    #[pyo3(signature = (tile_settings, reactions, species_names, interaction=None))]
    pub fn new(
        tile_settings: TileSettings,
        reactions: Vec<Reaction>,
        species_names: Vec<String>,
        interaction: Option<InteractionModel>,
    ) -> PyResult<Self> {
        // species_names fixes the species count: base = n_species, so a site
        // holds one of n_species codes and the state space is n_species ** l.
        // Every reaction code must index into it.
        let n_species = species_names.len();
        for rxn in &reactions {
            for &c in rxn.pattern_in.iter().chain(rxn.pattern_out.iter()) {
                if c as usize >= n_species {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "reaction {:?} uses species code {c} but only {n_species} species are named",
                        rxn.name
                    )));
                }
            }
        }
        // The tile only needs the ring geometry and the base, so it's built once
        // here and reused for every matrix assembly call.
        let tile = Tile::new(&tile_settings, n_species);
        let n_states = tile.n_states;
        let n_pairs = tile.neighbor_pairs.len();
        // No interaction model given -> noninteracting (all corrections = 1),
        // sized for n_species species.
        let interaction =
            interaction.unwrap_or_else(|| InteractionModel::noninteracting(n_species, 1.0));
        Ok(Self {
            tile,
            reactions,
            interaction,
            tile_settings,
            species_names,
            n_states,
            n_pairs,
        })
    }

    /// Ring length (number of sites), from the tile settings.
    #[getter]
    pub fn l(&self) -> usize {
        self.tile_settings.sites
    }

    /// Number of species, counting index 0. Equals the state-space base, so the
    /// microstate count is n_species ** l. (There is no separate empty species
    /// to add: index 0 is already one of the n_species.)
    #[getter]
    pub fn n_species(&self) -> usize {
        self.species_names.len()
    }

    /// Primary long-range offset; largest delta in the tile settings.
    /// Kept for convenience and Adams 2025 validate().
    #[getter]
    pub fn d(&self) -> usize {
        self.tile_settings.d()
    }

    // Mutators only swap the reaction list; species_names (and thus the state
    // space) is fixed at construction, so a swapped-in reaction must keep using
    // codes < n_species.
    pub fn add_reaction(&mut self, rxn: Reaction) {
        self.reactions.push(rxn);
    }
    pub fn set_reactions(&mut self, reactions: Vec<Reaction>) {
        self.reactions = reactions;
    }
    pub fn get_reactions(&self) -> Vec<Reaction> {
        self.reactions.clone()
    }
    pub fn clear_reactions(&mut self) {
        self.reactions.clear();
    }

    #[getter]
    pub fn n_rxns(&self) -> usize {
        self.reactions.len()
    }

    pub fn set_interaction(&mut self, interaction: InteractionModel) {
        self.interaction = interaction;
    }
    pub fn get_interaction(&self) -> InteractionModel {
        self.interaction.clone()
    }

    // Thin wrapper around Tile::validate that turns the (bool, bool, bool)
    // tuple into a Python dict with a human-readable "note" explaining
    // which rule failed first (rule1 takes priority over rule2, which
    // takes priority over checkerboard), so callers in Python don't have
    // to remember the rule ordering themselves.
    pub fn validate<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyDict> {
        let (r1, r2, cb) = Tile::validate(self.tile_settings.sites, self.tile_settings.d());
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("rule1_ok", r1).unwrap();
        dict.set_item("rule2_ok", r2).unwrap();
        dict.set_item("checkerboard", cb).unwrap();
        dict.set_item(
            "note",
            if !r1 {
                "Violates rule 1: site abuts its own periodic image"
            } else if !r2 {
                "Violates rule 2: needs l even and d odd for a valid square-lattice fold"
            } else if !cb {
                "Valid Tile but cannot represent checkerboard superlattice"
            } else {
                "Fully valid: satisfies both rules and supports checkerboard"
            },
        )
        .unwrap();
        dict
    }

    /// Undirected neighbor pairs (i < j), one entry per bond — the same
    /// geometry the W builder uses, exposed so Python doesn't re-derive it.
    pub fn neighbor_pairs(&self) -> Vec<(usize, usize)> {
        self.tile.neighbor_pairs.clone()
    }

    /// Partition every microstate by its coverage signature, returning one
    /// `(counts, indices)` entry per occupied class, sorted by `counts`.
    /// `counts[j]` is the number of sites carrying species `j + 1`; species 0's
    /// count is dropped from the key (it is fixed at `l - sum(counts)`, so it
    /// carries no extra information). This is the partition the graph viewer
    /// groups nodes by.
    pub fn coverage_classes(&self) -> Vec<(Vec<usize>, Vec<usize>)> {
        use std::collections::HashMap;
        let mut map: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();
        for idx in 0..self.tile.n_states {
            let state = decode(idx, self.tile.l, self.tile.base);
            let counts = count_species(&state, self.tile.base);
            // Drop species 0's count (redundant: l - sum of the rest).
            map.entry(counts[1..].to_vec()).or_default().push(idx);
        }
        let mut out: Vec<_> = map.into_iter().collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Microstate indices whose per-species site counts lie within the given
    /// inclusive bounds; `min_counts[s]`/`max_counts[s]` bound species `s` and
    /// both have length `n_species` (index 0 included -- no species is special).
    /// Fast path behind coverage-window queries (e.g. "coverage of A ≤ 0.2" →
    /// `max_counts[A] = ⌊0.2·l⌋`): one walk over the state space in Rust,
    /// returning only the matches.
    pub fn select_states(
        &self,
        min_counts: Vec<usize>,
        max_counts: Vec<usize>,
    ) -> PyResult<Vec<usize>> {
        let n_species = self.tile.base;
        if min_counts.len() != n_species || max_counts.len() != n_species {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "min_counts and max_counts must both have length n_species",
            ));
        }
        let mut out = Vec::new();
        for idx in 0..self.tile.n_states {
            let state = decode(idx, self.tile.l, self.tile.base);
            let counts = count_species(&state, self.tile.base);
            let ok =
                (0..n_species).all(|s| counts[s] >= min_counts[s] && counts[s] <= max_counts[s]);
            if ok {
                out.push(idx);
            }
        }
        Ok(out)
    }

    /// Reactive matches for `pattern_in` in a decoded microstate: sites equal
    /// to `pattern_in[0]` for an order-1 pattern, or ordered neighbor-pair
    /// matches for an order-2 pattern (each bond checked both ways, as the W
    /// builder does). Equals `-component.diagonal()` at unit rate; the graph
    /// exporter uses it as an edge multiplier.
    pub fn count_reactive(&self, state: Vec<u8>, pattern_in: Vec<u8>) -> usize {
        match pattern_in.len() {
            1 => state.iter().filter(|&&s| s == pattern_in[0]).count(),
            2 => {
                let mut c = 0;
                for &(si, sj) in &self.tile.neighbor_pairs {
                    for (s0, s1) in [(si, sj), (sj, si)] {
                        if state[s0] == pattern_in[0] && state[s1] == pattern_in[1] {
                            c += 1;
                        }
                    }
                }
                c
            }
            _ => 0,
        }
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
            "MEMKMBuilder(tile_settings={}, species_names={:?}, n_states={}, n_rxns={}, n_pairs={})",
            self.tile_settings.__repr__(),
            self.species_names,
            self.n_states,
            self.reactions.len(),
            self.tile.neighbor_pairs.len()
        )
    }
}

impl MEMKMBuilder {
    fn compute_offdiag(&self) -> (Vec<i32>, Vec<i32>, Vec<f64>, Vec<f64>) {
        let n = self.tile.n_states;
        // Upper bound on off-diagonal entries per state: each reaction can fire at
        // most once per matching site (order 1) or directed neighbor pair (order 2).
        let per_state_cap: usize = self
            .reactions
            .iter()
            .map(|rxn| match rxn.pattern_in.len() {
                1 => self.tile.l,
                2 => 2 * self.tile.neighbor_pairs.len(),
                _ => 0,
            })
            .sum();
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
                                let corr =
                                    im.rate_correction(&state, &[site], &self.tile.neighbors);
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
                            let mut fired_to: Option<usize> = None;
                            for (s0, s1) in [(si, sj), (sj, si)] {
                                if state[s0] == rxn.pattern_in[0] && state[s1] == rxn.pattern_in[1]
                                {
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
                                    let corr =
                                        im.rate_correction(&state, &[s0, s1], &self.tile.neighbors);
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
    /// one W = Σ_i(k_i * component_i), this builds each component_i
    /// separately, evaluated at k_i = 1 (i.e. corr instead of rxn.rate *
    /// corr, since rxn.rate is multiplied in on the Python side). Given the
    /// components, recovering W for any rate vector is just a weighted sum
    /// of sparse matrices; no re-walk of states/reactions/sites needed.
    /// It allows the Python side to change rates (sweep a concentration,
    /// drive k_i(t), or build derivative matrices like dW/dbeta) without
    /// ever calling back into Rust to rebuild the matrix from scratch.
    ///
    /// Loop body is otherwise identical to compute_offdiag, just keeping
    /// per-reaction rows/cols/vals/diag instead of one shared set.
    fn compute_w_components_coo(&self) -> Vec<(Vec<i32>, Vec<i32>, Vec<f64>)> {
        let n = self.tile.n_states;
        let n_rxns = self.reactions.len();
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
                                let corr =
                                    im.rate_correction(&state, &[site], &self.tile.neighbors);
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
                                if state[s0] == rxn.pattern_in[0] && state[s1] == rxn.pattern_in[1]
                                {
                                    let mut ns = state.clone();
                                    ns[s0] = rxn.pattern_out[0];
                                    ns[s1] = rxn.pattern_out[1];
                                    let to_idx = encode(&ns, self.tile.base);
                                    if to_idx == from_idx || fired_to == Some(to_idx) {
                                        continue;
                                    }
                                    fired_to = Some(to_idx);
                                    let corr =
                                        im.rate_correction(&state, &[s0, s1], &self.tile.neighbors);
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

        (0..n_rxns)
            .map(|ri| {
                let mut r = std::mem::take(&mut rows[ri]);
                let mut c = std::mem::take(&mut cols[ri]);
                let mut v = std::mem::take(&mut vals[ri]);
                for i in 0..n {
                    r.push(i as i32);
                    c.push(i as i32);
                    v.push(diag[ri][i]);
                }
                (r, c, v)
            })
            .collect()
    }

    /// Per-reaction ∂(dynamical-form W)/∂β at unit base rate, one pass over all
    /// states. Structurally identical to compute_w_components_coo, but each
    /// off-diagonal/diagonal entry is -ΔE·corr instead of corr, since
    /// corr = exp(-β·ΔE) ⇒ ∂corr/∂β = -ΔE·corr.
    fn compute_dw_dbeta_components_coo(&self) -> Vec<(Vec<i32>, Vec<i32>, Vec<f64>)> {
        let n = self.tile.n_states;
        let n_rxns = self.reactions.len();
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
                                    &state,
                                    &[site],
                                    &self.tile.neighbors,
                                );
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
                                if state[s0] == rxn.pattern_in[0] && state[s1] == rxn.pattern_in[1]
                                {
                                    let mut ns = state.clone();
                                    ns[s0] = rxn.pattern_out[0];
                                    ns[s1] = rxn.pattern_out[1];
                                    let to_idx = encode(&ns, self.tile.base);
                                    if to_idx == from_idx || fired_to == Some(to_idx) {
                                        continue;
                                    }
                                    fired_to = Some(to_idx);
                                    let (corr, delta_e) = im.rate_correction_and_delta_e(
                                        &state,
                                        &[s0, s1],
                                        &self.tile.neighbors,
                                    );
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
        (0..n_rxns)
            .map(|ri| {
                let mut r = std::mem::take(&mut rows[ri]);
                let mut c = std::mem::take(&mut cols[ri]);
                let mut v = std::mem::take(&mut vals[ri]);
                for i in 0..n {
                    r.push(i as i32);
                    c.push(i as i32);
                    v.push(diag[ri][i]);
                }
                (r, c, v)
            })
            .collect()
    }

    /// Steady-state W: last row replaced by normalisation (all 1s).
    ///
    /// W's rows are linearly dependent (each column sums to zero), so
    /// W @ Theta = 0 alone has a 1-dimensional null space and no unique
    /// solution. Dropping the last row and replacing it with all-1s plus a
    /// matching rhs of 1 in that slot turns it into Σ(Theta) = 1.
    fn compute_w_ss_coo(&self) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
        let n = self.tile.n_states;
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

// Everything pyo3-exposed gets registered here; this is what `import
// me_mkm._me_mkm` actually loads.
#[pymodule]
fn _me_mkm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TileSettings>()?;
    m.add_class::<InteractionModel>()?;
    m.add_class::<Reaction>()?;
    m.add_class::<MEMKMBuilder>()?;
    m.add_function(wrap_pyfunction!(decode_state, m)?)?;
    m.add_function(wrap_pyfunction!(encode_state, m)?)?;
    m.add_function(wrap_pyfunction!(state_counts, m)?)?;
    Ok(())
}
