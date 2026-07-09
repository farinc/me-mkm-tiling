"""
Exports ME-MKM coverage-class transition graphs to JSON for memkm_viewer.html.
If omitted, defaults to ['*', 'A*', 'B*', ...].
"""

from collections import defaultdict

from me_mkm._me_mkm import decode_state

_AUTO_COLORS = [
    "#1A5FA8",
    "#2E7D32",
    "#B71C1C",
    "#7B1FA2",
    "#E65100",
    "#00695C",
    "#00838F",
    "#AD1457",
    "#558B2F",
    "#6A1B9A",
]

def _first_present(obj, *names, default=None):
    """Return the first non-empty attribute from obj."""
    for name in names:
        value = getattr(obj, name, None)
        if value is not None and value != "":
            return value
    return default


def _plain_rate_symbol(value: str) -> str:
    """Best-effort plain-text fallback for simple LaTeX-ish rate symbols."""
    if value is None:
        return value
    return (
        str(value)
        .replace(r"\mathrm", "")
        .replace(r"\text", "")
        .replace(r"\lambda", "λ")
        .replace(r"\times", "×")
        .replace("{", "")
        .replace("}", "")
        .replace("\\", "")
    )


def _reaction_rate_symbols(src, fallback):
    """Return (plain, latex) rate symbols from a dict or Reaction-like object.

    Convention:
      rate_symbol       = plain/export-safe label for Cytoscape edges
      rate_symbol_latex = optional LaTeX label for KaTeX/sidebar math

    Backward compatibility:
      symbol is accepted as an older/plain alias.
    """
    if isinstance(src, dict):
        plain = src.get("rate_symbol") or src.get("symbol")
        latex = src.get("rate_symbol_latex")
    else:
        plain = _first_present(src, "rate_symbol", "symbol")
        latex = _first_present(src, "rate_symbol_latex")

    if plain is None or plain == "":
        plain = _plain_rate_symbol(latex) if latex else fallback
    if latex is None or latex == "":
        latex = plain
    return str(plain), str(latex)


def _reactions_from_builder(builder) -> list:
    """Convert Reaction objects from builder into display-dicts.

    Uses rxn.rate_symbol as the plain/export-safe edge label.
    Uses optional rxn.rate_symbol_latex for KaTeX/sidebar math.
    Falls back to rxn.symbol and then rxn.name for older code.
    """
    result = []
    for i, rxn in enumerate(builder.get_reactions()):
        name = rxn.name or f"Reaction {i}"
        plain, latex = _reaction_rate_symbols(rxn, name or f"k_{i}")
        result.append(
            {
                "name": name,
                "forward": {
                    "pattern_in": list(rxn.pattern_in),
                    "pattern_out": list(rxn.pattern_out),
                    "rate_symbol": plain,
                    "rate_symbol_latex": latex,
                    "color": _AUTO_COLORS[i % len(_AUTO_COLORS)],
                },
                "reversible": False,
            }
        )
    return result

def neighbor_pairs(l, topology):
    """All undirected neighbor pairs for topology on a ring of l sites."""
    return sorted(
        {
            (min(i, (i + d) % l), max(i, (i + d) % l))
            for i in range(l)
            for d in topology.deltas
        }
    )


def tile_style(topology):
    """Return topology.style as an int, defaulting to brickwork/square row."""
    try:
        return int(getattr(topology, "style", 0))
    except (TypeError, ValueError):
        return 0


def site_positions(l, topology=None):
    """Icon layout for the supported tile styles.

    The returned coordinates are display coordinates only; the reaction
    topology is still determined by topology.deltas.  Styles are encoded by
    topology.style:

      0 = brickwork / old single-row square tile
      1 = Greek cross / five upright square cells
      2 = hexagonal ``creamcups`` / flat-top hexagons

    For style 2, coordinates are axial hex-grid coordinates interpreted by
    memkm_viewer.html.
    """
    style = tile_style(topology) if topology is not None else 0

    if style == 1:
        # Upright Greek cross.  For l=5 this is exact.  For larger l, keep
        # the first five as the cross and place extras to the right so the
        # renderer remains usable rather than failing.
        base = [[1, 0], [1, 1], [1, 2], [0, 1], [2, 1]]
        if l <= len(base):
            return {i: base[i] for i in range(l)}
        out = {i: base[i] for i in range(len(base))}
        for i in range(len(base), l):
            out[i] = [1, 3 + i - len(base)]
        return out

    if style == 2:
        # Seven-site hexagonal cup: six neighbors around one central site.
        # Site 0 is the bottom cup, then sites 1..5 proceed around the ring;
        # site 6 is the center.  For non-seven-site tiles, continue a simple
        # axial spiral so the icon remains valid.
        base = [[0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1], [1, 0], [0, 0]]
        if l <= len(base):
            return {i: base[i] for i in range(l)}
        out = {i: base[i] for i in range(len(base))}
        dirs = [[1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1]]
        i = len(base)
        radius = 2
        while i < l:
            q, r = 0, -radius
            for dq, dr in dirs:
                for _ in range(radius):
                    if i >= l:
                        break
                    out[i] = [q, r]
                    i += 1
                    q += dq
                    r += dr
                if i >= l:
                    break
            radius += 1
        return out

    # style 0: original brickwork icon layout
    return {i: [0, i] for i in range(l)}

def build_coverage_groups(l, n_ads):
    base = n_ads + 1
    groups = defaultdict(list)
    for idx in range(base**l):
        state = decode_state(idx, l, base)
        key = tuple(state.count(sp) for sp in range(1, n_ads + 1))
        groups[key].append(state)
    return groups


def count_reactive_pairs(state, pattern_in, pairs):
    if len(pattern_in) == 1:
        return sum(1 for site in state if site == pattern_in[0])
    c = 0
    for si, sj in pairs:
        for s0, s1 in [(si, sj), (sj, si)]:
            if state[s0] == pattern_in[0] and state[s1] == pattern_in[1]:
                c += 1
    return c


def verify_uniformity(groups, flat_reactions, pairs):
    for key, states in groups.items():
        for rxn in flat_reactions:
            counts = {
                count_reactive_pairs(st, rxn["pattern_in"], pairs) for st in states
            }
            if len(counts) > 1:
                return False, key, rxn["name"]
    return True, None, None

def tex_name(s):
    """Convert species name to KaTeX math. '*' → '{*}', 'A*' → r'\text{A}^{*}'."""
    if s == "*":
        return "{*}"
    if s.endswith("*"):
        return r"\text{" + s[:-1] + r"}^{*}"
    return r"\text{" + s + "}"


def make_equation(pattern_in, pattern_out, all_species, rate_fwd=None, rate_bwd=None):
    """Build a LaTeX equation string.

    rate_fwd only  →  A* + * \\xrightarrow{k} 2A*
    both rates     →  A* + * \\underset{k_-}{\\overset{k_+}{\\rightleftharpoons}} 2A*
    neither        →  A* + * \\rightarrow 2A*
    """
    from collections import Counter

    def fmt_side(codes):
        cnt = Counter(codes)
        terms = []
        for code in sorted(cnt, key=lambda c: -c):
            n = cnt[code]
            name = all_species[code] if code < len(all_species) else f"sp{code}"
            prefix = "" if n == 1 else str(n)
            terms.append(f"{prefix}{tex_name(name)}")
        return " + ".join(terms)

    lhs = fmt_side(pattern_in)
    rhs = fmt_side(pattern_out)

    if rate_fwd is not None and rate_bwd is not None:
        arrow = (
            r"\underset{"
            + rate_bwd
            + r"}{\overset{"
            + rate_fwd
            + r"}{\rightleftharpoons}}"
        )
    elif rate_fwd is not None:
        arrow = r"\xrightarrow{" + rate_fwd + "}"
    else:
        arrow = r"\rightarrow"

    return f"{lhs} {arrow} {rhs}"

def delta_counts_from_patterns(pattern_in, pattern_out, n_ads):
    delta = [0] * n_ads
    for s_in, s_out in zip(pattern_in, pattern_out):
        if s_in != s_out:
            if s_in > 0:
                delta[s_in - 1] -= 1
            if s_out > 0:
                delta[s_out - 1] += 1
    return delta

def build_graph(
    builder, display_reactions=None, species_names=None, title=None, node_colors=None
):
    """
    Build the coverage-class transition graph from an MEMKMBuilder.

    Parameters
    ----------
    builder : MEMKMBuilder
        Provides l, topology, n_ads, and (if display_reactions is None)
        the reaction list.
    display_reactions : list of dict, optional
        Override how reactions appear in the viewer.  See module docstring.
    species_names : list of str, optional
        Full species list including the empty site as index 0.
        Length must be n_ads + 1.  Defaults to ['*', 'A*', 'B*', ...].
        This matches the convention of me_mkm.tile.coverages().
    title : str, optional
    node_colors : dict, optional
        Map from counts-tuple to CSS colour string.
    """
    l = builder.l
    n_ads = builder.n_ads
    topology = builder.topology
    d = topology.d()

    # species_names includes the empty site (index 0)
    if species_names is None:
        all_species = ["*"] + [chr(65 + i) + "*" for i in range(n_ads)]
    elif len(species_names) == n_ads + 1:
        all_species = list(species_names)
    else:
        # Adsorbate-only list passed — prepend empty site
        all_species = ["*"] + list(species_names)
    adsorbate_names = all_species[1:]  # for coverage-class labels

    reactions = (
        display_reactions
        if display_reactions is not None
        else _reactions_from_builder(builder)
    )

    pairs = neighbor_pairs(l, topology)
    groups = build_coverage_groups(l, n_ads)
    style = tile_style(topology)
    icon_pos = site_positions(l, topology)
    icon_max_col = max(c for [r, c] in icon_pos.values()) + 1
    icon_rows = max(r for [r, c] in icon_pos.values()) + 1

    # Flatten into directed edge descriptors
    flat = []
    for rxn in reactions:
        fwd = rxn["forward"]
        col = fwd.get("color", "#888")
        flat.append(
            {
                "name": rxn["name"],
                "pattern_in": fwd["pattern_in"],
                "pattern_out": fwd["pattern_out"],
                "color": col,
                "rate_symbol": _reaction_rate_symbols(fwd, rxn["name"])[0],
                "rate_symbol_latex": _reaction_rate_symbols(fwd, rxn["name"])[1],
                "direction": "forward",
            }
        )
        if rxn.get("reversible") and "backward" in rxn:
            bwd = rxn["backward"]
            flat.append(
                {
                    "name": rxn["name"],
                    "pattern_in": bwd["pattern_in"],
                    "pattern_out": bwd["pattern_out"],
                    "color": col,
                    "rate_symbol": _reaction_rate_symbols(bwd, rxn["name"] + "⁻¹")[0],
                    "rate_symbol_latex": _reaction_rate_symbols(bwd, rxn["name"] + "⁻¹")[1],
                    "direction": "backward",
                }
            )

    # Sidebar display reactions — equation embeds rate symbol via \xrightarrow
    display_rxns = []
    for rxn in reactions:
        fwd = rxn["forward"]
        col = fwd.get("color", "#888")
        rate_fwd = _reaction_rate_symbols(fwd, rxn["name"])[1]
        if rxn.get("reversible") and "backward" in rxn:
            bwd = rxn["backward"]
            rate_bwd = _reaction_rate_symbols(bwd, rxn["name"] + "⁻¹")[1]
            eq = make_equation(
                fwd["pattern_in"],
                fwd["pattern_out"],
                all_species,
                rate_fwd=rate_fwd,
                rate_bwd=rate_bwd,
            )
        else:
            eq = make_equation(
                fwd["pattern_in"], fwd["pattern_out"], all_species, rate_fwd=rate_fwd
            )
        display_rxns.append(
            {
                "name": rxn["name"],
                "equation": eq,
                "rate_symbol": _reaction_rate_symbols(fwd, rxn["name"])[0],
                "rate_symbol_latex": rate_fwd,
                "color": col,
            }
        )

    ok, bad_key, bad_rxn = verify_uniformity(groups, flat, pairs)
    if not ok:
        print(
            f'WARNING: class {bad_key} rxn "{bad_rxn}" non-uniform — '
            f"edge multipliers are averages over the coverage class"
        )

    def canonical(counts):
        state = []
        for sp in range(1, n_ads + 1):
            state += [sp] * counts[sp - 1]
        state += [0] * (l - sum(counts))
        return state

    def avg_pairs(counts, rxn):
        rep = canonical(counts)
        if ok:
            return count_reactive_pairs(rep, rxn["pattern_in"], pairs)
        return sum(
            count_reactive_pairs(st, rxn["pattern_in"], pairs) for st in groups[counts]
        ) / len(groups[counts])

    def is_absorbing(counts):
        return all(avg_pairs(counts, r) == 0 for r in flat)

    def default_color(counts, absorp):
        if absorp:
            return "#94a3b8"
        nonzero = [i for i, c in enumerate(counts) if c > 0]
        COLORS = ["#f59e0b", "#ef4444", "#818cf8", "#34d399", "#fb923c"]
        return COLORS[nonzero[0] % len(COLORS)] if len(nonzero) == 1 else "#10b981"

    def node_id(counts):
        return "n" + "_".join(str(c) for c in counts)

    def graph_pos(counts):
        if n_ads == 1:
            return (counts[0] * 200.0, 0.0)
        elif n_ads == 2:
            nA, nB = counts
            return (nA * 220.0 - nB * 110.0, -nB * 190.0)
        else:
            h = hash(counts) % 2000
            return (h % 50 * 25.0, h // 50 * 25.0)

    STATES = sorted(groups.keys())
    nodes, edges = [], []

    for counts in STATES:
        rep = canonical(counts)
        absorp = is_absorbing(counts)
        col = (node_colors or {}).get(counts, default_color(counts, absorp))
        rxn_pairs = {
            r["name"] + ":" + r["direction"]: round(avg_pairs(counts, r), 4)
            for r in flat
        }
        counts_label = ", ".join(
            f"n({adsorbate_names[i]})={c}" for i, c in enumerate(counts)
        )
        x, y = graph_pos(counts)
        nodes.append(
            {
                "id": node_id(counts),
                "counts": list(counts),
                "counts_label": counts_label,
                "x": x,
                "y": y,
                "canonical_state": rep,
                "degeneracy": len(groups[counts]),
                "is_absorbing": absorp,
                "rxn_pairs": rxn_pairs,
                "color": col,
            }
        )

    ei = 0
    for counts in STATES:
        for rxn in flat:
            n = avg_pairs(counts, rxn)
            if n > 0:
                dc = delta_counts_from_patterns(
                    rxn["pattern_in"], rxn["pattern_out"], n_ads
                )
                dst = tuple(c + dc[i] for i, c in enumerate(counts))
                if dst in groups and all(c >= 0 for c in dst):
                    ni = int(n) if n == int(n) else round(n, 2)
                    edges.append(
                        {
                            "id": f"edge_{ei:03d}",
                            "source": node_id(counts),
                            "target": node_id(dst),
                            "rxn_name": rxn["name"],
                            "direction": rxn["direction"],
                            "n_pairs": round(n, 4),
                            "label": f"{ni}×",
                            "rate_symbol": rxn["rate_symbol"],
                            "rate_symbol_latex": rxn.get("rate_symbol_latex", rxn["rate_symbol"]),
                            "color": rxn["color"],
                        }
                    )
                    ei += 1

    return {
        "meta": {
            "title": title or f"l={l}, d={d} — {n_ads}-adsorbate ME-MKM",
            "l": l,
            "d": d,
            "n_ads": n_ads,
            "n_states": sum(len(v) for v in groups.values()),
            "n_classes": len(STATES),
            "uniform": ok,
            "tile_style": style,
            "species_names": adsorbate_names,
            "all_species": all_species,
            "icon_rows": icon_rows,
            "icon_max_col": icon_max_col,
        },
        "site_positions": {str(i): rc for i, rc in icon_pos.items()},
        "display_reactions": display_rxns,
        "nodes": nodes,
        "edges": edges,
    }
