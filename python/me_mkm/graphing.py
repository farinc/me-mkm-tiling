"""
Exports ME-MKM coverage-class transition graphs to JSON for memkm_viewer.html
"""

from me_mkm.microstates import coverage_classes, pattern_delta_counts
from me_mkm.observables import class_average_matches

import json
from importlib.resources import files
from pathlib import Path

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


def tile_style(topology):
    """Return topology.style as an int, defaulting to brickwork/square row."""
    try:
        return int(getattr(topology, "style", 0))
    except TypeError, ValueError:
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


def build_graph(builder, display_reactions=None, title=None, node_colors=None, Theta=None):
    """
    Build the coverage-class transition graph from an MEMKMBuilder.

    Parameters
    ----------
    builder : MEMKMBuilder
        Provides l, topology, n_ads, and (if display_reactions is None)
        the reaction list.
    display_reactions : list of dict, optional
        Override how reactions appear in the viewer.  See module docstring.
    title : str, optional
    node_colors : dict, optional
        Map from counts-tuple to CSS colour string.
    Theta : array over microstates, optional
        Weight the edge multipliers by a solved distribution: each class's
        multiplier becomes the Theta-conditional expected match count instead
        of the plain average over class members (see
        observables.class_average_matches).
    """
    l = builder.l
    n_ads = builder.n_species - 1  # species 1..n_species-1 (index 0 is the reference)
    topology = builder.tile_settings
    d = topology.d()

    # species_names has the reference species at index 0; default to the
    # builder's own names.
    all_species = list(builder.species_names)

    if "*" in all_species:  # Explicitly the empty site
        adsorbate_names = all_species[1:]  # for coverage-class labels
    else:
        adsorbate_names = all_species

    reactions = (
        display_reactions
        if display_reactions is not None
        else _reactions_from_builder(builder)
    )

    # local counts-tuple -> indices lookup for the graph build below; plain-int
    # keys so counts land in the graph JSON as ints, not numpy int64.
    groups = {
        tuple(int(c) for c in counts): idxs
        for counts, idxs in coverage_classes(builder)
    }
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
                    "rate_symbol_latex": _reaction_rate_symbols(
                        bwd, rxn["name"] + "⁻¹"
                    )[1],
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

    # Per-class edge multipliers, one dict per flat reaction. The class-rate
    # math (and the optional Theta weighting) lives in
    # observables.class_average_matches; here we only report non-uniformity.
    multipliers = []
    ok = True
    for rxn in flat:
        avg, nonuniform = class_average_matches(builder, rxn["pattern_in"], Theta)
        multipliers.append(avg)
        if nonuniform and ok:
            ok = False
            print(
                f'WARNING: class {nonuniform[0]} rxn "{rxn["name"]}" non-uniform — '
                f"edge multipliers are averages over the coverage class"
            )

    def canonical(counts):
        state = []
        for sp in range(1, n_ads + 1):
            state += [sp] * counts[sp - 1]
        state += [0] * (l - sum(counts))
        return state

    def is_absorbing(counts):
        return all(m[counts] == 0 for m in multipliers)

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
            r["name"] + ":" + r["direction"]: round(m[counts], 4)
            for r, m in zip(flat, multipliers)
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
        for rxn, m in zip(flat, multipliers):
            n = m[counts]
            if n > 0:
                dc = pattern_delta_counts(
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
                            "rate_symbol_latex": rxn.get(
                                "rate_symbol_latex", rxn["rate_symbol"]
                            ),
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


def _template() -> str:
    return files("me_mkm").joinpath("memkm_viewer.html").read_text(encoding="utf-8")


def save_html(graph_data: dict, path: str = "me_mkm_graph.html") -> Path:
    """
    Write a self-contained HTML viewer with *graph_data* embedded.

    The output file has no external dependencies beyond CDN-hosted KaTeX
    and Cytoscape.js — just open it in any modern browser.

    Parameters
    ----------
    graph_data : dict
        Output of :func:`me_mkm.graphing.build_graph`.
    path : str or Path
        Destination file path (default: ``me_mkm_graph.html``).

    Returns
    -------
    Path
        Resolved absolute path of the written file.
    """
    html = _template()
    json_blob = json.dumps(graph_data, ensure_ascii=False)
    # Inject a load() call just before the closing tag of the module script.
    inject = f"\n// embedded data\nload({json_blob});\n"
    html = html.replace("</script>\n</body>", inject + "</script>\n</body>", 1)
    out = Path(path).resolve()
    out.write_text(html, encoding="utf-8")
    return out
