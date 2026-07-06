"""
The bundled memkm_viewer.html is a Cytoscape.js-based interactive viewer
that normally loads JSON via a file picker. The functions here produce a
self-contained HTML file with the graph data already embedded.

Usage
-----
    from me_mkm import build_graph
    from me_mkm.viewer import save_html, open_viewer

    data = build_graph(l=5, d=2, n_ads=1, reactions=[...])
    save_html(data, "my_graph.html")   # write only
"""

import json
from importlib.resources import files
from pathlib import Path


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
        Output of :func:`me_mkm.export_graph.build_graph`.
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
