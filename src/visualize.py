"""
visualize.py

Convert NetworkX graph to pyvis HTML for embedding.
"""
from pyvis.network import Network
import networkx as nx
import tempfile

def nx_to_pyvis(G: nx.Graph, height: str="700px", width: str="100%") -> str:
    """
    Returns path to a generated HTML file containing the pyvis visualization.
    Caller should remove the file when done.
    """
    net = Network(height=height, width=width, notebook=False)
    net.from_nx(G)
    #Set some physics options for nicer layout
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 200}
      },
      "interaction": {
        "hover": true,
        "multiselect": true
      }
    }
    """)
    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
    try:
        net.write_html(tmp.name)
    except AttributeError as e:
        # template.render() failed: template is None (missing Jinja2 or pyvis templates)
        raise RuntimeError(
            "pyvis failed to render HTML template. "
            "Fix: activate venv and run:\n\n"
            "  .venv\\Scripts\\Activate\n"
            "  pip install --force-reinstall pyvis jinja2\n\n"
            "Then restart the app."
        ) from e
    return tmp.name