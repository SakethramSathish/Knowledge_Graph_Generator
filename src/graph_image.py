"""
Convert a NetworkX graph into a PNG image (bytes) suitable for display/download.
Uses matplotlib (no headless browser required). Works for small-to-medium graphs.
"""
from typing import Optional, Tuple
import io
import math

import matplotlib.pyplot as plt
import networkx as nx

def graph_to_png_bytes(
        G: nx.Graph,
        figsize: Tuple[float, float] = (12, 12),
        dpi: int=200,
        layout: str="spring",
        node_size_base: int=600,
        font_size: int=8,
        show_labels: bool=True,
        cmap: str="tab20",
) -> bytes:
    """
    Render NetworkX graph G to a PNG image and return image bytes.

    Args:
      G: NetworkX graph (DiGraph or Graph)
      figsize: matplotlib figure size
      dpi: output DPI
      layout: 'spring' (default), 'kamada_kawai', 'spectral', or 'circular'
      node_size_base: base node size (scaled by degree)
      font_size: label font size
      show_labels: whether to draw node labels
      cmap: matplotlib colormap name for node coloring

    Returns:
      PNG image bytes
    """
    if G is None or len(G) == 0:
        raise ValueError("Graph is empty or None")
    
    if layout == "spring":
        #k controls spacing; scale with number of nodes
        n = G.number_of_nodes()
        k = 0.5 if n <= 50 else 0.9 if n <= 200 else 2.0
        pos = nx.spring_layout(G, k=k, seed=42, iterations=200)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    #Node sizes: Base scaled by node degree (so hubs are larger)
    degrees= dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1
    node_sizes = []
    for n in G.nodes():
        deg = degrees.get(n,0)
        #Scale non-linearly for better spread
        size = node_size_base * (1 + math.log1p(deg))
        node_sizes.append(size)

    #Node colors: optional - color by node 'type' attribute if present,
    #otherwise by degree buckets
    node_types = [G.nodes[n].get("type") for n in G.nodes()]
    use_types = any(node_types)
    if use_types:
        #map types to integers
        unique_types = {}
        colors = []
        idx = 0
        for t in node_types:
            if t not in unique_types:
                unique_types[t] = idx
                idx += 1
            colors.append(unique_types[t])
    else:
        #Bucket degrees into small number of colors
        colors = []
        for n in G.nodes():
            deg = degrees.get(n, 0)
            #Bucket into 5 bins
            if deg == 0:
                colors.append(0)
            else:
                colors.append(min(5, int(math.log1p(deg))))

    #Create figure
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    ax.set_facecolor("white")
    plt.axis("off")

    #Draw edges (thin)
    nx.draw_networkx_edges(G, pos, alpha=0.6, width=0.8, arrows=True, arrowstyle="-|>", arrowsize=12, edge_color="#666666")

    #Draw nodes
    cmap = plt.get_cmap(cmap)
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=colors,
        cmap=cmap,
        linewidths=0.5,
        edgecolors="#222222",
        alpha=0.95,
    )

    #Draw labels (truncate long labels and wrap if needed)
    if show_labels:
        labels = {}
        for n in G.nodes():
            lab = str(G.nodes[n].get("label", n))
            #Shorten very long labels
            if len(lab) > 30:
                lab = lab[:27] + "..."
            labels[n] = lab
        nx.draw_networkx_labels(G, pos, labels, font_size=font_size, font_family="sans-serif")

    # optionally draw edge labels (small)
    # build a label dict if edge has 'label'
    edge_labels = {}
    for u,v,d in G.edges(data=True):
        lab = d.get("label")
        if lab:
            edge_labels[(u, v)] = str(lab) if len(str(lab)) <=30 else str(lab)[:27] + "..."
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=max(6, font_size-1))

    #Tight layout and render to bytes
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    buf.seek(0)
    img_bytes = buf.getvalue()
    plt.close()
    return img_bytes