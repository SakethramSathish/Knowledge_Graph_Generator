"""
graph_builder.py

Create a NetworkX graph from triplets and provide basic aggregation behaviour.
"""
import networkx as nx
from typing import List, Tuple

def build_graph_from_triplets(triplets: List[Tuple[str, str, str]]) -> nx.DiGraph:
    """
    triplets: list of (subject, predicate, object)
    Aggregates edge 'weight' and set of predicates per edge.
    """
    G = nx.DiGraph()
    for s, p, o in triplets:
        if s is None or o is None:
            continue
        if not G.has_node(s):
            G.add_node(s, label=s, type="entity")
        if not G.has_node(o):
            G.add_node(o, label=o, type="entity")
        if G.has_edge(s,o):
            G[s][o]["weight"] += 1
            G[s][o]["preds"].add(p)
        else:
            G.add_edge(s, o, label=p, weight=1, preds={p})
    #Convert preds set to list for serialization convenience
    for u, v, data in G.edges(data=True):
        data["preds"] = list(data.get("preds", []))
    return G