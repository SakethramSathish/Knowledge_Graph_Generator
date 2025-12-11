"""
embeddings.py

Wraps sentence-transformers model for embedding and deduplication.
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed_texts(texts: List[str]):
    m = get_model()
    return m.encode(texts, show_progress_bar=False, convert_to_numpy=True)

def deduplicate_entities(entity_texts: List[str], threshold: float=0.75) -> List[str]:
    """
    Cluster by cosine similarity threshold (greedy) and return a representative string per cluster.
    Representative chosen as the longest mention in cluster.
    """
    if not entity_texts:
        return []
    embs = embed_texts(entity_texts)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    embs_norm = embs / norms
    n = len(embs_norm)
    visited = set()
    clusters = []
    for i in range(n):
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)
        for j in range(i+1, n):
            if j in visited:
                continue
            cos = float(np.dot(embs_norm[i], embs_norm[j]))
            if cos >= threshold:
                cluster.append(j)
                visited.add(j)
        clusters.append(cluster)
    reps = [max([entity_texts[i] for i in c], key=len) for c in clusters]
    return reps