"""
Streamlit application that drives the whole pipeline from upload -> KG -> visualize.
This version reads PDF bytes directly (no temp file required) and also generates
a static PNG snapshot of the knowledge graph.
"""

import streamlit as st
import json
import pandas as pd
import networkx as nx
from networkx.readwrite import json_graph
import os
import certifi

from src.ingestion import extract_text_from_pdf
from src.preprocess import split_sentences
from src.nlp_pipeline import (
    extract_entities_with_spans,
    extract_relations_from_sentence,
    naive_coref_resolution,
)
from src.embeddings import deduplicate_entities
from src.graph_builder import build_graph_from_triplets
from src.visualize import nx_to_pyvis

# New: image generator (must create src/graph_image.py as provided earlier)
from src.graph_image import graph_to_png_bytes

# Ensure requests/huggingface_hub use certifi's CA bundle
os.environ["SSL_CERT_FILE"] = certifi.where()

st.set_page_config(page_title="Knowledge Graph Generator", layout="wide")
st.title("Knowledge Graph Generator from PDFs")

st.sidebar.header("Settings")
ocr = st.sidebar.checkbox("Enable OCR for scanned PDFs (requires system tesseract)", value=True)
threshold = st.sidebar.slider("Deduplication similarity (%)", 50, 95, 80)
preview_sentences = st.sidebar.number_input("Preview sentences", min_value=3, max_value=200, value=20)

uploaded = st.file_uploader("Upload a PDF File", type=["pdf"])

if uploaded:
    st.success("PDF uploaded - processing...")
    pdf_bytes = uploaded.read()

    with st.spinner("Extracting text..."):
        pages = extract_text_from_pdf(pdf_bytes, ocr_if_needed=ocr)

    if not pages:
        st.error("No pages extracted from PDF.")
        st.stop()

    st.subheader("Extracted text (first page)")
    st.code(pages[0]["text"][:3000] or "No text on first page.")

    # Sentences
    sentences = []
    for p in pages:
        sentences.extend(split_sentences(p["text"]))

    st.info(f"Extracted {len(sentences)} sentences")

    # Naive Coref
    with st.spinner("Resolving pronouns (naive coref)..."):
        resolved = naive_coref_resolution(sentences)

    st.subheader("Sample sentences after coref")
    for s in resolved[:preview_sentences]:
        st.write("-", s)

    # Entities and Relations
    all_entities = []
    triplets = []
    with st.spinner("Extracting entities & relations..."):
        for sent in resolved:
            ents = extract_entities_with_spans(sent)
            for e in ents:
                all_entities.append(e["text"])
            rels = extract_relations_from_sentence(sent)
            for r in rels:
                triplets.append(r)

    st.subheader("Entities (sample)")
    st.write(list(dict.fromkeys(all_entities))[:100])

    # Deduplicate Entities
    with st.spinner("Deduplicating entitites..."):
        reps = deduplicate_entities(list(dict.fromkeys(all_entities)), threshold=threshold/100.0)

    st.subheader("Entity representative (sample)")
    st.write(reps[:100])

    # Normalize Triplets
    def normalize(e: str) -> str:
        if not e:
            return e
        for r in reps:
            if e.lower() in r.lower() or r.lower() in e.lower():
                return r
        return e

    normalized_triplets = [(normalize(s), p, normalize(o)) for (s, p, o) in triplets if s and o]

    st.info(f"Total triplets: {len(normalized_triplets)}")

    # Build graph
    with st.spinner("Building the knowledge graph..."):
        G = build_graph_from_triplets(normalized_triplets)

    st.subheader("Graph statistics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Nodes", G.number_of_nodes())
    c2.metric("Edges", G.number_of_edges())
    c3.metric("Triplets", len(normalized_triplets))

    if G.number_of_nodes() == 0:
        st.warning("Graph is empty - try another document or enable OCR for scanned PDFs.")
    else:
        st.subheader("Interactive Graph")
        html_path = None
        try:
            # Generate interactive PyVis HTML
            html_path = nx_to_pyvis(G)
            with open(html_path, "r", encoding="utf-8") as fh:
                html = fh.read()
            st.components.v1.html(html, height=700, scrolling=True)

            st.subheader("Download Options")
            # HTML
            with open(html_path, "rb") as fh:
                st.download_button("Download graph HTML", data=fh, file_name="knowledge_graph.html", mime="text/html")

            # JSON (node-link)
            graph_json = json.dumps(json_graph.node_link_data(G), indent=2)
            st.download_button("Download graph JSON", data=graph_json, file_name="knowledge_graph.json", mime="application/json")

            # triplets CSV
            if normalized_triplets:
                df = pd.DataFrame(normalized_triplets, columns=["subject", "predicate", "object"])
                st.download_button("Download triplets CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="triplets.csv", mime="text/csv")

            # --- NEW: generate PNG snapshot and display + download ---
            try:
                png_bytes = graph_to_png_bytes(G, figsize=(12, 12), dpi=200, layout="spring", font_size=8)
                st.subheader("Static Image Snapshot of the Graph")
                st.image(png_bytes, use_column_width=True)

                st.download_button(
                    label="Download graph as PNG",
                    data=png_bytes,
                    file_name="knowledge_graph.png",
                    mime="image/png"
                )
            except Exception as e_img:
                st.warning(f"Could not generate PNG snapshot: {e_img}")

            # show small edge table (optional)
            st.subheader("Graph edges (sample)")
            edges_list = []
            for u, v, d in G.edges(data=True):
                edges_list.append({
                    "source": u,
                    "target": v,
                    "label": d.get("label", ""),
                    "weight": d.get("weight", 1),
                    "preds": ",".join(d.get("preds", [])) if d.get("preds") else "",
                })
            if edges_list:
                st.dataframe(pd.DataFrame(edges_list).head(200))
            else:
                st.info("No edges to show.")
        finally:
            # cleanup the temporary pyvis html file if created
            if html_path:
                try:
                    import os
                    os.remove(html_path)
                except Exception:
                    pass
else:
    st.info("Please upload a PDF file to begin.")
