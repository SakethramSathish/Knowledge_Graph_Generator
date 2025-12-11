# Knowledge Graph Generator from PDFs

A comprehensive, production-ready Streamlit application that automatically extracts structured knowledge from PDF documents and visualizes it as an interactive knowledge graph.

## Overview

This tool leverages advanced NLP techniques to transform unstructured PDF text into structured knowledge representations (entity-relation triplets) and visualizes them as interactive and static network graphs. It's designed for researchers, analysts, and enterprises working with document-heavy workflows.

**Key Features:**
- 📄 **PDF Text Extraction** with OCR support for scanned documents
- 🤖 **Advanced NLP Pipeline** including entity recognition, relation extraction, and coreference resolution
- 🧠 **Entity Deduplication** using semantic embeddings (all-MiniLM-L6-v2)
- 🕸️ **Knowledge Graph Construction** from triplets (subject, predicate, object)
- 📊 **Interactive Visualization** with Pyvis for exploration
- 📸 **Static PNG Export** for reports and presentations
- ⬇️ **Multiple Export Formats** (HTML, JSON, CSV)

---

## Project Structure

```
Knowledge Graph Generator/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── constraints.txt                 # Pinned package versions (for stability)
├── README.md                       # This file
│
├── src/                            # Core pipeline modules
│   ├── __init__.py
│   ├── ingestion.py               # PDF text extraction (with OCR fallback)
│   ├── preprocess.py              # Text cleaning and sentence splitting
│   ├── nlp_pipeline.py            # Entity/relation extraction, coreference
│   ├── embeddings.py              # Semantic embeddings and deduplication
│   ├── graph_builder.py           # Triplet to graph conversion
│   ├── visualize.py               # Pyvis HTML generation
│   ├── graph_image.py             # Matplotlib PNG rendering
│   └── utils.py                   # Helper utilities (JSON I/O)
│
├── lib/                            # External libraries and assets
│   ├── tom-select/                # TomSelect dropdown component (CSS/JS)
│   ├── vis-9.1.2/                 # Vis.js network visualization
│   └── bindings/                  # Custom JS bindings for graph interaction
│
└── models/                         # Placeholder for model files (if any)
```

---

## Installation

### Prerequisites

- **Python 3.11+** (tested on 3.11.7)
- **System Dependencies:**
  - Tesseract OCR (optional, for scanned PDF support): [Install Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
  - Git (for cloning, if applicable)

### Setup Instructions

#### 1. Clone or Download the Project

```bash
cd /path/to/Knowledge\ Graph\ Generator
```

#### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Activate (Windows CMD)
.venv\Scripts\Activate

# Or PowerShell
.\.venv\Scripts\Activate
```

#### 3. Install Dependencies

```bash
# Upgrade pip, setuptools, wheel
pip install -U pip setuptools wheel

# Install core requirements (respecting constraints for stability)
pip install -r requirements.txt -c constraints.txt
```

**Note:** If you encounter numpy/thinc binary compatibility issues, the constraints file pins numpy==1.26.4 for stability. See [Troubleshooting](#troubleshooting) if issues persist.

#### 4. Verify Installation

```bash
python -c "import streamlit, spacy, sentence_transformers, pyvis; print('✓ All imports successful')"
```

---

## Usage

### Running the Application

```bash
# Activate virtual environment
.\.venv\Scripts\Activate

# Launch Streamlit app
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

### Workflow

1. **Upload PDF**: Click "Upload a PDF File" and select a document (native PDFs or scanned images)
2. **Configure Settings**:
   - Enable OCR for scanned PDFs (requires Tesseract)
   - Adjust entity deduplication similarity threshold (50-95%)
   - Set preview sentence count
3. **Process**:
   - Extract text and sentences
   - Resolve pronouns (naive coreference)
   - Extract entities and relations
   - Deduplicate entities via semantic similarity
   - Build knowledge graph
4. **Explore**:
   - View interactive graph (Pyvis)
   - Inspect graph statistics
   - Download as HTML, JSON, or CSV
5. **Export**:
   - Interactive HTML for sharing
   - PNG snapshot for reports
   - Triplets CSV for further analysis

---

## Core Modules

### `ingestion.py` - PDF Text Extraction
Extracts text from PDF pages using PyMuPDF, with OCR fallback for scanned documents.

**Key Functions:**
- `extract_text_from_pdf(path_or_bytes, ocr_if_needed=True, dpi=200)` → List[Dict]
  - Accepts PDF file path or raw bytes
  - Falls back to Tesseract OCR if text extraction fails

### `preprocess.py` - Text Cleaning & Sentence Splitting
Cleans text (normalize whitespace, remove control chars) and splits into sentences.

**Key Functions:**
- `clean_text(text)` → str
- `split_sentences(text)` → List[str]
  - Uses NLTK Punkt tokenizer (with fallback regex splitter)

### `nlp_pipeline.py` - Entity & Relation Extraction
Uses spaCy transformers model (en_core_web_trf) for NER and dependency parsing for relation extraction.

**Key Functions:**
- `extract_entities_with_spans(text)` → List[Dict]
- `extract_relations_from_sentence(sentence)` → List[Tuple[str, str, str]]
  - Heuristic-based: "X of Y" patterns, dependency SVO triples, sequential entity fallback
- `naive_coref_resolution(sentences)` → List[str]
  - Replaces pronouns with most recent entity mention

### `embeddings.py` - Semantic Embeddings & Deduplication
Uses `sentence-transformers` (all-MiniLM-L6-v2) for entity similarity.

**Key Functions:**
- `deduplicate_entities(entity_texts, threshold=0.75)` → List[str]
  - Greedy clustering by cosine similarity
  - Returns longest mention per cluster as representative

### `graph_builder.py` - Graph Construction
Converts triplets into a NetworkX DiGraph with edge aggregation.

**Key Functions:**
- `build_graph_from_triplets(triplets)` → nx.DiGraph
  - Aggregates edge weights and collects predicates per edge

### `visualize.py` - Interactive Visualization
Generates interactive Pyvis HTML for graph exploration.

**Key Functions:**
- `nx_to_pyvis(G, height="700px", width="100%")` → str
  - Physics simulation, hover interaction, multiselect
  - Returns path to temporary HTML file

### `graph_image.py` - Static PNG Rendering
Renders graphs as high-quality PNG images using Matplotlib.

**Key Functions:**
- `graph_to_png_bytes(G, figsize=(12, 12), dpi=200, layout="spring", ...)` → bytes
  - Supports layouts: spring, kamada-kawai, spectral, circular
  - Returns PNG bytes ready for download/display

---

## Dependencies

| Package | Purpose |
|---------|---------|
| **streamlit** | Web UI framework |
| **spacy + en_core_web_trf** | NLP: Named Entity Recognition, Dependency Parsing |
| **sentence-transformers** | Semantic embeddings for entity deduplication |
| **networkx** | Graph construction and algorithms |
| **pyvis** | Interactive graph visualization (Pyvis/Vis.js) |
| **matplotlib** | Static graph rendering (PNG) |
| **PyMuPDF (fitz)** | PDF text extraction |
| **Pillow** | Image processing (for OCR) |
| **pytesseract** | OCR support (requires system Tesseract) |
| **nltk** | Natural Language Toolkit (tokenization) |
| **torch + transformers** | Deep learning dependencies (for spaCy) |

---

## Configuration

### Environment Variables

Set before running the app for SSL/proxy support:

```bash
# Windows CMD
set SSL_CERT_FILE=C:\path\to\cacert.pem

# PowerShell
$env:SSL_CERT_FILE = "C:\path\to\cacert.pem"
```

The app automatically uses `certifi` for SSL validation.

### Streamlit Config (Optional)

Create `.streamlit/config.toml` to customize:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[server]
maxUploadSize = 500
```

---

## Performance Notes

- **Small graphs (< 50 nodes):** Instant rendering
- **Medium graphs (50-500 nodes):** 1-10 seconds
- **Large graphs (500+ nodes):** May slow down; consider filtering
- **OCR Processing:** 10-30 seconds per scanned page (depending on resolution)
- **Model Loading:** ~30 seconds on first run (spaCy, sentence-transformers cached after)

---

## Troubleshooting

### 1. NumPy/Thinc Binary Incompatibility

**Error:** `ValueError: numpy.dtype size changed...`

**Solution:**
```bash
pip install --force-reinstall numpy==1.26.4
pip install --force-reinstall pyvis jinja2
```

### 2. NLTK Punkt Missing

**Error:** `LookupError: Resource punkt_tab not found`

**Solution:** The app includes an automatic fallback, but you can pre-download:
```bash
python -c "import nltk; nltk.download('punkt')"
```

### 3. Tesseract Not Found (OCR)

**Error:** `TesseractNotFoundError`

**Solution:**
- Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
- Windows: Add to PATH or set `pytesseract.pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'`

### 4. SSL Certificate Verification Failed

**Error:** `SSLError: CERTIFICATE_VERIFY_FAILED`

**Solution:**
```bash
pip install -U certifi
# App automatically uses certifi now; restart Streamlit
```

### 5. Pyvis Template Missing

**Error:** `AttributeError: 'NoneType' object has no attribute 'render'`

**Solution:**
```bash
pip install --force-reinstall pyvis jinja2
```

### 6. Out of Memory

**Cause:** Large PDFs or dense graphs with many nodes/edges

**Solution:**
- Use a smaller PDF
- Increase deduplication threshold (→ fewer unique entities)
- Run on a machine with more RAM (minimum 8GB recommended)

---

## Example Workflow

### Sample Input
A PDF about "Artificial Intelligence and Healthcare" containing:
- 50 pages
- ~5000 sentences
- Mixed formatted text and tables

### Processing Steps
1. Extract ~8000 sentences via PyMuPDF
2. Split into 4500 clean sentences
3. Extract 200+ entity mentions (Person, Organization, Product)
4. Deduplicate to ~80 unique entities
5. Extract ~300 relation triplets
6. Build graph: **80 nodes, 300 edges**

### Output
- **Interactive HTML:** Explore graph, hover over nodes, zoom
- **PNG Image:** Embed in reports
- **JSON:** Load in D3, cytoscape, or custom apps
- **CSV:** Triplets for knowledge base ingestion

---

## Advanced Usage

### Custom NLP Models

To use a different spaCy model:

```python
# In src/nlp_pipeline.py
nlp = spacy.load("en_core_web_sm")  # Smaller, faster
# or
nlp = spacy.load("en_core_web_lg")  # Larger, more accurate
```

### Custom Embeddings Model

```python
# In src/embeddings.py
MODEL_NAME = "all-mpnet-base-v2"  # Larger, slower but more accurate
# or other sentence-transformers models
```

### Batch Processing

Extend `app.py` to process multiple PDFs:

```python
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
for uploaded_file in uploaded_files:
    # Process each file
```

### Graph Export to External Tools

```python
# Export to Cytoscape JSON
import cytoscape
cy = cytoscape.Cytoscape(G)
cy.to_file("graph.json")

# Or use graphml for Gephi
nx.write_graphml(G, "graph.graphml")
```

---

## Limitations & Future Work

### Current Limitations
- Relation extraction is heuristic-based (not trained models)
- Coreference resolution is naive (only pronouns, recent entity)
- Graph visualization can lag with 1000+ nodes
- No support for multi-page layout
- Single-document processing (batch mode not built-in)

### Planned Enhancements
- [ ] Trained relation extraction model (SpanBERT, etc.)
- [ ] Advanced coreference resolution (spaCy coref models)
- [ ] Graph clustering and community detection
- [ ] Batch PDF processing
- [ ] Database backend (Neo4j) for persistence
- [ ] API endpoint for programmatic access
- [ ] Multi-document entity linking and reconciliation

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -am 'Add your feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Submit a Pull Request

---

## License

This project is provided as-is for educational and research purposes.

---

## Author & Support

**Created:** December 2024

**Questions or Issues?**
- Check [Troubleshooting](#troubleshooting) section
- Review logs in terminal for detailed error messages
- Ensure all dependencies are correctly installed per [Installation](#installation)

---

## Citation

If you use this tool in research, please cite:

```bibtex
@software{KnowledgeGraphGenerator2024,
  title = {Knowledge Graph Generator from PDFs},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourrepo/knowledge-graph-generator}
}
```

---

## Acknowledgments

- **spaCy** for transformer-based NLP
- **Sentence-Transformers** for semantic embeddings
- **NetworkX** for graph algorithms
- **Pyvis** for interactive visualization
- **Streamlit** for rapid prototyping

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | Dec 2024 | Initial release with PDF extraction, NLP pipeline, and dual visualization |

---

**Last Updated:** December 11, 2024
