# ─────────────────────────────────────────────────────
#  Domain-Specific AI Research Assistant
#  with Knowledge Graph — README
# ─────────────────────────────────────────────────────

## 🔬 What This Project Does

An end-to-end, fully local, 100% free AI research assistant that:
- Ingests academic PDFs
- Retrieves relevant content using Hybrid RAG (Dense + Sparse + RRF)
- Generates answers using a local LLM (Ollama)
- Verifies every claim for hallucinations using NLI (bart-large-mnli)
- Builds a multi-paper knowledge graph (NetworkX + Pyvis)

---

## 📁 Project Structure

```
PROJECT/
├── app.py                  # Main Streamlit UI
├── config.py               # All configuration (models, thresholds, paths)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── src/
│   ├── pdf_processor.py    # Section-aware PDF extraction (PyMuPDF)
│   ├── chunker.py          # Semantic chunking with overlap
│   ├── retriever.py        # Hybrid RAG: ChromaDB + BM25 + RRF
│   ├── llm.py              # Ollama LLM generation
│   ├── nli_verifier.py     # Hallucination detection (bart-large-mnli)
│   ├── knowledge_graph.py  # NetworkX + Pyvis entity graph
│   └── utils.py            # File helpers, formatters
└── data/
    ├── uploads/            # Uploaded PDFs stored here
    └── chroma_db/          # ChromaDB vector store (auto-created)
```

---

## ⚙️ Setup & Installation

### Step 1 — Install Python dependencies
```bash
cd C:\Users\adhit\Downloads\PROJECT
pip install -r requirements.txt
```

### Step 2 — Install Ollama
Download from: https://ollama.com

Then pull a model:
```bash
ollama pull llama3
```

### Step 3 — Start Ollama server
```bash
ollama serve
```

### Step 4 — Run the app
```bash
streamlit run app.py
```

---

## 🧠 Model Stack (All Free & Local)

| Component         | Model / Tool                        |
|-------------------|-------------------------------------|
| PDF parsing       | PyMuPDF                             |
| Embeddings        | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB         | ChromaDB                            |
| Sparse search     | rank-bm25                           |
| LLM generation    | Ollama (llama3 / mistral / gemma2)  |
| NLI verification  | facebook/bart-large-mnli            |
| Knowledge graph   | NetworkX + Pyvis                    |
| UI                | Streamlit                           |

---

## 🔄 Switching LLM Models

Edit `config.py`:
```python
OLLAMA_MODEL = "mistral"   # or gemma2, phi3, llama3
```

---

## 🎨 Answer Color Codes

| Color  | Meaning                                    |
|--------|--------------------------------------------|
| 🟢 Green  | Grounded — directly supported by source    |
| 🟡 Yellow | Inferred — partially supported             |
| 🔴 Red    | Hallucinated — not found in source papers  |

---

## 📝 Notes

- First run will download NLI model (~1.6GB) automatically
- ChromaDB persists between sessions in `data/chroma_db/`
- Upload multiple PDFs — the graph links entities across all of them
