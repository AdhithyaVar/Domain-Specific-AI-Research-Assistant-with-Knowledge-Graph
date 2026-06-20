# ─────────────────────────────────────────────────────────────────────────────
#  config.py — Central configuration
#
#  SYSTEM REQUIREMENTS (Minimum):
#    RAM   : 4 GB
#    CPU   : Any modern 4-core (Intel i5 / AMD Ryzen 5 or better)
#    GPU   : Not required — fully CPU-based
#    Disk  : ~1.6 GB free (models + ChromaDB)
#    OS    : Windows 10/11, Ubuntu 20.04+, macOS 12+
#    Python: 3.10 or 3.11 recommended (NOT 3.14 — Pydantic v1 issues)
#
#  TOTAL MODEL DOWNLOAD SIZE:
#    all-MiniLM-L6-v2            →   80 MB  (embeddings)
#    nli-MiniLM2-L6-H768         →   67 MB  (NLI verifier)
#    ms-marco-MiniLM-L-6-v2      →   67 MB  (reranker) ← NEW
#    phi3.5:mini (Ollama)        →  2.2 GB  (LLM)
#    ─────────────────────────────────────────────────
#    Total                       ~  2.42 GB
#
#  AUDIT FIXES APPLIED (see RAG audit, Priority Fixes #1-#6):
#    #1 Per-chunk NLI scoring (was: 1000-char truncated concat premise)
#    #2 Preamble/meta-sentence filtering before claim split
#    #3 Real subword-tokenizer chunking (was: whitespace .split())
#    #4 Cross-encoder reranking stage added after RRF fusion
#    #5 NLI max_length raised 256 → 384 (safe now that scoring is per-chunk)
#    #6 Near-duplicate chunk deduplication before LLM context assembly
# ─────────────────────────────────────────────────────────────────────────────

import os

# ── Ollama LLM ───────────────────────────────────────────────────────────────
OLLAMA_MODEL     = "phi3.5:mini"
OLLAMA_BASE_URL  = "http://localhost:11434"
LLM_TEMPERATURE  = 0.1
LLM_MAX_TOKENS   = 1024

OLLAMA_FALLBACK_MODELS = ["phi3:mini", "llama3.2:3b", "gemma2:2b"]

# ── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"
EMBEDDING_MAX_SEQ_TOKENS = 256          # MiniLM-L6 hard limit — chunker must respect this

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR  = "./data/chroma_db"
CHROMA_COLLECTION   = "research_papers"

# ── Chunking (FIX #3) ─────────────────────────────────────────────────────────
#
#  CHUNK_SIZE/OVERLAP are now in REAL SUBWORD TOKENS (HF tokenizer-counted),
#  not whitespace words. Kept comfortably under EMBEDDING_MAX_SEQ_TOKENS (256)
#  and NLI_MAX_LENGTH (384) so nothing is silently truncated downstream.
#
CHUNK_SIZE          = 220              # subword tokens per chunk
CHUNK_OVERLAP       = 50              # ~23% overlap
CHUNK_MIN_TOKENS    = 15               # drop fragments smaller than this
SNAP_TO_SENTENCE    = True             # never cut mid-sentence

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_DENSE         = 10
TOP_K_SPARSE        = 10
TOP_K_RRF           = 8               # candidates surviving RRF, BEFORE rerank
TOP_K_FINAL         = 5               # final chunks after rerank, sent to LLM

# ── Reranker (FIX #4 — NEW) ───────────────────────────────────────────────────
#
#  cross-encoder/ms-marco-MiniLM-L-6-v2:
#    - 67 MB, CPU-friendly, ~10-15ms per pair on CPU
#    - Trained specifically for query-passage relevance reranking (MS MARCO)
#    - Closes the gap between RRF (fusion heuristic) and true relevance scoring
#
RERANKER_MODEL      = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_ENABLED    = True

# ── Deduplication (FIX #6 — NEW) ──────────────────────────────────────────────
DEDUP_SIMILARITY_THRESHOLD = 0.92      # cosine sim above this → treat as duplicate

# ── NLI Hallucination Verifier (FIXES #1, #2, #5) ─────────────────────────────
#
#  cross-encoder/nli-MiniLM2-L6-H768:
#    - 67 MB, CPU-friendly
#    - Label order: [contradiction=0, entailment=1, neutral=2]
#    - NOW scored PER-CHUNK (max entailment across chunks), not concatenated
#
NLI_MODEL               = "cross-encoder/nli-MiniLM2-L6-H768"
NLI_MAX_LENGTH           = 384         # raised from 256 — safe now (per-chunk, not concat)
NLI_THRESHOLD_ENTAIL     = 0.6         # >= this → Grounded  (green)
NLI_THRESHOLD_NEUTRAL    = 0.4         # >= this → Inferred  (yellow)
                                        # below   → Hallucinated (red)

# ── Knowledge Graph ───────────────────────────────────────────────────────────
KG_OUTPUT_PATH  = "./data/knowledge_graph.html"

# ── Upload Dir ────────────────────────────────────────────────────────────────
UPLOAD_DIR      = "./data/uploads"

# ── Eval / Observability Logging (Recommendation #4 — NEW) ───────────────────
#
#  JSONL log of every query: retrieved chunk IDs + scores, reranker scores,
#  generated answer, per-claim NLI verdicts. Without this, fix ROI cannot be
#  measured. Append-only, safe to delete anytime.
#
EVAL_LOG_PATH   = "./data/eval_logs.jsonl"
EVAL_LOGGING_ENABLED = True
