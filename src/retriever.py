# src/retriever.py
"""
Hybrid Retriever — ChromaDB (dense) + BM25 (sparse) fused via RRF,
then RERANKED by a cross-encoder, then DEDUPLICATED before returning.

═══════════════════════════════════════════════════════════════════════════════
 AUDIT FIXES APPLIED:

 FIX #4 (High) — search() now reranks the RRF survivors with a cross-encoder
   (src/reranker.py) before returning the final top-k. Previously RRF rank
   was treated as the final answer with no relevance-correction step.

 FIX #6 (Medium) — Near-duplicate chunks (e.g. heavily overlapping windows
   from adjacent chunk boundaries) are now collapsed via embedding cosine
   similarity BEFORE being sent to the LLM, freeing context budget that was
   previously wasted on redundant text.

 Also: TOP_K_RRF (candidates surviving fusion) is now distinct from
   TOP_K_FINAL (candidates surviving rerank) — RRF over-fetches slightly so
   the reranker has real signal to discriminate against, instead of RRF
   already having narrowed to exactly 5 with no room for correction.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL, EMBEDDING_DEVICE,
    CHROMA_PERSIST_DIR, CHROMA_COLLECTION,
    TOP_K_DENSE, TOP_K_SPARSE, TOP_K_RRF, TOP_K_FINAL,
    RERANKER_ENABLED, DEDUP_SIMILARITY_THRESHOLD,
)
from src.reranker import rerank as cross_encoder_rerank


class HybridRetriever:
    def __init__(self):
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

        # Dense: SentenceTransformer + ChromaDB
        self.embedder = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        # Sparse: BM25 (rebuilt from ChromaDB on demand)
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_chunks: List[Dict]   = []

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index_chunks(self, chunks: List[Dict]) -> None:
        """Index new chunks into ChromaDB and rebuild BM25. Skips duplicates."""
        existing_ids = set(self.collection.get()["ids"])
        new_chunks   = [c for c in chunks if c["chunk_id"] not in existing_ids]

        if not new_chunks:
            return

        texts      = [c["text"]    for c in new_chunks]
        ids        = [c["chunk_id"] for c in new_chunks]
        metadatas  = [{"source": c["source"], "section": c["section"]} for c in new_chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 index from all docs stored in ChromaDB."""
        result = self.collection.get(include=["documents", "metadatas"])
        self._bm25_chunks = [
            {
                "chunk_id": cid,
                "text":     text,
                "source":   meta.get("source", ""),
                "section":  meta.get("section", ""),
            }
            for cid, text, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        ]
        tokenized    = [c["text"].lower().split() for c in self._bm25_chunks]
        self._bm25   = BM25Okapi(tokenized) if tokenized else None

    # ── Search stages ────────────────────────────────────────────────────────

    def dense_search(self, query: str, top_k: int = TOP_K_DENSE) -> List[Tuple[str, float]]:
        """Semantic search via ChromaDB cosine similarity."""
        query_emb = self.embedder.encode([query]).tolist()
        n         = min(top_k, self.collection.count())
        if n == 0:
            return []
        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=n,
            include=["distances"],
        )
        ids   = results["ids"][0]
        dists = results["distances"][0]
        return [(cid, 1.0 - d) for cid, d in zip(ids, dists)]

    def sparse_search(self, query: str, top_k: int = TOP_K_SPARSE) -> List[Tuple[str, float]]:
        """Keyword search via BM25."""
        if self._bm25 is None:
            self._rebuild_bm25()
        if self._bm25 is None:
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            (self._bm25_chunks[i]["chunk_id"], score)
            for i, score in ranked
            if score > 0
        ]

    def rrf_fusion(
        self,
        dense:  List[Tuple[str, float]],
        sparse: List[Tuple[str, float]],
        k: int = 60,
        top_n: int = TOP_K_RRF,
    ) -> List[str]:
        """
        Reciprocal Rank Fusion. k=60 is the standard constant.
        Returns top_n survivors — intentionally MORE than TOP_K_FINAL so the
        reranker (FIX #4) has real candidates to discriminate between,
        instead of RRF already pre-deciding the final 5 with no correction.
        """
        scores: Dict[str, float] = {}
        for rank, (cid, _) in enumerate(dense):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        for rank, (cid, _) in enumerate(sparse):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores, key=lambda x: scores[x], reverse=True)[:top_n]

    # ── FIX #6: Near-duplicate deduplication ─────────────────────────────────

    def _deduplicate(self, chunks: List[Dict]) -> List[Dict]:
        """
        Collapses near-identical chunks (e.g. heavily overlapping windows from
        adjacent chunk boundaries of the same section) using embedding cosine
        similarity, BEFORE the LLM ever sees them. Keeps the higher-ranked
        chunk of each duplicate pair (input order = rank order).
        """
        if len(chunks) <= 1:
            return chunks

        texts = [c["text"] for c in chunks]
        embs  = self.embedder.encode(texts, show_progress_bar=False)
        norms = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
        sim_matrix = norms @ norms.T

        keep = []
        dropped = set()
        for i in range(len(chunks)):
            if i in dropped:
                continue
            keep.append(chunks[i])
            for j in range(i + 1, len(chunks)):
                if sim_matrix[i, j] >= DEDUP_SIMILARITY_THRESHOLD:
                    dropped.add(j)

        return keep

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def search(self, query: str) -> List[Dict]:
        """
        Full hybrid search pipeline:
          Dense + Sparse → RRF fusion (over-fetch TOP_K_RRF)
          → Cross-encoder rerank (FIX #4) → top TOP_K_FINAL
          → Near-duplicate removal (FIX #6)
        """
        if self.collection.count() == 0:
            return []

        dense   = self.dense_search(query)
        sparse  = self.sparse_search(query)
        top_ids = self.rrf_fusion(dense, sparse)

        if not top_ids:
            return []

        result = self.collection.get(
            ids=top_ids,
            include=["documents", "metadatas"],
        )
        candidates = [
            {
                "chunk_id": cid,
                "text":     text,
                "source":   meta.get("source", ""),
                "section":  meta.get("section", ""),
            }
            for cid, text, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        ]

        # FIX #4 — cross-encoder relevance rerank (replaces "RRF rank = final")
        if RERANKER_ENABLED:
            final = cross_encoder_rerank(query, candidates, top_k=TOP_K_FINAL)
        else:
            final = candidates[:TOP_K_FINAL]

        # FIX #6 — drop near-duplicate chunks before they reach the LLM
        final = self._deduplicate(final)

        return final

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_indexed_sources(self) -> List[str]:
        """Returns sorted list of unique source names."""
        if self.collection.count() == 0:
            return []
        result  = self.collection.get(include=["metadatas"])
        sources = list({m.get("source", "") for m in result["metadatas"]})
        return sorted(sources)

    def clear_index(self) -> None:
        """Wipes ChromaDB collection and BM25 index."""
        self.chroma_client.delete_collection(CHROMA_COLLECTION)
        self.collection = self.chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25        = None
        self._bm25_chunks = []
