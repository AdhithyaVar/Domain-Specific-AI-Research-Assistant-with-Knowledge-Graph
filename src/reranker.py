# src/reranker.py
"""
Reranker — Cross-encoder relevance reranking after RRF fusion.

═══════════════════════════════════════════════════════════════════════════════
 AUDIT FIX #4 (High) APPLIED:

 FINDING: The project's own documentation/pitch described a
   `cross-encoder/ms-marco` reranking stage, but it did NOT exist anywhere in
   retriever.py — RRF fusion (a rank-merging heuristic) was being treated as
   the final relevance signal, with zero correction step afterward. This was
   most visible on multi-entity comparison queries ("compare YOLO vs YOLOv8"),
   where RRF could promote a chunk that merely mentions both terms in passing
   over one that actually discusses the comparison.

 FIX: Added this dedicated reranker module using
   `cross-encoder/ms-marco-MiniLM-L-6-v2` (67MB, CPU-friendly, purpose-built
   for query-passage relevance on MS MARCO). It re-scores the TOP_K_RRF
   candidates that survive RRF fusion and returns only the TOP_K_FINAL best,
   now ordered by genuine query-relevance rather than fusion-rank alone.
═══════════════════════════════════════════════════════════════════════════════
"""

from typing import List, Dict
from sentence_transformers.cross_encoder import CrossEncoder

from config import RERANKER_MODEL, TOP_K_FINAL

_reranker_model = None


def get_reranker_model() -> CrossEncoder:
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(
            RERANKER_MODEL,
            max_length=384,
        )
    return _reranker_model


def rerank(query: str, chunks: List[Dict], top_k: int = TOP_K_FINAL) -> List[Dict]:
    """
    Re-scores `chunks` (already fused by RRF) against the raw query using a
    cross-encoder trained specifically for relevance ranking, and returns the
    top_k best, each annotated with `rerank_score`.

    This is a STRICTLY STRONGER relevance signal than RRF rank alone, because
    the cross-encoder jointly attends over (query, passage) — RRF only ever
    saw separate dense/sparse scores, never the literal query-passage pair.
    """
    if not chunks:
        return []
    if len(chunks) <= top_k:
        # Still score for transparency / debug visibility, but nothing to trim
        pass

    model = get_reranker_model()
    pairs = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs)

    scored = [
        {**chunk, "rerank_score": round(float(score), 4)}
        for chunk, score in zip(chunks, scores)
    ]
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)

    return scored[:top_k]
