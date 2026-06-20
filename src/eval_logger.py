# src/eval_logger.py
"""
Eval Logger — Append-only JSONL logging of every query's retrieval + NLI
verification trail, for offline measurement of RAG quality.

═══════════════════════════════════════════════════════════════════════════════
 AUDIT RECOMMENDATION #4 APPLIED:

 The original audit explicitly flagged that NO eval harness, ground-truth
 labels, or retrieval logs existed anywhere in the repo — meaning there was
 no way to measure whether any fix (reranker, NLI per-chunk scoring, chunking)
 actually improved anything. This module closes that gap with zero external
 dependencies (pure stdlib json + file I/O).

 Each line in the JSONL log captures, per query:
   - retrieved chunk IDs + sources + rerank scores
   - the generated answer
   - per-claim NLI verdicts (grounded/inferred/hallucinated) with raw scores
   - aggregate grounded/inferred/hallucinated counts

 This makes it possible to hand-label a 20-30 question eval set, re-run after
 a future change, and diff the JSONL outputs to get an actual before/after
 number — rather than relying on visual inspection of the Streamlit UI.
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import os
import time
from typing import List, Dict, Optional

from config import EVAL_LOG_PATH, EVAL_LOGGING_ENABLED


def log_query(
    query: str,
    chunks: List[Dict],
    answer: str,
    verified: List[Dict],
    model_used: Optional[str] = None,
) -> None:
    """
    Appends one JSON line capturing the full retrieval → generation →
    verification trail for a single query. Safe no-op if logging disabled
    or if writing fails for any reason (never breaks the user-facing flow).
    """
    if not EVAL_LOGGING_ENABLED:
        return

    try:
        grounded     = sum(1 for v in verified if v["verdict"] == "grounded")
        inferred     = sum(1 for v in verified if v["verdict"] == "inferred")
        hallucinated = sum(1 for v in verified if v["verdict"] == "hallucinated")

        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "query": query,
            "model_used": model_used,
            "retrieved_chunks": [
                {
                    "chunk_id": c.get("chunk_id"),
                    "source":   c.get("source"),
                    "section":  c.get("section"),
                    "rerank_score": c.get("rerank_score"),
                }
                for c in chunks
            ],
            "answer": answer,
            "claims": [
                {
                    "claim":         v["claim"],
                    "verdict":       v["verdict"],
                    "entail_score":  v["entail_score"],
                    "neutral_score": v["neutral_score"],
                    "contra_score":  v["contra_score"],
                    "best_chunk_source": v.get("best_chunk_source"),
                }
                for v in verified
            ],
            "summary": {
                "total_claims": len(verified),
                "grounded":     grounded,
                "inferred":     inferred,
                "hallucinated": hallucinated,
                "grounded_rate": round(grounded / len(verified), 3) if verified else None,
            },
        }

        parent = os.path.dirname(EVAL_LOG_PATH)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(EVAL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    except Exception:
        # Logging must never break the user-facing Q&A flow
        pass


def load_logs() -> List[Dict]:
    """Reads all logged records back in. Used by the Debug tab eval summary."""
    if not os.path.exists(EVAL_LOG_PATH):
        return []
    records = []
    with open(EVAL_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def aggregate_stats() -> Dict:
    """Computes overall grounded/inferred/hallucinated rates across all logged queries."""
    records = load_logs()
    if not records:
        return {"total_queries": 0}

    total_claims = sum(r["summary"]["total_claims"] for r in records)
    grounded     = sum(r["summary"]["grounded"] for r in records)
    inferred     = sum(r["summary"]["inferred"] for r in records)
    hallucinated = sum(r["summary"]["hallucinated"] for r in records)

    return {
        "total_queries":      len(records),
        "total_claims":       total_claims,
        "grounded":           grounded,
        "inferred":           inferred,
        "hallucinated":       hallucinated,
        "grounded_rate":      round(grounded / total_claims, 3) if total_claims else None,
        "hallucinated_rate":  round(hallucinated / total_claims, 3) if total_claims else None,
    }
