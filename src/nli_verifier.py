# src/nli_verifier.py
"""
NLI Verifier — Hallucination detection.
Uses cross-encoder/nli-MiniLM2-L6-H768 via sentence-transformers CrossEncoder.

═══════════════════════════════════════════════════════════════════════════════
 AUDIT FIXES APPLIED:

 FIX #1 (Critical) — Per-chunk scoring instead of concatenated-then-truncated
   premise. OLD: joined all 5 chunks into one string, then sliced [:1000 chars]
   — discarding ~80% of retrieved evidence before scoring even ran. NEW: score
   the claim against EACH chunk individually, take the MAX entailment across
   chunks. This means a claim grounded in chunk #4 is correctly recognized,
   instead of being judged only against a truncated chunk #1.

 FIX #2 (High) — Preamble/meta-commentary filtering. OLD: naive sentence split
   treated LLM filler like "Here are the individual claims about X:" and bare
   list markers ("1.") as claims, scored them, and polluted the verdict counts.
   NEW: regex filters strip these before claim extraction.

 FIX #5 (Medium) — max_length raised 256 → 384. Safe now that we score one
   chunk at a time instead of a multi-chunk concatenation.
═══════════════════════════════════════════════════════════════════════════════
"""

import re
from typing import List, Dict
import numpy as np
from sentence_transformers.cross_encoder import CrossEncoder

from config import NLI_MODEL, NLI_MAX_LENGTH, NLI_THRESHOLD_ENTAIL, NLI_THRESHOLD_NEUTRAL

# Singleton — load model once per session
_nli_model = None


def get_nli_model() -> CrossEncoder:
    global _nli_model
    if _nli_model is None:
        _nli_model = CrossEncoder(
            NLI_MODEL,
            num_labels=3,
            max_length=NLI_MAX_LENGTH,
        )
    return _nli_model


# ── FIX #2: Preamble / meta-commentary filtering ─────────────────────────────
#
# Patterns that indicate the LLM is talking ABOUT its answer, not making a
# factual claim. These must never reach the NLI scorer.
#
PREAMBLE_PATTERNS = [
    r"^here\s+(are|is)\s+the\s+(individual\s+)?claims?\b",
    r"^the\s+following\s+(claims?|points?|answer)\b",
    r"^based\s+on\s+the\s+(provided|available)\s+(context|information|papers?)\s*[:.]?\s*$",
    r"^in\s+summary\s*[:,]?\s*$",
    r"^to\s+summarize\s*[:,]?\s*$",
    r"^\d+\.\s*$",                      # bare list markers: "1.", "2."
    r"^[-•]\s*$",                       # bare bullet markers
    r"^\W*$",                           # punctuation/whitespace only
]
_PREAMBLE_REGEX = re.compile("|".join(PREAMBLE_PATTERNS), re.IGNORECASE)


def is_preamble(sentence: str) -> bool:
    """Returns True if a sentence is meta-commentary, not a factual claim."""
    stripped = sentence.strip()
    if len(stripped) < 8:
        return True
    return bool(_PREAMBLE_REGEX.match(stripped))


def split_into_claims(answer: str) -> List[str]:
    """
    Splits LLM answer into individual factual claims.
    FIX #2: Strips preamble/meta-sentences and bare list markers BEFORE
    they can be scored as claims.
    """
    # Strip leading numbered-list markers like "1. " "2) " so the sentence
    # splitter doesn't treat the bare number as its own fragment.
    cleaned_answer = re.sub(r'(?m)^\s*\d+[\.\)]\s*', '', answer.strip())

    sentences = re.split(r'(?<=[.!?])\s+', cleaned_answer)
    claims = []
    for s in sentences:
        clean = re.sub(r'\[Source:.*?\]', '', s).strip()
        clean = re.sub(r'\[Chunk\s*\d+\]', '', clean).strip()

        if not clean or len(clean.split()) < 5:
            continue
        if is_preamble(clean):
            continue

        claims.append(clean)
    return claims


def softmax(scores: List[float]) -> List[float]:
    """Converts raw logits to probabilities."""
    arr = np.array(scores, dtype=np.float32)
    arr -= arr.max()
    e   = np.exp(arr)
    return (e / e.sum()).tolist()


def _verdict_from_scores(entail: float, neutral: float, contra: float) -> tuple:
    """Shared threshold logic — single source of truth for verdict labels."""
    if entail >= NLI_THRESHOLD_ENTAIL:
        return "grounded", "✅"
    elif entail >= NLI_THRESHOLD_NEUTRAL or neutral > contra:
        return "inferred", "🟡"
    else:
        return "hallucinated", "❌"


def verify_claim(claim: str, context_chunks: List[Dict]) -> Dict:
    """
    FIX #1: Verifies a claim against EACH context chunk INDIVIDUALLY
    (not one giant truncated concatenation), and reports the BEST match
    (highest entailment score) across all chunks.

    Label order from CrossEncoder: [contradiction=0, entailment=1, neutral=2]
    """
    model = get_nli_model()

    if not context_chunks:
        return {
            "claim": claim, "verdict": "hallucinated", "icon": "❌",
            "entail_score": 0.0, "neutral_score": 0.0, "contra_score": 1.0,
            "best_chunk_idx": None,
        }

    # Score against every chunk in a single batched call (fast on CPU for 5 pairs)
    pairs = [(chunk["text"], claim) for chunk in context_chunks]
    raw_scores = model.predict(pairs)

    best_entail   = -1.0
    best_idx      = 0
    best_probs    = (0.0, 0.0, 1.0)  # contra, entail, neutral

    for i, raw in enumerate(raw_scores):
        probs = softmax(raw.tolist())
        contra, entail, neutral = probs[0], probs[1], probs[2]
        if entail > best_entail:
            best_entail = entail
            best_idx    = i
            best_probs  = (contra, entail, neutral)

    contra_score, entail_score, neutral_score = best_probs
    verdict, icon = _verdict_from_scores(entail_score, neutral_score, contra_score)

    return {
        "claim":          claim,
        "verdict":        verdict,
        "icon":           icon,
        "entail_score":   round(entail_score, 3),
        "neutral_score":  round(neutral_score, 3),
        "contra_score":   round(contra_score, 3),
        "best_chunk_idx":  best_idx,   # which chunk best supported this claim
        "best_chunk_source": context_chunks[best_idx].get("source", ""),
    }


def verify_answer(answer: str, context_chunks: List[Dict]) -> List[Dict]:
    """
    Verifies all real factual claims in an LLM answer against retrieved
    context, scoring each claim against each chunk individually.
    Returns list of verified claim dicts.
    """
    claims   = split_into_claims(answer)
    verified = []
    for claim in claims:
        result = verify_claim(claim, context_chunks)
        verified.append(result)
    return verified
