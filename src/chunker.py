# src/chunker.py
"""
Semantic Chunker — Section-aware, sentence-boundary-snapped, TOKEN-AWARE chunking.

═══════════════════════════════════════════════════════════════════════════════
 AUDIT FIX #3 (High) APPLIED:

 OLD: `_tokenize()` used `text.split()` — a whitespace-word count, NOT real
   tokenization. CHUNK_SIZE=200 meant "200 whitespace words", but every
   downstream consumer (embedder, NLI verifier, reranker) uses SUBWORD
   tokenization. For academic text (hyphens, numbers, citations), the
   subword/word ratio is typically 1.3-1.6x, meaning many "200-word" chunks
   silently exceeded the embedder's 256-token limit and were truncated
   during encoding with zero warning or error anywhere in the pipeline.

 NEW: Uses the ACTUAL HuggingFace tokenizer from the embedding model to count
   real subword tokens. CHUNK_SIZE/OVERLAP in config.py are now expressed in
   real tokens, kept safely under EMBEDDING_MAX_SEQ_TOKENS (256). Chunk
   boundaries are also snapped to sentence ends (SNAP_TO_SENTENCE) so chunks
   never cut a sentence in half, which both improves retrieval coherence and
   gives the NLI verifier and reranker cleaner premises to score against.
═══════════════════════════════════════════════════════════════════════════════
"""

import re
from typing import List, Dict
from transformers import AutoTokenizer

from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_MIN_TOKENS,
    SNAP_TO_SENTENCE, EMBEDDING_MODEL,
)

# Singleton tokenizer — same tokenizer the embedding model actually uses,
# so "token count" here means the SAME thing it means downstream.
_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    return _tokenizer


def _split_sentences(text: str) -> List[str]:
    """Splits text into sentences, preserving the delimiter."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _count_tokens(text: str) -> int:
    """Real subword token count via the embedding model's own tokenizer."""
    tok = _get_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))


def _chunk_text_sentence_aware(
    text: str, chunk_size: int, overlap: int
) -> List[str]:
    """
    Token-aware chunking that snaps to sentence boundaries.
    Greedily fills each chunk with whole sentences up to `chunk_size` real
    subword tokens, then backs up `overlap` tokens worth of trailing
    sentences to start the next chunk — never splitting a sentence.
    """
    if not SNAP_TO_SENTENCE:
        return _chunk_text_raw_tokens(text, chunk_size, overlap)

    sentences = _split_sentences(text)
    if not sentences:
        return []

    # Pre-compute token count per sentence once
    sent_tokens = [(s, _count_tokens(s)) for s in sentences]

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0
    i = 0

    while i < len(sent_tokens):
        sent, n_tok = sent_tokens[i]

        # Single sentence longer than chunk_size on its own — emit it alone
        if n_tok > chunk_size and not current:
            chunks.append(sent)
            i += 1
            continue

        if current_tokens + n_tok > chunk_size and current:
            chunks.append(" ".join(current))

            # Build overlap: walk backwards from end of current chunk,
            # collecting whole sentences until we've covered `overlap` tokens
            overlap_sents: List[str] = []
            overlap_tok = 0
            for s, t in reversed(list(zip(current, [_count_tokens(c) for c in current]))):
                if overlap_tok >= overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_tok += t

            current = overlap_sents
            current_tokens = overlap_tok
            continue  # re-process current sentence i against the reset window

        current.append(sent)
        current_tokens += n_tok
        i += 1

    if current:
        chunks.append(" ".join(current))

    return chunks


def _chunk_text_raw_tokens(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Fallback: pure token-window chunking (no sentence snapping).
    Used only if SNAP_TO_SENTENCE=False. Operates on the tokenizer's own
    token IDs and decodes back to text, so boundaries are always real
    subword-token boundaries — never silently truncated.
    """
    tok = _get_tokenizer()
    ids = tok.encode(text, add_special_tokens=False)

    chunks = []
    start = 0
    while start < len(ids):
        end = min(start + chunk_size, len(ids))
        chunk_ids = ids[start:end]
        chunk_text = tok.decode(chunk_ids, skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
        if end == len(ids):
            break
        start += chunk_size - overlap

    return chunks


def chunk_sections(
    sections: Dict[str, str],
    source_name: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Takes a {section_name: section_text} dict and produces token-aware,
    sentence-boundary-snapped chunks with metadata.

    Returns:
        List of {chunk_id, text, source, section, token_count}
    """
    all_chunks = []
    chunk_id   = 0

    for section_name, section_text in sections.items():
        if not section_text.strip():
            continue

        text_chunks = _chunk_text_sentence_aware(section_text, chunk_size, overlap)

        for chunk_text in text_chunks:
            n_tok = _count_tokens(chunk_text)
            if n_tok < CHUNK_MIN_TOKENS:
                continue
            all_chunks.append({
                "chunk_id":    f"{source_name}_{chunk_id}",
                "text":        chunk_text,
                "source":      source_name,
                "section":     section_name,
                "token_count": n_tok,
            })
            chunk_id += 1

    return all_chunks


def chunk_full_text(
    full_text: str,
    source_name: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Fallback: chunk entire text when section detection fails.
    Same token-aware, sentence-snapped logic as chunk_sections().
    """
    raw_chunks = _chunk_text_sentence_aware(full_text, chunk_size, overlap)
    result = []
    for i, c in enumerate(raw_chunks):
        n_tok = _count_tokens(c)
        if n_tok < CHUNK_MIN_TOKENS:
            continue
        result.append({
            "chunk_id":    f"{source_name}_{i}",
            "text":        c,
            "source":      source_name,
            "section":     "full_document",
            "token_count": n_tok,
        })
    return result
