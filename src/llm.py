# src/llm.py
"""
LLM Module — Ollama-based answer generation.
Auto-detects available Ollama models if configured model is not found.
"""

import subprocess
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Dict, Optional

from config import (
    OLLAMA_MODEL, OLLAMA_BASE_URL,
    LLM_TEMPERATURE, LLM_MAX_TOKENS,
    OLLAMA_FALLBACK_MODELS,
)


SYSTEM_PROMPT = """You are a precise academic research assistant.
Your job is to answer questions strictly based on the provided research paper excerpts.

Rules:
1. Answer ONLY from the provided context chunks.
2. Break your answer into clear, individual claims — one per sentence.
3. After each claim, cite the source like: [Source: filename, Section: section_name]
4. If the context does not contain enough information, say: "The provided papers do not contain sufficient information on this topic."
5. Do NOT hallucinate or add external knowledge.
6. Be concise, accurate, and technical.
"""


def get_available_models() -> List[str]:
    """Returns list of models currently pulled in Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return []
        lines  = result.stdout.strip().split("\n")[1:]  # skip header
        models = [l.split()[0] for l in lines if l.strip()]
        return models
    except Exception:
        return []


def resolve_model() -> Optional[str]:
    """
    Returns the best available model from Ollama.
    Priority: configured model → fallbacks → first available model.
    Returns None if no models are available.
    """
    available = get_available_models()
    if not available:
        return None

    # Check configured model (partial match — e.g. "phi3.5:mini" matches "phi3.5:mini")
    for m in available:
        if OLLAMA_MODEL.lower() in m.lower() or m.lower() in OLLAMA_MODEL.lower():
            return m

    # Try fallbacks
    for fallback in OLLAMA_FALLBACK_MODELS:
        for m in available:
            if fallback.lower() in m.lower() or m.lower() in fallback.lower():
                return m

    # Last resort — use whatever is first available
    return available[0]


def build_context(chunks: List[Dict]) -> str:
    """Formats retrieved chunks into a numbered context block."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Chunk {i}] Source: {chunk['source']} | Section: {chunk['section']}\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, chunks: List[Dict]) -> str:
    """
    Generates a cited answer using the best available Ollama model.
    Auto-detects and falls back gracefully if configured model not found.
    """
    if not chunks:
        return "No relevant context found. Please upload relevant PDFs first."

    model = resolve_model()

    if model is None:
        available = get_available_models()
        if not available:
            return (
                "❌ No Ollama models found.\n\n"
                "Run this command to pull a model:\n"
                "```\nollama pull phi3.5:mini\n```\n"
                "Or any other model: `ollama pull llama3.2:3b`"
            )

    llm = ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=LLM_TEMPERATURE,
        num_predict=LLM_MAX_TOKENS,
    )

    context      = build_context(chunks)
    user_message = (
        f"Context from research papers:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer with individual claims, each followed by its citation:"
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        err = str(e)
        # Provide helpful message for 404 model-not-found error
        if "404" in err or "not found" in err.lower():
            available = get_available_models()
            avail_str = "\n".join(f"  • {m}" for m in available) if available else "  (none)"
            return (
                f"❌ Model `{model}` not found in Ollama.\n\n"
                f"**Models you have pulled:**\n{avail_str}\n\n"
                f"**To pull the recommended model, run:**\n"
                f"```\nollama pull phi3.5:mini\n```\n\n"
                f"Or update `config.py` → `OLLAMA_MODEL` to one of your available models."
            )
        return f"LLM Error: {err}\n\nMake sure Ollama is running: `ollama serve`"
