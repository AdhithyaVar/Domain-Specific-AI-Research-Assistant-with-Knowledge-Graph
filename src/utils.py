# src/utils.py
"""
Utility functions — file handling, session helpers, display helpers.
"""

import os
import hashlib
from typing import List


def get_file_hash(file_path: str) -> str:
    """Returns MD5 hash of a file (used to detect duplicate uploads)."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def save_uploaded_file(uploaded_file, upload_dir: str) -> str:
    """
    Saves a Streamlit UploadedFile to disk.
    Resets file pointer before reading to handle re-uploads.
    Returns the saved file path.
    """
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    uploaded_file.seek(0)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    return file_path


def list_uploaded_pdfs(upload_dir: str) -> List[str]:
    """Returns list of PDF file paths in the upload directory."""
    if not os.path.exists(upload_dir):
        return []
    return sorted([
        os.path.join(upload_dir, f)
        for f in os.listdir(upload_dir)
        if f.lower().endswith(".pdf")
    ])


def format_verified_answer(verified_claims: list) -> str:
    """
    Formats verified claims into a readable markdown string with badges.
    """
    badge = {
        "grounded":     "🟢 **Grounded**",
        "inferred":     "🟡 **Inferred**",
        "hallucinated": "🔴 **Hallucinated**",
    }
    lines = []
    for item in verified_claims:
        verdict_badge = badge.get(item["verdict"], "⚪ **Unknown**")
        lines.append(f"{verdict_badge} — {item['claim']}")
    return "\n\n".join(lines)


def ollama_is_running() -> bool:
    """
    Checks if the Ollama server is accessible at localhost:11434.
    Returns True if running, False otherwise.
    """
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://localhost:11434",
            method="GET",
            headers={"User-Agent": "research-assistant-healthcheck"},
        )
        urllib.request.urlopen(req, timeout=2)
        return True
    except Exception:
        return False


def truncate_text(text: str, max_chars: int = 400) -> str:
    """Truncates text to max_chars with ellipsis."""
    return text[:max_chars] + "…" if len(text) > max_chars else text
