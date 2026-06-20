# verify_setup.py — Run before launching the app
"""
Checks all dependencies and Ollama status.
Usage: python verify_setup.py
"""

import sys
import importlib
import subprocess
import urllib.request

REQUIRED = [
    ("fitz",                    "pymupdf"),
    ("streamlit",               "streamlit"),
    ("chromadb",                "chromadb"),
    ("sentence_transformers",   "sentence-transformers"),
    ("rank_bm25",               "rank-bm25"),
    ("langchain",               "langchain"),
    ("langchain_core",          "langchain-core"),        # ← replaces langchain.schema
    ("langchain_ollama",        "langchain-ollama"),
    ("langchain_community",     "langchain-community"),
    ("ollama",                  "ollama"),
    ("transformers",            "transformers"),
    ("torch",                   "torch"),
    ("networkx",                "networkx"),
    ("pyvis",                   "pyvis"),
    ("numpy",                   "numpy"),
    ("pandas",                  "pandas"),
]

print("=" * 58)
print("  AI Research Assistant — Setup Verification")
print("=" * 58)

# ── Python version check ──────────────────────────────────────
major, minor = sys.version_info[:2]
py_ver = f"Python {major}.{minor}"
if major == 3 and 10 <= minor <= 12:
    print(f"  ✅  {py_ver:<35} supported")
elif major == 3 and minor >= 13:
    print(f"  ⚠️   {py_ver:<35} may have Pydantic v1 issues")
    print(f"        Recommended: Python 3.10 or 3.11")
else:
    print(f"  ❌  {py_ver:<35} too old — use Python 3.10+")

print()

# ── Package checks ────────────────────────────────────────────
all_ok = True
for import_name, pip_name in REQUIRED:
    try:
        importlib.import_module(import_name)
        print(f"  ✅  {pip_name:<35} installed")
    except ImportError:
        print(f"  ❌  {pip_name:<35} MISSING")
        print(f"        → pip install {pip_name}")
        all_ok = False

print()

# ── langchain_core.messages import check ─────────────────────
try:
    from langchain_core.messages import HumanMessage, SystemMessage
    print(f"  ✅  langchain_core.messages             OK")
except ImportError:
    print(f"  ❌  langchain_core.messages             FAILED")
    print(f"        → pip install langchain-core --upgrade")
    all_ok = False

print()

# ── Ollama server check ───────────────────────────────────────
try:
    urllib.request.urlopen("http://localhost:11434", timeout=2)
    print(f"  ✅  Ollama server                       running")
except Exception:
    print(f"  ❌  Ollama server                       OFFLINE")
    print(f"        → run: ollama serve")
    all_ok = False

print()

# ── Ollama model check ────────────────────────────────────────
try:
    result = subprocess.run(
        ["ollama", "list"], capture_output=True, text=True, timeout=5
    )
    if result.returncode == 0:
        lines = [l.strip() for l in result.stdout.strip().split("\n")[1:] if l.strip()]
        if lines:
            print(f"  ✅  Ollama models available:")
            for line in lines:
                model_name = line.split()[0]
                print(f"        • {model_name}")
            # Check for recommended model
            model_names = [l.split()[0] for l in lines]
            if any("phi3.5" in m or "phi3" in m for m in model_names):
                print(f"  ✅  phi3.5:mini detected — recommended model ready")
            else:
                print(f"  ⚠️   phi3.5:mini not found")
                print(f"        → run: ollama pull phi3.5:mini")
        else:
            print(f"  ⚠️   No Ollama models pulled yet")
            print(f"        → run: ollama pull phi3.5:mini")
    else:
        print(f"  ⚠️   Could not list models")
except Exception as e:
    print(f"  ⚠️   Ollama CLI error: {e}")

print()
print("=" * 58)
if all_ok:
    print("  ✅  All checks passed!")
    print("  →   Run: streamlit run app.py")
else:
    print("  ❌  Fix the issues above, then re-run this script.")
print("=" * 58)
