# app.py — Main Streamlit Application
"""
Domain-Specific AI Research Assistant with Knowledge Graph
100% Free · Fully Local · No API Keys Required
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore", message=".*Pydantic V1.*")
warnings.filterwarnings("ignore", message=".*pydantic.v1.*")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core.*")
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from config import UPLOAD_DIR, KG_OUTPUT_PATH, OLLAMA_MODEL
from src.pdf_processor   import extract_text_by_section, extract_full_text, get_pdf_metadata
from src.chunker         import chunk_sections, chunk_full_text
from src.retriever       import HybridRetriever
from src.llm             import generate_answer, get_available_models, resolve_model
from src.nli_verifier    import verify_answer
from src.knowledge_graph import KnowledgeGraph
from src.eval_logger      import log_query, aggregate_stats
from src.utils           import (
    save_uploaded_file, list_uploaded_pdfs,
    format_verified_answer, ollama_is_running, get_file_hash
)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.verdict-grounded    { background:#1a472a; color:#69db7c; padding:8px 14px;
                       border-radius:8px; margin:4px 0; font-size:15px; }
.verdict-inferred    { background:#5c4a00; color:#fcc419; padding:8px 14px;
                       border-radius:8px; margin:4px 0; font-size:15px; }
.verdict-hallucinated{ background:#6b1111; color:#ff6b6b; padding:8px 14px;
                       border-radius:8px; margin:4px 0; font-size:15px; }
.section-header      { border-left:4px solid #6c5ce7; padding-left:12px;
                       margin:16px 0 8px; }
.chunk-box           { background:#1e1e2e; border-radius:8px; padding:10px 14px;
                       margin:6px 0; font-size:13px; color:#cdd6f4; }
.model-ok            { background:#1a3a1a; color:#69db7c; padding:6px 10px;
                       border-radius:6px; font-size:13px; margin:4px 0; }
.model-missing       { background:#3a1a1a; color:#ff6b6b; padding:6px 10px;
                       border-radius:6px; font-size:13px; margin:4px 0; }
</style>
""", unsafe_allow_html=True)

# ── Session State Init ─────────────────────────────────────────────────────────
if "retriever"    not in st.session_state:
    st.session_state.retriever    = HybridRetriever()
if "kg"           not in st.session_state:
    st.session_state.kg           = KnowledgeGraph()
if "indexed"      not in st.session_state:
    st.session_state.indexed      = set()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

retriever: HybridRetriever = st.session_state.retriever
kg:        KnowledgeGraph  = st.session_state.kg


# ── Helper: display verified claims ───────────────────────────────────────────
def display_verified(verified: list):
    for item in verified:
        v         = item["verdict"]
        css_class = f"verdict-{v}"
        icon      = item["icon"]
        st.markdown(
            f'<div class="{css_class}">{icon} {item["claim"]}</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Entailment: {item['entail_score']:.2f}  |  "
            f"Neutral: {item['neutral_score']:.2f}  |  "
            f"Contradiction: {item['contra_score']:.2f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🔬 Research Assistant")
    st.caption("Hybrid RAG · NLI Verification · Knowledge Graph")
    st.divider()

    # ── Ollama + Model Status ──────────────────────────────────
    if ollama_is_running():
        st.success("✅ Ollama is running")

        available_models = get_available_models()
        active_model     = resolve_model()

        if active_model:
            st.markdown(
                f'<div class="model-ok">🤖 Model ready: <b>{active_model}</b></div>',
                unsafe_allow_html=True,
            )
        else:
            # No models pulled at all
            st.markdown(
                f'<div class="model-missing">⚠️ No models pulled yet</div>',
                unsafe_allow_html=True,
            )
            st.warning(f"Configured model `{OLLAMA_MODEL}` not found.")
            st.markdown("**Pull it now — run in terminal:**")
            st.code(f"ollama pull {OLLAMA_MODEL}", language="bash")
            st.caption("Or choose a smaller alternative:")
            st.code("ollama pull llama3.2:3b", language="bash")

        # Show all pulled models
        if available_models:
            with st.expander("📋 All pulled models"):
                for m in available_models:
                    icon = "✅" if m == active_model else "  "
                    st.caption(f"{icon} {m}")
    else:
        st.error("❌ Ollama offline")
        st.code("ollama serve", language="bash")

    st.divider()

    # ── PDF Upload ─────────────────────────────────────────────
    st.markdown("### 📄 Upload Research Papers")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for uf in uploaded_files:
            saved_path = save_uploaded_file(uf, UPLOAD_DIR)
            file_hash  = get_file_hash(saved_path)

            if file_hash not in st.session_state.indexed:
                with st.spinner(f"Indexing {uf.name}…"):
                    try:
                        sections   = extract_text_by_section(saved_path)
                        total_text = sum(len(v) for v in sections.values())

                        if not sections or total_text < 300:
                            full_text = extract_full_text(saved_path)
                            chunks    = chunk_full_text(full_text, uf.name)
                        else:
                            chunks = chunk_sections(sections, uf.name)

                        retriever.index_chunks(chunks)
                        kg.add_paper(chunks)
                        st.session_state.indexed.add(file_hash)
                        st.success(f"✅ {uf.name} — {len(chunks)} chunks")
                    except Exception as e:
                        st.error(f"❌ Failed: {uf.name}\n{e}")

    st.divider()

    # ── Indexed Sources ────────────────────────────────────────
    sources = retriever.get_indexed_sources()
    if sources:
        st.markdown("### 📚 Indexed Papers")
        for s in sources:
            st.markdown(f"- `{s}`")
    else:
        st.info("No papers indexed yet.\nUpload PDFs above.")

    st.divider()

    # ── Graph Stats ────────────────────────────────────────────
    if kg.graph.number_of_nodes() > 0:
        summary = kg.get_summary()
        st.markdown("### 🕸️ Graph Stats")
        c1, c2  = st.columns(2)
        c1.metric("Nodes", summary["total_nodes"])
        c2.metric("Edges", summary["total_edges"])

    st.divider()

    if st.button("🗑️ Clear All Data", use_container_width=True):
        retriever.clear_index()
        kg.clear()
        st.session_state.indexed      = set()
        st.session_state.chat_history = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_qa, tab_kg, tab_compare, tab_debug = st.tabs([
    "💬 Q&A",
    "🕸️ Knowledge Graph",
    "🔍 Compare Models",
    "🛠️ Debug / Settings",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Q&A
# ─────────────────────────────────────────────────────────────────────────────
with tab_qa:
    st.markdown('<div class="section-header"><h3>Ask a Question</h3></div>',
                unsafe_allow_html=True)

    # Show model-not-ready banner at top of Q&A tab
    active_model = resolve_model() if ollama_is_running() else None
    if not ollama_is_running():
        st.error("❌ Ollama is not running. Start it with: `ollama serve`")
    elif active_model is None:
        st.error(
            f"❌ No Ollama model found. Pull one in your terminal:\n\n"
            f"```\nollama pull {OLLAMA_MODEL}\n```"
        )
    elif not sources:
        st.warning("⬅️ Upload at least one PDF in the sidebar to start asking questions.")
    else:
        st.info(f"🤖 Using model: **{active_model}**", icon="ℹ️")

    if sources:
        for entry in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(entry["question"])
            with st.chat_message("assistant"):
                display_verified(entry["verified"])
                st.caption(
                    f"🟢 Grounded: {entry['stats']['grounded']}  |  "
                    f"🟡 Inferred: {entry['stats']['inferred']}  |  "
                    f"🔴 Hallucinated: {entry['stats']['hallucinated']}"
                )

        query = st.chat_input("Ask anything about the uploaded research papers…")

        if query:
            with st.chat_message("user"):
                st.write(query)

            with st.chat_message("assistant"):
                with st.spinner("🔍 Retrieving relevant context…"):
                    chunks = retriever.search(query)

                if not chunks:
                    st.warning("No relevant context found. Try rephrasing your question.")
                else:
                    with st.spinner("🤖 Generating answer…"):
                        answer = generate_answer(query, chunks)

                    with st.spinner("🔬 Verifying claims for hallucinations…"):
                        verified = verify_answer(answer, chunks)

                    display_verified(verified)

                    grounded     = sum(1 for v in verified if v["verdict"] == "grounded")
                    inferred     = sum(1 for v in verified if v["verdict"] == "inferred")
                    hallucinated = sum(1 for v in verified if v["verdict"] == "hallucinated")

                    st.caption(
                        f"🟢 Grounded: {grounded}  |  "
                        f"🟡 Inferred: {inferred}  |  "
                        f"🔴 Hallucinated: {hallucinated}  |  "
                        f"Total claims: {len(verified)}"
                    )
                    st.caption(
                        "🟢 = supported  |  🟡 = partially supported  |  🔴 = not found in source"
                    )

                    with st.expander(f"📎 View {len(chunks)} Source Chunks Used"):
                        for i, c in enumerate(chunks, 1):
                            rerank_badge = (
                                f' | <span style="color:#888">rerank: {c["rerank_score"]:.3f}</span>'
                                if "rerank_score" in c else ""
                            )
                            st.markdown(
                                f'<div class="chunk-box">'
                                f'<b>Chunk {i}</b> — '
                                f'<code>{c["source"]}</code> | '
                                f'<i>{c["section"]}</i>{rerank_badge}<br><br>'
                                f'{c["text"][:400]}{"…" if len(c["text"]) > 400 else ""}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                    # Recommendation #4 — log this turn for offline eval measurement
                    log_query(
                        query=query,
                        chunks=chunks,
                        answer=answer,
                        verified=verified,
                        model_used=resolve_model(),
                    )

                    st.session_state.chat_history.append({
                        "question": query,
                        "verified": verified,
                        "stats": {
                            "grounded":     grounded,
                            "inferred":     inferred,
                            "hallucinated": hallucinated,
                        },
                    })

        if st.session_state.chat_history:
            if st.button("🧹 Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Knowledge Graph
# ─────────────────────────────────────────────────────────────────────────────
with tab_kg:
    st.markdown('<div class="section-header"><h3>Multi-Paper Knowledge Graph</h3></div>',
                unsafe_allow_html=True)

    if kg.graph.number_of_nodes() == 0:
        st.info("Upload papers in the sidebar to build the knowledge graph.")
    else:
        summary = kg.get_summary()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📄 Papers",   len(summary["papers"]))
        c2.metric("🤖 Models",   len(summary["models"]))
        c3.metric("🗃️ Datasets", len(summary["datasets"]))
        c4.metric("📊 Metrics",  len(summary["metrics"]))

        st.divider()

        with st.spinner("Rendering knowledge graph…"):
            graph_path = kg.render_html(KG_OUTPUT_PATH)
            with open(graph_path, "r", encoding="utf-8") as f:
                html_content = f.read()

        st.components.v1.html(html_content, height=640, scrolling=False)

        st.divider()
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### 🤖 Detected Models")
            if summary["models"]:
                for m in sorted(summary["models"]):
                    st.markdown(f"- `{m}`")
            else:
                st.caption("None detected")

        with col_right:
            st.markdown("#### 🗃️ Detected Datasets")
            if summary["datasets"]:
                for d in sorted(summary["datasets"]):
                    st.markdown(f"- `{d}`")
            else:
                st.caption("None detected")

        st.divider()
        st.markdown("#### 🔎 Find Papers by Entity Name")
        entity_query = st.text_input(
            "Entity name", placeholder="e.g. BERT, ImageNet, accuracy…"
        )
        if entity_query:
            papers = kg.query_entity(entity_query)
            if papers:
                st.success(f"Found **{len(papers)}** paper(s) mentioning `{entity_query}`:")
                for p in papers:
                    st.markdown(f"- 📄 `{p}`")
            else:
                st.warning(f"No papers found mentioning `{entity_query}`.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Model Comparison
# ─────────────────────────────────────────────────────────────────────────────
with tab_compare:
    st.markdown('<div class="section-header"><h3>Compare Models Across Papers</h3></div>',
                unsafe_allow_html=True)

    if kg.graph.number_of_nodes() == 0:
        st.info("Upload papers to enable model comparison.")
    else:
        summary = kg.get_summary()
        models  = sorted(summary.get("models", []))

        if len(models) < 2:
            st.warning("At least 2 models must be detected in the uploaded papers.")
        else:
            col_a, col_b = st.columns(2)
            model_a = col_a.selectbox("Model A", options=models, index=0)
            model_b = col_b.selectbox("Model B", options=models,
                                       index=min(1, len(models) - 1))

            if st.button("🔍 Compare Models", use_container_width=True, type="primary"):
                if model_a == model_b:
                    st.warning("Please select two different models.")
                else:
                    result = kg.compare_models(model_a, model_b)

                    col1, col2, col3 = st.columns(3)
                    col1.metric(f"Only {model_a}", len(result["only_a"]))
                    col2.metric("Both", len(result["papers_with_both"]))
                    col3.metric(f"Only {model_b}", len(result["only_b"]))

                    st.divider()

                    if result["papers_with_both"]:
                        st.success(
                            f"✅ **{len(result['papers_with_both'])}** paper(s) "
                            f"compare `{model_a}` vs `{model_b}`:"
                        )
                        for p in result["papers_with_both"]:
                            st.markdown(f"- 📄 `{p}`")
                    else:
                        st.info("No single paper compares both models directly.")

                    st.divider()

                    active = resolve_model()
                    if active:
                        with st.spinner(f"Generating analysis: {model_a} vs {model_b}…"):
                            comp_query = (
                                f"Compare {model_a} and {model_b}: differences in "
                                f"architecture, performance, accuracy, training, and results."
                            )
                            chunks = retriever.search(comp_query)
                            if chunks:
                                answer   = generate_answer(comp_query, chunks)
                                verified = verify_answer(answer, chunks)
                                st.markdown(f"#### 📝 {model_a} vs {model_b}")
                                display_verified(verified)
                                log_query(
                                    query=comp_query,
                                    chunks=chunks,
                                    answer=answer,
                                    verified=verified,
                                    model_used=active,
                                )
                            else:
                                st.warning("No relevant context found for this comparison.")
                    else:
                        st.warning(
                            f"Pull an Ollama model to generate the analysis:\n"
                            f"```\nollama pull {OLLAMA_MODEL}\n```"
                        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Debug / Settings
# ─────────────────────────────────────────────────────────────────────────────
with tab_debug:
    st.markdown('<div class="section-header"><h3>Debug & Settings</h3></div>',
                unsafe_allow_html=True)

    # ── System Status ──────────────────────────────────────────
    st.markdown("#### ⚙️ System Status")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**Ollama**")
        if ollama_is_running():
            st.success("Running ✅")
        else:
            st.error("Offline ❌")

    with col2:
        st.markdown("**Active Model**")
        active = resolve_model() if ollama_is_running() else None
        if active:
            st.success(active)
        else:
            st.error("None pulled")

    with col3:
        st.markdown("**Indexed Chunks**")
        count = retriever.collection.count() if retriever else 0
        st.info(f"{count} chunks")

    with col4:
        st.markdown("**Graph Nodes**")
        nodes = kg.graph.number_of_nodes()
        st.info(f"{nodes} nodes")

    st.divider()

    # ── Pull Model Helper ──────────────────────────────────────
    if ollama_is_running() and resolve_model() is None:
        st.error("⚠️ No model is available. Pull one to use the app.")
        st.markdown("**Run this in your terminal:**")
        st.code(f"ollama pull {OLLAMA_MODEL}", language="bash")
        st.caption("After pulling, refresh this page.")
        st.divider()

    # ── All Available Models ───────────────────────────────────
    st.markdown("#### 🤖 Available Ollama Models")
    avail = get_available_models() if ollama_is_running() else []
    if avail:
        for m in avail:
            active = resolve_model()
            badge  = " ← **active**" if m == active else ""
            st.markdown(f"- `{m}`{badge}")
    else:
        st.caption("No models found.")
        st.code(f"ollama pull {OLLAMA_MODEL}", language="bash")

    st.divider()

    # ── Config Table ───────────────────────────────────────────
    # FIX: all values cast to str to avoid PyArrow ArrowTypeError
    st.markdown("#### 🔧 Current Configuration")
    import config as cfg
    import pandas as pd

    config_rows = [
        ["LLM Model (config)",      str(cfg.OLLAMA_MODEL)],
        ["Active Model",            str(resolve_model() or "⚠️ None pulled")],
        ["Embedding Model",         str(cfg.EMBEDDING_MODEL)],
        ["NLI Model",               str(cfg.NLI_MODEL)],
        ["NLI Max Length",          str(cfg.NLI_MAX_LENGTH) + " tokens (fix #5)"],
        ["Reranker Model",          str(cfg.RERANKER_MODEL) + " (fix #4)"],
        ["Reranker Enabled",        str(cfg.RERANKER_ENABLED)],
        ["Chunk Size",              str(cfg.CHUNK_SIZE) + " subword tokens (fix #3)"],
        ["Chunk Overlap",           str(cfg.CHUNK_OVERLAP) + " subword tokens"],
        ["Sentence-Snap Chunks",    str(cfg.SNAP_TO_SENTENCE)],
        ["Dedup Similarity",        str(cfg.DEDUP_SIMILARITY_THRESHOLD) + " (fix #6)"],
        ["Top-K Dense",             str(cfg.TOP_K_DENSE)],
        ["Top-K Sparse",            str(cfg.TOP_K_SPARSE)],
        ["Top-K RRF (pre-rerank)",  str(cfg.TOP_K_RRF)],
        ["Top-K Final",             str(cfg.TOP_K_FINAL)],
        ["Entail Threshold",        str(cfg.NLI_THRESHOLD_ENTAIL)],
        ["Neutral Threshold",       str(cfg.NLI_THRESHOLD_NEUTRAL)],
        ["ChromaDB Path",           str(cfg.CHROMA_PERSIST_DIR)],
        ["Upload Dir",              str(cfg.UPLOAD_DIR)],
        ["Eval Log Path",           str(cfg.EVAL_LOG_PATH)],
    ]
    config_df = pd.DataFrame(config_rows, columns=["Setting", "Value"])
    st.table(config_df)

    st.divider()

    st.markdown("#### 📊 Eval Stats (from logged queries)")
    st.caption(
        "Aggregated from data/eval_logs.jsonl -- every Q&A turn is logged "
        "automatically. Use this to measure whether a future change actually "
        "improves grounded-rate, instead of eyeballing the UI."
    )
    eval_stats = aggregate_stats()
    if eval_stats.get("total_queries", 0) == 0:
        st.info("No queries logged yet. Ask a question in the Q&A tab to start building eval history.")
    else:
        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.metric("Queries Logged", eval_stats["total_queries"])
        ec2.metric("Total Claims", eval_stats["total_claims"])
        gr = eval_stats["grounded_rate"]
        hr = eval_stats["hallucinated_rate"]
        ec3.metric("Grounded Rate", f"{gr*100:.1f}%" if gr is not None else "—")
        ec4.metric("Hallucinated Rate", f"{hr*100:.1f}%" if hr is not None else "—")
        st.caption(
            f"🟢 Grounded: {eval_stats['grounded']}  |  "
            f"🟡 Inferred: {eval_stats['inferred']}  |  "
            f"🔴 Hallucinated: {eval_stats['hallucinated']}"
        )

    st.divider()

    # ── Indexed Sources ────────────────────────────────────────
    st.markdown("#### 📚 Indexed Sources")
    sources_list = retriever.get_indexed_sources()
    if sources_list:
        for s in sources_list:
            st.markdown(f"- `{s}`")
    else:
        st.caption("No sources indexed.")

    st.divider()

    # ── Raw LLM Test ──────────────────────────────────────────
    st.markdown("#### 🧪 Quick LLM Test (No NLI)")
    active_now = resolve_model() if ollama_is_running() else None
    if not active_now:
        st.warning(f"Pull a model first: `ollama pull {OLLAMA_MODEL}`")
    else:
        st.caption(f"Testing with model: `{active_now}`")
        test_query = st.text_input("Test query", placeholder="Enter a test question…")
        if st.button("▶ Run Test") and test_query:
            with st.spinner("Running…"):
                chunks = retriever.search(test_query)
                if chunks:
                    answer = generate_answer(test_query, chunks)
                    st.text_area("Raw LLM Output", value=answer, height=300)
                else:
                    st.warning("No context found. Upload a PDF first.")
