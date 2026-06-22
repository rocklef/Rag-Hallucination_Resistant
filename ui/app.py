"""
Streamlit Web UI for the Multi-Agent Hallucination-Resistant RAG System.
Run: streamlit run ui/app.py
"""
import sys
import time
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from core.config import config

logging.basicConfig(level=logging.WARNING)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent RAG | Hallucination Resistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero gradient */
    .hero-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .hero-header h1 {
        color: #fff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-header p {
        color: rgba(255,255,255,0.7);
        margin: 0.5rem 0 0;
        font-size: 0.95rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .metric-card .label {
        color: rgba(255,255,255,0.5);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .metric-card .value {
        color: #fff;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Answer box */
    .answer-box {
        background: linear-gradient(145deg, #0a0a1a, #111128);
        border-left: 4px solid #6c63ff;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    /* Agent trace */
    .trace-item {
        background: rgba(255,255,255,0.04);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        font-family: monospace;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.8);
        border-left: 3px solid #6c63ff;
    }

    /* Verdict badges */
    .badge-grounded {
        background: linear-gradient(90deg, #00c853, #64dd17);
        color: #000;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .badge-partial {
        background: linear-gradient(90deg, #ff9800, #ffb300);
        color: #000;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .badge-hallucinated {
        background: linear-gradient(90deg, #f44336, #e53935);
        color: #fff;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.8rem;
    }

    /* Input styling */
    .stTextArea textarea {
        background: #0f0f1a !important;
        color: #fff !important;
        border: 1px solid rgba(108,99,255,0.5) !important;
        border-radius: 8px !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #6c63ff, #4b44cc);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #7c74ff, #5b54dc);
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(108,99,255,0.4);
    }

    /* Source chips */
    .source-chip {
        display: inline-block;
        background: rgba(108,99,255,0.2);
        border: 1px solid rgba(108,99,255,0.4);
        color: #9c94ff;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.8rem;
        margin: 3px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1a1a2e) !important;
    }

    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 0;
        color: rgba(255,255,255,0.7);
        font-size: 0.85rem;
    }
    .pipeline-step .dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #6c63ff;
        flex-shrink: 0;
    }
    .pipeline-arrow {
        color: #6c63ff;
        text-align: center;
        margin: 2px 0 2px 4px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = None
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0


@st.cache_resource(show_spinner=False)
def load_orchestrator():
    from agents.orchestrator import OrchestratorAgent
    return OrchestratorAgent()


@st.cache_resource(show_spinner=False)
def get_store_count():
    from core.vector_store import get_vector_store
    return get_vector_store().count()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🧠 Multi-Agent Hallucination-Resistant RAG</h1>
    <p>7 specialized AI agents working together to give you grounded, verified answers</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown(f"**LLM:** `{config.LLM_PROVIDER.upper()} / {config.OLLAMA_MODEL if config.LLM_PROVIDER == 'ollama' else config.OPENAI_MODEL}`")
    st.markdown(f"**Embeddings:** `{config.EMBEDDING_MODEL}`")
    st.markdown(f"**Vector Store:** `ChromaDB`")

    try:
        doc_count = get_store_count()
        st.metric("Documents in KB", doc_count)
    except Exception:
        st.warning("Vector store not initialized — run `python ingest.py` first")

    st.markdown("---")
    st.markdown("### 🔄 Agent Pipeline")
    pipeline_steps = [
        "1️⃣ Query Reformulation",
        "2️⃣ Multi-Query Retrieval",
        "3️⃣ Relevance Filtering",
        "4️⃣ Cross-Reference Check",
        "5️⃣ Answer Generation",
        "6️⃣ Hallucination Detection",
        "↩️ Auto-Retry if Hallucinated",
    ]
    for step in pipeline_steps:
        st.markdown(f'<div class="pipeline-step"><span class="dot"></span>{step}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📂 Add Documents")
    uploaded = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("📥 Ingest Files"):
        from core.document_loader import load_text
        from core.vector_store import get_vector_store
        store = get_vector_store()
        total = 0
        with st.spinner("Ingesting..."):
            for f in uploaded:
                text = f.read().decode("utf-8", errors="ignore")
                docs = load_text(text, source=f.name)
                added = store.add_documents(docs)
                total += added
        st.success(f"✅ Ingested {total} chunks from {len(uploaded)} file(s)")
        st.cache_resource.clear()

    st.markdown("---")
    use_graph = st.checkbox("Use LangGraph backend", value=False)

    if st.button("🗑️ Clear Chat History"):
        st.session_state.history = []
        st.rerun()

# ── Main Query Area ───────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_area(
        "Ask a question",
        placeholder="e.g., What is RAG and how does it reduce hallucination?",
        height=100,
        label_visibility="collapsed",
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🚀 Ask", use_container_width=True)

# ── Process Query ─────────────────────────────────────────────────────────────
if run_btn and query.strip():
    with st.spinner("🤖 Agents working..."):
        try:
            t0 = time.time()
            if use_graph:
                from graph.rag_graph import run_graph
                state = run_graph(query.strip(), max_retries=config.MAX_RETRIES)
                result_data = {
                    "query": state["query"],
                    "answer": state["answer"],
                    "sources": state.get("sources", []),
                    "sub_queries": state.get("sub_queries", []),
                    "retrieved_chunks": len(state.get("retrieved_chunks", [])),
                    "filtered_chunks": len(state.get("filtered_chunks", [])),
                    "consistency": state.get("cross_ref_report", {}).get("consistency", "?"),
                    "conflicts": state.get("cross_ref_report", {}).get("conflicts", ""),
                    "key_facts": state.get("cross_ref_report", {}).get("key_facts", ""),
                    "hallucination_verdict": state.get("hallucination_verdict", "?"),
                    "hallucination_confidence": state.get("hallucination_confidence", 0),
                    "is_hallucinated": state.get("is_hallucinated", False),
                    "retry_count": state.get("attempt", 0),
                    "elapsed_seconds": round(time.time() - t0, 2),
                    "has_context": state.get("has_context", True),
                    "agent_trace": state.get("agent_trace", []),
                }
            else:
                orchestrator = load_orchestrator()
                result = orchestrator.run(query.strip())
                result_data = result.to_dict()

            st.session_state.history.append(result_data)
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

elif run_btn and not query.strip():
    st.warning("Please enter a question.")

# ── Display Results ───────────────────────────────────────────────────────────
for result in reversed(st.session_state.history):
    with st.container():
        st.markdown(f"**Q:** {result['query']}")

        # Answer
        st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

        # Hallucination verdict
        verdict = result.get("hallucination_verdict", "?")
        conf = result.get("hallucination_confidence", 0)
        if verdict == "GROUNDED":
            badge = f'<span class="badge-grounded">✅ GROUNDED</span>'
        elif verdict == "PARTIALLY_GROUNDED":
            badge = f'<span class="badge-partial">⚠️ PARTIALLY GROUNDED</span>'
        elif verdict in {"HALLUCINATED"}:
            badge = f'<span class="badge-hallucinated">❌ HALLUCINATED</span>'
        else:
            badge = f'<span class="badge-partial">❓ {verdict}</span>'

        st.markdown(
            f'{badge} &nbsp; Confidence: **{conf:.0%}** &nbsp; Time: **{result.get("elapsed_seconds", 0):.1f}s**',
            unsafe_allow_html=True,
        )

        # Expandable details
        with st.expander("📊 Pipeline Details"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sub-queries", len(result.get("sub_queries", [])))
            c2.metric("Retrieved", result.get("retrieved_chunks", 0))
            c3.metric("Filtered", result.get("filtered_chunks", 0))
            c4.metric("Retries", result.get("retry_count", 0))

            if result.get("sub_queries"):
                st.markdown("**Sub-queries generated:**")
                for sq in result["sub_queries"]:
                    st.markdown(f"  - {sq}")

            if result.get("consistency"):
                st.markdown(f"**Cross-reference consistency:** `{result['consistency']}`")

            if result.get("conflicts") and result["conflicts"] != "None detected":
                st.warning(f"⚠️ Conflicts: {result['conflicts']}")

            if result.get("key_facts"):
                st.markdown("**Key verified facts:**")
                st.markdown(result["key_facts"])

        # Sources
        sources = result.get("sources", [])
        if sources:
            source_html = "".join(f'<span class="source-chip">📄 {Path(s).name}</span>' for s in sources)
            st.markdown(f"**Sources:** {source_html}", unsafe_allow_html=True)

        # Agent trace
        with st.expander("🔍 Agent Execution Trace"):
            for step in result.get("agent_trace", []):
                st.markdown(f'<div class="trace-item">{step}</div>', unsafe_allow_html=True)

        st.divider()

# ── Empty state ───────────────────────────────────────────────────────────────
if not st.session_state.history:
    st.markdown("""
    <div style="text-align:center; padding: 3rem; opacity: 0.5;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">💬</div>
        <p>Ask a question above to get started.</p>
        <p style="font-size: 0.85rem;">Make sure to run <code>python ingest.py</code> first to load documents.</p>
    </div>
    """, unsafe_allow_html=True)
