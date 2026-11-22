"""
app.py - Flagship Private AI Search Engine & Local Document/Book Intelligence.
Features:
1. Book PDFs & Long Document Processing (TOC/Chapters, streaming extraction)
2. Tabular Data Intelligence (PDF tables, Word tables, Excel .xlsx, and CSV)
3. 100% In-Memory Executive Summarization (Books, Reports, Datasets)
4. Private Document Q&A with exact page & table citations
5. Live Untracked Web Search & Hybrid mode
"""

import os
import time
from typing import Dict, List
import requests
import streamlit as st
from dotenv import load_dotenv

from search_engine import PrivateSearchEngine
from document_engine import DocumentEngine
from llm_engine import LLMManager

load_dotenv()

st.set_page_config(
    page_title="Private AI Search & Book/Document RAG",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------- MODERN DARK CSS -----------------
st.markdown(
    """
<style>
.stApp {
    background-color: #070a13;
    color: #f1f5f9;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.main-header {
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.5rem;
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    backdrop-filter: blur(14px);
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}
.header-title-box h1 {
    font-size: 1.65rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header-sub {
    font-size: 0.84rem;
    color: #94a3b8;
    margin-top: 4px;
}
.header-badge {
    background: rgba(99, 102, 241, 0.15);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.35);
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.header-badge-green {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.35);
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}
.hero-box {
    text-align: center;
    padding: 2.5rem 1.5rem 1.8rem 1.5rem;
    max-width: 800px;
    margin: 0 auto;
}
.hero-icon {
    font-size: 3rem;
    margin-bottom: 0.8rem;
    display: inline-block;
    filter: drop-shadow(0 0 20px rgba(99, 102, 241, 0.4));
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.8px;
    margin-bottom: 0.8rem;
    line-height: 1.25;
}
.hero-desc {
    font-size: 1.02rem;
    color: #94a3b8;
    line-height: 1.6;
    margin-bottom: 1.8rem;
}
.feature-pills-row {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 2rem;
}
.feat-pill {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: #cbd5e1;
    font-size: 0.8rem;
    font-weight: 500;
    padding: 6px 14px;
    border-radius: 24px;
}
.sources-container {
    margin-top: 1.4rem;
    padding-top: 1.1rem;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
}
.sources-header {
    font-size: 0.9rem;
    font-weight: 600;
    color: #cbd5e1;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.75rem;
}
.sources-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px;
}
.source-card {
    background: rgba(23, 32, 51, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 13px 15px;
    text-decoration: none !important;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: all 0.22s ease-in-out;
}
.source-card:hover {
    background: rgba(30, 41, 59, 0.95);
    border-color: rgba(99, 102, 241, 0.5);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
}
.source-card-doc:hover {
    border-color: rgba(16, 185, 129, 0.5) !important;
}
.badge-doc {
    background: #059669;
    color: #ffffff;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 5px;
}
.badge-table {
    background: #d97706;
    color: #ffffff;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 5px;
    margin-left: 4px;
}
.badge-web {
    background: #2563eb;
    color: #ffffff;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 5px;
}
.match-score-pill {
    font-size: 0.72rem;
    font-weight: 600;
    color: #34d399;
}
.source-domain-tag {
    font-size: 0.76rem;
    color: #94a3b8;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 150px;
}
.source-card-title {
    font-size: 0.88rem;
    font-weight: 600;
    color: #f1f5f9;
    line-height: 1.35;
    margin: 6px 0;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.source-card-snippet {
    font-size: 0.77rem;
    color: #94a3b8;
    line-height: 1.45;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.sidebar-file-item {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 9px 12px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.sidebar-file-name {
    font-size: 0.82rem;
    font-weight: 600;
    color: #e2e8f0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 170px;
}
.sidebar-file-sub {
    font-size: 0.72rem;
    color: #64748b;
}
.action-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(16, 185, 129, 0.1) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Re-initialize if missing or if from older class version
    if "doc_engine" not in st.session_state or not hasattr(st.session_state.doc_engine, "build_summary_context"):
        st.session_state.doc_engine = DocumentEngine()
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = set()


init_session()


def get_installed_ollama_models(base_url: str = "http://localhost:11434") -> List[str]:
    """Retrieves installed models from the local Ollama daemon."""
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=1.5)
        if resp.status_code == 200:
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []


# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.markdown("### 🔍 Search Mode")

    mode_labels = [
        "📄 Document & Book Intelligence (Local RAG)",
        "🌐 Web Search (DuckDuckGo)",
        "🔀 Hybrid Search (Docs + Web)",
    ]
    selected_mode_label = st.radio(
        "Mode Selector",
        options=mode_labels,
        index=0,
        label_visibility="collapsed",
    )

    if "Document" in selected_mode_label:
        active_mode = "document"
    elif "Hybrid" in selected_mode_label:
        active_mode = "hybrid"
    else:
        active_mode = "web"

    st.divider()

    # --- Document & Book Vault ---
    trigger_summary = False
    trigger_table_analysis = False

    if active_mode in ("document", "hybrid"):
        st.markdown("### 📁 Document & Book Vault")
        st.caption("100% in-memory. Supports Books, PDFs, Word, Excel, CSV & Images.")

        uploaded_files = st.file_uploader(
            "Upload Book, Document or Sheet",
            type=["pdf", "docx", "xlsx", "xls", "csv", "txt", "md", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Upload Books (PDF), Financial Sheets (.xlsx, .csv), Word docs, or Scans.",
            label_visibility="collapsed",
        )

        if uploaded_files:
            uploaded_names = {f.name for f in uploaded_files}

            # 1. Ingest any uploaded file not yet indexed in the document engine
            for f in uploaded_files:
                if f.name not in st.session_state.doc_engine.documents:
                    with st.spinner(f"Ingesting '{f.name}'..."):
                        success, msg, count = st.session_state.doc_engine.load_document(f.getvalue(), f.name)
                        if success:
                            st.toast(f"✅ {msg}", icon="📚")
                        else:
                            st.error(f"Error loading {f.name}: {msg}")

            # 2. Sync deletions (if user clicked 'x' on file uploader)
            for indexed_name in list(st.session_state.doc_engine.documents.keys()):
                if indexed_name not in uploaded_names:
                    st.session_state.doc_engine.remove_document(indexed_name)
        else:
            # If uploader is empty, clear in-memory engine
            if st.session_state.doc_engine.documents:
                st.session_state.doc_engine.clear()

        doc_stats = st.session_state.doc_engine.get_summary()
        total_docs = doc_stats.get("total_documents", 0)
        total_chunks = doc_stats.get("total_chunks", 0)
        total_tables = doc_stats.get("total_tables", 0)

        if total_docs > 0:
            table_info = f" • <b>{total_tables} Tables</b>" if total_tables > 0 else ""
            st.markdown(
                f"<div style='font-size:0.8rem; font-weight:600; color:#34d399; margin-bottom:10px;'>"
                f"🟢 <b>{total_docs} Document(s)</b> • <b>{total_chunks} Chunks</b>{table_info}"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Quick Actions Box
            st.markdown(
                "<div class='action-card'>"
                "<div style='font-size:0.82rem; font-weight:700; color:#cbd5e1; margin-bottom:6px;'>⚡ Document Intelligence Actions</div>",
                unsafe_allow_html=True,
            )
            col_act1, col_act2 = st.columns(2)
            with col_act1:
                if st.button("📖 Summarize Book / Doc", use_container_width=True, help="Generates structured executive outline"):
                    trigger_summary = True
            with col_act2:
                if st.button("📊 Tabular Analysis", use_container_width=True, help="Queries metrics, numbers and tables"):
                    trigger_table_analysis = True
            st.markdown("</div>", unsafe_allow_html=True)

            for file_info in doc_stats.get("files", []):
                col_info, col_del = st.columns([0.85, 0.15])
                with col_info:
                    fmt = file_info.get("file_type", "DOC")
                    fname = file_info.get("filename", "Document")
                    fpages = file_info.get("pages", 1)
                    fchunks = file_info.get("chunks_count", 0)
                    icon = "📖" if fmt == "PDF" and fpages > 20 else ("📄" if fmt == "PDF" else ("📊" if fmt in ("XLSX", "XLS", "CSV") else ("📝" if fmt in ("DOCX", "DOC") else "🖼️")))
                    table_tag = f" • {file_info.get('tables_count', 0)} tabs" if file_info.get("tables_count", 0) > 0 else ""
                    st.markdown(
                        f"""
                        <div class="sidebar-file-item">
                            <div>
                                <div class="sidebar-file-name">{icon} {fname}</div>
                                <div class="sidebar-file-sub">{fpages} page(s) • {fchunks} chunks{table_tag}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col_del:
                    if st.button("✕", key=f"del_{fname}", help="Remove file"):
                        st.session_state.doc_engine.remove_document(fname)
                        st.rerun()

            if st.button("🗑️ Clear All Documents", use_container_width=True):
                st.session_state.doc_engine.clear()
                st.rerun()
        else:
            st.info("💡 Upload a Book (PDF), Financial Report (.xlsx), or Word doc above.")

        st.divider()

    # --- AI Inference Backend ---
    st.markdown("### ⚙️ Inference Engine")

    backend_options = ["Ollama (100% Local / Offline)", "Groq (Ultra-Fast Cloud)"]
    selected_backend = st.radio(
        "Inference Backend",
        options=backend_options,
        index=0,
        label_visibility="collapsed",
    )
    provider = "ollama" if "Ollama" in selected_backend else "groq"

    if provider == "ollama":
        ollama_url = st.text_input("Host URL", value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        installed_models = get_installed_ollama_models(ollama_url)
        if installed_models:
            st.success(f"Connected: {len(installed_models)} model(s)")
            model_name = st.selectbox("Model", options=installed_models, index=0)
        else:
            st.warning("⚠️ No models detected.")
            st.code("ollama pull llama3.2", language="bash")
            model_name = st.text_input("Model Name", value="llama3.2")
        groq_api_key = None
    else:
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            placeholder="gsk_...",
        )
        groq_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "deepseek-r1-distill-llama-70b",
        ]
        model_name = st.selectbox("Groq Model", options=groq_models, index=0)
        ollama_url = ""

    # Web retrieval settings if in web or hybrid mode
    if active_mode in ("web", "hybrid"):
        st.divider()
        st.markdown("**🌐 Web Engine**")
        search_backend = st.selectbox("Search Backend", options=["DuckDuckGo (Untracked)", "SearXNG (Self-Hosted)"], index=0)
        backend_key = "duckduckgo" if "DuckDuckGo" in search_backend else "searxng"
        searxng_url = st.text_input("SearXNG URL", value=os.getenv("SEARXNG_URL", "http://localhost:8080")) if backend_key == "searxng" else None
        deep_scrape = st.toggle("Deep Page Scraping", value=True)
    else:
        backend_key = "duckduckgo"
        searxng_url = None
        deep_scrape = False

    max_results = st.slider("Max Context Sources", min_value=2, max_value=8, value=5, step=1)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    st.divider()
    if st.button("🗑️ Reset Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ----------------- MAIN VIEW HEADER -----------------
mode_info = {
    "document": ("📖 Book & Document RAG", "header-badge-green"),
    "web": ("🌐 Untracked Web Search", "header-badge"),
    "hybrid": ("🔀 Hybrid Docs + Web", "header-badge"),
}[active_mode]

st.markdown(
    f"""
    <div class="main-header">
        <div class="header-title-box">
            <h1>Private AI Search & Document Intelligence</h1>
            <div class="header-sub">
                Book PDF Chapters • Structured Tables • Deep Summarization • In-Memory Local RAG
            </div>
        </div>
        <div style="display:flex; gap:10px; align-items:center;">
            <span class="{mode_info[1]}">{mode_info[0]}</span>
            <span class="header-badge">{'🏠 Ollama' if provider == 'ollama' else '⚡ Groq'} ({model_name})</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def render_sources_grid(sources: List[Dict]):
    """Renders modern Perplexity-style citation cards with match indicators and table tags."""
    if not sources:
        return

    cards = []
    for s in sources:
        idx = s.get("id", 1)
        title = s.get("title", "Source")
        snippet = s.get("snippet", "")
        is_doc = s.get("is_document", False)

        if is_doc:
            filename = s.get("filename", "Document")
            match_score = s.get("match_pct", "92% match")
            table_badge = '<span class="badge-table">📊 TABLE</span>' if s.get("is_table") else ""

            card = f"""
            <div class="source-card source-card-doc">
                <div>
                    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
                        <div>
                            <span class="badge-doc">[{idx}] DOC</span>
                            {table_badge}
                        </div>
                        <span class="match-score-pill">{match_score}</span>
                    </div>
                    <div class="source-card-title" title="{title}">{title}</div>
                </div>
                <div class="source-card-snippet">{snippet}</div>
            </div>
            """
        else:
            url = s.get("url", "#")
            domain = s.get("domain", "web")
            card = f"""
            <a class="source-card" href="{url}" target="_blank" rel="noopener noreferrer">
                <div>
                    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
                        <span class="badge-web">[{idx}] WEB</span>
                        <span class="source-domain-tag">{domain}</span>
                    </div>
                    <div class="source-card-title" title="{title}">{title}</div>
                </div>
                <div class="source-card-snippet">{snippet}</div>
            </a>
            """
        cards.append(card)

    st.markdown(
        f"""
        <div class="sources-container">
            <div class="sources-header">
                <span>📚 Sources & Cited References ({len(sources)})</span>
            </div>
            <div class="sources-grid">
                {''.join(cards)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------- EMPTY STATE HERO & PROMPT CHIPS -----------------
preset_triggered_query = None

if not st.session_state.messages and not trigger_summary and not trigger_table_analysis:
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-icon">📖</div>
            <div class="hero-title">Book, Document & Tabular Intelligence.</div>
            <div class="hero-desc">
                Interrogate 100+ page book PDFs, extract tabular sheets (.xlsx / .csv),
                generate executive document summaries, or search the live web privately.
            </div>
            <div class="feature-pills-row">
                <span class="feat-pill">📖 Book PDF Chapter Indexing</span>
                <span class="feat-pill">📊 PDF & Excel Table Extraction</span>
                <span class="feat-pill">📋 Executive Stratified Summaries</span>
                <span class="feat-pill">🛡️ 100% In-Memory Private RAG</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='text-align:center; font-size:0.88rem; color:#64748b; margin-bottom:12px;'>QUICK INQUIRY PRESETS:</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📋 Summarize Document / Book", use_container_width=True):
            trigger_summary = True
    with c2:
        if st.button("📊 Extract Key Tables & Numbers", use_container_width=True):
            trigger_table_analysis = True
    with c3:
        if st.button("🔍 Research Live Web Breakthroughs", use_container_width=True):
            preset_triggered_query = "What are the latest breakthroughs and releases in generative AI and LLMs this week?"


# ----------------- RENDER CHAT HISTORY -----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("thought_process"):
            t = msg["thought_process"]
            with st.expander(f"🔍 Context & Search Execution • {t.get('source_count', 0)} sources ({t.get('duration', 0.0):.2f}s)", expanded=False):
                st.markdown(f"**Dispatched Query:** `{t.get('query')}`")
                st.markdown(f"**Active Mode:** `{t.get('mode', 'search').capitalize()}`")
                if msg.get("sources"):
                    st.markdown("**Referenced Passages:**")
                    for s in msg["sources"]:
                        if s.get("is_document"):
                            tab_lbl = " [📊 Table]" if s.get("is_table") else ""
                            st.markdown(f"- **[{s['id']}] 📄 {s['filename']}** (Page {s.get('page', 1)}){tab_lbl} — *Match: {s.get('match_pct', '90%')}*")
                        else:
                            st.markdown(f"- **[{s['id']}] 🌐 [{s['title']}]({s['url']})** (`{s.get('domain', 'web')}`)")

        st.markdown(msg["content"])

        if msg.get("sources"):
            render_sources_grid(msg["sources"])


# ----------------- CHAT INPUT & EXECUTION -----------------
placeholder_text = {
    "document": "Ask about book chapters, tabular data, formulas, or key findings...",
    "web": "Search the live web privately without tracking...",
    "hybrid": "Ask anything (cross-referencing local files and live web)...",
}[active_mode]

chat_input_val = st.chat_input(placeholder_text)
user_query = preset_triggered_query or chat_input_val

# Handle Explicit Summary Trigger
is_summary_request = trigger_summary
if is_summary_request:
    user_query = "Generate an executive-level structured summary of the document, detailing core objectives, chapter roadmap, key data/tables, and practical conclusions."

# Handle Explicit Tabular Analysis Trigger
if trigger_table_analysis:
    user_query = "Identify and extract all primary tables, numerical metrics, financial figures, and quantitative statistics from the document, presenting them in structured markdown tables."

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        start_time = time.time()
        retrieved_sources: List[Dict] = []
        custom_context = None

        status_box = st.status("🔍 Analyzing documents with local intelligence...", expanded=True)

        try:
            # 1. Summary Workflow
            if is_summary_request or ("summarize" in user_query.lower() and len(user_query.split()) < 10):
                doc_stats = st.session_state.doc_engine.get_summary()
                if doc_stats.get("total_documents", 0) == 0:
                    status_box.update(label="No documents to summarize", state="error", expanded=False)
                    st.warning("⚠️ Please upload a book PDF, report, or sheet in the sidebar to summarize!")
                    st.stop()

                status_box.write("Synthesizing stratified book outline, chapter headers, key tables, and conclusion...")
                custom_context, retrieved_sources = st.session_state.doc_engine.build_summary_context(max_chunks=8)
                exec_mode = "summary"
                elapsed_search = time.time() - start_time
                status_box.write(f"✅ Prepared structured digest across {len(retrieved_sources)} strategic document anchors in {elapsed_search:.2f}s")

            # 2. Document / Tabular Mode
            elif active_mode == "document":
                doc_stats = st.session_state.doc_engine.get_summary()
                if doc_stats.get("total_documents", 0) == 0:
                    status_box.update(label="No documents available", state="error", expanded=False)
                    st.warning("⚠️ No documents uploaded yet. Please upload a book PDF, Word doc, or table in the left sidebar.")
                    st.stop()

                total_chunks_cnt = doc_stats.get("total_chunks", 0)
                total_tables_cnt = doc_stats.get("total_tables", 0)
                status_box.write(f"Scanning {total_chunks_cnt} chunks ({total_tables_cnt} tables) in RAM...")
                retrieved_sources = st.session_state.doc_engine.search(user_query, top_k=max_results)
                exec_mode = "document"
                elapsed_search = time.time() - start_time
                table_hits = sum(1 for s in retrieved_sources if s.get("is_table"))
                tab_info = f" ({table_hits} structured table passages)" if table_hits > 0 else ""
                status_box.write(f"✅ Retrieved **{len(retrieved_sources)} relevant passages**{tab_info} in {elapsed_search:.2f}s")

            # 3. Web Mode
            elif active_mode == "web":
                status_box.write(f"Querying DuckDuckGo without tracking for: `{user_query}`")
                search_engine = PrivateSearchEngine(searxng_url=searxng_url)
                retrieved_sources = search_engine.search_and_enrich(
                    query=user_query,
                    backend=backend_key,
                    max_results=max_results,
                    deep_scrape=deep_scrape,
                )
                exec_mode = "web"
                scraped_count = sum(1 for s in retrieved_sources if s.get("scraped"))
                elapsed_search = time.time() - start_time
                status_box.write(f"✅ Retrieved **{len(retrieved_sources)} web sources** ({scraped_count} deep-scraped) in {elapsed_search:.2f}s")

            # 4. Hybrid Mode
            elif active_mode == "hybrid":
                status_box.write("Cross-referencing in-memory documents and live DuckDuckGo web results...")
                doc_matches = st.session_state.doc_engine.search(user_query, top_k=max(2, max_results // 2))

                search_engine = PrivateSearchEngine(searxng_url=searxng_url)
                web_matches = search_engine.search_and_enrich(
                    query=user_query,
                    backend=backend_key,
                    max_results=max(2, max_results - len(doc_matches)),
                    deep_scrape=deep_scrape,
                )

                retrieved_sources = []
                idx = 1
                for d in doc_matches:
                    d["id"] = idx
                    retrieved_sources.append(d)
                    idx += 1
                for w in web_matches:
                    w["id"] = idx
                    retrieved_sources.append(w)
                    idx += 1

                exec_mode = "hybrid"
                elapsed_search = time.time() - start_time
                status_box.write(f"✅ Combined {len(doc_matches)} document passage(s) + {len(web_matches)} web source(s)")

            # Conclude Thought Accordion
            elapsed_total = time.time() - start_time
            status_box.write(f"Synthesizing answer with **{model_name}**...")
            status_box.update(
                label=f"Retrieved {len(retrieved_sources)} references ({elapsed_total:.2f}s)",
                state="complete",
                expanded=False,
            )

            # 5. LLM Synthesis
            synthesized_answer = ""
            try:
                llm_manager = LLMManager(
                    provider=provider,
                    model_name=model_name,
                    temperature=temperature,
                    groq_api_key=groq_api_key,
                    ollama_base_url=ollama_url,
                )
                stream = llm_manager.stream_answer(
                    query=user_query,
                    sources=retrieved_sources,
                    mode=exec_mode,
                    chat_history=st.session_state.messages[:-1],
                    custom_context=custom_context,
                )
                synthesized_answer = st.write_stream(stream)
            except Exception as llm_err:
                st.warning(f"AI synthesis notice: {llm_err}")
                # Extractive Fallback
                lines = [f"### Relevant Excerpts for: *{user_query}*\n"]
                for s in retrieved_sources:
                    if s.get("is_document"):
                        lines.append(f"- **[{s['id']}] 📄 {s['filename']}** ({s.get('page', 1)}):\n  > {s['content'][:300]}...\n")
                    else:
                        lines.append(f"- **[{s['id']}] 🌐 [{s['title']}]({s['url']})**:\n  > {s['content'][:300]}...\n")
                synthesized_answer = "\n".join(lines)
                st.markdown(synthesized_answer)

            # 6. Render Citations Grid
            if retrieved_sources:
                render_sources_grid(retrieved_sources)

            # 7. Save to Session State
            st.session_state.messages.append({
                "role": "assistant",
                "content": synthesized_answer,
                "thought_process": {
                    "query": user_query,
                    "mode": exec_mode,
                    "duration": elapsed_total,
                    "source_count": len(retrieved_sources),
                },
                "sources": retrieved_sources,
            })

        except Exception as err:
            status_box.update(label="Process encountered an issue", state="error", expanded=True)
            st.error(f"Error: {err}")
            if retrieved_sources:
                render_sources_grid(retrieved_sources)
