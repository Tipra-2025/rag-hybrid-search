"""PRECISION RAG — Streamlit Enterprise UI.

Provides:
- Dashboard: real system stats + component health
- Knowledge Bases: list, create, open, delete
- KB Detail: Overview, Documents, Chunks, Settings tabs
- Document upload + ingestion with real status tracking
- Chunk inspector with search and filters
- Settings: per-KB retrieval configuration

Usage:
    streamlit run src/rag/ui.py
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Bootstrap: must happen before any other streamlit calls
# ---------------------------------------------------------------------------

import os
import sys
from pathlib import Path

# Ensure src is on the path when run directly
_HERE = Path(__file__).parent.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE / "src"))


def _get_api_url() -> str:
    return os.environ.get("PRECISION_RAG_API_URL", "http://localhost:8100")


def render() -> None:  # pragma: no cover — streamlit-only
    import asyncio
    import time

    import requests
    import streamlit as st

    # ------------------------------------------------------------------
    # Page config — MUST be first streamlit call
    # ------------------------------------------------------------------
    st.set_page_config(
        page_title="PRECISION RAG",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ------------------------------------------------------------------
    # CSS: premium dark enterprise design
    # ------------------------------------------------------------------
    st.markdown(
        """
<style>
/* Import Inter font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Root variables */
:root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-card: #1c2333;
    --bg-card-hover: #212940;
    --border: #30363d;
    --accent: #2563eb;
    --accent-light: #3b82f6;
    --accent-glow: rgba(37,99,235,0.15);
    --green: #10b981;
    --red: #ef4444;
    --yellow: #f59e0b;
    --gray: #6b7280;
    --text-primary: #e2e8f0;
    --text-secondary: #8b949e;
    --text-muted: #6b7280;
}

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #111827 100%);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    text-align: left;
    background: transparent;
    border: none;
    color: var(--text-secondary);
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 0.9rem;
    transition: all 0.2s;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--accent-glow);
    color: var(--accent-light);
}

/* Cards */
.rag-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    transition: all 0.2s ease;
}

.rag-card:hover {
    border-color: var(--accent);
    background: var(--bg-card-hover);
    box-shadow: 0 0 0 1px var(--accent-glow), 0 4px 24px rgba(0,0,0,0.4);
}

/* Status badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.badge-green { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.badge-red { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-yellow { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.badge-blue { background: rgba(37,99,235,0.15); color: #3b82f6; border: 1px solid rgba(37,99,235,0.3); }
.badge-gray { background: rgba(107,114,128,0.15); color: #9ca3af; border: 1px solid rgba(107,114,128,0.3); }

/* Metrics row */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent-light);
    line-height: 1;
}

.metric-label {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Status dot */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.dot-green { background: #10b981; box-shadow: 0 0 6px #10b981; }
.dot-red { background: #ef4444; }
.dot-yellow { background: #f59e0b; }
.dot-gray { background: #6b7280; }

/* Table */
.rag-table {
    width: 100%;
    border-collapse: collapse;
}

.rag-table th {
    background: var(--bg-secondary);
    color: var(--text-muted);
    padding: 10px 14px;
    text-align: left;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-bottom: 1px solid var(--border);
}

.rag-table td {
    padding: 12px 14px;
    border-bottom: 1px solid rgba(48,54,61,0.5);
    font-size: 0.9rem;
}

/* Input fields */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > select {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.2s;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    border: none;
    box-shadow: 0 4px 12px rgba(37,99,235,0.3);
}

.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(37,99,235,0.5);
    transform: translateY(-1px);
}

/* Header */
.page-header {
    border-bottom: 1px solid var(--border);
    padding-bottom: 16px;
    margin-bottom: 24px;
}

.page-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
}

.page-subtitle {
    color: var(--text-muted);
    font-size: 0.9rem;
}

/* Divider */
hr {
    border-color: var(--border) !important;
    margin: 16px 0;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #1d4ed8, #7c3aed);
    border-radius: 4px;
}

/* Expander */
.streamlit-expander {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    background: var(--bg-card) !important;
}
</style>
""",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # Session state init
    # ------------------------------------------------------------------
    if "page" not in st.session_state:
        st.session_state.page = "dashboard"
    if "selected_kb" not in st.session_state:
        st.session_state.selected_kb = None
    if "kb_tab" not in st.session_state:
        st.session_state.kb_tab = "overview"

    api_url = _get_api_url()

    # ------------------------------------------------------------------
    # API helpers
    # ------------------------------------------------------------------

    def api_get(path: str, params: dict | None = None) -> dict | None:
        try:
            r = requests.get(f"{api_url}{path}", params=params, timeout=10)
            if r.status_code == 200:
                return r.json()
            return {"error": r.text, "status_code": r.status_code}
        except Exception as e:
            return {"error": str(e)}

    def api_post(path: str, data: dict | None = None) -> dict | None:
        try:
            r = requests.post(f"{api_url}{path}", json=data, timeout=30)
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def api_delete(path: str) -> dict | None:
        try:
            r = requests.delete(f"{api_url}{path}", timeout=10)
            if r.status_code in (200, 204):
                return r.json() if r.content else {"deleted": True}
            return {"error": r.text}
        except Exception as e:
            return {"error": str(e)}

    def api_upload(path: str, file_bytes: bytes, filename: str) -> dict | None:
        try:
            r = requests.post(
                f"{api_url}{path}",
                files={"file": (filename, file_bytes)},
                timeout=120,
            )
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------

    with st.sidebar:
        # Logo
        st.markdown(
            """
            <div style="padding: 20px 0 8px; text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 4px;">🎯</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #3b82f6; letter-spacing: -0.5px;">
                    PRECISION RAG
                </div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 2px;">
                    Enterprise Retrieval Platform
                </div>
            </div>
            <hr>
            """,
            unsafe_allow_html=True,
        )

        # Nav items
        nav_items = [
            ("📊", "Dashboard", "dashboard"),
            ("🗂️", "Knowledge Bases", "knowledge_bases"),
            ("🔍", "Search", "search"),
            ("💬", "Chat", "chat"),
            ("📈", "Evaluations", "evaluations"),
            ("⚙️", "Settings", "settings"),
        ]

        for icon, label, page_key in nav_items:
            active = st.session_state.page == page_key
            btn_label = f"{icon}  {label}" if not active else f"**{icon}  {label}**"
            if st.button(
                btn_label,
                key=f"nav_{page_key}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                st.session_state.page = page_key
                if page_key != "knowledge_bases":
                    st.session_state.selected_kb = None
                st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)

        # API status indicator
        health = api_get("/health")
        if health and "error" not in health:
            st.markdown(
                '<span class="status-dot dot-green"></span> **API Connected**',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-dot dot-red"></span> **API Disconnected**',
                unsafe_allow_html=True,
            )
            st.caption(f"[{api_url}]")

    # ------------------------------------------------------------------
    # Page routing
    # ------------------------------------------------------------------

    page = st.session_state.page

    if page == "dashboard":
        _render_dashboard(api_get, api_url)
    elif page == "knowledge_bases":
        if st.session_state.selected_kb:
            _render_kb_detail(
                st.session_state.selected_kb,
                api_get,
                api_post,
                api_delete,
                api_upload,
            )
        else:
            _render_knowledge_bases(api_get, api_post, api_delete)
    elif page == "chat":
        _render_chat(api_post)
    elif page == "search":
        _render_search_placeholder()
    elif page == "evaluations":
        _render_eval_placeholder()
    elif page == "settings":
        _render_settings_placeholder()
    else:
        _render_dashboard(api_get, api_url)


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------


def _render_dashboard(api_get, api_url) -> None:  # pragma: no cover
    import streamlit as st

    st.markdown(
        """
        <div class="page-header">
            <div class="page-title">📊 Dashboard</div>
            <div class="page-subtitle">Real-time system overview — all values from live data</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # System stats
    stats = api_get("/system/stats") or {}
    health = api_get("/health") or {}

    # Metrics row
    cols = st.columns(5)
    metric_data = [
        ("Knowledge Bases", stats.get("knowledge_bases", 0), "📚"),
        ("Documents", stats.get("total_documents", 0), "📄"),
        ("Chunks", stats.get("total_chunks", 0), "🧩"),
        ("Indexed", stats.get("indexed_documents", 0), "✅"),
        ("Failed", stats.get("failed_documents", 0), "❌"),
    ]

    for col, (label, value, icon) in zip(cols, metric_data):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:1.5rem">{icon}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # System health
    st.markdown("### System Health")

    def _status_badge(status: str) -> str:
        if status == "ok":
            return '<span class="badge badge-green">● OK</span>'
        elif status in ("not_configured", "disabled"):
            return '<span class="badge badge-gray">○ Not Configured</span>'
        elif status == "empty":
            return '<span class="badge badge-yellow">◌ Empty</span>'
        else:
            return f'<span class="badge badge-red">✗ {status}</span>'

    components = [
        ("API Server", health.get("api", {}).get("status", "unknown"), api_url),
        (
            "Indexes",
            health.get("indexes", {}).get("status", "unknown"),
            f"{health.get('indexes', {}).get('n_chunks_dense', 0)} dense chunks",
        ),
        (
            "Embedding Service",
            health.get("embedding_service", {}).get("status", "unknown"),
            health.get("embedding_service", {}).get("model", "—"),
        ),
        (
            "Reranker",
            health.get("reranker", {}).get("status", "unknown"),
            f"{health.get('reranker', {}).get('kind', '—')} / "
            f"{health.get('reranker', {}).get('model', '—')}",
        ),
        (
            "LLM Generation",
            health.get("llm", {}).get("status", "unknown"),
            health.get("llm", {}).get("model", "—"),
        ),
    ]

    col1, col2 = st.columns([3, 2])
    with col1:
        rows = "".join(
            f"""
            <tr>
                <td style="font-weight:500">{name}</td>
                <td>{_status_badge(status)}</td>
                <td style="color:#6b7280;font-size:0.85rem">{detail}</td>
            </tr>
            """
            for name, status, detail in components
        )
        st.markdown(
            f"""
            <div class="rag-card">
                <table class="rag-table">
                    <thead>
                        <tr>
                            <th>Component</th>
                            <th>Status</th>
                            <th>Details</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        # Recent KBs
        kbs_data = api_get("/knowledge-bases") or {}
        kbs = kbs_data.get("knowledge_bases", [])

        st.markdown("**Recent Knowledge Bases**")
        if not kbs:
            st.info("No knowledge bases yet. Create one to get started.")
        else:
            for kb in kbs[:5]:
                st.markdown(
                    f"""
                    <div class="rag-card" style="padding:14px">
                        <div style="font-weight:600;margin-bottom:4px">{kb['name']}</div>
                        <div style="font-size:0.8rem;color:#6b7280">
                            {kb['document_count']} docs · {kb['chunk_count']} chunks
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Page: Knowledge Bases list
# ---------------------------------------------------------------------------


def _render_knowledge_bases(api_get, api_post, api_delete) -> None:  # pragma: no cover
    import streamlit as st

    st.markdown(
        """
        <div class="page-header">
            <div class="page-title">🗂️ Knowledge Bases</div>
            <div class="page-subtitle">Manage isolated document collections for retrieval</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Create KB button
    col_h, col_btn = st.columns([4, 1])
    with col_btn:
        if st.button("➕ New Knowledge Base", type="primary", use_container_width=True):
            st.session_state.show_create_kb = True

    # Create KB dialog
    if st.session_state.get("show_create_kb"):
        with st.form("create_kb_form"):
            st.markdown("### Create Knowledge Base")
            name = st.text_input("Name *", placeholder="e.g. HR Policies")
            description = st.text_area("Description", placeholder="What documents will this KB contain?")
            col_s, col_c = st.columns(2)
            with col_s:
                submitted = st.form_submit_button("Create", type="primary")
            with col_c:
                if st.form_submit_button("Cancel"):
                    st.session_state.show_create_kb = False
                    st.rerun()

            if submitted:
                if not name.strip():
                    st.error("Name is required.")
                else:
                    result = api_post(
                        "/knowledge-bases",
                        {"name": name.strip(), "description": description.strip()},
                    )
                    if result and "error" not in result:
                        st.success(f"Created '{name}'")
                        st.session_state.show_create_kb = False
                        st.rerun()
                    else:
                        st.error(f"Failed: {result}")

    # Load KBs
    data = api_get("/knowledge-bases") or {}
    kbs = data.get("knowledge_bases", [])

    if not kbs:
        st.markdown(
            """
            <div class="rag-card" style="text-align:center;padding:48px">
                <div style="font-size:3rem;margin-bottom:12px">📂</div>
                <div style="font-size:1.1rem;font-weight:600;margin-bottom:8px">No knowledge bases yet</div>
                <div style="color:#6b7280">Create your first knowledge base to start ingesting documents</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # KB cards
    for kb in kbs:
        status = kb.get("status", "active")
        status_badge = (
            '<span class="badge badge-green">● Active</span>'
            if status == "active"
            else f'<span class="badge badge-yellow">◌ {status}</span>'
        )
        created = kb.get("created_at", "")[:10]
        doc_count = kb.get("document_count", 0)
        chunk_count = kb.get("chunk_count", 0)

        col_info, col_actions = st.columns([5, 1])
        with col_info:
            st.markdown(
                f"""
                <div class="rag-card">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start">
                        <div>
                            <div style="font-size:1.1rem;font-weight:600;margin-bottom:4px">
                                {kb['name']}
                            </div>
                            <div style="color:#6b7280;font-size:0.9rem;margin-bottom:12px">
                                {kb.get('description', '') or '<em>No description</em>'}
                            </div>
                            <div style="display:flex;gap:16px;font-size:0.85rem">
                                <span>📄 {doc_count} documents</span>
                                <span>🧩 {chunk_count} chunks</span>
                                <span>📅 {created}</span>
                                <span>{status_badge}</span>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_actions:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Open →", key=f"open_{kb['id']}", type="primary"):
                st.session_state.selected_kb = kb
                st.session_state.kb_tab = "overview"
                st.rerun()
            if st.button("Delete", key=f"del_{kb['id']}"):
                result = api_delete(f"/knowledge-bases/{kb['id']}")
                if result and result.get("deleted"):
                    st.success("Deleted")
                    st.rerun()


# ---------------------------------------------------------------------------
# Page: KB Detail
# ---------------------------------------------------------------------------


def _render_kb_detail(kb, api_get, api_post, api_delete, api_upload) -> None:  # pragma: no cover
    import streamlit as st

    kb_id = kb["id"]

    # Breadcrumb
    col_back, col_title = st.columns([1, 8])
    with col_back:
        if st.button("← Back", type="secondary"):
            st.session_state.selected_kb = None
            st.rerun()
    with col_title:
        st.markdown(
            f"""
            <div style="padding-top:4px">
                <span style="font-size:1.3rem;font-weight:700">{kb['name']}</span>
                <span style="color:#6b7280;font-size:0.9rem;margin-left:12px">
                    {kb.get('description', '')}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Refresh KB data
    fresh_kb = api_get(f"/knowledge-bases/{kb_id}") or kb

    # Tabs
    tab_labels = ["📋 Overview", "📄 Documents", "🧩 Chunks", "⚙️ Settings"]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        _render_kb_overview(fresh_kb, api_get, kb_id)

    with tabs[1]:
        _render_kb_documents(kb_id, api_get, api_post, api_delete, api_upload)

    with tabs[2]:
        _render_kb_chunks(kb_id, api_get)

    with tabs[3]:
        _render_kb_settings(kb_id, fresh_kb, api_post)


def _render_kb_overview(kb, api_get, kb_id) -> None:  # pragma: no cover
    import streamlit as st

    st.markdown("### Knowledge Base Overview")

    # Stats row
    cols = st.columns(4)
    stats_data = [
        ("Documents", kb.get("document_count", 0), "📄"),
        ("Chunks", kb.get("chunk_count", 0), "🧩"),
        ("Status", kb.get("status", "active").title(), "⚡"),
        ("Created", kb.get("created_at", "")[:10], "📅"),
    ]
    for col, (label, value, icon) in zip(cols, stats_data):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:1.3rem">{icon}</div>
                    <div style="font-size:1.5rem;font-weight:700;color:#3b82f6">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Index configuration
    settings = kb.get("retrieval_settings", {})
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Chunking Configuration")
        st.markdown(
            f"""
            <div class="rag-card">
                <table style="width:100%">
                    <tr><td style="color:#6b7280">Strategy</td>
                        <td style="font-weight:600">{settings.get('chunking_strategy', 'recursive')}</td></tr>
                    <tr><td style="color:#6b7280">Chunk Size</td>
                        <td>{settings.get('chunk_size_tokens', 512)} tokens</td></tr>
                    <tr><td style="color:#6b7280">Overlap</td>
                        <td>{settings.get('chunk_overlap_tokens', 64)} tokens</td></tr>
                    <tr><td style="color:#6b7280">Dedup Threshold</td>
                        <td>{settings.get('dedup_cosine_threshold', 0.95)}</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("#### Retrieval Configuration")
        st.markdown(
            f"""
            <div class="rag-card">
                <table style="width:100%">
                    <tr><td style="color:#6b7280">Dense Top-K</td>
                        <td>{settings.get('dense_top_k', 20)}</td></tr>
                    <tr><td style="color:#6b7280">Sparse Top-K</td>
                        <td>{settings.get('sparse_top_k', 20)}</td></tr>
                    <tr><td style="color:#6b7280">RRF k</td>
                        <td>{settings.get('rrf_k', 60)}</td></tr>
                    <tr><td style="color:#6b7280">Final Top-K</td>
                        <td>{settings.get('final_top_k', 5)}</td></tr>
                    <tr><td style="color:#6b7280">Reranker</td>
                        <td>{settings.get('rerank_kind', 'llm')}</td></tr>
                    <tr><td style="color:#6b7280">IDK Threshold</td>
                        <td>{settings.get('idk_threshold', 0.35)}</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_kb_documents(kb_id, api_get, api_post, api_delete, api_upload) -> None:  # pragma: no cover
    import streamlit as st

    st.markdown("### Documents")

    # Upload section
    with st.expander("📤 Upload Documents", expanded=False):
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            accept_multiple_files=True,
            type=["md", "txt", "html", "pdf", "docx", "pptx", "xlsx", "csv", "json"],
            help="Supported: Markdown, Text, HTML, PDF, DOCX, PPTX, XLSX, CSV, JSON",
        )
        if uploaded_files and st.button("⬆️ Upload & Ingest All", type="primary"):
            progress = st.progress(0)
            status_area = st.empty()
            for i, uf in enumerate(uploaded_files):
                status_area.info(f"Processing {uf.name}...")
                result = api_upload(
                    f"/knowledge-bases/{kb_id}/documents",
                    uf.read(),
                    uf.name,
                )
                progress.progress((i + 1) / len(uploaded_files))
                if result and result.get("success"):
                    if result.get("skipped"):
                        status_area.warning(f"⚠️ {uf.name} — {result.get('message', 'skipped')}")
                    else:
                        status_area.success(
                            f"✅ {uf.name} — {result.get('chunks_added', 0)} chunks indexed"
                        )
                else:
                    err = result.get("error", "Unknown error") if result else "No response"
                    status_area.error(f"❌ {uf.name} — {err}")
            st.rerun()

    # Document list
    data = api_get(f"/knowledge-bases/{kb_id}/documents") or {}
    docs = data.get("documents", [])

    if not docs:
        st.info("No documents ingested yet. Upload files above to get started.")
        return

    def _doc_status_badge(status: str) -> str:
        badge_map = {
            "indexed": '<span class="badge badge-green">✓ Indexed</span>',
            "processing": '<span class="badge badge-yellow">⋯ Processing</span>',
            "uploaded": '<span class="badge badge-blue">↑ Uploaded</span>',
            "failed": '<span class="badge badge-red">✗ Failed</span>',
        }
        return badge_map.get(status, f'<span class="badge badge-gray">{status}</span>')

    for doc in docs:
        col_info, col_actions = st.columns([6, 1])
        with col_info:
            size_kb = doc.get("file_size", 0) / 1024
            indexed_at = doc.get("updated_at", "")[:16].replace("T", " ")
            error_msg = doc.get("error_message", "")
            error_html = (
                f'<div style="color:#ef4444;font-size:0.8rem;margin-top:4px">⚠ {error_msg}</div>'
                if error_msg
                else ""
            )
            st.markdown(
                f"""
                <div class="rag-card" style="padding:14px">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <div>
                            <span style="font-weight:600">{doc['filename']}</span>
                            <span style="color:#6b7280;font-size:0.85rem;margin-left:12px">
                                {doc.get('file_type', '').upper()} ·
                                {size_kb:.1f} KB ·
                                {doc.get('chunk_count', 0)} chunks ·
                                {indexed_at}
                            </span>
                            {error_html}
                        </div>
                        {_doc_status_badge(doc.get('status', 'uploaded'))}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_actions:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑", key=f"deldoc_{doc['id']}", help="Delete document"):
                result = api_delete(
                    f"/knowledge-bases/{kb_id}/documents/{doc['id']}"
                )
                if result and result.get("success"):
                    st.rerun()


def _render_kb_chunks(kb_id, api_get) -> None:  # pragma: no cover
    import streamlit as st

    st.markdown("### Chunk Inspector")

    # Filters
    col_search, col_filter_doc, col_limit = st.columns([3, 2, 1])
    with col_search:
        search = st.text_input("🔍 Search chunks", placeholder="Search text...")
    with col_filter_doc:
        docs_data = api_get(f"/knowledge-bases/{kb_id}/documents") or {}
        docs = docs_data.get("documents", [])
        doc_options = {"All documents": None}
        doc_options.update({d["filename"]: d["id"] for d in docs})
        selected_doc_name = st.selectbox("Filter by document", list(doc_options.keys()))
        selected_doc_id = doc_options[selected_doc_name]
    with col_limit:
        limit = st.number_input("Per page", min_value=10, max_value=200, value=50)

    # Pagination state
    if "chunk_offset" not in st.session_state:
        st.session_state.chunk_offset = 0

    params = {"limit": limit, "offset": st.session_state.chunk_offset}
    if search:
        params["search"] = search
    if selected_doc_id:
        params["document_id"] = selected_doc_id

    data = api_get(f"/knowledge-bases/{kb_id}/chunks", params=params) or {}
    chunks = data.get("chunks", [])
    total = data.get("total", 0)

    st.markdown(
        f'<div style="color:#6b7280;font-size:0.85rem;margin-bottom:12px">'
        f'Showing {len(chunks)} of {total} chunks</div>',
        unsafe_allow_html=True,
    )

    if not chunks:
        st.info("No chunks found matching the current filters.")
        return

    # Chunk list
    for chunk in chunks:
        token_count = chunk.get("token_count", 0)
        char_count = chunk.get("char_count", 0)
        text = chunk.get("text", "")
        preview = text[:120] + "..." if len(text) > 120 else text

        with st.expander(
            f"**{chunk['chunk_id'][:16]}…** | "
            f"{chunk.get('source', '')[-40:]!r} | "
            f"pos={chunk.get('position', 0)} | "
            f"{token_count}t / {char_count}c"
        ):
            col_meta, col_text = st.columns([1, 3])
            with col_meta:
                st.markdown(
                    f"""
                    **Metadata**
                    - **Document**: `{chunk.get('document_id', '')[:8]}…`
                    - **Strategy**: {chunk.get('strategy', '—')}
                    - **Position**: {chunk.get('position', 0)}
                    - **Page**: {chunk.get('page', 0)}
                    - **Tokens**: {token_count}
                    - **Chars**: {char_count}
                    - **Start**: {chunk.get('char_start', 0)}
                    - **End**: {chunk.get('char_end', 0)}
                    """
                )
            with col_text:
                st.code(text, language=None)

    # Pagination
    col_prev, col_info, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("← Previous") and st.session_state.chunk_offset > 0:
            st.session_state.chunk_offset = max(0, st.session_state.chunk_offset - limit)
            st.rerun()
    with col_info:
        page_num = st.session_state.chunk_offset // limit + 1
        total_pages = max(1, (total + limit - 1) // limit)
        st.markdown(
            f'<div style="text-align:center;color:#6b7280">Page {page_num} / {total_pages}</div>',
            unsafe_allow_html=True,
        )
    with col_next:
        if st.button("Next →") and st.session_state.chunk_offset + limit < total:
            st.session_state.chunk_offset += limit
            st.rerun()


def _render_kb_settings(kb_id, kb, api_post) -> None:  # pragma: no cover
    import streamlit as st
    import requests

    st.markdown("### Retrieval Settings")
    st.caption("These settings apply when querying this knowledge base. Defaults mirror global environment variables.")

    api_url = _get_api_url()
    settings = kb.get("retrieval_settings", {})

    with st.form("kb_settings_form"):
        st.markdown("#### Chunking")
        col1, col2, col3 = st.columns(3)
        with col1:
            strategy = st.selectbox(
                "Strategy",
                ["recursive", "fixed", "semantic"],
                index=["recursive", "fixed", "semantic"].index(
                    settings.get("chunking_strategy", "recursive")
                ),
            )
        with col2:
            chunk_size = st.number_input(
                "Chunk Size (tokens)", min_value=64, max_value=2048,
                value=settings.get("chunk_size_tokens", 512)
            )
        with col3:
            chunk_overlap = st.number_input(
                "Chunk Overlap (tokens)", min_value=0, max_value=512,
                value=settings.get("chunk_overlap_tokens", 64)
            )

        st.markdown("#### Retrieval")
        col4, col5, col6, col7 = st.columns(4)
        with col4:
            dense_top_k = st.number_input(
                "Dense Top-K", min_value=1, max_value=200,
                value=settings.get("dense_top_k", 20)
            )
        with col5:
            sparse_top_k = st.number_input(
                "Sparse Top-K", min_value=1, max_value=200,
                value=settings.get("sparse_top_k", 20)
            )
        with col6:
            rrf_k = st.number_input(
                "RRF k", min_value=1, max_value=1000,
                value=settings.get("rrf_k", 60),
                help="RRF(d) = Σ 1/(k + rank)"
            )
        with col7:
            final_top_k = st.number_input(
                "Final Top-K", min_value=1, max_value=50,
                value=settings.get("final_top_k", 5)
            )

        st.markdown("#### Generation")
        col8, col9, col10 = st.columns(3)
        with col8:
            rerank_kind = st.selectbox(
                "Reranker",
                ["llm", "none", "cross-encoder"],
                index=["llm", "none", "cross-encoder"].index(
                    settings.get("rerank_kind", "llm")
                ),
            )
        with col9:
            idk_threshold = st.slider(
                "IDK Threshold", 0.0, 1.0,
                value=float(settings.get("idk_threshold", 0.35)),
                step=0.05,
                help="Retrieval confidence below this → 'I don't know'"
            )
        with col10:
            judge_weight = st.slider(
                "Judge Weight", 0.0, 1.0,
                value=float(settings.get("judge_weight", 0.5)),
                step=0.05,
                help="composite = w·citation_accuracy + (1-w)·retrieval"
            )

        dedup_threshold = st.slider(
            "Dedup Cosine Threshold", 0.0, 1.0,
            value=float(settings.get("dedup_cosine_threshold", 0.95)),
            step=0.01,
        )

        if st.form_submit_button("💾 Save Settings", type="primary"):
            payload = {
                "retrieval_settings": {
                    "chunking_strategy": strategy,
                    "chunk_size_tokens": chunk_size,
                    "chunk_overlap_tokens": chunk_overlap,
                    "dense_top_k": dense_top_k,
                    "sparse_top_k": sparse_top_k,
                    "rrf_k": rrf_k,
                    "final_top_k": final_top_k,
                    "rerank_kind": rerank_kind,
                    "idk_threshold": idk_threshold,
                    "judge_weight": judge_weight,
                    "dedup_cosine_threshold": dedup_threshold,
                    "citation_verification_enabled": True,
                    "rerank_model": "gpt-4o-mini",
                }
            }
            try:
                r = requests.put(
                    f"{api_url}/knowledge-bases/{kb_id}/settings",
                    json=payload,
                    timeout=10,
                )
                if r.status_code == 200:
                    st.success("Settings saved!")
                    st.rerun()
                else:
                    st.error(f"Failed: {r.text}")
            except Exception as e:
                st.error(f"Error: {e}")


# ---------------------------------------------------------------------------
# Page: Chat (minimal — Phase 2)
# ---------------------------------------------------------------------------


def _render_chat(api_post) -> None:  # pragma: no cover
    import streamlit as st

    st.markdown(
        """
        <div class="page-header">
            <div class="page-title">💬 Chat</div>
            <div class="page-subtitle">Ask questions against the default index (Phase 2 will add KB selection)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("citations"):
                with st.expander("📎 Citations"):
                    for c in msg["citations"]:
                        st.caption(f"{c.get('marker', '')} — {c.get('source', '')}")
            if msg["role"] == "assistant":
                conf = msg.get("composite_confidence", 0)
                idk = msg.get("is_idk", False)
                ret = msg.get("retrieval_confidence", 0)
                st.caption(
                    f"retrieval={ret:.2f} | composite={conf:.2f} | "
                    f"{'⚠️ IDK' if idk else '✅ Answer'}"
                )

    # Input
    question = st.chat_input("Ask a question...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating..."):
                result = api_post("/v1/ask", {"question": question})
            if result and "error" not in result:
                text = result.get("text", "")
                st.write(text)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": text,
                    "citations": result.get("citations", []),
                    "composite_confidence": result.get("composite_confidence", 0),
                    "retrieval_confidence": result.get("retrieval_confidence", 0),
                    "is_idk": result.get("is_idk", False),
                })
            else:
                err = result.get("error", "Unknown error") if result else "No response from API"
                st.error(err)
        st.rerun()


def _render_search_placeholder() -> None:  # pragma: no cover
    import streamlit as st
    st.markdown("# 🔍 Search")
    st.info("Cross-knowledge-base search is coming in Phase 2.")


def _render_eval_placeholder() -> None:  # pragma: no cover
    import streamlit as st
    st.markdown("# 📈 Evaluations")
    st.info("Evaluation dashboard is coming in Phase 2.")


def _render_settings_placeholder() -> None:  # pragma: no cover
    import streamlit as st
    st.markdown("# ⚙️ Global Settings")
    st.info("Global configuration panel coming in Phase 2. Use .env for now.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    render()
