"""
J.A.R.V.I.S. — Personal Knowledge Assistant
Just A Rather Very Intelligent System

Theme: Iron Man HUD — deep black, crimson red, cyan arc reactor, Raleway font
RAG features: semantic chunking, hierarchical summaries, temporal versioning,
              hybrid BM25+semantic search, multi-query fusion, citation highlighting,
              document comparison
"""

import os, time
import streamlit as st
from datetime import datetime, timezone

from src.document_loader import extract_text
from src.semantic_chunker import SemanticChunker
from src.hierarchical_summarizer import HierarchicalSummarizer
from src.temporal_manager import TemporalVersionManager
from src.knowledge_base import KnowledgeBase
from src.rag_chain import build_rag_chain, format_sources
from src.hybrid_search import HybridRetriever
from src.multi_query import MultiQueryFuser
from src.citation_comparator import CitationHighlighter, DocumentComparator
from src.quiz_engine import QuizGenerator
from src.mindmap_generator import MindMapGenerator, render_mindmap_html
from src.logger import get_logger

log = get_logger("app")
log.info("JARVIS PKA starting up…")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="J.A.R.V.I.S. — Knowledge System",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── JARVIS HUD CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Raleway:wght@100;200;300;400;500;600;700;800;900&family=Raleway+Dots&display=swap');

:root {
    --black:    #000000;
    --deep:     #050508;
    --panel:    #0a0a0f;
    --panel2:   #0f0f18;
    --red:      #ff1a1a;
    --red-dim:  #cc1111;
    --red-glow: rgba(255,26,26,0.25);
    --red-faint:rgba(255,26,26,0.08);
    --cyan:     #00e5ff;
    --cyan-dim: #00b8cc;
    --cyan-glow:rgba(0,229,255,0.2);
    --gold:     #ffb700;
    --border:   rgba(255,26,26,0.3);
    --border2:  rgba(0,229,255,0.2);
    --text:     #e8e0d0;
    --muted:    #6b6870;
    --white:    #ffffff;
}

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stApp"] {
    background: var(--black) !important;
    font-family: 'Raleway', sans-serif !important;
    color: var(--text) !important;
}

/* Animated scanlines overlay */
[data-testid="stApp"]::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(255,26,26,0.015) 2px,
        rgba(255,26,26,0.015) 4px
    );
    pointer-events: none;
    z-index: 9999;
    animation: scanmove 8s linear infinite;
}
@keyframes scanmove {
    0%   { background-position: 0 0; }
    100% { background-position: 0 100px; }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--deep) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 4px 0 30px var(--red-glow) !important;
}
[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        90deg,
        transparent,
        transparent 40px,
        rgba(255,26,26,0.03) 40px,
        rgba(255,26,26,0.03) 41px
    );
    pointer-events: none;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebarContent"] { padding: 1.5rem 1rem !important; }

/* Sidebar labels */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] p {
    font-family: 'Raleway', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}

/* Sidebar inputs */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] textarea {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'Raleway', sans-serif !important;
    border-radius: 0 !important;
    box-shadow: inset 0 0 10px var(--red-faint) !important;
}
[data-testid="stSidebar"] input:focus {
    border-color: var(--red) !important;
    box-shadow: 0 0 12px var(--red-glow), inset 0 0 10px var(--red-faint) !important;
}

/* Sidebar buttons */
[data-testid="stSidebar"] button {
    background: transparent !important;
    border: 1px solid var(--red) !important;
    color: var(--red) !important;
    font-family: 'Raleway', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    border-radius: 0 !important;
    transition: all 0.2s !important;
}
[data-testid="stSidebar"] button:hover {
    background: var(--red) !important;
    color: var(--black) !important;
    box-shadow: 0 0 20px var(--red-glow) !important;
}
[data-testid="stSidebar"] button[kind="primary"] {
    background: var(--red) !important;
    color: var(--black) !important;
    box-shadow: 0 0 15px var(--red-glow) !important;
}
[data-testid="stSidebar"] button[kind="primary"]:hover {
    box-shadow: 0 0 30px var(--red-glow) !important;
    transform: translateY(-1px) !important;
}

/* Slider */
[data-testid="stSlider"] > div > div > div { background: var(--red) !important; }
[data-testid="stSlider"] [role="slider"] {
    background: var(--red) !important;
    box-shadow: 0 0 10px var(--red-glow) !important;
    border: none !important;
}

/* ── Main area ── */
.main .block-container { padding: 2rem 2rem 4rem !important; max-width: 1400px !important; }

/* ── JARVIS Header ── */
.jarvis-header {
    position: relative;
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
    overflow: hidden;
}
.jarvis-header::before {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--red), var(--cyan), var(--red), transparent);
    animation: headerline 3s ease-in-out infinite;
}
@keyframes headerline {
    0%,100% { opacity: 0.4; }
    50%      { opacity: 1; }
}
.jarvis-logo {
    font-family: 'Raleway', sans-serif;
    font-weight: 900;
    font-size: 3rem;
    letter-spacing: 0.3em;
    color: var(--red);
    text-shadow: 0 0 30px var(--red), 0 0 60px var(--red-glow);
    line-height: 1;
    text-transform: uppercase;
}
.jarvis-logo span { color: var(--cyan); text-shadow: 0 0 20px var(--cyan); }
.jarvis-tagline {
    font-family: 'Raleway', sans-serif;
    font-weight: 300;
    font-size: 0.72rem;
    letter-spacing: 0.35em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.4rem;
}
.jarvis-version {
    position: absolute;
    top: 2rem; right: 0;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: var(--red-dim);
    font-weight: 500;
    text-transform: uppercase;
}

/* ── HUD Panels ── */
.hud-panel {
    background: var(--panel);
    border: 1px solid var(--border);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    clip-path: polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 0 100%);
}
.hud-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 40px; height: 2px;
    background: var(--red);
    box-shadow: 0 0 8px var(--red);
}
.hud-panel-cyan {
    border-color: var(--border2);
    clip-path: polygon(12px 0, 100% 0, 100% 100%, 0 100%, 0 12px);
}
.hud-panel-cyan::before { background: var(--cyan); box-shadow: 0 0 8px var(--cyan); left: auto; right: 0; }
.hud-label {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--red);
    margin-bottom: 0.5rem;
}
.hud-label-cyan { color: var(--cyan); }

/* ── Stats ── */
.stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1.5rem 0; }
.stat-block {
    background: var(--panel);
    border: 1px solid var(--border);
    padding: 1rem;
    text-align: center;
    clip-path: polygon(0 0, calc(100% - 8px) 0, 100% 8px, 100% 100%, 0 100%);
    position: relative;
    overflow: hidden;
}
.stat-block::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, var(--red), transparent);
}
.stat-num {
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--red);
    text-shadow: 0 0 15px var(--red-glow);
    line-height: 1;
    font-variant-numeric: tabular-nums;
}
.stat-num.cyan { color: var(--cyan); text-shadow: 0 0 15px var(--cyan-glow); }
.stat-lbl {
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.3rem;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Raleway', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    border: none !important;
    border-right: 1px solid var(--border) !important;
    padding: 0.75rem 1.5rem !important;
    background: transparent !important;
    border-radius: 0 !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] [role="tab"]:hover {
    color: var(--red) !important;
    background: var(--red-faint) !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--red) !important;
    background: var(--red-faint) !important;
    box-shadow: inset 0 -2px 0 var(--red) !important;
}

/* ── Chat messages ── */
.msg-user {
    background: var(--panel2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--red);
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 0 100%);
    box-shadow: -4px 0 15px var(--red-faint), inset 0 0 30px var(--red-faint);
}
.msg-assistant {
    background: var(--panel);
    border: 1px solid var(--border2);
    border-left: 3px solid var(--cyan);
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    clip-path: polygon(0 0, 100% 0, 100% 100%, 10px 100%, 0 calc(100% - 10px));
    box-shadow: -4px 0 15px var(--cyan-glow), inset 0 0 30px rgba(0,229,255,0.03);
    line-height: 1.8;
}
.msg-label {
    font-size: 0.6rem;
    font-weight: 800;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.msg-label.user  { color: var(--red); }
.msg-label.asst  { color: var(--cyan); }

/* Citation markers */
.cite-marker {
    display: inline-block;
    background: var(--red);
    color: var(--black) !important;
    font-size: 0.6rem;
    font-weight: 800;
    padding: 0.05rem 0.3rem;
    border-radius: 1px;
    margin-left: 2px;
    vertical-align: super;
}
.cite-block {
    background: var(--panel2);
    border: 1px solid var(--border);
    border-left: 2px solid var(--red);
    padding: 0.5rem 0.75rem;
    margin: 0.3rem 0;
    font-size: 0.8rem;
}
.cite-filename { color: var(--red); font-weight: 700; font-size: 0.7rem; letter-spacing: 0.1em; }

/* Source chips */
.src-chip {
    display: inline-block;
    background: var(--panel2);
    border: 1px solid var(--border);
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    padding: 0.2rem 0.6rem;
    margin: 0.15rem 0.1rem 0;
    text-transform: uppercase;
    color: var(--red);
}
.src-chip.cyan { border-color: var(--border2); color: var(--cyan); }

/* Query chips */
.query-chip {
    display: inline-block;
    background: rgba(255,183,0,0.08);
    border: 1px solid rgba(255,183,0,0.3);
    color: var(--gold);
    font-size: 0.65rem;
    padding: 0.2rem 0.6rem;
    margin: 0.15rem 0.1rem 0;
    font-weight: 600;
    letter-spacing: 0.05em;
}

/* Doc card in sidebar */
.doc-entry {
    background: var(--panel2);
    border: 1px solid var(--border);
    border-left: 2px solid var(--red);
    padding: 0.6rem 0.75rem;
    margin-bottom: 0.4rem;
    font-size: 0.78rem;
}
.doc-entry.updated { border-left-color: var(--gold); }
.doc-entry-name { font-weight: 700; color: var(--text); font-size: 0.8rem; word-break: break-word; }
.doc-entry-meta { font-size: 0.65rem; color: var(--muted); margin-top: 0.2rem; letter-spacing: 0.05em; }
.doc-entry-diff { font-size: 0.68rem; color: var(--gold); margin-top: 0.2rem; }

/* Version badge */
.vbadge {
    display: inline-block;
    background: var(--red);
    color: var(--black) !important;
    font-size: 0.55rem;
    font-weight: 900;
    padding: 0.05rem 0.35rem;
    margin-left: 0.3rem;
    letter-spacing: 0.05em;
    vertical-align: middle;
}
.vbadge.cyan { background: var(--cyan); }

/* Mode selector */
.mode-btn {
    display: inline-block;
    border: 1px solid var(--border);
    padding: 0.4rem 1rem;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    cursor: pointer;
    margin-right: 0.5rem;
    color: var(--muted);
    transition: all 0.2s;
}
.mode-btn.active { border-color: var(--red); color: var(--red); background: var(--red-faint); box-shadow: 0 0 10px var(--red-glow); }

/* Comparison table */
.compare-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin: 1rem 0; }
.compare-table th {
    background: var(--panel2);
    border: 1px solid var(--border);
    padding: 0.6rem 1rem;
    text-align: left;
    font-weight: 700;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--red);
}
.compare-table td {
    border: 1px solid var(--border);
    padding: 0.6rem 1rem;
    vertical-align: top;
    color: var(--text);
    background: var(--panel);
}

/* Empty state */
.empty-hud {
    text-align: center;
    padding: 5rem 2rem;
}
.empty-ring {
    width: 100px;
    height: 100px;
    border: 2px solid var(--red);
    border-top-color: var(--cyan);
    border-radius: 50%;
    margin: 0 auto 2rem;
    animation: spin 3s linear infinite;
    box-shadow: 0 0 20px var(--red-glow);
}
@keyframes spin { to { transform: rotate(360deg); } }
.empty-title {
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--red);
    text-shadow: 0 0 15px var(--red-glow);
    margin-bottom: 0.5rem;
}
.empty-sub { font-size: 0.82rem; color: var(--muted); letter-spacing: 0.08em; line-height: 1.8; }

/* Expander */
.streamlit-expanderHeader {
    font-family: 'Raleway', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--red) !important;
    border: 1px solid var(--border) !important;
    background: var(--panel) !important;
    border-radius: 0 !important;
    padding: 0.5rem 1rem !important;
}
.streamlit-expanderContent {
    border: 1px solid var(--border) !important;
    border-top: none !important;
    background: var(--panel2) !important;
    border-radius: 0 !important;
}

/* Progress bar */
.stProgress > div > div { background: var(--red) !important; box-shadow: 0 0 8px var(--red-glow) !important; }
.stProgress > div { background: var(--panel2) !important; border: 1px solid var(--border) !important; }

/* Chat input */
[data-testid="stChatInput"] { border-top: 1px solid var(--border) !important; background: var(--panel) !important; }
[data-testid="stChatInput"] textarea {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'Raleway', sans-serif !important;
    border-radius: 0 !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--red) !important;
    box-shadow: 0 0 10px var(--red-glow) !important;
}

/* Checkbox / selectbox */
[data-testid="stCheckbox"] label { font-size: 0.78rem !important; font-weight: 500 !important; letter-spacing: 0.05em !important; }
[data-testid="stSelectbox"] > div > div {
    background: var(--panel) !important;
    border-color: var(--border) !important;
    border-radius: 0 !important;
    color: var(--text) !important;
}

/* Info / warning boxes */
.stAlert {
    background: var(--panel2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    color: var(--text) !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--deep); }
::-webkit-scrollbar-thumb { background: var(--red-dim); }

/* Pulse animation for "online" indicators */
@keyframes pulse {
    0%,100% { opacity: 1; box-shadow: 0 0 6px var(--red); }
    50%      { opacity: 0.4; box-shadow: 0 0 2px var(--red); }
}
.pulse-dot {
    display: inline-block;
    width: 6px; height: 6px;
    background: var(--red);
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s ease-in-out infinite;
    vertical-align: middle;
}
.pulse-dot.cyan { background: var(--cyan); box-shadow: 0 0 6px var(--cyan); animation-delay: 0.5s; }

/* Sidebar section title */
.sid-title {
    font-size: 0.6rem;
    font-weight: 800;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--red);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin: 1rem 0 0.6rem;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def init():
    d = {
        "chat_history": [], "conversation": None, "kb": None,
        "version_mgr": None, "doc_summaries": {}, "active_doc_ids": [],
        "processing_log": [], "api_key_set": False,
        "chat_mode": "standard",   # standard | multiquery | compare
        "hybrid_retriever": None,
        "citation_hl": None,
        "comparator": None,
    }
    for k, v in d.items():
        if k not in st.session_state:
            st.session_state[k] = v

init()

@st.cache_resource(show_spinner="⚙ Loading embeddings model…")
def _cached_embeddings():
    """Cached so all-MiniLM-L6-v2 is only downloaded once per session."""
    from src.llm_provider import _load_embeddings
    return _load_embeddings()


def get_kb():
    if st.session_state.kb is None:
        log.info("Initialising KnowledgeBase…")
        st.session_state.kb = KnowledgeBase()
    return st.session_state.kb

def get_vm():
    if st.session_state.version_mgr is None:
        st.session_state.version_mgr = TemporalVersionManager()
    return st.session_state.version_mgr


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="text-align:center;padding:1rem 0 0.5rem">
            <div style="font-family:'Raleway',sans-serif;font-size:1.6rem;font-weight:900;
                        letter-spacing:0.3em;color:#ff1a1a;text-shadow:0 0 20px rgba(255,26,26,0.5);
                        text-transform:uppercase">JARVIS</div>
            <div style="font-size:0.55rem;font-weight:600;letter-spacing:0.3em;
                        color:#6b6870;text-transform:uppercase;margin-top:0.2rem">
                Knowledge System v2.0
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Provider + API Key
    st.markdown('<div class="sid-title">⚡ System Auth</div>', unsafe_allow_html=True)

    provider = st.radio(
        "AI Provider",
        ["OpenAI", "Gemini"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown(
        f'<div style="font-size:0.6rem;letter-spacing:0.1em;color:#6b6870;margin:-0.3rem 0 0.4rem">'
        f'{"OPENAI" if provider == "OpenAI" else "GOOGLE GEMINI"} SELECTED</div>',
        unsafe_allow_html=True,
    )

    if provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-…", label_visibility="collapsed")
        key_hint = "Get key → platform.openai.com"
    else:
        api_key = st.text_input("Gemini API Key", type="password", placeholder="AIza…", label_visibility="collapsed")
        key_hint = "Get key FREE → aistudio.google.com"

    st.markdown(f'<div style="font-size:0.6rem;color:#6b6870;margin-top:0.2rem">🔗 {key_hint}</div>', unsafe_allow_html=True)

    if api_key:
        from src.llm_provider import configure_provider
        configure_provider("gemini" if provider == "Gemini" else "openai", api_key)
        st.session_state.api_key_set = True
        st.session_state.provider = provider
        st.markdown('<div style="font-size:0.65rem;color:#ff1a1a;letter-spacing:0.1em">● AUTH ACCEPTED</div>', unsafe_allow_html=True)

    st.divider()

    # Model
    st.markdown('<div class="sid-title">🧠 Core Systems</div>', unsafe_allow_html=True)
    from src.llm_provider import available_models
    _active_provider = "gemini" if st.session_state.get("provider") == "Gemini" else "openai"
    model = st.selectbox("LLM Core", available_models(_active_provider), index=0, label_visibility="visible")

    # RAG mode
    st.markdown('<div class="sid-title">⚙ Retrieval Mode</div>', unsafe_allow_html=True)
    rag_mode = st.radio(
        "RAG Mode",
        ["Standard RAG", "Hybrid BM25+Semantic", "Multi-Query Fusion"],
        index=0,
        label_visibility="collapsed",
    )

    use_citations  = st.checkbox("Citation Highlighting", value=True)
    build_summaries = st.checkbox("Hierarchical Summaries", value=False)

    with st.expander("⚙ CHUNKING"):
        bp_pct   = st.slider("Breakpoint sensitivity", 70, 95, 85)
        min_tok  = st.slider("Min tokens/chunk", 40, 150, 80)
        max_tok  = st.slider("Max tokens/chunk", 200, 800, 400)
        win_sz   = st.slider("Smoothing window", 1, 5, 2)

    with st.expander("⚙ RETRIEVAL"):
        top_k    = st.slider("Retrieved chunks (k)", 3, 12, 6)
        n_queries = st.slider("Multi-query variants", 2, 5, 3)
        bm25_alpha = st.slider("Semantic weight α", 0.0, 1.0, 0.5, 0.1)
        memory_k = st.slider("Memory window", 2, 10, 5)

    st.divider()

    # Upload
    st.markdown('<div class="sid-title">📂 Document Upload</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "PDF / TXT / MD", type=["pdf","txt","md"],
        accept_multiple_files=True, label_visibility="collapsed",
    )
    process_btn = st.button("⚡  INITIALIZE SYSTEM", use_container_width=True, type="primary")

    st.divider()

    # Knowledge Base list
    vm = get_vm()
    all_docs = vm.get_all_documents()
    if all_docs:
        st.markdown('<div class="sid-title">📡 Active Intel</div>', unsafe_allow_html=True)
        for doc in all_docs:
            history = vm.get_document_history(doc.doc_id)
            badge = '<span class="vbadge">NEW</span>' if len(history)==1 else f'<span class="vbadge">{doc.version}</span>'
            diff_html = f'<div class="doc-entry-diff">↻ {doc.diff_summary}</div>' if doc.diff_summary else ""
            cls = "doc-entry updated" if doc.diff_summary else "doc-entry"
            st.markdown(
                f'<div class="{cls}">'
                f'<div class="doc-entry-name">{doc.filename[:28]}{badge}</div>'
                f'<div class="doc-entry-meta">{doc.word_count:,}w · {doc.age_label()}</div>'
                f'{diff_html}</div>', unsafe_allow_html=True
            )
        if st.button("🗑  PURGE DATABASE", use_container_width=True):
            st.session_state.kb = None
            st.session_state.conversation = None
            st.session_state.chat_history = []
            st.session_state.doc_summaries = {}
            st.session_state.active_doc_ids = []
            st.session_state.hybrid_retriever = None
            st.rerun()

    if st.button("🗑  CLEAR CHAT", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.conversation = None
        st.rerun()


# ── Ingestion pipeline ────────────────────────────────────────────────────────
if process_btn:
    if not st.session_state.api_key_set:
        st.sidebar.error("Auth required — enter API key.")
    elif not uploaded_files:
        st.sidebar.error("No documents uploaded.")
    else:
        prog = st.sidebar.progress(0, text="INITIALIZING…")
        log = []
        chunker = SemanticChunker(breakpoint_percentile=bp_pct, min_chunk_tokens=min_tok,
                                   max_chunk_tokens=max_tok, window_size=win_sz)
        summarizer = HierarchicalSummarizer(model=model, chunks_per_section=5)
        kb = get_kb(); vm = get_vm()

        all_ingested_docs = []

        for i, uf in enumerate(uploaded_files):
            base = i / len(uploaded_files)
            try:
                prog.progress(base + 0.0/len(uploaded_files), text=f"READING {uf.name}…")
                raw_text, pages = extract_text(uf)
                log.append(f"✓ {uf.name} — {pages}p, {len(raw_text.split()):,}w")

                prog.progress(base + 0.2/len(uploaded_files), text="SEMANTIC CHUNKING…")
                did_tmp = uf.name.replace(" ","_").replace(".","_")
                chunks = chunker.chunk(raw_text, source_file=uf.name, doc_id=did_tmp)
                log.append(f"  ↳ {len(chunks)} semantic chunks")

                prog.progress(base + 0.4/len(uploaded_files), text="VERSION CONTROL…")
                doc_id, is_new, diff = vm.register_document(uf.name, raw_text, chunks)
                for j, c in enumerate(chunks):
                    c.chunk_id = f"{doc_id}_chunk_{j:04d}"
                    c.source_file = uf.name
                log.append(f"  ↳ {'NEW v1' if is_new else f'Updated: {diff}'}")

                prog.progress(base + 0.6/len(uploaded_files), text="INDEXING…")
                current = vm.get_current_version(doc_id)
                kb.add_document(chunks=chunks, doc_id=doc_id, doc_version=current.version,
                                filename=uf.name, uploaded_at=current.uploaded_at)
                all_ingested_docs.extend([(c, doc_id, current.version, uf.name, current.uploaded_at) for c in chunks])

                if build_summaries:
                    prog.progress(base + 0.8/len(uploaded_files), text="HIERARCHICAL SUMMARY…")
                    summary = summarizer.summarize(chunks, doc_id=doc_id, filename=uf.name)
                    st.session_state.doc_summaries[doc_id] = summary
                    log.append(f"  ↳ {len(summary.section_summaries)} sections summarized")

                if doc_id not in st.session_state.active_doc_ids:
                    st.session_state.active_doc_ids.append(doc_id)

            except Exception as e:
                log.append(f"✗ {uf.name}: {e}")

        # Build hybrid retriever if needed
        if "Hybrid" in rag_mode and kb._vectorstore:
            from langchain_core.documents import Document as LCDoc
            all_lc_docs = []
            for chunk_id, meta in kb._chunk_meta.items():
                # rebuild docs from metadata
                all_lc_docs.append(LCDoc(
                    page_content=meta.get("text",""),
                    metadata=meta
                ))
            hr = HybridRetriever(kb._vectorstore, alpha=bm25_alpha)
            # we store it for query use
            st.session_state.hybrid_retriever = hr

        prog.progress(1.0, text="SYSTEM ONLINE")
        st.session_state.processing_log = log
        st.session_state.conversation = None
        st.rerun()


# ── Main UI ───────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="jarvis-header">
    <div class="jarvis-logo">J<span>.</span>A<span>.</span>R<span>.</span>V<span>.</span>I<span>.</span>S<span>.</span></div>
    <div class="jarvis-tagline">
        <span class="pulse-dot"></span>Just A Rather Very Intelligent System
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <span class="pulse-dot cyan"></span>Personal Knowledge Assistant
    </div>
    <div class="jarvis-version">MARK VII · BUILD 2025.02</div>
</div>
""", unsafe_allow_html=True)

# Processing log
if st.session_state.processing_log:
    with st.expander("📋 SYSTEM INITIALIZATION LOG", expanded=True):
        for line in st.session_state.processing_log:
            color = "#ff1a1a" if line.startswith("✓") else "#ffb700" if "↳" in line else "#cc1111"
            st.markdown(f'<div style="font-family:monospace;font-size:0.78rem;color:{color};padding:0.1rem 0">{line}</div>', unsafe_allow_html=True)
    st.session_state.processing_log = []

kb = get_kb()
vm = get_vm()
stats = kb.get_stats()
all_docs = vm.get_all_documents()

# Stats
if stats["total_chunks"] > 0:
    total_words = sum(d.word_count for d in all_docs)
    versions = sum(len(vm.get_document_history(d.doc_id)) for d in all_docs)
    st.markdown(f"""
    <div class="stats-grid">
        <div class="stat-block">
            <div class="stat-num">{stats['total_documents']}</div>
            <div class="stat-lbl">Documents</div>
        </div>
        <div class="stat-block">
            <div class="stat-num cyan">{stats['total_chunks']}</div>
            <div class="stat-lbl">Semantic Chunks</div>
        </div>
        <div class="stat-block">
            <div class="stat-num">{total_words:,}</div>
            <div class="stat-lbl">Words Indexed</div>
        </div>
        <div class="stat-block">
            <div class="stat-num cyan">{versions}</div>
            <div class="stat-lbl">Versions Tracked</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Empty state
if kb.is_empty():
    st.markdown("""
    <div class="empty-hud">
        <div class="empty-ring"></div>
        <div class="empty-title">System Offline</div>
        <div class="empty-sub">
            Upload documents in the sidebar and click<br>
            ⚡ INITIALIZE SYSTEM to bring JARVIS online.<br><br>
            Supported: PDF · TXT · Markdown
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_compare, tab_docs, tab_sums, tab_quiz, tab_mindmap, tab_guide = st.tabs([
    "◈ CHAT INTERFACE",
    "⚔ COMPARE DOCS",
    "📡 DOCUMENT REGISTRY",
    "📑 KNOWLEDGE MAP",
    "🎯 QUIZ MODE",
    "🧠 MIND MAP",
    "🚀 DEPLOY GUIDE",
])


# ── TAB: CHAT ─────────────────────────────────────────────────────────────────
with tab_chat:

    # Mode indicator
    mode_colors = {
        "Standard RAG": ("#ff1a1a", "STANDARD RAG"),
        "Hybrid BM25+Semantic": ("#ffb700", "HYBRID BM25+SEMANTIC"),
        "Multi-Query Fusion": ("#00e5ff", "MULTI-QUERY FUSION"),
    }
    mc, ml = mode_colors.get(rag_mode, ("#ff1a1a", "STANDARD"))
    st.markdown(
        f'<div style="font-size:0.62rem;font-weight:800;letter-spacing:0.2em;'
        f'color:{mc};border:1px solid {mc};display:inline-block;padding:0.3rem 0.8rem;margin-bottom:1rem;">'
        f'<span class="pulse-dot" style="background:{mc};box-shadow:0 0 6px {mc}"></span>{ml}</div>',
        unsafe_allow_html=True
    )

    # Build / rebuild chain
    if st.session_state.conversation is None and not kb.is_empty():
        doc_ids = [d.doc_id for d in all_docs]
        retriever = kb.get_retriever(k=top_k, doc_ids=doc_ids)
        if retriever:
            temporal_ctx = vm.get_temporal_context(doc_ids)
            hierarchy_ctx = ""
            for did, summary in st.session_state.doc_summaries.items():
                hierarchy_ctx += f"\n[{summary.filename}]\n{summary.document_summary}\n"
            st.session_state.conversation = build_rag_chain(
                retriever=retriever, kb=kb, model=model,
                temporal_context=temporal_ctx,
                hierarchy_context=hierarchy_ctx.strip(),
                memory_window=memory_k,
            )
        st.session_state.citation_hl = CitationHighlighter(model=model)

    # Render chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="msg-user">'
                f'<div class="msg-label user">▶ Operator Query</div>'
                f'{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            # Build answer content
            answer_html = msg.get("cited_answer") or msg["content"]

            # Query chips (multi-query mode)
            queries_html = ""
            if msg.get("queries"):
                chips = "".join(f'<span class="query-chip">{q[:60]}</span>' for q in msg["queries"][1:])
                queries_html = f'<div style="margin-bottom:0.6rem"><span style="font-size:0.6rem;color:#ffb700;letter-spacing:0.1em;text-transform:uppercase">Query variants: </span>{chips}</div>'

            # Source chips
            src_html = ""
            for src in msg.get("sources", []):
                src_html += f'<span class="src-chip">📄 {src["filename"][:20]} v{src["version"]}</span>'

            st.markdown(
                f'<div class="msg-assistant">'
                f'<div class="msg-label asst">{msg.get("intent_icon","◈")} JARVIS · {msg.get("intent","RESPONSE").upper()}</div>'
                f'{queries_html}'
                f'{answer_html}'
                f'{"<br/><br/>" + src_html if src_html else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Citations
            if msg.get("citations"):
                with st.expander("📎 CITATION INDEX"):
                    for cit in msg["citations"]:
                        st.markdown(
                            f'<div class="cite-block">'
                            f'<span class="cite-marker">{cit.marker}</span> '
                            f'<span class="cite-filename">{cit.source_filename}</span> '
                            f'<span style="font-size:0.65rem;color:#6b6870"> · chunk {cit.source_chunk_id}</span><br/>'
                            f'<span style="font-size:0.78rem;color:#e8e0d0">{cit.evidence}</span></div>',
                            unsafe_allow_html=True,
                        )

            # Sources
            if msg.get("sources"):
                with st.expander("🔍 SOURCE PASSAGES"):
                    for src in msg["sources"]:
                        st.markdown(
                            f'<div class="cite-block">'
                            f'<span class="cite-filename">{src["filename"]}</span> '
                            f'<span style="font-size:0.65rem;color:#6b6870">`{src["chunk_id"]}` · v{src["version"]}</span><br/>'
                            f'<span style="font-size:0.78rem;color:#9a9488">{src["text_preview"]}</span></div>',
                            unsafe_allow_html=True,
                        )

    # Input
    if prompt := st.chat_input("Query the knowledge base…"):
        if not st.session_state.api_key_set:
            st.error("System offline — no API key.")
        elif st.session_state.conversation is None:
            st.error("No documents indexed.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.rerun()

    # Generate response
    history = st.session_state.chat_history
    if history and history[-1]["role"] == "user" and st.session_state.conversation:
        last_q = history[-1]["content"]
        with st.spinner("◈ JARVIS processing…"):
            try:
                msg_data = {"role": "assistant", "content": "", "sources": [], "citations": [], "queries": []}

                if rag_mode == "Multi-Query Fusion":
                    doc_ids = [d.doc_id for d in all_docs]
                    retriever_obj = kb.get_retriever(k=top_k, doc_ids=doc_ids)
                    fuser = MultiQueryFuser(
                        retriever=retriever_obj, model=model,
                        n_queries=n_queries,
                        temporal_context=vm.get_temporal_context(doc_ids),
                    )
                    answer, src_docs, queries = fuser.answer(last_q)
                    msg_data["content"] = answer
                    msg_data["queries"] = queries
                    msg_data["sources"] = format_sources(src_docs)
                    if use_citations and st.session_state.citation_hl:
                        cited = st.session_state.citation_hl.highlight(answer, src_docs)
                        msg_data["cited_answer"] = cited.answer_with_markers
                        msg_data["citations"] = cited.citations
                else:
                    result = st.session_state.conversation({"question": last_q})
                    answer = result["answer"]
                    src_docs = result.get("source_documents", [])
                    intent = result.get("intent", "factual")
                    intent_icon = result.get("intent_icon", "◈")
                    msg_data["content"] = answer
                    msg_data["sources"] = format_sources(src_docs)
                    msg_data["intent"] = intent
                    msg_data["intent_icon"] = intent_icon
                    if use_citations and st.session_state.citation_hl:
                        cited = st.session_state.citation_hl.highlight(answer, src_docs)
                        msg_data["cited_answer"] = cited.answer_with_markers
                        msg_data["citations"] = cited.citations

                st.session_state.chat_history.append(msg_data)
            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "assistant", "content": f"⚠️ System error: {e}",
                    "sources": [], "citations": [], "queries": [],
                })
        st.rerun()


# ── TAB: COMPARE ──────────────────────────────────────────────────────────────
with tab_compare:
    st.markdown('<div class="hud-label">⚔ DOCUMENT COMPARISON ENGINE</div>', unsafe_allow_html=True)
    if len(all_docs) < 2:
        st.markdown('<div class="hud-panel"><span style="color:#6b6870;font-size:0.85rem">Upload at least 2 documents to enable comparison mode.</span></div>', unsafe_allow_html=True)
    else:
        doc_options = {d.filename: d.doc_id for d in all_docs}
        col1, col2 = st.columns(2)
        with col1:
            doc_a = st.selectbox("Document A", list(doc_options.keys()), key="cmp_a")
        with col2:
            remaining = [k for k in doc_options if k != doc_a]
            doc_b = st.selectbox("Document B", remaining, key="cmp_b")

        compare_q = st.text_input("Comparison question", placeholder="e.g. What are the key differences in approach?")
        if st.button("⚔ RUN COMPARISON", type="primary"):
            if compare_q and doc_a and doc_b:
                with st.spinner("Analyzing documents…"):
                    comparator = DocumentComparator(kb, model=model)
                    selected_ids = [doc_options[doc_a], doc_options[doc_b]]
                    names = {doc_options[doc_a]: doc_a, doc_options[doc_b]: doc_b}
                    result = comparator.compare(compare_q, selected_ids, names, k_per_doc=3)

                st.markdown(f'<div class="hud-label">COMPARISON: {compare_q}</div>', unsafe_allow_html=True)
                if result.comparison_table:
                    st.markdown(result.comparison_table)
                st.markdown(f'<div class="hud-panel-cyan hud-panel" style="margin-top:1rem">'
                           f'<div class="hud-label hud-label-cyan">NARRATIVE ANALYSIS</div>'
                           f'{result.narrative}</div>', unsafe_allow_html=True)


# ── TAB: DOCS ─────────────────────────────────────────────────────────────────
with tab_docs:
    st.markdown('<div class="hud-label">📡 DOCUMENT REGISTRY — VERSION CONTROL</div>', unsafe_allow_html=True)
    for doc in all_docs:
        history = vm.get_document_history(doc.doc_id)
        with st.expander(f"{'▶' if len(history) == 1 else '↻'} {doc.filename}  ·  v{doc.version}  ·  {doc.word_count:,} words  ·  {doc.age_label()}"):
            cols = st.columns([2, 1, 1])
            with cols[0]:
                st.markdown(f'<div style="font-size:0.78rem;color:#6b6870">DOC ID: <span style="color:#ff1a1a;font-family:monospace">{doc.doc_id}</span></div>', unsafe_allow_html=True)
                if doc.diff_summary:
                    st.markdown(f'<div style="font-size:0.78rem;color:#ffb700;margin-top:0.3rem">Latest changes: {doc.diff_summary}</div>', unsafe_allow_html=True)
            with cols[1]:
                st.metric("Versions", len(history))
            with cols[2]:
                st.metric("Words", f"{doc.word_count:,}")

            if len(history) > 1:
                st.markdown('<div style="font-size:0.65rem;font-weight:800;letter-spacing:0.15em;color:#6b6870;text-transform:uppercase;margin-top:0.5rem">Version History</div>', unsafe_allow_html=True)
                for v in reversed(history):
                    status = "🔴 CURRENT" if v.is_current else "⚫ ARCHIVED"
                    st.markdown(f'<div style="font-size:0.75rem;padding:0.3rem 0;border-bottom:1px solid rgba(255,26,26,0.1)">'
                               f'v{v.version} · {v.age_label()} · {status}'
                               f'{"  |  <span style=\"color:#ffb700\">" + v.diff_summary + "</span>" if v.diff_summary else ""}</div>',
                               unsafe_allow_html=True)


# ── TAB: SUMMARIES ────────────────────────────────────────────────────────────
with tab_sums:
    st.markdown('<div class="hud-label">📑 HIERARCHICAL KNOWLEDGE MAP</div>', unsafe_allow_html=True)
    if not st.session_state.doc_summaries:
        st.markdown('<div class="hud-panel"><span style="color:#6b6870;font-size:0.85rem">Enable "Hierarchical Summaries" and re-ingest to build the knowledge map.</span></div>', unsafe_allow_html=True)
    else:
        for doc_id, summary in st.session_state.doc_summaries.items():
            st.markdown(f'<div class="hud-panel"><div class="hud-label">{summary.filename}</div>'
                       f'<div style="font-size:0.85rem;line-height:1.7">{summary.document_summary}</div>'
                       f'<div style="font-size:0.65rem;color:#6b6870;margin-top:0.5rem;letter-spacing:0.08em">'
                       f'{summary.total_chunks} CHUNKS · {len(summary.section_summaries)} SECTIONS · {summary.total_words:,} WORDS</div>'
                       f'</div>', unsafe_allow_html=True)
            with st.expander("SECTION SUMMARIES"):
                for i, sec in enumerate(summary.section_summaries, 1):
                    st.markdown(f'<div class="cite-block"><span class="cite-filename">SECTION {i}</span> '
                               f'<span style="font-size:0.65rem;color:#6b6870">({len(sec.chunk_ids)} chunks)</span><br/>'
                               f'<span style="font-size:0.82rem;color:#e8e0d0">{sec.summary}</span></div>',
                               unsafe_allow_html=True)
            with st.expander("CHUNK SUMMARIES (first 20)"):
                for cs in summary.chunk_summaries[:20]:
                    st.markdown(f'<div style="font-size:0.72rem;color:#6b6870;padding:0.3rem 0;border-bottom:1px solid rgba(255,26,26,0.1)">'
                               f'<span style="color:#ff1a1a;font-family:monospace">{cs.chunk_id}</span> — {cs.summary}</div>',
                               unsafe_allow_html=True)


# ── TAB: QUIZ ─────────────────────────────────────────────────────────────────
with tab_quiz:
    st.markdown('<div class="hud-label">🎯 QUIZ MODE — TEST YOUR KNOWLEDGE</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        quiz_topic = st.text_input("Topic / focus area", placeholder="e.g. cloud security, EC2, networking… (leave blank for general)", label_visibility="visible")
    with col2:
        n_questions = st.selectbox("Questions", [3, 5, 7, 10], index=1)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        gen_quiz_btn = st.button("⚡ GENERATE QUIZ", type="primary", use_container_width=True)

    if gen_quiz_btn:
        if kb.is_empty():
            st.error("No documents indexed.")
        else:
            with st.spinner("◈ Generating quiz…"):
                try:
                    qgen = QuizGenerator(kb=kb, model=model)
                    doc_ids = [d.doc_id for d in all_docs]
                    quiz = qgen.generate(topic=quiz_topic, n_questions=n_questions, doc_ids=doc_ids)
                    if quiz.questions:
                        st.session_state["current_quiz"] = quiz
                        st.session_state["quiz_submitted"] = False
                        st.rerun()
                    else:
                        st.error("Could not generate questions. Try a different topic.")
                except Exception as e:
                    err = str(e).lower()
                    if "429" in str(e) or "quota" in err or "rate" in err or "exhausted" in err:
                        st.warning("⏳ Rate limit hit (5 req/min on free tier). Wait 60 seconds and try again.")
                    else:
                        st.error(f"Quiz error: {e}")

    quiz = st.session_state.get("current_quiz")
    quiz_submitted = st.session_state.get("quiz_submitted", False)

    if quiz and quiz.questions:
        # Score banner
        if quiz_submitted:
            score = quiz.calculate_score()
            pct = int(score / len(quiz.questions) * 100)
            color = "#ff1a1a" if pct < 50 else "#ffb700" if pct < 80 else "#00e5ff"
            st.markdown(f"""
            <div style="background:#0a0a0f;border:1px solid {color};padding:1.5rem;text-align:center;
                        margin-bottom:1.5rem;clip-path:polygon(0 0,calc(100% - 12px) 0,100% 12px,100% 100%,0 100%);">
                <div style="font-size:3rem;font-weight:900;color:{color};text-shadow:0 0 20px {color}66">{pct}%</div>
                <div style="font-size:0.7rem;letter-spacing:0.2em;color:#6b6870;text-transform:uppercase;margin-top:0.3rem">
                    {score} / {len(quiz.questions)} CORRECT
                </div>
            </div>""", unsafe_allow_html=True)

        # Questions
        user_answers = {}
        for i, q in enumerate(quiz.questions):
            answered = quiz_submitted

            # Status indicator
            if answered:
                status_color = "#00e5ff" if q.is_correct else "#ff1a1a"
                status_icon = "✓" if q.is_correct else "✗"
            else:
                status_color = "#6b6870"
                status_icon = str(i+1)

            st.markdown(f"""
            <div class="hud-panel" style="margin-bottom:0.8rem;border-color:{status_color}33">
                <div style="font-size:0.65rem;font-weight:800;letter-spacing:0.15em;color:{status_color};margin-bottom:0.6rem">
                    QUESTION {status_icon}
                </div>
                <div style="font-size:0.95rem;font-weight:600;color:#e8e0d0;line-height:1.5;margin-bottom:0.8rem">{q.question}</div>
            </div>""", unsafe_allow_html=True)

            if not answered:
                choice = st.radio(
                    f"q{i}",
                    q.options,
                    index=None,
                    label_visibility="collapsed",
                    key=f"quiz_q_{i}",
                )
                if choice:
                    user_answers[i] = choice[0]  # "A", "B", etc.
            else:
                for opt in q.options:
                    is_correct_opt = opt[0].upper() == q.answer.upper()
                    is_user_opt = q.user_answer and opt[0].upper() == q.user_answer.upper()
                    if is_correct_opt:
                        color_css = "color:#00e5ff;font-weight:700"
                        prefix = "✓ "
                    elif is_user_opt and not is_correct_opt:
                        color_css = "color:#ff1a1a;text-decoration:line-through"
                        prefix = "✗ "
                    else:
                        color_css = "color:#6b6870"
                        prefix = "  "
                    st.markdown(f'<div style="font-size:0.85rem;padding:0.2rem 0.5rem;{color_css}">{prefix}{opt}</div>', unsafe_allow_html=True)

                if q.explanation:
                    st.markdown(f'<div style="font-size:0.78rem;color:#ffb700;background:#0a0a0f;border-left:2px solid #ffb700;padding:0.5rem 0.75rem;margin-top:0.4rem">💡 {q.explanation}</div>', unsafe_allow_html=True)

        # Submit / retry
        if not quiz_submitted:
            if st.button("⚡ SUBMIT ANSWERS", type="primary", use_container_width=True):
                for i, q in enumerate(quiz.questions):
                    q.user_answer = user_answers.get(i, "")
                st.session_state["quiz_submitted"] = True
                st.rerun()
        else:
            if st.button("🔄 NEW QUIZ", use_container_width=True):
                st.session_state["current_quiz"] = None
                st.session_state["quiz_submitted"] = False
                st.rerun()


# ── TAB: MIND MAP ─────────────────────────────────────────────────────────────
with tab_mindmap:
    st.markdown('<div class="hud-label">🧠 KNOWLEDGE MIND MAP — INTERACTIVE GRAPH</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        mm_topic = st.text_input("Focus topic", placeholder="e.g. cloud services, security… (blank = full document)", label_visibility="visible", key="mm_topic")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        gen_mm_btn = st.button("🧠 GENERATE MAP", type="primary", use_container_width=True)

    if gen_mm_btn:
        if kb.is_empty():
            st.error("No documents indexed.")
        else:
            with st.spinner("◈ Mapping knowledge graph…"):
                try:
                    mmgen = MindMapGenerator(kb=kb, model=model)
                    doc_ids = [d.doc_id for d in all_docs]
                    mindmap = mmgen.generate(topic=mm_topic, doc_ids=doc_ids)
                    st.session_state["current_mindmap"] = mindmap
                except Exception as e:
                    err = str(e).lower()
                    if "429" in str(e) or "quota" in err or "rate" in err or "exhausted" in err:
                        st.warning("⏳ Rate limit hit (5 req/min on free tier). Wait 60 seconds and try again.")
                    else:
                        st.error(f"Mind map error: {e}")

    mindmap = st.session_state.get("current_mindmap")
    if mindmap and (mindmap.nodes or mindmap.central != "No Data"):
        st.markdown(f'<div style="font-size:0.65rem;color:#6b6870;letter-spacing:0.1em;margin-bottom:0.5rem">◈ {len(mindmap.nodes)} CONCEPTS · {len(mindmap.links)} CONNECTIONS · DRAG TO EXPLORE</div>', unsafe_allow_html=True)
        html_content = render_mindmap_html(mindmap)
        st.components.v1.html(html_content, height=600, scrolling=False)
    else:
        st.markdown("""
        <div class="empty-hud" style="padding:3rem">
            <div class="empty-ring" style="width:60px;height:60px;margin-bottom:1rem"></div>
            <div class="empty-title" style="font-size:1rem">AWAITING SCAN</div>
            <div class="empty-sub">Click GENERATE MAP to build an interactive<br>knowledge graph from your documents</div>
        </div>""", unsafe_allow_html=True)


# ── TAB: DEPLOY GUIDE ────────────────────────────────────────────────────────
with tab_guide:
    st.markdown('<div class="hud-label">🚀 DEPLOYMENT COMMAND CENTER</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="hud-panel">
    <div class="hud-label">STEP 1 — LOCAL SYSTEMS CHECK</div>
    <pre style="background:#000;border:1px solid rgba(255,26,26,0.2);padding:1rem;font-size:0.8rem;color:#ff1a1a;overflow-x:auto">
# Clone & enter
cd jarvis-pka

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\\Scripts\\activate

# Install dependencies (CPU-only, fast)
pip install -r requirements-cpu.txt

# Set your key (or enter it in the sidebar)
export OPENAI_API_KEY="sk-your-key-here"

# Launch JARVIS
streamlit run app.py
# → Open http://localhost:8501
    </pre>
    </div>

    <div class="hud-panel" style="margin-top:1rem">
    <div class="hud-label">STEP 2 — PUSH TO GITHUB</div>
    <pre style="background:#000;border:1px solid rgba(255,26,26,0.2);padding:1rem;font-size:0.8rem;color:#ff1a1a;overflow-x:auto">
# Initialize git repository
git init
git add .
git commit -m "feat: JARVIS PKA — semantic RAG, hybrid search, citations, versioning"

# Create repo at github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/jarvis-pka.git
git branch -M main
git push -u origin main

# Verify CI passes at: github.com/YOUR_USERNAME/jarvis-pka/actions
    </pre>
    </div>

    <div class="hud-panel" style="margin-top:1rem">
    <div class="hud-label">STEP 3 — DEPLOY TO STREAMLIT CLOUD (FREE)</div>
    <pre style="background:#000;border:1px solid rgba(255,26,26,0.2);padding:1rem;font-size:0.8rem;color:#00e5ff;overflow-x:auto">
1. Go to → https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Repository: YOUR_USERNAME/jarvis-pka
5. Branch: main
6. Main file path: app.py
7. Click "Advanced settings" → Secrets:

   OPENAI_API_KEY = "sk-your-key-here"

8. Under "Packages file" set: requirements-cpu.txt
9. Click DEPLOY
   → Live in ~3 minutes at https://YOUR_APP.streamlit.app
    </pre>
    </div>

    <div class="hud-panel-cyan hud-panel" style="margin-top:1rem">
    <div class="hud-label hud-label-cyan">OPTIONAL — HUGGING FACE SPACES (ALSO FREE)</div>
    <pre style="background:#000;border:1px solid rgba(0,229,255,0.2);padding:1rem;font-size:0.8rem;color:#00e5ff;overflow-x:auto">
1. Go to → https://huggingface.co/spaces
2. New Space → SDK: Streamlit → Python 3.11
3. Push your code to the Space's git repo
4. Add OPENAI_API_KEY as a Secret in Settings
5. Live at: https://huggingface.co/spaces/YOUR_USERNAME/jarvis-pka
    </pre>
    </div>
    """, unsafe_allow_html=True)