"""
Shared visual theme for the SFT Pipeline viz app.

Call apply_theme() at the top of every page (after st.set_page_config).
"""
from __future__ import annotations
import streamlit as st

# ── Plotly colour sequence (domain-aware) ─────────────────────────────────────
PLOTLY_COLORS = [
    "#6366f1",  # indigo
    "#06b6d4",  # cyan
    "#10b981",  # emerald
    "#f59e0b",  # amber
    "#ef4444",  # red
    "#8b5cf6",  # violet
    "#ec4899",  # pink
    "#14b8a6",  # teal
    "#f97316",  # orange
    "#a3e635",  # lime
]

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#cbd5e1"),
    colorway=PLOTLY_COLORS,
    xaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
    yaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
)

_CSS = """
<style>
/* ── Google Fonts ──────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global ────────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #070b14;
}

/* ── Sidebar ───────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0d1224 !important;
    border-right: 1px solid #1e293b !important;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stCheckbox label {
    color: #94a3b8 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stSubheader {
    color: #e2e8f0 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}
[data-testid="stSidebar"] hr {
    border-color: #1e293b !important;
}

/* ── Headers ───────────────────────────────────────────────────────────────── */
h1 {
    background: linear-gradient(90deg, #818cf8 0%, #06b6d4 60%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700 !important;
    font-size: 2rem !important;
    letter-spacing: -0.02em;
    padding-bottom: 0.25rem;
}
h2 {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    font-size: 1.15rem !important;
    letter-spacing: -0.01em;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 0.4rem;
    margin-top: 0.5rem !important;
}
h3 {
    color: #94a3b8 !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Metric cards ──────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #0d1224;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1rem 1.25rem !important;
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover {
    border-color: #334155;
}
[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
[data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: -0.02em;
}
[data-testid="stMetricDelta"] {
    color: #10b981 !important;
}

/* ── Captions ──────────────────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"], .stCaption, small {
    color: #475569 !important;
    font-size: 0.78rem !important;
}

/* ── Dividers ──────────────────────────────────────────────────────────────── */
hr {
    border-color: #1e293b !important;
}

/* ── Dataframe ─────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #1e293b;
    border-radius: 8px;
    overflow: hidden;
}
[data-testid="stDataFrame"] iframe {
    border-radius: 8px;
}

/* ── Text area ─────────────────────────────────────────────────────────────── */
textarea {
    background: #0d1224 !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    color: #cbd5e1 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.6 !important;
}
textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}

/* ── Inputs ────────────────────────────────────────────────────────────────── */
input[type="text"], input[type="search"] {
    background: #0d1224 !important;
    border: 1px solid #1e293b !important;
    border-radius: 6px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
input[type="text"]:focus, input[type="search"]:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}

/* ── Multiselect ───────────────────────────────────────────────────────────── */
[data-baseweb="select"] {
    background: #0d1224 !important;
}
[data-baseweb="tag"] {
    background: #1e1b4b !important;
    border: 1px solid #4338ca !important;
    border-radius: 4px !important;
}
[data-baseweb="tag"] span {
    color: #a5b4fc !important;
    font-size: 0.75rem !important;
}

/* ── Sliders ───────────────────────────────────────────────────────────────── */
[data-testid="stSlider"] [role="slider"] {
    background: #6366f1 !important;
}

/* ── Radio buttons ─────────────────────────────────────────────────────────── */
[data-testid="stRadio"] label[data-selected="true"] {
    color: #818cf8 !important;
}

/* ── Info / warning / error boxes ─────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    border-width: 1px !important;
}
.stAlert[data-baseweb="notification"] {
    background: #0d1224 !important;
}

/* Info */
div[data-testid="stAlert"] > div:first-child {
    background: rgba(99, 102, 241, 0.08) !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    border-radius: 8px !important;
    color: #a5b4fc !important;
}
/* Warning */
div[data-testid="stAlert"][kind="warning"] > div:first-child,
div.stWarning > div {
    background: rgba(245, 158, 11, 0.08) !important;
    border: 1px solid rgba(245, 158, 11, 0.25) !important;
    border-radius: 8px !important;
    color: #fcd34d !important;
}

/* ── Expanders ─────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    background: #0d1224 !important;
}
[data-testid="stExpander"] summary {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
}

/* ── Code blocks ───────────────────────────────────────────────────────────── */
code, pre {
    background: #0d1224 !important;
    border: 1px solid #1e293b !important;
    border-radius: 6px !important;
    color: #a5b4fc !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ── Plotly chart containers ───────────────────────────────────────────────── */
[data-testid="stPlotlyChart"] {
    border-radius: 10px;
    overflow: hidden;
}

/* ── Scrollbars ────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #070b14; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }

/* ── Main content padding ──────────────────────────────────────────────────── */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}

/* ── Subheader override ────────────────────────────────────────────────────── */
[data-testid="stHeading"] {
    color: #e2e8f0;
}
</style>
"""


def apply_theme() -> None:
    """Inject the dark CSS theme. Call once per page, after st.set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)


def plotly_defaults(fig, height: int | None = None) -> None:
    """Apply shared dark Plotly layout to an existing figure."""
    updates = dict(PLOTLY_LAYOUT)
    if height is not None:
        updates["height"] = height
    fig.update_layout(**updates)
