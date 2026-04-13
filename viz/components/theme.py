"""
Shared visual theme for the SFT Pipeline viz app.

Call apply_theme() at the top of every page (after st.set_page_config).
"""
from __future__ import annotations
import streamlit as st

# ── Plotly colour sequence ────────────────────────────────────────────────────
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

# NOTE: no xaxis/yaxis keys here — those conflict if a caller also passes
# xaxis=... directly to update_layout(). Apply axis styles with
# fig.update_xaxes() / fig.update_yaxes() separately when needed.
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0d1224",
    plot_bgcolor="#0d1224",
    font=dict(family="Inter, sans-serif", color="#cbd5e1"),
    colorway=PLOTLY_COLORS,
)

_CSS = """
<style>
/* ── Google Fonts ──────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Hide Streamlit top header bar ────────────────────────────────────────── */
header[data-testid="stHeader"] {
    background: #070b14 !important;
    border-bottom: 1px solid #1e293b !important;
}
/* Hamburger / deploy / manage buttons */
header[data-testid="stHeader"] button,
header[data-testid="stHeader"] a {
    color: #475569 !important;
}
header[data-testid="stHeader"] svg {
    fill: #475569 !important;
}

/* ── Global ────────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: #070b14;
}

/* ── Main content padding ──────────────────────────────────────────────────── */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
}

/* ── Sidebar ───────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0d1224 !important;
    border-right: 1px solid #1e293b !important;
}

/* Sidebar nav links (multipage) */
[data-testid="stSidebarNav"] a,
[data-testid="stSidebarNavItems"] a,
[data-testid="stSidebarNavLink"],
[data-testid="stSidebar"] nav a {
    color: #64748b !important;
    text-decoration: none !important;
    font-size: 0.875rem !important;
    font-weight: 400 !important;
    border-radius: 6px !important;
    transition: color 0.15s, background 0.15s !important;
}
[data-testid="stSidebarNav"] a:hover,
[data-testid="stSidebarNavItems"] a:hover,
[data-testid="stSidebarNavLink"]:hover {
    color: #e2e8f0 !important;
    background: rgba(99,102,241,0.1) !important;
}
/* Active page link */
[data-testid="stSidebarNav"] a[aria-current="page"],
[data-testid="stSidebarNavItems"] a[aria-current="page"],
[data-testid="stSidebarNavLink"][aria-current="page"] {
    color: #818cf8 !important;
    background: rgba(99,102,241,0.12) !important;
    font-weight: 500 !important;
}

/* Sidebar text / labels */
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stCheckbox label {
    color: #94a3b8 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #e2e8f0 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    /* cancel h1 gradient inside sidebar */
    background: none !important;
    -webkit-background-clip: unset !important;
    -webkit-text-fill-color: #e2e8f0 !important;
    background-clip: unset !important;
}
[data-testid="stSidebar"] hr {
    border-color: #1e293b !important;
}

/* ── Page title (h1 gradient) ──────────────────────────────────────────────── */
.block-container h1,
[data-testid="stHeadingWithActionElements"] h1 {
    background: linear-gradient(90deg, #818cf8 0%, #06b6d4 55%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    padding-bottom: 0.25rem;
}

/* ── Section subheaders ────────────────────────────────────────────────────── */
.block-container h2 {
    color: #94a3b8 !important;
    font-weight: 500 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 0.5rem;
    margin-top: 1.5rem !important;
    margin-bottom: 0.75rem !important;
}
.block-container h3 {
    color: #64748b !important;
    font-weight: 500 !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
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
[data-testid="stMetricLabel"] p {
    color: #64748b !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}

/* ── Captions ──────────────────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] p,
.stCaption p {
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
    outline: none !important;
}

/* ── Text inputs ───────────────────────────────────────────────────────────── */
input[type="text"], input[type="search"] {
    background: #0d1224 !important;
    border: 1px solid #1e293b !important;
    border-radius: 6px !important;
    color: #e2e8f0 !important;
}
input[type="text"]:focus, input[type="search"]:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}

/* ── Multiselect tags ──────────────────────────────────────────────────────── */
[data-baseweb="tag"] {
    background: #1e1b4b !important;
    border: 1px solid #3730a3 !important;
    border-radius: 4px !important;
}
[data-baseweb="tag"] span {
    color: #a5b4fc !important;
    font-size: 0.78rem !important;
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
    padding: 0.6rem 0.75rem !important;
}
[data-testid="stExpander"] summary:hover {
    color: #e2e8f0 !important;
}

/* ── Alert / info boxes ────────────────────────────────────────────────────── */
[data-testid="stAlert"] > div {
    border-radius: 8px !important;
}

/* ── Code ──────────────────────────────────────────────────────────────────── */
code {
    background: #1e293b !important;
    border-radius: 4px !important;
    color: #a5b4fc !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    padding: 1px 5px !important;
}
pre {
    background: #0d1224 !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
pre code {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* ── Number input ──────────────────────────────────────────────────────────── */
input[type="number"] {
    background: #0d1224 !important;
    border: 1px solid #1e293b !important;
    color: #e2e8f0 !important;
    border-radius: 6px !important;
}

/* ── Plotly chart wrapper ──────────────────────────────────────────────────── */
[data-testid="stPlotlyChart"] {
    border-radius: 10px;
    overflow: hidden;
    background: #0d1224;
    border: 1px solid #1e293b;
}

/* ── Scrollbars ────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }
</style>
"""


def apply_theme() -> None:
    """Inject the dark CSS theme. Call once per page, after st.set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)
