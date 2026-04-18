"""
Shared visual theme for the SFT Pipeline viz app.

Call apply_theme() at the top of every page (after st.set_page_config).

The base dark theme (backgrounds, input widgets, dataframes, text colours) is
handled by viz/.streamlit/config.toml so that ALL Streamlit components—including
st.dataframe and multiselect—are styled consistently without fighting the shadow
DOM.  This module adds polish on top: fonts, metric cards, chart containers, etc.
"""
from __future__ import annotations
import streamlit as st

# ── Plotly colour sequence ────────────────────────────────────────────────────
PLOTLY_COLORS = [
    "#818cf8",  # indigo-400
    "#22d3ee",  # cyan-400
    "#34d399",  # emerald-400
    "#fbbf24",  # amber-400
    "#f87171",  # red-400
    "#a78bfa",  # violet-400
    "#f472b6",  # pink-400
    "#2dd4bf",  # teal-400
    "#fb923c",  # orange-400
    "#a3e635",  # lime-400
]

# NOTE: no xaxis/yaxis keys here — those conflict if a caller also passes
# xaxis=... directly to update_layout(). Apply axis styles with
# fig.update_xaxes() / fig.update_yaxes() separately when needed.
#
# paper_bgcolor / plot_bgcolor must match config.toml backgroundColor so
# charts don't have a different background from the page.
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0a0f1e",
    plot_bgcolor="#0a0f1e",
    font=dict(family="Inter, sans-serif", color="#cbd5e1"),
    colorway=PLOTLY_COLORS,
)

_CSS = """
<style>
/* ── Google Fonts ──────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ── Page title — indigo→cyan gradient ────────────────────────────────────── */
.block-container h1,
[data-testid="stHeadingWithActionElements"] h1 {
    background: linear-gradient(90deg, #818cf8 0%, #22d3ee 60%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    padding-bottom: 0.25rem;
}

/* ── Subheaders ────────────────────────────────────────────────────────────── */
.block-container h2 {
    color: #64748b !important;
    font-weight: 500 !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 0.4rem;
    margin-top: 1.5rem !important;
    margin-bottom: 0.6rem !important;
}
.block-container h3 {
    color: #475569 !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Sidebar subheaders — cancel gradient + shrink ─────────────────────────── */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    background: none !important;
    -webkit-text-fill-color: #94a3b8 !important;
    font-size: 0.65rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 0.35rem;
    margin-top: 0.75rem !important;
}

/* ── Sidebar nav links ─────────────────────────────────────────────────────── */
[data-testid="stSidebarNav"] a,
[data-testid="stSidebarNavItems"] a,
[data-testid="stSidebarNavLink"] {
    color: #64748b !important;
    font-size: 0.875rem !important;
    font-weight: 400 !important;
    border-radius: 6px !important;
    transition: color 0.15s, background 0.15s !important;
}
[data-testid="stSidebarNav"] a:hover,
[data-testid="stSidebarNavItems"] a:hover,
[data-testid="stSidebarNavLink"]:hover {
    color: #e2e8f0 !important;
    background: rgba(99,102,241,0.12) !important;
}
[data-testid="stSidebarNav"] a[aria-current="page"],
[data-testid="stSidebarNavItems"] a[aria-current="page"],
[data-testid="stSidebarNavLink"][aria-current="page"] {
    color: #818cf8 !important;
    background: rgba(99,102,241,0.15) !important;
    font-weight: 600 !important;
}

/* ── Metric cards ──────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 0.9rem 1.1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stMetric"]:hover {
    border-color: #6366f1;
    box-shadow: 0 0 0 1px rgba(99,102,241,0.2);
}
[data-testid="stMetricLabel"] p {
    font-size: 0.68rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    color: #e2e8f0 !important;
}

/* ── Plotly chart container ────────────────────────────────────────────────── */
[data-testid="stPlotlyChart"] {
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #1e293b;
}

/* ── Dividers ──────────────────────────────────────────────────────────────── */
hr { border-color: #1e293b !important; }

/* ── Captions ──────────────────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] p, .stCaption p {
    font-size: 0.78rem !important;
    color: #475569 !important;
}

/* ── Code ──────────────────────────────────────────────────────────────────── */
code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}
pre {
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Text area (prompt viewer) ─────────────────────────────────────────────── */
textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.6 !important;
    border-radius: 8px !important;
}

/* ── Expanders ─────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.85rem !important;
    padding: 0.55rem 0.75rem !important;
}

/* ── Scrollbars ────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }

/* ── Main content padding ──────────────────────────────────────────────────── */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
}
</style>
"""


def apply_theme() -> None:
    """Inject polish CSS. Call once per page, after st.set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)
