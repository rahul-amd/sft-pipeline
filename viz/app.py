"""
SFT Pipeline — Visualization Dashboard
=======================================

Entry point for the Streamlit multipage app.

Run:
    streamlit run viz/app.py

Share via Cloudflare Tunnel (no account needed):
    cloudflared tunnel --protocol http2 --url http://localhost:8501
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import plotly.express as px
import streamlit as st

from components.data_loader import get_snapshot, has_stage, load_meta
from components.theme import PLOTLY_COLORS, PLOTLY_LAYOUT, apply_theme

st.set_page_config(
    page_title="SFT Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

st.title("SFT Pipeline Dashboard")

meta = load_meta()

if not (Path(__file__).parent / "data" / "snapshot.parquet").exists():
    st.warning(
        "No snapshot found. Generate one with:\n\n"
        "```bash\npython viz/export.py --run-dir /path/to/run\n```"
    )
    st.stop()

df, meta = get_snapshot()

# ── Stage status badges ───────────────────────────────────────────────────────
st.subheader("Pipeline Status")
stage_defs = [
    ("stage1", "Stage 1", "Collect"),
    ("stage3", "Stage 3", "Cluster"),
    ("stage5", "Stage 5", "Inference"),
    ("stage6", "Stage 6", "Filter"),
]
cols = st.columns(len(stage_defs))
for col, (key, label, desc) in zip(cols, stage_defs):
    done = has_stage(meta, key)
    col.metric(label=f"{label} — {desc}", value="✅ Done" if done else "⏳ Pending")

st.markdown("<br>", unsafe_allow_html=True)

# ── Snapshot summary ──────────────────────────────────────────────────────────
st.subheader("Snapshot Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Prompts in snapshot", f"{len(df):,}")
total = meta.get("total_prompts")
c2.metric("Total collected", f"{total:,}" if total else "—")
c3.metric("Sample size", f"{meta.get('sample_size', len(df)):,}")
exported_at = meta.get("exported_at", "")
c4.metric("Last export", exported_at[:19].replace("T", " ") if exported_at else "—")

st.markdown("<br>", unsafe_allow_html=True)

# ── Quick domain breakdown ────────────────────────────────────────────────────
st.subheader("Domain Breakdown")
domain_counts = df["domain"].value_counts().reset_index()
domain_counts.columns = ["domain", "count"]

fig = px.bar(
    domain_counts.sort_values("count", ascending=True),
    x="count", y="domain", orientation="h",
    color="domain",
    color_discrete_sequence=PLOTLY_COLORS,
    labels={"count": "Prompts", "domain": ""},
    height=max(200, len(domain_counts) * 44),
)
fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    "<p style='color:#475569;font-size:0.8rem;margin-top:1rem;'>"
    "Navigate using the sidebar → Stats · Prompts · Clusters · Answers</p>",
    unsafe_allow_html=True,
)
