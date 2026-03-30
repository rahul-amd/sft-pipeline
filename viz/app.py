"""
SFT Pipeline — Visualization Dashboard
=======================================

Entry point for the Streamlit multipage app.

Run:
    streamlit run viz/app.py

Share via Cloudflare Tunnel (no account needed):
    cloudflared tunnel --url http://localhost:8501
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make viz/ importable as a package root so pages can import components.
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from components.data_loader import get_snapshot, has_stage, load_meta

st.set_page_config(
    page_title="SFT Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔬 SFT Pipeline Dashboard")

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

# ── Snapshot summary ──────────────────────────────────────────────────────────
st.subheader("Snapshot Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Prompts in snapshot", f"{len(df):,}")
total = meta.get("total_prompts")
c2.metric("Total collected", f"{total:,}" if total else "—")
c3.metric("Sample size", f"{meta.get('sample_size', len(df)):,}")
exported_at = meta.get("exported_at", "")
c4.metric("Last export", exported_at[:19].replace("T", " ") if exported_at else "—")

# ── Quick domain breakdown ────────────────────────────────────────────────────
st.subheader("Domain Breakdown")
domain_counts = df["domain"].value_counts().reset_index()
domain_counts.columns = ["domain", "count"]

import plotly.express as px
fig = px.bar(
    domain_counts.sort_values("count", ascending=True),
    x="count", y="domain", orientation="h",
    color="domain",
    labels={"count": "Prompts", "domain": ""},
    height=max(200, len(domain_counts) * 40),
)
fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, use_container_width=True)

st.info("Use the **sidebar** to navigate to Stats, Prompts, Clusters, or Answers.")
