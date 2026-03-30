"""Stats Dashboard — domain, source, difficulty, dedup rate, shard progress."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from components.data_loader import get_snapshot, has_stage

st.set_page_config(page_title="Stats — SFT Pipeline", layout="wide")
st.title("📊 Stats")

df, meta = get_snapshot()

total = meta.get("total_prompts", len(df))
sample = meta.get("sample_size", len(df))
st.caption(
    f"Showing a {sample:,}-prompt sample out of {total:,} total collected. "
    f"Distributions below reflect the sample."
)

# ── Row 1: domain + source ────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Domain Distribution")
    dc = df["domain"].value_counts().reset_index()
    dc.columns = ["domain", "count"]
    fig = px.bar(
        dc.sort_values("count", ascending=True),
        x="count", y="domain", orientation="h",
        color="domain", labels={"count": "Prompts", "domain": ""},
    )
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Top Sources")
    sc = df["source"].value_counts().head(20).reset_index()
    sc.columns = ["source", "count"]
    fig = px.bar(
        sc.sort_values("count", ascending=True),
        x="count", y="source", orientation="h",
        labels={"count": "Prompts", "source": ""},
    )
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ── Row 2: difficulty + dedup ─────────────────────────────────────────────────
col_diff, col_dedup = st.columns(2)

with col_diff:
    st.subheader("Difficulty Distribution")
    if has_stage(meta, "stage3") and "difficulty" in df.columns and df["difficulty"].notna().any():
        order = ["easy", "medium", "hard"]
        difc = df["difficulty"].value_counts().reset_index()
        difc.columns = ["difficulty", "count"]
        difc["difficulty"] = pd.Categorical(difc["difficulty"], categories=order, ordered=True)
        difc = difc.sort_values("difficulty")
        fig = px.bar(
            difc, x="difficulty", y="count",
            color="difficulty",
            color_discrete_map={"easy": "#4caf50", "medium": "#ff9800", "hard": "#f44336"},
            labels={"count": "Prompts", "difficulty": ""},
        )
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Available after Stage 3 (clustering) completes.")

with col_dedup:
    st.subheader("Collection Stats")
    written = meta.get("sample_size", len(df))
    total_seen = meta.get("total_prompts")
    if total_seen and total_seen > written:
        dedup_pct = 100.0 * (1 - written / total_seen)
        st.metric("Total prompts seen", f"{total_seen:,}")
        st.metric("Written (after dedup)", f"{written:,}")
        st.metric("Dedup rate", f"{dedup_pct:.1f}%")
    else:
        st.metric("Prompts in snapshot", f"{len(df):,}")

# ── Row 3: prompts per source (full table) ────────────────────────────────────
with st.expander("All sources breakdown"):
    full_sc = df["source"].value_counts().reset_index()
    full_sc.columns = ["source", "count"]
    full_sc["% of sample"] = (full_sc["count"] / len(df) * 100).round(1)
    st.dataframe(full_sc, use_container_width=True, hide_index=True)
