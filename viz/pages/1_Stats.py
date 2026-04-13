"""Stats Dashboard — domain, source, difficulty, dedup rate, shard progress."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from components.data_loader import get_snapshot, has_stage
from components.theme import PLOTLY_COLORS, PLOTLY_LAYOUT, apply_theme

st.set_page_config(page_title="Stats — SFT Pipeline", layout="wide")
apply_theme()
st.title("Stats")

df, meta = get_snapshot()

total = meta.get("total_prompts", len(df))
sample = meta.get("sample_size", len(df))
st.caption(
    f"Showing a {sample:,}-prompt sample out of {total:,} total collected. "
    f"Distributions below reflect the sample."
)

st.markdown("<br>", unsafe_allow_html=True)

_layout = dict(PLOTLY_LAYOUT, showlegend=False, margin=dict(l=0, r=0, t=10, b=10))

# ── Row 1: domain + source ────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Domain Distribution")
    dc = df["domain"].value_counts().reset_index()
    dc.columns = ["domain", "count"]
    fig = px.bar(
        dc.sort_values("count", ascending=True),
        x="count", y="domain", orientation="h",
        color="domain",
        color_discrete_sequence=PLOTLY_COLORS,
        labels={"count": "Prompts", "domain": ""},
    )
    fig.update_layout(**_layout)
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Top Sources")
    sc = df["source"].value_counts().head(20).reset_index()
    sc.columns = ["source", "count"]
    fig = px.bar(
        sc.sort_values("count", ascending=True),
        x="count", y="source", orientation="h",
        color_discrete_sequence=[PLOTLY_COLORS[1]],
        labels={"count": "Prompts", "source": ""},
    )
    fig.update_layout(**_layout)
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
            color_discrete_map={"easy": "#10b981", "medium": "#f59e0b", "hard": "#ef4444"},
            labels={"count": "Prompts", "difficulty": ""},
        )
        fig.update_layout(**_layout)
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

# ── Row 3: language + topics (annotation fields) ─────────────────────────────
has_annotation = (
    "language" in df.columns
    and df["language"].notna().any()
    and df["language"].nunique() > 1
)
if has_annotation:
    st.markdown("<br>", unsafe_allow_html=True)
    col_lang, col_topics = st.columns(2)

    with col_lang:
        st.subheader("Language Distribution")
        lc = df["language"].value_counts().head(20).reset_index()
        lc.columns = ["language", "count"]
        fig = px.bar(
            lc.sort_values("count", ascending=True),
            x="count", y="language", orientation="h",
            color_discrete_sequence=[PLOTLY_COLORS[2]],
            labels={"count": "Prompts", "language": ""},
        )
        fig.update_layout(**_layout)
        st.plotly_chart(fig, use_container_width=True)

    with col_topics:
        st.subheader("Top Topics")
        if "topics" in df.columns and df["topics"].str.strip().any():
            all_topics = (
                df["topics"]
                .dropna()
                .str.split(", ")
                .explode()
                .str.strip()
                .loc[lambda s: s != ""]
            )
            tc = all_topics.value_counts().head(20).reset_index()
            tc.columns = ["topic", "count"]
            fig = px.bar(
                tc.sort_values("count", ascending=True),
                x="count", y="topic", orientation="h",
                color_discrete_sequence=[PLOTLY_COLORS[3]],
                labels={"count": "Prompts", "topic": ""},
            )
            fig.update_layout(**_layout)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Topics available after LLM annotation completes.")

# ── Row 4: prompts per source (full table) ────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("All sources breakdown"):
    full_sc = df["source"].value_counts().reset_index()
    full_sc.columns = ["source", "count"]
    full_sc["% of sample"] = (full_sc["count"] / len(df) * 100).round(1)
    st.dataframe(full_sc, use_container_width=True, hide_index=True)
