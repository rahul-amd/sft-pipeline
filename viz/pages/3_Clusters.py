"""Cluster Analysis — size distribution, domain breakdown, top clusters."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.data_loader import get_snapshot
from components.theme import PLOTLY_COLORS, PLOTLY_LAYOUT, apply_theme

st.set_page_config(page_title="Clusters — SFT Pipeline", layout="wide")
apply_theme()
st.title("Cluster Analysis")

df, meta = get_snapshot()
stats         = meta.get("stats", {})
cluster_stats = stats.get("cluster_stats", {})

if not cluster_stats:
    st.info(
        "Cluster statistics are not available in this snapshot.\n\n"
        "Re-run the export after Stage 3 completes:\n\n"
        "```bash\npython viz/export.py --run-dir /path/to/run\n```"
    )
    st.stop()

n_clusters  = cluster_stats.get("n_clusters", 0)
top_clusters = cluster_stats.get("top_clusters", [])
cpd          = cluster_stats.get("clusters_per_domain", {})
size_hist    = cluster_stats.get("size_histogram", {})
total        = meta.get("total_prompts", len(df))


def _hist_median(bin_edges: list, counts: list) -> float:
    if not counts:
        return 0.0
    n = sum(counts)
    half = n / 2
    cumulative = 0
    for i, count in enumerate(counts):
        cumulative += count
        if cumulative >= half:
            lo = bin_edges[i]
            hi = bin_edges[i + 1]
            excess = cumulative - count
            frac = (half - excess) / count if count else 0
            return lo + frac * (hi - lo)
    return 0.0


sizes_from_top = [c["size"] for c in top_clusters]
max_size    = max(sizes_from_top) if sizes_from_top else 0
median_size = _hist_median(size_hist.get("bin_edges", []), size_hist.get("counts", []))

# ── Metrics ─────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total clusters",          f"{n_clusters:,}")
m2.metric("Median cluster size",     f"{median_size:.0f}")
m3.metric("Largest cluster",         f"{max_size:,}")
m4.metric("Avg prompts / cluster",   f"{total / n_clusters:,.0f}" if n_clusters else "—")
st.markdown("<br>", unsafe_allow_html=True)

# ── Row 1: size histogram + clusters per domain ──────────────────────────────
col_hist, col_domain = st.columns(2)

with col_hist:
    st.subheader("Cluster Size Distribution")
    bin_edges   = size_hist.get("bin_edges", [])
    hist_counts = size_hist.get("counts", [])
    if bin_edges and hist_counts:
        bin_labels = [f"{bin_edges[i]}–{bin_edges[i+1]-1}" for i in range(len(hist_counts))]
        fig = go.Figure(go.Bar(
            x=bin_labels, y=hist_counts,
            marker_color=PLOTLY_COLORS[0],
            hovertemplate="Size %{x}<br>%{y:,} clusters<extra></extra>",
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT, height=320, showlegend=False,
            margin=dict(l=0, r=0, t=10, b=10),
            xaxis_title="Cluster size (# prompts)",
            yaxis_title="Number of clusters",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Size histogram not available.")

with col_domain:
    st.subheader("Clusters by Dominant Domain")
    if cpd:
        cpd_sorted = sorted(cpd.items(), key=lambda x: -x[1])
        labels = [k for k, _ in cpd_sorted]
        values = [v for _, v in cpd_sorted]
        fig = go.Figure(go.Bar(
            x=values[::-1], y=labels[::-1], orientation="h",
            marker_color=PLOTLY_COLORS[1],
            hovertemplate="%{y}: %{x:,} clusters<extra></extra>",
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT, height=max(300, len(labels) * 34), showlegend=False,
            margin=dict(l=0, r=0, t=10, b=10),
            xaxis_title="Clusters",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Domain breakdown not available.")

# ── Row 2: top clusters table ────────────────────────────────────────────────
if top_clusters:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Top 50 Clusters by Size")
    top_df = pd.DataFrame(top_clusters).rename(columns={
        "cluster_id": "Cluster ID",
        "size":       "Prompts",
        "domain":     "Dominant Domain",
    })
    top_df["% of total"] = (top_df["Prompts"] / total * 100).round(3)
    st.dataframe(
        top_df, use_container_width=True, hide_index=True,
        column_config={
            "Cluster ID":      st.column_config.NumberColumn(format="%d"),
            "Prompts":         st.column_config.NumberColumn(format="%d"),
            "% of total":      st.column_config.NumberColumn(format="%.3f%%"),
            "Dominant Domain": st.column_config.TextColumn(),
        },
    )
