"""Cluster Scatter — UMAP 2D projection coloured by domain or cluster."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import plotly.express as px
import streamlit as st

from components.data_loader import get_snapshot, has_stage
from components.filters import render_filters

st.set_page_config(page_title="Clusters — SFT Pipeline", layout="wide")
st.title("🔵 Cluster Explorer")

df, meta = get_snapshot()

if not has_stage(meta, "umap") or "umap_x" not in df.columns or df["umap_x"].isna().all():
    st.info(
        "UMAP coordinates are not available in this snapshot.\n\n"
        "They are computed automatically by `export.py` once Stage 3 embeddings exist. "
        "Re-run the export after Stage 3 completes:\n\n"
        "```bash\npython viz/export.py --run-dir /path/to/run\n```"
    )
    st.stop()

plot_df = df.dropna(subset=["umap_x", "umap_y"]).copy()
filtered = render_filters(plot_df, show_difficulty=True, show_search=False)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.subheader("Plot Controls")
    color_by = st.radio("Colour by", ["domain", "cluster_id", "difficulty"], index=0)
    point_size = st.slider("Point size", 1, 8, 3)
    opacity = st.slider("Opacity", 0.1, 1.0, 0.6, step=0.05)
    max_points = st.slider("Max points shown", 5_000, min(50_000, len(filtered)), min(20_000, len(filtered)), step=5_000)

# Subsample for rendering performance
if len(filtered) > max_points:
    plot_sample = filtered.sample(max_points, random_state=42)
    st.caption(f"Showing {max_points:,} of {len(filtered):,} filtered points (random subsample).")
else:
    plot_sample = filtered
    st.caption(f"Showing all {len(filtered):,} filtered points.")

# Build hover text
plot_sample = plot_sample.copy()
plot_sample["hover"] = (
    plot_sample["source"] + " | " + plot_sample["domain"]
    + "<br>" + plot_sample["prompt"].str.slice(0, 120) + "…"
)

color_col = color_by if color_by in plot_sample.columns and plot_sample[color_by].notna().any() else "domain"

fig = px.scatter(
    plot_sample,
    x="umap_x",
    y="umap_y",
    color=color_col,
    hover_name="hover",
    hover_data={"umap_x": False, "umap_y": False, "hover": False},
    opacity=opacity,
    labels={"umap_x": "", "umap_y": ""},
)
fig.update_traces(marker=dict(size=point_size))
fig.update_layout(
    height=650,
    legend=dict(itemsizing="constant"),
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
)

event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")

# Show prompt for clicked point
if event and hasattr(event, "selection") and event.selection.get("points"):
    pt = event.selection["points"][0]
    pt_idx = pt.get("point_index")
    if pt_idx is not None:
        row = plot_sample.iloc[pt_idx]
        st.divider()
        st.subheader("Selected Point")
        c1, c2 = st.columns([3, 1])
        with c1:
            st.text_area("Prompt", value=row["prompt"], height=200, disabled=True, label_visibility="visible")
        with c2:
            st.markdown(f"**Source:** {row['source']}")
            st.markdown(f"**Domain:** {row['domain']}")
            if row.get("difficulty"):
                st.markdown(f"**Difficulty:** {row['difficulty']}")
            if row.get("cluster_id") is not None:
                st.markdown(f"**Cluster:** {int(row['cluster_id'])}")
