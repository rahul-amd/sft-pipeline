"""Cluster Scatter — UMAP 2D projection coloured by domain or cluster."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import plotly.express as px
import streamlit as st

from components.data_loader import get_snapshot, has_stage
from components.filters import render_filters
from components.theme import PLOTLY_COLORS, PLOTLY_LAYOUT, apply_theme

st.set_page_config(page_title="Clusters — SFT Pipeline", layout="wide")
apply_theme()
st.title("Cluster Explorer")

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

has_difficulty = "difficulty" in plot_sample.columns and plot_sample["difficulty"].notna().any()
has_language   = "language"   in plot_sample.columns and plot_sample["language"].notna().any()
has_topics     = "topics"     in plot_sample.columns and plot_sample["topics"].str.strip().any()
has_summary    = "summary"    in plot_sample.columns and plot_sample["summary"].str.strip().any()

# Line 1: source | domain  [difficulty · language]
header = plot_sample["source"] + " | " + plot_sample["domain"]
if has_difficulty and has_language:
    header = header + "  [" + plot_sample["difficulty"].fillna("") + " · " + plot_sample["language"].fillna("") + "]"
elif has_difficulty:
    header = header + "  [" + plot_sample["difficulty"].fillna("") + "]"
elif has_language:
    header = header + "  [" + plot_sample["language"].fillna("") + "]"

hover = header

# Line 2: topics (if present)
if has_topics:
    hover = hover + "<br><i>" + plot_sample["topics"].fillna("").str.slice(0, 80) + "</i>"

# Line 3: summary or truncated prompt
if has_summary:
    snippet = plot_sample["summary"].fillna("").str.strip()
    no_summary = snippet == ""
    snippet[no_summary] = plot_sample.loc[no_summary, "prompt"].str.slice(0, 120) + "…"
    hover = hover + "<br>" + snippet
else:
    hover = hover + "<br>" + plot_sample["prompt"].str.slice(0, 120) + "…"

plot_sample["hover"] = hover

color_col = color_by if color_by in plot_sample.columns and plot_sample[color_by].notna().any() else "domain"

fig = px.scatter(
    plot_sample,
    x="umap_x",
    y="umap_y",
    color=color_col,
    color_discrete_sequence=PLOTLY_COLORS,
    hover_name="hover",
    hover_data={"umap_x": False, "umap_y": False, "hover": False},
    opacity=opacity,
    labels={"umap_x": "", "umap_y": ""},
)
fig.update_traces(marker=dict(size=point_size))
fig.update_layout(
    **PLOTLY_LAYOUT,
    height=660,
    legend=dict(
        itemsizing="constant",
        bgcolor="rgba(13,18,36,0.8)",
        bordercolor="#1e293b",
        borderwidth=1,
        font=dict(size=12),
    ),
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    hoverlabel=dict(
        bgcolor="#0d1224",
        bordercolor="#334155",
        font=dict(family="Inter, sans-serif", size=12, color="#e2e8f0"),
    ),
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
            st.markdown(
                f"""
                <div style="display:flex;flex-direction:column;gap:0.5rem;padding-top:0.25rem;">
                  <div>
                    <span style="color:#475569;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;">Source</span><br>
                    <span style="color:#e2e8f0;font-size:0.85rem;">{row['source']}</span>
                  </div>
                  <div>
                    <span style="color:#475569;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;">Domain</span><br>
                    <span style="background:#1e1b4b;color:#a5b4fc;padding:2px 8px;border-radius:4px;font-size:0.8rem;">{row['domain']}</span>
                  </div>
                """,
                unsafe_allow_html=True,
            )
            if row.get("difficulty"):
                colour = {"easy": "#064e3b", "medium": "#451a03", "hard": "#450a0a"}.get(row["difficulty"], "#1e293b")
                text_c = {"easy": "#6ee7b7", "medium": "#fcd34d", "hard": "#fca5a5"}.get(row["difficulty"], "#94a3b8")
                st.markdown(
                    f'<div><span style="color:#475569;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;">Difficulty</span><br>'
                    f'<span style="background:{colour};color:{text_c};padding:2px 8px;border-radius:4px;font-size:0.8rem;">{row["difficulty"]}</span></div>',
                    unsafe_allow_html=True,
                )
            if has_language and row.get("language"):
                st.markdown(
                    f'<div><span style="color:#475569;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;">Language</span><br>'
                    f'<span style="color:#e2e8f0;font-size:0.85rem;">{row["language"]}</span></div>',
                    unsafe_allow_html=True,
                )
            if has_topics and row.get("topics", "").strip():
                tags = "".join(
                    f'<span style="background:#0c1a2e;color:#67e8f9;border:1px solid #164e63;'
                    f'padding:2px 6px;border-radius:4px;font-size:0.75rem;margin:2px 2px 2px 0;display:inline-block;">'
                    f'{t.strip()}</span>'
                    for t in row["topics"].split(",") if t.strip()
                )
                st.markdown(
                    f'<div><span style="color:#475569;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;">Topics</span><br>'
                    f'<div style="margin-top:4px;">{tags}</div></div>',
                    unsafe_allow_html=True,
                )
            if has_summary and row.get("summary", "").strip():
                st.markdown(
                    f'<div style="padding:0.5rem 0.7rem;background:#0d1224;border-left:3px solid #6366f1;border-radius:0 6px 6px 0;">'
                    f'<span style="color:#475569;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;">Summary</span><br>'
                    f'<span style="color:#94a3b8;font-size:0.82rem;font-style:italic;">{row["summary"]}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            if row.get("cluster_id") is not None:
                st.markdown(
                    f'<div><span style="color:#475569;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;">Cluster</span><br>'
                    f'<span style="color:#e2e8f0;font-size:0.85rem;">#{int(row["cluster_id"])}</span></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
