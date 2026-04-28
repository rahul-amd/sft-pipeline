"""Stats Dashboard — full-data distributions, cross-tabs, topics."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.data_loader import get_snapshot, has_stage
from components.theme import PLOTLY_COLORS, PLOTLY_LAYOUT, apply_theme

st.set_page_config(page_title="Stats — SFT Pipeline", layout="wide")
apply_theme()
st.title("Stats")

df, meta = get_snapshot()
stats = meta.get("stats", {})

total  = meta.get("total_prompts", len(df))
sample = meta.get("sample_size",   len(df))

st.caption(
    f"Full-data stats computed over **{total:,}** prompts. "
    f"Prompt browser shows a {sample:,}-record sample."
)

_layout = dict(PLOTLY_LAYOUT, showlegend=False, margin=dict(l=0, r=0, t=10, b=10))


# ---------------------------------------------------------------------------
# Helper: build a bar chart from a {label: count} dict
# ---------------------------------------------------------------------------

def _bar(counts_dict: dict, top_n: int, color_idx: int = 0,
         height: int | None = None, label_name: str = "",
         x_label: str = "Prompts") -> go.Figure:
    items = sorted(counts_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    # reverse so largest is at top
    fig = px.bar(
        x=values[::-1], y=labels[::-1], orientation="h",
        color_discrete_sequence=[PLOTLY_COLORS[color_idx % len(PLOTLY_COLORS)]],
        labels={"x": x_label, "y": label_name},
        height=height or max(220, len(labels) * 34),
    )
    fig.update_layout(**_layout)
    return fig


# ---------------------------------------------------------------------------
# Row 1: Domain + Difficulty
# ---------------------------------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)
col_domain, col_diff = st.columns(2)

with col_domain:
    st.subheader("Domain Distribution")
    dc = stats.get("domain_counts") or df["domain"].value_counts().to_dict()
    st.plotly_chart(_bar(dc, top_n=30, color_idx=0), use_container_width=True)

with col_diff:
    st.subheader("Difficulty Distribution")
    raw_diff = stats.get("difficulty_counts") or (
        df["difficulty"].value_counts().to_dict()
        if "difficulty" in df.columns and df["difficulty"].notna().any()
        else {}
    )
    if raw_diff:
        order = ["easy", "medium", "hard"]
        diff_items = [(d, raw_diff.get(d, 0)) for d in order if raw_diff.get(d, 0) > 0]
        labels = [k for k, _ in diff_items]
        values = [v for _, v in diff_items]
        color_map = {"easy": "#34d399", "medium": "#fbbf24", "hard": "#f87171"}
        fig = px.bar(
            x=labels, y=values,
            color=labels,
            color_discrete_map=color_map,
            labels={"x": "", "y": "Prompts", "color": ""},
            height=280,
        )
        fig.update_layout(**_layout)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Available after Stage 3 completes.")

# ---------------------------------------------------------------------------
# Row 2: Language + Sources
# ---------------------------------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)
col_lang, col_src = st.columns(2)

with col_lang:
    st.subheader("Language Distribution")
    lc = stats.get("language_counts") or (
        df["language"].value_counts().head(20).to_dict()
        if "language" in df.columns and df["language"].notna().any()
        else {}
    )
    if lc and len(lc) > 1:
        st.plotly_chart(_bar(lc, top_n=20, color_idx=2), use_container_width=True)
    else:
        st.info("Language data available after LLM annotation completes.")

with col_src:
    st.subheader("Top Sources")
    sc = stats.get("source_counts") or df["source"].value_counts().head(20).to_dict()
    st.plotly_chart(_bar(sc, top_n=20, color_idx=1), use_container_width=True)

# ---------------------------------------------------------------------------
# Row 3: Cross-tab heatmaps
# ---------------------------------------------------------------------------

dd_mat = stats.get("domain_difficulty_matrix")
dl_mat = stats.get("domain_language_matrix")

if dd_mat or dl_mat:
    st.markdown("<br>", unsafe_allow_html=True)
    col_hm1, col_hm2 = st.columns(2)

    def _heatmap(
        row_labels: list[str],
        col_labels: list[str],
        counts: list[list[int]],
        title: str,
        height: int = 400,
    ) -> go.Figure:
        """Normalised heatmap (row %) with absolute count on hover."""
        z_counts = np.array(counts, dtype=float)
        row_totals = z_counts.sum(axis=1, keepdims=True)
        z_norm = np.where(row_totals > 0, z_counts / row_totals, 0.0)

        text = [[f"{v:.0%}" for v in row] for row in z_norm]

        fig = go.Figure(go.Heatmap(
            z=z_norm,
            x=col_labels,
            y=row_labels,
            customdata=z_counts.astype(int),
            colorscale=[[0, "#0a0f1e"], [0.5, "#4338ca"], [1, "#818cf8"]],
            hovertemplate=(
                "<b>%{y}</b> × <b>%{x}</b><br>"
                "%{customdata:,} prompts (%{z:.1%})<extra></extra>"
            ),
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorbar=dict(tickformat=".0%", thickness=12, len=0.8),
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=height,
            margin=dict(l=0, r=0, t=10, b=10),
            xaxis=dict(side="bottom"),
        )
        return fig

    with col_hm1:
        st.subheader("Domain × Difficulty")
        if dd_mat:
            n_domains = len(dd_mat["domains"])
            fig = _heatmap(
                dd_mat["domains"], dd_mat["difficulties"], dd_mat["counts"],
                "Domain × Difficulty",
                height=max(300, n_domains * 28 + 80),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Available after Stage 3 completes.")

    with col_hm2:
        st.subheader("Domain × Language (top 10 languages)")
        if dl_mat and dl_mat.get("languages"):
            n_domains = len(dl_mat["domains"])
            fig = _heatmap(
                dl_mat["domains"], dl_mat["languages"], dl_mat["counts"],
                "Domain × Language",
                height=max(300, n_domains * 28 + 80),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Available after LLM annotation completes.")

# ---------------------------------------------------------------------------
# Row 4: Topics
# ---------------------------------------------------------------------------

topics_counts   = stats.get("topics_counts", {})
topics_by_domain = stats.get("topics_by_domain", {})

if topics_counts or ("topics" in df.columns and df["topics"].str.strip().any()):
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Topics")
    col_top_overall, col_top_domain = st.columns([1, 1])

    with col_top_overall:
        st.markdown("###### Top topics overall")
        if topics_counts:
            st.plotly_chart(
                _bar(topics_counts, top_n=25, color_idx=3, height=520),
                use_container_width=True,
            )
        elif "topics" in df.columns:
            # Fall back to sample
            all_topics = (
                df["topics"].dropna()
                .str.split(", ").explode().str.strip()
                .loc[lambda s: s != ""]
            )
            tc = all_topics.value_counts().head(25).to_dict()
            st.plotly_chart(_bar(tc, top_n=25, color_idx=3, height=520), use_container_width=True)

    with col_top_domain:
        st.markdown("###### Top topics by domain")
        available_domains = (
            list(topics_by_domain.keys()) if topics_by_domain
            else sorted(df["domain"].dropna().unique().tolist() if "domain" in df.columns else [])
        )
        if available_domains:
            chosen = st.selectbox("Domain", available_domains, key="topic_domain_select")
            if topics_by_domain and chosen in topics_by_domain:
                domain_topic_data = {t: c for t, c in topics_by_domain[chosen]}
                st.plotly_chart(
                    _bar(domain_topic_data, top_n=15, color_idx=4, height=440),
                    use_container_width=True,
                )
            elif "topics" in df.columns:
                # Fall back to sample filtered by domain
                sub = df[df["domain"] == chosen]["topics"].dropna()
                sub_topics = (
                    sub.str.split(", ").explode().str.strip().loc[lambda s: s != ""]
                )
                domain_topic_data = sub_topics.value_counts().head(15).to_dict()
                if domain_topic_data:
                    st.plotly_chart(
                        _bar(domain_topic_data, top_n=15, color_idx=4, height=440),
                        use_container_width=True,
                    )
                else:
                    st.info(f"No topics found for domain '{chosen}'.")
        else:
            st.info("Topics available after LLM annotation completes.")

# ---------------------------------------------------------------------------
# Row 5: Collection summary + full sources table
# ---------------------------------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)
col_metrics, col_filler = st.columns([1, 2])

with col_metrics:
    st.subheader("Collection Summary")
    st.metric("Total prompts",     f"{total:,}")
    st.metric("Unique domains",    f"{len(stats.get('domain_counts', {})) or df['domain'].nunique():,}")
    st.metric("Unique sources",    f"{len(stats.get('source_counts', {})) or df['source'].nunique():,}")
    n_lang = len(stats.get("language_counts", {}))
    if n_lang:
        st.metric("Languages detected", f"{n_lang:,}")

with st.expander("All sources breakdown"):
    src_dict = stats.get("source_counts") or df["source"].value_counts().to_dict()
    src_df = pd.DataFrame(
        sorted(src_dict.items(), key=lambda x: -x[1]),
        columns=["source", "count"],
    )
    src_df["% of total"] = (src_df["count"] / src_df["count"].sum() * 100).round(2)
    st.dataframe(src_df, use_container_width=True, hide_index=True)
