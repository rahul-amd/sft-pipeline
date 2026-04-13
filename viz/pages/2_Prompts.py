"""Prompt Browser — searchable, filterable table with full-text expand."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from components.data_loader import get_snapshot
from components.filters import render_filters
from components.theme import apply_theme

st.set_page_config(page_title="Prompts — SFT Pipeline", layout="wide")
apply_theme()
st.title("Prompt Browser")

df, meta = get_snapshot()
filtered = render_filters(df, show_difficulty=True, show_search=True)

st.caption(f"Showing **{len(filtered):,}** of {len(df):,} prompts")

if filtered.empty:
    st.warning("No prompts match the current filters.")
    st.stop()

# Build display table — use summary as preview if annotation has run
has_summary = "summary" in filtered.columns and filtered["summary"].str.strip().any()
has_topics = "topics" in filtered.columns and filtered["topics"].str.strip().any()
has_language = "language" in filtered.columns and filtered["language"].notna().any()

display_cols = ["prompt_id", "source", "domain"]
if "difficulty" in filtered.columns:
    display_cols.append("difficulty")
if has_language:
    display_cols.append("language")
display_cols.append("prompt")
display = filtered[display_cols].copy()

if has_summary:
    display["prompt"] = filtered["summary"].fillna("").str.strip()
    display.loc[display["prompt"] == "", "prompt"] = (
        filtered.loc[display["prompt"] == "", "prompt"].str.slice(0, 120) + "…"
    )
    display = display.rename(columns={"prompt_id": "ID", "prompt": "Summary"})
else:
    display["prompt"] = display["prompt"].str.slice(0, 120) + "…"
    display = display.rename(columns={"prompt_id": "ID", "prompt": "Prompt (truncated)"})

st.markdown("<br>", unsafe_allow_html=True)

event = st.dataframe(
    display.reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
)

# Show full prompt when a row is selected
selected_rows = event.selection.get("rows", []) if event and hasattr(event, "selection") else []
if selected_rows:
    idx = selected_rows[0]
    row = filtered.iloc[idx]
    st.divider()
    st.subheader("Full Prompt")
    cols = st.columns([3, 1])
    with cols[0]:
        st.text_area(
            label="",
            value=row["prompt"],
            height=300,
            disabled=True,
            label_visibility="collapsed",
        )
    with cols[1]:
        # Render as styled markdown badges
        st.markdown(
            f"""
            <div style="display:flex;flex-direction:column;gap:0.6rem;padding-top:0.25rem;">
              <div>
                <span style="color:#475569;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;">ID</span><br>
                <code style="font-size:0.72rem;color:#a5b4fc;">{row['prompt_id'][:20]}…</code>
              </div>
              <div>
                <span style="color:#475569;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;">Source</span><br>
                <span style="color:#e2e8f0;font-size:0.85rem;">{row['source']}</span>
              </div>
              <div>
                <span style="color:#475569;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;">Domain</span><br>
                <span style="background:rgba(99,102,241,0.15);color:#818cf8;border:1px solid rgba(99,102,241,0.3);padding:2px 8px;border-radius:4px;font-size:0.8rem;">{row['domain']}</span>
              </div>
            """,
            unsafe_allow_html=True,
        )
        if row.get("difficulty"):
            colour = {"easy": "rgba(52,211,153,0.15)", "medium": "rgba(251,191,36,0.15)", "hard": "rgba(248,113,113,0.15)"}.get(row["difficulty"], "rgba(99,102,241,0.15)")
            text = {"easy": "#34d399", "medium": "#fbbf24", "hard": "#f87171"}.get(row["difficulty"], "#818cf8")
            st.markdown(
                f'<div style="margin-top:0.1rem;">'
                f'<span style="color:#475569;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;">Difficulty</span><br>'
                f'<span style="background:{colour};color:{text};padding:2px 8px;border-radius:4px;font-size:0.8rem;">{row["difficulty"]}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )
        if has_language and row.get("language"):
            st.markdown(
                f'<div style="margin-top:0.1rem;">'
                f'<span style="color:#475569;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;">Language</span><br>'
                f'<span style="color:#e2e8f0;font-size:0.85rem;">{row["language"]}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )
        if has_topics and row.get("topics", "").strip():
            topics_str = row["topics"] if isinstance(row["topics"], str) else ", ".join(row["topics"])
            if topics_str.strip():
                tags = "".join(
                    f'<span style="background:rgba(34,211,238,0.1);color:#22d3ee;border:1px solid rgba(34,211,238,0.25);'
                    f'padding:2px 7px;border-radius:4px;font-size:0.75rem;margin:2px 2px 2px 0;display:inline-block;">'
                    f'{t.strip()}</span>'
                    for t in topics_str.split(",") if t.strip()
                )
                st.markdown(
                    f'<div style="margin-top:0.4rem;">'
                    f'<span style="color:#475569;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;">Topics</span><br>'
                    f'<div style="margin-top:4px;">{tags}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
        if has_summary and row.get("summary", "").strip():
            st.markdown(
                f'<div style="margin-top:0.5rem;padding:0.6rem 0.75rem;background:rgba(99,102,241,0.08);'
                f'border-left:3px solid #6366f1;border-radius:0 6px 6px 0;">'
                f'<span style="color:#475569;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;">Summary</span><br>'
                f'<span style="color:#94a3b8;font-size:0.82rem;font-style:italic;">{row["summary"]}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )
        if row.get("cluster_id") is not None:
            st.markdown(
                f'<div style="margin-top:0.4rem;">'
                f'<span style="color:#475569;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;">Cluster</span><br>'
                f'<span style="color:#e2e8f0;font-size:0.85rem;">#{int(row["cluster_id"])}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
