"""Prompt Browser — searchable, filterable table with full-text expand."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from components.data_loader import get_snapshot
from components.filters import render_filters

st.set_page_config(page_title="Prompts — SFT Pipeline", layout="wide")
st.title("📋 Prompt Browser")

df, meta = get_snapshot()
filtered = render_filters(df, show_difficulty=True, show_search=True)

st.caption(f"Showing **{len(filtered):,}** of {len(df):,} prompts")

if filtered.empty:
    st.warning("No prompts match the current filters.")
    st.stop()

# Build display table with truncated prompt
display = filtered[["prompt_id", "source", "domain", "difficulty", "prompt"]].copy()
display["prompt"] = display["prompt"].str.slice(0, 120) + "…"
display = display.rename(columns={"prompt_id": "ID", "prompt": "Prompt (truncated)"})

# Render table — st.dataframe returns selection info in newer Streamlit
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
        st.markdown(f"**ID:** `{row['prompt_id']}`")
        st.markdown(f"**Source:** {row['source']}")
        st.markdown(f"**Domain:** {row['domain']}")
        if row.get("difficulty"):
            st.markdown(f"**Difficulty:** {row['difficulty']}")
        if row.get("cluster_id") is not None:
            st.markdown(f"**Cluster:** {int(row['cluster_id'])}")
