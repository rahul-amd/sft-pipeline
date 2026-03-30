"""Answers Viewer — prompt + reasoning + answer, post Stage 5."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from components.data_loader import get_snapshot, has_stage
from components.filters import render_filters

st.set_page_config(page_title="Answers — SFT Pipeline", layout="wide")
st.title("💬 Answers Viewer")

df, meta = get_snapshot()

if not has_stage(meta, "stage5") or "answer" not in df.columns or df["answer"].isna().all():
    st.info(
        "Answer data is not available in this snapshot.\n\n"
        "Re-run the export after Stage 5 (inference) completes:\n\n"
        "```bash\npython viz/export.py --run-dir /path/to/run\n```"
    )
    st.stop()

answers_df = df.dropna(subset=["answer"]).copy()
filtered = render_filters(
    answers_df,
    show_difficulty=True,
    show_search=True,
    show_passed_only=has_stage(meta, "stage6"),
)

st.caption(f"Showing **{len(filtered):,}** of {len(answers_df):,} answered prompts")

if filtered.empty:
    st.warning("No records match the current filters.")
    st.stop()

# Pagination
PAGE_SIZE = 10
total_pages = max(1, (len(filtered) - 1) // PAGE_SIZE + 1)
page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
start = (page - 1) * PAGE_SIZE
page_df = filtered.iloc[start: start + PAGE_SIZE]

st.divider()

for _, row in page_df.iterrows():
    passed = row.get("passed_filters")
    badge = " ✅" if passed is True else (" ❌" if passed is False else "")
    header = f"**{row['source']}** · {row['domain']}"
    if row.get("difficulty"):
        header += f" · {row['difficulty']}"
    header += badge

    with st.expander(header, expanded=False):
        col_prompt, col_reasoning, col_answer = st.columns([2, 3, 2])

        with col_prompt:
            st.markdown("**Prompt**")
            st.markdown(row["prompt"])

        with col_reasoning:
            st.markdown("**Reasoning**")
            reasoning = row.get("reasoning") or ""
            if reasoning:
                st.markdown(
                    f'<div style="max-height:300px;overflow-y:auto;'
                    f'background:#f8f8f8;padding:8px;border-radius:4px;'
                    f'font-size:0.85em">{reasoning}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No reasoning trace.")

        with col_answer:
            st.markdown("**Answer**")
            st.markdown(row.get("answer") or "—")
