"""Answers Viewer — prompt + reasoning + answer, post Stage 5."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from components.data_loader import get_snapshot, has_stage
from components.filters import render_filters
from components.theme import apply_theme

st.set_page_config(page_title="Answers — SFT Pipeline", layout="wide")
apply_theme()
st.title("Answers Viewer")

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

st.markdown("<br>", unsafe_allow_html=True)

# Pagination
PAGE_SIZE = 10
total_pages = max(1, (len(filtered) - 1) // PAGE_SIZE + 1)
page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
start = (page - 1) * PAGE_SIZE
page_df = filtered.iloc[start: start + PAGE_SIZE]

st.divider()

_DIFF_COLOURS = {
    "easy":   ("rgba(52,211,153,0.15)",  "#34d399"),
    "medium": ("rgba(251,191,36,0.15)",  "#fbbf24"),
    "hard":   ("rgba(248,113,113,0.15)", "#f87171"),
}

for _, row in page_df.iterrows():
    passed = row.get("passed_filters")
    if passed is True:
        filter_badge = '<span style="background:rgba(16,185,129,0.15);color:#6ee7b7;border:1px solid rgba(16,185,129,0.3);padding:1px 8px;border-radius:4px;font-size:0.75rem;margin-left:6px;">✓ passed</span>'
    elif passed is False:
        filter_badge = '<span style="background:rgba(239,68,68,0.15);color:#fca5a5;border:1px solid rgba(239,68,68,0.3);padding:1px 8px;border-radius:4px;font-size:0.75rem;margin-left:6px;">✗ filtered</span>'
    else:
        filter_badge = ""

    diff = row.get("difficulty", "")
    diff_bg, diff_text = _DIFF_COLOURS.get(diff, ("#1e293b", "#94a3b8"))
    diff_badge = (
        f'<span style="background:{diff_bg};color:{diff_text};border:1px solid {diff_text}33;'
        f'padding:1px 8px;border-radius:4px;font-size:0.75rem;margin-left:6px;">{diff}</span>'
        if diff else ""
    )

    header_html = (
        f'<span style="color:#e2e8f0;font-weight:600;">{row["source"]}</span>'
        f'<span style="color:#475569;margin:0 6px;">·</span>'
        f'<span style="color:#a5b4fc;">{row["domain"]}</span>'
        f"{diff_badge}{filter_badge}"
    )

    with st.expander(f"{row['source']} · {row['domain']}", expanded=False):
        # Re-render the styled header inside the expander
        st.markdown(
            f'<div style="margin-bottom:0.75rem;">{header_html}</div>',
            unsafe_allow_html=True,
        )
        col_prompt, col_reasoning, col_answer = st.columns([2, 3, 2])

        with col_prompt:
            st.markdown(
                '<p style="color:#475569;font-size:0.7rem;text-transform:uppercase;'
                'letter-spacing:0.07em;margin-bottom:0.4rem;">Prompt</p>',
                unsafe_allow_html=True,
            )
            st.markdown(row["prompt"])

        with col_reasoning:
            st.markdown(
                '<p style="color:#475569;font-size:0.7rem;text-transform:uppercase;'
                'letter-spacing:0.07em;margin-bottom:0.4rem;">Reasoning</p>',
                unsafe_allow_html=True,
            )
            reasoning = row.get("reasoning") or ""
            if reasoning:
                st.markdown(
                    f'<div style="max-height:320px;overflow-y:auto;'
                    f'background:rgba(0,0,0,0.25);border:1px solid #1e293b;'
                    f'padding:0.75rem 1rem;border-radius:8px;'
                    f'font-size:0.82rem;color:#94a3b8;line-height:1.65;'
                    f'font-family:\'JetBrains Mono\',monospace;">{reasoning}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No reasoning trace.")

        with col_answer:
            st.markdown(
                '<p style="color:#475569;font-size:0.7rem;text-transform:uppercase;'
                'letter-spacing:0.07em;margin-bottom:0.4rem;">Answer</p>',
                unsafe_allow_html=True,
            )
            st.markdown(row.get("answer") or "—")
