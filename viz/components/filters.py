"""Shared sidebar filter widgets, reused across Prompts, Clusters, and Answers pages."""
from __future__ import annotations

import pandas as pd
import streamlit as st


def render_filters(
    df: pd.DataFrame,
    show_difficulty: bool = True,
    show_search: bool = True,
    show_passed_only: bool = False,
) -> pd.DataFrame:
    """
    Render sidebar filters and return the filtered DataFrame.

    Parameters
    ----------
    df:               Full snapshot DataFrame.
    show_difficulty:  Show difficulty filter (only if the column exists).
    show_search:      Show free-text search box.
    show_passed_only: Show "passed filters only" toggle (Stage 6).
    """
    with st.sidebar:
        st.header("Filters")

        # Domain
        domains = sorted(df["domain"].dropna().unique().tolist())
        selected_domains = st.multiselect("Domain", domains, default=domains, key="f_domain")

        # Source
        sources = sorted(df["source"].dropna().unique().tolist())
        selected_sources = st.multiselect("Source", sources, default=sources, key="f_source")

        # Difficulty (optional)
        selected_difficulties = None
        if show_difficulty and "difficulty" in df.columns and df["difficulty"].notna().any():
            difficulties = sorted(df["difficulty"].dropna().unique().tolist())
            selected_difficulties = st.multiselect(
                "Difficulty", difficulties, default=difficulties, key="f_difficulty"
            )

        # Search
        search = ""
        if show_search:
            search = st.text_input("Search prompts", "", key="f_search")

        # Passed filters toggle
        passed_only = False
        if show_passed_only and "passed_filters" in df.columns and df["passed_filters"].notna().any():
            passed_only = st.checkbox("Passed filters only", value=False, key="f_passed")

    # Apply
    mask = (
        df["domain"].isin(selected_domains)
        & df["source"].isin(selected_sources)
    )
    if selected_difficulties is not None:
        mask &= df["difficulty"].isin(selected_difficulties)
    if search:
        mask &= df["prompt"].str.contains(search, case=False, na=False)
    if passed_only:
        mask &= df["passed_filters"] == True  # noqa: E712

    return df[mask].copy()
