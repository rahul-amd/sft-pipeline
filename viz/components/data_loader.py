"""Cached snapshot loader shared across all pages."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

_DATA_DIR = Path(__file__).parent.parent / "data"
SNAPSHOT_PATH = _DATA_DIR / "snapshot.parquet"
META_PATH = _DATA_DIR / "meta.json"


def load_meta() -> dict:
    if not META_PATH.exists():
        return {}
    try:
        return json.loads(META_PATH.read_text())
    except Exception:
        return {}


def has_stage(meta: dict, stage: str) -> bool:
    return bool(meta.get("stages", {}).get(stage))


@st.cache_data(show_spinner="Loading snapshot …")
def load_snapshot(mtime: float) -> pd.DataFrame:  # mtime busts cache when file changes
    return pd.read_parquet(SNAPSHOT_PATH)


def get_snapshot() -> tuple[pd.DataFrame, dict]:
    """Return (df, meta). Shows error + stops if snapshot missing."""
    meta = load_meta()
    if not SNAPSHOT_PATH.exists():
        st.error(
            "**No snapshot found.**  Run the export script first:\n\n"
            "```bash\npython viz/export.py --run-dir /path/to/run\n```"
        )
        st.stop()
    mtime = SNAPSHOT_PATH.stat().st_mtime
    df = load_snapshot(mtime)
    return df, meta
