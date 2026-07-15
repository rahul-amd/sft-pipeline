#!/usr/bin/env python3
"""
Gradio viz: baseline vs current Stage 6 verifiers, judged by glm-5.2.

Reads data/comparison.jsonl (produced by prep_comparison.py) and shows:
  - Overview: metric cards, confusion matrices, rejection-reason breakdowns
  - Verdict flips: explore records whose verdict changed, with judge labels
  - Improvement journey: the measured iteration-by-iteration loop

Usage:
    python sanity-checks/viz_compare.py          # http://localhost:7860
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr
import orjson
import pandas as pd
import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent
COMPARISON = HERE / "data" / "comparison.jsonl"
GOOD_THRESHOLD = 4.0

BASE_COLOR = "#d1495b"   # baseline = red
CURR_COLOR = "#00798c"   # current  = teal


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_df() -> pd.DataFrame:
    rows = []
    with COMPARISON.open("rb") as f:
        for line in f:
            if line.strip():
                rows.append(orjson.loads(line))
    df = pd.DataFrame(rows)
    df["judged"] = df["judge_correctness"].notna()
    df["bad"] = df["judge_correctness"] < GOOD_THRESHOLD
    for side in ("baseline", "current"):
        df[f"{side}_reason_bucket"] = (
            df[f"{side}_reason"].fillna("").str.split(":").str[:2].str.join(":")
        )
    def flip(row):
        b, c = row["baseline_passed"], row["current_passed"]
        if b and c:
            return "both pass"
        if not b and not c:
            return "both reject"
        if not b and c:
            return "reject → pass (baseline-only reject)"
        return "pass → reject (current-only reject)"
    df["flip"] = df.apply(flip, axis=1)
    return df


def confusion(df: pd.DataFrame, side: str) -> dict:
    """positive = verifier rejects; bad = judge correctness < threshold."""
    j = df[df["judged"]]
    rejected = ~j[f"{side}_passed"]
    bad = j["bad"]
    tp = int((rejected & bad).sum())
    fp = int((rejected & ~bad).sum())
    fn = int((~rejected & bad).sum())
    tn = int((~rejected & ~bad).sum())
    total = tp + fp + fn + tn
    sd = lambda a, b: (a / b) if b else None
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "rejected": tp + fp,
        "precision": sd(tp, tp + fp),
        "recall": sd(tp, tp + fn),
        "specificity": sd(tn, tn + fp),
        "accuracy": sd(tp + tn, total),
        "n": total,
    }


def fmt(x) -> str:
    return "—" if x is None else f"{x:.3f}"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def metric_bars(df: pd.DataFrame, domain: str) -> go.Figure:
    d = df[df["domain"] == domain]
    b, c = confusion(d, "baseline"), confusion(d, "current")
    metrics = ["specificity", "accuracy", "precision", "recall"]
    fig = go.Figure()
    for side, conf, color in (("baseline", b, BASE_COLOR), ("current", c, CURR_COLOR)):
        fig.add_bar(
            name=side, x=metrics,
            y=[conf[m] if conf[m] is not None else 0 for m in metrics],
            marker_color=color,
            text=[fmt(conf[m]) for m in metrics], textposition="outside",
        )
    fig.update_layout(
        title=f"{domain.upper()} — verifier quality vs LLM judge (n={b['n']} labeled)",
        yaxis=dict(range=[0, 1.15], title="score"),
        barmode="group", height=380, margin=dict(t=50, b=30),
        legend=dict(orientation="h", y=1.12, x=0.65),
    )
    return fig


def confusion_heatmap(df: pd.DataFrame, domain: str, side: str) -> go.Figure:
    conf = confusion(df[df["domain"] == domain], side)
    z = [[conf["TP"], conf["FP"]], [conf["FN"], conf["TN"]]]
    text = [[f"TP<br>{conf['TP']}", f"FP<br>{conf['FP']}"],
            [f"FN<br>{conf['FN']}", f"TN<br>{conf['TN']}"]]
    fig = go.Figure(go.Heatmap(
        z=z, x=["judge: bad", "judge: good"], y=["verifier: reject", "verifier: pass"],
        text=text, texttemplate="%{text}", showscale=False,
        colorscale=[[0, "#f0f4f8"], [1, BASE_COLOR if side == "baseline" else CURR_COLOR]],
    ))
    fig.update_layout(
        title=f"{side} — rejects {conf['rejected']}/{conf['n']}",
        height=300, margin=dict(t=45, b=20, l=10, r=10),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def reason_bars(df: pd.DataFrame, domain: str) -> go.Figure:
    d = df[df["domain"] == domain]
    fig = go.Figure()
    for side, color in (("baseline", BASE_COLOR), ("current", CURR_COLOR)):
        rej = d[~d[f"{side}_passed"]][f"{side}_reason_bucket"].value_counts()
        fig.add_bar(name=side, y=rej.index.tolist(), x=rej.values.tolist(),
                    orientation="h", marker_color=color)
    fig.update_layout(
        title=f"{domain.upper()} — rejection reasons (all {len(d)} sampled records)",
        height=420, barmode="group", margin=dict(t=50, b=30, l=10),
        xaxis_title="records rejected",
        legend=dict(orientation="h", y=1.1, x=0.65),
    )
    return fig


# Measured on the 100/domain dev set during the improvement loop.
JOURNEY = pd.DataFrame([
    # iteration, change, math_FP, math_TP, code_FP, code_TP
    (1, "Baseline production chain", 30, 1, 64, 3),
    (2, "Length-aware repetition · code verifier rewrite · contradiction gate", 7, 1, 38, 2),
    (3, "Parser newline fix · contradiction off by default", 1, 1, 1, 0),
    (4, "Repetition floor 30→40", 0, 1, 1, 0),
    (5, "True delimiters discovered (<|channel>thought … <channel|>)", 0, 1, 0, 0),
], columns=["iteration", "change", "math_FP", "math_TP", "code_FP", "code_TP"])


def journey_fig() -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=JOURNEY["iteration"], y=JOURNEY["math_FP"], name="math FPs",
                    mode="lines+markers+text", text=JOURNEY["math_FP"],
                    textposition="top center", line=dict(color=BASE_COLOR))
    fig.add_scatter(x=JOURNEY["iteration"], y=JOURNEY["code_FP"], name="code FPs",
                    mode="lines+markers+text", text=JOURNEY["code_FP"],
                    textposition="top center", line=dict(color="#edae49"))
    fig.update_layout(
        title="False positives per 100 records across loop iterations (dev set)",
        xaxis=dict(title="iteration", dtick=1), yaxis_title="false rejections / 100",
        height=400, margin=dict(t=50, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Flip explorer
# ---------------------------------------------------------------------------

FLIP_COLS = ["prompt_id", "domain", "judge_correctness", "baseline_reason", "current_reason", "prompt"]


def flip_table(df: pd.DataFrame, domain: str, flip_kind: str, only_judged: bool) -> pd.DataFrame:
    d = df[df["domain"] == domain]
    if flip_kind != "all":
        d = d[d["flip"] == flip_kind]
    if only_judged:
        d = d[d["judged"]]
    out = d[FLIP_COLS].copy()
    out["prompt"] = out["prompt"].str.slice(0, 120)
    out["baseline_reason"] = out["baseline_reason"].str.slice(0, 60)
    out["current_reason"] = out["current_reason"].str.slice(0, 60)
    return out.reset_index(drop=True)


def record_detail(df: pd.DataFrame, prompt_id: str) -> str:
    row = df[df["prompt_id"] == prompt_id]
    if row.empty:
        return "*Select a row above to inspect the record.*"
    r = row.iloc[0]
    corr = r["judge_correctness"]
    corr_s = "unjudged" if pd.isna(corr) else f"{corr:.0f}/5"
    b = "PASS ✅" if r["baseline_passed"] else f"REJECT ❌ `{r['baseline_reason']}`"
    c = "PASS ✅" if r["current_passed"] else f"REJECT ❌ `{r['current_reason']}`"
    return (
        f"### `{r['prompt_id']}`  ·  {r['domain']}  ·  source: {r['source']}\n\n"
        f"| | verdict |\n|---|---|\n| **baseline** | {b} |\n| **current** | {c} |\n"
        f"| **judge correctness** | {corr_s} |\n\n"
        f"**Judge said:** {r['judge_reasoning'] or '—'}\n\n"
        f"---\n\n**Prompt**\n\n{r['prompt']}\n\n---\n\n"
        f"**Response** *(truncated)*\n\n```\n{r['response'][:3000]}\n```\n"
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    df = load_df()
    domains = sorted(df["domain"].unique())

    with gr.Blocks(title="Verifier comparison — baseline vs current") as app:
        gr.Markdown(
            "# Stage 6 verifiers: baseline vs current\n"
            f"Same {len(df)} Stage 5 records scored by both verifier generations; "
            f"ground truth = glm-5.2 judge (bad = correctness < {GOOD_THRESHOLD:g}, "
            f"{int(df['judged'].sum())} records labeled)."
        )

        with gr.Tab("Overview"):
            for d in domains:
                gr.Markdown(f"## {d.upper()}")
                with gr.Row():
                    gr.Plot(metric_bars(df, d))
                    gr.Plot(reason_bars(df, d))
                with gr.Row():
                    gr.Plot(confusion_heatmap(df, d, "baseline"))
                    gr.Plot(confusion_heatmap(df, d, "current"))

        with gr.Tab("Verdict flips"):
            gr.Markdown(
                "Records whose verdict changed between generations. "
                "`reject → pass` under a **good** judge score = a false positive the loop fixed."
            )
            with gr.Row():
                dom = gr.Dropdown(domains, value=domains[0], label="Domain")
                kind = gr.Dropdown(
                    ["all", "reject → pass (baseline-only reject)",
                     "pass → reject (current-only reject)", "both reject", "both pass"],
                    value="reject → pass (baseline-only reject)", label="Flip type",
                )
                judged_only = gr.Checkbox(value=True, label="Judged records only")
            table = gr.Dataframe(
                flip_table(df, domains[0], "reject → pass (baseline-only reject)", True),
                interactive=False, max_height=320,
            )
            detail = gr.Markdown("*Select a row above to inspect the record.*")

            def refresh(dm, kd, jo):
                return flip_table(df, dm, kd, jo), "*Select a row above to inspect the record.*"

            for comp in (dom, kind, judged_only):
                comp.change(refresh, [dom, kind, judged_only], [table, detail])

            def on_select(tbl: pd.DataFrame, evt: gr.SelectData):
                if tbl is None or len(tbl) == 0:
                    return "*No rows.*"
                return record_detail(df, tbl.iloc[evt.index[0]]["prompt_id"])

            table.select(on_select, [table], [detail])

        with gr.Tab("Improvement journey"):
            gr.Plot(journey_fig())
            gr.Dataframe(JOURNEY, interactive=False)
            gr.Markdown(
                "Measured on the 100/domain dev subset with cached judge labels; "
                "the Overview tab shows the final generation on the full 1000/domain sample."
            )

    return app


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
