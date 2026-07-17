"""
Real-world decontamination smoke tests against small public HF eval datasets.

These exercise the parts the synthetic ``local_jsonl`` tests can't: the real
``datasets.load_dataset`` path, ``hf_configs`` list loading, real field shapes
(GSM8K ``question``, MMLU ``choices`` list), and the ``min_gram_size`` floor on
genuine data.

Network-gated: the whole module is skipped when ``datasets`` is unavailable,
when ``SFT_SKIP_NETWORK_TESTS`` is set, or when the dataset can't be fetched
(offline / HF down / gated). It downloads only small splits (GSM8K test ≈ 1319
rows; one MMLU config ≈ 100 rows) and relies on the HF cache on repeat runs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sft_pipeline.checkpoint import CheckpointManager, prompt_id
from sft_pipeline.config import load_config
from sft_pipeline.stages.decontaminate import run_decontaminate
from sft_pipeline.storage import iter_jsonl_dir

if os.environ.get("SFT_SKIP_NETWORK_TESTS"):
    pytest.skip("SFT_SKIP_NETWORK_TESTS set", allow_module_level=True)

datasets = pytest.importorskip("datasets", reason="datasets not installed")


def _load_or_skip(repo: str, name: str, split: str):
    try:
        return datasets.load_dataset(repo, name=name, split=split)
    except Exception as exc:  # noqa: BLE001 — offline / HF down / gated → skip, don't fail
        pytest.skip(f"could not fetch {repo} ({name}/{split}): {exc}")


def _decontaminate(tmp_path: Path, planted: list[tuple[str, str]], eval_block: str):
    """Write a planted stage1 pool + config, run the stage, return (survivors, removed, report).

    planted: list of (prompt_text, source). eval_block: YAML list body under `evals:`.
    """
    stage1 = tmp_path / "stage1"
    stage1.mkdir()
    with (stage1 / "part-000000.jsonl").open("w") as f:
        for text, source in planted:
            f.write(json.dumps(
                {"prompt_id": prompt_id(text), "prompt": text, "source": source}
            ) + "\n")

    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(
        "global: {run_id: rt, base_path: \"%s\", checkpoint_db: \"%s\", device: cpu}\n"
        "stage1_collect: {output_dir: \"%s\"}\n"
        "decontaminate:\n"
        "  enabled: true\n"
        "  n_workers: 1\n"
        "  evals:\n"
        "%s\n"
        % (
            tmp_path.as_posix(),
            (tmp_path / "ckpt.duckdb").as_posix(),
            stage1.as_posix(),
            eval_block,
        )
    )
    cfg = load_config(cfg_yaml)
    with CheckpointManager(cfg.global_.checkpoint_db) as cm:
        run_decontaminate(cfg, cm)

    survivors = {r["prompt_id"] for r in iter_jsonl_dir(Path(cfg.decontaminate.output_dir))}
    removed = {r["prompt_id"]: r for r in iter_jsonl_dir(Path(cfg.decontaminate.removed_dir))}
    report = json.loads(Path(cfg.decontaminate.report_path).read_text())
    return survivors, removed, report


def test_gsm8k_real(tmp_path):
    gsm = _load_or_skip("openai/gsm8k", "main", "test")
    q_verbatim = gsm[0]["question"]
    q_embedded = gsm[7]["question"]

    planted = [
        (q_verbatim, "src_verbatim"),
        (f"Please solve the following problem: {q_embedded}  Show all your working.", "src_embedded"),
        ("Write a limerick about a robot learning to paint.", "src_clean"),
        ("Summarize the main causes of the fall of the Western Roman Empire.", "src_clean"),
    ]
    eval_block = (
        "    - {name: gsm8k, source: hf_dataset, hf_repo_id: openai/gsm8k, "
        "hf_configs: [main], splits: [test], match_fields: [question]}"
    )
    survivors, removed, report = _decontaminate(tmp_path, planted, eval_block)

    # Verbatim and embedded eval questions are removed and attributed to gsm8k.
    assert prompt_id(q_verbatim) in removed
    assert removed[prompt_id(q_verbatim)]["matched_eval"] == "gsm8k"
    embedded_text = f"Please solve the following problem: {q_embedded}  Show all your working."
    assert prompt_id(embedded_text) in removed  # 13-gram containment on real text

    # Unrelated prompts survive.
    assert prompt_id("Write a limerick about a robot learning to paint.") in survivors

    assert report["total_input"] == 4
    assert report["total_removed"] == 2
    assert report["removed_per_eval"]["gsm8k"] == 2
    assert report["eval_items_loaded"]["gsm8k"] > 1000  # full test split


def test_mmlu_choices_floor_real(tmp_path):
    # abstract_algebra is a tiny MMLU config; its choices are often single tokens
    # ("0", "4", ...) — the min_gram_size floor must stop those from removing
    # legitimate prompts that merely contain such tokens.
    mmlu = _load_or_skip("cais/mmlu", "abstract_algebra", "test")
    real_q = mmlu[0]["question"]

    benign = "Compute 0 plus 1 and explain the result briefly."
    planted = [
        (real_q, "src_q"),
        (benign, "src_benign"),  # contains choice-like tokens 0/1 → must survive
        ("Describe how photosynthesis converts sunlight into chemical energy.", "src_clean"),
    ]
    eval_block = (
        "    - {name: mmlu, source: hf_dataset, hf_repo_id: cais/mmlu, "
        "hf_configs: [abstract_algebra], splits: [test], match_fields: [question, choices]}"
    )
    survivors, removed, report = _decontaminate(tmp_path, planted, eval_block)

    # The real question is removed; the benign short-overlap prompt is NOT.
    assert prompt_id(real_q) in removed
    assert prompt_id(benign) in survivors
    assert prompt_id("Describe how photosynthesis converts sunlight into chemical energy.") in survivors

    # The floor actually fired: some short choice texts were dropped from the index.
    assert report["eval_items_dropped_short"].get("mmlu", 0) > 0
    assert report["removed_per_eval"].get("mmlu", 0) == 1
