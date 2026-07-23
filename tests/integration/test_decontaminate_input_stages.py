"""
`decontaminate.input_stages` — choosing which upstream stages to decontaminate.

Sets up contaminated prompts in BOTH stage1 and stage2, decontaminates stage1
only, and verifies: stage1 is scrubbed into the clean pool, stage2 is left
untouched, and Stage 3 still reads stage2 (raw passthrough) so nothing is
dropped.
"""
from __future__ import annotations

import json
from pathlib import Path

from sft_pipeline.checkpoint import CheckpointManager, prompt_id
from sft_pipeline.config import load_config
from sft_pipeline.stages.decontaminate import run_decontaminate
from sft_pipeline.stages.stage3_cluster import _resolve_input_dirs
from sft_pipeline.storage import iter_jsonl_dir

_EVAL_Q = "the mitochondria is the powerhouse of the cell and produces energy for the body"
_S1_CONTAM = f"Explain why {_EVAL_Q} in one paragraph."
_S2_CONTAM = f"Note that {_EVAL_Q} according to biology textbooks."
_S1_CLEAN = "Write a haiku about the ocean at dawn"
_S2_CLEAN = "List three uses for a paperclip"


def _setup(tmp_path: Path, input_stages: list[str]) -> Path:
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(json.dumps({"question": _EVAL_Q}) + "\n")
    for name, contam, clean in (("stage1", _S1_CONTAM, _S1_CLEAN), ("stage2", _S2_CONTAM, _S2_CLEAN)):
        d = tmp_path / name
        d.mkdir()
        with (d / "part-000000.jsonl").open("w") as f:
            for p in (contam, clean):
                f.write(json.dumps({"prompt_id": prompt_id(p), "prompt": p, "source": name}) + "\n")

    cfg_text = (
        "global: {seed: 42, run_id: t, base_path: \"%s\", checkpoint_db: \"%s\", device: cpu}\n"
        "stage1_collect: {output_dir: \"%s\"}\n"
        "stage2_generate: {output_dir: \"%s\"}\n"
        "decontaminate:\n"
        "  enabled: true\n"
        "  n_workers: 1\n"
        "  input_stages: [%s]\n"
        "  output_dir: \"%s\"\n"
        "  report_path: \"%s\"\n"
        "  removed_dir: \"%s\"\n"
        "  evals:\n"
        "    - {name: bio, source: local_jsonl, path: \"%s\", match_fields: [question]}\n"
        % (
            tmp_path.as_posix(), (tmp_path / "ckpt.duckdb").as_posix(),
            (tmp_path / "stage1").as_posix(), (tmp_path / "stage2").as_posix(),
            ", ".join(input_stages),
            (tmp_path / "sd" / "clean").as_posix(),
            (tmp_path / "sd" / "report.json").as_posix(),
            (tmp_path / "sd" / "removed").as_posix(),
            eval_path.as_posix(),
        )
    )
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(cfg_text)
    return cfg_file


def test_decontaminate_stage1_only(tmp_path):
    cfg = load_config(_setup(tmp_path, ["stage1_collect"]))
    with CheckpointManager(cfg.global_.checkpoint_db) as cm:
        run_decontaminate(cfg, cm)

    clean = {r["prompt_id"] for r in iter_jsonl_dir(Path(cfg.decontaminate.output_dir))}
    # stage1: contaminated removed, clean kept.
    assert prompt_id(_S1_CONTAM) not in clean
    assert prompt_id(_S1_CLEAN) in clean
    # stage2 was NOT decontaminated → not in the clean pool at all.
    assert prompt_id(_S2_CONTAM) not in clean
    assert prompt_id(_S2_CLEAN) not in clean

    report = json.loads(Path(cfg.decontaminate.report_path).read_text())
    assert report["total_input"] == 2          # only stage1's 2 records scanned
    assert report["total_removed"] == 1
    assert report["removed_per_source"] == {"stage1": 1}

    # Stage 3 reads the clean pool PLUS raw stage2 (passthrough, not dropped).
    dirs = _resolve_input_dirs(cfg)
    assert Path(cfg.decontaminate.output_dir) in dirs
    assert Path(cfg.stage2_generate.output_dir) in dirs
    # The undecontaminated stage2 prompts (incl. its contaminated one) are still
    # reachable by Stage 3 via the raw passthrough dir.
    reachable = {r["prompt_id"] for d in dirs for r in iter_jsonl_dir(d)}
    assert prompt_id(_S2_CONTAM) in reachable
    assert prompt_id(_S2_CLEAN) in reachable
    assert prompt_id(_S1_CONTAM) not in reachable  # removed by decontam


def test_decontaminate_both_stages_default(tmp_path):
    # Default (both stages) scrubs stage1 AND stage2 into the clean pool.
    cfg = load_config(_setup(tmp_path, ["stage1_collect", "stage2_generate"]))
    with CheckpointManager(cfg.global_.checkpoint_db) as cm:
        run_decontaminate(cfg, cm)
    clean = {r["prompt_id"] for r in iter_jsonl_dir(Path(cfg.decontaminate.output_dir))}
    assert prompt_id(_S1_CONTAM) not in clean and prompt_id(_S2_CONTAM) not in clean
    assert prompt_id(_S1_CLEAN) in clean and prompt_id(_S2_CLEAN) in clean
    # Stage 3 reads only the clean pool (no raw passthrough needed).
    assert _resolve_input_dirs(cfg) == [Path(cfg.decontaminate.output_dir)]
    report = json.loads(Path(cfg.decontaminate.report_path).read_text())
    assert report["total_removed"] == 2
    assert report["removed_per_source"] == {"stage1": 1, "stage2": 1}
