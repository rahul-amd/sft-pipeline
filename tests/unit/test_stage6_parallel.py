"""
Stage 6 shard-map filtering tests.

Stage 6 filters one whole input shard per task (serial, single-node pool, or
Ray). These tests assert the per-shard function makes the right pass/reject
decisions and that the single-node pool path produces the same survivors as the
serial path. llm_judge is disabled throughout (the only non-deterministic,
network-dependent filter).
"""
from __future__ import annotations

import json
from pathlib import Path

from sft_pipeline.config import (
    CodeFilterConfig,
    LLMJudgeConfig,
    MathFilterConfig,
    ReasoningDelimiters,
    Stage6Config,
)
from sft_pipeline.stages.stage6_filter import _process_shard
from sft_pipeline.storage import iter_jsonl

_LONG_REASONING = (
    "Let me work through this problem carefully and methodically. "
    "First I identify the key quantities and the relationships between them. "
    "Then I apply the relevant rule step by step, checking each intermediate "
    "result against the constraints before moving on to the next stage. "
    "Combining the partial results gives the final conclusion."
)


def _record(pid: str, domain: str, answer: str, reasoning: str = _LONG_REASONING) -> dict:
    return {
        "prompt_id": pid,
        "prompt": f"Question {pid}?",
        "domain": domain,
        "response": f"<think>\n{reasoning}\n</think>\n<answer>\n{answer}\n</answer>",
    }


def _mixed_records() -> list[dict]:
    recs = []
    for i in range(20):
        domain = ["math", "code", "science", "general"][i % 4]
        recs.append(_record(f"p{i:03d}", domain, f"The final answer for item {i} is {i * 3}."))
    recs.append(_record("short", "general", "No.", reasoning="Too brief."))          # reject: structural
    recs.append(_record("code-ok", "code", "```python\nprint(sum(range(5)))\n```"))    # pass
    recs.append(_record("code-bad", "code", "```python\ndef broken(:\n    return 1\n```"))  # reject: code
    return recs


def _cfg() -> Stage6Config:
    return Stage6Config(
        llm_judge=LLMJudgeConfig(enabled=False),
        math=MathFilterConfig(enabled=True),
        code=CodeFilterConfig(enabled=True, sandbox="subprocess", timeout_seconds=5),
    )


def _write_shard(tmp_path: Path, records: list[dict], name: str = "part-000000") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / f"{name}.jsonl"
    with p.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


def _delims():
    return ReasoningDelimiters(think_start="<think>", think_end="</think>")


def test_process_shard_decisions(tmp_path):
    shard = _write_shard(tmp_path / "in", _mixed_records())
    out = (tmp_path / "out"); out.mkdir(parents=True)
    stats = _process_shard(str(shard), "part-000000", str(out), _cfg(), _delims(), seed=42)

    survivors = {r["prompt_id"] for r in iter_jsonl(out / "part-000000.jsonl")}
    assert "short" not in survivors           # structural reject
    assert "code-bad" not in survivors        # code reject
    assert "code-ok" in survivors             # valid code passes
    assert stats["input"] == 23
    assert stats["passed"] == len(survivors)
    assert stats["rejection_counts"].get("structural", 0) == 1
    assert stats["rejection_counts"].get("code", 0) == 1


def test_pool_matches_serial(tmp_path):
    # Same input across several shards, filtered serially vs via the process pool
    # through run_stage6 — survivors must be identical.
    from sft_pipeline.checkpoint import CheckpointManager
    from sft_pipeline.config import load_config
    from sft_pipeline.stages.stage6_filter import run_stage6
    from sft_pipeline.storage import iter_jsonl_dir

    def _run(base: Path, n_workers: int):
        base.mkdir(parents=True, exist_ok=True)
        stage5 = base / "stage5"; stage5.mkdir()
        recs = _mixed_records()
        # 3 shards so the pool path actually fans out.
        for s in range(3):
            with (stage5 / f"part-00000{s}.jsonl").open("w") as f:
                for i, r in enumerate(recs):
                    rr = dict(r); rr["prompt_id"] = f"s{s}-{r['prompt_id']}"
                    f.write(json.dumps(rr) + "\n")
        cfg_yaml = base / "cfg.yaml"
        cfg_yaml.write_text(
            "global: {run_id: r, base_path: \"%s\", checkpoint_db: \"%s\", device: cpu}\n"
            "stage5_inference: {output_dir: \"%s\"}\n"
            "stage6_filter:\n"
            "  enabled: true\n"
            "  n_workers: %d\n"
            "  output_dir: \"%s\"\n"
            "  report_path: \"%s\"\n"
            "  llm_judge: {enabled: false}\n"
            % (
                base.as_posix(), (base / "ckpt.duckdb").as_posix(), stage5.as_posix(),
                n_workers, (base / "stage6").as_posix(),
                (base / "stage6" / "report.json").as_posix(),
            )
        )
        cfg = load_config(cfg_yaml)
        with CheckpointManager(cfg.global_.checkpoint_db) as cm:
            run_stage6(cfg, cm)
        return {r["prompt_id"] for r in iter_jsonl_dir(Path(cfg.stage6_filter.output_dir))}

    serial = _run(tmp_path / "serial", n_workers=1)
    pool = _run(tmp_path / "pool", n_workers=3)
    assert serial == pool
    assert len(serial) > 0
