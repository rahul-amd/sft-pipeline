"""
Stage 6 parallel filtering equivalence tests.

The filter chain can run across a ProcessPoolExecutor (stage6_filter.n_workers).
These tests assert that the multi-process path produces the same per-record
pass/reject decision as the single-process path — same passes, same rejects —
so parallelism is a pure speedup with no behaviour change.

Note: the parallel path yields results in *completion* order, not input order
(this is what prevents one slow chunk from stalling the pipeline), so results
are compared keyed by prompt_id rather than as ordered lists. llm_judge is
disabled throughout (it is the only non-deterministic, network-dependent
filter).
"""
from __future__ import annotations

from sft_pipeline.config import (
    CodeFilterConfig,
    LLMJudgeConfig,
    MathFilterConfig,
    ReasoningDelimiters,
    Stage6Config,
)
from sft_pipeline.stages.stage6_filter import _iter_filtered

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
    # A spread of domains that route through different filters.
    for i in range(40):
        domain = ["math", "code", "science", "general"][i % 4]
        answer = f"The final answer for item {i} is {i * 3}."
        recs.append(_record(f"p{i:03d}", domain, answer))
    # A record that must be rejected (too short) — exercises the reject path.
    recs.append(_record("short", "general", "No.", reasoning="Too brief."))
    # A code record with a real (passing) python block.
    recs.append(
        _record(
            "code-ok",
            "code",
            "```python\nprint(sum(range(5)))\n```",
        )
    )
    # A code record with a genuine syntax error → code_syntax_error reject.
    recs.append(
        _record(
            "code-bad",
            "code",
            "```python\ndef broken(:\n    return 1\n```",
        )
    )
    return recs


def _cfg() -> Stage6Config:
    return Stage6Config(
        llm_judge=LLMJudgeConfig(enabled=False),
        math=MathFilterConfig(enabled=True),
        code=CodeFilterConfig(enabled=True, sandbox="subprocess", timeout_seconds=5),
    )


def _run(records: list[dict], n_workers: int) -> list[tuple[str, str | None]]:
    s6 = _cfg()
    delimiters = ReasoningDelimiters(think_start="<think>", think_end="</think>")
    out = []
    for rec, reason in _iter_filtered(
        records, s6, delimiters, seed=42, n_workers=n_workers, chunk_size=8
    ):
        out.append((rec.get("prompt_id"), reason))
    return out


def test_parallel_matches_serial():
    records = _mixed_records()
    serial = _run([dict(r) for r in records], n_workers=1)
    parallel = _run([dict(r) for r in records], n_workers=4)
    # Completion order differs from input order — compare by prompt_id.
    assert dict(parallel) == dict(serial)
    assert len(parallel) == len(serial)  # no records dropped or duplicated


def test_serial_rejects_expected():
    results = dict(_run([dict(r) for r in _mixed_records()], n_workers=1))
    # Too-short record is rejected by the structural filter.
    assert results["short"] is not None
    assert results["short"].startswith("structural:")
    # Syntactically broken code is rejected by the code filter.
    assert results["code-bad"] is not None
    assert results["code-bad"].startswith("code:")
    # Valid code passes.
    assert results["code-ok"] is None


def test_all_records_accounted_for():
    # Completion order is not input order, but every input record must appear
    # exactly once in the output.
    records = _mixed_records()
    ids_in = sorted(r["prompt_id"] for r in records)
    ids_out = sorted(pid for pid, _ in _run([dict(r) for r in records], n_workers=4))
    assert ids_out == ids_in
