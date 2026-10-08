"""Explain audit disagreements by replaying existing non-judge filters only."""
from collections import Counter
import json
from pathlib import Path
import random

from sft_pipeline.config import load_config
from sft_pipeline.stages.stage6_filter import _apply_filters
from sft_pipeline.storage import iter_jsonl_dir

root = Path("smoke/runs/freeze_validation/audit-freeze_scale_v2")
cfg = load_config("smoke/config-scale.yaml")
cfg.stage6_filter.llm_judge.enabled = False
verdicts = {r["prompt_id"]: r for r in map(json.loads, (root / "verdicts.jsonl").open())
            if r["pipeline_disposition"] == "rejected"}
results = []
for record in iter_jsonl_dir("smoke/runs/freeze_scale_v2/stage5"):
    pid = record["prompt_id"]
    if pid not in verdicts:
        continue
    reason = _apply_filters(dict(record), cfg.stage6_filter, random.Random(42),
                            cfg.stage5_inference.delimiters)
    results.append({"prompt_id": pid, "audit_verdict": verdicts[pid]["verdict"],
                    "finish_reason": record.get("finish_reason"),
                    "non_judge_replay": reason or "passed_non_judge_replay"})
assert len(results) == 40
report = {"status": "completed", "records": results,
          "audit_correct_replay_counts": dict(Counter(r["non_judge_replay"] for r in results
                                                        if r["audit_verdict"] == "correct")),
          "caveat": "Original per-record judge scores were not saved. Passing this replay suggests judge rejection, but does not exclude nondeterministic code execution. Reviewer correctness judgments are uncalibrated."}
(root / "non-judge-replay.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report["audit_correct_replay_counts"], indent=2))
