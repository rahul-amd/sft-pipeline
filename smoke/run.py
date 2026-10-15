#!/usr/bin/env python3
"""Run every stage, validate artifacts, then verify a no-work resume.

Use an already running OpenAI-compatible model server. See smoke/config.yaml.
No stage is mocked or manually imported by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sft_pipeline.checkpoint import CheckpointManager
from sft_pipeline.config import load_config
from sft_pipeline.storage import iter_jsonl_dir


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="smoke/config.yaml")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Pipeline wall-clock deadline in seconds")
    args = parser.parse_args()
    os.chdir(ROOT)
    cfg = load_config(args.config)
    run_dir = Path(cfg.base_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    # A failed revalidation must not leave an old success report behind.
    (run_dir / "smoke_report.json").unlink(missing_ok=True)
    config_bytes = Path(args.config).read_bytes()
    config_copy = run_dir / "config.used.yaml"
    config_copy.write_bytes(config_bytes)
    cmd = [sys.executable, "-m", "sft_pipeline.cli", "run", "--config", str(config_copy)]
    start = time.monotonic()
    print(f"Running pipeline; log: {run_dir / 'pipeline.log'}", flush=True)
    with (run_dir / "pipeline.log").open("w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=args.timeout)

    dirs = {name: getattr(cfg, name).output_dir for name in (
        "stage1_collect", "stage2_generate", "decontaminate", "stage3_cluster",
        "stage4_sample", "stage5_inference", "stage6_filter",
    )}
    dirs["export"] = str(Path(cfg.export.final_jsonl_path).parent)
    records = {name: list(iter_jsonl_dir(path)) for name, path in dirs.items()}
    with CheckpointManager(cfg.global_.checkpoint_db) as cm:
        statuses = cm.all_stage_statuses()
    require({r["stage"] for r in statuses if r["status"] == "completed"} == set(dirs),
            "All eight stages must be complete, including export")
    require(all(records.values()), "Every stage must produce records")
    for name, rows in records.items():
        key = "id" if name == "export" else "prompt_id"
        require(len({r[key] for r in rows}) == len(rows), f"Duplicate IDs in {name}")
    decontam = json.loads(Path(cfg.decontaminate.report_path).read_text())
    require(decontam["total_removed"] >= 1, "The contamination canary was not removed")
    removed_ids = {r["prompt_id"] for r in iter_jsonl_dir(cfg.decontaminate.removed_dir)}
    final_ids = {r["id"] for r in records["export"]}
    require(not (removed_ids & final_ids), "Contaminated prompts reached the final dataset")
    generated_ids = {r["prompt_id"] for r in records["stage2_generate"]}
    if cfg.stage2_generate.grounding_enabled:
        require(all(r.get("source_context") and r.get("reference_answer")
                    and r.get("grounding", {}).get("accepted") is True
                    and r["grounding"].get("evidence_matches_source") is True
                    for r in records["stage2_generate"]), "Generated prompts lack grounding evidence")
        require(all(r.get("source_context") and r.get("reference_answer")
                    for r in records["stage5_inference"] if r["prompt_id"] in generated_ids),
                "Source metadata was lost before filtering")
    decisions = list(iter_jsonl_dir(Path(cfg.stage6_filter.output_dir) / "_decisions"))
    decision_ids = {r["prompt_id"] for r in decisions}
    require(len(decision_ids) == len(decisions) and decision_ids == {
        r["prompt_id"] for r in records["stage5_inference"]}, "Filter decisions do not cover every input exactly once")
    require({r["prompt_id"] for r in decisions if r["passed"]} == {
        r["prompt_id"] for r in records["stage6_filter"]}, "Filter decisions disagree with survivor shards")
    require(not any(r["judge"]["status"] == "error" for r in decisions), "Judge errors occurred during smoke validation")
    require(bool(generated_ids & final_ids), "No generated prompt reached the final dataset")
    require(all(r["prompt"].strip() and r["reasoning"].strip() and r["answer"].strip()
                for r in records["export"]), "Final dataset contains empty training fields")

    sampling_report = json.loads((Path(cfg.stage4_sample.output_dir) /
                                  "sampling_report.json").read_text())
    max_cosine = None
    if cfg.stage4_sample.semantic_dedup_enabled:
        # Independently check all sampled pairs only when deduplication is enabled.
        import numpy as np
        from sft_pipeline.clustering.embedder import load_embeddings
        embedding_ids, vectors = load_embeddings(cfg.stage3_cluster.embeddings_dir)
        positions = {pid: i for i, pid in enumerate(embedding_ids)}
        selected = np.asarray([vectors[positions[r["prompt_id"]]]
                               for r in records["stage4_sample"]], dtype=np.float64)
        selected /= np.linalg.norm(selected, axis=1, keepdims=True)
        similarities = selected @ selected.T
        np.fill_diagonal(similarities, -np.inf)
        max_cosine = float(similarities.max()) if len(selected) > 1 else None
        require(max_cosine is None or max_cosine < cfg.stage4_sample.dedup_cosine_threshold + 1e-6,
                "Sampled prompts violate the semantic deduplication threshold")
        require(sampling_report["duplicate_candidates"] >= 1,
                "The semantic duplicate fixture was not rejected")
    else:
        require(sampling_report["duplicate_candidates"] == 0,
                "Semantic deduplication ran while disabled")

    def hashes():
        paths = {p for path in dirs.values() for p in Path(path).glob("*.jsonl")}
        paths.update((Path(cfg.stage2_generate.output_dir) / "_state/grounding").glob("*.json"))
        paths.update((Path(cfg.stage6_filter.output_dir) / "_decisions").glob("*.jsonl"))
        return {str(p.relative_to(run_dir)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    before = hashes()
    print("Validating no-work resume...", flush=True)
    with (run_dir / "resume.log").open("w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    require(hashes() == before, "Resume changed output artifacts")
    resume = (run_dir / "resume.log").read_text()
    require(resume.count("already complete") == 8, "Resume did not skip all completed stages")

    report = {
        "status": "passed", "counts": {k: len(v) for k, v in records.items()},
        "generated_prompts_in_final": len(generated_ids & final_ids),
        "grounding_enabled": cfg.stage2_generate.grounding_enabled,
        "filter_decisions": len(decisions),
        "decontamination": decontam,
        "filter_report": json.loads(Path(cfg.stage6_filter.report_path).read_text()),
        "resume_unchanged": True, "elapsed_seconds": round(time.monotonic() - start, 2),
        "teacher_model": cfg.stage5_inference.model,
        "embedding_model": cfg.stage3_cluster.embedding_model,
        "max_sampled_pairwise_cosine": max_cosine,
        "semantic_dedup_enabled": cfg.stage4_sample.semantic_dedup_enabled,
        "sampling_report": sampling_report,
        "code_sandbox": cfg.stage6_filter.code.sandbox,
        "sandbox_image": cfg.stage6_filter.code.docker_image,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "versions": {p: importlib.metadata.version(p) for p in
                     ("pydantic", "datasets", "sentence-transformers", "faiss-cpu", "openai")},
        "artifact_sha256": before,
    }
    (run_dir / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "counts", "resume_unchanged",
                                            "generated_prompts_in_final", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
