"""Real SIGKILL/retry checks in isolated smoke run directories.

Run after the scale run has produced Stage 4. No scale-run artifacts are edited.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml

from sft_pipeline.checkpoint import CheckpointManager
from sft_pipeline.config import load_config
from sft_pipeline.storage import ShardedJSONLWriter, iter_jsonl_dir

MATRIX = Path("smoke/runs/freeze_validation")


def cli(config, stage, log, kill_writes=False):
    command = ([sys.executable, __file__, "--child", "--config", str(config), "--stage", stage]
               if kill_writes else [sys.executable, "-m", "sft_pipeline.cli", "run-stage", stage, "--config", str(config)])
    output = open(log, "w")
    return subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True), output


def completed(process, stream):
    try:
        result = process.wait(timeout=1200)
        if result:
            raise RuntimeError(f"child returned {result}")
    finally:
        stream.close()


def hashes(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(directory).glob("part-*.jsonl")}


def write_input(directory, records):
    Path(directory).mkdir(parents=True, exist_ok=True)
    with ShardedJSONLWriter(directory) as writer:
        for record in records:
            writer.write(record)
        writer.flush()


def main(source_run):
    raw = yaml.safe_load(Path("smoke/config-scale.yaml").read_text())
    raw["global"]["run_id"] = "freeze_retry_stages34"
    raw["decontaminate"]["enabled"] = False
    raw["stage3_cluster"].update(annotation_enabled=False, n_clusters=4, faiss_nlist=4)
    raw["stage4_sample"].update(total_prompts=64)
    config = MATRIX / "recovery-stages34.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False))
    cfg = load_config(config)
    Path(cfg.base_path).mkdir(parents=True, exist_ok=False)
    rows = list(iter_jsonl_dir(Path("smoke/runs") / source_run / "stage1"))[:160]
    write_input(cfg.stage1_collect.output_dir, rows)
    report = {}
    for stage in ("stage3_cluster", "stage4_sample"):
        out = getattr(cfg, stage).output_dir
        completed(*cli(config, stage, MATRIX / f"recovery-{stage}-baseline.log"))
        before = hashes(out)
        before_ids = {r["prompt_id"] for r in iter_jsonl_dir(out)}
        assert before
        process, stream = cli(config, stage, MATRIX / f"recovery-{stage}-killed.log", kill_writes=True)
        status = process.wait(timeout=300)
        stream.close()
        assert status == -signal.SIGKILL, status
        assert hashes(out) == before, "Interrupted staging changed published output"
        with CheckpointManager(cfg.global_.checkpoint_db) as cm:
            assert not cm.is_stage_complete(stage)
        completed(*cli(config, stage, MATRIX / f"recovery-{stage}-retry.log"))
        after = list(iter_jsonl_dir(out))
        assert len(after) == len({r["prompt_id"] for r in after})
        assert {r["prompt_id"] for r in after} == before_ids
        with CheckpointManager(cfg.global_.checkpoint_db) as cm:
            assert cm.is_stage_complete(stage)
        report[stage] = {"status": "passed", "killed_exit_code": status,
                         "prior_output_preserved": True, "retry_rows": len(after), "unique_ids": True}
    (MATRIX / "recovery-stages34.json").write_text(json.dumps(report, indent=2)+"\n")

    raw["global"]["run_id"] = "freeze_retry_stage5"
    raw["stage5_inference"].update(concurrency=8, checkpoint_every=8, shard_size_mb=1)
    config = MATRIX / "recovery-stage5.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False))
    cfg = load_config(config)
    Path(cfg.base_path).mkdir(parents=True, exist_ok=False)
    rows = list(iter_jsonl_dir(Path("smoke/runs") / source_run / "stage4"))[:96]
    assert len(rows) == 96
    write_input(cfg.stage4_sample.output_dir, rows)
    process, stream = cli(config, "stage5_inference", MATRIX / "recovery-stage5-killed.log")
    deadline = time.monotonic()+600
    killed = False
    while process.poll() is None and time.monotonic() < deadline:
        durable = list(iter_jsonl_dir(cfg.stage5_inference.output_dir))
        if len(durable) >= 16:
            os.killpg(process.pid, signal.SIGKILL)
            killed = True
            break
        time.sleep(1)
    if not killed:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        raise RuntimeError("No progress within recovery-test deadline")
    status = process.wait(timeout=30)
    stream.close()
    assert status == -signal.SIGKILL, status
    before = {r["prompt_id"]: r for r in iter_jsonl_dir(cfg.stage5_inference.output_dir)}
    assert 16 <= len(before) < len(rows)
    completed(*cli(config, "stage5_inference", MATRIX / "recovery-stage5-retry.log"))
    after_rows = list(iter_jsonl_dir(cfg.stage5_inference.output_dir))
    after = {r["prompt_id"]: r for r in after_rows}
    assert len(after_rows) == len(after) == len(rows)
    assert set(after) == {r["prompt_id"] for r in rows}
    assert all(after[pid] == record for pid, record in before.items())
    with CheckpointManager(cfg.global_.checkpoint_db) as cm:
        assert cm.is_stage_complete("stage5_inference")
    (MATRIX / "recovery-stage5.json").write_text(json.dumps({"status": "passed",
        "killed_exit_code": status, "durable_before_kill": len(before), "after_retry": len(after),
        "unique_ids": True, "durable_responses_unchanged": True}, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--stage")
    parser.add_argument("--source-run", default="freeze_scale_v2")
    args = parser.parse_args()
    if args.child:
        original = ShardedJSONLWriter.write
        writes = 0
        def interrupted(writer, record):
            global writes
            original(writer, record)
            writes += 1
            if writes == 2:
                os.kill(os.getpid(), signal.SIGKILL)
        ShardedJSONLWriter.write = interrupted
        from sft_pipeline.cli import _dispatch_stage
        cfg = load_config(args.config)
        with CheckpointManager(cfg.global_.checkpoint_db) as cm:
            _dispatch_stage(args.stage, cfg, cm)
    else:
        main(args.source_run)
