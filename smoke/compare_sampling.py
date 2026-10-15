"""Compare optional semantic deduplication on the same annotated smoke pool."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import time

import numpy as np
import yaml

from sft_pipeline.checkpoint import CheckpointManager
from sft_pipeline.clustering.embedder import load_embeddings
from sft_pipeline.config import load_config
from sft_pipeline.stages.stage4_sample import run_stage4
from sft_pipeline.storage import iter_jsonl_dir


def main(source_run):
    source = Path("smoke/runs") / source_run
    root = Path("smoke/runs/freeze_validation")
    source_shards = list((source / "stage3").glob("part-*.jsonl"))
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_shards}
    ids, vectors = load_embeddings(source / "stage3/embeddings")
    positions = {pid: i for i, pid in enumerate(ids)}
    result = {}
    for enabled in (False, True):
        label = "on" if enabled else "off"
        raw = yaml.safe_load(Path("smoke/config-scale.yaml").read_text())
        raw["global"]["run_id"] = f"{source_run}_dedup_{label}"
        raw["stage3_cluster"]["output_dir"] = str(source / "stage3")
        raw["stage3_cluster"]["embeddings_dir"] = str(source / "stage3/embeddings")
        raw["stage4_sample"]["semantic_dedup_enabled"] = enabled
        config = root / f"sampling-{label}.yaml"
        config.write_text(yaml.safe_dump(raw, sort_keys=False))
        cfg = load_config(config)
        Path(cfg.base_path).mkdir(parents=True, exist_ok=False)
        start = time.monotonic()
        with CheckpointManager(cfg.global_.checkpoint_db) as cm:
            run_stage4(cfg, cm)
            assert cm.is_stage_complete("stage4_sample")
        elapsed = time.monotonic() - start
        rows = list(iter_jsonl_dir(cfg.stage4_sample.output_dir))
        chosen = np.asarray([vectors[positions[r["prompt_id"]]] for r in rows], dtype=np.float64)
        chosen /= np.linalg.norm(chosen, axis=1, keepdims=True)
        sims = chosen @ chosen.T
        pairs = sims[np.triu_indices(len(rows), k=1)]
        above = int(np.count_nonzero(pairs >= cfg.stage4_sample.dedup_cosine_threshold + 1e-6))
        if enabled:
            assert above == 0
        else:
            assert [r["prompt_id"] for r in rows] == [r["prompt_id"] for r in iter_jsonl_dir(source / "stage4")]
        report = json.loads((Path(cfg.stage4_sample.output_dir) / "sampling_report.json").read_text())
        assert sum(c["shortfall"] for c in report["cells"]) + report["quota_rounding_shortfall"] == report["shortfall"]
        result[label] = {"sampling_seconds": elapsed, "selected": len(rows),
                         "maximum_pairwise_cosine": float(pairs.max()),
                         "pairs_at_or_above_threshold_plus_tolerance": above,
                         "sampling_report": report}
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_shards}
    (root / "sampling-comparison.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: {a: b for a, b in v.items() if a != "sampling_report"} for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", default="freeze_scale_v2")
    main(parser.parse_args().source_run)
