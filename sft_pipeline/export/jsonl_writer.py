"""
Final dataset export utilities.

Merges filtered output shards into a final sharded JSONL dataset
with a clean, standardized schema. Optionally pushes to HuggingFace Hub.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from sft_pipeline.config import ExportConfig, PipelineConfig
from sft_pipeline.storage import ShardedJSONLWriter, ensure_dir, iter_jsonl

logger = logging.getLogger(__name__)

# Final output record schema — all other fields are dropped
_OUTPUT_FIELDS = {
    "id", "prompt", "reasoning", "answer",
    "domain", "difficulty", "language",
    "source", "teacher_model",
}


def _normalize_record(rec: dict, idx: int) -> dict:
    """Project to the final schema; fill defaults for missing fields."""
    return {
        "id": rec.get("prompt_id", f"gen_{idx:010d}"),
        "prompt": rec.get("prompt", ""),
        "reasoning": rec.get("reasoning", ""),
        "answer": rec.get("answer", ""),
        "domain": rec.get("domain", "general"),
        "difficulty": rec.get("difficulty", "medium"),
        "language": rec.get("language", "en"),
        "source": rec.get("source", "unknown"),
        "teacher_model": rec.get("teacher_model", ""),
    }


def export_final_dataset(cfg: PipelineConfig) -> int:
    """
    Read Stage 6 filtered shards, normalize schema, and write to final path.
    Returns total record count.
    """
    s6_dir = Path(cfg.stage6_filter.output_dir)
    exp = cfg.export
    out_dir = Path(exp.final_jsonl_path).parent
    if out_dir.resolve() == s6_dir.resolve():
        raise ValueError("Export directory must differ from Stage 6 output")
    if not list(s6_dir.glob("part-*.jsonl")):
        raise FileNotFoundError(f"No filtered shards in {s6_dir}")
    ensure_dir(out_dir.parent)

    total = 0
    seen = set()
    with tempfile.TemporaryDirectory(prefix=".sft-export-", dir=out_dir.parent) as tmp:
        with ShardedJSONLWriter(tmp, shard_size_mb=exp.shard_size_mb) as writer:
            for shard in sorted(s6_dir.glob("part-*.jsonl")):
                for rec in iter_jsonl(shard, skip_errors=False):
                    out_rec = _normalize_record(rec, total)
                    if any(not out_rec[k].strip() for k in ("prompt", "reasoning", "answer")):
                        raise ValueError(f"Export record {out_rec['id']} has empty training fields")
                    if out_rec["id"] in seen:
                        raise ValueError(f"Duplicate prompt ID in filtered output: {out_rec['id']}")
                    seen.add(out_rec["id"])
                    writer.write(out_rec)
                    total += 1
            writer.flush()
        if not total:
            raise ValueError("No records survived filtering; refusing an empty final dataset")
        ensure_dir(out_dir)
        new_shards = list(Path(tmp).glob("part-*.jsonl"))
        new_names = {p.name for p in new_shards}
        for shard in new_shards:
            shard.replace(out_dir / shard.name)
        for old in out_dir.glob("part-*.jsonl"):
            if old.name not in new_names:
                old.unlink()

    logger.info("Export complete: %d records in %d shards", total, len(writer.written_shards))

    if exp.push_to_hub and exp.hf_repo_id:
        _push_to_hub(out_dir, exp.hf_repo_id)

    return total


def _push_to_hub(dataset_dir: Path, repo_id: str) -> None:
    """Push the final JSONL shards to a HuggingFace Hub dataset repo."""
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi

        logger.info("Pushing dataset to HuggingFace Hub: %s", repo_id)
        ds = load_dataset("json", data_files=str(dataset_dir / "*.jsonl"), split="train")
        ds.push_to_hub(repo_id, private=True)
        logger.info("Successfully pushed to %s", repo_id)
    except Exception as exc:
        logger.error("Failed to push to Hub: %s", exc)
        raise
