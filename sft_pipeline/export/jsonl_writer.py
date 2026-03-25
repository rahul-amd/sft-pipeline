"""
Final dataset export utilities.

Merges filtered output shards into a final sharded JSONL dataset
with a clean, standardized schema. Optionally pushes to HuggingFace Hub.
"""
from __future__ import annotations

import logging
from pathlib import Path

from sft_pipeline.config import ExportConfig, PipelineConfig
from sft_pipeline.storage import ShardedJSONLWriter, ensure_dir, iter_jsonl_dir

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
    s6_dir = Path(cfg.stage6_filter.output_path).parent
    exp = cfg.export
    out_dir = Path(exp.final_jsonl_path).parent
    ensure_dir(out_dir)

    total = 0
    with ShardedJSONLWriter(out_dir, shard_size_mb=exp.shard_size_mb) as writer:
        for rec in iter_jsonl_dir(s6_dir):
            out_rec = _normalize_record(rec, total)
            writer.write(out_rec)
            total += 1
            if total % 500_000 == 0:
                logger.info("Export: wrote %d records...", total)

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
