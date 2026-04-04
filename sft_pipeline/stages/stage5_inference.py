"""
Stage 5 — Teacher Model Inference.

Reads sampled prompts (Stage 4), generates structured reasoning traces
using Qwen3.5-122B-A10B (or configured model) via vLLM, and writes
responses with reasoning + answer fields.

Supports two execution modes:
  - Single-node: runs vLLM directly in the current process (dev / small runs)
  - Multi-node: spawns Ray actors (one per vLLM replica) for 512-GPU cluster

Checkpointing: every `checkpoint_every` prompts, processed item records
are flushed to DuckDB and the shard is finalized. On resume, already-
processed prompt_ids are skipped.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterator

from sft_pipeline.checkpoint import CheckpointManager, ItemStatus, prompt_id
from sft_pipeline.config import PipelineConfig
from sft_pipeline.storage import ShardedJSONLWriter, ensure_dir, iter_jsonl_dir

logger = logging.getLogger(__name__)

STAGE = "stage5_inference"


def run_stage5(cfg: PipelineConfig, cm: CheckpointManager) -> None:
    s4 = cfg.stage4_sample
    s5 = cfg.stage5_inference

    stage4_dir = Path(s4.output_dir)
    out_dir = Path(s5.output_dir)
    ensure_dir(out_dir)

    cm.mark_stage_started(STAGE)
    cm.preload_processed(STAGE)

    # Load all sampled prompts, filtering already-processed ones
    all_prompts = list(iter_jsonl_dir(stage4_dir))
    pending = [p for p in all_prompts if not cm.is_processed(p["prompt_id"], STAGE)]

    logger.info(
        "Stage5: %d total prompts, %d already processed, %d pending",
        len(all_prompts), len(all_prompts) - len(pending), len(pending),
    )

    if not pending:
        logger.info("Stage5: all prompts already processed — stage complete")
        cm.mark_stage_complete(STAGE, output_count=cm.processed_count(STAGE))
        return

    total_written = cm.processed_count(STAGE)

    if s5.n_replicas > 1:
        # Multi-replica Ray execution
        total_written += _run_ray(pending, cfg, cm, out_dir, total_written)
    else:
        # Single-node execution (dev / small runs)
        total_written += _run_single_node(pending, cfg, cm, out_dir)

    cm.mark_stage_complete(STAGE, output_count=total_written)
    logger.info("Stage5 complete: %d responses written", total_written)


# ---------------------------------------------------------------------------
# Single-node execution
# ---------------------------------------------------------------------------

def _run_single_node(
    pending: list[dict],
    cfg: PipelineConfig,
    cm: CheckpointManager,
    out_dir: Path,
) -> int:
    from sft_pipeline.inference.vllm_batch import run_inference_batch

    s5 = cfg.stage5_inference
    batch_size = s5.batch_size
    checkpoint_every = s5.checkpoint_every
    total_written = 0

    with ShardedJSONLWriter(out_dir, shard_size_mb=500) as writer:
        batch_buffer: list[dict] = []
        checkpoint_buffer: list[tuple[str, ItemStatus, str | None]] = []

        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]

            for result in run_inference_batch(
                prompts=batch,
                model_name=s5.model,
                vllm_engine_cfg=s5.vllm_engine,
                generation_cfg=s5.generation,
                delimiters=s5.delimiters,
                device=cfg.global_.device,
            ):
                writer.write(result)
                total_written += 1
                shard = writer.written_shards[-1] if writer.written_shards else None
                checkpoint_buffer.append(
                    (result["prompt_id"], ItemStatus.SUCCESS, shard)
                )

            # Flush checkpoint every N processed
            if len(checkpoint_buffer) >= checkpoint_every:
                cm.mark_processed_batch(checkpoint_buffer, STAGE)
                checkpoint_buffer.clear()
                logger.info("Stage5: checkpoint flushed at %d", total_written)

        if checkpoint_buffer:
            cm.mark_processed_batch(checkpoint_buffer, STAGE)

    return total_written


# ---------------------------------------------------------------------------
# Multi-replica Ray execution
# ---------------------------------------------------------------------------

def _run_ray(
    pending: list[dict],
    cfg: PipelineConfig,
    cm: CheckpointManager,
    out_dir: Path,
    already_written: int,
) -> int:
    try:
        import ray
    except ImportError:
        logger.error("Ray not installed. Install with: pip install ray[default]")
        logger.info("Falling back to single-node execution")
        return _run_single_node(pending, cfg, cm, out_dir)

    from sft_pipeline.inference.vllm_batch import build_ray_actor_class

    s5 = cfg.stage5_inference
    n_replicas = s5.n_replicas

    if not ray.is_initialized():
        ray.init(address="auto")  # connects to running Ray cluster
        logger.info("Connected to Ray cluster")

    ActorClass = build_ray_actor_class()
    if ActorClass is None:
        return _run_single_node(pending, cfg, cm, out_dir)

    # Spawn actors
    actors = [
        ActorClass.remote(
            model_name=s5.model,
            vllm_engine_cfg=s5.vllm_engine,
            generation_cfg=s5.generation,
            delimiters=s5.delimiters,
            device=cfg.global_.device,
        )
        for _ in range(n_replicas)
    ]
    logger.info("Stage5: spawned %d Ray inference actors", n_replicas)

    # Shard prompts across actors
    shard_size = math.ceil(len(pending) / n_replicas)
    shards = [
        pending[i * shard_size : (i + 1) * shard_size]
        for i in range(n_replicas)
    ]

    # Each actor processes its shard in sub-batches, yielding results
    batch_size = s5.batch_size
    checkpoint_every = s5.checkpoint_every
    total_written = 0

    def _actor_batches(actor, shard: list[dict]):
        """Submit all batches for one actor and collect results."""
        futures = []
        for start in range(0, len(shard), batch_size):
            batch = shard[start : start + batch_size]
            futures.append(actor.process_batch.remote(batch))
        results = []
        for fut in futures:
            results.extend(ray.get(fut))
        return results

    all_futures = [
        (_actor_batches, actors[i], shards[i])
        for i in range(len(actors))
        if shards[i]
    ]

    checkpoint_buffer: list[tuple[str, ItemStatus, str | None]] = []

    with ShardedJSONLWriter(out_dir, shard_size_mb=500) as writer:
        for fn, actor, shard in all_futures:
            results = fn(actor, shard)
            for result in results:
                writer.write(result)
                total_written += 1
                shard_path = writer.written_shards[-1] if writer.written_shards else None
                checkpoint_buffer.append(
                    (result["prompt_id"], ItemStatus.SUCCESS, shard_path)
                )
                if len(checkpoint_buffer) >= checkpoint_every:
                    cm.mark_processed_batch(checkpoint_buffer, STAGE)
                    checkpoint_buffer.clear()
                    logger.info("Stage5 (Ray): checkpoint at %d total", already_written + total_written)

        if checkpoint_buffer:
            cm.mark_processed_batch(checkpoint_buffer, STAGE)

    return total_written
