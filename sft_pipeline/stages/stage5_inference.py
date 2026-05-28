"""
Stage 5 — Teacher Model Inference.

Reads sampled prompts (Stage 4), generates structured reasoning traces
using a teacher model (Qwen3.5-122B-A10B or similar), and writes responses
with `reasoning` and `answer` fields.

Two execution modes, selected by `stage5_inference.inference_mode`:

  openai_api   (recommended for cluster):
      Async HTTP calls to an already-running vLLM OpenAI-compatible server.
      No GPU required on the pipeline node — all compute is on the vLLM server.
      Semaphore-bounded concurrency; safe to interrupt and resume.
      Run the vLLM server first:  sbatch vllm/slurm_serve_array.sh

  vllm_offline (dev / single-node):
      Loads the model in-process via vLLM's LLM() class.
      Requires vLLM installed and a GPU.  Supports Ray multi-replica.

Checkpointing: every `checkpoint_every` responses are flushed to DuckDB and
the shard is closed.  Re-running the stage resumes from where it left off.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from pathlib import Path

from sft_pipeline.checkpoint import CheckpointManager, ItemStatus, prompt_id
from sft_pipeline.config import PipelineConfig
from sft_pipeline.storage import ShardedJSONLWriter, ensure_dir, iter_jsonl

logger = logging.getLogger(__name__)

STAGE = "stage5_inference"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_stage5(cfg: PipelineConfig, cm: CheckpointManager) -> None:
    s4 = cfg.stage4_sample
    s5 = cfg.stage5_inference

    stage4_dir = Path(s4.output_dir)
    out_dir = Path(s5.output_dir)
    ensure_dir(out_dir)

    cm.mark_stage_started(STAGE)
    cm.preload_processed(STAGE)

    # ------------------------------------------------------------------
    # Load Stage 4 prompts shard-by-shard with progress logs
    # ------------------------------------------------------------------
    shards = sorted(stage4_dir.glob("part-*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"No part-*.jsonl found in {stage4_dir}")

    logger.info("Stage5: loading %d shards from %s ...", len(shards), stage4_dir)
    t0 = time.time()
    all_prompts: list[dict] = []
    for i, shard in enumerate(shards):
        before = len(all_prompts)
        all_prompts.extend(iter_jsonl(shard))
        logger.info(
            "Stage5: loaded shard %d/%d  (%+d records, total %d)  [%.0fs]",
            i + 1, len(shards), len(all_prompts) - before, len(all_prompts),
            time.time() - t0,
        )

    pending = [p for p in all_prompts if not cm.is_processed(p["prompt_id"], STAGE)]

    logger.info(
        "Stage5: %d total prompts, %d already processed, %d pending  [%.0fs]",
        len(all_prompts), len(all_prompts) - len(pending), len(pending),
        time.time() - t0,
    )

    if not pending:
        logger.info("Stage5: all prompts already processed — stage complete")
        cm.mark_stage_complete(STAGE, output_count=cm.processed_count(STAGE))
        return

    already_written = cm.processed_count(STAGE)

    mode = s5.inference_mode
    if mode == "openai_api":
        n_written = _run_openai_api(pending, cfg, cm, out_dir)
    elif s5.n_replicas > 1:
        n_written = _run_ray(pending, cfg, cm, out_dir, already_written)
    else:
        n_written = _run_single_node(pending, cfg, cm, out_dir)

    total_written = already_written + n_written
    cm.mark_stage_complete(STAGE, output_count=total_written)
    logger.info("Stage5 complete: %d responses written", total_written)


# ---------------------------------------------------------------------------
# OpenAI-compatible API mode (async)
# ---------------------------------------------------------------------------

def _run_openai_api(
    pending: list[dict],
    cfg: PipelineConfig,
    cm: CheckpointManager,
    out_dir: Path,
) -> int:
    """Dispatch async inference against an external vLLM /v1 server."""
    return asyncio.run(_run_openai_api_async(pending, cfg, cm, out_dir))


async def _infer_one(
    client,
    model: str,
    record: dict,
    semaphore: asyncio.Semaphore,
    generation_cfg,
    counter: list,  # [done, failed, start_time]
    total: int,
) -> dict:
    """
    Send one prompt to the OpenAI-compatible endpoint and store the raw response.

    The full model output (including any <think>...</think> tokens the model
    produces natively) is stored verbatim in 'raw_response'.  No parsing is
    done here — that is Stage 6's job.

    Note on special tokens: <think> / </think> are regular text tokens for
    all major thinking models and will always appear in message.content.
    Actual HF special tokens like <|im_end|> are stripped by the vLLM server
    by default (pass extra_body={"skip_special_tokens": False} to keep them,
    but this is rarely needed for downstream use).
    """
    from sft_pipeline.inference.prompt_formatter import build_chat_messages

    pid = record["prompt_id"]
    messages = build_chat_messages(record["prompt"])
    raw_response = ""

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=generation_cfg.max_tokens,
                temperature=generation_cfg.temperature,
                top_p=generation_cfg.top_p,
                n=generation_cfg.n_candidates,
            )
            # Pick the longest candidate as the primary response.
            candidates = [choice.message.content or "" for choice in response.choices]
            raw_response = max(candidates, key=len) if candidates else ""
        except Exception as exc:
            logger.warning("Stage5: inference failed for %s: %s", pid, exc)
            counter[1] += 1

    counter[0] += 1
    done = counter[0]
    if done % 1_000 == 0 or done == total:
        elapsed = time.time() - counter[2]
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else float("inf")
        logger.info(
            "Stage5 (openai_api): %d / %d  (%.1f%%)  failures: %d"
            "  rate: %.1f/s  ETA: %.0fs",
            done, total, 100.0 * done / total, counter[1], rate, eta,
        )

    return {
        **record,
        "raw_response": raw_response,
        "teacher_model": model,
    }


async def _run_openai_api_async(
    pending: list[dict],
    cfg: PipelineConfig,
    cm: CheckpointManager,
    out_dir: Path,
) -> int:
    """
    Drive all pending prompts through the OpenAI-compatible API.

    Uses asyncio.as_completed so results are written to disk as soon as they
    arrive, rather than waiting for the full batch.  Checkpoints to DuckDB
    every `checkpoint_every` records so restarts are cheap.
    """
    import httpx
    from openai import AsyncOpenAI

    s5 = cfg.stage5_inference
    total = len(pending)

    # ── httpx client ─────────────────────────────────────────────────────────
    # Size the connection pool to match concurrency so requests don't queue
    # inside httpx.  Raise keep-alive timeout to avoid idle connection drops
    # between bursts of checkpointing.
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=s5.concurrency + 64,
            max_keepalive_connections=s5.concurrency,
        ),
        timeout=httpx.Timeout(s5.request_timeout),
    )
    oai_client = AsyncOpenAI(
        base_url=s5.api_base,
        api_key=s5.api_key or "none",
        http_client=http_client,
    )

    semaphore = asyncio.Semaphore(s5.concurrency)
    # counter: [done, failed, start_time]
    counter: list = [0, 0, time.time()]

    logger.info(
        "Stage5 (openai_api): %d prompts → %s  model=%s  concurrency=%d",
        total, s5.api_base, s5.model, s5.concurrency,
    )

    # ── submit all tasks ──────────────────────────────────────────────────────
    tasks = [
        asyncio.create_task(
            _infer_one(
                oai_client, s5.model, rec, semaphore,
                s5.generation, s5.delimiters, counter, total,
            )
        )
        for rec in pending
    ]

    # ── collect results as they complete, write + checkpoint ─────────────────
    total_written = 0
    checkpoint_buffer: list[tuple[str, ItemStatus, str | None]] = []

    try:
        with ShardedJSONLWriter(out_dir, shard_size_mb=500) as writer:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                writer.write(result)
                total_written += 1
                shard = writer.written_shards[-1] if writer.written_shards else None
                checkpoint_buffer.append(
                    (result["prompt_id"], ItemStatus.SUCCESS, shard)
                )
                if len(checkpoint_buffer) >= s5.checkpoint_every:
                    cm.mark_processed_batch(checkpoint_buffer, STAGE)
                    checkpoint_buffer.clear()
                    logger.info(
                        "Stage5 (openai_api): checkpoint flushed at %d written", total_written
                    )

            if checkpoint_buffer:
                cm.mark_processed_batch(checkpoint_buffer, STAGE)
    finally:
        await http_client.aclose()

    elapsed = time.time() - counter[2]
    logger.info(
        "Stage5 (openai_api): done — %d written, %d failures, %.0fs total  (%.1f/s avg)",
        total_written, counter[1], elapsed, total_written / elapsed if elapsed else 0,
    )
    return total_written


# ---------------------------------------------------------------------------
# Single-node offline vLLM mode
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
        checkpoint_buffer: list[tuple[str, ItemStatus, str | None]] = []

        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            logger.info(
                "Stage5 (vllm_offline): batch %d–%d / %d",
                start + 1, min(start + batch_size, len(pending)), len(pending),
            )

            for result in run_inference_batch(
                prompts=batch,
                model_name=s5.model,
                vllm_engine_cfg=s5.vllm_engine,
                generation_cfg=s5.generation,
                skip_special_tokens=s5.skip_special_tokens,
                device=cfg.global_.device,
            ):
                writer.write(result)
                total_written += 1
                shard = writer.written_shards[-1] if writer.written_shards else None
                checkpoint_buffer.append(
                    (result["prompt_id"], ItemStatus.SUCCESS, shard)
                )

            if len(checkpoint_buffer) >= checkpoint_every:
                cm.mark_processed_batch(checkpoint_buffer, STAGE)
                checkpoint_buffer.clear()
                logger.info("Stage5: checkpoint flushed at %d", total_written)

        if checkpoint_buffer:
            cm.mark_processed_batch(checkpoint_buffer, STAGE)

    return total_written


# ---------------------------------------------------------------------------
# Multi-replica Ray execution (vllm_offline)
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
        ray.init(address="auto")
        logger.info("Connected to Ray cluster")

    ActorClass = build_ray_actor_class()
    if ActorClass is None:
        return _run_single_node(pending, cfg, cm, out_dir)

    actors = [
        ActorClass.remote(
            model_name=s5.model,
            vllm_engine_cfg=s5.vllm_engine,
            generation_cfg=s5.generation,
            skip_special_tokens=s5.skip_special_tokens,
            device=cfg.global_.device,
        )
        for _ in range(n_replicas)
    ]
    logger.info("Stage5: spawned %d Ray inference actors", n_replicas)

    shard_size = math.ceil(len(pending) / n_replicas)
    shards = [
        pending[i * shard_size : (i + 1) * shard_size]
        for i in range(n_replicas)
    ]

    batch_size = s5.batch_size
    checkpoint_every = s5.checkpoint_every
    total_written = 0

    def _actor_batches(actor, shard: list[dict]):
        futures = []
        for start in range(0, len(shard), batch_size):
            futures.append(actor.process_batch.remote(shard[start : start + batch_size]))
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
                    logger.info(
                        "Stage5 (Ray): checkpoint at %d total",
                        already_written + total_written,
                    )

        if checkpoint_buffer:
            cm.mark_processed_batch(checkpoint_buffer, STAGE)

    return total_written
