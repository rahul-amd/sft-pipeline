"""
Stage 3 — Prompt Clustering by Domain and Difficulty.

Reads the combined prompt pool (Stages 1 + 2), embeds all prompts,
builds a FAISS IVFFlat index, runs HDBSCAN/flash_kmeans to get domain
clusters, assigns difficulty tiers via heuristics, and writes an
enriched JSONL with `domain`, `difficulty`, and `cluster_id` fields.

Distributed mode
----------------
When `stage3_cluster.distributed: true`, the embedding step (Step A) is
parallelised across Ray workers — one worker per chunk of input JSONL
shards, each running on a dedicated GPU.  Steps B–D (FAISS index build,
clustering, output write) always run on the head node.

Resume
------
Single-node:  if any embeddings_*.parquet shards exist in emb_dir, the
              entire embedding step is skipped.
Distributed:  each worker checks for a per-worker sentinel file
              embeddings_w{id:02d}.done; workers whose sentinel exists are
              skipped and their existing shards are used as-is.
"""
from __future__ import annotations

import logging
from pathlib import Path

from sft_pipeline.checkpoint import CheckpointManager
from sft_pipeline.clustering.clusterer import cluster_prompts
from sft_pipeline.clustering.embedder import embed_jsonl_shards, embed_prompts, load_embeddings
from sft_pipeline.clustering.faiss_index import (
    build_and_save,
    get_centroids,
    load_index,
)
from sft_pipeline.config import PipelineConfig
from sft_pipeline.storage import (
    ShardedJSONLWriter,
    ensure_dir,
    iter_jsonl_dir,
)

logger = logging.getLogger(__name__)

STAGE = "stage3_cluster"


# ---------------------------------------------------------------------------
# Distributed embedding helper
# ---------------------------------------------------------------------------

def _embed_distributed(
    stage1_dir: Path,
    stage2_dir: Path,
    emb_dir: Path,
    cfg: PipelineConfig,
) -> None:
    """
    Distribute embedding across Ray workers.

    Collects all part-*.jsonl shards from stage1_dir and stage2_dir,
    splits them round-robin into n_embedding_workers chunks, and submits
    one Ray remote task per chunk.  Each task calls embed_jsonl_shards()
    on its chunk and writes:
      embeddings_w{id:02d}_{shard:04d}.parquet  — embedding Parquet shards
      embeddings_w{id:02d}.done                  — sentinel on success

    Workers whose sentinel already exists are skipped (resume-safe).

    Ray is initialised with address=cfg.global_.ray_address.
    Each task requests num_gpus=1 to get a dedicated GCD on the cluster.
    """
    import ray

    s3 = cfg.stage3_cluster
    n = s3.n_embedding_workers

    # Collect + sort all input shards deterministically
    all_shards = sorted(stage1_dir.glob("part-*.jsonl")) + sorted(stage2_dir.glob("part-*.jsonl"))
    if not all_shards:
        logger.warning("Stage3 distributed: no input JSONL shards found in %s or %s",
                       stage1_dir, stage2_dir)
        return

    # Round-robin split → each worker gets every N-th shard, interleaved for
    # roughly equal file sizes across workers.
    chunks: list[list[str]] = [[] for _ in range(n)]
    for i, shard in enumerate(all_shards):
        chunks[i % n].append(str(shard))

    logger.info(
        "Stage3 distributed: %d input shards → %d workers (%s shards each)",
        len(all_shards), n,
        "/".join(str(len(c)) for c in chunks),
    )

    ray.init(address=cfg.global_.ray_address, ignore_reinit_error=True)

    # Wrap embed_jsonl_shards as a Ray remote task requiring 1 GPU per worker.
    # num_cpus=4 reserves enough CPUs for the sentence-transformer DataLoader.
    embed_remote = ray.remote(num_gpus=1, num_cpus=4)(embed_jsonl_shards)

    futures: dict = {}
    skipped = 0
    for worker_id, chunk in enumerate(chunks):
        if not chunk:
            continue
        sentinel = emb_dir / f"embeddings_w{worker_id:02d}.done"
        if sentinel.exists():
            logger.info("Stage3: worker %d already done (sentinel found), skipping.", worker_id)
            skipped += 1
            continue
        future = embed_remote.remote(
            jsonl_paths=chunk,
            worker_id=worker_id,
            model_name=s3.embedding_model,
            batch_size=s3.embedding_batch_size,
            device=cfg.global_.device,
            output_dir=str(emb_dir),
        )
        futures[future] = worker_id

    total = len(futures) + skipped
    logger.info(
        "Stage3 distributed: %d workers total — %d submitted, %d already done.",
        total, len(futures), skipped,
    )

    done = skipped
    failed = 0
    remaining = list(futures.keys())
    while remaining:
        ready, remaining = ray.wait(remaining, num_returns=1, timeout=None)
        future = ready[0]
        worker_id = futures[future]
        try:
            result = ray.get(future)
            done += 1
            logger.info(
                "Stage3 embedding [%d/%d] ✓  worker %d — %d prompts embedded",
                done, total, result["worker_id"], result["n_embedded"],
            )
        except Exception as exc:
            failed += 1
            done += 1
            logger.error(
                "Stage3 embedding [%d/%d] ✗  worker %d — %s",
                done, total, worker_id, exc,
            )

    if failed:
        raise RuntimeError(
            f"Stage3: {failed}/{len(futures)} embedding workers failed. "
            "Check logs above. Re-run to resume — completed workers are skipped."
        )

    logger.info("Stage3 distributed embedding complete.")


# ---------------------------------------------------------------------------
# Main stage entry point
# ---------------------------------------------------------------------------

def run_stage3(cfg: PipelineConfig, cm: CheckpointManager) -> None:
    s1 = cfg.stage1_collect
    s2 = cfg.stage2_generate
    s3 = cfg.stage3_cluster

    emb_dir = Path(s3.embeddings_dir)
    idx_path = Path(s3.faiss_index_path)
    out_dir = Path(s3.output_path).parent
    ensure_dir(out_dir)
    ensure_dir(emb_dir)

    cm.mark_stage_started(STAGE)

    # ------------------------------------------------------------------
    # Step A: Embed all prompts from Stages 1 and 2
    # ------------------------------------------------------------------
    stage1_dir = Path(s1.output_path).parent
    stage2_dir = Path(s2.output_path).parent

    def _all_prompts():
        yield from iter_jsonl_dir(stage1_dir)
        yield from iter_jsonl_dir(stage2_dir)

    emb_shards = list(emb_dir.glob("embeddings_*.parquet"))
    if emb_shards:
        logger.info("Stage3: %d embedding shards found, skipping re-embed", len(emb_shards))
    elif s3.distributed:
        logger.info("Stage3: distributed embedding — dispatching %d Ray workers", s3.n_embedding_workers)
        _embed_distributed(stage1_dir, stage2_dir, emb_dir, cfg)
    else:
        logger.info("Stage3: single-node embedding → %s", emb_dir)
        n_embedded = embed_prompts(
            prompt_iter=_all_prompts(),
            model_name=s3.embedding_model,
            batch_size=s3.embedding_batch_size,
            device=cfg.global_.device,
            output_dir=emb_dir,
        )
        logger.info("Stage3: embedded %d prompts", n_embedded)

    # ------------------------------------------------------------------
    # Step B: Build FAISS index
    # ------------------------------------------------------------------
    ids, vectors = load_embeddings(emb_dir)
    N = len(ids)
    logger.info("Stage3: loaded %d embedding vectors", N)

    if not idx_path.exists():
        build_and_save(
            ids=ids,
            vectors=vectors,
            index_path=idx_path,
            index_type=s3.faiss_index_type,
            nlist=s3.faiss_nlist,
            nprobe=s3.faiss_nprobe,
            training_sample=s3.faiss_training_sample,
            device=cfg.global_.device,
        )
    else:
        logger.info("Stage3: FAISS index found, skipping rebuild")

    # Load on CPU — get_centroids() needs a CPU index; Stage 4 handles GPU
    faiss_index = load_index(idx_path)

    # ------------------------------------------------------------------
    # Step C: Cluster and label
    # ------------------------------------------------------------------
    # Load all prompt texts into memory (needed for domain inference)
    id_to_prompt: dict[str, str] = {}
    for rec in _all_prompts():
        id_to_prompt[rec["prompt_id"]] = rec["prompt"]
    prompts_text = [id_to_prompt.get(pid, "") for pid in ids]

    algo = s3.clustering_algorithm
    if algo == "flash_kmeans":
        # Cluster ALL embeddings directly on GPU — no centroid indirection.
        logger.info("Stage3: using flash_kmeans (GPU) to cluster %d embeddings into %d clusters",
                    N, s3.n_clusters)
        cluster_results = cluster_prompts(
            prompt_ids=ids,
            prompts=prompts_text,
            embeddings=vectors,
            algorithm="flash_kmeans",
            n_clusters=s3.n_clusters,
            device=cfg.global_.device,
        )
    else:
        # Centroid-based path (hdbscan / kmeans, CPU).
        centroids = get_centroids(faiss_index)
        if centroids is not None:
            cluster_results = cluster_prompts(
                prompt_ids=ids,
                prompts=prompts_text,
                embeddings=vectors,
                centroids=centroids,
                faiss_index=faiss_index,
                algorithm=algo,
                min_cluster_size=s3.hdbscan_min_cluster_size,
                n_clusters=s3.n_clusters,
            )
        else:
            logger.warning(
                "Stage3: FAISS index has no centroids (Flat index?). "
                "Falling back to heuristic-only domain assignment."
            )
            from sft_pipeline.clustering.clusterer import score_difficulty
            from sft_pipeline.stages.stage1_collect import _infer_domain

            cluster_results = [
                {
                    "prompt_id": pid,
                    "domain": _infer_domain(prompts_text[i]),
                    "difficulty": score_difficulty(prompts_text[i]),
                    "cluster_id": -1,
                }
                for i, pid in enumerate(ids)
            ]

    # Build lookup: prompt_id → cluster result
    cluster_map: dict[str, dict] = {r["prompt_id"]: r for r in cluster_results}

    # ------------------------------------------------------------------
    # Step D: Write enriched output JSONL
    # ------------------------------------------------------------------
    total_written = 0
    with ShardedJSONLWriter(out_dir, shard_size_mb=300) as writer:
        for rec in _all_prompts():
            pid = rec["prompt_id"]
            cluster_info = cluster_map.get(pid, {})
            enriched = {
                **rec,
                "domain": cluster_info.get("domain", rec.get("domain", "general")),
                "difficulty": cluster_info.get("difficulty", "medium"),
                "cluster_id": cluster_info.get("cluster_id", -1),
            }
            writer.write(enriched)
            total_written += 1

    cm.mark_stage_complete(STAGE, output_count=total_written)
    logger.info("Stage3 complete: %d prompts enriched with domain+difficulty", total_written)
