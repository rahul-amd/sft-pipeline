"""
Stage 3 — Prompt Clustering by Domain and Difficulty.

Reads the combined prompt pool (Stages 1 + 2), embeds all prompts,
builds a FAISS IVFFlat index, runs HDBSCAN on centroids to get domain
clusters, assigns difficulty tiers via heuristics, and writes an
enriched JSONL with `domain`, `difficulty`, and `cluster_id` fields.
"""
from __future__ import annotations

import logging
from pathlib import Path

from sft_pipeline.checkpoint import CheckpointManager
from sft_pipeline.clustering.clusterer import cluster_prompts
from sft_pipeline.clustering.embedder import embed_prompts, load_embeddings
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
    if not emb_shards:
        logger.info("Stage3: embedding prompts → %s", emb_dir)
        n_embedded = embed_prompts(
            prompt_iter=_all_prompts(),
            model_name=s3.embedding_model,
            batch_size=s3.embedding_batch_size,
            device=cfg.global_.device,
            output_dir=emb_dir,
        )
        logger.info("Stage3: embedded %d prompts", n_embedded)
    else:
        logger.info("Stage3: %d embedding shards found, skipping re-embed", len(emb_shards))

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
        )
    else:
        logger.info("Stage3: FAISS index found, skipping rebuild")

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
        logger.info("Stage3: using flash_kmeans (GPU) to cluster %d embeddings", N)
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
