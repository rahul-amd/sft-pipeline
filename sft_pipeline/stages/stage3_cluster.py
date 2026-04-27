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

import json
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


def _all_prompts_iter(stage1_dir: Path, stage2_dir: Path):
    yield from iter_jsonl_dir(stage1_dir)
    yield from iter_jsonl_dir(stage2_dir)


# ---------------------------------------------------------------------------
# Distributed embedding helper
# ---------------------------------------------------------------------------

def _gpu_preflight() -> dict:
    """
    Lightweight GPU availability check run as a single Ray task before the
    main embedding workers are dispatched.

    Fails fast with a clear ROCm installation message if torch.cuda is
    unavailable, saving the user from seeing the same error 32 times.

    Returns basic diagnostic info on success.
    """
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() returned False on this worker.\n"
            "\n"
            "The sft-pipeline conda env needs PyTorch built against ROCm, not CUDA.\n"
            "Standard CUDA wheels (pip install torch) will not work on AMD GPUs.\n"
            "\n"
            "Install ROCm PyTorch inside the Singularity container:\n"
            "  singularity exec --overlay <overlay>:rw <sif> \\\n"
            "      scripts/run_in_env.sh \\\n"
            "      pip install --user torch \\\n"
            "          --index-url https://download.pytorch.org/whl/rocm6.2\n"
            "(Adjust rocm6.2 to match your cluster's ROCm stack version.)"
        )
    return {
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }


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

    # ── Pre-flight: verify GPU availability on one worker before dispatching all ──
    # Catches CUDA-vs-ROCm mismatches immediately with a useful error message
    # instead of letting all N workers fail with "Found no NVIDIA driver".
    logger.info("Stage3 distributed: running GPU pre-flight check ...")
    preflight_remote = ray.remote(num_gpus=1)(_gpu_preflight)
    try:
        gpu_info = ray.get(preflight_remote.remote())
        logger.info(
            "Stage3 GPU pre-flight OK: %d GPU(s), device=%s, torch=%s",
            gpu_info["device_count"], gpu_info["device_name"], gpu_info["torch_version"],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Stage3 GPU pre-flight failed on a worker node: {exc}"
        ) from exc

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
            batch_size=500,                         # I/O accumulation size
            device=cfg.global_.device,
            output_dir=str(emb_dir),
            gpu_batch_size=s3.embedding_batch_size, # GPU forward-pass cap
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

def run_stage3(
    cfg: PipelineConfig,
    cm: CheckpointManager,
    dump_annotations_path: Path | None = None,
    import_annotations_path: Path | None = None,
) -> None:
    s1 = cfg.stage1_collect
    s2 = cfg.stage2_generate
    s3 = cfg.stage3_cluster

    emb_dir = Path(s3.embeddings_dir)
    idx_path = Path(s3.faiss_index_path)
    out_dir = Path(s3.output_dir)
    ensure_dir(out_dir)
    ensure_dir(emb_dir)

    cm.mark_stage_started(STAGE)

    stage1_dir = Path(s1.output_dir)
    stage2_dir = Path(s2.output_dir)

    # ------------------------------------------------------------------
    # Dump mode fast-path: only needs the raw prompt list — skip all
    # embedding, FAISS, and clustering work.
    # ------------------------------------------------------------------
    if dump_annotations_path is not None:
        from sft_pipeline.clustering.annotator import build_annotation_request

        stage1_shards = list(stage1_dir.glob("part-*.jsonl"))
        if not stage1_shards:
            raise FileNotFoundError(
                f"Stage 3 --dump-annotations: no part-*.jsonl shards found in {stage1_dir}."
            )

        dump_annotations_path = Path(dump_annotations_path)
        dump_annotations_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Stage3 --dump-annotations: streaming prompts from %s + %s → %s",
            stage1_dir, stage2_dir, dump_annotations_path,
        )

        n_written = 0
        with open(dump_annotations_path, "w", encoding="utf-8") as fh:
            for rec in _all_prompts_iter(stage1_dir, stage2_dir):
                fh.write(json.dumps(build_annotation_request(rec)) + "\n")
                n_written += 1
                if n_written % 100_000 == 0:
                    logger.info("Stage3 --dump-annotations: %d records written ...", n_written)

        logger.info(
            "Stage3 --dump-annotations: done — %d annotation requests → %s\n"
            "Re-run with --import-annotations <results.jsonl> to continue.",
            n_written, dump_annotations_path,
        )
        return

    # ------------------------------------------------------------------
    # Step A: Embed all prompts from Stages 1 and 2
    # ------------------------------------------------------------------

    # Validate that at least the Stage 1 input directory has shards before
    # we start a potentially multi-hour embedding job.
    stage1_shards = list(stage1_dir.glob("part-*.jsonl"))
    if not stage1_shards:
        raise FileNotFoundError(
            f"Stage 3: no part-*.jsonl shards found in {stage1_dir}.\n"
            f"  This is derived from cfg.stage1_collect.output_dir:\n"
            f"    {s1.output_dir}\n"
            f"  Make sure Stage 1 has completed and the path above is correct."
        )
    logger.info("Stage3: found %d Stage 1 shard(s) in %s", len(stage1_shards), stage1_dir)

    def _all_prompts():
        yield from _all_prompts_iter(stage1_dir, stage2_dir)

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
            batch_size=500,                         # I/O accumulation size
            device=cfg.global_.device,
            output_dir=emb_dir,
            gpu_batch_size=s3.embedding_batch_size, # GPU forward-pass cap
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
    if algo in ("faiss_kmeans", "flash_kmeans"):
        # Direct k-means on all embeddings — no centroid indirection.
        logger.info(
            "Stage3: %s clustering %d embeddings into %d clusters",
            algo, N, s3.n_clusters,
        )
        cluster_results = cluster_prompts(
            prompt_ids=ids,
            prompts=prompts_text,
            embeddings=vectors,
            algorithm=algo,
            n_clusters=s3.n_clusters,
            device=cfg.global_.device,
            training_sample=s3.faiss_training_sample,
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
    # Step C': LLM annotation (optional)
    # Three modes:
    #   dump   — write annotation requests as JSONL and exit (offline workflow)
    #   import — read pre-computed responses from JSONL; fall back online for misses
    #   online — call the API directly (original behaviour)
    # ------------------------------------------------------------------
    annotation_records = [
        {"prompt_id": pid, "prompt": prompts_text[i]}
        for i, pid in enumerate(ids)
    ]
    annotation_map: dict[str, dict] = {}

    if import_annotations_path is not None:
        # ── Import mode ────────────────────────────────────────────────
        # Parse pre-computed raw responses from JSONL.  For any prompt_id
        # absent from the file, fall back to online inference (if enabled)
        # or use heuristic defaults with a loud warning.
        from sft_pipeline.clustering.annotator import parse_and_validate_annotation

        import_annotations_path = Path(import_annotations_path)
        logger.info("Stage3: importing annotation results from %s", import_annotations_path)

        results_map: dict[str, str] = {}
        n_loaded = 0
        with open(import_annotations_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                results_map[obj["prompt_id"]] = obj["response"]
                n_loaded += 1
                if n_loaded % 100_000 == 0:
                    logger.info("Stage3: loaded %d responses from import file ...", n_loaded)
        logger.info("Stage3: loaded %d responses from import file", n_loaded)

        missing_ids: list[str] = []
        n_parsed = 0
        n_empty = 0
        for rec in annotation_records:
            pid = rec["prompt_id"]
            if pid in results_map:
                raw = results_map[pid]
                if not raw or not raw.strip():
                    # Empty response — leave out of annotation_map so Step D
                    # falls back to the heuristic cluster labels naturally.
                    n_empty += 1
                else:
                    annotation_map[pid] = parse_and_validate_annotation(raw)
                    n_parsed += 1
                    if n_parsed % 100_000 == 0:
                        logger.info("Stage3: parsed %d / %d annotations ...", n_parsed, n_loaded)
            else:
                missing_ids.append(pid)
        if n_empty:
            logger.warning(
                "Stage3: %d empty responses in import file — "
                "heuristic cluster labels will be used for those prompts.",
                n_empty,
            )
        logger.info("Stage3: parsed %d annotations, %d empty, %d missing", n_parsed, n_empty, len(missing_ids))

        if missing_ids:
            logger.warning(
                "Stage3: %d prompt_id(s) missing from import file — "
                "these will be handled separately.",
                len(missing_ids),
            )
            if s3.annotation_enabled:
                logger.warning(
                    "Stage3: annotation_enabled=true — running online inference "
                    "for %d missing prompt(s).",
                    len(missing_ids),
                )
                from sft_pipeline.clustering.annotator import annotate_prompts

                missing_set = set(missing_ids)
                missing_records = [r for r in annotation_records if r["prompt_id"] in missing_set]
                ann_checkpoint = Path(s3.output_dir) / "annotations.parquet"
                online_map = annotate_prompts(
                    prompt_records=missing_records,
                    model=s3.annotation_model,
                    api_base=s3.annotation_api_base,
                    api_key=s3.annotation_api_key,
                    concurrency=s3.annotation_concurrency,
                    max_tokens=s3.annotation_max_tokens,
                    temperature=s3.annotation_temperature,
                    checkpoint_path=ann_checkpoint,
                    checkpoint_every=s3.annotation_checkpoint_every,
                )
                annotation_map.update(online_map)
            else:
                logger.warning(
                    "Stage3: annotation_enabled=false — using heuristic defaults "
                    "(domain='other', difficulty='medium') for %d missing prompt(s). "
                    "Set annotation_enabled: true in config to enable online fallback.",
                    len(missing_ids),
                )

        logger.info("Stage3: import complete — %d records annotated", len(annotation_map))

    elif s3.annotation_enabled:
        # ── Online mode (original behaviour) ──────────────────────────
        from sft_pipeline.clustering.annotator import annotate_prompts

        ann_checkpoint = Path(s3.output_dir) / "annotations.parquet"
        logger.info(
            "Stage3: LLM annotation enabled — %d prompts → %s  (concurrency=%d)",
            len(annotation_records), s3.annotation_api_base, s3.annotation_concurrency,
        )
        annotation_map = annotate_prompts(
            prompt_records=annotation_records,
            model=s3.annotation_model,
            api_base=s3.annotation_api_base,
            api_key=s3.annotation_api_key,
            concurrency=s3.annotation_concurrency,
            max_tokens=s3.annotation_max_tokens,
            temperature=s3.annotation_temperature,
            checkpoint_path=ann_checkpoint,
            checkpoint_every=s3.annotation_checkpoint_every,
        )
        logger.info("Stage3: annotation complete — %d records", len(annotation_map))

    else:
        logger.info("Stage3: LLM annotation disabled (annotation_enabled: false)")

    # ------------------------------------------------------------------
    # Step D: Write enriched output JSONL
    # LLM annotation values take priority over heuristic cluster labels.
    # New fields added when annotation is enabled: topics, language.
    # ------------------------------------------------------------------
    total_written = 0
    with ShardedJSONLWriter(out_dir, shard_size_mb=300) as writer:
        for rec in _all_prompts():
            pid = rec["prompt_id"]
            cluster_info = cluster_map.get(pid, {})
            ann = annotation_map.get(pid, {})
            enriched = {
                **rec,
                # LLM annotation wins over heuristic; heuristic wins over raw record
                "domain": (ann.get("domain")
                           or cluster_info.get("domain")
                           or rec.get("domain", "general")),
                "difficulty": (ann.get("difficulty")
                               or cluster_info.get("difficulty", "medium")),
                "cluster_id": cluster_info.get("cluster_id", -1),
                # Fields only populated when annotation is enabled; empty/default otherwise
                "topics": ann.get("topics", []),
                "language": ann.get("language", "en"),
                "summary": ann.get("summary", ""),
            }
            writer.write(enriched)
            total_written += 1

    cm.mark_stage_complete(STAGE, output_count=total_written)
    logger.info("Stage3 complete: %d prompts enriched with domain+difficulty", total_written)
