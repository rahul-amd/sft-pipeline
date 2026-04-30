"""
Stage 4 — Prompt Sampling under Compute Constraints.

Reads the clustered prompt pool (Stage 3), enforces domain and difficulty
quotas, removes near-duplicates via embedding cosine similarity, and
sorts the final set by embedding similarity (KV-cache optimization).

The output is a deterministic, deduplicated set of ~7M prompts ready
for teacher-model inference.
"""
from __future__ import annotations

import logging
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from sft_pipeline.checkpoint import CheckpointManager
from sft_pipeline.clustering.embedder import load_embeddings
from sft_pipeline.clustering.faiss_index import make_flat_index
from sft_pipeline.config import PipelineConfig
from sft_pipeline.storage import ShardedJSONLWriter, ensure_dir, iter_jsonl

logger = logging.getLogger(__name__)

STAGE = "stage4_sample"


def run_stage4(cfg: PipelineConfig, cm: CheckpointManager) -> None:
    s3 = cfg.stage3_cluster
    s4 = cfg.stage4_sample

    stage3_dir = Path(s3.output_dir)
    emb_dir = Path(s3.embeddings_dir)
    out_dir = Path(s4.output_dir)
    ensure_dir(out_dir)

    cm.mark_stage_started(STAGE)

    rng = random.Random(cfg.global_.seed)

    # ------------------------------------------------------------------
    # Step A: Load all clustered prompts into a Polars DataFrame
    # ------------------------------------------------------------------
    shards = sorted(stage3_dir.glob("part-*.jsonl"))
    if not shards:
        logger.error("Stage4: no prompts found in %s — was Stage 3 run?", stage3_dir)
        cm.mark_stage_failed(STAGE, "No input prompts")
        return

    logger.info("Stage4: loading clustered prompts from %s (%d shards) ...", stage3_dir, len(shards))
    t0 = time.time()
    records: list[dict] = []
    for i, shard in enumerate(shards, 1):
        shard_records = list(iter_jsonl(shard))
        records.extend(shard_records)
        logger.info(
            "Stage4: loaded shard %d/%d — %d records this shard, %d total (%.1f s)",
            i, len(shards), len(shard_records), len(records), time.time() - t0,
        )

    df = pl.DataFrame(records)
    logger.info("Stage4: all shards loaded — %d candidate prompts in %.1f s", len(df), time.time() - t0)

    # Ensure required columns exist
    for col, default in [("domain", "general"), ("difficulty", "medium")]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(default).alias(col))

    # ------------------------------------------------------------------
    # Step B: Load embeddings for dedup
    # ------------------------------------------------------------------
    emb_ids: list[str] = []
    emb_vectors: np.ndarray | None = None
    emb_shards = list(emb_dir.glob("embeddings_*.parquet")) if emb_dir.is_dir() else []
    if emb_shards:
        logger.info("Stage4: loading embeddings from %s (%d shards) ...", emb_dir, len(emb_shards))
        t0 = time.time()
        emb_ids, emb_vectors = load_embeddings(emb_dir)
        id_to_emb_idx = {pid: i for i, pid in enumerate(emb_ids)}
        logger.info(
            "Stage4: embeddings loaded — %d vectors, shape %s, dtype %s (%.1f s)",
            len(emb_ids), emb_vectors.shape, emb_vectors.dtype, time.time() - t0,
        )
    else:
        logger.warning("Stage4: no embedding shards found in %s — skipping cosine dedup", emb_dir)
        id_to_emb_idx = {}

    # ------------------------------------------------------------------
    # Step C: Quota-enforced sampling per (domain, difficulty) cell
    # ------------------------------------------------------------------
    total_target = s4.total_prompts
    domain_quotas = s4.domain_quotas
    diff_quotas_map = s4.difficulty_quotas  # domain → {easy/medium/hard → frac}

    def _diff_quotas_for(domain: str) -> dict[str, float]:
        return diff_quotas_map.get(domain) or diff_quotas_map["default"]

    sampled_ids: list[str] = []
    known_domains = set(df["domain"].unique().to_list())

    # Normalise quota keys — if a domain in data doesn't appear in config, lump into 'general'
    def _resolve_domain(d: str) -> str:
        return d if d in domain_quotas else "general"

    df = df.with_columns(
        pl.col("domain").map_elements(_resolve_domain, return_dtype=pl.Utf8).alias("domain_resolved")
    )

    for domain, d_frac in domain_quotas.items():
        domain_target = int(total_target * d_frac)
        domain_df = df.filter(pl.col("domain_resolved") == domain)
        if len(domain_df) == 0:
            logger.warning("Stage4: no prompts for domain '%s' — skipping", domain)
            continue

        for diff, diff_frac in _diff_quotas_for(domain).items():
            cell_target = int(domain_target * diff_frac)
            cell_df = domain_df.filter(pl.col("difficulty") == diff)
            n_available = len(cell_df)

            if n_available == 0:
                logger.warning(
                    "Stage4: QUOTA MISS (%s, %s): 0 available, target=%d"
                    " — cell skipped entirely. Adjust quotas in config.",
                    domain, diff, cell_target,
                )
                continue

            cell_ids = cell_df["prompt_id"].to_list()
            if n_available < cell_target:
                logger.warning(
                    "Stage4: QUOTA MISS (%s, %s): %d available < %d target"
                    " — shortfall of %d prompts (%.1f%% of cell target)."
                    " Adjust domain_quotas / difficulty_quotas in config.",
                    domain, diff, n_available, cell_target,
                    cell_target - n_available,
                    100.0 * (cell_target - n_available) / cell_target,
                )

            rng.shuffle(cell_ids)
            sampled_ids.extend(cell_ids[:cell_target])

            logger.debug("Stage4: (%s, %s): target=%d available=%d", domain, diff, cell_target, n_available)

    n_sampled = len(sampled_ids)
    if n_sampled < total_target:
        logger.warning(
            "Stage4: sampled %d prompts — shortfall of %d vs target %d (%.1f%%)."
            " Check QUOTA MISS warnings above and adjust config quotas.",
            n_sampled, total_target - n_sampled, total_target,
            100.0 * (total_target - n_sampled) / total_target,
        )
    else:
        logger.info("Stage4: sampled %d prompts before dedup", n_sampled)

    device = cfg.global_.device

    # ------------------------------------------------------------------
    # Step D: Within-cluster near-duplicate removal
    # ------------------------------------------------------------------
    if emb_vectors is not None and len(sampled_ids) > 0:
        pid_to_cluster: dict[str, int] = dict(
            zip(df["prompt_id"].to_list(), df["cluster_id"].cast(pl.Int64).to_list())
        ) if "cluster_id" in df.columns else {}

        logger.info(
            "Stage4: within-cluster cosine dedup on %d prompts "
            "(threshold=%.2f, device=%s) ...",
            len(sampled_ids), s4.dedup_cosine_threshold, device,
        )
        t0 = time.time()
        sampled_ids = _remove_near_duplicates_within_clusters(
            sampled_ids,
            pid_to_cluster,
            emb_vectors,
            id_to_emb_idx,
            threshold=s4.dedup_cosine_threshold,
            device=device,
        )
        logger.info("Stage4: %d prompts after cosine dedup (%.1f s)", len(sampled_ids), time.time() - t0)

    # ------------------------------------------------------------------
    # Step E: Sort by embedding similarity (KV-cache optimization)
    # ------------------------------------------------------------------
    if emb_vectors is not None and len(sampled_ids) > 0:
        t0 = time.time()
        sampled_ids = _sort_by_similarity(sampled_ids, emb_vectors, id_to_emb_idx, device=device)
        logger.info("Stage4: prompts sorted by embedding similarity (%.1f s)", time.time() - t0)

    # ------------------------------------------------------------------
    # Step F: Write output
    # ------------------------------------------------------------------
    sampled_set = set(sampled_ids)
    logger.info("Stage4: collecting %d sampled records from %d shards for output ...",
                len(sampled_set), len(shards))
    t0 = time.time()
    id_to_record: dict[str, dict] = {}
    for i, shard in enumerate(shards, 1):
        for rec in iter_jsonl(shard):
            if rec["prompt_id"] in sampled_set:
                id_to_record[rec["prompt_id"]] = rec
        if i % 20 == 0 or i == len(shards):
            logger.info("Stage4: scanned %d/%d shards, collected %d records (%.1f s)",
                        i, len(shards), len(id_to_record), time.time() - t0)

    total_written = 0
    with ShardedJSONLWriter(out_dir, shard_size_mb=300) as writer:
        for pid in sampled_ids:
            rec = id_to_record.get(pid)
            if rec:
                writer.write(rec)
                total_written += 1

    cm.mark_stage_complete(STAGE, output_count=total_written)
    logger.info(
        "Stage4 complete: %d prompts written (target was %d)", total_written, total_target
    )


# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------

def _remove_near_duplicates_within_clusters(
    sampled_ids: list[str],
    pid_to_cluster: dict[str, int],
    all_embeddings: np.ndarray,
    id_to_idx: dict[str, int],
    threshold: float,
    device: str = "cpu",
    log_every: int = 10_000,
) -> list[str]:
    """
    Within-cluster near-duplicate removal using torch cosine similarity.

    For each cluster:
      1. Compute pairwise cosine similarities among the cluster's sampled prompts.
      2. Rank prompts by similarity to the cluster centroid (most central = highest priority).
      3. Greedily keep the most central prompt; drop any subsequent prompt whose
         cosine similarity to any kept prompt exceeds `threshold`.

    This is O(C × K²) where C is the number of clusters and K is the average
    cluster size (~70 for 7M prompts / 100K clusters) — fast on CPU and GPU.
    """
    import torch

    torch_device = torch.device("cuda" if device in ("cuda", "rocm") else "cpu")
    logger.info("Stage4 dedup: using device=%s", torch_device)

    # Group sampled indices by cluster
    cluster_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, pid in enumerate(sampled_ids):
        cid = pid_to_cluster.get(pid, -1)
        cluster_to_indices[cid].append(idx)

    keep_mask = np.ones(len(sampled_ids), dtype=bool)
    n_clusters = len(cluster_to_indices)
    n_removed = 0
    t0 = time.time()

    for i, (cid, indices) in enumerate(cluster_to_indices.items()):
        if len(indices) <= 1:
            continue

        pids = [sampled_ids[j] for j in indices]
        valid = [(j, p) for j, p in zip(indices, pids) if p in id_to_idx]
        if len(valid) <= 1:
            continue

        valid_indices, valid_pids = zip(*valid)

        vecs = torch.tensor(
            np.array([all_embeddings[id_to_idx[p]] for p in valid_pids], dtype=np.float32),
            device=torch_device,
        )
        # L2-normalise so dot product = cosine similarity
        vecs = vecs / vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)

        # Centroid similarity — higher means more central / more representative
        centroid = vecs.mean(dim=0)
        centroid = centroid / centroid.norm().clamp(min=1e-8)
        centroid_sim = (vecs @ centroid).cpu().numpy()

        # Pairwise cosine similarity matrix
        sim_matrix = (vecs @ vecs.T).cpu().numpy()

        # Greedy keep: process most-central first; drop anything too similar to a kept prompt
        order = np.argsort(-centroid_sim)
        local_keep = np.ones(len(valid_pids), dtype=bool)
        for a in range(len(order)):
            ia = order[a]
            if not local_keep[ia]:
                continue
            for b in range(a + 1, len(order)):
                ib = order[b]
                if local_keep[ib] and sim_matrix[ia, ib] >= threshold:
                    local_keep[ib] = False

        cluster_removed = int((~local_keep).sum())
        n_removed += cluster_removed
        for j, keep in enumerate(local_keep):
            if not keep:
                keep_mask[valid_indices[j]] = False

        if (i + 1) % log_every == 0 or (i + 1) == n_clusters:
            logger.info(
                "Stage4 dedup: %d/%d clusters processed, %d removed so far (%.1f s)",
                i + 1, n_clusters, n_removed, time.time() - t0,
            )

    kept = [pid for pid, keep in zip(sampled_ids, keep_mask) if keep]
    logger.info(
        "Stage4 dedup: %d → %d (removed %d near-duplicates within clusters)",
        len(sampled_ids), len(kept), n_removed,
    )
    return kept


def _sort_by_similarity(
    prompt_ids: list[str],
    all_embeddings: np.ndarray,
    id_to_idx: dict[str, int],
    device: str = "cpu",
) -> list[str]:
    """
    Greedy nearest-neighbour traversal to sort prompts by similarity.
    Improves vLLM KV-cache utilization during inference.
    Uses a simple greedy approach (not TSP-optimal) for tractability.
    """
    valid_ids = [pid for pid in prompt_ids if pid in id_to_idx]
    if len(valid_ids) < 2:
        return valid_ids

    vecs = np.array(
        [all_embeddings[id_to_idx[pid]] for pid in valid_ids], dtype=np.float32
    )
    N = len(vecs)

    # For very large sets, skip sorting (too slow) and just return as-is
    if N > 500_000:
        logger.info("Stage4: skipping similarity sort for N=%d (too large, >500K)", N)
        return valid_ids

    logger.info(
        "Stage4: running greedy similarity sort on %d prompts (dim=%d, device=%s)",
        N, vecs.shape[1], device,
    )
    index = make_flat_index(vecs, device=device)

    visited = np.zeros(N, dtype=bool)
    order = []
    current = 0  # start from first prompt
    visited[current] = True
    order.append(current)

    for _ in range(N - 1):
        _, nbrs = index.search(vecs[current : current + 1], k=min(20, N))
        moved = False
        for nbr in nbrs[0]:
            if nbr != -1 and not visited[nbr]:
                current = int(nbr)
                visited[current] = True
                order.append(current)
                moved = True
                break
        if not moved:
            # Fall back to first unvisited
            unvisited = np.where(~visited)[0]
            if len(unvisited):
                current = int(unvisited[0])
                visited[current] = True
                order.append(current)

    return [valid_ids[i] for i in order]
