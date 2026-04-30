"""
Stage 4 — Prompt Sampling under Compute Constraints.

Reads the clustered prompt pool (Stage 3), enforces domain and difficulty
quotas using centroid-ordered round-robin sampling, and writes the final
set of ~7M prompts ready for teacher-model inference.

Sampling strategy per (domain, difficulty) cell:
  - Group prompts by cluster_id.
  - Per cluster: centroid representative (highest centroid_sim) first,
    then most peripheral prompts (lowest centroid_sim) if more are needed.
  - Round-robin across all clusters until cell quota is met.

This naturally avoids sampling near-duplicates (which cluster near the
centroid) while ensuring broad coverage across all clusters.

If centroid_sim is absent from Stage 3 data (old runs), Stage 4 computes
it from the embeddings, patches the Stage 3 shards in-place, and continues.
"""
from __future__ import annotations

import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from sft_pipeline.checkpoint import CheckpointManager
from sft_pipeline.clustering.embedder import load_embeddings
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
    device = cfg.global_.device

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
    for col, default in [("domain", "general"), ("difficulty", "medium"), ("cluster_id", -1)]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(default).alias(col))

    # ------------------------------------------------------------------
    # Step B: Load embeddings
    # ------------------------------------------------------------------
    emb_vectors: np.ndarray | None = None
    id_to_emb_idx: dict[str, int] = {}
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
        logger.warning(
            "Stage4: no embedding shards found in %s — centroid_sim will use defaults",
            emb_dir,
        )

    # ------------------------------------------------------------------
    # Step B.5: Patch centroid_sim into Stage 3 shards if missing
    # ------------------------------------------------------------------
    needs_patch = (
        "centroid_sim" not in df.columns
        or df["centroid_sim"].is_null().any()
    )
    if needs_patch and emb_vectors is not None:
        logger.info(
            "Stage4: centroid_sim missing from Stage 3 data — computing and patching shards ..."
        )
        t0 = time.time()
        df = _patch_centroid_sims(df, emb_vectors, id_to_emb_idx, stage3_dir, device=device)
        logger.info("Stage4: centroid_sim patch complete (%.1f s)", time.time() - t0)
    elif needs_patch:
        logger.warning(
            "Stage4: centroid_sim missing and no embeddings available — using default 0.5"
        )
        df = df.with_columns(pl.lit(0.5).cast(pl.Float32).alias("centroid_sim"))

    # ------------------------------------------------------------------
    # Step C: Centroid-ordered quota sampling per (domain, difficulty) cell
    # ------------------------------------------------------------------
    total_target = s4.total_prompts
    domain_quotas = s4.domain_quotas
    diff_quotas_map = s4.difficulty_quotas  # domain → {easy/medium/hard → frac}

    def _diff_quotas_for(domain: str) -> dict[str, float]:
        return diff_quotas_map.get(domain) or diff_quotas_map["default"]

    # Normalise quota keys — if a domain in data doesn't appear in config, lump into 'general'
    def _resolve_domain(d: str) -> str:
        return d if d in domain_quotas else "general"

    df = df.with_columns(
        pl.col("domain").map_elements(_resolve_domain, return_dtype=pl.Utf8).alias("domain_resolved")
    )

    sampled_ids: list[str] = []

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

            if n_available < cell_target:
                logger.warning(
                    "Stage4: QUOTA MISS (%s, %s): %d available < %d target"
                    " — shortfall of %d prompts (%.1f%% of cell target)."
                    " Adjust domain_quotas / difficulty_quotas in config.",
                    domain, diff, n_available, cell_target,
                    cell_target - n_available,
                    100.0 * (cell_target - n_available) / cell_target,
                )

            cell_sampled = _sample_cell_with_centroid_ordering(cell_df, cell_target, rng)
            sampled_ids.extend(cell_sampled)

            logger.debug(
                "Stage4: (%s, %s): target=%d available=%d sampled=%d",
                domain, diff, cell_target, n_available, len(cell_sampled),
            )

    n_sampled = len(sampled_ids)
    if n_sampled < total_target:
        logger.warning(
            "Stage4: sampled %d prompts — shortfall of %d vs target %d (%.1f%%)."
            " Check QUOTA MISS warnings above and adjust config quotas.",
            n_sampled, total_target - n_sampled, total_target,
            100.0 * (total_target - n_sampled) / total_target,
        )
    else:
        logger.info("Stage4: sampled %d prompts", n_sampled)

    # ------------------------------------------------------------------
    # Step D: Write output
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
# Sampling helper
# ---------------------------------------------------------------------------

def _sample_cell_with_centroid_ordering(
    cell_df: pl.DataFrame,
    cell_target: int,
    rng: random.Random,
) -> list[str]:
    """
    Sample up to cell_target prompts using centroid-ordered round-robin.

    Per cluster:
      - Round 1: the most central prompt (highest centroid_sim).
      - Round 2+: remaining prompts from most peripheral inward
        (ascending centroid_sim), maximising within-cluster diversity.

    Clusters are shuffled once at the start so no cluster is
    systematically favoured across domains/difficulties.
    """
    pids = cell_df["prompt_id"].to_list()
    cids = cell_df["cluster_id"].cast(pl.Int64).to_list()
    sims = (
        cell_df["centroid_sim"].to_list()
        if "centroid_sim" in cell_df.columns
        else [0.5] * len(pids)
    )

    # Group by cluster_id
    cluster_to_items: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for pid, cid, sim in zip(pids, cids, sims):
        cluster_to_items[int(cid)].append((pid, float(sim)))

    # Build per-cluster ordered queue:
    #   [most_central, most_peripheral, 2nd_most_peripheral, ...]
    cluster_queues: list[list[str]] = []
    for cid in sorted(cluster_to_items.keys()):
        items = cluster_to_items[cid]
        items.sort(key=lambda x: -x[1])  # descending centroid_sim
        if len(items) == 1:
            cluster_queues.append([items[0][0]])
        else:
            # centroid representative first, then periphery → centre
            centroid_pid = items[0][0]
            rest_pids = [x[0] for x in reversed(items[1:])]  # ascending sim
            cluster_queues.append([centroid_pid] + rest_pids)

    # Shuffle cluster order for stochastic diversity
    rng.shuffle(cluster_queues)

    n_available = sum(len(q) for q in cluster_queues)
    if n_available <= cell_target:
        return [pid for q in cluster_queues for pid in q]

    # Round-robin: take one from each cluster per round
    sampled: list[str] = []
    round_n = 0
    while len(sampled) < cell_target:
        added = 0
        for q in cluster_queues:
            if round_n < len(q):
                sampled.append(q[round_n])
                added += 1
                if len(sampled) >= cell_target:
                    break
        if added == 0:
            break
        round_n += 1

    return sampled


# ---------------------------------------------------------------------------
# centroid_sim patch helper (for Stage 3 data produced before centroid_sim
# was added to cluster_prompts())
# ---------------------------------------------------------------------------

def _patch_centroid_sims(
    df: pl.DataFrame,
    all_embeddings: np.ndarray,
    id_to_emb_idx: dict[str, int],
    stage3_dir: Path,
    device: str = "cpu",
) -> pl.DataFrame:
    """
    Compute centroid_sim for all prompts and rewrite Stage 3 shards in-place.

    Cluster centres are computed as the mean of L2-normalised embeddings for
    all prompts assigned to that cluster.  Prompts whose embeddings are absent
    receive a neutral default of 0.5.
    """
    import torch
    import torch.nn.functional as F

    torch_device = torch.device("cuda" if device in ("cuda", "rocm") else "cpu")

    pids = df["prompt_id"].to_list()
    cluster_ids = df["cluster_id"].cast(pl.Int64).to_list()
    N = len(pids)
    D = all_embeddings.shape[1]

    # Map each prompt to its embedding row index (-1 if absent)
    pid_emb_idx = np.array(
        [id_to_emb_idx.get(pid, -1) for pid in pids], dtype=np.int64
    )

    # Collect embedding indices per cluster
    cluster_to_emb_indices: dict[int, list[int]] = defaultdict(list)
    for i, cid in enumerate(cluster_ids):
        if pid_emb_idx[i] >= 0:
            cluster_to_emb_indices[int(cid)].append(int(pid_emb_idx[i]))

    # Compute per-cluster centres (mean of L2-normalised embeddings)
    unique_cids = sorted(cluster_to_emb_indices.keys())
    cid_to_center_idx = {cid: i for i, cid in enumerate(unique_cids)}
    K = len(unique_cids)
    logger.info("Stage4 patch: computing centres for %d clusters (device=%s) ...", K, torch_device)

    centers = np.zeros((K, D), dtype=np.float32)
    for ci_idx, (cid, emb_indices) in enumerate(cluster_to_emb_indices.items()):
        ci = cid_to_center_idx[cid]
        vecs = all_embeddings[emb_indices].astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs /= np.maximum(norms, 1e-8)
        center = vecs.mean(axis=0)
        norm = np.linalg.norm(center)
        centers[ci] = center / max(norm, 1e-8)
        if (ci_idx + 1) % 10_000 == 0 or (ci_idx + 1) == K:
            logger.info("Stage4 patch: computed centres for %d/%d clusters", ci_idx + 1, K)

    # Compute centroid_sim in chunks via torch
    from sft_pipeline.clustering.clusterer import compute_centroid_similarities

    labels_for_sim = np.array(
        [cid_to_center_idx.get(int(cid), 0) for cid in cluster_ids], dtype=np.int64
    )
    # Build embedding matrix for all prompts (zeros for missing)
    valid_mask = pid_emb_idx >= 0
    logger.info(
        "Stage4 patch: building embedding matrix for %d prompts (%d have embeddings) ...",
        N, int(valid_mask.sum()),
    )
    emb_for_sim = np.zeros((N, D), dtype=np.float16)
    emb_for_sim[valid_mask] = all_embeddings[pid_emb_idx[valid_mask]]
    logger.info("Stage4 patch: embedding matrix ready — computing cosine sims ...")

    centroid_sims = compute_centroid_similarities(
        emb_for_sim, labels_for_sim, centers, device=device
    )
    centroid_sims[~valid_mask] = 0.5  # neutral default for prompts without embeddings

    logger.info(
        "Stage4 patch: centroid_sim computed — min=%.3f mean=%.3f max=%.3f",
        float(centroid_sims.min()), float(centroid_sims.mean()), float(centroid_sims.max()),
    )

    # Add column to df
    df = df.with_columns(pl.Series("centroid_sim", centroid_sims.tolist()).cast(pl.Float32))

    # Rewrite Stage 3 shards in-place
    pid_to_sim: dict[str, float] = dict(zip(pids, centroid_sims.tolist()))
    shards = sorted(stage3_dir.glob("part-*.jsonl"))
    logger.info("Stage4 patch: rewriting %d Stage 3 shards with centroid_sim ...", len(shards))
    for i, shard in enumerate(shards, 1):
        updated: list[dict] = []
        for rec in iter_jsonl(shard):
            sim = pid_to_sim.get(rec["prompt_id"])
            if sim is not None:
                rec["centroid_sim"] = sim
            updated.append(rec)
        with open(shard, "w", encoding="utf-8") as fh:
            for rec in updated:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if i % 20 == 0 or i == len(shards):
            logger.info("Stage4 patch: rewrote %d/%d shards", i, len(shards))

    logger.info("Stage4 patch: Stage 3 shards updated — centroid_sim will be present on next run")
    return df
