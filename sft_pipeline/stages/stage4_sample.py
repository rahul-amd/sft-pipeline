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
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from sft_pipeline.checkpoint import CheckpointManager
from sft_pipeline.clustering.embedder import load_embeddings
from sft_pipeline.clustering.faiss_index import make_flat_index
from sft_pipeline.config import PipelineConfig
from sft_pipeline.storage import ShardedJSONLWriter, ensure_dir, iter_jsonl_dir

logger = logging.getLogger(__name__)

STAGE = "stage4_sample"


def run_stage4(cfg: PipelineConfig, cm: CheckpointManager) -> None:
    s3 = cfg.stage3_cluster
    s4 = cfg.stage4_sample

    stage3_dir = Path(s3.output_path).parent
    emb_dir = Path(s3.embeddings_dir)
    out_dir = Path(s4.output_path).parent
    ensure_dir(out_dir)

    cm.mark_stage_started(STAGE)

    rng = random.Random(cfg.global_.seed)

    # ------------------------------------------------------------------
    # Step A: Load all clustered prompts into a Polars DataFrame
    # ------------------------------------------------------------------
    logger.info("Stage4: loading clustered prompts from %s", stage3_dir)
    records = list(iter_jsonl_dir(stage3_dir))
    if not records:
        logger.error("Stage4: no prompts found in %s — was Stage 3 run?", stage3_dir)
        cm.mark_stage_failed(STAGE, "No input prompts")
        return

    df = pl.DataFrame(records)
    logger.info("Stage4: loaded %d candidate prompts", len(df))

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
        emb_ids, emb_vectors = load_embeddings(emb_dir)
        id_to_emb_idx = {pid: i for i, pid in enumerate(emb_ids)}
    else:
        logger.warning("Stage4: no embedding shards found in %s — skipping cosine dedup", emb_dir)
        id_to_emb_idx = {}

    # ------------------------------------------------------------------
    # Step C: Quota-enforced sampling per (domain, difficulty) cell
    # ------------------------------------------------------------------
    total_target = s4.total_prompts
    domain_quotas = s4.domain_quotas
    diff_quotas = s4.difficulty_quotas

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

        for diff, diff_frac in diff_quotas.items():
            cell_target = int(domain_target * diff_frac)
            cell_df = domain_df.filter(pl.col("difficulty") == diff)
            n_available = len(cell_df)

            if n_available == 0:
                logger.warning(
                    "Stage4: no prompts for (%s, %s) — trying neighbouring difficulty",
                    domain, diff,
                )
                continue

            # Sample with seed
            n_sample = min(cell_target, n_available)
            cell_ids = cell_df["prompt_id"].to_list()
            rng.shuffle(cell_ids)
            sampled_ids.extend(cell_ids[:n_sample])

            logger.debug(
                "Stage4: (%s, %s): target=%d available=%d sampled=%d",
                domain, diff, cell_target, n_available, n_sample,
            )

    logger.info("Stage4: sampled %d prompts before dedup", len(sampled_ids))

    device = cfg.global_.device

    # ------------------------------------------------------------------
    # Step D: Near-duplicate removal via cosine similarity
    # ------------------------------------------------------------------
    if emb_vectors is not None and len(sampled_ids) > 0:
        sampled_ids = _remove_near_duplicates(
            sampled_ids,
            emb_vectors,
            id_to_emb_idx,
            threshold=s4.dedup_cosine_threshold,
            device=device,
        )
        logger.info("Stage4: %d prompts after cosine dedup", len(sampled_ids))

    # ------------------------------------------------------------------
    # Step E: Sort by embedding similarity (KV-cache optimization)
    # ------------------------------------------------------------------
    if emb_vectors is not None and len(sampled_ids) > 0:
        sampled_ids = _sort_by_similarity(sampled_ids, emb_vectors, id_to_emb_idx, device=device)
        logger.info("Stage4: prompts sorted by embedding similarity")

    # ------------------------------------------------------------------
    # Step F: Write output
    # ------------------------------------------------------------------
    sampled_set = set(sampled_ids)
    # Build a lookup for full records
    id_to_record: dict[str, dict] = {}
    for rec in iter_jsonl_dir(stage3_dir):
        if rec["prompt_id"] in sampled_set:
            id_to_record[rec["prompt_id"]] = rec

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

def _remove_near_duplicates(
    prompt_ids: list[str],
    all_embeddings: np.ndarray,
    id_to_idx: dict[str, int],
    threshold: float,
    batch_size: int = 50_000,
    device: str = "cpu",
) -> list[str]:
    """
    Remove near-duplicates from prompt_ids using cosine similarity.
    Returns a deduplicated list preserving order.
    """
    # Extract embeddings for the sampled set
    valid_ids = [pid for pid in prompt_ids if pid in id_to_idx]
    if not valid_ids:
        return prompt_ids

    vecs = np.array(
        [all_embeddings[id_to_idx[pid]] for pid in valid_ids], dtype=np.float32
    )
    N, D = vecs.shape
    logger.info("Stage4 dedup: building temp FAISS index for %d vectors (device=%s)", N, device)

    index = make_flat_index(vecs, device=device)

    keep_mask = np.ones(N, dtype=bool)
    for i in range(0, N, batch_size):
        batch_end = min(i + batch_size, N)
        batch_vecs = vecs[i:batch_end]
        # k=6: first result is the point itself (distance 1.0)
        distances, indices = index.search(batch_vecs, k=min(6, N))
        for local_idx, (dists, nbrs) in enumerate(zip(distances, indices)):
            global_idx = i + local_idx
            if not keep_mask[global_idx]:
                continue
            for dist, nbr in zip(dists[1:], nbrs[1:]):  # skip self
                if nbr == -1:
                    break
                if dist >= threshold and keep_mask[nbr]:
                    # Remove the later-appearing duplicate
                    if nbr > global_idx:
                        keep_mask[nbr] = False

    kept = [valid_ids[i] for i in range(N) if keep_mask[i]]
    logger.info("Stage4 dedup: %d → %d (removed %d)", N, len(kept), N - len(kept))
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
