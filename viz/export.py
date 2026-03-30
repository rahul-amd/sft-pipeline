"""
Export pipeline run outputs to a single snapshot.parquet for the viz app.

Usage
-----
    python viz/export.py --run-dir /path/to/run --out viz/data/snapshot.parquet

Options
-------
    --run-dir   Path to the pipeline run directory (the base_path in config).
    --out       Output path for the snapshot Parquet file.
                Default: viz/data/snapshot.parquet
    --sample    Maximum number of prompts to include. Default: 50000
    --seed      Random seed for sampling. Default: 42

The script auto-detects which stages are complete by probing for output
directories, and joins available data (cluster labels, embeddings → UMAP,
responses, filter results) in a single pass. Stages that aren't done yet
are silently skipped; their columns will be absent from the snapshot.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Allow importing sft_pipeline from the project root.
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def _read_stage_jsonl(stage_dir: Path) -> list[dict]:
    from sft_pipeline.storage import iter_jsonl_dir
    return list(iter_jsonl_dir(stage_dir))


def _sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    if len(rows) <= n:
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, n)


def _compute_umap(
    sampled_ids: list[str],
    embeddings_dir: Path,
    seed: int,
) -> dict[str, tuple[float, float]] | None:
    """
    Load embeddings for sampled_ids, run UMAP, return {prompt_id: (x, y)}.
    Returns None if embeddings are unavailable or UMAP fails.
    """
    shards = list(embeddings_dir.glob("embeddings_*.parquet"))
    if not shards:
        logger.info("  No embedding shards found in %s — skipping UMAP.", embeddings_dir)
        return None

    try:
        import numpy as np
        import polars as pl
        import umap as umap_lib

        id_set = set(sampled_ids)
        logger.info("  Loading embeddings for %d sampled prompts …", len(id_set))
        emb_df = (
            pl.scan_parquet(str(embeddings_dir / "embeddings_*.parquet"))
            .filter(pl.col("prompt_id").is_in(list(id_set)))
            .collect()
        )

        if emb_df.is_empty():
            logger.warning("  No embeddings matched sampled prompt IDs.")
            return None

        ids_found = emb_df["prompt_id"].to_list()
        vectors = np.array(emb_df["embedding"].to_list(), dtype=np.float32)
        logger.info(
            "  Running UMAP on %d vectors (dim=%d) — this may take a minute …",
            len(vectors), vectors.shape[1],
        )
        reducer = umap_lib.UMAP(
            n_components=2, n_neighbors=15, min_dist=0.1,
            random_state=seed, verbose=False,
        )
        coords = reducer.fit_transform(vectors)
        logger.info("  UMAP complete.")
        return {pid: (float(coords[i, 0]), float(coords[i, 1])) for i, pid in enumerate(ids_found)}

    except Exception as exc:
        logger.warning("  UMAP failed (%s) — cluster scatter will be unavailable.", exc)
        return None


def export(run_dir: Path, out_path: Path, sample: int, seed: int) -> None:
    run_dir = run_dir.resolve()
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta: dict = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "stages": {},
    }

    # ------------------------------------------------------------------
    # 1. Prompts — prefer Stage 3 (enriched with domain/difficulty/cluster_id)
    #    fall back to Stage 1 if Stage 3 not yet done.
    # ------------------------------------------------------------------
    stage3_dir = run_dir / "stage3"
    stage1_dir = run_dir / "stage1"

    if list(stage3_dir.glob("part-*.jsonl")):
        logger.info("Reading Stage 3 enriched prompts from %s …", stage3_dir)
        rows = _read_stage_jsonl(stage3_dir)
        meta["stages"]["stage3"] = True
        meta["stages"]["stage1"] = True
    elif list(stage1_dir.glob("part-*.jsonl")):
        logger.info("Stage 3 not found — reading Stage 1 prompts from %s …", stage1_dir)
        rows = _read_stage_jsonl(stage1_dir)
        meta["stages"]["stage1"] = True
    else:
        logger.error("No Stage 1 or Stage 3 output found in %s", run_dir)
        sys.exit(1)

    meta["total_prompts"] = len(rows)
    logger.info("  %d prompts loaded.", len(rows))

    rows = _sample(rows, sample, seed)
    meta["sample_size"] = len(rows)
    logger.info("  Sampled %d prompts (seed=%d).", len(rows), seed)

    sampled_ids = [r["prompt_id"] for r in rows]

    # ------------------------------------------------------------------
    # 2. UMAP from Stage 3 embeddings
    # ------------------------------------------------------------------
    embeddings_dir = stage3_dir / "embeddings"
    umap_coords: dict[str, tuple[float, float]] | None = None
    if embeddings_dir.exists():
        umap_coords = _compute_umap(sampled_ids, embeddings_dir, seed)
        if umap_coords:
            meta["stages"]["umap"] = True

    # ------------------------------------------------------------------
    # 3. Stage 5 responses
    # ------------------------------------------------------------------
    stage5_dir = run_dir / "stage5"
    resp_lookup: dict[str, dict] = {}
    if list(stage5_dir.glob("part-*.jsonl")):
        logger.info("Reading Stage 5 responses …")
        for rec in _read_stage_jsonl(stage5_dir):
            pid = rec.get("prompt_id")
            if pid:
                resp_lookup[pid] = rec
        logger.info("  %d responses loaded.", len(resp_lookup))
        meta["stages"]["stage5"] = True

    # ------------------------------------------------------------------
    # 4. Stage 6 filter results
    # ------------------------------------------------------------------
    stage6_dir = run_dir / "stage6"
    filter_lookup: dict[str, bool] = {}
    if list(stage6_dir.glob("part-*.jsonl")):
        logger.info("Reading Stage 6 filter results …")
        for rec in _read_stage_jsonl(stage6_dir):
            pid = rec.get("prompt_id")
            if pid and "passed_filters" in rec:
                filter_lookup[pid] = bool(rec["passed_filters"])
        logger.info("  %d filter results loaded.", len(filter_lookup))
        meta["stages"]["stage6"] = True

    # ------------------------------------------------------------------
    # 5. Assemble final records
    # ------------------------------------------------------------------
    logger.info("Assembling snapshot …")
    final_rows = []
    for rec in rows:
        pid = rec["prompt_id"]
        row: dict = {
            "prompt_id": pid,
            "prompt": rec.get("prompt", ""),
            "source": rec.get("source", ""),
            "domain": rec.get("domain", "general"),
            "difficulty": rec.get("difficulty"),
            "cluster_id": rec.get("cluster_id"),
        }
        if umap_coords and pid in umap_coords:
            row["umap_x"], row["umap_y"] = umap_coords[pid]
        else:
            row["umap_x"] = row["umap_y"] = None
        if pid in resp_lookup:
            row["reasoning"] = resp_lookup[pid].get("reasoning")
            row["answer"] = resp_lookup[pid].get("answer")
        else:
            row["reasoning"] = row["answer"] = None
        row["passed_filters"] = filter_lookup.get(pid)
        final_rows.append(row)

    # ------------------------------------------------------------------
    # 6. Write Parquet + meta.json
    # ------------------------------------------------------------------
    import polars as pl

    df = pl.DataFrame(final_rows)
    df.write_parquet(str(out_path), compression="zstd")
    logger.info("Snapshot written → %s  (%d rows, %d columns)", out_path, len(df), len(df.columns))

    meta_path = out_path.parent / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Metadata written → %s", meta_path)
    logger.info("Stages present: %s", [k for k, v in meta["stages"].items() if v])


def main() -> None:
    parser = argparse.ArgumentParser(description="Export pipeline outputs to viz snapshot.")
    parser.add_argument("--run-dir", required=True, help="Pipeline run directory (base_path)")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent / "data" / "snapshot.parquet"),
        help="Output snapshot.parquet path",
    )
    parser.add_argument("--sample", type=int, default=50_000, help="Max prompts to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    export(
        run_dir=Path(args.run_dir),
        out_path=Path(args.out),
        sample=args.sample,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
