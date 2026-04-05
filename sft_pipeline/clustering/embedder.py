"""
Batched sentence embedding using sentence-transformers.

Streams JSONL prompt records, embeds in batches, and writes
(prompt_id, embedding) pairs to sharded Parquet files in float16.

Memory: each shard holds `rows_per_shard` rows. Default 500K rows at
1024 dims × 2 bytes = ~1GB float16 per shard. At 7M prompts → ~14 shards.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

_SHARD_GLOB = "embeddings_*.parquet"


def embed_prompts(
    prompt_iter: Iterator[dict],
    model_name: str,
    batch_size: int,
    device: str,
    output_dir: str | Path,
    rows_per_shard: int = 500_000,
    gpu_batch_size: int = 32,
) -> int:
    """
    Embed all prompts and save to sharded Parquet files under `output_dir`.

    Shards are named embeddings_0000.parquet, embeddings_0001.parquet, …
    Each shard holds at most `rows_per_shard` rows, so peak RAM stays bounded
    regardless of total dataset size.

    Args:
        prompt_iter: Iterator of dicts with at least 'prompt_id' and 'prompt'.
        model_name: HuggingFace model name for sentence-transformers.
        batch_size: Number of prompts to accumulate before calling model.encode().
            Can be large (e.g. 500) for I/O efficiency — the actual GPU forward
            pass is capped at gpu_batch_size regardless.
        device: 'cuda', 'rocm' (maps to 'cuda' in PyTorch), or 'cpu'.
        output_dir: Directory to write sharded Parquet files.
        rows_per_shard: Flush a new shard file after this many rows.
        gpu_batch_size: Max sequences per GPU forward pass inside model.encode().
            Separate from batch_size — controls peak GPU memory.  32 is safe for
            bge-m3 on MI250X; increase to 64/128 once stable.

    Returns:
        Total number of prompts embedded.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from sentence_transformers import SentenceTransformer

    # ROCm uses 'cuda' device string in PyTorch
    pt_device = "cuda" if device in ("cuda", "rocm") else "cpu"

    logger.info("Loading embedding model %s on device=%s", model_name, pt_device)
    model = SentenceTransformer(model_name, device=pt_device)
    logger.info("Embedding dim: %d", model.get_sentence_embedding_dimension())

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    shard_ids: list[str] = []
    shard_vecs: list[np.ndarray] = []
    shard_rows = 0

    batch_ids: list[str] = []
    batch_texts: list[str] = []
    total = 0

    def _write_shard() -> None:
        nonlocal shard_idx, shard_rows
        if not shard_ids:
            return
        vecs = np.vstack(shard_vecs)  # (rows, dim)
        shard_path = out_dir / f"embeddings_{shard_idx:04d}.parquet"
        table = pa.table({
            "prompt_id": pa.array(shard_ids, type=pa.string()),
            "embedding": pa.array(
                [row.tolist() for row in vecs],
                type=pa.list_(pa.float16()),
            ),
        })
        pq.write_table(table, str(shard_path), compression="zstd")
        logger.info(
            "Wrote shard %d → %s (%d rows)", shard_idx, shard_path.name, len(shard_ids)
        )
        shard_idx += 1
        shard_ids.clear()
        shard_vecs.clear()
        shard_rows = 0

    def _flush_batch() -> None:
        nonlocal shard_rows
        if not batch_texts:
            return
        vecs = model.encode(
            batch_texts,
            batch_size=gpu_batch_size,   # GPU forward-pass size; << batch_size
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float16)
        shard_ids.extend(batch_ids)
        shard_vecs.append(vecs)
        shard_rows += len(batch_texts)
        batch_ids.clear()
        batch_texts.clear()

        if shard_rows >= rows_per_shard:
            _write_shard()

    for record in prompt_iter:
        batch_ids.append(record["prompt_id"])
        batch_texts.append(record["prompt"])
        total += 1

        if len(batch_texts) >= batch_size:
            _flush_batch()
            if total % 100_000 == 0:
                logger.info("Embedded %d prompts so far...", total)

    _flush_batch()   # remaining batch
    _write_shard()   # remaining shard

    if total == 0:
        logger.warning("No prompts to embed.")
    else:
        logger.info("Embedding complete: %d prompts → %d shards in %s", total, shard_idx, out_dir)

    return total


def embed_jsonl_shards(
    jsonl_paths: list[str],
    worker_id: int,
    model_name: str,
    batch_size: int,
    device: str,
    output_dir: str | Path,
    rows_per_shard: int = 500_000,
    gpu_batch_size: int = 32,
) -> dict:
    """
    Embed prompts from a list of JSONL shard files and write to Parquet.

    This is the Ray worker entry point for distributed Stage 3 embedding.
    It must be a standalone importable function (not a closure) so Ray can
    serialise it.  Mirrors embed_prompts() but:

      - Takes explicit file paths instead of an iterator so each Ray worker
        can process an independent chunk of the prompt pool.
      - Prefixes output shard names with the worker_id to avoid collisions:
        embeddings_w{worker_id:02d}_{shard_idx:04d}.parquet
      - Writes a sentinel file  embeddings_w{worker_id:02d}.done  on success
        so the caller can detect completion and skip on resume.

    Args:
        jsonl_paths:   List of absolute paths to part-*.jsonl shard files.
        worker_id:     Zero-based worker index; determines output file prefix.
        model_name:    HuggingFace model name for sentence-transformers.
        batch_size:    Number of prompts to accumulate before calling model.encode().
            Can be large (e.g. 500) for I/O efficiency — the actual GPU forward
            pass is capped at gpu_batch_size regardless.
        device:        'cuda', 'rocm' (maps to 'cuda' in PyTorch), or 'cpu'.
        output_dir:    Shared directory to write Parquet shards.
        rows_per_shard: Max rows per output Parquet shard.
        gpu_batch_size: Max sequences per GPU forward pass inside model.encode().
            Separate from batch_size — controls peak GPU memory.  32 is safe for
            bge-m3 on MI250X; increase to 64/128 once stable.

    Returns:
        {"worker_id": int, "n_embedded": int, "n_shards": int}
    """
    import logging as _logging
    import pyarrow as pa
    import pyarrow.parquet as pq
    import numpy as np
    from pathlib import Path as _Path
    from sentence_transformers import SentenceTransformer

    from sft_pipeline.storage import iter_jsonl

    _log = _logging.getLogger(__name__)

    # ROCm uses 'cuda' device string in PyTorch
    pt_device = "cuda" if device in ("cuda", "rocm") else "cpu"

    if pt_device == "cuda":
        import torch as _torch
        if not _torch.cuda.is_available():
            raise RuntimeError(
                f"Worker {worker_id}: GPU requested (device={device!r}) but "
                "torch.cuda.is_available() is False.\n"
                "The sft-pipeline env needs ROCm PyTorch, not the default CUDA wheel.\n"
                "Install: pip install --user torch "
                "--index-url https://download.pytorch.org/whl/rocm6.2"
            )

    _log.info(
        "Worker %d: loading embedding model %s on device=%s",
        worker_id, model_name, pt_device,
    )
    model = SentenceTransformer(model_name, device=pt_device)

    out_dir = _Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"embeddings_w{worker_id:02d}"
    shard_idx = 0
    shard_ids: list[str] = []
    shard_vecs: list[np.ndarray] = []
    shard_rows = 0

    batch_ids: list[str] = []
    batch_texts: list[str] = []
    total = 0

    def _write_shard() -> None:
        nonlocal shard_idx, shard_rows
        if not shard_ids:
            return
        vecs = np.vstack(shard_vecs)
        shard_path = out_dir / f"{prefix}_{shard_idx:04d}.parquet"
        table = pa.table({
            "prompt_id": pa.array(shard_ids, type=pa.string()),
            "embedding": pa.array(
                [row.tolist() for row in vecs],
                type=pa.list_(pa.float16()),
            ),
        })
        pq.write_table(table, str(shard_path), compression="zstd")
        _log.info(
            "Worker %d: wrote shard %d → %s (%d rows)",
            worker_id, shard_idx, shard_path.name, len(shard_ids),
        )
        shard_idx += 1
        shard_ids.clear()
        shard_vecs.clear()
        shard_rows = 0

    def _flush_batch() -> None:
        nonlocal shard_rows
        if not batch_texts:
            return
        vecs = model.encode(
            batch_texts,
            batch_size=gpu_batch_size,   # GPU forward-pass size; << batch_size
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float16)
        shard_ids.extend(batch_ids)
        shard_vecs.append(vecs)
        shard_rows += len(batch_texts)
        batch_ids.clear()
        batch_texts.clear()

        if shard_rows >= rows_per_shard:
            _write_shard()

    for jsonl_path in jsonl_paths:
        for record in iter_jsonl(jsonl_path):
            batch_ids.append(record["prompt_id"])
            batch_texts.append(record["prompt"])
            total += 1

            if len(batch_texts) >= batch_size:
                _flush_batch()
                if total % 100_000 == 0:
                    _log.info("Worker %d: embedded %d prompts so far...", worker_id, total)

    _flush_batch()
    _write_shard()

    # Write sentinel file — presence signals that this worker completed successfully
    sentinel = out_dir / f"{prefix}.done"
    sentinel.write_text(f"n_embedded={total}\n")

    _log.info(
        "Worker %d: embedding complete — %d prompts, %d shards",
        worker_id, total, shard_idx,
    )
    return {"worker_id": worker_id, "n_embedded": total, "n_shards": shard_idx}


def load_embeddings(parquet_path: str | Path) -> tuple[list[str], np.ndarray]:
    """
    Load (prompt_ids, embeddings_float32) from a Parquet file or directory of shards.

    Accepts:
      - A single .parquet file (legacy / test usage)
      - A directory containing embeddings_*.parquet shards (production)

    Returns float32 for FAISS compatibility.
    """
    import pyarrow.parquet as pq

    path = Path(parquet_path)

    if path.is_dir():
        shard_files = sorted(path.glob(_SHARD_GLOB))
        if not shard_files:
            raise FileNotFoundError(f"No embedding shards found in {path}")
        tables = [pq.read_table(str(f)) for f in shard_files]
        import pyarrow as pa
        table = pa.concat_tables(tables)
        logger.info("Loaded %d shards from %s", len(shard_files), path)
    else:
        table = pq.read_table(str(path))

    ids = table["prompt_id"].to_pylist()
    vecs = np.array(table["embedding"].to_pylist(), dtype=np.float32)
    logger.info("Loaded %d embeddings, shape=%s", len(ids), vecs.shape)
    return ids, vecs
