"""
FAISS index management — CPU IVFFlat (works on all hardware including ROCm clusters).

Build procedure:
  1. Sample training vectors from the full set
  2. Train IVFFlat quantizer
  3. Stream all vectors into the index in batches
  4. Save index to disk

At query time, load the saved index and call search().
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def build_and_save(
    ids: list[str],
    vectors: np.ndarray,  # float32, shape (N, D)
    index_path: str | Path,
    index_type: str = "IVFFlat",
    nlist: int = 1000,
    nprobe: int = 50,
    training_sample: int = 500_000,
    batch_size: int = 100_000,
) -> None:
    """
    Build a FAISS index from vectors and save to disk.

    Args:
        ids:            Parallel list of string IDs (not stored in FAISS; used externally).
        vectors:        Float32 numpy array of shape (N, embedding_dim).
        index_path:     Path to save the serialized FAISS index.
        index_type:     'Flat', 'IVFFlat', or 'IVFPQ'.
        nlist:          Number of IVF centroids (for IVFFlat / IVFPQ).
        nprobe:         Centroids probed at query time.
        training_sample: Number of vectors to use for IVF training.
        batch_size:     Vectors added per batch.
    """
    import faiss

    N, D = vectors.shape
    logger.info("Building FAISS %s index: N=%d, D=%d, nlist=%d", index_type, N, D, nlist)

    if index_type == "Flat":
        index = faiss.IndexFlatIP(D)
    elif index_type == "IVFFlat":
        quantizer = faiss.IndexFlatIP(D)
        index = faiss.IndexIVFFlat(quantizer, D, min(nlist, N // 10 or 1), faiss.METRIC_INNER_PRODUCT)
        # Train on a subsample
        train_size = min(training_sample, N)
        rng = np.random.default_rng(42)
        train_idx = rng.choice(N, size=train_size, replace=False)
        train_vecs = vectors[train_idx].astype(np.float32)
        logger.info("Training IVF quantizer on %d vectors...", train_size)
        index.train(train_vecs)
        index.nprobe = nprobe
    elif index_type == "IVFPQ":
        quantizer = faiss.IndexFlatIP(D)
        m = 64  # number of sub-quantizers; D must be divisible by m
        index = faiss.IndexIVFPQ(quantizer, D, min(nlist, N // 10 or 1), m, 8)
        train_size = min(training_sample, N)
        rng = np.random.default_rng(42)
        train_idx = rng.choice(N, size=train_size, replace=False)
        logger.info("Training IVFPQ quantizer on %d vectors...", train_size)
        index.train(vectors[train_idx].astype(np.float32))
        index.nprobe = nprobe
    else:
        raise ValueError(f"Unknown index_type: {index_type}")

    # Add vectors in batches
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = vectors[start:end].astype(np.float32)
        index.add(batch)
        if (start // batch_size) % 10 == 0:
            logger.info("FAISS: added %d / %d vectors", end, N)

    logger.info("FAISS index built: ntotal=%d", index.ntotal)

    out_path = Path(index_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_path))
    logger.info("FAISS index saved to %s", out_path)


def load_index(index_path: str | Path):
    """Load a saved FAISS index from disk."""
    import faiss
    path = str(index_path)
    logger.info("Loading FAISS index from %s", path)
    return faiss.read_index(path)


def search(
    index,
    query_vectors: np.ndarray,  # float32, shape (Q, D)
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Search for k nearest neighbors.
    Returns (distances, indices) arrays of shape (Q, k).
    """
    return index.search(query_vectors.astype(np.float32), k)


def get_centroids(index) -> np.ndarray | None:
    """
    Extract centroid vectors from an IVF index.
    Returns float32 array of shape (nlist, D), or None if not applicable.

    Uses index.quantizer.reconstruct(i) rather than accessing the internal
    .xb attribute, which is not reliably exposed in all faiss-cpu versions.
    """
    import faiss

    if isinstance(index, (faiss.IndexIVFFlat, faiss.IndexIVFPQ)):
        nlist = index.nlist
        centroids = np.stack([index.quantizer.reconstruct(i) for i in range(nlist)])
        return centroids.astype(np.float32)
    return None
