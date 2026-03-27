"""
FAISS index management — GPU-accelerated where available, CPU fallback everywhere else.

Build procedure:
  1. Sample training vectors from the full set
  2. Train IVFFlat quantizer (on GPU if device=cuda/rocm, else CPU)
  3. Stream all vectors into the index in batches (on GPU if available)
  4. Convert back to CPU index and save to disk (GPU indexes can't be serialized)

At query time, load the saved index and optionally move to GPU for fast search.

GPU notes:
  - Requires faiss-gpu package (pip install faiss-gpu). faiss-cpu silently falls back.
  - ROCm: faiss-rocm wheel required; device string "rocm" maps to GPU device 0.
  - Always save/load as CPU index; move to GPU per-operation as needed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _get_gpu_res(device: str):
    """
    Try to acquire a FAISS GPU resource handle.

    Returns:
        (res, device_id) if a GPU is available and faiss-gpu is installed.
        (None, -1) if CPU-only (faiss-cpu, no GPUs, or device not cuda/rocm).
    """
    if device not in ("cuda", "rocm"):
        return None, -1
    try:
        import faiss
        n = faiss.get_num_gpus()
        if n == 0:
            logger.warning("FAISS: device=%s requested but no GPUs detected, using CPU", device)
            return None, -1
        res = faiss.StandardGpuResources()
        logger.info("FAISS: using GPU device 0 (%d GPU(s) available)", n)
        return res, 0
    except AttributeError:
        # faiss-cpu package: StandardGpuResources not present
        logger.warning(
            "FAISS: faiss-gpu not installed (faiss-cpu detected). "
            "Install faiss-gpu to enable GPU acceleration. Falling back to CPU."
        )
        return None, -1
    except Exception as e:
        logger.warning("FAISS: could not initialize GPU resources (%s), falling back to CPU", e)
        return None, -1


def build_and_save(
    ids: list[str],
    vectors: np.ndarray,  # float32, shape (N, D)
    index_path: str | Path,
    index_type: str = "IVFFlat",
    nlist: int = 1000,
    nprobe: int = 50,
    training_sample: int = 500_000,
    batch_size: int = 100_000,
    device: str = "cpu",
) -> None:
    """
    Build a FAISS index from vectors and save to disk.

    Training and vector insertion run on GPU when device='cuda'/'rocm' and
    faiss-gpu is installed. The index is always saved as a CPU index.

    Args:
        ids:             Parallel list of string IDs (not stored in FAISS; used externally).
        vectors:         Float32 numpy array of shape (N, embedding_dim).
        index_path:      Path to save the serialized FAISS index.
        index_type:      'Flat', 'IVFFlat', or 'IVFPQ'.
        nlist:           Number of IVF centroids (for IVFFlat / IVFPQ).
        nprobe:          Centroids probed at query time (set after GPU→CPU conversion).
        training_sample: Number of vectors to use for IVF training subsample.
        batch_size:      Vectors added per batch.
        device:          'cpu', 'cuda', or 'rocm'. 'rocm' maps to GPU device 0.
    """
    import faiss

    N, D = vectors.shape
    logger.info(
        "Building FAISS %s index: N=%d, D=%d, nlist=%d, device=%s",
        index_type, N, D, nlist, device,
    )

    res, gpu_id = _get_gpu_res(device)
    use_gpu = res is not None

    # --- Build CPU index, move to GPU, train, add, convert back ---

    if index_type == "Flat":
        index = faiss.IndexFlatIP(D)
        if use_gpu:
            index = faiss.index_cpu_to_gpu(res, gpu_id, index)

    elif index_type == "IVFFlat":
        quantizer = faiss.IndexFlatIP(D)
        nlist_actual = min(nlist, max(1, N // 10))
        cpu_index = faiss.IndexIVFFlat(
            quantizer, D, nlist_actual, faiss.METRIC_INNER_PRODUCT
        )
        index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index) if use_gpu else cpu_index

        train_size = min(training_sample, N)
        rng = np.random.default_rng(42)
        train_idx = rng.choice(N, size=train_size, replace=False)
        train_vecs = vectors[train_idx].astype(np.float32)
        logger.info(
            "Training IVF quantizer on %d vectors (GPU=%s)...", train_size, use_gpu
        )
        index.train(train_vecs)

    elif index_type == "IVFPQ":
        quantizer = faiss.IndexFlatIP(D)
        m = 64  # sub-quantizers; D must be divisible by m
        nlist_actual = min(nlist, max(1, N // 10))
        cpu_index = faiss.IndexIVFPQ(quantizer, D, nlist_actual, m, 8)
        index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index) if use_gpu else cpu_index

        train_size = min(training_sample, N)
        rng = np.random.default_rng(42)
        train_idx = rng.choice(N, size=train_size, replace=False)
        logger.info(
            "Training IVFPQ quantizer on %d vectors (GPU=%s)...", train_size, use_gpu
        )
        index.train(vectors[train_idx].astype(np.float32))

    else:
        raise ValueError(f"Unknown index_type: {index_type!r}")

    # Add vectors in batches (on GPU if index is GPU-resident)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        index.add(vectors[start:end].astype(np.float32))
        if (start // batch_size) % 10 == 0:
            logger.info("FAISS: added %d / %d vectors", end, N)

    logger.info("FAISS index built: ntotal=%d", index.ntotal)

    # Convert GPU index → CPU index before serialisation.
    # nprobe is set here (after conversion) because GpuIndexIVF exposes it
    # differently and it's cleanest to set on the final CPU index.
    if use_gpu:
        logger.info("FAISS: converting GPU index back to CPU for saving")
        index = faiss.index_gpu_to_cpu(index)
        # res goes out of scope here; GPU memory freed after this function returns.

    if hasattr(index, "nprobe"):
        index.nprobe = nprobe

    out_path = Path(index_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_path))
    logger.info("FAISS index saved to %s", out_path)


def load_index(index_path: str | Path, device: str = "cpu"):
    """
    Load a saved FAISS index from disk.

    If device='cuda' or 'rocm' and faiss-gpu is available, the index is
    moved to GPU and returned as a GpuIndex for fast search. The caller must
    keep the returned (index, res) tuple alive for the lifetime of any searches.

    Returns:
        index — CPU index (device='cpu') or GPU index (device='cuda'/'rocm').
        The GPU resources handle is stored as ``index._gpu_res`` to keep it
        alive as long as the index object lives.
    """
    import faiss
    path = str(index_path)
    logger.info("Loading FAISS index from %s", path)
    cpu_index = faiss.read_index(path)

    res, gpu_id = _get_gpu_res(device)
    if res is not None:
        try:
            gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
            # Attach res to the index object so Python keeps it alive
            gpu_index._gpu_res = res
            logger.info("FAISS: index loaded on GPU (device %d)", gpu_id)
            return gpu_index
        except Exception as e:
            logger.warning("FAISS: could not move loaded index to GPU (%s), using CPU", e)

    return cpu_index


def search(
    index,
    query_vectors: np.ndarray,  # float32, shape (Q, D)
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Search for k nearest neighbors.
    Returns (distances, indices) arrays of shape (Q, k).
    Works with both CPU and GPU FAISS indexes.
    """
    return index.search(query_vectors.astype(np.float32), k)


def get_centroids(index) -> Optional[np.ndarray]:
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


def make_flat_index(
    vectors: np.ndarray,
    device: str = "cpu",
):
    """
    Build an IndexFlatIP, add vectors, and optionally move to GPU.

    Returns the index. If GPU is used, the GPU resource handle is stored as
    ``index._gpu_res`` to keep it alive.

    Use this for temporary indexes in dedup / similarity-sort operations.
    """
    import faiss

    N, D = vectors.shape
    cpu_index = faiss.IndexFlatIP(D)
    cpu_index.add(vectors.astype(np.float32))

    res, gpu_id = _get_gpu_res(device)
    if res is not None:
        try:
            gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
            gpu_index._gpu_res = res  # keep alive
            logger.debug("FAISS: flat index on GPU (N=%d, D=%d)", N, D)
            return gpu_index
        except Exception as e:
            logger.warning("FAISS: could not move flat index to GPU (%s), using CPU", e)

    return cpu_index
