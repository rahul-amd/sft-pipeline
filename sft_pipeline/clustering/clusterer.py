"""
Domain clustering and difficulty scoring.

Three algorithm options:
  hdbscan      — HDBSCAN on FAISS IVF centroids (CPU, no GPU required)
  kmeans       — sklearn MiniBatchKMeans on FAISS IVF centroids (CPU)
  flash_kmeans — flash-kmeans Triton kernels directly on all embeddings (CUDA/ROCm GPU)

flash_kmeans is strongly preferred at scale (7M+ points) because it avoids
the centroid-indirection and runs in O(N·K·iter) on GPU instead of O(n²).
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Phrases suggesting a multi-step / harder prompt
_MULTI_STEP_PATTERNS = re.compile(
    r"\b(step[- ]by[- ]step|derive|prove|explain why|how does|compare|contrast|"
    r"analyze|evaluate|critique|design|implement|calculate.*and then|first.*then|"
    r"multiple|several|list all)\b",
    re.IGNORECASE,
)

_DOMAIN_LABEL_KEYWORDS: dict[str, list[str]] = {
    "math": ["equation", "solve", "calculate", "proof", "algebra", "geometry",
             "integral", "derivative", "theorem", "probability"],
    "code": ["function", "algorithm", "python", "javascript", "implement", "code",
             "program", "class", "debug", "api", "sql", "data structure"],
    "science": ["physics", "chemistry", "biology", "molecule", "reaction", "force",
                "energy", "quantum", "cell", "evolution", "atom", "element"],
    "language": ["translate", "grammar", "essay", "summarize", "vocabulary",
                 "sentence", "paragraph", "write", "synonym", "paraphrase"],
}


# ---------------------------------------------------------------------------
# Difficulty scoring
# ---------------------------------------------------------------------------

def score_difficulty(prompt: str) -> str:
    """Assign a difficulty tier. Returns 'easy', 'medium', or 'hard'."""
    tokens = prompt.split()
    n_tokens = len(tokens)
    if n_tokens >= 200 or _MULTI_STEP_PATTERNS.search(prompt):
        return "hard"
    if n_tokens <= 50 and not _MULTI_STEP_PATTERNS.search(prompt):
        return "easy"
    return "medium"


# ---------------------------------------------------------------------------
# Domain inference
# ---------------------------------------------------------------------------

def _infer_cluster_domain(centroid_prompts: list[str]) -> str:
    """Infer a domain label from a sample of prompts near a centroid."""
    text = " ".join(centroid_prompts).lower()
    scores: dict[str, int] = {
        domain: sum(text.count(kw) for kw in keywords)
        for domain, keywords in _DOMAIN_LABEL_KEYWORDS.items()
    }
    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] > 0 else "general"


# ---------------------------------------------------------------------------
# FAISS k-means path (CPU or GPU via faiss-gpu; NO Triton dependency)
# ---------------------------------------------------------------------------

def _cluster_with_faiss_kmeans(
    embeddings: np.ndarray,   # float16 or float32, shape (N, D)
    n_clusters: int,
    training_sample: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run FAISS built-in k-means on the full embedding matrix.

    Uses faiss.Kmeans which relies on FAISS's own optimised BLAS/GPU kernels —
    no Triton, so it works on MI250X (flash-kmeans Triton kernels require
    128 KB shared memory per CU but MI250X only provides 64 KB).

    Training uses a random subsample of `training_sample` vectors; assignment
    converts float16 chunks to float32 on-the-fly to avoid a 44 GB float32 copy.
    """
    import faiss

    N, D = embeddings.shape
    k = min(n_clusters, N)

    # --- Train on a random subsample ---
    train_size = min(training_sample, N)
    rng = np.random.default_rng(42)
    train_idx = rng.choice(N, size=train_size, replace=False)
    train_vecs = embeddings[train_idx].astype(np.float32)

    logger.info(
        "faiss_kmeans: training on %d / %d vectors → %d clusters ...",
        train_size, N, k,
    )

    # Try GPU (faiss-gpu / faiss-rocm); fall back to CPU silently.
    use_gpu = device in ("cuda", "rocm")
    try:
        kmeans = faiss.Kmeans(D, k, niter=20, verbose=True, gpu=use_gpu, seed=42)
        kmeans.train(train_vecs)
    except Exception as e:
        if use_gpu:
            logger.warning("faiss_kmeans: GPU mode failed (%s), retrying on CPU", e)
            kmeans = faiss.Kmeans(D, k, niter=20, verbose=True, gpu=False, seed=42)
            kmeans.train(train_vecs)
        else:
            raise
    del train_vecs

    # --- Assign all vectors in float32 chunks (avoids a full 88 GB float32 copy) ---
    logger.info("faiss_kmeans: assigning %d vectors to %d centroids ...", N, k)
    chunk = 500_000
    labels = np.empty(N, dtype=np.int64)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        batch_f32 = embeddings[start:end].astype(np.float32)
        _, I = kmeans.index.search(batch_f32, 1)
        labels[start:end] = I[:, 0]
        if start % 5_000_000 == 0 and start > 0:
            logger.info("faiss_kmeans: assigned %d / %d", end, N)

    n_unique = len(np.unique(labels))
    logger.info("faiss_kmeans: done. Unique clusters: %d", n_unique)
    return labels, kmeans.centroids.astype(np.float32)


# ---------------------------------------------------------------------------
# flash-kmeans path (GPU, Triton) — NOT compatible with MI250X
# (requires 128 KB shared memory per CU; MI250X limit is 64 KB)
# ---------------------------------------------------------------------------

def _cluster_with_flash_kmeans(
    embeddings: np.ndarray,
    n_clusters: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run flash-kmeans (Triton GPU kernels) on the full embedding matrix.

    Args:
        embeddings: float32 array of shape (N, D).
        n_clusters: number of clusters K.
        device: 'cuda' or 'rocm' (both map to CUDA device string for PyTorch/Triton).

    Returns:
        labels:  int64 array of shape (N,) — cluster assignment per point.
        centers: float32 array of shape (K, D) — cluster centroids.

    Raises:
        RuntimeError: if no CUDA-capable GPU is available.
        ImportError: if flash-kmeans is not installed.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "flash_kmeans requires a CUDA-capable GPU (also works with ROCm via "
            "HIP/Triton). No CUDA device found. Use algorithm='hdbscan' or 'kmeans' "
            "for CPU-only environments."
        )

    from flash_kmeans import batch_kmeans_Euclid  # pip install flash-kmeans

    N, D = embeddings.shape
    mem_gb = N * D * 2 / 1024**3  # float16
    if mem_gb > 8.0:
        logger.warning(
            "flash_kmeans: tensor will occupy ~%.1f GB of GPU memory (float16). "
            "Ensure your GPU has sufficient VRAM.",
            mem_gb,
        )

    k = min(n_clusters, N)
    logger.info("flash_kmeans: clustering %d points into %d clusters (%.1f GB)", N, k, mem_gb)

    # flash_kmeans expects: (batch, N, D), float16, on CUDA.
    # load_embeddings() now returns float16 so copy=False avoids a 44 GB copy.
    x = torch.from_numpy(embeddings.astype(np.float16, copy=False)).unsqueeze(0).cuda()

    cluster_ids, centers, _ = batch_kmeans_Euclid(x, n_clusters=k, tol=1e-4, verbose=False)

    # Remove batch dim: (1, N) → (N,);  (1, K, D) → (K, D)
    labels = cluster_ids.squeeze(0).cpu().numpy().astype(np.int64)
    centers_np = centers.squeeze(0).cpu().to(torch.float32).numpy()

    logger.info("flash_kmeans: done. Unique clusters: %d", len(np.unique(labels)))
    return labels, centers_np


# ---------------------------------------------------------------------------
# Centroid-based path (HDBSCAN / sklearn KMeans — CPU)
# ---------------------------------------------------------------------------

def _cluster_centroids(
    centroids: np.ndarray,
    algorithm: str,
    min_cluster_size: int,
    n_clusters: int,
) -> np.ndarray:
    """Cluster the centroid vectors (small, fast). Returns label array."""
    if algorithm == "hdbscan":
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min(min_cluster_size, max(2, len(centroids) // 10)),
            metric="euclidean",
            core_dist_n_jobs=-1,
        )
        return clusterer.fit_predict(centroids.astype(np.float64))
    else:
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=min(n_clusters, len(centroids)), random_state=42)
        return km.fit_predict(centroids)


def _assign_to_centroids(embeddings: np.ndarray, faiss_index) -> list[int]:
    """Return the nearest centroid index for each embedding, via FAISS."""
    import faiss

    batch_size = 50_000
    assignments: list[int] = []
    for start in range(0, len(embeddings), batch_size):
        batch = embeddings[start : start + batch_size].astype(np.float32)
        if hasattr(faiss_index, "quantizer"):
            quantizer = faiss.downcast_index(faiss_index.quantizer)
            _, idx = quantizer.search(batch, 1)
        else:
            _, idx = faiss_index.search(batch, 1)
        assignments.extend(idx[:, 0].tolist())
    return assignments


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def cluster_prompts(
    prompt_ids: list[str],
    prompts: list[str],
    embeddings: np.ndarray,          # float16 or float32, shape (N, D)
    centroids: Optional[np.ndarray] = None,  # required for hdbscan/kmeans
    faiss_index=None,                        # required for hdbscan/kmeans
    algorithm: str = "hdbscan",
    min_cluster_size: int = 100,
    n_clusters: int = 50,
    device: str = "cpu",
    training_sample: int = 1_000_000,
) -> list[dict]:
    """
    Assign each prompt a domain label and difficulty tier.

    For algorithm='faiss_kmeans': FAISS built-in k-means (no Triton).
      Trains on `training_sample` vectors, assigns all N. Works on MI250X.
    For algorithm='flash_kmeans': Triton GPU k-means. NOT compatible with MI250X
      (requires 128 KB shared memory; MI250X CU limit is 64 KB).
    For algorithm='hdbscan' or 'kmeans': centroid-based (CPU).
      centroids and faiss_index must be provided.

    Returns a list of dicts:
      {"prompt_id": ..., "domain": ..., "difficulty": ..., "cluster_id": int}
    """
    N = len(prompt_ids)
    logger.info("cluster_prompts: N=%d, algorithm=%s", N, algorithm)

    if algorithm in ("faiss_kmeans", "flash_kmeans"):
        if algorithm == "faiss_kmeans":
            labels, centers = _cluster_with_faiss_kmeans(
                embeddings, n_clusters, training_sample, device
            )
        else:
            labels, centers = _cluster_with_flash_kmeans(embeddings, n_clusters, device)

        # Build domain map: for each cluster, inspect a sample of its member prompts
        cluster_domain: dict[int, str] = {}
        for k in range(centers.shape[0]):
            members = [prompts[i] for i in range(N) if labels[i] == k][:50]
            cluster_domain[k] = _infer_cluster_domain(members)

        results: list[dict] = []
        for i, pid in enumerate(prompt_ids):
            cid = int(labels[i])
            results.append({
                "prompt_id": pid,
                "domain": cluster_domain.get(cid, "general"),
                "difficulty": score_difficulty(prompts[i]),
                "cluster_id": cid,
            })
        return results

    # --- centroid-based path (hdbscan / kmeans) ---
    if centroids is None or faiss_index is None:
        raise ValueError(
            f"algorithm='{algorithm}' requires centroids and faiss_index. "
            "Use algorithm='faiss_kmeans' or 'flash_kmeans' to cluster without a FAISS index."
        )

    centroid_labels = _cluster_centroids(centroids, algorithm, min_cluster_size, n_clusters)
    logger.info("Centroid clustering done: %d unique labels", len(set(centroid_labels)))

    centroid_assignments = _assign_to_centroids(embeddings, faiss_index)

    nlist = len(centroids)
    centroid_domain: dict[int, str] = {}
    for cidx in range(nlist):
        members = [prompts[i] for i, ca in enumerate(centroid_assignments) if ca == cidx][:50]
        centroid_domain[cidx] = _infer_cluster_domain(members)

    results = []
    for i, pid in enumerate(prompt_ids):
        cidx = centroid_assignments[i]
        cluster_label = int(centroid_labels[cidx]) if cidx < len(centroid_labels) else -1
        results.append({
            "prompt_id": pid,
            "domain": centroid_domain.get(cidx, "general"),
            "difficulty": score_difficulty(prompts[i]),
            "cluster_id": cluster_label,
        })
    return results
