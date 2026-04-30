"""
Domain clustering and difficulty scoring.

Algorithm options:
  hdbscan      — HDBSCAN on FAISS IVF centroids (CPU, no GPU required)
  kmeans       — sklearn MiniBatchKMeans on FAISS IVF centroids (CPU)
  faiss_kmeans — FAISS built-in k-means (CPU BLAS; no Triton); slow on large N
  torch_kmeans — PyTorch matmul k-means on GPU (cosine distance); recommended
                 for MI250X — no Triton, no faiss-gpu, works on ROCm natively
  flash_kmeans — flash-kmeans Triton k-means; NOT compatible with MI250X
                 (requires 128 KB shared memory; MI250X CU limit is 64 KB)

torch_kmeans is the recommended algorithm at scale on MI250X.
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
             "integral", "derivative", "theorem", "probability", "matrix",
             "statistics", "arithmetic", "calculus", "combinatorics"],
    "code": ["function", "algorithm", "python", "javascript", "implement", "code",
             "program", "class", "debug", "api", "sql", "data structure",
             "typescript", "java", "c++", "rust", "compiler", "runtime", "loop"],
    "science": ["physics", "chemistry", "biology", "molecule", "reaction", "force",
                "energy", "quantum", "cell", "evolution", "atom", "element",
                "gravity", "photosynthesis", "dna", "orbit", "thermodynamics"],
    "reasoning": ["logic", "deduce", "conclude", "premise", "fallacy", "puzzle",
                  "riddle", "infer", "argument", "valid", "syllogism", "paradox",
                  "critical thinking", "analyze the argument"],
    "writing": ["essay", "story", "creative", "draft", "edit", "narrative",
                "fiction", "poem", "blog post", "article", "rewrite", "tone",
                "introduction", "conclusion", "persuasive"],
    "language": ["translate", "grammar", "vocabulary", "synonym", "paraphrase",
                 "conjugate", "tense", "linguistic", "idiom", "pronoun",
                 "sentence structure", "spanish", "french", "mandarin"],
    "knowledge": ["history", "geography", "capital", "who was", "when did",
                  "what is", "define", "explain", "culture", "biography",
                  "encyclopedia", "fact", "event", "founded"],
    "instruction": ["how to", "steps to", "how do i", "guide me", "tutorial",
                    "walk me through", "recipe", "procedure", "instructions for",
                    "set up", "configure", "install"],
}


# ---------------------------------------------------------------------------
# Difficulty scoring
# ---------------------------------------------------------------------------

def score_difficulty(prompt: str) -> str:
    """Assign a difficulty tier. Returns 'easy', 'medium', or 'hard'."""
    tokens = prompt.split()
    n_tokens = len(tokens)
    multi_step = _MULTI_STEP_PATTERNS.search(prompt) is not None
    if n_tokens >= 200 or multi_step:
        return "hard"
    if n_tokens <= 50 and not multi_step:
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
    return best if scores[best] > 0 else "other"


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
# PyTorch GPU k-means (cosine distance, chunked matmul) — works on MI250X
# ---------------------------------------------------------------------------

def _cluster_with_torch_kmeans(
    embeddings: np.ndarray,   # float16 or float32, shape (N, D)
    n_clusters: int,
    training_sample: int,
    device: str,
    n_iter: int = 20,
    chunk_size: int = 20_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    GPU k-means via PyTorch matmuls on L2-normalised embeddings (cosine distance).

    Works on MI250X/ROCm without Triton and without faiss-gpu.
    Trains on a random subsample of `training_sample` vectors, then assigns all N.

    Memory footprint (4M training × 1024 dim, 100K clusters):
      - Training set on GPU: 4M × 1024 × 2B (float16) = 8 GB
      - Centroids: 100K × 1024 × 4B (float32) = 400 MB
      - Distance matrix per chunk: 20K × 100K × 2B (float16) = 4 GB
      Total peak: ~13 GB — well within MI250X 64 GB HBM.
    """
    import torch
    import torch.nn.functional as F

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        logger.warning("torch_kmeans: no GPU available, falling back to CPU (slow)")

    N, D = embeddings.shape
    k = min(n_clusters, N)
    train_size = min(max(training_sample, k), N)   # at least k to init centroids

    rng = np.random.default_rng(42)
    train_idx = rng.choice(N, size=train_size, replace=False)

    # Load training set to GPU as float16 (4M × 1024 × 2B = 8 GB)
    X = torch.from_numpy(
        embeddings[train_idx].astype(np.float16, copy=False)
    ).to(torch_device)
    X = F.normalize(X.float(), dim=1).half()   # L2-normalise; keep float16

    # Init centroids from random training vectors; float32 for stable updates
    init_idx = torch.randperm(train_size, device=torch_device)[:k]
    C = F.normalize(X[init_idx].float(), dim=1)   # (k, D) float32

    logger.info(
        "torch_kmeans: training on %d / %d vectors → %d clusters, up to %d iters ...",
        train_size, N, k, n_iter,
    )

    ones = torch.ones(train_size, dtype=torch.float32, device=torch_device)

    for it in range(n_iter):
        C_h = C.half()   # float16 copy for fast matmul
        labels = torch.empty(train_size, dtype=torch.int32, device=torch_device)

        for start in range(0, train_size, chunk_size):
            end = min(start + chunk_size, train_size)
            sims = X[start:end] @ C_h.T                    # (c, k) cosine similarities
            labels[start:end] = sims.argmax(dim=1).int()

        # Centroid update: sum assigned vectors, divide by counts
        new_C = torch.zeros(k, D, dtype=torch.float32, device=torch_device)
        counts = torch.zeros(k, dtype=torch.float32, device=torch_device)
        new_C.index_add_(0, labels.long(), X.float())
        counts.index_add_(0, labels.long(), ones)

        # Re-init empty clusters to random training vectors
        empty_mask = counts == 0
        n_empty = int(empty_mask.sum().item())
        if n_empty:
            ri = torch.randint(train_size, (n_empty,), device=torch_device)
            new_C[empty_mask] = X[ri].float()
            counts[empty_mask] = 1.0

        new_C.div_(counts.unsqueeze(1))
        new_C = F.normalize(new_C, dim=1)

        shift = (new_C - C).norm(dim=1).max().item()
        C = new_C
        logger.info(
            "torch_kmeans: iter %2d / %d  max-shift=%.6f  empty=%d",
            it + 1, n_iter, shift, n_empty,
        )
        if shift < 1e-4 and it > 2:
            logger.info("torch_kmeans: converged at iter %d", it + 1)
            break

    del X, ones
    torch.cuda.empty_cache()

    # Assign all N vectors; stream from CPU numpy in chunks to avoid OOM
    logger.info("torch_kmeans: assigning all %d vectors ...", N)
    C_h = F.normalize(C, dim=1).half()   # (k, D) float16
    all_labels = np.empty(N, dtype=np.int32)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = torch.from_numpy(
            embeddings[start:end].astype(np.float16, copy=False)
        ).to(torch_device)
        chunk = F.normalize(chunk.float(), dim=1).half()
        all_labels[start:end] = (chunk @ C_h.T).argmax(dim=1).cpu().numpy()
        if start % 2_000_000 == 0 and start > 0:
            logger.info("torch_kmeans: assigned %d / %d", start, N)

    n_unique = len(np.unique(all_labels))
    logger.info("torch_kmeans: done. Unique clusters: %d", n_unique)
    return all_labels.astype(np.int64), C.float().cpu().numpy()


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

def compute_centroid_similarities(
    embeddings: np.ndarray,   # (N, D) float16 or float32
    labels: np.ndarray,       # (N,) int — cluster assignment per point
    centers: np.ndarray,      # (K, D) float32 — cluster centroids
    device: str = "cpu",
    chunk_size: int = 500_000,
) -> np.ndarray:
    """
    Cosine similarity of each embedding to its assigned cluster centroid.

    Returns float32 array of shape (N,).  Labels < 0 (noise points in
    HDBSCAN) are clamped to 0 and will receive the similarity to cluster 0
    rather than NaN — callers should treat those values as unreliable.
    """
    import torch
    import torch.nn.functional as F

    torch_device = torch.device("cuda" if device in ("cuda", "rocm") else "cpu")
    N = len(embeddings)

    C = F.normalize(
        torch.from_numpy(centers.astype(np.float32)).to(torch_device), dim=1
    )  # (K, D)

    sims = np.empty(N, dtype=np.float32)
    n_chunks = (N + chunk_size - 1) // chunk_size
    for chunk_i, start in enumerate(range(0, N, chunk_size)):
        end = min(start + chunk_size, N)
        vecs = F.normalize(
            torch.from_numpy(embeddings[start:end].astype(np.float32)).to(torch_device),
            dim=1,
        )  # (c, D)
        lbls = torch.from_numpy(
            np.clip(labels[start:end], 0, C.shape[0] - 1).astype(np.int64)
        ).to(torch_device)
        sims[start:end] = (vecs * C[lbls]).sum(dim=1).cpu().numpy()
        if n_chunks > 4 and (
            (chunk_i + 1) % max(1, n_chunks // 4) == 0 or (chunk_i + 1) == n_chunks
        ):
            logger.info(
                "compute_centroid_similarities: %d/%d vectors (%.0f%%)", end, N, 100.0 * end / N
            )

    return sims


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

    For algorithm='torch_kmeans': PyTorch matmul k-means on GPU (cosine distance).
      Recommended for MI250X — no Triton, no faiss-gpu. Trains on `training_sample`
      vectors, assigns all N.
    For algorithm='faiss_kmeans': FAISS built-in k-means (CPU BLAS; no Triton).
      Trains on `training_sample` vectors, assigns all N. Works on MI250X but slow.
    For algorithm='flash_kmeans': Triton GPU k-means. NOT compatible with MI250X
      (requires 128 KB shared memory; MI250X CU limit is 64 KB).
    For algorithm='hdbscan' or 'kmeans': centroid-based (CPU).
      centroids and faiss_index must be provided.

    Returns a list of dicts:
      {"prompt_id": ..., "domain": ..., "difficulty": ..., "cluster_id": int}
    """
    N = len(prompt_ids)
    logger.info("cluster_prompts: N=%d, algorithm=%s", N, algorithm)

    if algorithm in ("torch_kmeans", "faiss_kmeans", "flash_kmeans"):
        if algorithm == "torch_kmeans":
            labels, centers = _cluster_with_torch_kmeans(
                embeddings, n_clusters, training_sample, device
            )
        elif algorithm == "faiss_kmeans":
            labels, centers = _cluster_with_faiss_kmeans(
                embeddings, n_clusters, training_sample, device
            )
        else:
            labels, centers = _cluster_with_flash_kmeans(embeddings, n_clusters, device)

        # Build domain map: for each cluster, inspect a sample of its member prompts.
        # Do this in a single O(N) pass to collect up to 50 member indices per cluster,
        # rather than the naive O(K×N) loop that scans all N prompts for every cluster.
        _SAMPLE = 50
        cluster_members: dict[int, list[int]] = {}
        for i, lbl in enumerate(labels):
            k_lbl = int(lbl)
            lst = cluster_members.get(k_lbl)
            if lst is None:
                cluster_members[k_lbl] = [i]
            elif len(lst) < _SAMPLE:
                lst.append(i)

        cluster_domain: dict[int, str] = {}
        for k in range(centers.shape[0]):
            members = [prompts[i] for i in cluster_members.get(k, [])]
            cluster_domain[k] = _infer_cluster_domain(members)

        logger.info("cluster_prompts: computing centroid similarities ...")
        centroid_sims = compute_centroid_similarities(embeddings, labels, centers, device=device)

        results: list[dict] = []
        for i, pid in enumerate(prompt_ids):
            cid = int(labels[i])
            results.append({
                "prompt_id": pid,
                "domain": cluster_domain.get(cid, "general"),
                "difficulty": score_difficulty(prompts[i]),
                "cluster_id": cid,
                "centroid_sim": float(centroid_sims[i]),
            })
            if (i + 1) % 1_000_000 == 0:
                logger.info("cluster_prompts: labelled %d / %d", i + 1, N)
        logger.info("cluster_prompts: done — %d records labelled", N)
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
    # Single O(N) pass to collect up to 50 member indices per centroid.
    _SAMPLE = 50
    centroid_members: dict[int, list[int]] = {}
    for i, ca in enumerate(centroid_assignments):
        lst = centroid_members.get(ca)
        if lst is None:
            centroid_members[ca] = [i]
        elif len(lst) < _SAMPLE:
            lst.append(i)

    centroid_domain: dict[int, str] = {}
    for cidx in range(nlist):
        members = [prompts[i] for i in centroid_members.get(cidx, [])]
        centroid_domain[cidx] = _infer_cluster_domain(members)

    logger.info("cluster_prompts: computing centroid similarities ...")
    centroid_assignments_arr = np.array(centroid_assignments, dtype=np.int64)
    centroid_sims = compute_centroid_similarities(embeddings, centroid_assignments_arr, centroids, device=device)

    results = []
    for i, pid in enumerate(prompt_ids):
        cidx = centroid_assignments[i]
        cluster_label = int(centroid_labels[cidx]) if cidx < len(centroid_labels) else -1
        results.append({
            "prompt_id": pid,
            "domain": centroid_domain.get(cidx, "general"),
            "difficulty": score_difficulty(prompts[i]),
            "cluster_id": cluster_label,
            "centroid_sim": float(centroid_sims[i]),
        })
    return results
