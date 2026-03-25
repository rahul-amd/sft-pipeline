"""
Tests for the flash_kmeans clustering path in clusterer.py.

Most tests require a CUDA GPU and will be skipped on CPU-only machines.
The benchmark test prints a side-by-side timing table; run with -s to see it:

    pytest tests/unit/clustering/test_clusterer_flash_kmeans.py -v -s
"""
from __future__ import annotations

import time

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_embeddings(n: int, d: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, d), dtype=np.float32)


def _make_prompts(n: int) -> tuple[list[str], list[str]]:
    ids = [f"id_{i:06d}" for i in range(n)]
    texts = [
        "solve the equation x^2 + 3x - 4 = 0" if i % 4 == 0
        else "write a python function that reverses a string" if i % 4 == 1
        else "explain the water cycle step by step" if i % 4 == 2
        else "translate the sentence into French"
        for i in range(n)
    ]
    return ids, texts


# Probe CUDA and flash_kmeans availability once at module load.
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

try:
    import flash_kmeans  # noqa: F401
    FLASH_KMEANS_AVAILABLE = True
except ImportError:
    FLASH_KMEANS_AVAILABLE = False

needs_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="requires a CUDA-capable GPU",
)
needs_flash = pytest.mark.skipif(
    not FLASH_KMEANS_AVAILABLE,
    reason="flash-kmeans not installed (pip install flash-kmeans)",
)

# Combined marker: skip unless both CUDA and flash_kmeans are present.
needs_cuda_and_flash = pytest.mark.skipif(
    not (CUDA_AVAILABLE and FLASH_KMEANS_AVAILABLE),
    reason="requires CUDA GPU + flash-kmeans (pip install flash-kmeans)",
)


# ---------------------------------------------------------------------------
# Unit: _cluster_with_flash_kmeans
# ---------------------------------------------------------------------------

@needs_cuda_and_flash
def test_flash_kmeans_raw_output_shapes():
    """_cluster_with_flash_kmeans returns arrays with the right shapes."""
    from sft_pipeline.clustering.clusterer import _cluster_with_flash_kmeans

    N, D, K = 2_000, 64, 20
    embeddings = _make_embeddings(N, D)

    labels, centers = _cluster_with_flash_kmeans(embeddings, n_clusters=K, device="cuda")

    assert labels.shape == (N,), f"Expected labels shape ({N},), got {labels.shape}"
    assert centers.shape == (K, D), f"Expected centers shape ({K}, {D}), got {centers.shape}"
    assert labels.dtype == np.int64
    assert centers.dtype == np.float32


@needs_cuda_and_flash
def test_flash_kmeans_valid_cluster_ids():
    """All returned cluster IDs are in [0, K)."""
    from sft_pipeline.clustering.clusterer import _cluster_with_flash_kmeans

    N, D, K = 3_000, 32, 15
    embeddings = _make_embeddings(N, D)
    labels, _ = _cluster_with_flash_kmeans(embeddings, n_clusters=K, device="cuda")

    assert labels.min() >= 0, "Negative cluster ID returned"
    assert labels.max() < K, f"Cluster ID {labels.max()} out of range [0, {K})"


@needs_cuda_and_flash
def test_flash_kmeans_all_points_assigned():
    """Every input point gets a cluster assignment (no unassigned -1)."""
    from sft_pipeline.clustering.clusterer import _cluster_with_flash_kmeans

    N, D, K = 1_000, 64, 10
    embeddings = _make_embeddings(N, D)
    labels, _ = _cluster_with_flash_kmeans(embeddings, n_clusters=K, device="cuda")

    assert len(labels) == N
    assert not np.any(labels < 0), "Some points were left unassigned"


@needs_cuda_and_flash
def test_flash_kmeans_deterministic():
    """Same data produces the same cluster centroids across two runs."""
    from sft_pipeline.clustering.clusterer import _cluster_with_flash_kmeans

    N, D, K = 2_000, 32, 10
    embeddings = _make_embeddings(N, D, seed=7)

    _, centers1 = _cluster_with_flash_kmeans(embeddings, n_clusters=K, device="cuda")
    _, centers2 = _cluster_with_flash_kmeans(embeddings, n_clusters=K, device="cuda")

    # Center positions should be very close (within float16 rounding)
    # Sort centers by their first dimension before comparing, since cluster
    # ordering may differ between runs.
    c1 = np.sort(centers1, axis=0)
    c2 = np.sort(centers2, axis=0)
    np.testing.assert_allclose(c1, c2, atol=1e-2,
                               err_msg="Centers differ between two runs on same data")


@needs_cuda_and_flash
def test_flash_kmeans_k_larger_than_n_clamped():
    """Requesting more clusters than points doesn't crash — K is clamped."""
    from sft_pipeline.clustering.clusterer import _cluster_with_flash_kmeans

    N, D = 50, 16
    embeddings = _make_embeddings(N, D)
    labels, centers = _cluster_with_flash_kmeans(embeddings, n_clusters=200, device="cuda")

    assert len(labels) == N
    assert centers.shape[0] <= N


@needs_flash
def test_flash_kmeans_raises_without_cuda(monkeypatch):
    """_cluster_with_flash_kmeans raises RuntimeError when no CUDA GPU is available."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    from sft_pipeline.clustering import clusterer
    # Force re-import so the monkeypatched value is seen inside the function
    import importlib
    importlib.reload(clusterer)

    with pytest.raises(RuntimeError, match="flash_kmeans requires a CUDA"):
        clusterer._cluster_with_flash_kmeans(_make_embeddings(10, 8), n_clusters=3, device="cuda")


# ---------------------------------------------------------------------------
# Integration: cluster_prompts with algorithm='flash_kmeans'
# ---------------------------------------------------------------------------

@needs_cuda_and_flash
def test_cluster_prompts_flash_kmeans_full_pipeline():
    """cluster_prompts() with flash_kmeans returns one result per prompt."""
    from sft_pipeline.clustering.clusterer import cluster_prompts

    N, D, K = 500, 64, 10
    embeddings = _make_embeddings(N, D)
    ids, texts = _make_prompts(N)

    results = cluster_prompts(
        prompt_ids=ids,
        prompts=texts,
        embeddings=embeddings,
        algorithm="flash_kmeans",
        n_clusters=K,
        device="cuda",
    )

    assert len(results) == N, "Expected one result per prompt"

    for r in results:
        assert "prompt_id" in r
        assert "domain" in r
        assert "difficulty" in r
        assert "cluster_id" in r
        assert r["domain"] in {"math", "code", "science", "language", "general"}
        assert r["difficulty"] in {"easy", "medium", "hard"}
        assert 0 <= r["cluster_id"] < K


@needs_cuda_and_flash
def test_cluster_prompts_flash_kmeans_domain_inference():
    """Domain labels reflect the content of the prompts."""
    from sft_pipeline.clustering.clusterer import cluster_prompts

    # Build two tight clusters: 200 math prompts, 200 code prompts.
    # Embeddings: math cluster near [1, 0, …], code cluster near [0, 1, …].
    D = 32
    math_vecs = np.zeros((200, D), dtype=np.float32)
    math_vecs[:, 0] = 1.0
    code_vecs = np.zeros((200, D), dtype=np.float32)
    code_vecs[:, 1] = 1.0
    embeddings = np.vstack([math_vecs, code_vecs])

    ids = [f"id_{i}" for i in range(400)]
    texts = (
        ["solve the equation x^2 + 2x - 3 = 0"] * 200
        + ["write a python function to sort a list"] * 200
    )

    results = cluster_prompts(
        prompt_ids=ids,
        prompts=texts,
        embeddings=embeddings,
        algorithm="flash_kmeans",
        n_clusters=2,
        device="cuda",
    )

    # Both groups should land in their own cluster; each should be labelled correctly.
    domain_by_id = {r["prompt_id"]: r["domain"] for r in results}
    math_domains = [domain_by_id[f"id_{i}"] for i in range(200)]
    code_domains = [domain_by_id[f"id_{i}"] for i in range(200, 400)]

    assert all(d == "math" for d in math_domains), f"Math prompts mislabelled: {set(math_domains)}"
    assert all(d == "code" for d in code_domains), f"Code prompts mislabelled: {set(code_domains)}"


@needs_cuda_and_flash
def test_cluster_prompts_flash_kmeans_ignores_centroids_arg():
    """Passing centroids=None with flash_kmeans must not raise."""
    from sft_pipeline.clustering.clusterer import cluster_prompts

    N, D = 200, 16
    ids, texts = _make_prompts(N)
    results = cluster_prompts(
        prompt_ids=ids,
        prompts=texts,
        embeddings=_make_embeddings(N, D),
        centroids=None,     # should be silently ignored for flash_kmeans
        faiss_index=None,
        algorithm="flash_kmeans",
        n_clusters=5,
        device="cuda",
    )
    assert len(results) == N


# ---------------------------------------------------------------------------
# Benchmark: flash_kmeans vs sklearn MiniBatchKMeans
# ---------------------------------------------------------------------------

@needs_cuda_and_flash
@pytest.mark.parametrize("N,D,K", [
    (100_000,  64,  50),
    (500_000, 128, 100),
])
def test_benchmark_flash_vs_sklearn(N, D, K, capsys):
    """
    Side-by-side timing comparison of flash_kmeans vs sklearn MiniBatchKMeans.

    Run with -s to see the printed table:
        pytest tests/unit/clustering/test_clusterer_flash_kmeans.py -v -s \
               -k test_benchmark
    """
    from sklearn.cluster import MiniBatchKMeans

    from sft_pipeline.clustering.clusterer import _cluster_with_flash_kmeans

    import torch

    embeddings = _make_embeddings(N, D, seed=0)

    # --- sklearn MiniBatchKMeans (CPU) ---
    km = MiniBatchKMeans(n_clusters=K, random_state=42, n_init=3, batch_size=4096)
    t0 = time.perf_counter()
    sklearn_labels = km.fit_predict(embeddings)
    sklearn_time = time.perf_counter() - t0

    # --- flash_kmeans (GPU) ---
    # Warm-up: first call includes Triton JIT compile time; time the second call.
    _cluster_with_flash_kmeans(_make_embeddings(100, D), n_clusters=5, device="cuda")
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    flash_labels, flash_centers = _cluster_with_flash_kmeans(embeddings, n_clusters=K, device="cuda")
    torch.cuda.synchronize()
    flash_time = time.perf_counter() - t0

    speedup = sklearn_time / flash_time if flash_time > 0 else float("inf")

    with capsys.disabled():
        print(
            f"\n{'─'*58}\n"
            f"  Clustering benchmark  N={N:,}  D={D}  K={K}\n"
            f"{'─'*58}\n"
            f"  flash_kmeans (GPU):          {flash_time:7.3f} s\n"
            f"  sklearn MiniBatchKMeans (CPU): {sklearn_time:7.3f} s\n"
            f"  speedup:                     {speedup:7.1f}×\n"
            f"{'─'*58}"
        )

    # Sanity checks (not correctness comparisons — different algorithms)
    assert flash_labels.shape == (N,)
    assert flash_centers.shape == (K, D)
    assert set(sklearn_labels.tolist()).issubset(set(range(K)))
    assert set(flash_labels.tolist()).issubset(set(range(K)))
