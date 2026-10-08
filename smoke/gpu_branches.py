"""Live branch probes. Run one case per process, on an allocated compute node.

All inputs/results stay in smoke/. Failures are saved before the process exits.
These are component probes complementing the full runs, not extra full runs.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MATRIX = ROOT / "smoke/runs/gpu_matrix_20261008"


def config(name, changes=None):
    from sft_pipeline.config import load_config
    raw = yaml.safe_load((ROOT / "smoke/config-gpu.yaml").read_text())
    raw["global"]["run_id"] = f"gpu_case_{name}_20261008"
    raw["global"]["ray_address"] = os.environ.get("SFT_RAY_ADDRESS", "auto")
    if changes:
        for key, values in changes.items():
            raw[key].update(values)
    path = MATRIX / f"config-{name}.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    cfg = load_config(path)
    Path(cfg.base_path).mkdir(parents=True, exist_ok=True)
    return cfg, path


def cli(path, *args):
    command = [sys.executable, "-m", "sft_pipeline.cli", *args, "--config", str(path)]
    subprocess.run(command, check=True, timeout=1200)


def write_rows(path, rows):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "part-000000.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def ray_probe():
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    ray.init(address=os.environ["SFT_RAY_ADDRESS"])

    @ray.remote(num_gpus=1)
    def gpu():
        import torch
        return {"host": socket.gethostname(), "gpu_count": torch.cuda.device_count(),
                "hip_visible": os.environ.get("HIP_VISIBLE_DEVICES"),
                "rocr_visible": os.environ.get("ROCR_VISIBLE_DEVICES"),
                "kernel_result": (torch.ones(3, device="cuda") * 2).cpu().tolist()}

    nodes = [n for n in ray.nodes() if n["Alive"]]
    assert len(nodes) == 2, nodes
    results = ray.get([gpu.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
        n["NodeID"], soft=False)).remote() for n in nodes])
    assert len({r["host"] for r in results}) == 2
    return {"workers": results, "resources": ray.cluster_resources()}


def components():
    import faiss
    import numpy as np
    import torch
    from sft_pipeline.clustering.clusterer import cluster_prompts
    from sft_pipeline.clustering.faiss_index import build_and_save, load_index
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(512, 64)).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = [f"synthetic-{i}" for i in range(len(vectors))]
    texts = ["Solve this equation and explain the solution." for _ in ids]
    results = {}
    for kind in ("Flat", "IVFFlat", "IVFPQ"):
        path = MATRIX / f"index-{kind}.faiss"
        build_and_save(ids, vectors, path, index_type=kind, nlist=16,
                       nprobe=16, training_sample=512, batch_size=128, device="rocm")
        index = load_index(path, device="rocm")
        distances, neighbors = index.search(vectors[:3], 3)
        assert index.ntotal == len(ids)
        assert all(i in row for i, row in enumerate(neighbors)), neighbors
        results[kind] = {"count": index.ntotal, "neighbors": neighbors.tolist(),
                         "metric_type": index.metric_type}
    index = load_index(MATRIX / "index-IVFFlat.faiss")
    centroids = np.vstack([index.quantizer.reconstruct(i) for i in range(index.nlist)])
    for algorithm in ("kmeans", "hdbscan", "faiss_kmeans", "torch_kmeans", "flash_kmeans"):
        rows = cluster_prompts(ids, texts, vectors, centroids, index, algorithm=algorithm,
                               min_cluster_size=2, n_clusters=4, device="rocm",
                               training_sample=512)
        assert len(rows) == len(ids) and all("centroid_sim" in r for r in rows)
        results[algorithm] = {"rows": len(rows), "clusters": len({r["cluster_id"] for r in rows})}
    return {"gpu_kernel": torch.ones(1, device="cuda").cpu().tolist(),
            "faiss_gpu_count": faiss.get_num_gpus(), "cases": results}


def offline(replicas):
    from sft_pipeline.checkpoint import CheckpointManager, prompt_id
    from sft_pipeline.stages.stage5_inference import run_stage5
    from sft_pipeline.storage import iter_jsonl_dir
    cfg, path = config(f"offline_{replicas}", {"stage5_inference": {
        "inference_mode": "vllm_offline", "model": "Qwen/Qwen3-4B",
        "n_replicas": replicas, "batch_size": 1, "checkpoint_every": 1,
        "generation": {"temperature": 0.2, "top_p": 0.95, "max_tokens": 1024, "n_candidates": 2},
        "vllm_engine": {"gpu_memory_utilization": 0.15, "max_model_len": 2048}}})
    prompts = ["What is 2 + 3? Explain briefly.", "What is 6 times 7? Explain briefly."]
    write_rows(cfg.stage4_sample.output_dir,
               [{"prompt_id": prompt_id(p), "prompt": p, "domain": "math"} for p in prompts])
    with CheckpointManager(cfg.global_.checkpoint_db) as cm:
        run_stage5(cfg, cm)
    rows = list(iter_jsonl_dir(cfg.stage5_inference.output_dir))
    assert len(rows) == 2 and all(r["response"].strip() for r in rows)
    return {"responses": len(rows), "config": str(path),
            "finish_reason_recorded": all("finish_reason" in r for r in rows)}


def corpus_hf():
    from sft_pipeline.config import CorpusSource
    from sft_pipeline.stages.stage2_generate import _iter_hf_corpus, _chunk_text, _call_generator
    # Bound iteration explicitly: the Stage 2 corpus schema currently has no cap.
    src = CorpusSource(source="hf_dataset", hf_repo_id="roneneldan/TinyStories", text_field="text")
    docs = list(itertools.islice(_iter_hf_corpus(src), 1))
    assert len(docs) == 1
    doc_id, text = docs[0]
    chunks = _chunk_text(text, 256, 32)
    questions = _call_generator(chunks[0], 2, os.environ["SFT_API_BASE"],
                                os.environ["SFT_MODEL"], 0.2, 4096, os.environ["SFT_API_KEY"])
    assert questions and all(isinstance(q, str) and q.strip() for q in questions)
    return {"source": src.hf_repo_id, "document_id": doc_id, "chunks": len(chunks),
            "questions": questions, "scope": "one streamed document; first chunk sent to real API"}


def annotations():
    from openai import OpenAI
    from sft_pipeline.storage import iter_jsonl_dir
    cfg, path = config("annotations", {"decontaminate": {"enabled": False},
                                       "stage3_cluster": {"annotation_enabled": False}})
    source = ROOT / "smoke/runs/gpu_full_v2_20261008/stage3"
    rows = list(iter_jsonl_dir(source))[:3]
    assert len(rows) == 3
    write_rows(cfg.stage1_collect.output_dir, rows)
    write_rows(cfg.stage3_cluster.output_dir, rows)
    requests = Path(cfg.base_path) / "requests.jsonl"
    responses = Path(cfg.base_path) / "responses.jsonl"
    cli(path, "run-stage", "stage3_cluster", "--dump-annotations", str(requests))
    client = OpenAI(base_url=os.environ["SFT_API_BASE"], api_key=os.environ["SFT_API_KEY"])
    replies = []
    for request in map(json.loads, requests.read_text().splitlines()):
        response = client.chat.completions.create(model=os.environ["SFT_MODEL"],
            messages=request["messages"], max_tokens=4096, temperature=0.1)
        replies.append({"prompt_id": request["prompt_id"], "response": response.choices[0].message.content})
    responses.write_text("".join(json.dumps(r) + "\n" for r in replies))
    cli(path, "run-stage", "stage3_cluster", "--import-annotations", str(responses))
    imported = list(iter_jsonl_dir(cfg.stage3_cluster.output_dir))
    assert {r["prompt_id"] for r in imported} == {r["prompt_id"] for r in rows}
    assert {r["prompt_id"]: r["centroid_sim"] for r in imported} == {
        r["prompt_id"]: r["centroid_sim"] for r in rows}
    cli(path, "annotate")  # disabled branch
    raw = yaml.safe_load(path.read_text())
    raw["stage3_cluster"]["annotation_enabled"] = True
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    # Delete only the probe checkpoint to test fresh online annotation and resume.
    (Path(cfg.stage3_cluster.output_dir) / "annotations.parquet").unlink()
    cli(path, "annotate")
    cli(path, "annotate")
    return {"requests": len(replies), "imported": len(imported), "centroid_sim_preserved": True,
            "standalone_annotation_and_resume": True}


def cli_paths():
    cfg, path = config("cli")
    cli(path, "status")
    cli(path, "estimate")
    cli(path, "run", "--dry-run")
    raw = yaml.safe_load(path.read_text())
    for name, value in raw.items():
        if name != "global":
            value["enabled"] = False
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    cli(path, "run")
    cli(path, "run", "--no-resume")
    return {"status_estimate_dry_run_disabled_no_resume": True}


def cached_flat():
    import hashlib
    from sft_pipeline.checkpoint import CheckpointManager
    from sft_pipeline.stages.stage3_cluster import run_stage3
    from sft_pipeline.storage import iter_jsonl_dir
    cfg, path = config("cached_flat", {"decontaminate": {"enabled": False},
        "stage3_cluster": {"annotation_enabled": False, "faiss_index_type": "Flat",
                           "clustering_algorithm": "hdbscan"}})
    source = ROOT / "smoke/runs/gpu_full_v3_20261008/stage3"
    rows = list(iter_jsonl_dir(source))[:4]
    assert len(rows) == 4
    write_rows(cfg.stage1_collect.output_dir, rows)
    def hashes():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(cfg.stage3_cluster.embeddings_dir).glob("*.parquet")}
    with CheckpointManager(cfg.global_.checkpoint_db) as cm:
        run_stage3(cfg, cm)
        before = hashes()
        first = list(iter_jsonl_dir(cfg.stage3_cluster.output_dir))
        run_stage3(cfg, cm)
    after = list(iter_jsonl_dir(cfg.stage3_cluster.output_dir))
    result = {"first_count": len(first), "rerun_count": len(after),
              "unique_ids": len({r["prompt_id"] for r in after}),
              "embedding_files_unchanged": before == hashes(),
              "heuristic_flat_fallback": all(r["cluster_id"] == -1 for r in after)}
    (MATRIX / "cached-flat-observations.json").write_text(json.dumps(result, indent=2) + "\n")
    assert before == hashes()
    assert len(after) == result["unique_ids"] == len(rows), result
    return result


CASES = {"ray_probe": ray_probe, "components": components,
         "offline_single": lambda: offline(1), "offline_ray": lambda: offline(2),
         "corpus_hf": corpus_hf, "annotations": annotations, "cli_paths": cli_paths,
         "cached_flat": cached_flat}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=CASES)
    args = parser.parse_args()
    os.chdir(ROOT)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    start = time.monotonic()
    report = {"case": args.case, "host": socket.gethostname(), "status": "failed"}
    try:
        report["result"] = CASES[args.case]()
        report["status"] = "passed"
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - start, 2)
        (MATRIX / f"{args.case}.json").write_text(json.dumps(report, indent=2) + "\n")
