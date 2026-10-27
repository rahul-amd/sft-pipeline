"""Uniform sample of 200 physical conversations per exact source, without replacement.

Select source-relative ranks first, then fetch only the Parquet row groups needed
to retrieve them. Retain every original message verbatim. Restartable shard caches
are keyed by the pinned revision and the hash of the complete sampling plan.
"""
from bisect import bisect_left
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

from inventory_sources import ROOT, REPO, REVISION

SEED = 20261027
PER_SOURCE = 200


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False).encode()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main(workers=1):
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    destination = ROOT / "samples.jsonl"
    if destination.exists():
        manifest = json.loads((ROOT / "sample-manifest.json").read_text())
        assert sha(destination.read_bytes()) == manifest["samples_sha256"]
        print("Frozen sample already exists and its hash is valid.", flush=True)
        return
    inventory = json.loads((ROOT / "sources.json").read_text())
    populations = {r["source"]: r["population"] for r in inventory}
    ranks = {}
    for source, population in populations.items():
        rng = random.Random(int.from_bytes(hashlib.sha256(f"{SEED}:{source}".encode()).digest(), "big"))
        ranks[source] = sorted(rng.sample(range(population), PER_SOURCE))
    plan = {"dataset": REPO, "revision": REVISION, "seed": SEED,
            "per_source": PER_SOURCE, "source_ranks_zero_based": ranks}
    plan_hash = sha(encoded(plan))
    atomic_json(ROOT / "sampling-plan.json", plan)
    shards = [json.loads(p.read_text()) for p in sorted((ROOT / "shards" / REVISION).glob("*.json"))]
    offsets, tasks = Counter(), []
    for shard in shards:
        wanted = {}
        for entry in shard["sources"]:
            source, count = entry["source"], entry["population"]
            start, end = offsets[source], offsets[source] + count
            chosen = ranks[source][bisect_left(ranks[source], start):bisect_left(ranks[source], end)]
            if chosen:
                wanted[source] = {rank - start: rank for rank in chosen}
            offsets[source] += count
        if wanted:
            tasks.append((shard, wanted))
    assert dict(offsets) == populations
    cache_dir = ROOT / "sampled_shards" / plan_hash
    cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(task):
        shard, wanted = task
        cache = cache_dir / (Path(shard["path"]).stem + ".json")
        if cache.exists():
            return json.loads(cache.read_text())
        for attempt in range(4):
            try:
                url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{shard['path']}"
                with fsspec.open(url, "rb", block_size=65536) as stream:
                    parquet = pq.ParquetFile(stream)
                    assert shard["rows"] == parquet.metadata.num_rows
                    if len(shard["sources"]) == 1:
                        # The exact inventory already scanned this immutable
                        # shard: source-relative row rank is its physical row.
                        source = shard["sources"][0]["source"]
                        indices = {local: (source, rank) for local, rank in wanted[source].items()}
                    else:
                        sources = parquet.read(columns=["dataset_source"], use_threads=False).column(0).to_pylist()
                        assert Counter(sources) == Counter({r["source"]: r["population"] for r in shard["sources"]})
                        assert len(sources) == shard["rows"]
                        seen, indices = Counter(), {}
                        for index, source in enumerate(sources):
                            if seen[source] in wanted.get(source, {}):
                                indices[index] = (source, wanted[source][seen[source]])
                            seen[source] += 1
                    assert len(indices) == sum(len(rs) for rs in wanted.values())
                    selected, offset = [], 0
                    for group in range(parquet.num_row_groups):
                        length = parquet.metadata.row_group(group).num_rows
                        subset = sorted(i for i in indices if offset <= i < offset + length)
                        if subset:
                            table = parquet.read_row_group(group, columns=["messages", "dataset_source", "id"], use_threads=False)
                            records = table.take(pa.array([i - offset for i in subset], type=pa.int64())).to_pylist()
                            for index, record in zip(subset, records):
                                source, rank = indices[index]
                                assert record["dataset_source"] == source
                                assert isinstance(record["messages"], list)
                                record_hash = sha(encoded(record))
                                audit_id = sha(f"{REPO}:{REVISION}:{shard['path']}:{index}:{record_hash}".encode())[:24]
                                selected.append({"audit_id": audit_id, "source": source,
                                    "prompt_id": record["id"], "input_shard": shard["path"],
                                    "input_row_zero_based": index, "input_row_group": group,
                                    "source_rank_zero_based": rank, "record_sha256": record_hash,
                                    "record": record})
                            del table
                        offset += length
                    result = {"path": shard["path"], "plan_sha256": plan_hash,
                              "range_bytes_requested": stream.cache.total_requested_bytes, "samples": selected}
                atomic_json(cache, result)
                return result
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(fetch, task) for task in tasks]):
            results.append(future.result())
            progress = {"completed_shards": len(results), "total_shards": len(tasks),
                        "fetched_samples": sum(len(r["samples"]) for r in results), "total_samples": PER_SOURCE * len(populations)}
            atomic_json(ROOT / "sampling-progress.json", progress)
            print(json.dumps(progress), flush=True)
    samples = sorted((s for r in results for s in r["samples"]), key=lambda s: (s["source"], s["audit_id"]))
    assert len(samples) == len({s["audit_id"] for s in samples}) == PER_SOURCE * len(populations)
    assert Counter(s["source"] for s in samples) == Counter({source: PER_SOURCE for source in populations})
    for source in ranks:
        assert sorted(s["source_rank_zero_based"] for s in samples if s["source"] == source) == ranks[source]
    data = b"".join(encoded(s) + b"\n" for s in samples)
    temporary = destination.with_suffix(".jsonl.tmp")
    temporary.write_bytes(data)
    manifest = {"dataset": REPO, "revision": REVISION, "split": "train",
        "created_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED, "per_source_limit": PER_SOURCE,
        "method": "Uniform source-relative ranks without replacement; independent source RNGs; sorted shard/row order",
        "sampling_unit": "physical conversation record; all original messages retained; duplicate IDs not collapsed",
        "population": sum(populations.values()), "source_count": len(populations), "sample_size": len(samples),
        "samples_sha256": sha(data), "plan_sha256": plan_hash, "sampler_sha256": sha(Path(__file__).read_bytes()),
        "source_inventory_sha256": sha((ROOT / "sources.json").read_bytes()),
        "range_bytes_requested": sum(r["range_bytes_requested"] for r in results)}
    atomic_json(ROOT / "sample-manifest.json", manifest)
    temporary.replace(destination)
    print(json.dumps({"complete": True, **manifest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    main(parser.parse_args().workers)
