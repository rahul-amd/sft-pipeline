"""Uniform per-source reservoir sample of the retained dataset; read-only inputs."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import orjson

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[1] / "runs/research_stage3_001/stage6_v2_decontam/clean"
SEED = 20261026
LIMIT = 100


def main():
    destination = ROOT / "samples.jsonl"
    if destination.exists():
        raise SystemExit("Sample already exists; refusing to replace a frozen sample")
    files = sorted(DATA.glob("*.jsonl"))
    assert files, DATA
    counts, reservoirs, rngs = Counter(), {}, {}
    manifest = []
    started = time.monotonic()
    for number, path in enumerate(files, 1):
        before = path.stat()
        digest = hashlib.sha256()
        rows = 0
        with path.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                digest.update(raw)
                if not raw.strip():
                    continue
                record = orjson.loads(raw)
                source = record.get("source")
                if not isinstance(source, str) or not source.strip():
                    raise ValueError(f"Missing source: {path}:{line_number}")
                counts[source] += 1
                rows += 1
                if source not in reservoirs:
                    reservoirs[source] = []
                    rngs[source] = random.Random(int.from_bytes(hashlib.sha256(f"{SEED}:{source}".encode()).digest(), "big"))
                pool = reservoirs[source]
                slot = len(pool) if len(pool) < LIMIT else rngs[source].randrange(counts[source])
                if slot < LIMIT:
                    row_hash = hashlib.sha256(raw).hexdigest()
                    audit_id = hashlib.sha256(f"{path.name}:{line_number}:{row_hash}".encode()).hexdigest()[:24]
                    selected = {"audit_id": audit_id, "source": source, "prompt_id": record.get("prompt_id"),
                                "input_shard": path.name, "input_line": line_number, "raw_line_sha256": row_hash,
                                "record": record}
                    if slot == len(pool):
                        pool.append(selected)
                    else:
                        pool[slot] = selected
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Input changed during sampling: {path}")
        manifest.append({"file": path.name, "bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
                         "rows": rows, "sha256": digest.hexdigest()})
        if number % 10 == 0 or number == len(files):
            print(json.dumps({"shards": number, "total_shards": len(files), "rows": sum(counts.values()),
                              "sources": len(counts), "elapsed_seconds": round(time.monotonic()-started, 1)}), flush=True)
    selected = [r for source in sorted(reservoirs) for r in sorted(reservoirs[source], key=lambda r: r["audit_id"])]
    temp = destination.with_suffix(".jsonl.tmp")
    with temp.open("wb") as stream:
        for record in selected:
            stream.write(orjson.dumps(record) + b"\n")
    temp.replace(destination)
    inventory = [{"source": s, "population": counts[s], "sampled": len(reservoirs[s])} for s in sorted(counts)]
    (ROOT / "sources.json").write_text(json.dumps(inventory, indent=2) + "\n")
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "input_directory": str(DATA),
                "seed": SEED, "per_source_limit": LIMIT, "method": "uniform reservoir sampling without replacement within source; sorted shard/line traversal; independent source RNGs",
                "sampling_unit": "physical record (duplicate prompt IDs are not silently collapsed)",
                "population": sum(counts.values()), "source_count": len(counts), "sample_size": len(selected),
                "samples_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "sampler_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "shards": manifest}
    (ROOT / "sample-manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"complete": True, "population": sum(counts.values()), "sample_size": len(selected), "sources": inventory}), flush=True)


if __name__ == "__main__":
    main()
