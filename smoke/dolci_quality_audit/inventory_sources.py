"""Count exact dataset_source values across a pinned Dolci dataset snapshot.

Reads only the source column over HTTP ranges; does not download conversations.
Requires requests, fsspec[http], and pyarrow. All outputs stay beside this script.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import fsspec
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parent
REPO = "allenai/Dolci-Think-SFT-7B"
REVISION = "83f618e2fe0aed29ee3440a3658a0326c175d09d"


def inspect_shard(path):
    cache = ROOT / "shards" / REVISION / (Path(path).stem + ".json")
    if cache.exists():
        return json.loads(cache.read_text())
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{path}"
    for attempt in range(4):
        try:
            with fsspec.open(url, "rb", block_size=65536) as stream:
                parquet = pq.ParquetFile(stream)
                column = parquet.read(columns=["dataset_source"], use_threads=False).column(0)
                counts = column.value_counts().to_pylist()
                result = {
                    "path": path, "revision": REVISION,
                    "rows": parquet.metadata.num_rows,
                    "null_count": column.null_count,
                    "sources": [{"source": x["values"], "population": x["counts"]} for x in counts],
                    "range_bytes_requested": stream.cache.total_requested_bytes,
                }
                assert sum(x["population"] for x in result["sources"]) == result["rows"]
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(result, indent=2) + "\n")
            return result
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def main():
    response = requests.get(f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}", timeout=60)
    response.raise_for_status()
    metadata = response.json()
    assert metadata["sha"] == REVISION
    (ROOT / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    paths = sorted(x["rfilename"] for x in metadata["siblings"] if x["rfilename"].startswith("data/train-") and x["rfilename"].endswith(".parquet"))
    assert paths
    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(inspect_shard, path) for path in paths]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 10 == 0 or len(results) == len(paths):
                print(f"Scanned {len(results)}/{len(paths)} shards", flush=True)
    totals = Counter()
    for shard in results:
        for row in shard["sources"]:
            totals[row["source"]] += row["population"]
    n = sum(totals.values())
    expected = sum(x["num_examples"] for x in metadata["cardData"]["dataset_info"]["splits"] if x["name"] == "train")
    assert n == expected, (n, expected)
    rows = [{"source": source, "population": population, "percentage": 100 * population / n} for source, population in totals.most_common()]
    (ROOT / "sources.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (ROOT / "sources.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source", "population", "percentage"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "dataset": REPO, "revision": REVISION, "split": "train",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Exact value_counts of dataset_source over every original Parquet shard using HTTP range reads",
        "shards": len(results), "total_records": n,
        "distinct_non_null_sources": sum(source is not None for source in totals),
        "null_source_records": totals.get(None, 0), "empty_source_records": totals.get("", 0),
        "range_bytes_requested": sum(x["range_bytes_requested"] for x in results),
        "sampling_started": False, "reviews_started": False,
    }
    (ROOT / "inventory-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": manifest, "sources": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
