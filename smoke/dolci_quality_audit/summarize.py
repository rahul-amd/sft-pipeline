"""Validate final coverage and compute source-stratified audit aggregates."""
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

from review import ROOT, MODEL, sha, to_review, validate, atomic_json


def main():
    sample_bytes = (ROOT / "samples.jsonl").read_bytes()
    samples = [json.loads(line) for line in sample_bytes.splitlines()]
    verdict_bytes = (ROOT / "verdicts.jsonl").read_bytes()
    verdicts = [json.loads(line) for line in verdict_bytes.splitlines()]
    assert sha(sample_bytes) == json.loads((ROOT / "sample-manifest.json").read_text())["samples_sha256"]
    originals = {s["audit_id"]: s for s in samples}
    assert len(samples) == len(originals) == len(verdicts) == 3200
    assert {v["audit_id"] for v in verdicts} == set(originals)
    fields = ("audit_id", "overall", "summary", "uncertainty_reason", "dimensions", "issues")
    for verdict in verdicts:
        sample = originals[verdict["audit_id"]]
        assert verdict["source"] == sample["source"] and verdict["model"] == MODEL
        validate({"reviews": [{k: verdict[k] for k in fields}]}, [to_review(sample)])
    populations = {r["source"]: r["population"] for r in json.loads((ROOT / "sources.json").read_text())}
    groups = defaultdict(list)
    for v in verdicts:
        groups[v["source"]].append(v)
    assert set(groups) == set(populations) and all(len(vs) == 200 for vs in groups.values())
    total = sum(populations.values())
    labels = ["major_defect", "minor_defect", "uncertain", "no_defect_found"]
    weighted = {}
    for label in labels:
        estimate = variance = 0.0
        for source, values in groups.items():
            n, population = len(values), populations[source]
            weight = population / total
            p = sum(v["overall"] == label for v in values) / n
            estimate += weight * p
            variance += weight * weight * (1 - n / population) * p * (1 - p) / (n - 1)
        half = 1.959963984540054 * math.sqrt(variance)
        weighted[label] = {"estimate": estimate, "interval_95": [max(0, estimate-half), min(1, estimate+half)]}
    sources = []
    for source, values in groups.items():
        counts = Counter(v["overall"] for v in values)
        p, n, z = counts["major_defect"] / len(values), len(values), 1.959963984540054
        centre = (p+z*z/(2*n))/(1+z*z/n)
        half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        sources.append({"source": source, "population": populations[source], "sampled": n,
                        **{label: counts[label] for label in labels}, "major_rate": p,
                        "major_wilson_lower": centre-half, "major_wilson_upper": centre+half,
                        "major_contribution_pp": 100*populations[source]/total*p})
    sources.sort(key=lambda s: -s["major_contribution_pp"])
    turns = Counter(sum(m["role"] == "assistant" for m in s["record"]["messages"]) for s in samples)
    result = {"dataset": "allenai/Dolci-Think-SFT-7B", "model": MODEL,
              "population": total, "sample_size": len(samples), "source_count": len(groups),
              "samples_sha256": sha(sample_bytes), "verdicts_sha256": sha(verdict_bytes),
              "counts": dict(Counter(v["overall"] for v in verdicts)), "population_weighted": weighted,
              "sources": sources, "assistant_turn_count_distribution": dict(sorted(turns.items())),
              "dimensions": {d: dict(Counter(v["dimensions"][d] for v in verdicts)) for d in verdicts[0]["dimensions"]},
              "limitations": ["Single-model judgments; no human adjudication, code execution, or external fact verification",
                  "Sampling intervals condition on reviewer labels and omit model judgment uncertainty",
                  "Conversation-level labels cover every assistant turn; longer conversations have more opportunities for defects",
                  "Cross-dataset aggregate differences can reflect source, topic, language, and conversation-length mixtures"]}
    atomic_json(ROOT / "analysis.json", result)
    with (ROOT / "source_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sources[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(sources)
    print(json.dumps({"complete": True, "sample_size": len(samples), "counts": result["counts"], "weighted_major": weighted["major_defect"]}), flush=True)


if __name__ == "__main__":
    main()
