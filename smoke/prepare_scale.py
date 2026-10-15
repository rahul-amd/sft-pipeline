"""Snapshot bounded public source samples for the release validation.

Run on compute. Preserves context when instructions depend on it; keeps
references separately for audit, never feeds them to the teacher as answers.
"""
import hashlib
import itertools
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

from sft_pipeline.checkpoint import prompt_id
from sft_pipeline.stages.stage1_collect import _normalize, _extract_prompt

ROOT = Path("smoke/runs/freeze_validation/sources")
ROOT.mkdir(parents=True, exist_ok=True)
manifest = {"datasets": {}, "files": {}}
references = []


def dataset(repo, name=None):
    revision = HfApi().dataset_info(repo).sha
    manifest["datasets"][repo] = {"revision": revision, "config": name}
    return load_dataset(repo, name=name, split="train", revision=revision, streaming=True)


def save(name, rows):
    path = ROOT / name
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    manifest["files"][name] = {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    print(name, len(rows), flush=True)


code = []
for i, row in enumerate(dataset("iamtarun/python_code_instructions_18k_alpaca")):
    instruction = row["instruction"].strip()
    context = (row.get("input") or "").strip()
    prompt = instruction + ("\n\n" + context if context else "")
    if len(prompt) < 10 or len(prompt) > 6000:
        continue
    code.append({"prompt": prompt})
    references.append({"prompt_id": prompt_id(_normalize(_extract_prompt(prompt))), "source_row": i,
                       "repo": "iamtarun/python_code_instructions_18k_alpaca", "reference": row.get("output")})
    if len(code) == 2000:
        break
assert len(code) == 2000
save("code.jsonl", code)

general, documents = [], []
seen_documents = set()
for i, row in enumerate(dataset("databricks/databricks-dolly-15k")):
    context = (row.get("context") or "").strip()
    instruction = row["instruction"].strip()
    prompt = instruction + ("\n\nContext:\n" + context if context else "")
    if len(general) < 1800 and 10 <= len(prompt) <= 6000:
        general.append({"prompt": prompt})
        references.append({"prompt_id": prompt_id(_normalize(_extract_prompt(prompt))), "source_row": i,
                           "repo": "databricks/databricks-dolly-15k", "reference": row.get("response")})
    passage = context[:3500]
    if 200 <= len(passage) and passage not in seen_documents and len(documents) < 500:
        documents.append({"text": passage, "source_row": i})
        seen_documents.add(passage)
    if len(general) == 1800 and len(documents) == 500:
        break
assert len(general) == 1800 and len(documents) == 500
save("general.jsonl", general)
save("corpus.jsonl", documents)

# Record revisions and references for sources used by Stage 1.
for i, row in enumerate(itertools.islice(dataset("openai/gsm8k", "main"), 5000)):
    references.append({"prompt_id": prompt_id(_normalize(_extract_prompt(row["question"]))),
        "repo": "openai/gsm8k", "source_row": i, "reference": row["answer"]})
science = []
for i, row in enumerate(dataset("allenai/ai2_arc", "ARC-Challenge")):
    choices = row["choices"]
    prompt = row["question"] + "\n\nOptions:\n" + "\n".join(
        f"{label}. {text}" for label, text in zip(choices["label"], choices["text"]))
    science.append({"prompt": prompt})
    references.append({"prompt_id": prompt_id(_normalize(_extract_prompt(prompt))),
        "repo": "allenai/ai2_arc", "source_row": i,
        "reference": {"answerKey": row["answerKey"], "choices": row["choices"]}})
save("science.jsonl", science)
save("references.jsonl", references)
(ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
