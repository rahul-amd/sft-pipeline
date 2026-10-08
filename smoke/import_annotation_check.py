"""Live API check for missing and empty imported annotations, in isolation."""
import json
from pathlib import Path
import yaml

from sft_pipeline.checkpoint import CheckpointManager
from sft_pipeline.config import load_config
from sft_pipeline.stages.stage3_cluster import run_stage3
from sft_pipeline.storage import iter_jsonl_dir

root = Path("smoke/runs/freeze_validation")
raw = yaml.safe_load(Path("smoke/config-scale.yaml").read_text())
raw["global"]["run_id"] = "freeze_import_fallback"
raw["decontaminate"]["enabled"] = False
path = root / "import-fallback.yaml"
path.write_text(yaml.safe_dump(raw, sort_keys=False))
cfg = load_config(path)
Path(cfg.base_path).mkdir(parents=True, exist_ok=False)
records = list(iter_jsonl_dir("smoke/runs/freeze_retry_stages34/stage3"))[:3]
assert len(records) == 3
for directory in (cfg.stage1_collect.output_dir, cfg.stage3_cluster.output_dir):
    dst = Path(directory)
    dst.mkdir(parents=True)
    (dst / "part-000000.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
imported = root / "import-fallback-responses.jsonl"
imported.write_text(json.dumps({"prompt_id": records[0]["prompt_id"], "response": {
    "domain": "science", "difficulty": "hard", "summary": "Imported fixture annotation"}})+"\n"+
    json.dumps({"prompt_id": records[1]["prompt_id"], "response": ""})+"\n")
with CheckpointManager(cfg.global_.checkpoint_db) as cm:
    run_stage3(cfg, cm, import_annotations_path=imported)
    assert cm.is_stage_complete("stage3_cluster")
after = {r["prompt_id"]: r for r in iter_jsonl_dir(cfg.stage3_cluster.output_dir)}
assert len(after) == 3
assert after[records[0]["prompt_id"]]["summary"] == "Imported fixture annotation"
assert all(after[r["prompt_id"]]["summary"] for r in records[1:])
assert all(after[r["prompt_id"]]["centroid_sim"] == r["centroid_sim"] for r in records)
(root / "import-fallback.json").write_text(json.dumps({"status": "passed", "rows": 3,
    "imported": 1, "empty_response_filled_online": 1, "missing_response_filled_online": 1,
    "centroid_similarity_preserved": True}, indent=2)+"\n")
