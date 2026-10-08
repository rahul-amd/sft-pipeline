"""Exercise the real CLI and HTTP clients against a deterministic protocol fixture.

The server is not a quality benchmark or live model. Embeddings are synthetic;
FAISS, clustering, sampling, checkpointing, filtering and export are real.
The separate smoke/config.yaml run exercises real embeddings and live inference.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from sft_pipeline.checkpoint import CheckpointManager
from sft_pipeline.cli import app
from sft_pipeline.config import load_config
from sft_pipeline.storage import iter_jsonl_dir


@pytest.fixture
def api_server():
    state = {"calls": [], "fail_teacher": False, "empty_teacher": False,
             "truncated_teacher": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            if self.headers.get("Authorization") != "Bearer test-key":
                self.send_error(401)
                return
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["calls"].append(req)
            system = req["messages"][0]["content"]
            message = {}
            if "creating educational" in system:
                content = json.dumps([
                    "Calculate the perimeter of a rectangle with sides seven and five.",
                    "How many outcomes are possible when rolling two six-sided dice?",
                ])
            elif "prompt classifier" in system:
                content = json.dumps({"domain": "math", "difficulty": "easy",
                                      "topics": ["arithmetic"], "language": "en",
                                      "summary": "Solve a short arithmetic problem."})
            elif "expert evaluator" in system:
                content = '{"overall": 9}'
            else:
                if state["fail_teacher"]:
                    self.send_error(503)
                    return
                content = "" if state["empty_teacher"] else "The result follows from the calculation."
                reasoning = (
                    "We identify the quantities in the question and apply the relevant rule. "
                    "Checking each step against the original conditions gives a consistent result. "
                    "This calculation accounts for all the information provided in the question."
                )
                if not state["empty_teacher"]:
                    message["reasoning_content"] = reasoning
                if state["truncated_teacher"]:
                    content = ""
                    message = {"reasoning": reasoning}
            payload = json.dumps({"id": "test", "object": "chat.completion", "created": 0,
                                  "model": "fixture", "choices": [{"index": 0,
                                  "finish_reason": "length" if state["truncated_teacher"] else "stop",
                                  "message": {"role": "assistant",
                                  "content": content, **message}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1", state
    server.shutdown()
    server.server_close()
    thread.join()


def make_config(tmp_path, api_base):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("\n".join(json.dumps({"prompt": p}) for p in [
        "What is the sum of twelve and nineteen?",
        "What is the product of thirteen and seven?",
        "Explain why the old library clock rings exactly thirteen times at midnight every winter.",
    ]) + "\n")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("A rectangle has four sides and four right angles. Its perimeter is "
                      "twice the sum of its length and width. Rolling two ordinary dice "
                      "produces thirty-six equally likely ordered outcomes.")
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(json.dumps({"question":
        "the old library clock rings exactly thirteen times at midnight every winter"}) + "\n")
    data = {
        "global": {"base_path": str(tmp_path / "run"), "device": "cpu"},
        "stage1_collect": {"datasets": [{"source": "local_jsonl", "path": str(prompts)}]},
        "stage2_generate": {"corpora": [{"source": "local", "path": str(corpus)}],
                            "generator_endpoint": api_base, "generator_api_key": "test-key"},
        "decontaminate": {"evals": [{"name": "fixture", "source": "local_jsonl",
                                    "path": str(benchmark), "match_fields": ["question"]}]},
        "stage3_cluster": {"clustering_algorithm": "kmeans", "n_clusters": 2,
                           "faiss_nlist": 1, "annotation_enabled": True,
                           "annotation_api_base": api_base, "annotation_api_key": "test-key"},
        "stage4_sample": {"total_prompts": 4, "domain_quotas": {"math": 1.0},
                          "difficulty_quotas": {"default": {"easy": 1.0}}},
        "stage5_inference": {"inference_mode": "openai_api", "api_base": api_base,
                             "api_key": "test-key", "concurrency": 2, "max_retries": 0,
                             "checkpoint_every": 1},
        "stage6_filter": {"llm_judge": {"enabled": False}},
        "export": {"push_to_hub": False},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_all_stages_cli_and_resume(tmp_path, api_server, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from sft_pipeline.stages import stage3_cluster

    def embed(prompt_iter, output_dir, **kwargs):
        records = list(prompt_iter)
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(len(records), 8))
        vectors = (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float16)
        pq.write_table(pa.table({"prompt_id": [r["prompt_id"] for r in records],
                                "embedding": pa.FixedSizeListArray.from_arrays(
                                    pa.array(vectors.ravel(), type=pa.float16()), 8)}),
                       Path(output_dir) / "embeddings_0000.parquet")
        return len(records)

    monkeypatch.setattr(stage3_cluster, "embed_prompts", embed)
    api_base, state = api_server
    config = make_config(tmp_path, api_base)
    result = CliRunner().invoke(app, ["run", "--config", str(config)])
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    cfg = load_config(config)
    final = list(iter_jsonl_dir(Path(cfg.export.final_jsonl_path).parent))
    assert len(final) == 4
    assert all(r["reasoning"] and r["answer"] for r in final)
    assert len({r["id"] for r in final}) == 4
    assert any("corpus.txt" in r["source"] for r in final)
    report = json.loads(Path(cfg.decontaminate.report_path).read_text())
    assert report["total_removed"] == 1
    with CheckpointManager(cfg.global_.checkpoint_db) as cm:
        assert len(cm.all_stage_statuses()) == 8
        assert all(r["status"] == "completed" for r in cm.all_stage_statuses())
    before = len(state["calls"])
    second = CliRunner().invoke(app, ["run", "--config", str(config)])
    assert second.exit_code == 0, second.output
    assert len(state["calls"]) == before
    assert list(iter_jsonl_dir(Path(cfg.export.final_jsonl_path).parent)) == final
    # An explicit export rerun replaces the dataset instead of duplicating it.
    rerun = CliRunner().invoke(app, ["run-stage", "export", "--config", str(config)])
    assert rerun.exit_code == 0, rerun.output
    assert list(iter_jsonl_dir(Path(cfg.export.final_jsonl_path).parent)) == final


@pytest.mark.parametrize("failure", ["fail_teacher", "empty_teacher"])
def test_api_failure_is_retried_on_resume(tmp_path, api_server, failure):
    api_base, state = api_server
    config = make_config(tmp_path, api_base)
    cfg = load_config(config)
    out = Path(cfg.stage4_sample.output_dir)
    out.mkdir(parents=True)
    (out / "part-000000.jsonl").write_text(json.dumps({
        "prompt_id": "p1", "prompt": "Compute two plus three.", "domain": "math"}) + "\n")
    state[failure] = True
    runner = CliRunner()
    failed = runner.invoke(app, ["run-stage", "stage5_inference", "--config", str(config)])
    assert failed.exit_code != 0
    assert not list(iter_jsonl_dir(cfg.stage5_inference.output_dir))
    with CheckpointManager(cfg.global_.checkpoint_db) as cm:
        assert not cm.is_processed("p1", "stage5_inference")
        assert not cm.is_stage_complete("stage5_inference")
    state[failure] = False
    passed = runner.invoke(app, ["run-stage", "stage5_inference", "--config", str(config)])
    assert passed.exit_code == 0, passed.output
    assert len(list(iter_jsonl_dir(cfg.stage5_inference.output_dir))) == 1


def test_truncated_separate_reasoning_reaches_quality_filter(tmp_path, api_server):
    api_base, state = api_server
    config = make_config(tmp_path, api_base)
    cfg = load_config(config)
    path = Path(cfg.stage4_sample.output_dir)
    path.mkdir(parents=True)
    (path / "part-000000.jsonl").write_text(json.dumps({
        "prompt_id": "p1", "prompt": "Compute two plus three.", "domain": "math"}) + "\n")
    state["truncated_teacher"] = True
    runner = CliRunner()
    for stage in ("stage5_inference", "stage6_filter"):
        result = runner.invoke(app, ["run-stage", stage, "--config", str(config)])
        assert result.exit_code == 0, result.output
    rows = list(iter_jsonl_dir(cfg.stage5_inference.output_dir))
    assert rows[0]["reasoning"] and not rows[0]["answer"]
    report = json.loads(Path(cfg.stage6_filter.report_path).read_text())
    assert report["total_input"] == 1
    assert report["total_passed"] == 0
    assert report["rejection_counts"] == {"structural": 1}
