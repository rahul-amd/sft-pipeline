"""Restricted, disposable Docker execution with no host data mounts.

Docker must be available on each filtering worker and the image pre-pulled.
The daemon/image are trusted infrastructure; containers share the host kernel.
"""
from __future__ import annotations

import json
import subprocess
import threading
import uuid


class SandboxError(RuntimeError):
    """Sandbox infrastructure failed; do not turn this into a quality verdict."""


def _capture_bounded(command, source: bytes, timeout: int, limit: int):
    """Drain pipes continuously, retaining only a bounded tail of each stream."""
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    tails = [bytearray(), bytearray()]

    def drain(stream, tail):
        try:
            while chunk := stream.read(8192):
                tail.extend(chunk)
                del tail[:-limit]
        finally:
            stream.close()

    def feed():
        try:
            proc.stdin.write(source)
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        finally:
            proc.stdin.close()

    threads = [threading.Thread(target=drain, args=(stream, tail), daemon=True)
               for stream, tail in zip((proc.stdout, proc.stderr), tails)]
    threads.append(threading.Thread(target=feed, daemon=True))
    for thread in threads:
        thread.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
        for thread in threads:
            thread.join(timeout=5)
    return proc.returncode, *(bytes(t).decode("utf-8", errors="replace") for t in tails), timed_out


def run_docker(code: str, cfg):
    from sft_pipeline.filters.code_verifier import ExecutionResult

    name = "sft-sandbox-" + uuid.uuid4().hex
    command = [
        "docker", "create", "--name", name, "--pull=never", "--interactive", "--init",
        "--network=none", "--ipc=none", "--read-only", "--cap-drop=ALL",
        "--security-opt=no-new-privileges", "--user=65534:65534",
        "--pids-limit", str(cfg.docker_pids_limit),
        "--memory", f"{cfg.docker_memory_mb}m", "--memory-swap", f"{cfg.docker_memory_mb}m",
        "--cpus=1", "--ulimit=nofile=64:64", "--ulimit=core=0:0",
        "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777",
        "--workdir=/tmp", "--log-driver=none", "--env=HOME=/tmp",
        "--env=PYTHONIOENCODING=utf-8", "--entrypoint=/usr/bin/timeout",
        cfg.docker_image, "--signal=KILL", str(cfg.timeout_seconds + 5),
        "python", "-I", "-u", "-c",
        "import sys; exec(compile(sys.stdin.read(), '<response>', 'exec'))",
    ]
    created = False
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise SandboxError(f"Cannot create Docker sandbox: {result.stderr[-1000:]}")
        created = True
        _, stdout, stderr, timed_out = _capture_bounded(
            ["docker", "start", "--attach", "--interactive", name], code.encode(),
            cfg.timeout_seconds, cfg.max_output_bytes,
        )
        if timed_out:
            return ExecutionResult(False, stdout, stderr, timed_out=True)
        inspected = subprocess.run(["docker", "inspect", "--format", "{{json .State}}", name],
                                   capture_output=True, text=True, timeout=10, check=True)
        state = json.loads(inspected.stdout)
        if state.get("Error") or state.get("Status") != "exited":
            raise SandboxError(f"Docker sandbox did not execute normally: {state.get('Error')}")
        if state.get("OOMKilled"):
            return ExecutionResult(False, stdout, "Sandbox memory limit exceeded")
        return ExecutionResult(state["ExitCode"] == 0, stdout, stderr)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise SandboxError(f"Docker sandbox unavailable: {exc}") from exc
    finally:
        # Always address only our unique container. Also handles a create timeout
        # after the daemon created it but before the CLI returned its ID.
        try:
            removed = subprocess.run(["docker", "rm", "--force", name], capture_output=True,
                                     text=True, timeout=10)
            if removed.returncode and (created or "No such container" not in removed.stderr):
                raise SandboxError(f"Could not remove sandbox {name}: {removed.stderr[-500:]}")
        except (OSError, subprocess.SubprocessError) as exc:
            raise SandboxError(f"Could not clean up sandbox {name}: {exc}") from exc
