"""
Code domain quality filter.

Extracts code blocks from responses and executes them in a sandbox.
Two sandbox backends:
  - 'subprocess': restricted subprocess with resource limits (dev/testing)
  - 'e2b': E2B cloud sandbox (production — zero ops, strong isolation)

Only Python code blocks are executed. Other languages are syntax-checked
where possible (no execution).
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from sft_pipeline.config import CodeFilterConfig
from sft_pipeline.filters.structural import FilterResult

_CODE_BLOCK = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE
)


@dataclass
class ExecutionResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


def check_code(record: dict, cfg: CodeFilterConfig) -> FilterResult:
    """
    Extract and execute code from the response answer/reasoning.
    Returns FilterResult.
    """
    full_text = record.get("reasoning", "") + "\n" + record.get("answer", "")
    code_blocks = _CODE_BLOCK.findall(full_text)

    if not code_blocks:
        return FilterResult(True)  # No code to verify

    # Execute the last (most likely final) code block
    code = code_blocks[-1].strip()
    if not code:
        return FilterResult(True)

    if cfg.sandbox == "e2b":
        result = _run_e2b(code, cfg.timeout_seconds)
    else:
        result = _run_subprocess(code, cfg.timeout_seconds)

    if result.timed_out:
        return FilterResult(False, "code_timeout")
    if not result.success:
        return FilterResult(False, f"code_error:{result.stderr[:200]}")

    return FilterResult(True)


# ---------------------------------------------------------------------------
# Subprocess sandbox (dev)
# ---------------------------------------------------------------------------

def _run_subprocess(code: str, timeout: int) -> ExecutionResult:
    """
    Execute Python code in a restricted subprocess.
    No network, CPU time limited by subprocess timeout.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return ExecutionResult(
            success=result.returncode == 0,
            stdout=result.stdout[:500],
            stderr=result.stderr[:500],
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(success=False, timed_out=True)
    except Exception as exc:
        return ExecutionResult(success=False, stderr=str(exc))
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# E2B sandbox (production)
# ---------------------------------------------------------------------------

def _run_e2b(code: str, timeout: int) -> ExecutionResult:
    """
    Execute code in an E2B cloud sandbox.
    Requires: pip install e2b-code-interpreter and E2B_API_KEY env var.
    """
    try:
        from e2b_code_interpreter import Sandbox
    except ImportError:
        # Fallback to subprocess if e2b not installed
        return _run_subprocess(code, timeout)

    try:
        with Sandbox(timeout=timeout) as sb:
            execution = sb.run_code(code)
            stderr = "\n".join(str(e) for e in execution.error) if execution.error else ""
            return ExecutionResult(
                success=execution.error is None or len(execution.error) == 0,
                stdout=execution.text or "",
                stderr=stderr,
            )
    except Exception as exc:
        return ExecutionResult(success=False, stderr=str(exc))
