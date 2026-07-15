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

import os
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path

from sft_pipeline.config import CodeFilterConfig
from sft_pipeline.filters.structural import FilterResult

# The language tag is REQUIRED. With an optional tag, any fenced block (yaml,
# robots.txt, markdown prose, sample I/O …) gets executed as Python and fails
# with a spurious SyntaxError — measured at 100% false rejections on real
# Stage 5 data.
_CODE_BLOCK = re.compile(
    r"```(?:python|py)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE
)

# Runtime errors caused by the sandbox environment, not by wrong code.
# Rejecting on these punishes correct responses (measured on real data):
#   ModuleNotFound/Import — sandbox has no third-party deps
#   EOFError              — code calls input(); sandbox has no stdin
#   FileNotFound/Permission/network — external resources unavailable
#   NameError             — illustrative fragment using names defined in prose
#   Unicode*Error         — console-encoding artifact (e.g. emoji on cp1252)
_ENV_ERROR_MARKERS = (
    "ModuleNotFoundError",
    "ImportError",
    "EOFError",
    "FileNotFoundError",
    "PermissionError",
    "ConnectionError",
    "SSLError",
    "Max retries exceeded",
    "gaierror",
    "URLError",
    "HTTPError",
    "TimeoutError",
    "NameError",
    "KeyboardInterrupt",
    "MemoryError",
    "UnicodeEncodeError",
    "UnicodeDecodeError",
)

# Code that is *designed* to run indefinitely or wait on external resources
# (GUIs, servers, polling loops, model/dataset downloads). A timeout on these
# is expected behaviour, not an infinite-loop bug.
_LONG_RUNNING_MARKERS = (
    "input(",
    "mainloop(",
    "plt.show(",
    "app.run(",
    "serve_forever",
    "run_forever",
    "time.sleep(",
    "import turtle",
    "import tkinter",
    "import pygame",
    "from_pretrained(",
)


@dataclass
class ExecutionResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


def check_code(record: dict, cfg: CodeFilterConfig) -> FilterResult:
    """
    Extract and execute the final Python code block from the response.

    Only genuine code defects reject; environment limitations (missing deps,
    no stdin, no network) and non-executable illustration (REPL transcripts,
    fragments) pass with an informational reason.
    """
    full_text = record.get("reasoning", "") + "\n" + record.get("answer", "")
    code_blocks = _CODE_BLOCK.findall(full_text)

    if not code_blocks:
        return FilterResult(True)  # No code to verify

    # Execute the last (most likely final) code block
    code = code_blocks[-1].strip()
    if not code:
        return FilterResult(True)

    # REPL transcripts (">>> ..." lines) are illustration, not programs.
    if ">>>" in code:
        return FilterResult(True, "code_repl_transcript")

    # Notebook magics / shell commands inside a python fence ("!pip install",
    # "%matplotlib") — sloppy fencing, but not incorrect code.
    if re.search(r"^\s*[!%]\w", code, re.MULTILINE):
        return FilterResult(True, "code_notebook_shell")

    # Syntax gate. A fenced fragment (e.g. a method body with a bare `return`)
    # is legitimate illustration — detect it by checking whether the block
    # compiles once dedented or wrapped in a function. Only reject when the
    # block is not valid Python in any of those forms.
    try:
        _compile_quiet(code)
    except SyntaxError:
        dedented = textwrap.dedent(code)
        wrapped = "def _sft_wrap_():\n" + textwrap.indent(dedented, "    ")
        for candidate in (dedented, wrapped):
            try:
                _compile_quiet(candidate)
                return FilterResult(True, "code_fragment")
            except SyntaxError:
                continue
        return FilterResult(False, "code_syntax_error")
    except (ValueError, MemoryError, RecursionError):
        return FilterResult(True, "code_compile_uncertain")

    if cfg.sandbox == "e2b":
        result = _run_e2b(code, cfg.timeout_seconds)
    else:
        result = _run_subprocess(code, cfg.timeout_seconds)

    if result.timed_out:
        if any(m in code for m in _LONG_RUNNING_MARKERS):
            return FilterResult(True, "code_long_running")
        return FilterResult(False, "code_timeout")

    if not result.success:
        last_line = _last_stderr_line(result.stderr)
        if any(m in result.stderr for m in _ENV_ERROR_MARKERS):
            return FilterResult(True, f"code_env_uncertain:{last_line[:80]}")
        return FilterResult(False, f"code_error:{last_line[:200]}")

    return FilterResult(True)


def _compile_quiet(src: str) -> None:
    """``compile()`` with ``SyntaxWarning`` silenced.

    Arbitrary response code routinely contains invalid escape sequences (an
    un-raw regex string like ``"\\d"``, a Windows path), which make ``compile()``
    emit a ``SyntaxWarning`` to stderr. At Stage 6 scale these flood the logs
    and say nothing about response quality — the syntax *gate* only cares about
    ``SyntaxError``. Suppress the warnings; let ``SyntaxError`` propagate.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        compile(src, "<response>", "exec")


def _last_stderr_line(stderr: str) -> str:
    """The final non-empty stderr line — for Python tracebacks, the actual error."""
    lines = [ln.strip() for ln in (stderr or "").strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


# ---------------------------------------------------------------------------
# Subprocess sandbox (dev)
# ---------------------------------------------------------------------------

def _run_subprocess(code: str, timeout: int) -> ExecutionResult:
    """
    Execute Python code in a restricted subprocess.
    No network, CPU time limited by subprocess timeout.

    Runs inside a throwaway temp directory: dataset code frequently writes
    files (csv/json/zip demos), and with the caller's cwd those would litter
    the repository. The directory is removed afterwards, along with anything
    the code created in it.

    The snippet runs in its **own process group** (``start_new_session=True``)
    so that on timeout we can kill the entire tree, not just the direct child.
    This matters: ``subprocess.run(timeout=)`` only kills the immediate child,
    but a snippet that backgrounds a process (``multiprocessing``, ``Popen``,
    ``os.system('… &')``) leaves a grandchild holding the stdout pipe open, and
    ``communicate()`` then blocks waiting for pipe EOF *long past the timeout* —
    freezing the whole filter run on a single record (observed as a multi-minute
    stall in Stage 6). Killing the process group closes those fds so we return
    within the timeout.
    """
    import shutil

    workdir = tempfile.mkdtemp(prefix="sft_code_sandbox_")
    tmp_path = Path(workdir) / "snippet.py"
    tmp_path.write_text(code, encoding="utf-8")

    # No stdin: input() raises EOFError immediately instead of blocking until
    # the timeout. UTF-8 I/O: print(emoji) must not die with UnicodeEncodeError
    # on platforms whose console defaults to a legacy codepage (Windows cp1252).
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = None
    try:
        proc = subprocess.Popen(
            [sys.executable, str(tmp_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=workdir,
            # POSIX: new session → child is its own process-group leader, so
            # os.killpg reaches every descendant. Ignored on Windows (we fall
            # back to proc.kill(), which on Windows terminates child trees).
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return ExecutionResult(
                success=proc.returncode == 0,
                stdout=(stdout or "")[:500],
                # Keep the TAIL: for tracebacks the exception type is on the last line.
                stderr=(stderr or "")[-500:],
            )
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            # Drain the pipes now that the tree is dead. Bounded by a short
            # secondary timeout so a stuck fd can never hang us indefinitely.
            try:
                proc.communicate(timeout=5)
            except (subprocess.TimeoutExpired, Exception):
                pass
            return ExecutionResult(success=False, timed_out=True)
    except Exception as exc:
        if proc is not None:
            _kill_process_tree(proc)
        return ExecutionResult(success=False, stderr=str(exc))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill *proc* and every descendant it spawned.

    On POSIX the child leads its own process group (see ``start_new_session``),
    so ``killpg`` takes out the whole tree in one signal. On Windows, and if the
    group lookup fails (child already gone), fall back to killing the child.
    """
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except Exception:
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
