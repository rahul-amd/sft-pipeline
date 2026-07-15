"""Tests for the code verifier filter (subprocess sandbox)."""
from __future__ import annotations

import os
import time

import pytest

from sft_pipeline.config import CodeFilterConfig
from sft_pipeline.filters.code_verifier import check_code

CFG = CodeFilterConfig(enabled=True, sandbox="subprocess", timeout_seconds=5, domains=["code"])


def _rec(answer: str, reasoning: str = "Some reasoning about the problem.") -> dict:
    return {"reasoning": reasoning, "answer": answer}


def test_no_code_passes():
    result = check_code(_rec("Just prose, no code blocks."), CFG)
    assert result.passed


def test_untagged_fence_not_executed():
    # yaml/robots.txt/markdown inside a bare ``` fence must NOT run as Python
    answer = "Here is the config:\n```\nUser-agent: *\nDisallow: /\n```"
    result = check_code(_rec(answer), CFG)
    assert result.passed
    assert result.reason == ""


def test_valid_python_passes():
    answer = "```python\nprint(sum(range(10)))\n```"
    result = check_code(_rec(answer), CFG)
    assert result.passed


def test_repl_transcript_passes():
    answer = "```python\n>>> 1 + 1\n2\n```"
    result = check_code(_rec(answer), CFG)
    assert result.passed
    assert result.reason == "code_repl_transcript"


def test_notebook_shell_passes():
    answer = "```python\n!pip install requests\nimport requests\n```"
    result = check_code(_rec(answer), CFG)
    assert result.passed
    assert result.reason == "code_notebook_shell"


def test_fragment_with_bare_return_passes():
    # A method body shown for illustration — valid once wrapped in a function
    answer = "```python\nresult = compute(x)\nreturn result\n```"
    result = check_code(_rec(answer), CFG)
    assert result.passed
    assert result.reason == "code_fragment"


def test_true_syntax_error_rejects():
    answer = "```python\ndef broken(:\n    pass pass\n```"
    result = check_code(_rec(answer), CFG)
    assert not result.passed
    assert result.reason == "code_syntax_error"


def test_missing_dependency_is_uncertain():
    answer = "```python\nimport nonexistent_module_xyz_42\n```"
    result = check_code(_rec(answer), CFG)
    assert result.passed
    assert result.reason.startswith("code_env_uncertain")


def test_input_call_is_uncertain():
    # No stdin in the sandbox → EOFError → environment, not a code bug
    answer = "```python\nname = input('Your name: ')\nprint(name)\n```"
    result = check_code(_rec(answer), CFG)
    assert result.passed
    assert result.reason.startswith("code_env_uncertain")


def test_unicode_print_passes():
    answer = "```python\nprint('check: ✅')\n```"
    result = check_code(_rec(answer), CFG)
    assert result.passed


def test_genuine_runtime_error_rejects():
    answer = "```python\nx = [1, 2, 3]\nprint(x[10])\n```"
    result = check_code(_rec(answer), CFG)
    assert not result.passed
    assert result.reason.startswith("code_error")
    assert "IndexError" in result.reason


def test_failing_assertion_rejects():
    answer = "```python\ndef add(a, b):\n    return a - b\nassert add(2, 2) == 4\n```"
    result = check_code(_rec(answer), CFG)
    assert not result.passed
    assert result.reason.startswith("code_error")


@pytest.mark.slow
def test_infinite_loop_rejects():
    answer = "```python\nwhile True:\n    x = 1\n```"
    cfg = CodeFilterConfig(enabled=True, sandbox="subprocess", timeout_seconds=3, domains=["code"])
    result = check_code(_rec(answer), cfg)
    assert not result.passed
    assert result.reason == "code_timeout"


@pytest.mark.slow
def test_long_running_marker_timeout_is_uncertain():
    answer = "```python\nimport time\ntime.sleep(30)\n```"
    cfg = CodeFilterConfig(enabled=True, sandbox="subprocess", timeout_seconds=3, domains=["code"])
    result = check_code(_rec(answer), cfg)
    assert result.passed
    assert result.reason == "code_long_running"


@pytest.mark.slow
@pytest.mark.skipif(os.name != "posix", reason="process-group kill is POSIX-specific")
def test_backgrounded_child_does_not_hang_past_timeout():
    # The direct child exits instantly but backgrounds a grandchild that
    # inherits the stdout pipe and blocks forever. Without process-group kill,
    # communicate() waits for pipe EOF long past the timeout — the multi-minute
    # Stage 6 freeze. The sandbox must still return within the timeout window.
    answer = (
        "```python\n"
        "import subprocess, sys\n"
        "subprocess.Popen([sys.executable, '-c', 'import signal; signal.pause()'])\n"
        "print('parent exits now')\n"
        "```"
    )
    cfg = CodeFilterConfig(enabled=True, sandbox="subprocess", timeout_seconds=3, domains=["code"])
    start = time.monotonic()
    result = check_code(_rec(answer), cfg)
    elapsed = time.monotonic() - start
    assert not result.passed
    assert result.reason == "code_timeout"
    # timeout (3s) + bounded drain (5s) + slack — must not hang indefinitely.
    assert elapsed < 20, f"sandbox hung for {elapsed:.1f}s past its timeout"


def test_last_block_is_executed():
    # Earlier broken block is illustration; the final corrected block is what counts
    answer = (
        "First attempt:\n```python\nraise ValueError('old buggy version')\n```\n"
        "Corrected:\n```python\nprint('ok')\n```"
    )
    result = check_code(_rec(answer), CFG)
    assert result.passed
