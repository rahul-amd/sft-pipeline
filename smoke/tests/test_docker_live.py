"""Opt-in real isolation checks: SFT_TEST_DOCKER=1, pre-pull python:3.12-slim."""
import os
import subprocess
import time
import uuid

import pytest

from sft_pipeline.config import CodeFilterConfig
from sft_pipeline.filters.docker_sandbox import run_docker, SandboxError

pytestmark = pytest.mark.skipif(os.getenv('SFT_TEST_DOCKER') != '1',
                                reason='Set SFT_TEST_DOCKER=1 on a Docker worker')


def test_host_files_secrets_network_privileges_and_resources_are_isolated(tmp_path, monkeypatch):
    secret = tmp_path / 'host-only.txt'
    secret.write_text('host secret')
    monkeypatch.setenv('SFT_SANDBOX_SECRET', 'must-not-reach-snippet')
    code = f'''
import os, socket
from pathlib import Path
assert not Path({str(secret)!r}).exists()
assert 'SFT_SANDBOX_SECRET' not in os.environ
assert not Path('/var/run/docker.sock').exists()
assert os.getuid() == 65534
assert 'NoNewPrivs:\\t1' in Path('/proc/self/status').read_text()
assert 'CapEff:\\t0000000000000000' in Path('/proc/self/status').read_text()
assert 'Seccomp:\\t2' in Path('/proc/self/status').read_text()
assert int(Path('/sys/fs/cgroup/memory.max').read_text()) == 256 * 1024 * 1024
assert int(Path('/sys/fs/cgroup/pids.max').read_text()) == 32
quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
assert int(quota) == int(period)
try:
    Path('/usr/local/sandbox-write').write_text('bad')
except OSError:
    pass
else:
    raise AssertionError('root filesystem writable')
Path('/tmp/allowed').write_text('ok')
sock = socket.socket()
sock.settimeout(.2)
try:
    sock.connect(('1.1.1.1', 443))
except OSError:
    pass
else:
    raise AssertionError('outbound network reachable')
print('isolation checks passed')
'''
    result = run_docker(code, CodeFilterConfig(timeout_seconds=10))
    assert result.success, result.stderr
    assert 'isolation checks passed' in result.stdout
    assert secret.read_text() == 'host secret'


def test_stdout_and_stderr_are_bounded():
    result = run_docker("import sys; print('x' * 1000000); print('e' * 1000000, file=sys.stderr)",
                        CodeFilterConfig(timeout_seconds=10, max_output_bytes=1024))
    assert result.success, result.stderr
    assert len(result.stdout) <= 1024 and len(result.stderr) <= 1024


def test_timeout_removes_container_and_detached_descendants(monkeypatch):
    identity = uuid.uuid4()
    monkeypatch.setattr('sft_pipeline.filters.docker_sandbox.uuid.uuid4', lambda: identity)
    started = time.monotonic()
    result = run_docker("import subprocess,sys,time; subprocess.Popen([sys.executable, '-c', "
                        "'import time; time.sleep(120)'], start_new_session=True); time.sleep(120)",
                        CodeFilterConfig(timeout_seconds=2))
    assert result.timed_out
    assert time.monotonic() - started < 20
    status = subprocess.run(['docker', 'inspect', 'sft-sandbox-' + identity.hex], capture_output=True)
    assert status.returncode != 0


def test_memory_limit_rejects_run():
    result = run_docker('x = bytearray(1024 * 1024 * 1024)',
                        CodeFilterConfig(timeout_seconds=10, docker_memory_mb=64))
    assert not result.success


def test_missing_image_is_an_infrastructure_error():
    with pytest.raises(SandboxError, match='Cannot create'):
        run_docker('print(42)', CodeFilterConfig(docker_image='sft-missing-test-image:never'))
