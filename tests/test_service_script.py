"""Regression tests for the `service.sh` lifecycle wrapper."""

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


def _spawn_stubborn_process() -> subprocess.Popen[str]:
    """Start a process that ignores SIGTERM so stop must escalate."""
    script = (
        "import signal\n"
        "import time\n"
        "signal.signal(signal.SIGTERM, lambda *_: None)\n"
        "while True:\n"
        "    time.sleep(0.1)\n"
    )
    return subprocess.Popen([sys.executable, "-c", script], text=True)


def _wait_for_exit(process: subprocess.Popen[str], timeout_seconds: float) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if process.poll() is not None:
            return True
        time.sleep(0.1)
    return process.poll() is not None


def test_stop_kills_stubborn_process_and_removes_pid_file(tmp_path: Path) -> None:
    """`service.sh stop` should SIGKILL if graceful shutdown times out."""
    source_script = Path(__file__).resolve().parents[1] / "service.sh"
    script_copy = tmp_path / "service.sh"
    shutil.copy2(source_script, script_copy)
    script_copy.chmod(0o755)

    process = _spawn_stubborn_process()
    pid_file = tmp_path / ".service.pid"
    pid_file.write_text(f"{process.pid}\n", encoding="utf-8")

    try:
        completed = subprocess.run(
            ["bash", str(script_copy), "stop"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=15,
            env={**os.environ, "BJJ_STOP_TIMEOUT_SECONDS": "1"},
        )

        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert not pid_file.exists()
        assert _wait_for_exit(process, timeout_seconds=2.0)
    finally:
        if process.poll() is None:
            os.kill(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
