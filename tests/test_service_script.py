"""Regression tests for the `service.sh` lifecycle wrapper."""

import os
import shutil
import signal
import socket
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


def _wait_for_port(port: int, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        with socket.socket() as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise AssertionError(f"Timed out waiting for port {port} to accept connections")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        return int(sock.getsockname()[1])


def _spawn_uvicorn_app(tmp_path: Path, port: int) -> subprocess.Popen[str]:
    app_file = tmp_path / "dummy_app.py"
    app_file.write_text(
        "async def app(scope, receive, send):\n"
        "    assert scope['type'] == 'http'\n"
        "    await send({'type': 'http.response.start', 'status': 200, 'headers': []})\n"
        "    await send({'type': 'http.response.body', 'body': b'ok'})\n",
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "dummy_app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        text=True,
    )
    _wait_for_port(port, timeout_seconds=5.0)
    return process


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


def test_stop_finds_running_service_when_pid_file_is_missing(tmp_path: Path) -> None:
    """`service.sh stop` should recover from a missing pid file."""
    source_script = Path(__file__).resolve().parents[1] / "service.sh"
    script_copy = tmp_path / "service.sh"
    shutil.copy2(source_script, script_copy)
    script_copy.chmod(0o755)

    port = _free_port()
    process = _spawn_uvicorn_app(tmp_path, port)

    try:
        completed = subprocess.run(
            ["bash", str(script_copy), "stop"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=10,
            env={
                **os.environ,
                "BJJ_HOST": "127.0.0.1",
                "BJJ_PORT": str(port),
                "BJJ_APP_MODULE": "dummy_app:app",
                "BJJ_STOP_TIMEOUT_SECONDS": "1",
            },
        )

        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert _wait_for_exit(process, timeout_seconds=2.0)
    finally:
        if process.poll() is None:
            os.kill(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def test_start_recovers_existing_service_when_pid_file_is_missing(tmp_path: Path) -> None:
    """`service.sh start` should reuse a live matching service instead of double-starting."""
    source_script = Path(__file__).resolve().parents[1] / "service.sh"
    script_copy = tmp_path / "service.sh"
    shutil.copy2(source_script, script_copy)
    script_copy.chmod(0o755)

    port = _free_port()
    process = _spawn_uvicorn_app(tmp_path, port)
    pid_file = tmp_path / ".service.pid"

    try:
        completed = subprocess.run(
            ["bash", str(script_copy), "start"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=10,
            env={
                **os.environ,
                "BJJ_HOST": "127.0.0.1",
                "BJJ_PORT": str(port),
                "BJJ_APP_MODULE": "dummy_app:app",
            },
        )

        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert pid_file.read_text(encoding="utf-8").strip() == str(process.pid)
        assert process.poll() is None
        assert "already running" in completed.stdout.lower()
    finally:
        if process.poll() is None:
            os.kill(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
