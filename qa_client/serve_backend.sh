#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${QA_BACKEND_PORT:-8000}"
PYTHON="${QA_BACKEND_PYTHON:-/Users/stanliu/bjj-proj/whole-video-analysis/venv/bin/python}"

cd "$ROOT"
echo "QA backend on :${PORT} — no Cassandra/GPU needed; serve the page via qa_client/serve.sh on :8765"
exec "$PYTHON" -m uvicorn service.qa_app:app --host 127.0.0.1 --port "$PORT"
