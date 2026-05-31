#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${QA_CLIENT_PORT:-8765}"
cd "$ROOT/qa_client"
echo "QA client: http://127.0.0.1:${PORT}/"
echo "Point Service URL at http://localhost:8000 (vision engine must be running)."
exec python3 -m http.server "$PORT"
