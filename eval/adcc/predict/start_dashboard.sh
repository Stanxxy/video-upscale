#!/usr/bin/env bash
# Start the ADCC eval cost-control dashboard on bjj-server, bound to
# 0.0.0.0:8321 (verified free — reachable at http://100.79.167.101:8321 over
# Tailscale). Run this ON the server (deploy_to_bjj_server.sh puts the code
# there first). Loads ~/vlm-eval/.env for GEMINI_API_KEY (server-side only —
# dashboard_app.py never sends it to the browser or logs it).
#
# Usage (foreground, for interactive debugging):
#   bash eval/adcc/predict/start_dashboard.sh
#
# Usage (background, survives the SSH session — see also the systemd user
# unit at eval/adcc/predict/bjj-adcc-dashboard.service for a supervised,
# auto-restarting alternative):
#   nohup bash eval/adcc/predict/start_dashboard.sh > ~/vlm-eval/dashboard.log 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"   # ~/vlm-eval/code
VENV_PYTHON="${VLM_EVAL_VENV_PYTHON:-$HOME/vlm-eval/venv/bin/python}"
PORT="${VLM_EVAL_DASHBOARD_PORT:-8321}"

cd "$CODE_DIR"
echo "ADCC eval cost-control dashboard starting on 0.0.0.0:${PORT}"
echo "Tailscale URL: http://100.79.167.101:${PORT}"
exec "$VENV_PYTHON" -m uvicorn eval.adcc.predict.dashboard_app:app --host 0.0.0.0 --port "$PORT"
