#!/usr/bin/env bash
# Deploy sam3_scout package to gx10 (code only — not full worktree).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GX10_HOST="${GX10_HOST:-gx10}"
REMOTE_DIR="${GX10_SCOUT_DIR:-/home/sliu1143/sam3_scout/code/sam3_scout}"
REMOTE_PARENT="$(dirname "$REMOTE_DIR")"

ssh "$GX10_HOST" "mkdir -p '$REMOTE_PARENT'"
rsync -avP --delete \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache/' \
  "$SCRIPT_DIR"/ "${GX10_HOST}:${REMOTE_DIR}/"

echo "Deployed to ${GX10_HOST}:${REMOTE_DIR}"
echo "Next: dry-run one short clip with the gx10 SAM3 venv (see README / dry_run_one_clip.sh)."
