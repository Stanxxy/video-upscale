#!/usr/bin/env bash
# Deploy the ADCC eval cost-control dashboard (and the underlying
# chunk-segment-tags capture engine it wraps) from the Mac worktree to
# bjj-server. Mac is the source of truth (feature/qa-vlm-studio worktree,
# uncommitted) — this script only rsyncs, it never commits/pushes anything.
#
# Mirrors the EXACT minimal file set the Operator's original feasibility-spike
# port already established on the server (analyzer.py/pipeline.py/utils.py +
# service/{config.py,pipelines/,routes/qa_vlm.py} + eval/ — no service/worker,
# no tracking_pipeline, no bjj_analysis_taxonomy.md: none of those are on the
# chunk-segment-tags import path at all — see
# service/pipelines/executors.py's module docstring / CALL_BUDGET_ESTIMATE.md).
#
# CRITICAL — two server-only shims must survive every redeploy UNTOUCHED:
#   service/routes/__init__.py  (emptied barrel — see its own docstring)
#   service/routes/state.py     (reduced to `_config = ServiceConfig()`)
# This script therefore does each path as its OWN narrow rsync invocation
# (never a whole-directory sync of service/routes/) — it copies ONLY
# service/routes/qa_vlm.py there, and never runs rsync --delete against a
# directory that contains the shims, so there is no filter-rule ordering to
# get wrong and no way for a stale/real Mac copy of those two files to ever
# land on the server.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAC_REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"   # .../whole-video-analysis-wt-vlmstudio
SERVER_HOST="${VLM_EVAL_SERVER_HOST:-bjj-server}"
SERVER_CODE_DIR="${VLM_EVAL_SERVER_CODE_DIR:-vlm-eval/code}"  # relative to $HOME on the server

echo "Deploying ADCC eval dashboard: ${MAC_REPO_ROOT} -> ${SERVER_HOST}:~/${SERVER_CODE_DIR}"

ssh "${SERVER_HOST}" "mkdir -p ~/${SERVER_CODE_DIR}/service/pipelines ~/${SERVER_CODE_DIR}/service/routes ~/${SERVER_CODE_DIR}/eval"

# --- Top-level single files ------------------------------------------------
rsync -avz \
  "${MAC_REPO_ROOT}/analyzer.py" \
  "${MAC_REPO_ROOT}/pipeline.py" \
  "${MAC_REPO_ROOT}/utils.py" \
  "${MAC_REPO_ROOT}/bjj_taxonomy_simplified.md" \
  "${SERVER_HOST}:~/${SERVER_CODE_DIR}/"

# --- service/ single files --------------------------------------------------
rsync -avz \
  "${MAC_REPO_ROOT}/service/__init__.py" \
  "${MAC_REPO_ROOT}/service/config.py" \
  "${SERVER_HOST}:~/${SERVER_CODE_DIR}/service/"

# --- service/pipelines/** (safe to fully sync + delete stale files — no shims here) ---
rsync -avz --delete \
  --exclude='__pycache__' \
  "${MAC_REPO_ROOT}/service/pipelines/" \
  "${SERVER_HOST}:~/${SERVER_CODE_DIR}/service/pipelines/"

# --- service/routes/qa_vlm.py ONLY — never the directory, never __init__.py/state.py ---
rsync -avz \
  "${MAC_REPO_ROOT}/service/routes/qa_vlm.py" \
  "${SERVER_HOST}:~/${SERVER_CODE_DIR}/service/routes/qa_vlm.py"

# --- eval/** (safe to fully sync + delete stale files — no shims here either) ---
rsync -avz --delete \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='predictions/' \
  --exclude='predictions_smoke/' \
  "${MAC_REPO_ROOT}/eval/" \
  "${SERVER_HOST}:~/${SERVER_CODE_DIR}/eval/"

echo
echo "Deploy complete. Verifying server-only shims survived untouched:"
ssh "${SERVER_HOST}" "grep -l 'SERVER-SPIKE SHIM' ~/${SERVER_CODE_DIR}/service/routes/__init__.py ~/${SERVER_CODE_DIR}/service/routes/state.py"
echo "OK — shims intact."
