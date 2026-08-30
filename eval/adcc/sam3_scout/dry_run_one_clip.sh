#!/usr/bin/env bash
# gx10 dry-run: one short H.264 clip through chunked SAM3 at concurrency 1.
#
# Ops note: if CUDA OOM on model load, temporarily stop competing GPU users:
#   docker stop vllm-spark vllm-spark-27b
# …run this script… then:
#   docker start vllm-spark vllm-spark-27b
set -euo pipefail

SCOUT_ROOT="${SCOUT_ROOT:-/home/sliu1143/sam3_scout}"
CODE_DIR="${SCOUT_ROOT}/code"
CLIP="${1:-}"
OUT="${SCOUT_ROOT}/runs/dry_run"
VENV="${SAM3_VENV:-/home/sliu1143/Documents/vision/venv}"
export SAM3_TEXT_PROMPT="${SAM3_TEXT_PROMPT:-person}"

if [[ -z "$CLIP" ]]; then
  FIRST="$(ls -1 "${SCOUT_ROOT}/clips/"*.mp4 2>/dev/null | head -1 || true)"
  if [[ -n "${FIRST}" ]]; then
    CLIP="$FIRST"
  else
    mkdir -p "${SCOUT_ROOT}/clips"
    CLIP="${SCOUT_ROOT}/clips/dry_run_synth.mp4"
    ffmpeg -y -f lavfi -i testsrc=size=640x360:rate=30 -t 8 \
      -c:v libx264 -pix_fmt yuv420p -an "$CLIP"
  fi
fi
export SCOUT_DRY_CLIP="$CLIP"
export SCOUT_DRY_OUT="$OUT"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
mkdir -p "$OUT"
if [[ ! -d "${CODE_DIR}/eval/adcc/sam3_scout" ]]; then
  mkdir -p "${CODE_DIR}/eval/adcc"
  ln -sfn "${SCOUT_ROOT}/code/sam3_scout" "${CODE_DIR}/eval/adcc/sam3_scout"
  printf '' > "${CODE_DIR}/eval/__init__.py"
  printf '' > "${CODE_DIR}/eval/adcc/__init__.py"
fi
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

python - <<'PY'
import json, os
from pathlib import Path
from eval.adcc.sam3_scout.sam3_chunk_runner import run_chunked_sam3, probe_duration_seconds
from eval.adcc.sam3_scout.advisory_metrics import compute_advisory_metrics

clip = Path(os.environ["SCOUT_DRY_CLIP"])
out = Path(os.environ["SCOUT_DRY_OUT"])
prompt = os.environ.get("SAM3_TEXT_PROMPT", "person")
print("dry-run clip:", clip, "duration=", probe_duration_seconds(str(clip)))
manifest = run_chunked_sam3(
    str(clip), out, clip_id="dry_run",
    text_prompt=prompt,
    persist_masks=True,
    write_preview=True,
)
adv = compute_advisory_metrics(manifest.tracks_jsonl)
print(json.dumps({"manifest": manifest.__dict__, "advisory": adv}, indent=2, default=str))
PY

echo "Dry-run complete under $OUT"
