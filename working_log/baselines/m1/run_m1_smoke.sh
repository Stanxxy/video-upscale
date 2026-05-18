#!/usr/bin/env bash
# M1 driver -- DGX Spark smoke after S6+S2+S1+S3.
# Launches the FastAPI service, submits a /track job with pre-detected boxes
# on the same 30-sec segment as M0, polls until terminal, then dumps
# per-stage timing + worker log into the artifact dir.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/bjj/whole-video-analysis}"
VENV="${VENV:-$HOME/bjj/.venv-spark}"
PORT="${BJJ_PORT:-8000}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$HOME/bjj/m1-artifacts/smoke}"
S3_BUCKET="${S3_BUCKET:-bjj-video-analysis}"
S3_KEY="${S3_KEY:-aHR0cHM6Ly93d3cueW91dHViZS5jb20vd2F0Y2g_dj1DOEZrVWtaSGxGYw==.mp4}"
BOXES_JSON="${BOXES_JSON:-$PROJECT_DIR/working_log/baselines/m0/boxes.json}"
GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL:-15}"
START_TIME="0:00"
END_TIME="0:30"
# M1 should land in <15 min; cap at 30 to leave headroom for cold model load.
WALL_BUDGET_SEC=1800

mkdir -p "$ARTIFACT_DIR"
echo "[m1] artifact_dir=$ARTIFACT_DIR" | tee -a "$ARTIFACT_DIR/driver.log"

# Source env so BJJ_GEMINI_API_KEY etc. flow through
set -a
. "$PROJECT_DIR/.env"
set +a

# Match M0 env overrides for like-for-like comparison.
export BJJ_UPSCALE_HEARTBEAT_INTERVAL_SEC=5
export BJJ_MAX_CONCURRENT_JOBS=1
export BJJ_TEMP_DIR="$ARTIFACT_DIR/temp"
export BJJ_MODEL_PATH="${BJJ_MODEL_PATH:-RealESRGAN_x4plus.pth}"
mkdir -p "$BJJ_TEMP_DIR"

cd "$PROJECT_DIR"
source "$VENV/bin/activate"

BOX_A=$(python -c "import json; d=json.load(open('$BOXES_JSON')); print(json.dumps(d['box_a']))")
BOX_B=$(python -c "import json; d=json.load(open('$BOXES_JSON')); print(json.dumps(d['box_b']))")
echo "[m1] box_a=$BOX_A" | tee -a "$ARTIFACT_DIR/driver.log"
echo "[m1] box_b=$BOX_B" | tee -a "$ARTIFACT_DIR/driver.log"

SVC_LOG="$ARTIFACT_DIR/service.log"
echo "[m1] starting uvicorn on port $PORT, log=$SVC_LOG" | tee -a "$ARTIFACT_DIR/driver.log"
nohup python -m uvicorn service.app:app --host 0.0.0.0 --port "$PORT" --log-level info > "$SVC_LOG" 2>&1 &
SVC_PID=$!
echo "$SVC_PID" > "$ARTIFACT_DIR/service.pid"
echo "[m1] uvicorn pid=$SVC_PID" | tee -a "$ARTIFACT_DIR/driver.log"

cleanup() {
    echo "[m1] cleanup: killing service pid=$SVC_PID and gpu poller pid=${GPU_PID:-NA}" | tee -a "$ARTIFACT_DIR/driver.log"
    kill "$SVC_PID" 2>/dev/null || true
    if [[ -n "${GPU_PID:-}" ]]; then kill "$GPU_PID" 2>/dev/null || true; fi
    sleep 2
    kill -9 "$SVC_PID" 2>/dev/null || true
    if [[ -n "${GPU_PID:-}" ]]; then kill -9 "$GPU_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

for i in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "[m1] service healthy after ${i}s" | tee -a "$ARTIFACT_DIR/driver.log"
        break
    fi
    sleep 1
done

if ! curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[m1] FATAL: service never came up" | tee -a "$ARTIFACT_DIR/driver.log"
    exit 2
fi

SAMPLES_CSV="$ARTIFACT_DIR/nvidia-smi-samples.csv"
echo "timestamp,gpu_util_pct,torch_alloc_mb,torch_reserved_mb,sys_mem_used_mb,sys_mem_total_mb" > "$SAMPLES_CSV"

cat > "$ARTIFACT_DIR/gpu_poller.py" <<'PYEOF'
import csv, time, sys, os
import subprocess
def smi_util():
    try:
        out = subprocess.check_output(
            ["nvidia-smi","--query-gpu=utilization.gpu","--format=csv,noheader,nounits"],
            text=True, timeout=5,
        ).strip()
        return out.split("\n")[0]
    except Exception:
        return ""
def torch_mem():
    try:
        import torch
        if torch.cuda.is_available():
            a = torch.cuda.memory_allocated() / 1024 / 1024
            r = torch.cuda.memory_reserved() / 1024 / 1024
            return f"{a:.1f}", f"{r:.1f}"
    except Exception:
        pass
    return "", ""
def sys_mem():
    try:
        out = subprocess.check_output(["free","-m"], text=True, timeout=5).splitlines()
        parts = out[1].split()
        return parts[2], parts[1]
    except Exception:
        return "", ""
INTERVAL = float(os.environ.get("GPU_SAMPLE_INTERVAL","15"))
OUT = sys.argv[1]
while True:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    u = smi_util()
    a, r = torch_mem()
    used, total = sys_mem()
    with open(OUT, "a") as f:
        f.write(f"{ts},{u},{a},{r},{used},{total}\n")
    time.sleep(INTERVAL)
PYEOF

GPU_SAMPLE_INTERVAL=$GPU_SAMPLE_INTERVAL nohup python "$ARTIFACT_DIR/gpu_poller.py" "$SAMPLES_CSV" > "$ARTIFACT_DIR/gpu_poller.log" 2>&1 &
GPU_PID=$!
echo "[m1] gpu poller pid=$GPU_PID" | tee -a "$ARTIFACT_DIR/driver.log"

REQ_JSON="$ARTIFACT_DIR/track_request.json"
python - "$BOXES_JSON" "$S3_BUCKET" "$S3_KEY" "$START_TIME" "$END_TIME" "$REQ_JSON" <<'PYEOF'
import json, sys
boxes_path, bucket, key, start_time, end_time, out = sys.argv[1:7]
d = json.load(open(boxes_path))
req = {
    "bucket": bucket,
    "key": key,
    "box_a": d["box_a"],
    "box_b": d["box_b"],
    "analyzer_mode": "single",
    "method": "esrgan",
    "sam2_model": "facebook/sam2.1-hiera-base-plus",
    "sampling_rate": 1,
    # Same M0 sentinel: bypass the mid-clip human-correction suspend so the
    # smoke runs unattended.
    "max_missing_frames": 999999,
}
if start_time: req["start_time"] = start_time
if end_time: req["end_time"] = end_time
json.dump(req, open(out, "w"), indent=2)
print(json.dumps(req, indent=2))
PYEOF
echo "[m1] track request written to $REQ_JSON" | tee -a "$ARTIFACT_DIR/driver.log"

WALL_START_EPOCH=$(date +%s)
echo "[m1] WALL_START_EPOCH=$WALL_START_EPOCH" | tee -a "$ARTIFACT_DIR/driver.log"

RESP_JSON="$ARTIFACT_DIR/track_response.json"
curl -fsS -X POST "http://127.0.0.1:$PORT/track" \
    -H 'Content-Type: application/json' \
    -d @"$REQ_JSON" > "$RESP_JSON"
JOB_ID=$(python -c "import json; print(json.load(open('$RESP_JSON'))['job_id'])")
echo "[m1] job_id=$JOB_ID" | tee -a "$ARTIFACT_DIR/driver.log"
echo "$JOB_ID" > "$ARTIFACT_DIR/job_id.txt"

LIFECYCLE_LOG="$ARTIFACT_DIR/lifecycle.jsonl"
> "$LIFECYCLE_LOG"
DEADLINE=$(( WALL_START_EPOCH + WALL_BUDGET_SEC ))
prev_state=""
while true; do
    NOW=$(date +%s)
    if [[ "$NOW" -ge "$DEADLINE" ]]; then
        echo "[m1] WALL BUDGET EXCEEDED (${WALL_BUDGET_SEC}s); aborting poll" | tee -a "$ARTIFACT_DIR/driver.log"
        break
    fi
    LIFE=$(curl -fsS "http://127.0.0.1:$PORT/job/$JOB_ID" 2>/dev/null || echo '{}')
    TS=$(date +%Y-%m-%dT%H:%M:%S%z)
    echo "{\"ts\":\"$TS\",\"life\":$LIFE}" >> "$LIFECYCLE_LOG"
    STATE=$(python -c "import json,sys; d=json.loads('''$LIFE'''); print(d.get('status',''))" 2>/dev/null || echo "")
    if [[ "$STATE" != "$prev_state" ]]; then
        echo "[m1] state -> $STATE @ $TS (job=$JOB_ID)" | tee -a "$ARTIFACT_DIR/driver.log"
        prev_state="$STATE"
    fi
    case "$STATE" in
        completed|failed|cancelled)
            echo "[m1] reached terminal state: $STATE" | tee -a "$ARTIFACT_DIR/driver.log"
            break
            ;;
    esac
    sleep 5
done

WALL_END_EPOCH=$(date +%s)
echo "[m1] WALL_END_EPOCH=$WALL_END_EPOCH" | tee -a "$ARTIFACT_DIR/driver.log"
echo "[m1] WALL_TOTAL_SEC=$(( WALL_END_EPOCH - WALL_START_EPOCH ))" | tee -a "$ARTIFACT_DIR/driver.log"

cleanup
trap - EXIT INT TERM
echo "[m1] done. artifacts in $ARTIFACT_DIR" | tee -a "$ARTIFACT_DIR/driver.log"
