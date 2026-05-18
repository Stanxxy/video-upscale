#!/usr/bin/env bash
# M5 smoke driver — measures standard mode (prop_stride=5) and fast mode (prop_stride=24).
# Assumes the service is already running on port 8000 with BJJ_STANDARD_PROP_STRIDE=5
# and BJJ_FAST_PROP_STRIDE=24 set in the environment.
# Usage:
#   MODE=standard bash run_m5_smoke.sh
#   MODE=fast bash run_m5_smoke.sh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/bjj/whole-video-analysis}"
PORT="${BJJ_PORT:-8000}"
MODE="${MODE:-standard}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$HOME/bjj/m5-artifacts/$MODE}"
S3_BUCKET="bjj-video-analysis"
S3_KEY="aHR0cHM6Ly93d3cueW91dHViZS5jb20vd2F0Y2g_dj1DOEZrVWtaSGxGYw==.mp4"
BOXES_JSON="$PROJECT_DIR/working_log/baselines/m0/boxes.json"
START_TIME="0:00"
END_TIME="0:30"
# Standard mode: ~5 min smoke expected; cap at 20 min.
# Fast mode: ~30s smoke expected; cap at 5 min.
if [[ "$MODE" == "fast" ]]; then
    WALL_BUDGET_SEC=300
else
    WALL_BUDGET_SEC=1200
fi

mkdir -p "$ARTIFACT_DIR"
echo "[m5] mode=$MODE artifact_dir=$ARTIFACT_DIR" | tee -a "$ARTIFACT_DIR/driver.log"

BOX_A=$(python3 -c "import json; d=json.load(open('$BOXES_JSON')); print(json.dumps(d['box_a']))")
BOX_B=$(python3 -c "import json; d=json.load(open('$BOXES_JSON')); print(json.dumps(d['box_b']))")
echo "[m5] box_a=$BOX_A" | tee -a "$ARTIFACT_DIR/driver.log"
echo "[m5] box_b=$BOX_B" | tee -a "$ARTIFACT_DIR/driver.log"

REQ_JSON="$ARTIFACT_DIR/track_request.json"
python3 - "$BOXES_JSON" "$S3_BUCKET" "$S3_KEY" "$START_TIME" "$END_TIME" "$MODE" "$REQ_JSON" <<'PYEOF'
import json, sys
boxes_path, bucket, key, start_time, end_time, mode, out = sys.argv[1:8]
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
    "max_missing_frames": 999999,
    "processing_mode": mode,
}
if start_time: req["start_time"] = start_time
if end_time: req["end_time"] = end_time
json.dump(req, open(out, "w"), indent=2)
print(json.dumps(req, indent=2))
PYEOF
echo "[m5] track request written to $REQ_JSON" | tee -a "$ARTIFACT_DIR/driver.log"

WALL_START_EPOCH=$(date +%s)
echo "[m5] WALL_START=$WALL_START_EPOCH ($(date -u))" | tee -a "$ARTIFACT_DIR/driver.log"

RESP_JSON="$ARTIFACT_DIR/track_response.json"
curl -fsS -X POST "http://127.0.0.1:$PORT/track" \
    -H 'Content-Type: application/json' \
    -d @"$REQ_JSON" > "$RESP_JSON"
JOB_ID=$(python3 -c "import json; print(json.load(open('$RESP_JSON'))['job_id'])")
echo "[m5] job_id=$JOB_ID" | tee -a "$ARTIFACT_DIR/driver.log"
echo "$JOB_ID" > "$ARTIFACT_DIR/job_id.txt"

LIFECYCLE_LOG="$ARTIFACT_DIR/lifecycle.jsonl"
> "$LIFECYCLE_LOG"
DEADLINE=$(( WALL_START_EPOCH + WALL_BUDGET_SEC ))
prev_state=""
while true; do
    NOW=$(date +%s)
    if [[ "$NOW" -ge "$DEADLINE" ]]; then
        echo "[m5] WALL BUDGET EXCEEDED (${WALL_BUDGET_SEC}s); aborting poll" | tee -a "$ARTIFACT_DIR/driver.log"
        break
    fi
    LIFE=$(curl -fsS "http://127.0.0.1:$PORT/job/$JOB_ID" 2>/dev/null || echo '{}')
    TS=$(date +%Y-%m-%dT%H:%M:%S%z)
    echo "{\"ts\":\"$TS\",\"life\":$LIFE}" >> "$LIFECYCLE_LOG"
    STATE=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d.get('status',''))" "$LIFE" 2>/dev/null || echo "")
    if [[ "$STATE" != "$prev_state" ]]; then
        echo "[m5] state -> $STATE @ $TS (job=$JOB_ID)" | tee -a "$ARTIFACT_DIR/driver.log"
        prev_state="$STATE"
    fi
    case "$STATE" in
        completed|failed|cancelled)
            echo "[m5] reached terminal state: $STATE" | tee -a "$ARTIFACT_DIR/driver.log"
            break
            ;;
    esac
    sleep 5
done

WALL_END_EPOCH=$(date +%s)
WALL_TOTAL=$(( WALL_END_EPOCH - WALL_START_EPOCH ))
echo "[m5] WALL_END=$WALL_END_EPOCH ($(date -u))" | tee -a "$ARTIFACT_DIR/driver.log"
echo "[m5] WALL_TOTAL_SEC=$WALL_TOTAL" | tee -a "$ARTIFACT_DIR/driver.log"
echo "[m5] WALL_TOTAL_MIN=$(python3 -c "print(round($WALL_TOTAL/60,2))")" | tee -a "$ARTIFACT_DIR/driver.log"

# Extract tracking and upscale stage timing from service log
echo "[m5] === Extracting stage timings from service log ===" | tee -a "$ARTIFACT_DIR/driver.log"
grep -E "tracking executor|upscal|Gemini|prop_stride|frame_stride|sam2" /tmp/m5-service.log 2>/dev/null | grep "$JOB_ID" | tee -a "$ARTIFACT_DIR/driver.log" || true

echo "[m5] done. artifacts in $ARTIFACT_DIR" | tee -a "$ARTIFACT_DIR/driver.log"
