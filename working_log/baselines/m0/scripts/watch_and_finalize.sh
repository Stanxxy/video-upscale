#!/usr/bin/env bash
# Watch a running M0 job to terminal, then dump timing and cleanup.
set -euo pipefail

JOB_ID="${1:?usage: watch_and_finalize.sh <job_id> <artifact_dir> [<wall_start_epoch>]}"
ARTIFACT_DIR="${2:?missing artifact dir}"
WALL_START="${3:-}"
PORT="${BJJ_PORT:-8000}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/bjj/whole-video-analysis}"
VENV="${VENV:-$HOME/bjj/.venv-spark}"
WALL_BUDGET_SEC="${WALL_BUDGET_SEC:-21600}"

source "$VENV/bin/activate"

# Recover wall_start from existing driver.log if not provided
if [[ -z "$WALL_START" ]]; then
    WALL_START=$(awk -F= '/WALL_START_EPOCH=/ {print $2; exit}' "$ARTIFACT_DIR/driver.log")
fi
if [[ -z "$WALL_START" ]]; then
    WALL_START=$(date +%s)
    echo "[watch] WARN: could not recover wall_start, using now=$WALL_START"
fi
echo "[watch] WALL_START_EPOCH=$WALL_START" | tee -a "$ARTIFACT_DIR/driver.log"

LIFECYCLE_LOG="$ARTIFACT_DIR/lifecycle.jsonl"
DEADLINE=$(( WALL_START + WALL_BUDGET_SEC ))
prev_state=""
while true; do
    NOW=$(date +%s)
    if [[ "$NOW" -ge "$DEADLINE" ]]; then
        echo "[watch] WALL BUDGET EXCEEDED (${WALL_BUDGET_SEC}s); aborting poll" | tee -a "$ARTIFACT_DIR/driver.log"
        break
    fi
    LIFE=$(curl -fsS "http://127.0.0.1:$PORT/job/$JOB_ID" 2>/dev/null || echo '{}')
    TS=$(date +%Y-%m-%dT%H:%M:%S%z)
    echo "{\"ts\":\"$TS\",\"life\":$LIFE}" >> "$LIFECYCLE_LOG"
    STATE=$(python -c "import json,sys; d=json.loads('''$LIFE'''); print(d.get('status',''))" 2>/dev/null || echo "")
    if [[ "$STATE" != "$prev_state" ]]; then
        echo "[watch] state -> $STATE @ $TS (job=$JOB_ID)" | tee -a "$ARTIFACT_DIR/driver.log"
        prev_state="$STATE"
    fi
    case "$STATE" in
        completed|failed|cancelled)
            echo "[watch] reached terminal state: $STATE" | tee -a "$ARTIFACT_DIR/driver.log"
            break
            ;;
    esac
    sleep 5
done

WALL_END=$(date +%s)
echo "[watch] WALL_END_EPOCH=$WALL_END" | tee -a "$ARTIFACT_DIR/driver.log"
echo "[watch] WALL_TOTAL_SEC=$(( WALL_END - WALL_START ))" | tee -a "$ARTIFACT_DIR/driver.log"

echo "[watch] dumping timing" | tee -a "$ARTIFACT_DIR/driver.log"
cd "$PROJECT_DIR"
python "$PROJECT_DIR/working_log/baselines/m0/scripts/dump_timing.py" \
    --job-id "$JOB_ID" \
    --out-dir "$ARTIFACT_DIR" \
    --wall-start "$WALL_START" \
    --wall-end "$WALL_END" \
    2>&1 | tee -a "$ARTIFACT_DIR/driver.log"

echo "[watch] done" | tee -a "$ARTIFACT_DIR/driver.log"
