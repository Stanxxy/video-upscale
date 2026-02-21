#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────
APP_MODULE="service.app:app"
HOST="${BJJ_HOST:-0.0.0.0}"
PORT="${BJJ_PORT:-9001}"
WORKERS="${BJJ_WORKERS:-1}"
LOG_LEVEL="${BJJ_LOG_LEVEL:-info}"

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$PROJECT_DIR/.service.pid"
LOG_FILE="$PROJECT_DIR/service.log"

# ── Helpers ────────────────────────────────────────────────────
red()   { printf '\033[0;31m%s\033[0m\n' "$*"; }
green() { printf '\033[0;32m%s\033[0m\n' "$*"; }
yellow(){ printf '\033[0;33m%s\033[0m\n' "$*"; }

is_running() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid=$(<"$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        # Stale pid file
        rm -f "$PID_FILE"
    fi
    return 1
}

get_pid() {
    if [[ -f "$PID_FILE" ]]; then
        cat "$PID_FILE"
    fi
}

# ── Commands ───────────────────────────────────────────────────
cmd_start() {
    if is_running; then
        yellow "Service is already running (PID $(get_pid))"
        return 0
    fi

    echo "Starting service on $HOST:$PORT ..."
    cd "$PROJECT_DIR"

    nohup uvicorn "$APP_MODULE" \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level "$LOG_LEVEL" \
        >> "$LOG_FILE" 2>&1 &

    local pid=$!
    echo "$pid" > "$PID_FILE"

    # Brief wait to verify process started
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
        green "Service started (PID $pid)"
        echo "  Logs: $LOG_FILE"
        echo "  Docs: http://localhost:$PORT/docs"
    else
        red "Service failed to start — check $LOG_FILE"
        rm -f "$PID_FILE"
        return 1
    fi
}

cmd_stop() {
    if ! is_running; then
        yellow "Service is not running"
        return 0
    fi

    local pid
    pid=$(get_pid)
    echo "Stopping service (PID $pid) ..."

    kill "$pid"

    # Wait up to 10s for graceful shutdown
    local waited=0
    while kill -0 "$pid" 2>/dev/null && (( waited < 10 )); do
        sleep 1
        (( waited++ ))
    done

    if kill -0 "$pid" 2>/dev/null; then
        yellow "Graceful stop timed out, sending SIGKILL ..."
        kill -9 "$pid" 2>/dev/null || true
    fi

    rm -f "$PID_FILE"
    green "Service stopped"
}

cmd_restart() {
    cmd_stop
    cmd_start
}

cmd_status() {
    if is_running; then
        local pid
        pid=$(get_pid)
        green "Service is running (PID $pid)"
        # Try a quick health check
        if command -v curl &>/dev/null; then
            echo -n "  Health: "
            curl -sf "http://localhost:$PORT/health" 2>/dev/null || echo "unreachable"
        fi
    else
        red "Service is not running"
        return 1
    fi
}

cmd_logs() {
    if [[ ! -f "$LOG_FILE" ]]; then
        yellow "No log file found"
        return 1
    fi
    tail -f "$LOG_FILE"
}

# ── Usage ──────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") {start|stop|restart|status|logs}

Commands:
  start     Start the service in the background
  stop      Stop the running service
  restart   Restart the service
  status    Check if the service is running
  logs      Tail the service log file

Environment variables:
  BJJ_HOST        Listen host     (default: 0.0.0.0)
  BJJ_PORT        Listen port     (default: 9001)
  BJJ_WORKERS     Uvicorn workers (default: 1)
  BJJ_LOG_LEVEL   Log level       (default: info)
EOF
}

# ── Main ───────────────────────────────────────────────────────
case "${1:-}" in
    start)   cmd_start   ;;
    stop)    cmd_stop    ;;
    restart) cmd_restart ;;
    status)  cmd_status  ;;
    logs)    cmd_logs    ;;
    *)       usage; exit 1 ;;
esac
