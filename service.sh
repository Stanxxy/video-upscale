#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────
APP_MODULE="${BJJ_APP_MODULE:-service.app:app}"
HOST="${BJJ_HOST:-0.0.0.0}"
PORT="${BJJ_PORT:-8000}"
LOG_LEVEL="${BJJ_LOG_LEVEL:-info}"
STOP_TIMEOUT_SECONDS="${BJJ_STOP_TIMEOUT_SECONDS:-10}"
# Comma-separated extra uvicorn CLI args (dev command only), e.g. BJJ_UVICORN_EXTRA="--access-log"
BJJ_UVICORN_EXTRA="${BJJ_UVICORN_EXTRA:-}"

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

    if recover_pid_file; then
        return 0
    fi

    return 1
}

get_pid() {
    if [[ -f "$PID_FILE" ]]; then
        cat "$PID_FILE"
    fi
}

find_service_pid() {
    local lsof_bin=""
    for candidate in "$(command -v lsof 2>/dev/null || true)" /usr/sbin/lsof /usr/bin/lsof; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            lsof_bin="$candidate"
            break
        fi
    done

    if [[ -z "$lsof_bin" ]]; then
        return 1
    fi

    local pid
    while IFS= read -r pid; do
        [[ -n "$pid" ]] || continue

        local command
        command=$(ps -p "$pid" -o command= 2>/dev/null || true)
        if [[ "$command" == *"uvicorn"* && "$command" == *"$APP_MODULE"* && "$command" == *"--port $PORT"* ]]; then
            printf '%s\n' "$pid"
            return 0
        fi
    done < <("$lsof_bin" -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)

    return 1
}

recover_pid_file() {
    local pid
    if ! pid=$(find_service_pid); then
        return 1
    fi

    echo "$pid" > "$PID_FILE"
    return 0
}

stop_pid() {
    local pid="$1"
    echo "Stopping service (PID $pid) ..."

    kill "$pid" 2>/dev/null || true

    # Wait for graceful shutdown before escalating.
    local waited=0
    while kill -0 "$pid" 2>/dev/null && (( waited < STOP_TIMEOUT_SECONDS )); do
        sleep 1
        waited=$((waited + 1))
    done

    if kill -0 "$pid" 2>/dev/null; then
        yellow "Graceful stop timed out, sending SIGKILL ..."
        kill -9 "$pid" 2>/dev/null || true
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

    # Locate virtual environment python
    local venv_python="$PROJECT_DIR/venv/bin/python"
    if [[ ! -x "$venv_python" ]]; then
        red "Virtual environment not found at $PROJECT_DIR/venv"
        echo "  Create one with:"
        echo "    python3 -m venv venv"
        echo "    venv/bin/pip install -r requirements-service.txt"
        echo "    venv/bin/pip install 'git+https://github.com/facebookresearch/sam2.git'"
        return 1
    fi

    # Load .env if present (without overriding existing vars)
    if [[ -f "$PROJECT_DIR/.env" ]]; then
        set -a
        # shellcheck disable=SC1091
        source "$PROJECT_DIR/.env"
        set +a
    fi

    # Single worker required: WebSocket + in-memory state don't survive forking
    local -a uvicorn_cmd=(
        "$venv_python" -m uvicorn "$APP_MODULE"
        --host "$HOST"
        --port "$PORT"
        --log-level "$LOG_LEVEL"
    )
    if [[ "${BJJ_RELOAD:-}" == 1 ]]; then
        yellow "BJJ_RELOAD=1: adding --reload (PID file tracks initial process only; prefer './service.sh dev' for local dev)"
        uvicorn_cmd+=(--reload)
        if [[ -n "${BJJ_RELOAD_DIR:-}" ]]; then
            uvicorn_cmd+=(--reload-dir "$BJJ_RELOAD_DIR")
        fi
    fi
    if [[ -n "$BJJ_UVICORN_EXTRA" ]]; then
        # shellcheck disable=SC2206
        uvicorn_cmd+=($BJJ_UVICORN_EXTRA)
    fi

    if [[ "${BJJ_FOREGROUND:-}" == 1 ]]; then
        yellow "BJJ_FOREGROUND=1: logs go to terminal (not $LOG_FILE)"
        exec "${uvicorn_cmd[@]}"
    fi

    nohup "${uvicorn_cmd[@]}" >> "$LOG_FILE" 2>&1 &

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
    stop_pid "$pid"

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

cmd_dev() {
    if is_running; then
        yellow "Service is already running (PID $(get_pid)) — stop it first or use a different BJJ_PORT"
        return 1
    fi

    echo "Starting dev server (foreground, --reload) on $HOST:$PORT ..."
    cd "$PROJECT_DIR"

    local venv_python="$PROJECT_DIR/venv/bin/python"
    if [[ ! -x "$venv_python" ]]; then
        red "Virtual environment not found at $PROJECT_DIR/venv"
        return 1
    fi

    if [[ -f "$PROJECT_DIR/.env" ]]; then
        set -a
        # shellcheck disable=SC1091
        source "$PROJECT_DIR/.env"
        set +a
    fi

    local dev_log_level="${BJJ_LOG_LEVEL:-debug}"
    local -a uvicorn_cmd=(
        "$venv_python" -m uvicorn "$APP_MODULE"
        --host "$HOST"
        --port "$PORT"
        --log-level "$dev_log_level"
        --reload
    )
    if [[ -n "${BJJ_RELOAD_DIR:-}" ]]; then
        uvicorn_cmd+=(--reload-dir "$BJJ_RELOAD_DIR")
    fi
    if [[ -n "$BJJ_UVICORN_EXTRA" ]]; then
        # shellcheck disable=SC2206
        uvicorn_cmd+=($BJJ_UVICORN_EXTRA)
    fi

    yellow "Log level: $dev_log_level (set BJJ_LOG_LEVEL to override)"
    exec "${uvicorn_cmd[@]}"
}

# ── Usage ──────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") {start|dev|stop|restart|status|logs}

Commands:
  start     Start the service in the background (nohup -> $LOG_FILE)
  dev       Start in the foreground with uvicorn --reload (local development)
  stop      Stop the running service
  restart   Restart the service
  status    Check if the service is running
  logs      Tail the service log file

Environment variables:
  BJJ_HOST           Listen host     (default: 0.0.0.0)
  BJJ_PORT           Listen port     (default: 8000)
  BJJ_LOG_LEVEL      Log level       (default: info for start, debug for dev)
  BJJ_FOREGROUND=1   With start: run in foreground (logs to terminal, no nohup)
  BJJ_RELOAD=1       With start: add --reload (prefer ./service.sh dev)
  BJJ_RELOAD_DIR     Optional directory for uvicorn --reload-dir (repeat via dev/start+reload)
  BJJ_UVICORN_EXTRA  Extra uvicorn CLI tokens (word-split; dev and start)
EOF
}

# ── Main ───────────────────────────────────────────────────────
case "${1:-}" in
    start)   cmd_start   ;;
    dev)     cmd_dev     ;;
    stop)    cmd_stop    ;;
    restart) cmd_restart ;;
    status)  cmd_status  ;;
    logs)    cmd_logs    ;;
    *)       usage; exit 1 ;;
esac
