#!/bin/bash
# Kill all leftover processes related to slime training.
# Safe to use from any directory or standalone.

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$(dirname "$WORKSPACE_DIR")/slime_env"

echo "[$(date)] Cleaning up slime processes..."

# 1) kill sglang engines launched from this venv
pkill -9 -f "$VENV_DIR/.*sglang" 2>/dev/null && echo "  killed sglang processes" || echo "  no sglang processes"
sleep 1

# 2) kill sglang router/engine by port name pattern (in case they escaped venv matching)
pkill -9 -f "sglang.launch_server\|sglang.srt.entrypoints\|sglang.srt.managers" 2>/dev/null && echo "  killed sglang extra" || echo "  no sglang extra"

# 3) stop ray (all open clusters)
ray stop --force 2>/dev/null && echo "  ray stopped" || echo "  ray was not running"
sleep 2

# 4) kill any lingering ray/actor/nccld processes
pkill -9 -f "$VENV_DIR/.*(ray|python|train\.py|nccld)" 2>/dev/null && echo "  killed ray/python processes" || echo "  no ray/python processes"

# 5) kill any leftover gunicorn/uvicorn (if using)
pkill -9 -f "gunicorn.*$VENV_DIR\|uvicorn.*$VENV_DIR" 2>/dev/null && echo "  killed gunicorn/uvicorn" || echo "  no gunicorn/uvicorn"

echo "[$(date)] Cleanup done."
