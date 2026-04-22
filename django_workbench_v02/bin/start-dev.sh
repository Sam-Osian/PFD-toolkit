#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export SECRET_KEY="${SECRET_KEY:-local-dev-check-key-only}"
# Force a safe local-dev runtime even when the parent shell exports non-boolean DEBUG
# values (for example DEBUG=release), which Django interprets as DEBUG=False.
export DEBUG="True"
export ARTIFACT_STORAGE_BACKEND="${ARTIFACT_STORAGE_BACKEND:-file}"
export ARTIFACT_ENFORCE_OBJECT_STORAGE_IN_PRODUCTION="False"
HOST="${DEV_HOST:-127.0.0.1}"
PORT="${DEV_PORT:-8000}"
WORKER_ID="${DEV_WORKER_ID:-local-worker-1}"
WORKER_POLL_SECONDS="${DEV_WORKER_POLL_SECONDS:-3}"
RUN_DISPATCHER="${DEV_RUN_DISPATCHER:-0}"
BASE_URL="http://${HOST}:${PORT}"

# Force local Auth0 callback/logout URLs to match the chosen dev host/port.
# This avoids stale shell env values sending callbacks to another port.
export WORKBENCH_BASE_URL="${BASE_URL}"
export AUTH0_CALLBACK_URL="${BASE_URL}/auth/callback/"
export AUTH0_POST_LOGOUT_REDIRECT_URI="${BASE_URL}/"

cleanup() {
  local exit_code=$?
  trap - INT TERM EXIT
  if [[ -n "${WEB_PID:-}" ]]; then kill "$WEB_PID" 2>/dev/null || true; fi
  if [[ -n "${WORKER_PID:-}" ]]; then kill "$WORKER_PID" 2>/dev/null || true; fi
  if [[ -n "${DISPATCHER_PID:-}" ]]; then kill "$DISPATCHER_PID" 2>/dev/null || true; fi
  wait 2>/dev/null || true
  exit "$exit_code"
}
trap cleanup INT TERM EXIT

echo "[dev] Applying migrations..."
uv run python manage.py migrate

echo "[dev] Starting worker (id=${WORKER_ID}, poll=${WORKER_POLL_SECONDS}s)..."
uv run python manage.py run_runs_worker --worker-id "$WORKER_ID" --poll-seconds "$WORKER_POLL_SECONDS" &
WORKER_PID=$!

if [[ "$RUN_DISPATCHER" == "1" ]]; then
  echo "[dev] Starting notification dispatcher..."
  uv run python manage.py run_notification_dispatcher --poll-seconds 5 --max-items 50 &
  DISPATCHER_PID=$!
fi

echo "[dev] Starting web server at http://${HOST}:${PORT}/ ..."
uv run python manage.py runserver "${HOST}:${PORT}" &
WEB_PID=$!

echo "[dev] Running. Press Ctrl+C to stop all services."
wait
