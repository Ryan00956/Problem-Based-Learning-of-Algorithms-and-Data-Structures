#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8013}"
DATASET="${2:-movielens}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$PROJECT_ROOT"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Local virtual environment was not found. Creating it now..."
  ./setup_venv.sh
fi

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Python is required but was not found in PATH. Run ./setup_venv.sh after installing Python 3.10+." >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c "import pandas, numpy, fastapi, uvicorn, duckdb" >/dev/null 2>&1; then
  echo "Python dependencies are missing. Installing the demo requirements now..."
  ./setup_venv.sh
  PYTHON_BIN=".venv/bin/python"
fi

stop_project_api_on_port() {
  local port="$1"

  if ! command -v lsof >/dev/null 2>&1; then
    return
  fi

  local pids
  pids="$(lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -z "$pids" ]]; then
    return
  fi

  local pid cmd
  for pid in $pids; do
    cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    if [[ "$cmd" == *"src.api"* && "$cmd" == *"--port"* && "$cmd" == *"${port}"* ]]; then
      echo "Stopping existing backend process ${pid} on port ${port}..."
      kill "$pid" 2>/dev/null || true
      for _ in $(seq 1 20); do
        if ! kill -0 "$pid" 2>/dev/null; then
          break
        fi
        sleep 0.25
      done
      if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
      fi
    fi
  done
}

stop_project_api_on_port "$PORT"

if "$PYTHON_BIN" - <<PY >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:${PORT}/api/health", timeout=2)
PY
then
  echo "Port ${PORT} is already serving an API that was not started by this script. Stop that process or choose another port." >&2
  exit 1
fi

if "$PYTHON_BIN" - <<PY >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:${PORT}/", timeout=2)
PY
then
  echo "Port ${PORT} is already serving a non-API frontend. Stop that process or choose another port." >&2
  exit 1
fi

if command -v xdg-open >/dev/null 2>&1; then
  (
    for _ in $(seq 1 40); do
      if "$PYTHON_BIN" - <<PY >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:${PORT}/api/health", timeout=2)
PY
      then
        xdg-open "http://127.0.0.1:${PORT}/" >/dev/null 2>&1 || true
        break
      fi
      sleep 0.5
    done
  ) &
elif command -v open >/dev/null 2>&1; then
  (
    for _ in $(seq 1 40); do
      if "$PYTHON_BIN" - <<PY >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:${PORT}/api/health", timeout=2)
PY
      then
        open "http://127.0.0.1:${PORT}/" >/dev/null 2>&1 || true
        break
      fi
      sleep 0.5
    done
  ) &
fi

echo "Frontend URL: http://127.0.0.1:${PORT}/"
echo "API Health: http://127.0.0.1:${PORT}/api/health"
echo "Dataset: ${DATASET}"
echo "Backend is running in this terminal. Press Ctrl+C or close this window to stop it."

"$PYTHON_BIN" -m src.api --port "$PORT" --dataset "$DATASET" &
SERVER_PID="$!"

cleanup_server() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}

trap cleanup_server INT TERM EXIT
wait "$SERVER_PID"
