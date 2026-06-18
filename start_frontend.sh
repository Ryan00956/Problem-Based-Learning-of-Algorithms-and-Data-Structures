#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8013}"
DATASET="${2:-movielens}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$PROJECT_ROOT"

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

if ! "$PYTHON_BIN" -c "import fastapi, uvicorn" >/dev/null 2>&1; then
  echo "FastAPI dependencies are missing. Run ./setup_venv.sh to install requirements.txt." >&2
  exit 1
fi

if "$PYTHON_BIN" - <<PY >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:${PORT}/api/health", timeout=2)
PY
then
  echo "Frontend URL: http://127.0.0.1:${PORT}/"
  echo "API Health: http://127.0.0.1:${PORT}/api/health"
  echo "Dataset: ${DATASET}"
  exit 0
fi

if "$PYTHON_BIN" - <<PY >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:${PORT}/", timeout=2)
PY
then
  echo "Port ${PORT} is already serving a non-API frontend. Stop that process or choose another port." >&2
  exit 1
fi

nohup "$PYTHON_BIN" -m src.api --port "$PORT" --dataset "$DATASET" >/tmp/movie_recommendation_lab_${PORT}.log 2>&1 &
READY=0
for _ in $(seq 1 20); do
  sleep 0.5
  if "$PYTHON_BIN" - <<PY >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:${PORT}/api/health", timeout=2)
PY
  then
    READY=1
    break
  fi
done

if [[ "$READY" != "1" ]]; then
  echo "FastAPI server did not become ready on port ${PORT}." >&2
  echo "Server log: /tmp/movie_recommendation_lab_${PORT}.log" >&2
  exit 1
fi

echo "Frontend URL: http://127.0.0.1:${PORT}/"
echo "API Health: http://127.0.0.1:${PORT}/api/health"
echo "Dataset: ${DATASET}"
echo "Server log: /tmp/movie_recommendation_lab_${PORT}.log"
