#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8013}"
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

"$PYTHON_BIN" -m src.export_frontend_data

if "$PYTHON_BIN" - <<PY >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:${PORT}/", timeout=2)
PY
then
  echo "Frontend URL: http://127.0.0.1:${PORT}/"
  exit 0
fi

nohup "$PYTHON_BIN" -m http.server "$PORT" --directory web >/tmp/movie_recommendation_lab_${PORT}.log 2>&1 &
sleep 1

echo "Frontend URL: http://127.0.0.1:${PORT}/"
echo "Server log: /tmp/movie_recommendation_lab_${PORT}.log"
