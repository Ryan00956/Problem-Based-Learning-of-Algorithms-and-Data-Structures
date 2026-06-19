#!/usr/bin/env bash
set -euo pipefail

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

DATASET="movielens"
PREV=""
for ARG in "$@"; do
  if [[ "$PREV" == "--dataset" ]]; then
    DATASET="$ARG"
    break
  fi
  PREV="$ARG"
done

"$PYTHON_BIN" -m src.bootstrap_data --dataset "$DATASET"
"$PYTHON_BIN" -m src.main "$@"
