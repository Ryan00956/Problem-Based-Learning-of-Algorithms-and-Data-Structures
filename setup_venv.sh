#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="${1:-requirements.txt}"
cd "$PROJECT_ROOT"

if [[ ! -x ".venv/bin/python" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    python3 -m venv .venv
  elif command -v python >/dev/null 2>&1; then
    python -m venv .venv
  else
    echo "Python 3.10+ was not found. Install Python, then rerun this script." >&2
    exit 1
  fi
fi

".venv/bin/python" -m pip install --upgrade pip
".venv/bin/python" -m pip install -r "$REQUIREMENTS"

echo "Virtual environment is ready: $PROJECT_ROOT/.venv"
echo "Installed dependencies from: $REQUIREMENTS"
echo "Run commands with ./run.sh or start the dashboard with ./start_frontend.sh"
