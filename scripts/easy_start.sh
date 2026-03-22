#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

info() { echo "[local-research-agent] $*"; }

if command -v docker >/dev/null 2>&1; then
  info "Docker detected. Starting with docker compose..."
  if command -v docker-compose >/dev/null 2>&1; then
    docker-compose up --build -d
  else
    docker compose up --build -d
  fi
  info "App is starting in Docker. Open http://localhost:8000"
  info "To stop: docker compose down"
  exit 0
fi

info "Docker not found. Falling back to local Python mode."
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3 is required but not found. Please install Python 3.11+ and rerun."
  exit 1
fi

if [ ! -d ".venv" ]; then
  info "Creating virtual environment..."
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

info "Installing dependencies..."
pip install -r requirements.txt

info "Installing Playwright Chromium (needed for browser fallback)..."
python -m playwright install chromium || info "Playwright install failed; app can still run in degraded mode."

info "Starting backend on http://127.0.0.1:8000"
exec uvicorn app.main:app --host 127.0.0.1 --port 8000
