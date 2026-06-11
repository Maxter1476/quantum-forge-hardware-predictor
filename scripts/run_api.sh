#!/usr/bin/env bash
# Start the FastAPI backend.
set -euo pipefail
cd "$(dirname "$0")/.."
HOST="${QFHP_API_HOST:-127.0.0.1}"
PORT="${QFHP_API_PORT:-8000}"
exec python3 -m uvicorn app.api.main:app --host "$HOST" --port "$PORT" "$@"
