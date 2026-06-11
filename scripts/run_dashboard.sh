#!/usr/bin/env bash
# Start the Streamlit dashboard.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 -m streamlit run app/dashboard/app.py "$@"
