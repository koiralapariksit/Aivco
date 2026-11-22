#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

echo "========================================================"
echo "       STARTING AIVCO ARGON EDITION SERVER...           "
echo "========================================================"
echo "Access URL: http://127.0.0.1:8000"
echo

open "http://127.0.0.1:8000" >/dev/null 2>&1 || true
exec ./.venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port 8000 --reload
