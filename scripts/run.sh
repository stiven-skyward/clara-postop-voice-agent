#!/usr/bin/env bash
# Arranca el agente completo (llama-server se lanza solo desde la app si no está corriendo).
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="${POSTOP_VENV:-.venv}"
exec "$VENV/bin/python" -m uvicorn app.main:app --host "${POSTOP_HOST:-127.0.0.1}" --port "${POSTOP_PORT:-8000}"
