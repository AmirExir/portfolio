#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
if [[ -z "${ERCOT_REPO_ROOT:-}" ]]; then
  if [[ -d "$SCRIPT_DIR/../chatbot_ercot_all_in_one" ]]; then
    ERCOT_REPO_ROOT="${SCRIPT_DIR:h}"
  else
    # Supports the existing launchd setup, which copies this script beneath
    # ~/Library/Application Support. Override ERCOT_REPO_ROOT for other clones.
    ERCOT_REPO_ROOT="$HOME/Documents/GitHub/portfolio"
  fi
fi
APP_DIR="${ERCOT_RAG_APP_DIR:-$ERCOT_REPO_ROOT/chatbot_ercot_all_in_one}"
PYTHON="${ERCOT_RAG_PYTHON:-$ERCOT_REPO_ROOT/.venv/bin/python}"
SERVICE_NAME="openai-api-key"
ACCOUNT="${USER:-$(/usr/bin/id -un)}"

export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"
export PYTHONPATH="$ERCOT_REPO_ROOT:$APP_DIR${PYTHONPATH:+:$PYTHONPATH}"
export ERCOT_RAG_STORE="${ERCOT_RAG_STORE:-$ERCOT_REPO_ROOT/ERCOTAPI/.rag_store}"
API_HOST="${ERCOT_RAG_API_HOST:-0.0.0.0}"
API_PORT="${ERCOT_RAG_API_PORT:-8000}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERCOT RAG Python is not executable: $PYTHON" >&2
  echo "Set ERCOT_RAG_PYTHON to a Python environment with chatbot requirements installed." >&2
  exit 78
fi
if [[ ! -d "$APP_DIR" ]]; then
  echo "ERCOT RAG app directory does not exist: $APP_DIR" >&2
  echo "Set ERCOT_REPO_ROOT or ERCOT_RAG_APP_DIR for this checkout." >&2
  exit 78
fi

cd "$APP_DIR"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  OPENAI_API_KEY="$(/usr/bin/security find-generic-password -a "$ACCOUNT" -s "$SERVICE_NAME" -w 2>/dev/null || true)"
  export OPENAI_API_KEY
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not available. Add it once with:"
  echo "security add-generic-password -a \"\$USER\" -s \"$SERVICE_NAME\" -w \"your-openai-key\" -U"
  sleep 300
  exit 78
fi

exec "$PYTHON" -m uvicorn ercot_rag_api:app --host "$API_HOST" --port "$API_PORT"
