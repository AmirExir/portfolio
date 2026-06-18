#!/bin/zsh
set -euo pipefail

APP_DIR="/Users/amirexir/Documents/GitHub/portfolio/chatbot_ercot_all_in_one"
PYTHON="/opt/homebrew/opt/python@3.13/bin/python3.13"
VENV_SITE_PACKAGES="/Users/amirexir/Documents/GitHub/portfolio/.venv/lib/python3.13/site-packages"
SERVICE_NAME="openai-api-key"
ACCOUNT="${USER:-amirexir}"

export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"
export PYTHONPATH="$APP_DIR:$VENV_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"

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

exec "$PYTHON" -m uvicorn ercot_rag_api:app --host 127.0.0.1 --port 8000
