#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${ERCOT_REPO_ROOT:-${SCRIPT_DIR:h}}"
MONITOR_DIR="${ERCOT_MONITOR_DIR:-$REPO_ROOT/ERCOTAPI}"
PYTHON="${ERCOT_MONITOR_PYTHON:-$MONITOR_DIR/.venv/bin/python}"
KEYCHAIN_SERVICE="${ERCOT_OPENAI_KEYCHAIN_SERVICE:-openai-api-key}"
KEYCHAIN_ACCOUNT="${ERCOT_OPENAI_KEYCHAIN_ACCOUNT:-${USER:-$(/usr/bin/id -un)}}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERCOT monitor Python is not executable: $PYTHON" >&2
  echo "Set ERCOT_MONITOR_PYTHON to the scheduled job's Python environment." >&2
  exit 78
fi
if [[ ! -f "$MONITOR_DIR/ercot_link_monitor.py" ]]; then
  echo "ERCOT monitor entry point does not exist: $MONITOR_DIR/ercot_link_monitor.py" >&2
  echo "Set ERCOT_REPO_ROOT or ERCOT_MONITOR_DIR for this checkout." >&2
  exit 78
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "${OPENAI_API_KEY:-}" && -x /usr/bin/security ]]; then
  OPENAI_API_KEY="$(/usr/bin/security find-generic-password -a "$KEYCHAIN_ACCOUNT" -s "$KEYCHAIN_SERVICE" -w 2>/dev/null || true)"
  export OPENAI_API_KEY
fi
cd "$MONITOR_DIR"
exec "$PYTHON" "$MONITOR_DIR/ercot_link_monitor.py"
