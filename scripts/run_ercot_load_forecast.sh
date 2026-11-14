#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
FORECAST_REPO="${ERCOT_REPO_ROOT:-${SCRIPT_DIR:h}}"
FORECAST_PYTHON="${ERCOT_FORECAST_PYTHON:-$FORECAST_REPO/.venv/bin/python}"

if [[ ! -x "$FORECAST_PYTHON" ]]; then
  echo "Set ERCOT_FORECAST_PYTHON to an environment with the forecast dependencies." >&2
  exit 78
fi
export PYTHONPATH="$FORECAST_REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$FORECAST_REPO"
exec "$FORECAST_PYTHON" -m ERCOTAPI.load_forecast_report
