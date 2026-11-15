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
# Build real forecast-vintage history without changing the active load model.
# Explicit off/model settings and a custom NWS identity remain configurable.
export ERCOT_WEATHER_MODE="${ERCOT_WEATHER_MODE:-collect}"
export ERCOT_WEATHER_USER_AGENT="${ERCOT_WEATHER_USER_AGENT:-ercot-load-forecast/1.0}"
cd "$FORECAST_REPO"
exec "$FORECAST_PYTHON" -m ERCOTAPI.load_forecast_report "$@"
