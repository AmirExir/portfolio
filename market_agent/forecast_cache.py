from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agent.forecast import ForecastResult


CACHE_VERSION = 3


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def signal_quality(confidence_pct: float) -> str:
    edge_pct = max(float(confidence_pct) - 50.0, 0.0)
    if edge_pct >= 15.0:
        return "High Edge"
    if edge_pct >= 8.0:
        return "Moderate Edge"
    if edge_pct >= 3.0:
        return "Weak Edge"
    return "No Edge"


def model_call(forecast_change_pct: float, expected_error_pct: float, confidence_pct: float) -> str:
    edge_pct = max(float(confidence_pct) - 50.0, 0.0)
    if edge_pct < 3.0 or abs(forecast_change_pct) < expected_error_pct * 0.5:
        return "Neutral / No Edge"
    if forecast_change_pct >= expected_error_pct and edge_pct >= 8.0:
        return "Strong Buy"
    if forecast_change_pct > 0:
        return "Buy"
    if forecast_change_pct <= -expected_error_pct and edge_pct >= 8.0:
        return "Strong Sell"
    if forecast_change_pct < 0:
        return "Sell"
    return "Neutral / No Edge"


def build_cache_key(
    symbols: tuple[str, ...] | list[str],
    history_days: int,
    horizon_days: int,
    lookback_window: int,
    ridge_alpha: float,
    optimize_model: bool,
    use_market_context: bool,
    sequence_model: str = "off",
) -> str:
    payload = {
        "cache_version": CACHE_VERSION,
        "symbols": list(symbols),
        "history_days": int(history_days),
        "horizon_days": int(horizon_days),
        "lookback_window": int(lookback_window),
        "ridge_alpha": float(ridge_alpha),
        "optimize_model": bool(optimize_model),
        "use_market_context": bool(use_market_context),
        "sequence_model": str(sequence_model or "off"),
    }
    raw = json.dumps(payload, sort_keys=True, default=_json_safe).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def forecast_cache_path(
    output_dir: str | Path,
    symbols: tuple[str, ...] | list[str],
    history_days: int,
    horizon_days: int,
    lookback_window: int,
    ridge_alpha: float,
    optimize_model: bool,
    use_market_context: bool,
    sequence_model: str = "off",
) -> Path:
    output_path = Path(output_dir)
    cache_key = build_cache_key(
        symbols=symbols,
        history_days=history_days,
        horizon_days=horizon_days,
        lookback_window=lookback_window,
        ridge_alpha=ridge_alpha,
        optimize_model=optimize_model,
        use_market_context=use_market_context,
        sequence_model=sequence_model,
    )
    return output_path / f"ml_forecast_rankings_cache_{cache_key}.json"


def forecast_result_to_dict(result: ForecastResult) -> dict:
    forecast_rows = []
    if result.forecast is not None and not result.forecast.empty:
        forecast_frame = result.forecast.reset_index().copy()
        if "date" in forecast_frame.columns:
            forecast_frame["date"] = pd.to_datetime(forecast_frame["date"], errors="coerce").astype(str)
        forecast_rows = forecast_frame.to_dict(orient="records")

    return {
        "model_name": result.model_name,
        "metrics": _json_safe(result.metrics),
        "forecast": _json_safe(forecast_rows),
    }


def forecast_result_from_dict(payload: dict) -> ForecastResult:
    forecast_rows = payload.get("forecast") or []
    if forecast_rows:
        forecast = pd.DataFrame(forecast_rows)
        if "date" in forecast.columns:
            forecast["date"] = pd.to_datetime(forecast["date"], errors="coerce")
            forecast = forecast.set_index("date")
            forecast.index.name = "date"
    else:
        forecast = pd.DataFrame()

    return ForecastResult(
        forecast=forecast,
        metrics=payload.get("metrics", {}),
        model_name=payload.get("model_name", ""),
    )


def select_model_name(model_results: dict[str, ForecastResult | dict], preferred: str = "Ensemble") -> str:
    preferred = (preferred or "").strip()
    if preferred == "Best Validation":
        candidates = [
            (name, result)
            for name, result in model_results.items()
            if _has_forecast(result)
        ]
        if not candidates:
            return ""
        return min(candidates, key=lambda item: _metric(item[1], "holdout_mae_pct", np.inf))[0]

    if preferred and preferred in model_results and _has_forecast(model_results[preferred]):
        return preferred

    candidates = [(name, result) for name, result in model_results.items() if _has_forecast(result)]
    if not candidates:
        return ""
    if preferred and preferred in model_results:
        return preferred
    return min(candidates, key=lambda item: _metric(item[1], "holdout_mae_pct", np.inf))[0]


def snapshot_from_model_results(symbol: str, last_price: float, model_results: dict[str, ForecastResult]) -> dict:
    return {
        "symbol": symbol,
        "last_price": float(last_price),
        "models": {name: forecast_result_to_dict(result) for name, result in model_results.items()},
    }


def snapshot_to_ranking_row(snapshot: dict, primary_model_choice: str) -> dict:
    model_payloads = snapshot.get("models", {}) or {}
    selected_model = select_model_name(model_payloads, preferred=primary_model_choice)
    if not selected_model:
        raise ValueError("No usable forecast model result.")

    selected_payload = model_payloads[selected_model]
    selected_result = forecast_result_from_dict(selected_payload)
    metrics = selected_result.metrics or {}
    forecast_rows = selected_result.forecast
    if forecast_rows is None or forecast_rows.empty:
        raise ValueError("No usable forecast model result.")

    confidence = float(metrics.get("confidence_pct", 0.0))
    expected_error = float(metrics.get("expected_error_pct", 0.0))
    forecast_return = float(metrics.get("forecast_change_pct", 0.0))
    probability_up = float(metrics.get("probability_up_pct", 50.0))
    edge = max(confidence - 50.0, 0.0)
    score = float(metrics.get("forecast_score", 0.0))
    forecast_price = float(forecast_rows["forecast_close"].iloc[-1])

    def _return_for(model_name: str):
        model_payload = model_payloads.get(model_name)
        if not model_payload:
            return np.nan
        return float((model_payload.get("metrics") or {}).get("forecast_change_pct", np.nan))

    return {
        "Symbol": snapshot.get("symbol", ""),
        "Model Call": model_call(forecast_return, expected_error, confidence),
        "Selected Model": selected_model,
        "Last Price": float(snapshot.get("last_price", np.nan)),
        "Forecast Price": forecast_price,
        "Forecast Return %": forecast_return,
        "Ridge Return %": _return_for("Ridge"),
        "XGBoost Return %": _return_for("XGBoost"),
        "Neural Net Return %": _return_for("Neural Net"),
        "LSTM Return %": _return_for("LSTM"),
        "Transformer Return %": _return_for("Transformer"),
        "Ensemble Return %": _return_for("Ensemble"),
        "Probability Up %": probability_up,
        "Probability Down %": 100.0 - probability_up,
        "Directional Probability %": confidence,
        "Model Edge %": edge,
        "Signal Quality": signal_quality(confidence),
        "Expected Error %": expected_error,
        "Validation MAE %": float(metrics.get("holdout_mae_pct", np.nan)),
        "Direction Hit Rate %": float(metrics.get("holdout_direction_accuracy", np.nan)),
        "Score": score,
    }


def snapshots_to_ranking_frame(snapshots: list[dict], primary_model_choice: str) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    errors = []
    for snapshot in snapshots:
        try:
            rows.append(snapshot_to_ranking_row(snapshot, primary_model_choice))
        except Exception as exc:
            errors.append(f"{snapshot.get('symbol', 'Unknown')}: {exc}")
    return pd.DataFrame(rows), errors


def _has_forecast(result: ForecastResult | dict) -> bool:
    if isinstance(result, ForecastResult):
        return result.forecast is not None and not result.forecast.empty
    return bool(result and result.get("forecast"))


def _metric(result: ForecastResult | dict, metric_name: str, default=np.inf):
    if isinstance(result, ForecastResult):
        return result.metrics.get(metric_name, default)
    return (result.get("metrics") or {}).get(metric_name, default)
