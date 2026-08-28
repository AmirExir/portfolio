from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from agent.forecast import ForecastResult
except ImportError:  # pragma: no cover - package import path
    from .agent.forecast import ForecastResult


CACHE_VERSION = 9
MODEL_RESULT_CACHE_VERSION = 8
SHADOW_MODEL_NAMES = frozenset({"RL Policy", "LSTM", "Transformer"})
FIXED_LIVE_CHAMPION_ORDER = (
    "Ensemble",
    "Ridge",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
)


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


def _safe_float(value, default=np.nan) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except Exception:
        return default


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
    include_rl_policy: bool = False,
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
        "include_rl_policy": bool(include_rl_policy),
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
    include_rl_policy: bool = False,
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
        include_rl_policy=include_rl_policy,
    )
    return output_path / f"ml_forecast_rankings_cache_{cache_key}.json"


def frame_fingerprint(frame: pd.DataFrame | pd.Series | None) -> dict:
    if frame is None or frame.empty:
        return {"rows": 0, "start": None, "end": None, "hash": "empty"}

    normalized = frame.copy()
    if isinstance(normalized, pd.Series):
        normalized = normalized.to_frame("value")
    normalized = normalized.sort_index()
    normalized.index = pd.to_datetime(normalized.index, errors="coerce")
    normalized = normalized[~normalized.index.isna()]
    normalized = normalized.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if normalized.empty:
        return {"rows": 0, "start": None, "end": None, "hash": "empty"}

    hash_values = pd.util.hash_pandas_object(normalized.round(10), index=True).values.tobytes()
    digest = hashlib.sha256(hash_values).hexdigest()[:16]
    return {
        "rows": int(len(normalized)),
        "start": normalized.index.min().isoformat(),
        "end": normalized.index.max().isoformat(),
        "hash": digest,
    }


def model_result_cache_path(
    output_dir: str | Path,
    symbol: str,
    history_days: int,
    horizon_days: int,
    lookback_window: int,
    ridge_alpha: float,
    optimize_model: bool,
    use_market_context: bool,
    sequence_model: str,
    data_fingerprint: dict,
    context_fingerprint: dict | None = None,
    include_rl_policy: bool = False,
) -> Path:
    payload = {
        "cache_version": MODEL_RESULT_CACHE_VERSION,
        "symbol": str(symbol),
        "history_days": int(history_days),
        "horizon_days": int(horizon_days),
        "lookback_window": int(lookback_window),
        "ridge_alpha": float(ridge_alpha),
        "optimize_model": bool(optimize_model),
        "use_market_context": bool(use_market_context),
        "sequence_model": str(sequence_model or "off"),
        "data_fingerprint": data_fingerprint,
        # Earnings/event identity remains relevant when broad market-context
        # features are disabled.
        "context_fingerprint": context_fingerprint,
        "include_rl_policy": bool(include_rl_policy),
    }
    raw = json.dumps(payload, sort_keys=True, default=_json_safe).encode("utf-8")
    cache_key = hashlib.sha256(raw).hexdigest()[:20]
    return Path(output_dir) / "model_artifacts" / f"model_results_{cache_key}.json"


def cache_payload_fresh(payload: dict, max_age_days: float | None = None) -> bool:
    if not payload:
        return False
    if int(payload.get("model_result_cache_version", 0)) != MODEL_RESULT_CACHE_VERSION:
        return False
    if max_age_days is None or float(max_age_days) <= 0:
        return True

    generated_at = payload.get("generated_at")
    if not generated_at:
        return False
    try:
        generated_dt = dt.datetime.fromisoformat(str(generated_at).replace("Z", "+00:00"))
        if generated_dt.tzinfo is None:
            generated_dt = generated_dt.replace(tzinfo=dt.timezone.utc)
    except Exception:
        return False
    age = dt.datetime.now(dt.timezone.utc) - generated_dt.astimezone(dt.timezone.utc)
    return age <= dt.timedelta(days=float(max_age_days))


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
    """Select a live-eligible forecasting model.

    Shadow/research policies remain in snapshots for evaluation, but they can
    never become the champion model, even when explicitly requested or when
    their diagnostic metric appears better than a forecasting model.
    """
    preferred = (preferred or "").strip()
    if preferred in SHADOW_MODEL_NAMES:
        preferred = "Best Validation"

    def fixed_live_champion() -> str:
        """Return the pre-registered champion without reusing holdout scores."""
        eligible = {
            name
            for name, result in model_results.items()
            if _has_forecast(result) and _is_live_eligible(name, result)
        }
        for name in FIXED_LIVE_CHAMPION_ORDER:
            if name in eligible:
                return name
        return sorted(eligible)[0] if eligible else ""

    if preferred == "Best Validation":
        return fixed_live_champion()

    if (
        preferred
        and preferred in model_results
        and _has_forecast(model_results[preferred])
        and _is_live_eligible(preferred, model_results[preferred])
    ):
        return preferred

    return fixed_live_champion()


def snapshot_from_model_results(
    symbol: str,
    last_price: float,
    model_results: dict[str, ForecastResult],
    smart_policy: dict | None = None,
    as_of_session: object | None = None,
    data_cutoff_utc: object | None = None,
) -> dict:
    return {
        "symbol": symbol,
        "last_price": float(last_price),
        "as_of_session": (
            pd.Timestamp(as_of_session).date().isoformat()
            if as_of_session is not None
            else None
        ),
        "data_cutoff_utc": (
            pd.Timestamp(data_cutoff_utc).isoformat()
            if data_cutoff_utc is not None
            else None
        ),
        "models": {name: forecast_result_to_dict(result) for name, result in model_results.items()},
        "smart_policy": _json_safe(smart_policy or {}),
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
    target_session = pd.Timestamp(forecast_rows.index[-1]).date().isoformat()
    smart_policy = snapshot.get("smart_policy") or {}

    def _return_for(model_name: str):
        model_payload = model_payloads.get(model_name)
        if not model_payload:
            return np.nan
        return float((model_payload.get("metrics") or {}).get("forecast_change_pct", np.nan))

    return {
        "Symbol": snapshot.get("symbol", ""),
        "As Of Session": snapshot.get("as_of_session"),
        "Target Session": target_session,
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
        "RL Policy Return %": _return_for("RL Policy"),
        "RL Mode": "Shadow" if "RL Policy" in model_payloads else "Off",
        "RL Shadow Action": _shadow_metric(model_payloads, "rl_action", "hold"),
        "RL Shadow Target %": _safe_float(
            _shadow_metric(model_payloads, "rl_target_weight", 0.0),
            0.0,
        )
        * 100.0,
        "RL Shadow Visits": _safe_float(
            _shadow_metric(model_payloads, "rl_state_visits", 0.0),
            0.0,
        ),
        "Ensemble Return %": _return_for("Ensemble"),
        "Probability Up %": probability_up,
        "Probability Down %": 100.0 - probability_up,
        "Directional Probability %": confidence,
        "Model Edge %": edge,
        "Signal Quality": signal_quality(confidence),
        "Expected Error %": expected_error,
        "Validation MAE %": float(metrics.get("holdout_mae_pct", np.nan)),
        "Zero-Return MAE %": _safe_float(
            metrics.get("zero_return_mae_pct")
        ),
        "MAE Skill Score": _safe_float(metrics.get("mae_skill_score")),
        "Direction Hit Rate %": float(metrics.get("holdout_direction_accuracy", np.nan)),
        "Direction Baseline Accuracy %": _safe_float(
            metrics.get("direction_baseline_accuracy_pct")
        ),
        "Direction Skill %": _safe_float(metrics.get("direction_skill_pct")),
        "Validation Samples": _safe_float(metrics.get("holdout_samples"), 0.0),
        "Nonoverlapping Validation Samples": _safe_float(
            metrics.get("holdout_nonoverlapping_samples"),
            0.0,
        ),
        "Effective Validation Samples": _safe_float(
            metrics.get("holdout_effective_samples"),
            0.0,
        ),
        "Validation Overlap Stride Sessions": _safe_float(
            metrics.get("holdout_overlap_stride_sessions"),
            0.0,
        ),
        "Validation Is OOS": metrics.get("validation_is_oos") is True,
        "Calibration Error %": _safe_float(metrics.get("calibration_error_pct")),
        "Brier Score": _safe_float(metrics.get("brier_score")),
        "Brier Baseline Score": _safe_float(
            metrics.get("probability_baseline_brier_score")
        ),
        "Brier Skill Score": _safe_float(metrics.get("brier_skill_score")),
        "Postprocessor Version": str(
            metrics.get("postprocessor_version", "") or ""
        ),
        "Forecast Outlier": bool(metrics.get("forecast_outlier", False)),
        "Score": score,
        "Smart Policy": smart_policy.get("policy_call", ""),
        "Policy Score": _safe_float(smart_policy.get("policy_score")),
        "Policy Target %": _safe_float(smart_policy.get("policy_target_pct")),
        "Policy Reason": smart_policy.get("policy_reason", ""),
        "Policy Forecast Score": _safe_float(smart_policy.get("forecast_component")),
        "Policy Trend Score": _safe_float(smart_policy.get("trend_component")),
        "Policy Momentum Score": _safe_float(smart_policy.get("momentum_component")),
        "Policy Earnings Score": _safe_float(smart_policy.get("earnings_component")),
        "Earnings Event": bool(smart_policy.get("earnings_event_flag", False)),
        "Earnings Score": _safe_float(smart_policy.get("earnings_event_score"), 0.0),
        "Earnings Confidence %": _safe_float(
            smart_policy.get("earnings_confidence"),
            0.0,
        )
        * 100.0,
        "Earnings Summary": smart_policy.get("earnings_summary", ""),
        "Earnings Reported At": smart_policy.get(
            "earnings_reported_at",
            "",
        ),
        "Earnings Effective Session": smart_policy.get(
            "earnings_effective_session",
            "",
        ),
        "Earnings Stale": bool(
            smart_policy.get("earnings_is_stale", False)
        ),
        "Earnings Policy Eligible": bool(
            smart_policy.get("earnings_policy_eligible", False)
        ),
        "Earnings Blockers": list(
            smart_policy.get("earnings_blockers", [])
        ),
        "Earnings Data Quality": list(
            smart_policy.get("earnings_data_quality_flags", [])
        ),
        "Earnings Calendar Source": smart_policy.get(
            "earnings_calendar_source",
            "",
        ),
        "Earnings Error": smart_policy.get("earnings_error_code", ""),
        "Policy Allocation Eligible": bool(smart_policy.get("allocation_eligible", False)),
        "Policy Allocation Blockers": list(smart_policy.get("allocation_blockers", [])),
        "Policy Lower Bound %": _safe_float(smart_policy.get("lower_confidence_bound_pct")),
        "Policy Agreeing Models": _safe_float(smart_policy.get("agreeing_models"), 0.0),
        "Policy RL Score": 0.0,
        "Policy RL Action": "",
        "Policy Volatility %": _safe_float(smart_policy.get("annual_volatility_pct")),
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
    if hasattr(result, "forecast"):
        forecast = getattr(result, "forecast")
        return forecast is not None and not forecast.empty
    return bool(result and result.get("forecast"))


def _metric(result: ForecastResult | dict, metric_name: str, default=np.inf):
    if hasattr(result, "metrics"):
        return getattr(result, "metrics").get(metric_name, default)
    return (result.get("metrics") or {}).get(metric_name, default)


def _is_live_eligible(name: str, result: ForecastResult | dict) -> bool:
    if name in SHADOW_MODEL_NAMES:
        return False
    metrics = (
        getattr(result, "metrics")
        if hasattr(result, "metrics")
        else (result.get("metrics") or {})
    )
    return not bool(metrics.get("shadow_mode")) and metrics.get("live_eligible", True) is not False


def _shadow_metric(model_payloads: dict, metric_name: str, default):
    payload = model_payloads.get("RL Policy") or {}
    return (payload.get("metrics") or {}).get(metric_name, default)
