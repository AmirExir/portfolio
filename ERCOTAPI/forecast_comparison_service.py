"""Run the existing load model against a contemporaneous official ERCOT vintage."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import math
import os
from pathlib import Path
from typing import Any, Callable

import pandas as pd

if __package__:
    from .actual_load_report import fetch_actual_load_history
    from .forecast_comparison import ForecastComparisonStore, compare_forecasts
    from .load_forecast import forecast_load, train_load_forecast_model, validate_hourly_load
else:  # Direct Streamlit entry-point imports.
    from actual_load_report import fetch_actual_load_history
    from forecast_comparison import ForecastComparisonStore, compare_forecasts
    from load_forecast import forecast_load, train_load_forecast_model, validate_hourly_load


CENTRAL = "America/Chicago"


def utc_now() -> datetime:
    """Return the real issuance clock, separate from the latest observation."""
    return datetime.now(timezone.utc)


def comparison_store_path() -> Path:
    """Return a configurable server-local ledger outside the source checkout."""
    override = os.getenv("ERCOT_COMPARISON_STORE")
    if override:
        return Path(override).expanduser()
    root = Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return root / "ercot-grid-dashboard" / "forecast-comparisons.sqlite"


def _aware_timestamp(value: Any, label: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is None:
            raise ValueError("missing timezone")
        return stamp.tz_convert("UTC")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{label} must be a valid timestamp with an explicit timezone.") from exc


def _check_official_vintage(official: dict[str, Any], *, now: datetime) -> None:
    if (not isinstance(official, dict) or official.get("ok") is not True
            or official.get("units") != "MW" or official.get("forecast_kind") != "official_ercot_np3_561_cd"):
        raise ValueError("A successful official NP3-561-CD forecast in MW is required.")
    published = _aware_timestamp(official.get("published_at"), "ERCOT publication time")
    retrieved = _aware_timestamp(official.get("generated_at"), "ERCOT retrieval time")
    current = _aware_timestamp(now, "Current time")
    if not published <= retrieved <= current:
        raise ValueError("ERCOT publication and retrieval must precede the model comparison.")
    if (current - published).total_seconds() > 3 * 3600:
        raise ValueError("The official forecast vintage is no longer fresh; refresh it before comparing.")


def prepare_model_history(bundle: dict[str, Any], *, now: datetime | None = None) -> pd.DataFrame:
    """Require real consecutive hourly observations, completed and recently available."""
    if not isinstance(bundle, dict) or not isinstance(bundle.get("history"), pd.DataFrame):
        raise ValueError("Actual-load history must contain an hourly observation dataframe.")
    metadata = bundle.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Actual-load history requires source and cutoff metadata.")
    if metadata.get("units", "MW") != "MW":
        raise ValueError("Actual-load inputs must be expressed in MW.")
    frame = validate_hourly_load(bundle["history"])
    if frame.index.tz is None:
        raise ValueError("Actual load needs explicit timezone offsets for a forecast comparison.")
    if pd.api.types.is_bool_dtype(bundle["history"]["load"]):
        raise ValueError("Actual loads must be numeric MW rather than boolean values.")
    frame.index = frame.index.tz_convert(CENTRAL)
    if not (frame.index.tz_convert("UTC") == frame.index.tz_convert("UTC").floor("h")).all():
        raise ValueError("Actual-load timestamps must identify exact hourly interval starts.")
    if len(frame) < 48:
        raise ValueError("At least 48 complete actual-load hours are required to train the model.")
    cutoff = frame.index[-1].tz_convert("UTC") + pd.Timedelta(hours=1)
    current = _aware_timestamp(now or utc_now(), "Current time")
    age = (current - cutoff).total_seconds() / 3600
    if age < 0:
        raise ValueError("Actual-load history includes an hour that has not finished.")
    if age > 4:
        raise ValueError(f"Latest complete actual load is {age:.1f} hours old; a current model comparison was not generated.")
    observed = metadata.get("retrieved_at") or metadata.get("generated_at")
    if observed is not None:
        observed_at = _aware_timestamp(observed, "Actual-load observation time")
        if not cutoff <= observed_at <= current:
            raise ValueError("Actual-load observation time must follow the completed intervals and precede the comparison.")
    declared = metadata.get("latest_interval_end_utc")
    if declared is not None and _aware_timestamp(declared, "Actual-load cutoff") != cutoff:
        raise ValueError("Actual-load cutoff metadata disagrees with its hourly observations.")
    return frame


def build_current_comparison(
    official_report: dict[str, Any], actual_bundle: dict[str, Any], *,
    tuning_mode: str = "auto", manual_params: dict[str, Any] | None = None,
    trainer: Callable[..., tuple] = train_load_forecast_model,
) -> dict[str, Any]:
    """Fit existing models and align only upcoming hours from one saved vintage.

    Missing recent observations are not filled. The recursive model predicts
    the short unobserved bridge before the next complete target hour; those
    bridge predictions are excluded from the comparison itself.
    """
    initial = utc_now()
    history = prepare_model_history(actual_bundle, now=initial)
    _check_official_vintage(official_report, now=initial)
    fitted = trainer(history, tuning_mode=tuning_mode, manual_params=manual_params)
    model, scaler, _, _, _, metrics = fitted
    if model is None or scaler is None or not isinstance(metrics, dict):
        raise ValueError("The selected history did not produce a fitted load model.")
    prediction_started = utc_now()
    history = prepare_model_history(actual_bundle, now=prediction_started)
    _check_official_vintage(official_report, now=prediction_started)
    cutoff = history.index[-1].tz_convert("UTC") + pd.Timedelta(hours=1)
    # A prospective snapshot must be saved strictly before its first target.
    # At an exact hour, that hour has already begun; always choose the next one.
    future_start = _aware_timestamp(prediction_started, "Prediction start").floor("h") + pd.Timedelta(hours=1)
    future_end = future_start + pd.Timedelta(hours=24)
    steps = math.ceil((future_end - cutoff).total_seconds() / 3600)
    prediction = forecast_load(model, scaler, history, hours=steps)
    prediction = prediction.loc[(prediction.index >= future_start) & (prediction.index < future_end)]
    if len(prediction) != 24:
        raise ValueError("The model did not produce the 24 requested future hourly intervals.")
    issued = utc_now()
    if _aware_timestamp(issued, "Model issue time") < _aware_timestamp(prediction_started, "Prediction start"):
        raise ValueError("The issue clock moved backwards while generating the model forecast.")
    if pd.Timestamp(issued) >= future_start:
        raise ValueError("An hourly boundary passed while generating the model forecast; retry for 24 fully future hours.")
    prepare_model_history(actual_bundle, now=issued)
    _check_official_vintage(official_report, now=issued)
    metadata = copy.deepcopy(metrics)
    metadata.update({
        "model_data_cutoff_utc": cutoff.isoformat(),
        "source_metadata": copy.deepcopy(actual_bundle["metadata"]),
        "recursive_steps": steps,
        "unobserved_bridge_hours": int((future_start - cutoff).total_seconds() / 3600),
        "history_hours": len(history),
        "calendar_timezone": CENTRAL,
        "target_semantics": "hourly system-wide load MW; timestamps identify interval starts",
    })
    snapshot = compare_forecasts(
        prediction, official_report, model_issued_at=issued,
        model_data_cutoff_utc=cutoff.to_pydatetime(), model_metadata=metadata,
        now=utc_now(),
    )
    if snapshot["summary"]["matched_hours"] != 24:
        raise ValueError("The official vintage and model do not cover all 24 future hourly intervals; refresh and retry.")
    return snapshot


def run_and_save_comparison(
    official_report: dict[str, Any], *, actual_bundle: dict[str, Any] | None = None,
    history_days: int = 30, tuning_mode: str = "auto",
    manual_params: dict[str, Any] | None = None,
    store: ForecastComparisonStore | None = None,
    trainer: Callable[..., tuple] = train_load_forecast_model,
) -> dict[str, Any]:
    """Record a prospective run and evaluate saved runs on observed actual load."""
    bundle = actual_bundle or fetch_actual_load_history(history_days=history_days)
    snapshot = build_current_comparison(official_report, bundle, tuning_mode=tuning_mode,
                                        manual_params=manual_params, trainer=trainer)
    ledger = store or ForecastComparisonStore(comparison_store_path())
    observed_at = bundle["metadata"].get("retrieved_at") or bundle["metadata"].get("generated_at")
    ledger.append_actuals(bundle["history"], observed_at=observed_at,
                          source_metadata=bundle["metadata"])
    identifier = ledger.append_snapshot(snapshot)
    return {
        "snapshot_id": identifier, "snapshot": snapshot,
        "evaluation": ledger.evaluate_snapshot(identifier),
        "recent_evaluations": [
            {"snapshot_id": saved["snapshot_id"], "model_issued_at": saved["model_issued_at"],
             **ledger.evaluate_snapshot(saved["snapshot_id"])}
            for saved in ledger.list_snapshots(limit=5)
        ],
    }
