"""Prepare concise, named-method load comparisons for report delivery."""

from __future__ import annotations

import copy
from datetime import datetime
import logging
import sqlite3
from typing import Any
from zoneinfo import ZoneInfo

from .forecast_comparison import validate_comparison_snapshot
from .forecast_comparison_service import run_and_save_comparison
from .forecast_labels import model_display_name
from .load_forecast import WeatherInputError
from .weather_forecast import WeatherError


CENTRAL = ZoneInfo("America/Chicago")


def _local_time(value: str) -> str:
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(CENTRAL)
    return f"{stamp:%a, %b} {stamp.day}, {stamp.year} at {stamp.hour % 12 or 12}:{stamp.minute:02d} {stamp:%p %Z}"


def _peak_text(rows: list[dict[str, Any]], key: str) -> str:
    peak = max(rows, key=lambda row: row[key])
    start = datetime.fromisoformat(peak["interval_start_utc"]).astimezone(CENTRAL)
    end = datetime.fromisoformat(peak["interval_end_utc"]).astimezone(CENTRAL)
    end_label = f"{end.hour % 12 or 12}:{end.minute:02d} {end:%p %Z}"
    if end.date() != start.date():
        end_label += " (next day)"
    return f"{peak[key]:,.0f} MW · {_local_time(peak['interval_start_utc'])}–{end_label}"


def _score_summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Keep delivery bounded; full evidence remains in the local ledger."""
    fields = ("snapshot_id", "status", "as_of", "model_issued_at", "ercot_published_at", "matched_count",
              "pending_count", "missing_actual_count", "model_metrics", "ercot_metrics")
    return {key: copy.deepcopy(evaluation[key]) for key in fields if key in evaluation}


def _method_note(method: str, metadata: dict[str, Any]) -> str:
    weather = metadata.get("weather", {})
    if weather.get("status") == "used":
        return (f"{method} is an experimental recursive forecast using past load, calendar features, "
                "and archived NWS temperature forecasts. Weather has not been established to improve accuracy.")
    note = f"{method} is an experimental recursive forecast using past load and calendar features; it does not use weather inputs."
    if weather.get("status") == "collecting":
        note += " Weather forecasts are being archived for evaluation; they are not yet model inputs."
    elif weather.get("status") == "collection_failed":
        note += " Weather collection failed this run; check the local logs."
    return note


def _completed_score(evaluations: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select one fully observed run without pooling overlapping forecasts."""
    completed = [row for row in evaluations if row.get("status") == "complete"
                 and row.get("matched_count") == 24 and row.get("pending_count") == 0
                 and row.get("missing_actual_count") == 0
                 and row.get("model_metrics") and row.get("ercot_metrics")]
    if not completed:
        return None
    latest = max(completed, key=lambda row: datetime.fromisoformat(row["model_issued_at"]))
    summary = _score_summary(latest)
    metadata = latest.get("model_metadata", {})
    summary["model_name"] = model_display_name(metadata.get("model_type", ""))
    summary["uses_weather"] = metadata.get("weather", {}).get("status") == "used"
    return summary


def format_comparison_report(report: dict[str, Any], comparison: dict[str, Any]) -> tuple[str, str]:
    """Return readable report text and a caption for one matched-hour chart."""
    snapshot = validate_comparison_snapshot(comparison["snapshot"])
    rows, summary = snapshot["rows"], snapshot["summary"]
    if len(rows) != 24 or summary["matched_hours"] != 24:
        raise ValueError("Telegram comparison requires exactly 24 matching future hours.")
    method = model_display_name(snapshot["model_metadata"]["model_type"])
    text = [
        f"ERCOT vs {method} Load Forecast",
        "24 matching future hours · system-wide demand · MW",
        f"From {_local_time(rows[0]['interval_start_utc'])}",
        f"Through {_local_time(rows[-1]['interval_end_utc'])}",
        "",
        f"ERCOT published: {_local_time(snapshot['ercot_published_at'])}",
        f"{method} issued: {_local_time(snapshot['model_issued_at'])}",
        f"Actual-load inputs complete through: {_local_time(snapshot['model_data_cutoff_utc'])}",
        "",
        f"ERCOT peak: {_peak_text(rows, 'ercot_mw')}",
        f"{method} peak: {_peak_text(rows, 'model_mw')}",
        f"Average {method} − ERCOT: {summary['mean_difference_mw']:+,.0f} MW",
        f"Largest absolute hourly difference: {summary['maximum_absolute_difference_mw']:,.0f} MW",
        "",
        "Forecast differences show disagreement, not accuracy. This run awaits actual load.",
        _method_note(method, snapshot["model_metadata"]),
        "Forecasts are estimates, not actual load or a grid-emergency prediction.",
        f"Source: {report['source_product_url']}",
        f"Exact ERCOT archive: {report['source_url']}",
    ]
    score = comparison.get("latest_completed_evaluation")
    if score:
        feature_label = "with weather" if score["uses_weather"] else "load/calendar"
        text.extend([
            "", f"Recent completed run · issued {_local_time(score['model_issued_at'])}",
            f"24 observed hours · {score['model_name']} ({feature_label}) MAE: {score['model_metrics']['mae_mw']:,.0f} MW "
            f"· ERCOT MAE: {score['ercot_metrics']['mae_mw']:,.0f} MW",
            "One run only; forecast issue times differ. Saved scores do not automatically tune the model.",
        ])
    caption = f"Official ERCOT + {method}\n24 matching hourly intervals · system demand (MW) · Central time"
    return "\n".join(text), caption


def attach_model_comparison(report: dict[str, Any]) -> dict[str, Any]:
    """Attach a current saved comparison without sacrificing valid official text.

    A failed model comparison receives no substitute chart. Default official-only
    reporting remains a separate CLI mode. Exception details stay out of the
    delivered report; the failure type is logged for local diagnostics.
    """
    result = copy.deepcopy(report)
    if result.get("ok") is not True:
        return result
    result.pop("chart", None)
    try:
        generated = run_and_save_comparison(result)
        comparison = {
            "snapshot_id": generated["snapshot_id"],
            "snapshot": generated["snapshot"],
            "evaluation": _score_summary(generated["evaluation"]),
        }
        score = _completed_score(generated.get("recent_evaluations", []))
        if score is not None:
            comparison["latest_completed_evaluation"] = score
        text, caption = format_comparison_report(result, comparison)
    except (WeatherInputError, WeatherError) as exc:
        # These classes contain controlled, actionable messages; never expose
        # arbitrary database/source exception text through the generic handler.
        logging.getLogger(__name__).warning("ERCOT weather comparison unavailable: %s", exc)
        result["comparison_error"] = "weather_unavailable"
        result["chart_error"] = "comparison_unavailable"
        result["text"] += (
            f"\n\nWeather model comparison unavailable: {exc}\n"
            "Use weather collection mode to build forecast history; model mode requires complete archived coverage. "
            "The official ERCOT forecast remains available above; no alternate model or comparison image was substituted."
        )
        return result
    except (ValueError, OSError, RuntimeError, sqlite3.Error) as exc:
        logging.getLogger(__name__).error("ERCOT comparison unavailable (%s)", type(exc).__name__)
        result["comparison_error"] = "comparison_unavailable"
        result["chart_error"] = "comparison_unavailable"
        result["text"] += "\n\nModel comparison unavailable for this run. The official ERCOT forecast remains available above; no comparison image was substituted. Please retry /ercot_load."
        return result
    result["comparison"] = comparison
    result["text"] = text
    result["chart_caption"] = caption
    return result
