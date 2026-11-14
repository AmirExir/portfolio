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
        f"{method} is an experimental recursive forecast using past load and calendar features; it does not use weather inputs.",
        "Forecasts are estimates, not actual load or a grid-emergency prediction.",
        f"Source: {report['source_product_url']}",
        f"Exact ERCOT archive: {report['source_url']}",
    ]
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
        text, caption = format_comparison_report(result, comparison)
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
