"""Read-only projections of saved dashboard scans and published market reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

if __package__:
    from .forecast_cache import snapshots_to_ranking_frame
else:
    from forecast_cache import snapshots_to_ranking_frame


def dashboard_result_path(directory: str | Path, name: str, settings: tuple) -> Path:
    """Identify a compact saved dashboard result by the settings used to compute it."""
    key = hashlib.sha256(json.dumps(settings, separators=(",", ":")).encode()).hexdigest()[:24]
    return Path(directory) / "dashboard_results" / f"{name}_{key}.json"


def _selected_rows(rows: object, symbols: Sequence[str], aliases: dict[str, str]) -> pd.DataFrame:
    if not isinstance(rows, list):
        return pd.DataFrame()
    lookup = {aliases.get(symbol, symbol): symbol for symbol in symbols}
    lookup.update({symbol: symbol for symbol in symbols})
    selected = []
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("Symbol"), str) and row["Symbol"] in lookup:
            selected.append({**row, "Symbol": lookup[row["Symbol"]]})
    return pd.DataFrame(selected).drop_duplicates("Symbol") if selected else pd.DataFrame()


def scheduled_rows_for_symbols(
    payload: dict | None, symbols: Sequence[str], horizon: int, aliases: dict[str, str],
) -> pd.DataFrame:
    """Read matching published forecast rows using the caller's provider-to-report aliases."""
    if not isinstance(payload, dict) or payload.get("run_complete") is False:
        return pd.DataFrame()
    for report in (payload, *(payload.get("short_horizon_reports") or [])):
        if (
            not isinstance(report, dict) or report.get("horizon_days") != horizon
            or report.get("run_complete") is False or (report.get("timings") or {}).get("run_complete") is False
        ):
            continue
        universe = report.get("symbols") or payload.get("symbols")
        requested = [symbol for symbol in symbols if not isinstance(universe, list) or symbol in universe]
        candidates = report.get("rows") or [*(report.get("top_buys") or []), *(report.get("top_sells") or [])]
        rows = _selected_rows(candidates, requested, aliases)
        if not rows.empty and "Forecast Return %" in rows:
            return rows
    return pd.DataFrame()


def stored_ranking_rows(payload: dict, symbols: Sequence[str], horizon: int, primary_model: str) -> pd.DataFrame:
    """Read a dashboard ranking at its exact configuration path, including older snapshot-only caches."""
    if not isinstance(payload, dict) or payload.get("run_complete") is False:
        return pd.DataFrame()
    universe = payload.get("symbols")
    if (
        not isinstance(universe, list) or not all(isinstance(symbol, str) for symbol in universe)
        or sorted(universe) != sorted(symbols) or payload.get("horizon_days") != horizon
        or payload.get("primary_model") != primary_model
    ):
        return pd.DataFrame()
    rows = _selected_rows(payload.get("rows"), symbols, {})
    if not rows.empty and "Forecast Return %" in rows:
        return rows
    # Old on-demand runs saved model snapshots before portfolio allocation.
    # Their per-symbol targets cannot be presented as final portfolio allocations.
    snapshots = payload.get("snapshots")
    if not isinstance(snapshots, list) or "snapshot" in payload:
        return pd.DataFrame()
    rows, _ = snapshots_to_ranking_frame(
        [row for row in snapshots if isinstance(row, dict) and row.get("symbol") in symbols], primary_model,
    )
    return rows.drop(columns=["Policy Target %"], errors="ignore")


def scheduled_pattern_rows(
    payload: dict | None, symbols: Sequence[str], aliases: dict[str, str],
) -> pd.DataFrame:
    """Recover published pattern labels and any explicitly recorded close, date and RSI."""
    if not isinstance(payload, dict) or payload.get("run_complete") is False:
        return pd.DataFrame()
    universe = payload.get("symbols")
    requested = [symbol for symbol in symbols if not isinstance(universe, list) or symbol in universe]
    details = {}
    columns = ("Symbol", "As Of Session", "Primary Pattern", "All Patterns", "Last Price")
    rows = _selected_rows(payload.get("rows"), requested, aliases)
    for row in rows.to_dict(orient="records"):
        labels = row.get("All Patterns")
        if isinstance(labels, str) and labels.strip():
            record = {column: row[column] for column in columns if column in row}
            record.setdefault("Primary Pattern", labels.split(", ")[0])
            details[row["Symbol"]] = record

    # Forecasts can fail for a symbol whose completed-close scan succeeded.
    for item in (payload.get("price_watch") or {}).get("observations") or []:
        if not isinstance(item, dict) or item.get("symbol") not in requested:
            continue
        technical = item.get("technical")
        if not isinstance(technical, dict) or not isinstance(technical.get("patterns"), list):
            continue
        symbol = item["symbol"]
        if symbol in details:
            if details[symbol].get("As Of Session") == item.get("as_of_session"):
                details[symbol]["RSI"] = technical.get("rsi")
            continue
        labels = [name for name in technical["patterns"] if isinstance(name, str) and name]
        if technical.get("trend") in ("Uptrend", "Downtrend"):
            labels.insert(0, technical["trend"])
        if technical.get("rsi_state") in ("Oversold", "Overbought"):
            labels.append(technical["rsi_state"])
        details[symbol] = {
            "Symbol": symbol, "As Of Session": item.get("as_of_session"), "Last Price": item.get("close"),
            "Primary Pattern": labels[0] if labels else "No named price setup",
            "All Patterns": ", ".join(dict.fromkeys(labels)), "RSI": technical.get("rsi"),
            "Pattern source": "Closing-price watch",
        }
    frame = pd.DataFrame(details.values())
    if "Pattern source" in frame:
        frame["Pattern source"] = frame["Pattern source"].fillna("Scheduled pattern scan")
    return frame
