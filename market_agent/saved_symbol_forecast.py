"""Read-only presentation of a previously calculated single-symbol forecast."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from math import isfinite
from pathlib import Path
import re

import pandas as pd
import streamlit as st

if __package__:
    from .forecast_cache import forecast_result_from_dict, snapshots_to_ranking_frame
    from .forecast_settings import tuning_results_table
    from .heatmaps import format_signed_return
    from .report_dates import format_report_timestamp, format_session_date
    from .report_freshness import parse_report_timestamp
    from .saved_rankings import dashboard_result_path, scheduled_rows_for_symbols, stored_ranking_rows
else:  # Streamlit script entrypoint
    from forecast_cache import forecast_result_from_dict, snapshots_to_ranking_frame
    from forecast_settings import tuning_results_table
    from heatmaps import format_signed_return
    from report_dates import format_report_timestamp, format_session_date
    from report_freshness import parse_report_timestamp
    from saved_rankings import dashboard_result_path, scheduled_rows_for_symbols, stored_ranking_rows


@dataclass
class SavedSymbolForecast:
    """A historical forecast and the source that produced it."""

    row: dict
    generated_at: datetime
    source: str
    models: dict


def historical_accuracy_path(
    directory: str, config: dict, symbol: str, model: str, test_points: int,
) -> Path | None:
    """Identify a walk-forward test by the fitted inputs and both price series fingerprints."""
    prices = config.get("data_fingerprint") or {}
    market = config.get("context_fingerprint") or {}
    if isinstance(market, dict) and "market" in market:
        market = market.get("market") or {}
    use_context = config.get("use_market_context")
    if (
        not isinstance(prices, dict) or not prices.get("hash") or prices["hash"] == "empty"
        or not prices.get("end") or not isinstance(use_context, bool)
        or (use_context and (not isinstance(market, dict) or not market.get("hash")))
    ):
        return None
    try:
        parameters = (
            int(config["history_days"]), int(config["horizon_days"]),
            int(config["lookback_window"]), float(config["ridge_alpha"]), use_context,
        )
    except (KeyError, ValueError, OverflowError, TypeError):
        return None
    return dashboard_result_path(directory, "walk_forward_v1", (
        symbol, model, test_points, *parameters,
        prices["end"], prices["hash"], market.get("hash") if use_context else None,
    ))


def render_historical_accuracy(history: pd.DataFrame, generated_at: object = None) -> None:
    """Show the dated walk-forward predictions and the prices observed at their targets."""
    with st.expander("Previous forecast accuracy"):
        stamp = parse_report_timestamp(generated_at)
        if stamp:
            st.caption(f"Walk-forward test completed {format_report_timestamp(stamp)}")
        st.caption("Each forecast uses the history available on the date shown.")
        display = history.copy()
        for date in ("as_of_date", "forecast_date"):
            if date in display:
                display[date] = pd.to_datetime(display[date], errors="coerce").dt.date
        st.dataframe(display.rename(columns={
            "as_of_date": "Forecast Made On", "forecast_date": "Forecast Target Date",
            "as_of_price": "Price Then", "forecast_close": "Forecast Close", "actual_close": "Actual Close",
            "predicted_change_pct": "Forecast Return %", "actual_change_pct": "Actual Return %",
            "error_pct": "Error %", "abs_error_pct": "Absolute Error %", "confidence_pct": "Confidence %",
            "expected_error_pct": "Expected Error %", "direction_correct": "Direction Correct",
        }))


@st.cache_resource(max_entries=1, show_spinner=False)
def _archived_snapshots(path: Path, modified_ns: int) -> dict:
    """Index the referenced full local run once, invalidating it if the file changes."""
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict) or report.get("run_complete") is False:
        return {}
    return {
        (part.get("horizon_days"), snap["symbol"]): snap
        for part in (report, *(report.get("short_horizon_reports") or []))
        if isinstance(part, dict) and part.get("run_complete") is not False
        for snap in (part.get("snapshots") or []) if isinstance(snap, dict) and isinstance(snap.get("symbol"), str)
    }


def attach_archived_models(
    saved: SavedSymbolForecast, report: dict, directory: str, symbol: str, horizon: int,
) -> None:
    """Attach the forecast path only when the local run matches the published endpoint."""
    if saved.models or saved.source != "Scheduled report":
        return
    name = Path(str((report.get("paths") or {}).get("json") or "")).name
    if not re.fullmatch(r"ml_forecast_rankings_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z\.json", name):
        return
    path = Path(directory) / name
    try:
        snapshot = _archived_snapshots(path, path.stat().st_mtime_ns).get((horizon, symbol), {})
    except (OSError, ValueError):
        return
    models = snapshot.get("models") or {}
    selected = models.get(saved.row.get("Selected Model")) or {}
    values = selected.get("forecast") or []
    if not values or not isinstance(values[-1], dict):
        return
    end = values[-1]
    if (
        snapshot.get("as_of_session") == saved.row.get("As Of Session")
        and end.get("date") == saved.row.get("Target Session")
        and end.get("forecast_close") == saved.row.get("Forecast Price")
    ):
        saved.models = models


def _symbol_snapshot(report: dict, symbol: str) -> dict:
    snapshots = report.get("snapshots") or [report.get("snapshot")]
    return next((s for s in snapshots if isinstance(s, dict) and s.get("symbol") == symbol), {})


def choose_saved_forecast(
    local: dict, scheduled: dict | None, symbol: str, alias: str, horizon: int, primary: str,
) -> SavedSymbolForecast | None:
    """Choose a matching, dated result by price session, then generation time."""
    symbols = (symbol,)
    scheduled = scheduled or {}
    local_rows = stored_ranking_rows(local, symbols, horizon, primary)
    snapshot = _symbol_snapshot(local, symbol)
    if (
        local_rows.empty and local.get("horizon_days") == horizon and local.get("run_complete") is not False
        and local.get("symbols", [local.get("symbol")]) == [symbol] and snapshot
    ):
        local_rows, _ = snapshots_to_ranking_frame([snapshot], primary)
        if not local_rows.empty:
            # Legacy ranking construction supplies defaults for absent metrics. Only
            # show a stored return or a derived call when its inputs actually exist.
            model = local_rows.iloc[0]["Selected Model"]
            metrics = snapshot["models"][model].get("metrics") or {}
            for column, metric in (
                ("Forecast Return %", "forecast_change_pct"), ("Probability Up %", "probability_up_pct"),
                ("Expected Error %", "expected_error_pct"), ("Validation MAE %", "holdout_mae_pct"),
            ):
                local_rows[column] = metrics.get(metric)
            if any(pd.isna(metrics.get(k)) for k in ("forecast_change_pct", "confidence_pct", "expected_error_pct")):
                local_rows["Model Call"] = None
    candidates = []
    sources = [("Interactive run", local, local_rows)]
    if scheduled.get("run_complete") is not False:
        for report in (scheduled, *(scheduled.get("short_horizon_reports") or [])):
            if isinstance(report, dict) and report.get("horizon_days") == horizon:
                frame = scheduled_rows_for_symbols({**report, "short_horizon_reports": []}, symbols, horizon, {symbol: alias})
                sources.append(("Scheduled report", report, frame))
    for name, report, frame in sources:
        if frame.empty:
            continue
        row = frame.iloc[0].to_dict()
        snapshot = _symbol_snapshot(report, symbol)
        as_of = row.get("As Of Session")
        if pd.isna(as_of):
            as_of = snapshot.get("as_of_session")
        if pd.isna(as_of) and name == "Interactive run":
            as_of = (local.get("data_fingerprint") or {}).get("end")
        price = pd.to_numeric(row.get("Forecast Price"), errors="coerce")
        generated = report.get("generated_at")
        if name == "Scheduled report" and not generated:
            generated = scheduled.get("generated_at")
        timestamp = parse_report_timestamp(generated)
        if not timestamp or pd.isna(price) or not 0 < price < float("inf"):
            continue
        session = pd.to_datetime(as_of, errors="coerce")
        row["As Of Session"] = session.date().isoformat() if pd.notna(session) else None
        candidates.append((
            row["As Of Session"] or "", timestamp,
            SavedSymbolForecast(row, timestamp, name, snapshot.get("models") or {}),
        ))
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate[:2])[2]


def render_saved_symbol_forecast(result: SavedSymbolForecast, symbol: str, horizon: int) -> None:
    """Display dated output from the saved run without fitting or touching trading state."""
    row = result.row
    model = str(row.get("Selected Model") or "Unavailable")
    st.subheader(f"{symbol} · Saved {horizon}-session forecast")
    st.caption(
        f"{result.source} · {format_report_timestamp(result.generated_at)} · Model {model} · "
        f"Prices as of {format_session_date(row.get('As Of Session'))} · "
        f"Target {format_session_date(row.get('Target Session'))}"
    )
    metrics = st.columns(3)
    quote = float(row["Forecast Price"])
    metrics[0].metric("Forecast price", f"${quote:,.2f}" if quote >= 1 else f"${quote:.5g}")
    metrics[1].metric("Forecast return", format_signed_return(row.get("Forecast Return %")))
    metrics[2].metric("Saved model call", row.get("Model Call") or "Unavailable")

    selected = result.models.get(model) or {}
    if selected.get("forecast"):
        path = forecast_result_from_dict(selected).forecast
        if "forecast_close" in path and isinstance(path.index, pd.DatetimeIndex):
            with st.expander(f"{symbol} · {horizon}-session forecast path", expanded=True):
                draw = st.scatter_chart if path["forecast_close"].count() == 1 else st.line_chart
                draw(path["forecast_close"].rename("Forecast price"), x_label="Target session", y_label="Forecast price (USD)")
                st.dataframe(path, use_container_width=True)

    with st.expander(f"{symbol} · {horizon}-session model details"):
        diagnostics = []
        for name in ("Probability Up", "Expected Error", "Validation MAE"):
            value = pd.to_numeric(row.get(name + " %"), errors="coerce")
            if pd.notna(value) and isfinite(value):
                diagnostics.append(f"{name} {value:.2f}%")
        if diagnostics:
            st.caption(" · ".join(diagnostics))
        comparable = {
            col.removesuffix(" Return %"): format_signed_return(value)
            for col, value in row.items()
            if col.endswith(" Return %") and col != "Forecast Return %" and pd.notna(value)
        }
        if comparable:
            st.dataframe(pd.DataFrame(comparable.items(), columns=["Model", "Forecast return"]),
                         hide_index=True, use_container_width=True)
        settings = tuning_results_table(result.models, model) if result.models else pd.DataFrame()
        if not settings.empty:
            st.dataframe(settings, hide_index=True, use_container_width=True)
        else:
            st.caption("Fitted model settings are unavailable for this saved result.")
