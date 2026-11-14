"""Streamlit presentation for official forecasts and saved model comparisons."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

if __package__:
    from .actual_load_report import fetch_actual_load_history
    from .forecast_comparison import ForecastComparisonStore
    from .forecast_comparison_service import comparison_store_path, run_and_save_comparison
    from .forecast_labels import model_display_name
    from .load_forecast import HAS_XGBOOST, train_load_forecast_model
    from .load_forecast_report import fetch_forecast_report
else:  # Streamlit runs the entry point with ERCOTAPI itself on sys.path.
    from actual_load_report import fetch_actual_load_history
    from forecast_comparison import ForecastComparisonStore
    from forecast_comparison_service import comparison_store_path, run_and_save_comparison
    from forecast_labels import model_display_name
    from load_forecast import HAS_XGBOOST, train_load_forecast_model
    from load_forecast_report import fetch_forecast_report

CENTRAL = "America/Chicago"


def local_time(value: str) -> str:
    """Format a source instant with a date and explicit Central-time offset name."""
    return pd.Timestamp(value).tz_convert(CENTRAL).strftime("%a, %b %d, %Y · %I:%M %p %Z")


@st.cache_data(ttl=300, show_spinner=False, max_entries=3)
def official_forecast() -> dict[str, Any]:
    """Cache a verified public forecast vintage for five minutes."""
    return fetch_forecast_report()


@st.cache_data(ttl=300, show_spinner=False, max_entries=3)
def actual_history(days: int) -> dict[str, Any]:
    """Cache public actuals without transforming or filling their observations."""
    return fetch_actual_load_history(history_days=days)


@st.cache_resource(ttl=3600, max_entries=3, show_spinner=False)
def fitted_model(history: pd.DataFrame, tuning_mode: str, manual_params: dict[str, Any] | None) -> tuple:
    """Reuse the existing model fit for identical observations and settings."""
    return train_load_forecast_model(history, tuning_mode=tuning_mode, manual_params=manual_params)


def hourly_figure(rows: list[dict[str, Any]], series: list[tuple[str, str, str]]) -> go.Figure:
    """Plot interval-start MW steps with explicit Central-time ticks and hovers."""
    figure = go.Figure()
    if not rows:
        return figure
    rows = sorted(rows, key=lambda row: pd.Timestamp(row["interval_start_utc"]))
    starts = pd.to_datetime([row["interval_start_utc"] for row in rows], utc=True)
    ends = pd.to_datetime([row["interval_end_utc"] for row in rows], utc=True)
    labels = [f"{start.tz_convert(CENTRAL):%a, %b %d · %I:%M %p %Z} – {end.tz_convert(CENTRAL):%I:%M %p %Z}"
              for start, end in zip(starts, ends)]
    for key, name, color in series:
        values = [row.get(key) for row in rows]
        plot_times, plot_values, plot_labels = [], [], []
        for index, (start, end, value, label) in enumerate(zip(starts, ends, values, labels)):
            plot_times.append(start)
            plot_values.append(value)
            plot_labels.append(label)
            # End the step at its actual interval boundary. A missing hour
            # must be a visible gap, not an extension of the previous MW.
            next_value_missing = index + 1 < len(values) and values[index + 1] is None
            if index == len(starts) - 1 or end != starts[index + 1] or (value is not None and next_value_missing):
                plot_times.append(end)
                plot_values.append(value)
                plot_labels.append(label)
                if index < len(starts) - 1:
                    plot_times.append(end)
                    plot_values.append(None)
                    plot_labels.append(None)
        figure.add_trace(go.Scatter(
            x=plot_times, y=plot_values, mode="lines",
            name=name, line={"color": color, "width": 3, "shape": "hv"},
            customdata=plot_labels, connectgaps=False,
            hovertemplate="%{customdata}<br>%{y:,.0f} MW<extra>%{fullData.name}</extra>",
        ))
    stride = max(1, len(starts) // 7)
    ticks = starts[::stride]
    figure.update_layout(
        height=420, margin={"l": 15, "r": 15, "t": 20, "b": 25},
        legend={"orientation": "h", "y": 1.12}, hovermode="x",
        yaxis={"title": "System-wide demand (MW)", "tickformat": ",.0f"},
        xaxis={"title": "Hourly intervals · Central time", "tickvals": ticks,
               "ticktext": [f"{tick.tz_convert(CENTRAL):%b %d}<br>{tick.tz_convert(CENTRAL):%I %p %Z}" for tick in ticks]},
    )
    return figure


def accuracy_table(evaluations: list[dict[str, Any]]) -> pd.DataFrame:
    """Keep accuracy per saved run and expose pending/missing observations."""
    rows = []
    for result in evaluations:
        model, official = result["model_metrics"] or {}, result["ercot_metrics"] or {}
        rows.append({
            "Model method": model_display_name(result["model_metadata"].get("model_type", "")),
            "Model issued (Central)": local_time(result["model_issued_at"]),
            "ERCOT published (Central)": local_time(result["ercot_published_at"]),
            "Scored hours": result["matched_count"], "Future hours": result["pending_count"],
            "Awaiting actuals": result["missing_actual_count"],
            "Model MAE (MW)": model.get("mae_mw"), "ERCOT MAE (MW)": official.get("mae_mw"),
            "Model RMSE (MW)": model.get("rmse_mw"), "ERCOT RMSE (MW)": official.get("rmse_mw"),
            "Model bias (MW)": model.get("bias_mw"), "ERCOT bias (MW)": official.get("bias_mw"),
        })
    return pd.DataFrame(rows)


def _model_settings() -> tuple[int, str, dict[str, Any]]:
    with st.expander("Model settings", expanded=False):
        days = st.selectbox("Actual-load history", [30, 14, 7], format_func=lambda day: f"{day} days")
        mode = st.radio("Model tuning", ["auto", "manual"], horizontal=True,
                        format_func=lambda value: "Time-series cross-validation" if value == "auto" else "Manual settings")
        params = {}
        if mode == "manual":
            left, right = st.columns(2)
            params["n_estimators"] = left.slider("Trees", 50, 500, 200, 10)
            params["max_depth"] = right.slider("Maximum depth", 3, 30, 6 if HAS_XGBOOST else 15)
            if HAS_XGBOOST:
                params["learning_rate"] = left.slider("Learning rate", 0.01, 0.30, 0.10, 0.01)
                params["subsample"] = right.slider("Training row fraction", 0.50, 1.0, 0.80, 0.05)
                params["colsample_bytree"] = left.slider("Feature fraction", 0.50, 1.0, 0.80, 0.05)
                params["min_child_weight"] = right.slider("Minimum child weight", 1, 10, 1)
            else:
                params["min_samples_split"] = left.slider("Minimum split samples", 2, 20, 5)
                params["min_samples_leaf"] = right.slider("Minimum leaf samples", 1, 10, 1)
        st.caption("The existing model uses calendar features and past load. Weather is not an input. Missing actual hours are never filled.")
    return days, mode, params


def _render_actual_history(days: int) -> None:
    """Retain historical load charts and metrics using the same validated source."""
    with st.expander("Historical actual load", expanded=False):
        if st.button("Load / refresh actual-load history"):
            try:
                with st.spinner("Loading official actual-load history…"):
                    st.session_state["forecast_actual_history"] = {"days": days, **actual_history(days)}
            except (ValueError, OSError) as exc:
                st.error(f"Actual-load history unavailable: {exc}")
        bundle = st.session_state.get("forecast_actual_history")
        if not bundle or bundle["days"] != days:
            st.caption(f"Load {days} days of actual system demand, or run a model comparison above.")
            return
        history, metadata = bundle["history"], bundle["metadata"]
        st.caption(f"Official actual load · inputs complete through {local_time(metadata['latest_interval_end_utc'])} · Retrieved {local_time(metadata['retrieved_at'])}")
        for warning in metadata.get("warnings", []):
            st.warning(warning)
        latest, average, peak, factor = st.columns(4)
        latest.metric("Latest complete hour", f"{history['load'].iloc[-1]:,.0f} MW")
        average.metric("Average load", f"{history['load'].mean():,.0f} MW")
        peak_value = history["load"].max()
        peak.metric("Peak observed load", f"{peak_value:,.0f} MW")
        factor.metric("Load factor", f"{100 * history['load'].mean() / peak_value:.1f}%" if peak_value > 0 else "—")
        rows = [{"interval_start_utc": stamp.tz_convert("UTC").isoformat(),
                 "interval_end_utc": (stamp.tz_convert("UTC") + pd.Timedelta(hours=1)).isoformat(),
                 "load_mw": value} for stamp, value in history["load"].items()]
        st.plotly_chart(hourly_figure(rows, [("load_mw", "Official actual system load", "#e3a323")]),
                        use_container_width=True, key="forecast_actual_history_chart")
        st.caption("Load factor is the mean divided by the peak over the displayed hourly sample. Current-day coverage ends at the latest completed hour ERCOT has published.")


def render_forecast_comparison() -> None:
    """Show an official outlook, generate a matched comparison and score saved runs."""
    st.subheader("Load Forecast Comparison")
    st.write("ERCOT’s official outlook alongside an experimental load forecast, labeled by its model method. Accuracy is measured later against actual system demand.")
    report = None
    try:
        with st.spinner("Loading ERCOT’s official forecast…"):
            report = official_forecast()
        st.caption(f"Official forecast published {local_time(report['published_at'])} · Retrieved {local_time(report['generated_at'])}")
    except (ValueError, OSError) as exc:
        st.error(f"Official forecast unavailable: {exc}")

    days, mode, params = _model_settings()
    run_col, score_col = st.columns([2, 1])
    run = run_col.button("Run model comparison", type="primary", disabled=report is None, use_container_width=True)
    refresh = score_col.button("Update accuracy scores", use_container_width=True)
    try:
        store = ForecastComparisonStore(comparison_store_path())
        if run or refresh:
            try:
                with st.spinner("Loading completed actual hours and preparing the comparison. The first history download can take a minute…"):
                    bundle = actual_history(days)
                    st.session_state["forecast_actual_history"] = {"days": days, **bundle}
                    for warning in bundle["metadata"].get("warnings", []):
                        st.warning(warning)
                    if run:
                        result = run_and_save_comparison(report, actual_bundle=bundle, tuning_mode=mode,
                                                         manual_params=params, store=store, trainer=fitted_model)
                        st.success(f"Saved {result['snapshot']['summary']['matched_hours']} matching future hours. Accuracy will become available as actual load is reported.")
                    else:
                        store.append_actuals(bundle["history"], observed_at=bundle["metadata"]["retrieved_at"],
                                             source_metadata=bundle["metadata"])
                        st.success("Updated saved runs using available actual load.")
            except (ValueError, OSError, RuntimeError, sqlite3.Error) as exc:
                st.error(f"Comparison update unavailable: {exc}. Previously saved runs remain available below.")
        saved = store.list_snapshots(limit=30)
        evaluations = [store.evaluate_snapshot(item["snapshot_id"]) for item in saved]
    except (ValueError, OSError, sqlite3.Error) as exc:
        st.error(f"Model comparison unavailable: {exc}")
        saved, evaluations = [], []

    comparison_tab, official_tab, accuracy_tab, method_tab = st.tabs([
        "Model vs ERCOT", "Official 7-day outlook", "Accuracy history", "Method & sources",
    ])
    selected = None
    with comparison_tab:
        if not saved:
            st.info("Run the model comparison to create the first saved run. Both forecasts will be saved before the target hours begin.")
            if report:
                st.caption("Latest official outlook while the model comparison is pending")
                st.plotly_chart(hourly_figure(report["hourly_series"], [("load_mw", "Official ERCOT", "#008b8b")]), use_container_width=True, key="forecast_official_initial")
        else:
            selected_id = st.selectbox("Saved comparison", [item["snapshot_id"] for item in saved],
                format_func=lambda identifier: local_time(next(item["model_issued_at"] for item in saved if item["snapshot_id"] == identifier)))
            selected = next(item for item in saved if item["snapshot_id"] == selected_id)
            evaluation = next(item for item in evaluations if item["snapshot_id"] == selected_id)
            metadata, summary = selected["model_metadata"], selected["summary"]
            model_name = model_display_name(metadata.get("model_type", ""))
            st.caption(f"{model_name} forecast issued {local_time(selected['model_issued_at'])}")
            st.caption(f"Paired ERCOT forecast published {local_time(selected['ercot_published_at'])} · Actual inputs complete through {local_time(selected['model_data_cutoff_utc'])}")
            st.caption("The chart uses the saved ERCOT vintage from this run. The separate official outlook tab shows the latest available vintage.")
            one, two, three = st.columns(3)
            one.metric("Matching forecast hours", summary["matched_hours"])
            two.metric(f"Average {model_name} − ERCOT", f"{summary['mean_difference_mw']:+,.0f} MW")
            three.metric("Largest absolute difference", f"{summary['maximum_absolute_difference_mw']:,.0f} MW")
            scored = {row["interval_start_utc"]: row["actual_mw"] for row in evaluation["rows"]}
            plot_rows = [{**row, "actual_mw": scored.get(row["interval_start_utc"])} for row in selected["rows"]]
            series = [("ercot_mw", "Official ERCOT · saved vintage", "#008b8b"), ("model_mw", f"{model_name} · experimental", "#8b5cf6")]
            if scored:
                series.append(("actual_mw", "Observed actual load", "#e3a323"))
            st.plotly_chart(hourly_figure(plot_rows, series), use_container_width=True, key="model_official_comparison")
            st.info("Forecast differences show disagreement, not accuracy. Compare errors only after the same hours have completed and actual load is available.")
            with st.expander("Hourly comparison and export"):
                table = pd.DataFrame(selected["rows"])
                table.insert(0, "Hour starts (Central)", table["interval_start_utc"].map(local_time))
                st.dataframe(table, hide_index=True, use_container_width=True)
                st.download_button("Download this comparison (JSON)", json.dumps({"snapshot": selected, "evaluation": evaluation}, indent=2, allow_nan=False),
                                   file_name=f"ercot-comparison-{selected_id[:12]}.json", mime="application/json")

    with official_tab:
        if report:
            st.write("Official ERCOT forecast · NP3-561-CD · system-wide demand · MW")
            st.plotly_chart(hourly_figure(report["hourly_series"], [("load_mw", "Official ERCOT", "#008b8b")]), use_container_width=True, key="forecast_official_week")
            st.caption("Every reported hour is shown. Each step is the forecast for that full hourly interval; this is not a grid-emergency prediction.")
            st.markdown(f"[ERCOT product]({report['source_product_url']}) · [Exact source archive]({report['source_url']})")
        else:
            st.info("Refresh data to retry the official forecast source.")

    with accuracy_tab:
        st.write("Both forecasts are scored on exactly the same observed hours in each saved run.")
        if evaluations:
            st.dataframe(accuracy_table(evaluations).round(1), hide_index=True, use_container_width=True)
        else:
            st.info("No prospective runs have been saved yet.")
        st.caption("MAE and RMSE are in MW; lower is better. Bias is forecast minus actual. Missing actuals are excluded from both scores. Runs may overlap, so their scores are not pooled. Different forecast issue times and lead times are retained in the export.")
        st.caption("Scores use the first actual observation recorded for each hour; subsequent source revisions are preserved without rewriting that benchmark.")
        st.caption("Saved runs are kept on this dashboard server. Download comparison JSON to retain a copy; hosting restarts or redeployments may reset local history.")

    with method_tab:
        st.markdown("**Official forecast:** ERCOT NP3-561-CD, one identified publication. **Model target:** hourly system-wide load in MW. **Actual observations:** NP6-345-CD daily archives plus the completed hours in ERCOT’s GEN-527-UI current-day display.")
        st.write("The experimental forecast uses XGBoost when installed, with Random Forest as the fallback. Each saved run is labeled by the estimator that actually produced it. Inputs are Central-time calendar features, lagged load and rolling statistics; weather is not an input. The recursive forecast predicts each next hour using previous predictions. A short gap after the latest completed observation is predicted as a bridge and excluded from the scored future window.")
        st.write("Training requires continuous, finite hourly observations. No missing hours are interpolated. Time is matched in UTC to preserve daylight-saving transitions. Forecasts and source hashes are saved before target hours begin, preventing hindsight comparisons.")
        st.markdown("[Forecast product](https://www.ercot.com/mp/data-products/data-product-details?id=NP3-561-CD) · [Actual-load product](https://www.ercot.com/mp/data-products/data-product-details?id=NP6-345-CD) · [Current-day actuals](https://www.ercot.com/content/cdr/html/actual_loads_of_weather_zones.html)")
        if selected:
            metadata = selected["model_metadata"]
            st.write("Internal model validation uses a chronological one-hour-ahead holdout with observed lag updates. These diagnostics do not measure recursive 24-hour skill and are separate from the prospective ERCOT comparison.")
            diagnostics = {model_display_name(metadata.get("model_type", "")): metadata.get("test_mae"), "Previous-hour baseline": metadata.get("persistence_mae"), "Previous-day baseline": metadata.get("seasonal_mae")}
            st.dataframe(pd.DataFrame({"One-hour holdout": diagnostics.keys(), "MAE (MW)": diagnostics.values()}).round(1), hide_index=True)
            with st.expander("Model configuration and data provenance"):
                st.json(metadata)

    _render_actual_history(days)
