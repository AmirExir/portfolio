"""Streamlit forecast settings and the recorded parameters of a fitted forecast."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class ForecastSettings:
    """Inputs shared by the dashboard's universe and individual forecast runs."""

    automatic: bool = True
    forecast_horizon: int = 30
    history_days: int = 1825
    forecast_lookback: int = 20
    forecast_alpha: float = 10.0
    historical_test_points: int = 50
    optimize_forecast_model: bool = True
    use_market_context: bool = True
    primary_model_choice: str = "Best Validation"
    sequence_model_choice: str = "off"
    one_day_sequence_model_choice: str = "off"
    include_rl_policy: bool = False


def render_forecast_settings() -> ForecastSettings:
    """Read forecast inputs from the current container without starting a fit."""
    mode_col, horizon_col = st.columns(2)
    mode = mode_col.radio("Forecast settings mode", ("Automatic", "Manual"), horizontal=True, key="forecast_settings_mode")
    horizon = horizon_col.slider("Forecast horizon (sessions)", 1, 60, 30, key="dashboard_forecast_horizon")

    if mode == "Automatic":
        # Streamlit otherwise clears values for manual widgets on the first run that hides them.
        for key in list(st.session_state):
            if key.startswith("forecast_manual_"):
                st.session_state[key] = st.session_state[key]
        st.caption(
            "On a forecast run, the agent requests five years of prices and market context. "
            "It tests lag lengths and regularization on earlier data, then evaluates the selected settings "
            "on a later holdout when enough history is available. The registered ensemble combines eligible models."
        )
        return ForecastSettings(forecast_horizon=horizon)

    inputs, diagnostics = st.columns(2)
    history_labels = {365: "~1 year", 913: "~2.5 years", 1825: "~5 years", 2738: "~7.5 years"}
    history = inputs.selectbox(
        "Historical data lookback", tuple(history_labels), index=2,
        format_func=history_labels.__getitem__, key="forecast_manual_history",
    )
    model = inputs.selectbox(
        "Primary forecast model", ("Best Validation", "Ensemble", "Ridge", "XGBoost", "Neural Net"),
        format_func=lambda name: "Registered default" if name == "Best Validation" else name,
        key="forecast_manual_model",
        help="The registered default uses the ensemble when available, followed by eligible fallback models.",
    )
    market_context = inputs.checkbox("Use market context features", value=True, key="forecast_manual_context")
    optimize = inputs.checkbox(
        "Tune model settings", value=True, key="forecast_manual_optimize",
        help="Include your lag and regularization values in the candidate grid. Turn off to fit your exact values.",
    )
    lag = inputs.slider("Lag window (sessions)", 5, 60, 20, step=5, key="forecast_manual_lag")
    alpha = inputs.number_input("Ridge regularization", 0.1, 100.0, 10.0, 0.5, key="forecast_manual_alpha")

    points = diagnostics.slider("Previous forecast test points", 10, 100, 50, step=10, key="forecast_manual_points")
    sequences = {"Off": "off", "LSTM": "lstm", "Transformer": "transformer", "Both": "both"}
    sequence = diagnostics.selectbox(
        "Sequence model diagnostics", tuple(sequences), key="forecast_manual_sequence",
        help="Runs additional research models alongside the eligible primary forecasts.",
    )
    one_day = diagnostics.selectbox(
        "One-session sequence diagnostics", ("Both", "LSTM", "Transformer", "Off", "Same as main forecast"),
        key="forecast_manual_one_day_sequence",
    )
    rl_diagnostics = diagnostics.checkbox("Generate RL shadow diagnostics", value=True, key="forecast_manual_rl")
    return ForecastSettings(
        automatic=False, forecast_horizon=horizon, history_days=history, forecast_lookback=lag,
        forecast_alpha=alpha, historical_test_points=points, optimize_forecast_model=optimize,
        use_market_context=market_context, primary_model_choice=model, sequence_model_choice=sequences[sequence],
        one_day_sequence_model_choice=sequences[sequence if one_day == "Same as main forecast" else one_day],
        include_rl_policy=rl_diagnostics,
    )


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if isfinite(result) else None


def _metrics(result: Any) -> Mapping:
    metrics = result.get("metrics") if isinstance(result, Mapping) else getattr(result, "metrics", None)
    return metrics if isinstance(metrics, Mapping) else {}


def tuning_results_table(model_results: Mapping[str, Any], primary_model: str) -> pd.DataFrame:
    """Read actual fit settings for the selected model or its contributing ensemble components."""
    primary = _metrics(model_results.get(primary_model))
    weights = primary.get("component_weights")
    names = (
        [name for name, weight in weights.items() if (_number(weight) or 0) > 0]
        if isinstance(weights, Mapping) else [primary_model]
    )
    rows = []
    for name in names:
        metrics = _metrics(model_results.get(name))
        lag = _number(metrics.get("selected_lookback_window"))
        alpha = _number(metrics.get("selected_ridge_alpha"))
        if lag is None and alpha is None:
            continue
        rows.append({
            "Model": name,
            "Lag (sessions)": lag,
            "Regularization": alpha,
            "Tuning MAE (%)": _number(metrics.get("selection_validation_mae_pct")),
            "Out-of-sample MAE (%)": (
                _number(metrics.get("holdout_mae_pct")) if metrics.get("validation_is_oos") is True else None
            ),
        })
    return pd.DataFrame(rows)
