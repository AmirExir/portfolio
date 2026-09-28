"""Streamlit interface for reproducible, causal hourly-load forecasting."""

from hashlib import sha256
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from xgboost.core import XGBoostError

try:
    from energy_forcast.forecasting import Experiment, FEATURES, read_load_csv, recursive_forecast, run_experiment
except ModuleNotFoundError as exc:
    if exc.name != "energy_forcast":
        raise
    # Support launching from this project directory as well as the repository root.
    from forecasting import Experiment, FEATURES, read_load_csv, recursive_forecast, run_experiment


@st.cache_data(max_entries=3)
def load_data(content: bytes, duplicate_policy: str, core_fingerprint: str) -> tuple[pd.Series, dict]:
    """Cache validated input using its content and preprocessing implementation."""
    return read_load_csv(content, duplicate_policy)


@st.cache_resource(max_entries=2)
def fit_models(content: bytes, duplicate_policy: str, core_fingerprint: str) -> Experiment:
    """Reuse immutable fitted models when users change the forecast display horizon."""
    load, report = load_data(content, duplicate_policy, core_fingerprint)
    return run_experiment(load, report)


st.set_page_config(page_title="Hourly Load Forecasting", layout="wide")
st.title("Hourly Load Forecasting")
st.caption("AEP / PJM example by Amir Exir · Load and forecast values in MW")
st.write("Compare XGBoost with persistence, previous-day, and previous-week forecasts using chronological holdouts.")

uploaded = st.sidebar.file_uploader("Optional hourly data", type="csv", help="Datetime and MW (or AEP_MW) columns")
duplicate_policy = "mean" if uploaded is None else "error"
if uploaded is not None and st.sidebar.checkbox("Average measurements with duplicate timestamps", value=False):
    duplicate_policy = "mean"
source_name = uploaded.name if uploaded is not None else "AEP_hourly.csv"
core_fingerprint = sha256(Path(__file__).with_name("forecasting.py").read_bytes()).hexdigest()
try:
    content = uploaded.getvalue() if uploaded is not None else Path(__file__).with_name("AEP_hourly.csv").read_bytes()
    load, report = load_data(content, duplicate_policy, core_fingerprint)
except (OSError, ValueError) as exc:
    st.error(f"Could not prepare load data: {exc}")
    st.stop()

st.caption(f"Source: {source_name} · {load.index[0]} to {load.index[-1]} · timezone-naive clock labels")
if report["duplicate_extra_rows"] or report["missing_load_hours"]:
    st.warning(
        f"Data quality: {report['duplicate_extra_rows']} duplicate rows averaged; "
        f"{report['inserted_missing_hours']} absent hourly slots inserted; "
        f"{report['missing_load_hours']} hourly loads missing. No interpolation is used. "
        "Training/evaluation exclude targets and history windows requiring missing measurements. "
        "Timezone and daylight-saving transitions cannot be recovered from these clock labels."
    )
st.subheader("Historical load · last 7 days")
st.line_chart(load.iloc[-168:].rename("Load (MW)"))

try:
    with st.spinner("Training, validating, and evaluating chronological holdouts…"):
        experiment = fit_models(content, duplicate_policy, core_fingerprint)
except (ValueError, XGBoostError) as exc:
    st.error(f"Could not run forecasting experiment: {exc}")
    st.stop()

periods = experiment.metadata["periods"]
with st.expander("Experiment setup and data quality", expanded=False):
    st.write("Eligible rows are split 70% training, 15% validation, and 15% test in timestamp order. "
             "Depth is selected on validation one-step MAE. The evaluation model is refitted on training + validation; "
             "the final forecasting model is separately fitted on all eligible historical data.")
    st.dataframe(pd.DataFrame(periods).T, use_container_width=True)
    st.write(f"Excluded hourly slots (warmup, missing target, or incomplete history): {experiment.metadata['excluded_hours']:,}")
    st.dataframe(pd.DataFrame(experiment.metadata["candidate_validation_metrics"]), hide_index=True)
    st.json(report)

one_step, multi_step = st.tabs(["One-hour-ahead evaluation", "24-hour recursive evaluation"])
with one_step:
    st.write("Every prediction uses measured load through the previous hour, including earlier holdout observations. "
             "This measures one-hour-ahead operation when actual measurements arrive each hour.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Validation · model selection**")
        st.dataframe(experiment.validation_metrics.style.format("{:.2f}"), use_container_width=True)
    with right:
        st.markdown("**Held-out test · after selection**")
        st.dataframe(experiment.test_metrics.style.format("{:.2f}"), use_container_width=True)
    st.caption("Lower MAE and RMSE are better. Validation scores were used for selection and are not unbiased test estimates.")
    st.line_chart(experiment.test_predictions.iloc[-168:][["Actual (MW)", "XGBoost", "Previous day"]])

with multi_step:
    origin_count = len(experiment.metadata["recursive_origins"])
    st.write(f"{origin_count} nonoverlapping 24-hour windows sampled evenly across the test period. "
             "Each forecast uses observations only through its origin and then feeds predictions back. "
             "All methods use identical windows. The fitted evaluation model never sees test targets during training.")
    st.dataframe(experiment.recursive_metrics.style.format("{:.2f}"), use_container_width=True)
    errors = experiment.recursive_predictions.assign(
        absolute_error_MW=lambda rows: (rows["Actual (MW)"] - rows["XGBoost"]).abs()
    ).groupby("Lead time (hours)")["absolute_error_MW"].mean()
    st.line_chart(errors.rename("XGBoost MAE (MW)"))
    st.caption("This 24-hour backtest does not validate every possible forecast horizon or operating condition.")

st.subheader("Feature importance")
importance = pd.Series(experiment.model.feature_importances_, index=FEATURES, name="Relative importance")
st.bar_chart(importance.sort_values())
st.caption("Relative split-gain importance from the final model; it does not establish causation.")

st.subheader("Forecast after the end of the dataset")
hours = st.slider("Hours to forecast", min_value=1, max_value=48, value=12)
st.caption(f"Forecast starts {experiment.metadata['forecast_start']}. The bundled dataset ends in 2018; "
           "its forecast is a historical demonstration, not a current grid forecast. No weather inputs or uncertainty intervals are included.")
forecast = None
try:
    forecast = recursive_forecast(experiment.model, load, hours)
except ValueError as exc:
    st.error(f"Cannot forecast from the end of this dataset: {exc}")
else:
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(load.iloc[-48:], label="Measured load")
    axis.plot(forecast, label="Recursive forecast", linestyle="--")
    axis.set(xlabel="Datetime (source clock)", ylabel="Load (MW)")
    axis.legend()
    axis.grid(alpha=0.3)
    st.pyplot(figure)
    plt.close(figure)

st.subheader("Download results")
metadata = {**experiment.metadata, "source_name": source_name, "requested_forecast_hours": hours,
            "forecast_available": forecast is not None}
st.download_button("Experiment metadata (JSON)", json.dumps(metadata, indent=2),
                   file_name="experiment_metadata.json", mime="application/json")
st.download_button("One-step test predictions (CSV)", experiment.test_predictions.to_csv(),
                   file_name="test_predictions.csv", mime="text/csv")
st.download_button("Recursive backtest (CSV)", experiment.recursive_predictions.to_csv(),
                   file_name="recursive_backtest.csv", mime="text/csv")
if forecast is not None:
    st.download_button("Forecast (CSV)", forecast.to_csv(), file_name="forecast.csv", mime="text/csv")
