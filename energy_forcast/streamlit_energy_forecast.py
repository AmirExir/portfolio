"""Streamlit interface for reproducible, causal hourly-load forecasting."""

from hashlib import sha256
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from xgboost.core import XGBoostError

try:
    from energy_forcast.forecasting import Experiment, read_load_csv, forecast_with_intervals, run_experiment
    from energy_forcast.weather import read_weather_csv
    from energy_forcast.evaluation import evaluation_summary
except ModuleNotFoundError as exc:
    if exc.name != "energy_forcast":
        raise
    # Support launching from this project directory as well as the repository root.
    from forecasting import Experiment, read_load_csv, forecast_with_intervals, run_experiment
    from weather import read_weather_csv
    from evaluation import evaluation_summary


@st.cache_data(max_entries=3)
def load_data(content: bytes, duplicate_policy: str, core_fingerprint: str) -> tuple[pd.Series, dict]:
    """Cache validated input using its content and preprocessing implementation."""
    return read_load_csv(content, duplicate_policy)


@st.cache_resource(max_entries=2)
def fit_models(content: bytes, duplicate_policy: str, core_fingerprint: str,
               weather_content: bytes | None = None) -> Experiment:
    """Reuse immutable fitted models when users change the forecast display horizon."""
    load, report = load_data(content, duplicate_policy, core_fingerprint)
    weather = read_weather_csv(weather_content) if weather_content is not None else None
    return run_experiment(load, report, weather=weather)


st.set_page_config(page_title="Hourly Load Forecasting", layout="wide")
st.title("Hourly Load Forecasting")
st.caption("AEP / PJM example by Amir Exir · Load and forecast values in MW")
st.write("Compare XGBoost with persistence, previous-day, and previous-week forecasts using chronological holdouts.")

uploaded = st.sidebar.file_uploader("Optional hourly data", type="csv", help="Datetime and MW (or AEP_MW) columns")
duplicate_policy = "mean" if uploaded is None else "error"
if uploaded is not None and st.sidebar.checkbox("Average measurements with duplicate timestamps", value=False):
    duplicate_policy = "mean"
source_name = uploaded.name if uploaded is not None else "AEP_hourly.csv"
weather_upload = st.sidebar.file_uploader(
    "Optional archived weather forecasts", type="csv",
    help="Columns: valid_time, issued_at, available_at, record_type=forecast, temperature_c, source. "
         "Use first availability timestamps and the same clock as load; observed weather is rejected.")
weather_content = weather_upload.getvalue() if weather_upload is not None else None
core_fingerprint = sha256(b"".join(Path(__file__).with_name(name).read_bytes()
                                for name in ("forecasting.py", "weather.py", "evaluation.py"))).hexdigest()
try:
    content = uploaded.getvalue() if uploaded is not None else Path(__file__).with_name("AEP_hourly.csv").read_bytes()
    load, report = load_data(content, duplicate_policy, core_fingerprint)
    weather = read_weather_csv(weather_content) if weather_content is not None else None
except (OSError, ValueError) as exc:
    st.error(f"Could not prepare load data: {exc}")
    st.stop()

st.caption(f"Source: {source_name} · {load.index[0]} to {load.index[-1]} · timezone-naive clock labels")
if weather is None:
    st.info("No archived weather has been supplied. This run uses load history and calendar fields only.")
else:
    st.caption("Weather uses only forecasts issued and first available by each prediction origin. "
               "Missing forecast vintages exclude affected training rows and evaluation windows; no weather is synthesized.")
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
        experiment = fit_models(content, duplicate_policy, core_fingerprint, weather_content)
except (ValueError, XGBoostError) as exc:
    st.error(f"Could not run forecasting experiment: {exc}")
    st.stop()

periods = experiment.metadata["periods"]
with st.expander("Experiment setup and data quality", expanded=False):
    st.write("Eligible rows are split 70% training, 15% validation, and 15% test in timestamp order. "
             "The first half of validation selects depth; its second half calibrates intervals. "
             "The evaluation model is fitted on training + selection rows only; "
             "the final forecasting model is separately fitted on all eligible historical data.")
    st.dataframe(pd.DataFrame(periods).T, use_container_width=True)
    st.write(f"Excluded hourly slots (warmup, missing target, or incomplete history): {experiment.metadata['excluded_hours']:,}")
    st.dataframe(pd.DataFrame(experiment.metadata["candidate_validation_metrics"]), hide_index=True)
    st.json(report)
    st.json(experiment.metadata["weather"])
    st.json(experiment.metadata["window_availability"])

one_step, multi_step, seasonal = st.tabs(["One-hour-ahead evaluation", "24 / 48-hour evaluation", "Seasonal rolling evaluation"])
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
    st.write(f"{origin_count} nonoverlapping 24-hour windows, plus separate 48-hour windows, sampled across the test period. "
             "Each forecast uses observations only through its origin and then feeds predictions back. "
             "All methods use identical windows. The fitted evaluation model never sees test targets during training.")
    st.dataframe(experiment.horizon_metrics, hide_index=True, use_container_width=True)
    lead_metrics = evaluation_summary(experiment.horizon_predictions, ["Horizon (hours)", "Lead time (hours)"])
    st.dataframe(lead_metrics.loc[lead_metrics.Model == "XGBoost"], hide_index=True, use_container_width=True)
    st.caption("Nominal interval coverage is 90%. Empirical coverage and width use held-out targets. "
               "An unavailable interval has fewer than ten calibration origins or an unsupported quantile; "
               "missing coverage is not a zero-error result.")
    with st.expander("Interval calibration by lead"):
        st.dataframe(experiment.calibration, use_container_width=True)

with seasonal:
    st.write("Up to four complete meteorological seasons are evaluated with expanding historical training. "
             "Each fold independently selects depth and calibrates intervals using data before that season. "
             "All models use the same complete windows within each horizon; the model is fixed within a season.")
    if experiment.rolling_predictions.empty:
        st.info("No complete seasonal folds are available with sufficient earlier data. Upload longer history for seasonal evaluation.")
    else:
        st.dataframe(experiment.seasonal_metrics, hide_index=True, use_container_width=True)
    st.caption("Seasonal folds may overlap the main holdout in calendar time and are separate evaluations. "
               "Do not pool their errors or treat a historical coverage estimate as a guarantee.")

st.subheader("Feature importance")
importance = pd.Series(experiment.model.feature_importances_, index=experiment.metadata["features"], name="Relative importance")
st.bar_chart(importance.sort_values())
st.caption("Relative split-gain importance from the final model; it does not establish causation.")

st.subheader("Forecast after the end of the dataset")
hours = st.slider("Hours to forecast", min_value=1, max_value=48, value=12)
st.caption(f"Forecast starts {experiment.metadata['forecast_start']}. The bundled dataset ends in 2018; "
           "its forecast is a historical demonstration, not a current grid forecast. "
           "Empirical interval widths are transferred from the pre-test calibration model to the model refitted on all history; "
           "coverage for the refitted model is not independently established.")
forecast = None
try:
    forecast = forecast_with_intervals(experiment, load, hours, weather)
except ValueError as exc:
    st.error(f"Cannot forecast from the end of this dataset: {exc}")
else:
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(load.iloc[-48:], label="Measured load")
    axis.plot(forecast.index, forecast.XGBoost, label="Recursive forecast", linestyle="--")
    if forecast["Lower (MW)"].notna().all():
        axis.fill_between(forecast.index, forecast["Lower (MW)"], forecast["Upper (MW)"],
                          alpha=0.2, label="90% nominal empirical interval")
    else:
        st.warning("Insufficient separate calibration windows for all requested lead times; unsupported interval bounds are blank.")
    axis.set(xlabel="Datetime (source clock)", ylabel="Load (MW)")
    axis.legend()
    axis.grid(alpha=0.3)
    st.pyplot(figure)
    plt.close(figure)

st.subheader("Download results")
metadata = {**experiment.metadata, "source_name": source_name,
            "weather_source_name": weather_upload.name if weather_upload else None, "requested_forecast_hours": hours,
            "forecast_available": forecast is not None}
st.download_button("Experiment metadata (JSON)", json.dumps(metadata, indent=2),
                   file_name="experiment_metadata.json", mime="application/json")
st.download_button("One-step test predictions (CSV)", experiment.test_predictions.to_csv(),
                   file_name="test_predictions.csv", mime="text/csv")
st.download_button("Recursive backtest (CSV)", experiment.horizon_predictions.to_csv(),
                   file_name="recursive_backtest.csv", mime="text/csv")
if not experiment.rolling_predictions.empty:
    st.download_button("Seasonal backtests (CSV)", experiment.rolling_predictions.to_csv(),
                       file_name="seasonal_backtests.csv", mime="text/csv")
if forecast is not None:
    st.download_button("Forecast (CSV)", forecast.to_csv(), file_name="forecast.csv", mime="text/csv")
