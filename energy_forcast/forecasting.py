"""Causal hourly-load features, temporal evaluation, and recursive forecasts.

Input timestamps are timezone-naive clock labels. No timezone or daylight-saving
correction is inferred, and missing loads are never interpolated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
import platform
from typing import Any, Protocol

import numpy as np
import pandas as pd

try:
    from .evaluation import add_intervals, calibrate_intervals, evaluation_summary, season_labels
    from .weather import WEATHER_FEATURES, WeatherArchive
except ImportError:
    from evaluation import add_intervals, calibrate_intervals, evaluation_summary, season_labels
    from weather import WEATHER_FEATURES, WeatherArchive


FEATURES = (
    "hour", "dayofweek", "month", "is_weekend", "lag_1", "lag_24", "lag_168",
    "rolling_24h_mean", "rolling_168h_mean",
)
BASELINES = {"Persistence": 1, "Previous day": 24, "Previous week": 168}
MAX_HISTORY_HOURS = 1_000_000


class Predictor(Protocol):
    """Minimal interface shared by the fitted model and deterministic test models."""

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return one load prediction in MW for each feature row."""
        ...


@dataclass(frozen=True)
class ForecastConfig:
    """Reproducible split, model-selection, and recursive-backtest settings."""

    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    seed: int = 42
    estimators: int = 100
    candidate_depths: tuple[int, ...] = (4, 6)
    learning_rate: float = 0.1
    recursive_horizon: int = 24
    max_origins: int = 30
    evaluation_horizons: tuple[int, ...] = (24, 48)
    rolling_seasons: int = 4
    interval_coverage: float = 0.9
    min_calibration_origins: int = 10

    def __post_init__(self) -> None:
        if not (0 < self.train_fraction < 1 and 0 < self.validation_fraction < 1
                and self.train_fraction + self.validation_fraction < 1):
            raise ValueError("Training and validation fractions must leave a nonempty test set.")
        if (not self.candidate_depths or any(depth < 1 for depth in self.candidate_depths)
                or self.estimators < 1 or not 0 < self.learning_rate <= 1):
            raise ValueError("Model depth, estimator count, and learning rate must be positive.")
        if (not isinstance(self.recursive_horizon, int) or isinstance(self.recursive_horizon, bool)
                or not 1 <= self.recursive_horizon <= 168 or not isinstance(self.max_origins, int)
                or isinstance(self.max_origins, bool) or self.max_origins < 1):
            raise ValueError("Use a recursive horizon of 1–168 hours and at least one origin.")
        if (not self.evaluation_horizons or any(not isinstance(h, int) or isinstance(h, bool) or not 1 <= h <= 168
                                              for h in self.evaluation_horizons)):
            raise ValueError("Evaluation horizons must be integers between 1 and 168 hours.")
        if (not 0 < self.interval_coverage < 1 or not isinstance(self.min_calibration_origins, int)
                or isinstance(self.min_calibration_origins, bool) or self.min_calibration_origins < 1
                or not isinstance(self.rolling_seasons, int) or isinstance(self.rolling_seasons, bool)
                or not 0 <= self.rolling_seasons <= 12):
            raise ValueError("Use coverage between zero and one, positive calibration count, and 0–12 seasons.")


@dataclass
class Experiment:
    """Fitted deployment model and reviewable results from untouched test data."""

    model: Predictor
    validation_metrics: pd.DataFrame
    test_metrics: pd.DataFrame
    recursive_metrics: pd.DataFrame
    test_predictions: pd.DataFrame
    recursive_predictions: pd.DataFrame
    metadata: dict[str, Any]
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)
    horizon_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    rolling_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    horizon_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    seasonal_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)


def read_load_csv(content: bytes, duplicate_policy: str = "error") -> tuple[pd.Series, dict[str, Any]]:
    """Read Datetime + MW/AEP_MW CSV and report explicit data-quality processing.

    Duplicate timestamps fail unless the caller explicitly requests ``mean``.
    Absent clock hours are inserted as NaN, preserving elapsed-hour lag semantics.
    """
    if duplicate_policy not in {"error", "mean"}:
        raise ValueError("Duplicate policy must be 'error' or 'mean'.")
    try:
        raw = pd.read_csv(BytesIO(content))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError("Provide a readable UTF-8 CSV with Datetime and MW columns.") from exc
    value_column = "MW" if "MW" in raw else "AEP_MW"
    if "Datetime" not in raw or value_column not in raw or raw.empty:
        raise ValueError("CSV must contain Datetime and MW (or AEP_MW), with at least one row.")
    try:
        timestamps = pd.DatetimeIndex(pd.to_datetime(raw["Datetime"], errors="raise", format="mixed"))
        values = pd.to_numeric(raw[value_column], errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Datetime must be valid timestamps and MW must be numeric or empty.") from exc
    if timestamps.hasnans or timestamps.tz is not None:
        raise ValueError("Use nonempty timezone-naive timestamps; normalize timezones before upload.")
    if not timestamps.equals(timestamps.floor("h")):
        raise ValueError("Every timestamp must lie exactly on an hour boundary.")
    if np.isinf(values).any() or (values < 0).any():
        raise ValueError("MW must be finite and nonnegative; leave missing measurements empty.")
    duplicates = int(timestamps.duplicated().sum())
    if duplicates and duplicate_policy == "error":
        raise ValueError(f"Found {duplicates} duplicate timestamp rows. Resolve them or explicitly select averaging.")
    series = pd.Series(values, index=timestamps, name="MW").sort_index()
    if duplicates:
        # A mean is only defined when every duplicate has a measurement.
        # Do not silently discard a missing observation within a duplicate group.
        grouped = series.groupby(level=0)
        series = grouped.mean().where(grouped.count() == grouped.size())
    hour_count = int((series.index[-1] - series.index[0]) / pd.Timedelta(hours=1)) + 1
    if hour_count > MAX_HISTORY_HOURS:
        raise ValueError(f"Timestamp range exceeds the supported {MAX_HISTORY_HOURS:,} hourly slots.")
    full_index = pd.date_range(series.index[0], series.index[-1], freq="h", name="Datetime")
    inserted = len(full_index) - len(series)
    series = series.reindex(full_index)
    report = {
        "source_sha256": sha256(content).hexdigest(),
        "source_rows": len(raw),
        "duplicate_extra_rows": duplicates,
        "duplicate_policy": duplicate_policy,
        "inserted_missing_hours": inserted,
        "missing_load_hours": int(series.isna().sum()),
        "hourly_slots": len(series),
        "timestamp_assumption": "Timezone-naive hourly clock labels; no inferred DST correction",
        "missing_data_policy": "No interpolation; exclude targets/features requiring missing loads",
    }
    return series, report


def _validate_hourly(load: pd.Series) -> None:
    if not isinstance(load.index, pd.DatetimeIndex) or load.empty:
        raise ValueError("Load requires a nonempty DatetimeIndex.")
    index = load.index
    if (index.tz is not None or index.hasnans or not index.is_unique
            or not index.is_monotonic_increasing or not index.equals(index.floor("h"))
            or not (index.to_series().diff().iloc[1:] == pd.Timedelta(hours=1)).all()):
        raise ValueError("Load must use a sorted, unique, timezone-naive hourly grid.")
    if np.isinf(load.to_numpy(dtype=float)).any() or (load < 0).any():
        raise ValueError("Load values must be nonnegative and finite, or missing.")


def make_features(load: pd.Series, weather: WeatherArchive | None = None) -> pd.DataFrame:
    """Build predictors of MW(t) using clock fields and measurements strictly before t."""
    _validate_hourly(load)
    frame = pd.DataFrame(index=load.index)
    frame["hour"] = load.index.hour
    frame["dayofweek"] = load.index.dayofweek
    frame["month"] = load.index.month
    frame["is_weekend"] = (load.index.dayofweek >= 5).astype(int)
    for lag in (1, 24, 168):
        frame[f"lag_{lag}"] = load.shift(lag)
    past = load.shift(1)
    for window in (24, 168):
        frame[f"rolling_{window}h_mean"] = past.rolling(window, min_periods=window).mean()
    if weather is not None:
        frame = frame.join(weather.at_origins(load.index, load.index - pd.Timedelta(hours=1)))
    return frame


def temporal_split(frame: pd.DataFrame, config: ForecastConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split eligible rows chronologically; require a week in each evaluation partition."""
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Temporal splits require a sorted, unique index.")
    first = int(len(frame) * config.train_fraction)
    second = first + int(len(frame) * config.validation_fraction)
    splits = (frame.iloc[:first], frame.iloc[first:second], frame.iloc[second:])
    if min(len(part) for part in splits) < 168:
        raise ValueError("Need at least 168 eligible hourly rows per split; provide roughly eight weeks of complete data.")
    return splits


def regression_metrics(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> dict[str, float]:
    """Calculate MAE/RMSE in MW without undefined percentage errors at zero load."""
    observed, estimate = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if (observed.ndim != 1 or observed.size == 0 or observed.shape != estimate.shape
            or not np.isfinite(observed).all() or not np.isfinite(estimate).all()):
        raise ValueError("Metrics require equally sized, nonempty, finite one-dimensional arrays.")
    residual = observed - estimate
    return {"MAE (MW)": float(np.mean(np.abs(residual))), "RMSE (MW)": float(np.sqrt(np.mean(residual ** 2)))}


def recursive_forecast(model: Predictor, history: pd.Series, hours: int,
                       weather: WeatherArchive | None = None) -> pd.Series:
    """Forecast using only past history and earlier predictions, without mutating input."""
    _validate_hourly(history)
    if not isinstance(hours, int) or isinstance(hours, bool) or not 1 <= hours <= 168:
        raise ValueError("Forecast length must be an integer between 1 and 168 hours.")
    if len(history) < 168 or history.iloc[-168:].isna().any():
        raise ValueError("Forecasting requires 168 consecutive measured hours immediately before the origin.")
    values = list(history.iloc[-168:].to_numpy(dtype=float))
    index = pd.date_range(history.index[-1] + pd.Timedelta(hours=1), periods=hours, freq="h", name="Datetime")
    weather_rows = None
    if weather is not None:
        weather_rows = weather.at_origins(index, pd.DatetimeIndex([history.index[-1]] * hours))
        if weather_rows.isna().any().any():
            raise ValueError("Archived forecast weather available at this origin does not cover the requested horizon.")
    predictions = []
    for stamp in index:
        row = [stamp.hour, stamp.dayofweek, stamp.month, int(stamp.dayofweek >= 5),
               values[-1], values[-24], values[-168], np.mean(values[-24:]), np.mean(values[-168:])]
        columns = list(FEATURES)
        if weather_rows is not None:
            row.extend(weather_rows.loc[stamp].tolist())
            columns.extend(WEATHER_FEATURES)
        predicted = np.asarray(model.predict(pd.DataFrame([row], columns=columns)), dtype=float)
        if predicted.shape != (1,) or not np.isfinite(predicted[0]) or predicted[0] < 0:
            raise ValueError("Model produced an invalid or negative MW forecast; inspect the data and fitted model.")
        predictions.append(float(predicted[0]))
        values.append(float(predicted[0]))
    return pd.Series(predictions, index=index, name="Forecast (MW)")


def recursive_backtest(model: Predictor, load: pd.Series, test_start: pd.Timestamp,
                       horizon: int = 24, max_origins: int = 30,
                       weather: WeatherArchive | None = None) -> pd.DataFrame:
    """Evaluate evenly sampled, nonoverlapping test windows with no within-window observations.

    The model must already have been fitted without test data. Actual observations
    before each origin are available, as in operational rolling-origin evaluation.
    """
    _validate_hourly(load)
    if not 1 <= horizon <= 168 or max_origins < 1:
        raise ValueError("Backtest requires a horizon of 1–168 hours and at least one origin.")
    start = max(168, int(load.index.searchsorted(test_start)))
    candidates = [position for position in range(start, len(load) - horizon + 1, horizon)
                  if load.iloc[position - 168:position + horizon].notna().all()]
    load_complete_count = len(candidates)
    if weather is not None and candidates:
        targets = pd.DatetimeIndex([stamp for position in candidates for stamp in load.index[position:position + horizon]])
        origins = pd.DatetimeIndex([load.index[position - 1] for position in candidates for _ in range(horizon)])
        complete = weather.at_origins(targets, origins).notna().all(axis=1).to_numpy().reshape(-1, horizon).all(axis=1)
        candidates = [position for position, valid in zip(candidates, complete) if valid]
    if not candidates:
        if weather is not None and load_complete_count:
            raise ValueError(f"No complete recursive test windows; {load_complete_count} load-complete origins "
                             "lack full archived-weather coverage available at their origins.")
        raise ValueError("No complete recursive test windows; use more continuous hourly data.")
    chosen = np.linspace(0, len(candidates) - 1, min(max_origins, len(candidates)), dtype=int)
    frames = []
    for candidate in chosen:
        position = candidates[candidate]
        history = load.iloc[position - 168:position]
        forecast = recursive_forecast(model, history, horizon, weather)
        frame = pd.DataFrame({"Actual (MW)": load.iloc[position:position + horizon], "XGBoost": forecast})
        for name, period in BASELINES.items():
            pattern = history.iloc[-period:].to_numpy()
            frame[name] = np.resize(pattern, horizon)
        frame["Origin"] = history.index[-1]
        frame["Lead time (hours)"] = np.arange(1, horizon + 1)
        frames.append(frame)
    result = pd.concat(frames)
    result.attrs["load_complete_origins"] = load_complete_count
    result.attrs["weather_incomplete_origins"] = load_complete_count - len(candidates)
    result.attrs["available_origins"] = len(candidates)
    return result


def _metrics_table(predictions: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({name: regression_metrics(predictions["Actual (MW)"], predictions[name])
                         for name in ("XGBoost", *BASELINES)}).T.rename_axis("Model")


def _one_step_predictions(model: Predictor, rows: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    predictions = pd.DataFrame({"Actual (MW)": rows["MW"], "XGBoost": model.predict(rows[features])})
    for name, period in BASELINES.items():
        predictions[name] = rows[f"lag_{period}"]
    return predictions


def _period(rows: pd.DataFrame) -> dict[str, Any]:
    return {"start": rows.index[0].isoformat(), "end": rows.index[-1].isoformat(), "eligible_rows": len(rows)}


def _fit_partition(train: pd.DataFrame, selection: pd.DataFrame, features: list[str],
                   config: ForecastConfig) -> tuple[Predictor, Predictor, dict[str, Any], list[dict[str, Any]]]:
    """Select on earlier validation rows, then fit before calibration starts."""
    from xgboost import XGBRegressor

    common = {"n_estimators": config.estimators, "learning_rate": config.learning_rate,
              "random_state": config.seed, "n_jobs": 1, "tree_method": "hist", "objective": "reg:squarederror"}
    candidates = []
    best: tuple[float, int, Predictor] | None = None
    for depth in config.candidate_depths:
        candidate = XGBRegressor(max_depth=depth, **common)
        candidate.fit(train[features], train.MW)
        metrics = regression_metrics(selection.MW, candidate.predict(selection[features]))
        candidates.append({"max_depth": depth, **metrics})
        if best is None or metrics["MAE (MW)"] < best[0]:
            best = (metrics["MAE (MW)"], depth, candidate)
    assert best is not None
    parameters = {"max_depth": best[1], **common}
    model = XGBRegressor(**parameters)
    fit_rows = pd.concat([train, selection])
    model.fit(fit_rows[features], fit_rows.MW)
    return model, best[2], parameters, candidates


def _available_backtest(model: Predictor, load: pd.Series, start: pd.Timestamp,
                        horizon: int, config: ForecastConfig,
                        weather: WeatherArchive | None) -> pd.DataFrame:
    """Represent absent complete windows explicitly, without hiding model failures."""
    try:
        return recursive_backtest(model, load, start, horizon, config.max_origins, weather)
    except ValueError as exc:
        if str(exc).startswith("No complete recursive test windows"):
            result = pd.DataFrame()
            result.attrs["unavailable_reason"] = str(exc)
            return result
        raise


def _calibration(model: Predictor, load: pd.Series, rows: pd.DataFrame,
                 config: ForecastConfig, weather: WeatherArchive | None) -> pd.DataFrame:
    horizon = max(config.recursive_horizon, *config.evaluation_horizons)
    predictions = _available_backtest(model, load.loc[:rows.index[-1]], rows.index[0], horizon, config, weather)
    result = calibrate_intervals(predictions, config.interval_coverage, config.min_calibration_origins)
    result.attrs.update(predictions.attrs)
    result.attrs["origins"] = [stamp.isoformat() for stamp in predictions.Origin.unique()] if not predictions.empty else []
    return result


def _horizon_backtests(model: Predictor, load: pd.Series, start: pd.Timestamp,
                       config: ForecastConfig, weather: WeatherArchive | None,
                       calibration: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames, availability = [], []
    for horizon in sorted(set((*config.evaluation_horizons, config.recursive_horizon))):
        frame = _available_backtest(model, load, start, horizon, config, weather)
        availability.append({"horizon_hours": horizon, "evaluated_hours": len(frame), **frame.attrs})
        if not frame.empty:
            frame = add_intervals(frame, calibration)
            frame["Horizon (hours)"] = horizon
            frames.append(frame)
    return (pd.concat(frames) if frames else pd.DataFrame()), availability


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Produce strict-JSON records, preserving unavailable metrics as null."""
    return frame.astype(object).where(frame.notna(), None).to_dict(orient="records")


def _rolling_evaluation(load: pd.Series, eligible: pd.DataFrame, features: list[str],
                        config: ForecastConfig, weather: WeatherArchive | None
                        ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Independently tune, calibrate, and test expanding-history seasonal folds."""
    boundaries = pd.date_range((load.index[0] - pd.Timedelta(days=95)).normalize(),
                               (load.index[-1] + pd.Timedelta(days=95)).normalize(), freq="QS-DEC")
    windows = [(start, end) for start, end in zip(boundaries[:-1], boundaries[1:])
               if start >= load.index[0] and end <= load.index[-1] + pd.Timedelta(hours=1)
               and len(eligible.loc[eligible.index < start]) >= 1120]
    frames, reports = [], []
    for start, end in windows[-config.rolling_seasons:] if config.rolling_seasons else []:
        historical = eligible.loc[eligible.index < start]
        try:
            train, selection, calibration_rows = temporal_split(historical, config)
        except ValueError as exc:
            reports.append({"season": season_labels(pd.DatetimeIndex([start]))[0], "status": str(exc)})
            continue
        model, _, parameters, candidates = _fit_partition(train, selection, features, config)
        calibration = _calibration(model, load, calibration_rows, config, weather)
        predictions, availability = _horizon_backtests(
            model, load.loc[load.index < end], start, config, weather, calibration)
        label = str(season_labels(pd.DatetimeIndex([start]))[0])
        reports.append({"season": label, "test_start": start.isoformat(), "test_end_exclusive": end.isoformat(),
                        "training": _period(train), "selection": _period(selection),
                        "calibration": _period(calibration_rows), "selected_parameters": parameters,
                        "candidate_validation_metrics": candidates, "window_availability": availability,
                        "interval_calibration": _records(calibration.reset_index()),
                        "calibration_availability": calibration.attrs})
        if not predictions.empty:
            predictions["Fold"] = label
            frames.append(predictions)
    return (pd.concat(frames) if frames else pd.DataFrame()), reports


def forecast_with_intervals(experiment: Experiment, history: pd.Series, hours: int,
                            weather: WeatherArchive | None = None) -> pd.DataFrame:
    """Forecast with empirical widths transferred from pre-test calibration.

    Deployment refits on all history; transferred widths are estimates, not a
    coverage guarantee for the refitted model or a new operating regime.
    """
    expects_weather = bool(experiment.metadata.get("weather", {}).get("enabled"))
    if expects_weather != (weather is not None):
        raise ValueError("Forecast weather inputs must match the fitted experiment's feature set.")
    point = recursive_forecast(experiment.model, history, hours, weather)
    frame = pd.DataFrame({"XGBoost": point, "Lead time (hours)": np.arange(1, hours + 1)})
    frame["Origin"] = history.index[-1]
    return add_intervals(frame, experiment.calibration)


def run_experiment(load: pd.Series, data_report: dict[str, Any],
                   config: ForecastConfig = ForecastConfig(),
                   weather: WeatherArchive | None = None) -> Experiment:
    """Separate selection/calibration/test data and evaluate seasonal rolling folds."""
    from xgboost import XGBRegressor

    features_frame = make_features(load, weather)
    features = list(features_frame.columns)
    eligible = features_frame.assign(MW=load).dropna()
    train, validation, test = temporal_split(eligible, config)
    midpoint = len(validation) // 2
    selection, calibration_rows = validation.iloc[:midpoint], validation.iloc[midpoint:]
    model, selection_model, parameters, candidates = _fit_partition(train, selection, features, config)
    validation_predictions = _one_step_predictions(selection_model, selection, features)
    test_predictions = _one_step_predictions(model, test, features)
    calibration = _calibration(model, load, calibration_rows, config, weather)
    horizon_predictions, availability = _horizon_backtests(model, load, test.index[0], config, weather, calibration)
    if horizon_predictions.empty or config.recursive_horizon not in horizon_predictions["Horizon (hours)"].unique():
        raise ValueError("No complete recursive test windows for the configured horizon; inspect load/weather coverage.")
    recursive_predictions = horizon_predictions.loc[
        horizon_predictions["Horizon (hours)"] == config.recursive_horizon].copy()
    validation_metrics = _metrics_table(validation_predictions)
    test_metrics = _metrics_table(test_predictions)
    recursive_metrics = _metrics_table(recursive_predictions)
    rolling_predictions, rolling_reports = _rolling_evaluation(load, eligible, features, config, weather)
    horizon_metrics = evaluation_summary(horizon_predictions, ["Horizon (hours)"])
    seasonal_metrics = evaluation_summary(rolling_predictions, ["Fold", "Horizon (hours)", "Season"])
    # Deployment consumes all observations only after evaluation and calibration are fixed.
    deployment_model = XGBRegressor(**parameters)
    deployment_model.fit(eligible[features], eligible.MW)
    fit_rows = pd.concat([train, selection])
    metadata = {
        "tool_version": "3.0.0", "core_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "supporting_source_sha256": {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                     for name in ("weather.py", "evaluation.py")},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": platform.python_version(), **{name: version(name) for name in
                    ("numpy", "pandas", "xgboost", "scikit-learn")}},
        "data": data_report, "configuration": asdict(config), "target": "Hourly load (MW)",
        "features": features, "excluded_hours": len(load) - len(eligible),
        "weather": {"enabled": weather is not None, **(weather.report if weather else {
            "status": "No archived weather supplied; load/calendar-only model"}),
            "unavailable_one_step_hours": int(features_frame[list(WEATHER_FEATURES)].isna().any(axis=1).sum())
            if weather else 0},
        "periods": {"training": _period(train), "validation": _period(selection),
                    "interval_calibration": _period(calibration_rows), "test": _period(test),
                    "evaluation_fit": _period(fit_rows), "deployment_fit": _period(eligible)},
        "selection_rule": "Lowest one-step MAE in first half of validation; second half reserved for intervals",
        "candidate_validation_metrics": candidates, "selected_parameters": parameters,
        "one_step_assumption": "Actual loads through t-1; weather issued and available by t-1",
        "recursive_assumption": "Actual history and forecast weather only through origin; no within-window observations",
        "recursive_origins": [stamp.isoformat() for stamp in recursive_predictions.Origin.unique()],
        "validation_metrics": validation_metrics.to_dict(orient="index"),
        "test_metrics": test_metrics.to_dict(orient="index"),
        "recursive_metrics": recursive_metrics.to_dict(orient="index"),
        "horizon_metrics": _records(horizon_metrics), "window_availability": availability,
        "interval_calibration": _records(calibration.reset_index()),
        "calibration_availability": calibration.attrs,
        "interval_method": "Lead-specific absolute residual order statistic ceil((n+1)*coverage); nonnegative lower bound",
        "interval_limit": "Temporal dependence and deployment refit mean nominal coverage is not guaranteed",
        "rolling_seasons": rolling_reports,
        "seasonal_metrics": _records(seasonal_metrics),
        "rolling_status": "Completed available seasonal folds" if rolling_reports else "No complete eligible seasonal folds",
        "forecast_start": (load.index[-1] + pd.Timedelta(hours=1)).isoformat(),
        "outputs": "User downloads: metadata, test predictions, horizon backtests, seasonal backtests, forecast",
    }
    return Experiment(deployment_model, validation_metrics, test_metrics, recursive_metrics,
                      test_predictions, recursive_predictions, metadata, calibration, horizon_predictions,
                      rolling_predictions, horizon_metrics, seasonal_metrics)
