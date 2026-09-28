"""Causal hourly load forecasting, independent of the dashboard and API client."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from importlib.metadata import version
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


FEATURES = [
    "hour", "day_of_week", "is_weekend", "hour_sin", "hour_cos",
    "load_lag_1h", "load_lag_24h", "rolling_mean_3h",
    "rolling_mean_24h", "rolling_std_24h",
]


def validate_hourly_load(data: pd.DataFrame) -> pd.DataFrame:
    """Sort a copy and require finite MW on an unambiguous hourly time axis.

    No interpolation, aggregation, timezone inference, or synthetic timestamps
    are applied. Offset-aware timestamps support daylight-saving transitions.
    """
    if "load" not in data.columns or data.columns.duplicated().any():
        raise ValueError("Load data must have one uniquely named 'load' column in MW.")
    if not isinstance(data.index, pd.DatetimeIndex) or data.index.hasnans:
        raise ValueError("Load forecasting requires valid source timestamps.")
    if data.index.has_duplicates:
        raise ValueError("Duplicate load timestamps must be resolved at the source (including DST offsets).")
    frame = data[["load"]].copy().sort_index()
    try:
        frame["load"] = pd.to_numeric(frame["load"], errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Load values must be numeric MW.") from exc
    if not np.isfinite(frame["load"]).all() or (frame["load"] < 0).any():
        raise ValueError("Load values must be finite, nonnegative MW; missing values are not filled.")
    if len(frame) > 1 and not (frame.index.to_series().diff().iloc[1:] == pd.Timedelta(hours=1)).all():
        raise ValueError("Load forecasting requires consecutive hourly observations; resolve gaps or interval mismatches first.")
    return frame


def build_load_features(data: pd.DataFrame) -> pd.DataFrame:
    """Use only observations strictly before each prediction timestamp."""
    frame = validate_hourly_load(data)
    frame["hour"] = frame.index.hour
    frame["day_of_week"] = frame.index.dayofweek
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(int)
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
    prior = frame["load"].shift(1)
    frame["load_lag_1h"] = prior
    frame["load_lag_24h"] = frame["load"].shift(24)
    frame["rolling_mean_3h"] = prior.rolling(3).mean()
    frame["rolling_mean_24h"] = prior.rolling(24).mean()
    frame["rolling_std_24h"] = prior.rolling(24).std()
    return frame


def forecast_load(model: Any, scaler: Any, history: pd.DataFrame, hours: int = 24) -> pd.DataFrame:
    """Predict recursively, updating every lag/window with prior predictions."""
    if isinstance(hours, bool) or not isinstance(hours, int) or hours < 1:
        raise ValueError("Forecast hours must be a positive integer.")
    frame = validate_hourly_load(history)
    if len(frame) < 24:
        raise ValueError("Recursive forecasts require at least 24 consecutive hourly observations.")
    values = frame["load"].tolist()
    times = pd.date_range(frame.index[-1] + pd.Timedelta(hours=1), periods=hours, freq="h")
    predictions = []
    for timestamp in times:
        row = pd.DataFrame([[
            timestamp.hour, timestamp.dayofweek, int(timestamp.dayofweek >= 5),
            np.sin(2 * np.pi * timestamp.hour / 24),
            np.cos(2 * np.pi * timestamp.hour / 24),
            values[-1], values[-24], float(np.mean(values[-3:])),
            float(np.mean(values[-24:])), float(np.std(values[-24:], ddof=1)),
        ]], columns=FEATURES)
        prediction = float(np.asarray(model.predict(scaler.transform(row))).reshape(-1)[0])
        if not np.isfinite(prediction) or prediction < 0:
            raise ValueError("The model produced invalid load; the forecast was not published.")
        predictions.append(prediction)
        values.append(prediction)
    return pd.DataFrame({"predicted_load_mw": predictions}, index=times)


def _model_and_parameters(tuning_mode: str, manual: dict[str, Any]) -> tuple[Any, dict[str, list[Any]], int]:
    if HAS_XGBOOST:
        defaults = dict(n_estimators=100, max_depth=6, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8, min_child_weight=1)
        model = XGBRegressor(objective="reg:squarederror", random_state=42, verbosity=0, n_jobs=1)
        distributions = dict(
            n_estimators=[100, 150, 200, 250, 300, 400], max_depth=[3, 4, 5, 6, 7, 8, 10],
            learning_rate=[0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3],
            subsample=[0.7, 0.8, 0.9, 1.0], colsample_bytree=[0.7, 0.8, 0.9, 1.0],
            min_child_weight=[1, 3, 5],
        )
        iterations = 20
    else:
        defaults = dict(n_estimators=100, max_depth=15, min_samples_split=5, min_samples_leaf=1)
        model = RandomForestRegressor(random_state=42, n_jobs=1)
        distributions = dict(n_estimators=[100, 150, 200, 300, 400], max_depth=[8, 12, 15, 20, None],
                             min_samples_split=[2, 5, 10], min_samples_leaf=[1, 2, 4])
        iterations = 12
    if tuning_mode == "manual":
        model.set_params(**{key: manual.get(key, value) for key, value in defaults.items()})
    return model, distributions, iterations


def train_load_forecast_model(
    historical_data: pd.DataFrame,
    tuning_mode: str = "auto",
    manual_params: dict[str, Any] | None = None,
) -> tuple:
    """Evaluate chronological one-step predictions, then refit for future use.

    Keep the dashboard's six-value return contract. The returned model/scaler
    use all eligible rows; reported metrics come from the earlier 80/20 split.
    Hyperparameter search and scaling see only training folds. Holdout lags use
    actual preceding observations: these scores do not measure 24-hour skill.
    """
    if tuning_mode not in {"auto", "manual"}:
        raise ValueError("Tuning mode must be 'auto' or 'manual'.")
    raw = validate_hourly_load(historical_data)
    if len(raw) < 48:
        return (None,) * 6
    frame = build_load_features(raw).dropna(subset=FEATURES)
    split_at = int(len(frame) * 0.8)
    train, test = frame.iloc[:split_at], frame.iloc[split_at:]
    x_train, y_train = train[FEATURES], train["load"]
    x_test, y_test = test[FEATURES], test["load"]
    model, distributions, iterations = _model_and_parameters(tuning_mode, manual_params or {})
    pipeline = Pipeline([("scale", StandardScaler()), ("model", model)])
    cv_mae = None
    if tuning_mode == "auto":
        search = RandomizedSearchCV(
            pipeline, {f"model__{key}": values for key, values in distributions.items()},
            n_iter=iterations, scoring="neg_mean_absolute_error",
            cv=TimeSeriesSplit(n_splits=min(5, max(2, len(train) // 10))),
            random_state=42, n_jobs=1, error_score="raise",
        )
        search.fit(x_train, y_train)
        pipeline = search.best_estimator_
        cv_mae = float(-search.best_score_)
        parameters = {key.removeprefix("model__"): value for key, value in search.best_params_.items()}
    else:
        pipeline.fit(x_train, y_train)
        parameters = {key: pipeline.named_steps["model"].get_params()[key] for key in distributions}
    train_prediction, test_prediction = pipeline.predict(x_train), pipeline.predict(x_test)
    metrics: dict[str, Any] = {}
    for name, truth, prediction in (("train", y_train, train_prediction), ("test", y_test, test_prediction)):
        metrics.update({
            f"{name}_mae": float(mean_absolute_error(truth, prediction)),
            f"{name}_rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
            f"{name}_r2": float(r2_score(truth, prediction)),
        })
    metrics.update({
        "persistence_mae": float(mean_absolute_error(y_test, x_test["load_lag_1h"])),
        "seasonal_mae": float(mean_absolute_error(y_test, x_test["load_lag_24h"])),
        "best_params": parameters, "cv_mae": cv_mae, "tuning_mode": tuning_mode,
        "evaluation": "chronological one-hour-ahead holdout with observed lag updates",
        "train_start": str(train.index[0]), "train_end": str(train.index[-1]),
        "test_start": str(test.index[0]), "test_end": str(test.index[-1]),
        "train_rows": len(train), "test_rows": len(test), "random_seed": 42,
        "forecast_refit_rows": len(frame), "forecast_origin": str(raw.index[-1]),
        "features": list(FEATURES), "units": "MW",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": hashlib.sha256(raw.to_csv().encode("utf-8")).hexdigest(),
        "method_version": "causal-hourly-v1",
        "model_type": type(model).__name__,
        "library_versions": {name: version(name) for name in (
            "numpy", "pandas", "scikit-learn", *(('xgboost',) if HAS_XGBOOST else ()),
        )},
    })
    # Freeze evaluation above before incorporating the holdout into future fits.
    production = clone(pipeline).fit(frame[FEATURES], frame["load"])
    return (production.named_steps["model"], production.named_steps["scale"], frame,
            (x_train, x_test, y_train, y_test), (train_prediction, test_prediction), metrics)
