from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastResult:
    forecast: pd.DataFrame
    metrics: dict
    model_name: str


def forecast_close_prices(
    df: pd.DataFrame,
    horizon_days: int = 30,
    lookback_window: int = 20,
    ridge_alpha: float = 10.0,
) -> ForecastResult:
    """
    Forecast future close prices from the OHLCV frame already loaded by the app.

    The model is a ridge-regularized autoregressive learner on lagged log returns.
    It deliberately avoids fetching data so it uses the same source as the chart.
    """
    close = _clean_close(df)
    horizon_days = max(1, int(horizon_days))
    lookback_window = max(5, int(lookback_window))

    log_returns = np.diff(np.log(close.to_numpy(dtype=float)))
    if len(log_returns) < lookback_window + 20:
        raise ValueError(
            f"Need at least {lookback_window + 21} clean close prices for ML forecasting."
        )

    x, y = _build_training_data(log_returns, lookback_window)
    if len(y) < 20:
        raise ValueError("Not enough training samples for ML forecasting.")

    metrics = _holdout_metrics(x, y, ridge_alpha)
    model = _fit_ridge(x, y, ridge_alpha)

    recent_returns = log_returns[-lookback_window:].copy()
    current_log_price = float(np.log(close.iloc[-1]))
    future_log_returns = []

    for _ in range(horizon_days):
        features = _features_from_window(recent_returns)
        predicted_return = float(_predict_row(features, model))
        predicted_return = float(np.clip(predicted_return, -0.15, 0.15))
        future_log_returns.append(predicted_return)

        current_log_price += predicted_return
        recent_returns = np.append(recent_returns[1:], predicted_return)

    future_log_returns = np.asarray(future_log_returns, dtype=float)
    cumulative_returns = np.cumsum(future_log_returns)
    last_log_price = float(np.log(close.iloc[-1]))
    residual_std = max(float(model["residual_std"]), 1e-6)
    steps = np.arange(1, horizon_days + 1, dtype=float)

    forecast_close = np.exp(last_log_price + cumulative_returns)
    lower_estimate = np.exp(last_log_price + cumulative_returns - 1.645 * residual_std * np.sqrt(steps))
    upper_estimate = np.exp(last_log_price + cumulative_returns + 1.645 * residual_std * np.sqrt(steps))

    forecast = pd.DataFrame(
        {
            "forecast_close": forecast_close,
            "lower_estimate": lower_estimate,
            "upper_estimate": upper_estimate,
            "expected_daily_return_pct": np.expm1(future_log_returns) * 100.0,
        },
        index=_future_index(close.index, horizon_days),
    )
    forecast.index.name = "date"

    metrics["forecast_change_pct"] = (forecast_close[-1] / float(close.iloc[-1]) - 1.0) * 100.0
    metrics["training_samples"] = int(len(y))

    return ForecastResult(
        forecast=forecast,
        metrics=metrics,
        model_name="ridge autoregressive lag model",
    )


def _clean_close(df: pd.DataFrame) -> pd.Series:
    if "close" not in df.columns:
        raise ValueError("Input data must include a 'close' column.")

    close = df["close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = pd.to_numeric(close, errors="coerce").dropna()
    close = close[close > 0].sort_index()
    if close.empty:
        raise ValueError("No positive close prices available for forecasting.")
    return close


def _build_training_data(log_returns: np.ndarray, lookback_window: int) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    targets = []

    for target_idx in range(lookback_window, len(log_returns)):
        window = log_returns[target_idx - lookback_window : target_idx]
        target = log_returns[target_idx]
        if np.isfinite(window).all() and np.isfinite(target):
            rows.append(_features_from_window(window))
            targets.append(target)

    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def _features_from_window(window: np.ndarray) -> np.ndarray:
    window = np.asarray(window, dtype=float)
    recent_5 = window[-min(5, len(window)) :]
    recent_10 = window[-min(10, len(window)) :]

    summary_features = np.asarray(
        [
            recent_5.mean(),
            recent_10.mean(),
            recent_10.std(ddof=0),
            recent_5.sum(),
            window[-1],
        ],
        dtype=float,
    )
    return np.concatenate([window, summary_features])


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std == 0] = 1.0

    x_scaled = (x - x_mean) / x_std
    design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0

    lhs = design.T @ design + float(alpha) * penalty
    rhs = design.T @ y
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs) @ rhs

    fitted = design @ beta
    residuals = y - fitted
    residual_std = residuals.std(ddof=1) if len(residuals) > 1 else residuals.std(ddof=0)

    return {
        "beta": beta,
        "x_mean": x_mean,
        "x_std": x_std,
        "residual_std": residual_std,
    }


def _predict_matrix(x: np.ndarray, model: dict) -> np.ndarray:
    x_scaled = (x - model["x_mean"]) / model["x_std"]
    design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
    return design @ model["beta"]


def _predict_row(row: np.ndarray, model: dict) -> float:
    return float(_predict_matrix(np.asarray([row], dtype=float), model)[0])


def _holdout_metrics(x: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    holdout_size = min(max(5, int(len(y) * 0.2)), len(y) - 10)
    if holdout_size < 5:
        return {}

    train_cut = len(y) - holdout_size
    model = _fit_ridge(x[:train_cut], y[:train_cut], alpha)
    predicted = np.clip(_predict_matrix(x[train_cut:], model), -0.15, 0.15)
    actual = y[train_cut:]

    predicted_pct = np.expm1(predicted) * 100.0
    actual_pct = np.expm1(actual) * 100.0

    return {
        "holdout_direction_accuracy": float((np.sign(predicted) == np.sign(actual)).mean() * 100.0),
        "holdout_mae_pct": float(np.abs(predicted_pct - actual_pct).mean()),
        "holdout_rmse_pct": float(np.sqrt(np.square(predicted_pct - actual_pct).mean())),
    }


def _future_index(index: pd.Index, horizon_days: int) -> pd.Index:
    last_value = index[-1]
    if isinstance(last_value, pd.Timestamp):
        datetime_index = pd.DatetimeIndex(index)
        has_weekend_data = bool((datetime_index.dayofweek >= 5).any())
        if has_weekend_data:
            start = last_value + pd.Timedelta(days=1)
            return pd.date_range(start=start, periods=horizon_days, freq="D")

        start = last_value + pd.tseries.offsets.BDay(1)
        return pd.bdate_range(start=start, periods=horizon_days)

    return pd.RangeIndex(start=len(index), stop=len(index) + horizon_days)
