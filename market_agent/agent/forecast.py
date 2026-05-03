from dataclasses import dataclass
from math import erf, sqrt
import warnings

import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional dependency fallback
    XGBRegressor = None

try:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPRegressor
except Exception:  # pragma: no cover - optional dependency fallback
    ConvergenceWarning = Warning
    MLPRegressor = None


@dataclass(frozen=True)
class ForecastResult:
    forecast: pd.DataFrame
    metrics: dict
    model_name: str


@dataclass(frozen=True)
class HistoricalForecastResult:
    forecasts: pd.DataFrame
    metrics: dict


def forecast_close_prices(
    df: pd.DataFrame,
    horizon_days: int = 30,
    lookback_window: int = 20,
    ridge_alpha: float = 10.0,
    optimize_model: bool = True,
    model_type: str = "ridge",
    context_df: pd.DataFrame | None = None,
) -> ForecastResult:
    """
    Forecast future close prices from the OHLCV frame already loaded by the app.

    The model is a ridge-regularized direct horizon learner on lagged log returns.
    It deliberately avoids fetching data so it uses the same source as the chart.
    """
    model_type = _normalize_model_type(model_type)
    close = _clean_close(df)
    horizon_days = max(1, int(horizon_days))
    lookback_window = max(5, int(lookback_window))

    log_returns = np.diff(np.log(close.to_numpy(dtype=float)))
    if len(log_returns) < lookback_window + horizon_days + 20:
        raise ValueError(
            f"Need at least {lookback_window + horizon_days + 21} clean close prices for ML forecasting."
        )

    extra_feature_arrays = _extra_feature_arrays(df, context_df, close)

    if optimize_model:
        selected = _select_direct_model(
            log_returns,
            horizon_days,
            lookback_window,
            ridge_alpha,
            model_type,
            extra_feature_arrays,
        )
        lookback_window = selected["lookback_window"]
        ridge_alpha = selected["ridge_alpha"]
        metrics = selected["metrics"]
    else:
        metrics = _direct_holdout_metrics(
            log_returns,
            lookback_window,
            horizon_days,
            ridge_alpha,
            model_type,
            extra_feature_arrays,
        )

    x, y = _build_direct_training_data(log_returns, lookback_window, horizon_days, extra_feature_arrays)
    if len(y) < 20:
        raise ValueError("Not enough training samples for ML forecasting.")

    model = _fit_model(x, y, ridge_alpha, model_type)
    latest_features = _features_for_position(log_returns, extra_feature_arrays, len(log_returns), lookback_window)
    raw_horizon_return = float(_predict_row(latest_features, model))
    raw_horizon_return = _clip_horizon_return(raw_horizon_return, y)
    shrink_factor = _prediction_shrink_factor(raw_horizon_return, metrics)
    horizon_return = raw_horizon_return * shrink_factor

    steps = np.arange(1, horizon_days + 1, dtype=float)
    cumulative_returns = horizon_return * (steps / float(horizon_days))
    future_log_returns = np.diff(np.concatenate([[0.0], cumulative_returns]))
    last_log_price = float(np.log(close.iloc[-1]))
    validation_rmse_pct = metrics.get("holdout_rmse_pct") or metrics.get("validation_rmse_pct")
    validation_rmse_log = np.log1p(max(float(validation_rmse_pct or 0.0), 0.0) / 100.0)
    residual_std = max(float(model["residual_std"]), validation_rmse_log, 1e-6)
    interval_scale = np.sqrt(steps / float(horizon_days))

    forecast_close = np.exp(last_log_price + cumulative_returns)
    lower_estimate = np.exp(last_log_price + cumulative_returns - 1.645 * residual_std * interval_scale)
    upper_estimate = np.exp(last_log_price + cumulative_returns + 1.645 * residual_std * interval_scale)

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

    forecast_change_pct = np.expm1(horizon_return) * 100.0
    projected_error_log = residual_std
    probability_up_pct = _normal_cdf(float(horizon_return) / projected_error_log) * 100.0
    confidence_pct = max(probability_up_pct, 100.0 - probability_up_pct)
    expected_error_pct = np.expm1(projected_error_log) * 100.0

    metrics["forecast_change_pct"] = forecast_change_pct
    metrics["probability_up_pct"] = probability_up_pct
    metrics["probability_down_pct"] = 100.0 - probability_up_pct
    metrics["confidence_pct"] = confidence_pct
    metrics["expected_error_pct"] = expected_error_pct
    metrics["forecast_score"] = _forecast_score(forecast_change_pct, expected_error_pct, confidence_pct)
    metrics["raw_forecast_change_pct"] = np.expm1(raw_horizon_return) * 100.0
    metrics["shrink_factor"] = float(shrink_factor)
    metrics["selected_lookback_window"] = int(lookback_window)
    metrics["selected_ridge_alpha"] = float(ridge_alpha)
    metrics["training_samples"] = int(len(y))

    return ForecastResult(
        forecast=forecast,
        metrics=metrics,
        model_name=_model_display_name(model_type, optimize_model),
    )


def backtest_forecasts(
    df: pd.DataFrame,
    horizon_days: int = 30,
    lookback_window: int = 20,
    ridge_alpha: float = 10.0,
    max_points: int = 60,
    min_train_samples: int = 80,
    optimize_model: bool = True,
    model_type: str = "ridge",
    context_df: pd.DataFrame | None = None,
) -> HistoricalForecastResult:
    """
    Walk-forward forecast backtest.

    Each row trains only on data available at that past date, forecasts the
    chosen horizon, then compares the forecast to the realized future close.
    """
    close = _clean_close(df)
    horizon_days = max(1, int(horizon_days))
    lookback_window = max(5, int(lookback_window))
    max_points = max(5, int(max_points))
    min_train_samples = max(20, int(min_train_samples))

    first_as_of_pos = lookback_window + min_train_samples
    last_as_of_pos = len(close) - horizon_days - 1
    if last_as_of_pos <= first_as_of_pos:
        raise ValueError("Not enough history for walk-forward ML forecast testing.")

    positions = np.arange(first_as_of_pos, last_as_of_pos + 1)
    if len(positions) > max_points:
        positions = np.unique(np.linspace(positions[0], positions[-1], max_points).round().astype(int))

    rows = []
    for as_of_pos in positions:
        train_close = close.iloc[: as_of_pos + 1]
        train_df = pd.DataFrame({"close": train_close}, index=train_close.index)
        if (model_type or "").strip().lower() == "ensemble":
            comparison = compare_forecast_models(
                train_df,
                horizon_days=horizon_days,
                lookback_window=lookback_window,
                ridge_alpha=ridge_alpha,
                optimize_model=optimize_model,
                context_df=context_df,
            )
            result = comparison["Ensemble"]
        else:
            result = forecast_close_prices(
                train_df,
                horizon_days=horizon_days,
                lookback_window=lookback_window,
                ridge_alpha=ridge_alpha,
                optimize_model=optimize_model,
                model_type=model_type,
                context_df=context_df,
            )

        as_of_price = float(close.iloc[as_of_pos])
        actual_price = float(close.iloc[as_of_pos + horizon_days])
        predicted_price = float(result.forecast["forecast_close"].iloc[-1])
        predicted_change_pct = (predicted_price / as_of_price - 1.0) * 100.0
        actual_change_pct = (actual_price / as_of_price - 1.0) * 100.0
        error_pct = (predicted_price / actual_price - 1.0) * 100.0

        rows.append(
            {
                "as_of_date": close.index[as_of_pos],
                "forecast_date": close.index[as_of_pos + horizon_days],
                "as_of_price": as_of_price,
                "forecast_close": predicted_price,
                "actual_close": actual_price,
                "predicted_change_pct": predicted_change_pct,
                "actual_change_pct": actual_change_pct,
                "error_pct": error_pct,
                "abs_error_pct": abs(error_pct),
                "probability_up_pct": result.metrics.get("probability_up_pct", np.nan),
                "confidence_pct": result.metrics.get("confidence_pct", np.nan),
                "expected_error_pct": result.metrics.get("expected_error_pct", np.nan),
                "selected_lookback_window": result.metrics.get("selected_lookback_window", np.nan),
                "selected_ridge_alpha": result.metrics.get("selected_ridge_alpha", np.nan),
                "direction_correct": np.sign(predicted_change_pct) == np.sign(actual_change_pct),
            }
        )

    forecasts = pd.DataFrame(rows)
    metrics = {
        "historical_points": int(len(forecasts)),
        "historical_direction_accuracy": float(forecasts["direction_correct"].mean() * 100.0),
        "historical_mae_pct": float(forecasts["abs_error_pct"].mean()),
        "historical_rmse_pct": float(np.sqrt(np.square(forecasts["error_pct"]).mean())),
        "historical_bias_pct": float(forecasts["error_pct"].mean()),
        "within_expected_error_pct": float(
            (forecasts["abs_error_pct"] <= forecasts["expected_error_pct"]).mean() * 100.0
        ),
    }

    return HistoricalForecastResult(forecasts=forecasts, metrics=metrics)


def compare_forecast_models(
    df: pd.DataFrame,
    horizon_days: int = 30,
    lookback_window: int = 20,
    ridge_alpha: float = 10.0,
    optimize_model: bool = True,
    context_df: pd.DataFrame | None = None,
) -> dict[str, ForecastResult]:
    """Run Ridge, XGBoost, Neural Net, and Ensemble forecasts on the same feature set."""
    results = {}
    results["Ridge"] = forecast_close_prices(
        df,
        horizon_days=horizon_days,
        lookback_window=lookback_window,
        ridge_alpha=ridge_alpha,
        optimize_model=optimize_model,
        model_type="ridge",
        context_df=context_df,
    )

    if XGBRegressor is not None:
        try:
            results["XGBoost"] = forecast_close_prices(
                df,
                horizon_days=horizon_days,
                lookback_window=lookback_window,
                ridge_alpha=ridge_alpha,
                optimize_model=optimize_model,
                model_type="xgboost",
                context_df=context_df,
            )
        except Exception as exc:
            results["XGBoost"] = _unavailable_result("xgboost unavailable", exc)
    else:
        results["XGBoost"] = _unavailable_result("xgboost unavailable", "xgboost package is not installed")

    if MLPRegressor is not None:
        try:
            results["Neural Net"] = forecast_close_prices(
                df,
                horizon_days=horizon_days,
                lookback_window=lookback_window,
                ridge_alpha=ridge_alpha,
                optimize_model=optimize_model,
                model_type="neural_net",
                context_df=context_df,
            )
        except Exception as exc:
            results["Neural Net"] = _unavailable_result("neural net unavailable", exc)
    else:
        results["Neural Net"] = _unavailable_result("neural net unavailable", "scikit-learn package is not installed")

    available_components = {
        name: result
        for name, result in results.items()
        if result.forecast is not None and not result.forecast.empty
    }
    if available_components:
        results["Ensemble"] = _ensemble_result(available_components)

    return results


def _unavailable_result(model_name: str, error: Exception | str) -> ForecastResult:
    return ForecastResult(
        forecast=pd.DataFrame(),
        metrics={"error": str(error), "forecast_score": np.nan},
        model_name=model_name,
    )


def _ensemble_result(components: dict[str, ForecastResult]) -> ForecastResult:
    usable = {
        name: result
        for name, result in components.items()
        if result.forecast is not None and not result.forecast.empty
    }
    if not usable:
        return _unavailable_result("ensemble unavailable", "No component forecasts available.")

    raw_weights = {}
    for name, result in usable.items():
        mae = result.metrics.get("holdout_mae_pct") or result.metrics.get("expected_error_pct") or 25.0
        raw_weights[name] = 1.0 / max(float(mae), 0.25)

    weight_sum = sum(raw_weights.values())
    weights = {name: value / weight_sum for name, value in raw_weights.items()}
    first = next(iter(usable.values())).forecast.copy()
    forecast = pd.DataFrame(index=first.index)

    for column in ["forecast_close", "lower_estimate", "upper_estimate", "expected_daily_return_pct"]:
        forecast[column] = sum(
            weights[name] * result.forecast[column].reindex(first.index)
            for name, result in usable.items()
        )

    probability_up = sum(weights[name] * result.metrics.get("probability_up_pct", 50.0) for name, result in usable.items())
    confidence = max(probability_up, 100.0 - probability_up)
    expected_error = sum(weights[name] * result.metrics.get("expected_error_pct", 0.0) for name, result in usable.items())
    forecast_change = sum(weights[name] * result.metrics.get("forecast_change_pct", 0.0) for name, result in usable.items())
    holdout_mae = sum(weights[name] * result.metrics.get("holdout_mae_pct", expected_error) for name, result in usable.items())
    holdout_rmse = sum(weights[name] * result.metrics.get("holdout_rmse_pct", expected_error) for name, result in usable.items())
    direction_accuracy = sum(
        weights[name] * result.metrics.get("holdout_direction_accuracy", 50.0)
        for name, result in usable.items()
    )

    metrics = {
        "forecast_change_pct": float(forecast_change),
        "probability_up_pct": float(probability_up),
        "probability_down_pct": float(100.0 - probability_up),
        "confidence_pct": float(confidence),
        "expected_error_pct": float(expected_error),
        "forecast_score": _forecast_score(forecast_change, expected_error, confidence),
        "holdout_mae_pct": float(holdout_mae),
        "holdout_rmse_pct": float(holdout_rmse),
        "holdout_direction_accuracy": float(direction_accuracy),
        "component_weights": {name: round(weight, 3) for name, weight in weights.items()},
    }

    return ForecastResult(
        forecast=forecast,
        metrics=metrics,
        model_name="weighted ensemble forecast model",
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


def _normalize_model_type(model_type: str) -> str:
    model_type = (model_type or "ridge").strip().lower()
    if model_type in {"ridge", "linear"}:
        return "ridge"
    if model_type in {"xgboost", "xgb"}:
        if XGBRegressor is None:
            raise ValueError("XGBoost package is not installed.")
        return "xgboost"
    if model_type in {"neural", "neural_net", "neural net", "mlp", "nn"}:
        if MLPRegressor is None:
            raise ValueError("scikit-learn package is not installed.")
        return "neural_net"
    raise ValueError(f"Unsupported model type: {model_type}")


def _model_display_name(model_type: str, optimize_model: bool) -> str:
    prefix = "optimized " if optimize_model else ""
    if model_type == "xgboost":
        return f"{prefix}direct XGBoost horizon model"
    if model_type == "neural_net":
        return f"{prefix}direct neural net horizon model"
    return f"{prefix}direct ridge horizon model"


def _extra_feature_arrays(
    df: pd.DataFrame,
    context_df: pd.DataFrame | None,
    close: pd.Series,
) -> list[np.ndarray]:
    arrays = []
    aligned_df = df.reindex(close.index)

    for column in ["volume", "high", "low", "open"]:
        if column in aligned_df.columns:
            series = aligned_df[column]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            series = pd.to_numeric(series, errors="coerce").ffill()
            if column == "volume":
                transformed = np.diff(np.log1p(series.clip(lower=0).to_numpy(dtype=float)))
            elif column in {"high", "low", "open"}:
                ratio = (series / close).replace([np.inf, -np.inf], np.nan).ffill()
                transformed = ratio.iloc[1:].to_numpy(dtype=float)
            else:
                continue
            if len(transformed) == len(close) - 1 and np.isfinite(transformed).any():
                arrays.append(np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0))

    if context_df is not None and not context_df.empty:
        context = context_df.reindex(close.index).ffill()
        for column in context.columns:
            series = pd.to_numeric(context[column], errors="coerce").ffill()
            if series.notna().sum() < 10:
                continue
            values = series.to_numpy(dtype=float)
            if np.nanmin(values) > 0:
                transformed = np.diff(np.log(values))
            else:
                transformed = np.diff(values)
            if len(transformed) == len(close) - 1 and np.isfinite(transformed).any():
                arrays.append(np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0))

    return arrays


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


def _build_direct_training_data(
    log_returns: np.ndarray,
    lookback_window: int,
    horizon_days: int,
    extra_feature_arrays: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    targets = []
    last_start = len(log_returns) - horizon_days

    for as_of_pos in range(lookback_window, last_start + 1):
        features = _features_for_position(log_returns, extra_feature_arrays, as_of_pos, lookback_window)
        target = log_returns[as_of_pos : as_of_pos + horizon_days].sum()
        if np.isfinite(features).all() and np.isfinite(target):
            rows.append(features)
            targets.append(target)

    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def _features_for_position(
    log_returns: np.ndarray,
    extra_feature_arrays: list[np.ndarray] | None,
    as_of_pos: int,
    lookback_window: int,
) -> np.ndarray:
    start = as_of_pos - lookback_window
    target_window = log_returns[start:as_of_pos]
    features = [_features_from_window(target_window)]
    for array in extra_feature_arrays or []:
        if len(array) >= as_of_pos:
            features.append(_summary_features(array[start:as_of_pos]))
    return np.concatenate(features)


def _features_from_window(window: np.ndarray) -> np.ndarray:
    window = np.asarray(window, dtype=float)
    return np.concatenate([window, _summary_features(window)])


def _summary_features(window: np.ndarray) -> np.ndarray:
    window = np.asarray(window, dtype=float)
    recent_5 = window[-min(5, len(window)) :]
    recent_10 = window[-min(10, len(window)) :]
    recent_20 = window[-min(20, len(window)) :]
    volatility = recent_20.std(ddof=0)
    downside_volatility = np.minimum(recent_20, 0.0).std(ddof=0)
    z_score = 0.0 if volatility == 0 else recent_5.mean() / volatility

    summary_features = np.asarray(
        [
            recent_5.mean(),
            recent_10.mean(),
            recent_10.std(ddof=0),
            recent_20.mean(),
            volatility,
            downside_volatility,
            z_score,
            float((recent_20 > 0).mean()),
            recent_5.sum(),
            recent_10.sum(),
            recent_20.sum(),
            window[-1],
        ],
        dtype=float,
    )
    return summary_features


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
        "kind": "ridge",
        "beta": beta,
        "x_mean": x_mean,
        "x_std": x_std,
        "residual_std": residual_std,
    }


def _fit_xgboost(x: np.ndarray, y: np.ndarray, reg_lambda: float) -> dict:
    if XGBRegressor is None:
        raise ValueError("XGBoost package is not installed.")

    model = XGBRegressor(
        n_estimators=90,
        max_depth=2,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_lambda=float(reg_lambda),
        reg_alpha=0.0,
        objective="reg:squarederror",
        random_state=7,
        n_jobs=1,
    )
    model.fit(x, y)
    fitted = model.predict(x)
    residuals = y - fitted
    residual_std = residuals.std(ddof=1) if len(residuals) > 1 else residuals.std(ddof=0)
    return {
        "kind": "xgboost",
        "model": model,
        "residual_std": residual_std,
    }


def _fit_neural_net(x: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    if MLPRegressor is None:
        raise ValueError("scikit-learn package is not installed.")

    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std == 0] = 1.0
    x_scaled = (x - x_mean) / x_std

    y_mean = float(y.mean())
    y_std = float(y.std(ddof=0))
    if y_std == 0.0:
        y_std = 1.0
    y_scaled = (y - y_mean) / y_std

    model = MLPRegressor(
        hidden_layer_sizes=(24, 12),
        activation="relu",
        solver="adam",
        alpha=max(float(alpha), 0.01) / 10000.0,
        learning_rate_init=0.003,
        max_iter=300,
        random_state=11,
        shuffle=False,
        early_stopping=False,
        tol=1e-4,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_scaled, y_scaled)

    fitted = model.predict(x_scaled) * y_std + y_mean
    residuals = y - fitted
    residual_std = residuals.std(ddof=1) if len(residuals) > 1 else residuals.std(ddof=0)
    return {
        "kind": "neural_net",
        "model": model,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "residual_std": residual_std,
    }


def _fit_model(x: np.ndarray, y: np.ndarray, alpha: float, model_type: str) -> dict:
    if model_type == "xgboost":
        return _fit_xgboost(x, y, alpha)
    if model_type == "neural_net":
        return _fit_neural_net(x, y, alpha)
    return _fit_ridge(x, y, alpha)


def _predict_matrix(x: np.ndarray, model: dict) -> np.ndarray:
    if model.get("kind") == "xgboost":
        return model["model"].predict(x)

    if model.get("kind") == "neural_net":
        x_scaled = (x - model["x_mean"]) / model["x_std"]
        return model["model"].predict(x_scaled) * model["y_std"] + model["y_mean"]

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


def _select_direct_model(
    log_returns: np.ndarray,
    horizon_days: int,
    requested_lookback: int,
    requested_alpha: float,
    model_type: str,
    extra_feature_arrays: list[np.ndarray] | None = None,
) -> dict:
    candidates = []
    max_window = max(5, min(90, (len(log_returns) - horizon_days - 25) // 2))
    window_candidates = _candidate_values(
        [5, 10, 20, 30, 45, 60, requested_lookback, requested_lookback // 2, requested_lookback * 2],
        lower=5,
        upper=max_window,
        cast=int,
    )
    alpha_candidates = _candidate_values(
        [0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, requested_alpha],
        lower=0.01,
        upper=250.0,
        cast=float,
    )
    if model_type == "xgboost":
        window_candidates = _candidate_values(
            [10, 20, 30, 45, requested_lookback],
            lower=5,
            upper=max_window,
            cast=int,
        )
        alpha_candidates = _candidate_values(
            [5.0, 25.0, requested_alpha],
            lower=0.01,
            upper=250.0,
            cast=float,
        )
    if model_type == "neural_net":
        window_candidates = _candidate_values(
            [requested_lookback, 30],
            lower=5,
            upper=max_window,
            cast=int,
        )
        alpha_candidates = _candidate_values(
            [requested_alpha],
            lower=0.01,
            upper=250.0,
            cast=float,
        )

    for window in window_candidates:
        for alpha in alpha_candidates:
            metrics = _direct_holdout_metrics(
                log_returns,
                window,
                horizon_days,
                alpha,
                model_type,
                extra_feature_arrays,
            )
            if metrics:
                candidates.append(
                    {
                        "lookback_window": window,
                        "ridge_alpha": alpha,
                        "metrics": metrics,
                    }
                )

    if not candidates:
        fallback_metrics = _direct_holdout_metrics(
            log_returns,
            requested_lookback,
            horizon_days,
            requested_alpha,
            model_type,
            extra_feature_arrays,
        )
        return {
            "lookback_window": requested_lookback,
            "ridge_alpha": requested_alpha,
            "metrics": fallback_metrics,
        }

    return min(
        candidates,
        key=lambda item: (
            item["metrics"].get("holdout_mae_pct", np.inf),
            item["metrics"].get("holdout_rmse_pct", np.inf),
            -item["metrics"].get("holdout_direction_accuracy", 0.0),
        ),
    )


def _direct_holdout_metrics(
    log_returns: np.ndarray,
    lookback_window: int,
    horizon_days: int,
    alpha: float,
    model_type: str = "ridge",
    extra_feature_arrays: list[np.ndarray] | None = None,
) -> dict:
    x, y = _build_direct_training_data(log_returns, lookback_window, horizon_days, extra_feature_arrays)
    if len(y) < 35:
        return {}

    holdout_size = min(max(8, int(len(y) * 0.25)), len(y) - 20)
    if holdout_size < 8:
        return {}

    train_cut = len(y) - holdout_size
    model = _fit_model(x[:train_cut], y[:train_cut], alpha, model_type)
    predicted = np.asarray(
        [_clip_horizon_return(value, y[:train_cut]) for value in _predict_matrix(x[train_cut:], model)],
        dtype=float,
    )
    actual = y[train_cut:]

    predicted_pct = np.expm1(predicted) * 100.0
    actual_pct = np.expm1(actual) * 100.0
    errors = predicted_pct - actual_pct
    directional_accuracy = float((np.sign(predicted) == np.sign(actual)).mean() * 100.0)

    return {
        "holdout_direction_accuracy": directional_accuracy,
        "holdout_mae_pct": float(np.abs(errors).mean()),
        "holdout_rmse_pct": float(np.sqrt(np.square(errors).mean())),
        "holdout_bias_pct": float(errors.mean()),
        "holdout_samples": int(len(actual)),
    }


def _candidate_values(values: list, lower: float, upper: float, cast) -> list:
    candidates = []
    for value in values:
        if value is None:
            continue
        cast_value = cast(value)
        if lower <= cast_value <= upper and cast_value not in candidates:
            candidates.append(cast_value)
    return sorted(candidates)


def _clip_horizon_return(prediction: float, target_returns: np.ndarray) -> float:
    target_returns = np.asarray(target_returns, dtype=float)
    if len(target_returns) < 10:
        return float(np.clip(prediction, -0.35, 0.35))

    lower, upper = np.quantile(target_returns, [0.02, 0.98])
    return float(np.clip(prediction, lower, upper))


def _prediction_shrink_factor(prediction: float, metrics: dict) -> float:
    mae_pct = max(float(metrics.get("holdout_mae_pct", 0.0)), 0.25)
    direction_accuracy = float(metrics.get("holdout_direction_accuracy", 50.0))
    predicted_pct = abs(np.expm1(prediction) * 100.0)
    signal_to_error = predicted_pct / mae_pct
    signal_strength = signal_to_error / (1.0 + signal_to_error)
    directional_strength = np.clip((direction_accuracy - 45.0) / 25.0, 0.0, 1.0)
    return float(np.clip(0.35 + 0.45 * signal_strength + 0.20 * directional_strength, 0.25, 1.0))


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _forecast_score(forecast_change_pct: float, expected_error_pct: float, confidence_pct: float) -> float:
    error_floor = max(abs(expected_error_pct), 0.25)
    edge_weight = max(float(confidence_pct) - 50.0, 0.0) / 50.0
    return float((forecast_change_pct / error_floor) * edge_weight)


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
