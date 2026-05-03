from dataclasses import dataclass
import hashlib
from math import erf, sqrt
import os
from pathlib import Path
import pickle
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

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency fallback
    torch = None
    nn = None


MODEL_WEIGHT_ARTIFACT_VERSION = 1


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
    symbol: str | None = None,
    warm_start: bool = True,
    force_retrain: bool = False,
    model_artifact_dir: str | Path | None = None,
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

    x, y = _training_data_for_model(log_returns, lookback_window, horizon_days, model_type, extra_feature_arrays)
    if len(y) < 20:
        raise ValueError("Not enough training samples for ML forecasting.")

    model = _fit_model(
        x,
        y,
        ridge_alpha,
        model_type,
        symbol=symbol,
        horizon_days=horizon_days,
        lookback_window=lookback_window,
        optimize_model=optimize_model,
        warm_start=warm_start,
        force_retrain=force_retrain,
        artifact_dir=model_artifact_dir,
    )
    latest_features = _latest_features_for_model(log_returns, extra_feature_arrays, len(log_returns), lookback_window, model_type)
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
    metrics.update(model.get("training_metadata", {}))

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
    sequence_model: str = "off",
    symbol: str | None = None,
    warm_start: bool = True,
    force_retrain: bool = False,
    model_artifact_dir: str | Path | None = None,
    include_rl: bool = False,
) -> dict[str, ForecastResult]:
    """Run baseline models plus an optional LSTM/Transformer sequence model."""
    results = {}
    results["Ridge"] = forecast_close_prices(
        df,
        horizon_days=horizon_days,
        lookback_window=lookback_window,
        ridge_alpha=ridge_alpha,
        optimize_model=optimize_model,
        model_type="ridge",
        context_df=context_df,
        symbol=symbol,
        warm_start=warm_start,
        force_retrain=force_retrain,
        model_artifact_dir=model_artifact_dir,
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
                symbol=symbol,
                warm_start=warm_start,
                force_retrain=force_retrain,
                model_artifact_dir=model_artifact_dir,
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
                symbol=symbol,
                warm_start=warm_start,
                force_retrain=force_retrain,
                model_artifact_dir=model_artifact_dir,
            )
        except Exception as exc:
            results["Neural Net"] = _unavailable_result("neural net unavailable", exc)
    else:
        results["Neural Net"] = _unavailable_result("neural net unavailable", "scikit-learn package is not installed")

    for model_key, display_name in _sequence_model_choices(sequence_model):
        try:
            results[display_name] = forecast_close_prices(
                df,
                horizon_days=horizon_days,
                lookback_window=lookback_window,
                ridge_alpha=ridge_alpha,
                optimize_model=optimize_model,
                model_type=model_key,
                context_df=context_df,
                symbol=symbol,
                warm_start=warm_start,
                force_retrain=force_retrain,
                model_artifact_dir=model_artifact_dir,
            )
        except Exception as exc:
            results[display_name] = _unavailable_result(f"{display_name.lower()} unavailable", exc)

    if include_rl:
        try:
            results["RL Policy"] = reinforcement_policy_forecast(
                df,
                horizon_days=horizon_days,
                lookback_window=lookback_window,
                symbol=symbol,
                warm_start=warm_start,
                force_retrain=force_retrain,
                artifact_dir=model_artifact_dir,
            )
        except Exception as exc:
            results["RL Policy"] = _unavailable_result("rl policy unavailable", exc)

    available_components = {
        name: result
        for name, result in results.items()
        if result.forecast is not None and not result.forecast.empty
    }
    if available_components:
        results["Ensemble"] = _ensemble_result(available_components)

    return results


def reinforcement_policy_forecast(
    df: pd.DataFrame,
    horizon_days: int = 30,
    lookback_window: int = 20,
    symbol: str | None = None,
    warm_start: bool = True,
    force_retrain: bool = False,
    artifact_dir: str | Path | None = None,
) -> ForecastResult:
    close = _clean_close(df)
    horizon_days = max(1, int(horizon_days))
    lookback_window = max(5, int(lookback_window))
    log_returns = np.diff(np.log(close.to_numpy(dtype=float)))
    states, targets = _rl_training_data(log_returns, lookback_window, horizon_days)
    if len(targets) < 35:
        raise ValueError("Not enough training samples for RL policy forecasting.")

    artifact_path = None
    artifact = {}
    if warm_start and symbol:
        artifact_path = _model_weight_artifact_path(
            symbol=symbol,
            model_type="rl_policy",
            horizon_days=horizon_days,
            lookback_window=lookback_window,
            alpha=0.0,
            optimize_model=False,
            x_shape=states.shape,
            artifact_dir=artifact_dir,
        )
        if not force_retrain:
            artifact = _load_model_weight_artifact(artifact_path)

    q_table = {}
    previous_samples = 0
    mode = "full_fit"
    if artifact and _artifact_training_prefix_matches(artifact, states, targets):
        q_table = {
            key: np.asarray(value, dtype=float)
            for key, value in (artifact.get("q_table") or {}).items()
        }
        previous_samples = int(artifact.get("training_samples", 0) or 0)
        mode = "incremental_q_update" if len(targets) > previous_samples else "reused_artifact"

    start_index = previous_samples if q_table and previous_samples < len(targets) else 0 if not q_table else len(targets)
    if start_index < len(targets):
        _update_rl_q_table(q_table, states, targets, start_index=start_index)

    latest_state = _rl_state_from_window(log_returns[-lookback_window:])
    q_values = q_table.get(_rl_state_key(latest_state), np.zeros(3, dtype=float))
    action_index = int(np.argmax(q_values))
    action_name = ["short", "hold", "long"][action_index]
    if action_index == 2:
        raw_horizon_return = float(q_values[2])
    elif action_index == 0:
        raw_horizon_return = float(-q_values[0])
    else:
        raw_horizon_return = 0.0
    horizon_return = _clip_horizon_return(raw_horizon_return, targets)

    predicted_targets = np.asarray([_rl_forecast_for_state(q_table, state) for state in states], dtype=float)
    errors_pct = (np.expm1(predicted_targets) - np.expm1(targets)) * 100.0
    holdout_size = min(max(8, int(len(targets) * 0.25)), len(targets) - 20)
    holdout_errors = errors_pct[-holdout_size:] if holdout_size > 0 else errors_pct
    holdout_predicted = predicted_targets[-holdout_size:] if holdout_size > 0 else predicted_targets
    holdout_actual = targets[-holdout_size:] if holdout_size > 0 else targets
    residual_std = float(np.std(targets - predicted_targets, ddof=1)) if len(targets) > 1 else 1e-6
    residual_std = max(residual_std, 1e-6)

    metrics = {
        "holdout_direction_accuracy": float((np.sign(holdout_predicted) == np.sign(holdout_actual)).mean() * 100.0),
        "holdout_mae_pct": float(np.abs(holdout_errors).mean()),
        "holdout_rmse_pct": float(np.sqrt(np.square(holdout_errors).mean())),
        "holdout_bias_pct": float(holdout_errors.mean()),
        "holdout_samples": int(len(holdout_actual)),
        "selected_lookback_window": int(lookback_window),
        "selected_ridge_alpha": 0.0,
        "training_samples": int(len(targets)),
        "model_training_mode": mode,
        "warm_start_used": bool(q_table and previous_samples > 0),
        "new_labeled_samples": int(max(len(targets) - previous_samples, 0)) if previous_samples else int(len(targets)),
        "rl_action": action_name,
        "rl_q_short": float(q_values[0]),
        "rl_q_hold": float(q_values[1]),
        "rl_q_long": float(q_values[2]),
    }
    result = _forecast_from_horizon_return(
        close=close,
        horizon_days=horizon_days,
        horizon_return=horizon_return,
        raw_horizon_return=raw_horizon_return,
        residual_std=residual_std,
        metrics=metrics,
        model_name="warm-start tabular Q-learning trading policy",
    )

    if artifact_path is not None:
        model = {
            "kind": "rl_policy",
            "q_table": {key: value.tolist() for key, value in q_table.items()},
            "residual_std": residual_std,
            "training_metadata": result.metrics,
        }
        _save_model_weight_artifact(artifact_path, model, states, targets, symbol or "", "rl_policy")
        result.metrics["model_artifact_path"] = str(artifact_path)
    return result


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
    if model_type in {"lstm", "rnn"}:
        if torch is None or nn is None:
            raise ValueError("PyTorch package is not installed.")
        return "lstm"
    if model_type in {"transformer", "tft"}:
        if torch is None or nn is None:
            raise ValueError("PyTorch package is not installed.")
        return "transformer"
    raise ValueError(f"Unsupported model type: {model_type}")


def _normalize_sequence_model(sequence_model: str | None) -> str:
    sequence_model = (sequence_model or "off").strip().lower()
    if sequence_model in {"", "off", "none", "false", "0"}:
        return "off"
    if sequence_model in {"lstm", "rnn"}:
        return "lstm"
    if sequence_model in {"transformer", "tft"}:
        return "transformer"
    if sequence_model in {"both", "all", "lstm+transformer", "transformer+lstm"}:
        return "both"
    raise ValueError(f"Unsupported sequence model: {sequence_model}")


def _sequence_model_choices(sequence_model: str | None) -> list[tuple[str, str]]:
    sequence_model = _normalize_sequence_model(sequence_model)
    if sequence_model == "lstm":
        return [("lstm", "LSTM")]
    if sequence_model == "transformer":
        return [("transformer", "Transformer")]
    if sequence_model == "both":
        return [("lstm", "LSTM"), ("transformer", "Transformer")]
    return []


def _model_display_name(model_type: str, optimize_model: bool) -> str:
    prefix = "optimized " if optimize_model else ""
    if model_type == "xgboost":
        return f"{prefix}direct XGBoost horizon model"
    if model_type == "neural_net":
        return f"{prefix}direct neural net horizon model"
    if model_type == "lstm":
        return f"{prefix}direct LSTM sequence horizon model"
    if model_type == "transformer":
        return f"{prefix}direct transformer sequence horizon model"
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


def _is_sequence_model(model_type: str) -> bool:
    return model_type in {"lstm", "transformer"}


def _training_data_for_model(
    log_returns: np.ndarray,
    lookback_window: int,
    horizon_days: int,
    model_type: str,
    extra_feature_arrays: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if _is_sequence_model(model_type):
        return _build_sequence_training_data(log_returns, lookback_window, horizon_days, extra_feature_arrays)
    return _build_direct_training_data(log_returns, lookback_window, horizon_days, extra_feature_arrays)


def _latest_features_for_model(
    log_returns: np.ndarray,
    extra_feature_arrays: list[np.ndarray] | None,
    as_of_pos: int,
    lookback_window: int,
    model_type: str,
) -> np.ndarray:
    if _is_sequence_model(model_type):
        return _sequence_features_for_position(log_returns, extra_feature_arrays, as_of_pos, lookback_window)
    return _features_for_position(log_returns, extra_feature_arrays, as_of_pos, lookback_window)


def _build_sequence_training_data(
    log_returns: np.ndarray,
    lookback_window: int,
    horizon_days: int,
    extra_feature_arrays: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    targets = []
    last_start = len(log_returns) - horizon_days

    for as_of_pos in range(lookback_window, last_start + 1):
        features = _sequence_features_for_position(log_returns, extra_feature_arrays, as_of_pos, lookback_window)
        target = log_returns[as_of_pos : as_of_pos + horizon_days].sum()
        if np.isfinite(features).all() and np.isfinite(target):
            rows.append(features)
            targets.append(target)

    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def _sequence_features_for_position(
    log_returns: np.ndarray,
    extra_feature_arrays: list[np.ndarray] | None,
    as_of_pos: int,
    lookback_window: int,
) -> np.ndarray:
    start = as_of_pos - lookback_window
    columns = [np.asarray(log_returns[start:as_of_pos], dtype=float)]
    for array in extra_feature_arrays or []:
        if len(array) >= as_of_pos:
            columns.append(np.asarray(array[start:as_of_pos], dtype=float))
    return np.column_stack(columns)


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


def _model_weight_root(artifact_dir: str | Path | None = None) -> Path:
    configured = artifact_dir or os.getenv("MARKET_AGENT_MODEL_WEIGHT_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[1] / "reports" / "model_weights"


def _model_weight_artifact_path(
    symbol: str,
    model_type: str,
    horizon_days: int | None,
    lookback_window: int | None,
    alpha: float,
    optimize_model: bool,
    x_shape: tuple[int, ...],
    artifact_dir: str | Path | None = None,
) -> Path:
    feature_shape = tuple(int(value) for value in x_shape[1:])
    payload = {
        "artifact_version": MODEL_WEIGHT_ARTIFACT_VERSION,
        "symbol": str(symbol),
        "model_type": str(model_type),
        "horizon_days": int(horizon_days or 0),
        "lookback_window": int(lookback_window or 0),
        "alpha": round(float(alpha), 8),
        "optimize_model": bool(optimize_model),
        "feature_shape": feature_shape,
    }
    digest = hashlib.sha256(repr(sorted(payload.items())).encode("utf-8")).hexdigest()[:18]
    safe_symbol = "".join(ch if ch.isalnum() else "_" for ch in str(symbol).upper()).strip("_") or "SYMBOL"
    return _model_weight_root(artifact_dir) / safe_symbol / f"{model_type}_{digest}.pkl"


def _training_hash(x: np.ndarray, y: np.ndarray) -> str:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(x_arr.shape).encode("utf-8"))
    digest.update(np.ascontiguousarray(np.round(x_arr, 12)).tobytes())
    digest.update(str(y_arr.shape).encode("utf-8"))
    digest.update(np.ascontiguousarray(np.round(y_arr, 12)).tobytes())
    return digest.hexdigest()


def _artifact_training_prefix_matches(artifact: dict | None, x: np.ndarray, y: np.ndarray) -> bool:
    if not artifact:
        return False
    previous_samples = int(artifact.get("training_samples", 0) or 0)
    if previous_samples <= 0 or previous_samples > len(y):
        return False
    if artifact.get("feature_shape") != tuple(int(value) for value in np.asarray(x).shape[1:]):
        return False
    previous_hash = artifact.get("training_hash")
    if not previous_hash:
        return False
    return previous_hash == _training_hash(x[:previous_samples], y[:previous_samples])


def _load_model_weight_artifact(path: Path) -> dict:
    try:
        if not path.exists():
            return {}
        with path.open("rb") as handle:
            artifact = pickle.load(handle)
        if int(artifact.get("artifact_version", 0)) != MODEL_WEIGHT_ARTIFACT_VERSION:
            return {}
        return artifact
    except Exception:
        return {}


def _save_model_weight_artifact(
    path: Path,
    model: dict,
    x: np.ndarray,
    y: np.ndarray,
    symbol: str,
    model_type: str,
) -> None:
    artifact = {
        "artifact_version": MODEL_WEIGHT_ARTIFACT_VERSION,
        "symbol": str(symbol),
        "model_type": str(model_type),
        "feature_shape": tuple(int(value) for value in np.asarray(x).shape[1:]),
        "training_samples": int(len(y)),
        "training_hash": _training_hash(x, y),
        "residual_std": float(model.get("residual_std", 0.0)),
        "training_metadata": dict(model.get("training_metadata", {})),
    }

    for key in [
        "kind",
        "beta",
        "x_mean",
        "x_std",
        "y_mean",
        "y_std",
        "ridge_stats",
        "sequence_length",
        "feature_count",
        "model",
        "q_table",
        "rl_counts",
    ]:
        if key in model:
            artifact[key] = model[key]

    if torch is not None and model.get("kind") in {"lstm", "transformer"} and "model" in model:
        artifact["state_dict"] = {
            key: value.detach().cpu()
            for key, value in model["model"].state_dict().items()
        }
        artifact.pop("model", None)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(artifact, handle)


def _ridge_stats_from_xy(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return {
        "n": int(len(y)),
        "x_sum": x.sum(axis=0),
        "x_square_sum": np.square(x).sum(axis=0),
        "xtx_raw": x.T @ x,
        "xty_raw": x.T @ y,
        "y_sum": float(y.sum()),
        "yy_sum": float(np.dot(y, y)),
    }


def _append_ridge_stats(stats: dict, x: np.ndarray, y: np.ndarray) -> dict:
    updated = {key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value for key, value in stats.items()}
    tail_stats = _ridge_stats_from_xy(x, y)
    updated["n"] = int(updated["n"]) + int(tail_stats["n"])
    for key in ["x_sum", "x_square_sum", "xtx_raw", "xty_raw"]:
        updated[key] = updated[key] + tail_stats[key]
    updated["y_sum"] = float(updated["y_sum"]) + float(tail_stats["y_sum"])
    updated["yy_sum"] = float(updated["yy_sum"]) + float(tail_stats["yy_sum"])
    return updated


def _fit_ridge_from_stats(stats: dict, alpha: float) -> dict:
    n = int(stats["n"])
    x_sum = np.asarray(stats["x_sum"], dtype=float)
    x_square_sum = np.asarray(stats["x_square_sum"], dtype=float)
    xtx_raw = np.asarray(stats["xtx_raw"], dtype=float)
    xty_raw = np.asarray(stats["xty_raw"], dtype=float)
    y_sum = float(stats["y_sum"])
    yy_sum = float(stats["yy_sum"])

    x_mean = x_sum / float(n)
    variance = np.maximum(x_square_sum / float(n) - np.square(x_mean), 0.0)
    x_std = np.sqrt(variance)
    x_std[x_std == 0] = 1.0

    centered_xtx = xtx_raw - np.outer(x_sum, x_mean) - np.outer(x_mean, x_sum) + n * np.outer(x_mean, x_mean)
    scaled_xtx = centered_xtx / np.outer(x_std, x_std)
    centered_xty = xty_raw - x_mean * y_sum
    scaled_xty = centered_xty / x_std

    ztz = np.empty((scaled_xtx.shape[0] + 1, scaled_xtx.shape[1] + 1), dtype=float)
    ztz[0, 0] = n
    ztz[0, 1:] = 0.0
    ztz[1:, 0] = 0.0
    ztz[1:, 1:] = scaled_xtx
    zty = np.concatenate([[y_sum], scaled_xty])
    penalty = np.eye(ztz.shape[0])
    penalty[0, 0] = 0.0
    lhs = ztz + float(alpha) * penalty
    try:
        beta = np.linalg.solve(lhs, zty)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs) @ zty

    sse = yy_sum - 2.0 * float(beta @ zty) + float(beta @ ztz @ beta)
    denominator = max(n - 1, 1)
    residual_std = float(np.sqrt(max(sse / denominator, 0.0)))
    return {
        "kind": "ridge",
        "beta": beta,
        "x_mean": x_mean,
        "x_std": x_std,
        "residual_std": residual_std,
        "ridge_stats": stats,
    }


def _fit_ridge_warm_start(x: np.ndarray, y: np.ndarray, alpha: float, artifact: dict) -> dict | None:
    if artifact.get("kind") != "ridge":
        return None
    if not _artifact_training_prefix_matches(artifact, x, y):
        return None
    previous_samples = int(artifact.get("training_samples", 0) or 0)
    stats = artifact.get("ridge_stats")
    if not stats:
        return None
    if previous_samples < len(y):
        stats = _append_ridge_stats(stats, x[previous_samples:], y[previous_samples:])
        mode = "incremental_stats_update"
    else:
        mode = "reused_artifact"
    model = _fit_ridge_from_stats(stats, alpha)
    model["training_metadata"] = {
        "model_training_mode": mode,
        "warm_start_used": True,
        "new_labeled_samples": int(max(len(y) - previous_samples, 0)),
    }
    return model


def _fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    artifact: dict | None = None,
    force_full_fit: bool = False,
) -> dict:
    if artifact and not force_full_fit:
        warm_model = _fit_ridge_warm_start(x, y, alpha, artifact)
        if warm_model is not None:
            return warm_model

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
        "ridge_stats": _ridge_stats_from_xy(x, y),
        "training_metadata": {
            "model_training_mode": "full_fit",
            "warm_start_used": False,
            "new_labeled_samples": int(len(y)),
        },
    }


def _fit_xgboost(
    x: np.ndarray,
    y: np.ndarray,
    reg_lambda: float,
    artifact: dict | None = None,
    force_full_fit: bool = False,
) -> dict:
    if XGBRegressor is None:
        raise ValueError("XGBoost package is not installed.")

    previous_model = None
    previous_samples = 0
    warm_start_mode = "full_fit"
    if artifact and not force_full_fit and _artifact_training_prefix_matches(artifact, x, y):
        previous_model = artifact.get("model")
        previous_samples = int(artifact.get("training_samples", 0) or 0)

    if previous_model is not None and previous_samples > 0 and previous_samples <= len(y):
        new_samples = len(y) - previous_samples
        if new_samples <= 0:
            fitted = previous_model.predict(x)
            residuals = y - fitted
            residual_std = residuals.std(ddof=1) if len(residuals) > 1 else residuals.std(ddof=0)
            return {
                "kind": "xgboost",
                "model": previous_model,
                "residual_std": residual_std,
                "training_metadata": {
                    "model_training_mode": "reused_artifact",
                    "warm_start_used": True,
                    "new_labeled_samples": 0,
                },
            }

        fine_tune_start = max(0, previous_samples - 30)
        fit_x = x[fine_tune_start:]
        fit_y = y[fine_tune_start:]
        warm_start_mode = "warm_booster_update"
        n_estimators = 24
    else:
        fit_x = x
        fit_y = y
        new_samples = len(y)
        n_estimators = 90

    model = XGBRegressor(
        n_estimators=n_estimators,
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
    if previous_model is not None and warm_start_mode == "warm_booster_update":
        model.fit(fit_x, fit_y, xgb_model=previous_model.get_booster())
    else:
        model.fit(fit_x, fit_y)
    fitted = model.predict(x)
    residuals = y - fitted
    residual_std = residuals.std(ddof=1) if len(residuals) > 1 else residuals.std(ddof=0)
    return {
        "kind": "xgboost",
        "model": model,
        "residual_std": residual_std,
        "training_metadata": {
            "model_training_mode": warm_start_mode,
            "warm_start_used": warm_start_mode != "full_fit",
            "new_labeled_samples": int(new_samples),
        },
    }


def _fit_neural_net(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    artifact: dict | None = None,
    force_full_fit: bool = False,
) -> dict:
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

    previous_model = None
    previous_samples = 0
    warm_start_mode = "full_fit"
    if artifact and not force_full_fit and _artifact_training_prefix_matches(artifact, x, y):
        previous_model = artifact.get("model")
        previous_samples = int(artifact.get("training_samples", 0) or 0)
        if previous_model is not None:
            x_mean = np.asarray(artifact.get("x_mean", x_mean), dtype=float)
            x_std = np.asarray(artifact.get("x_std", x_std), dtype=float)
            x_std[x_std == 0] = 1.0
            y_mean = float(artifact.get("y_mean", y_mean))
            y_std = float(artifact.get("y_std", y_std) or 1.0)
            x_scaled = (x - x_mean) / x_std
            y_scaled = (y - y_mean) / y_std

    if previous_model is not None and previous_samples > 0 and previous_samples <= len(y):
        new_samples = len(y) - previous_samples
        if new_samples <= 0:
            model = previous_model
        else:
            model = previous_model
            model.warm_start = True
            model.max_iter = 80
            warm_start_mode = "warm_partial_fit"
    else:
        new_samples = len(y)
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
        if previous_model is None or warm_start_mode == "warm_partial_fit":
            fit_start = max(0, previous_samples - 30) if previous_model is not None else 0
            model.fit(x_scaled[fit_start:], y_scaled[fit_start:])

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
        "training_metadata": {
            "model_training_mode": "reused_artifact" if previous_model is not None and new_samples <= 0 else warm_start_mode,
            "warm_start_used": previous_model is not None,
            "new_labeled_samples": int(max(new_samples, 0)),
        },
    }


def _fit_sequence_model(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    model_type: str,
    artifact: dict | None = None,
    force_full_fit: bool = False,
) -> dict:
    if torch is None or nn is None:
        raise ValueError("PyTorch package is not installed.")
    if x.ndim != 3:
        raise ValueError("Sequence model needs 3D training data.")

    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    torch.manual_seed(17)

    sample_count, sequence_length, feature_count = x.shape
    hidden_size = 18
    x_mean = x.reshape(-1, feature_count).mean(axis=0)
    x_std = x.reshape(-1, feature_count).std(axis=0)
    x_std[x_std == 0] = 1.0

    y_mean = float(y.mean())
    y_std = float(y.std(ddof=0))
    if y_std == 0.0:
        y_std = 1.0

    previous_state = None
    previous_samples = 0
    warm_start_mode = "full_fit"
    if (
        artifact
        and not force_full_fit
        and artifact.get("kind") == model_type
        and int(artifact.get("sequence_length", 0) or 0) == sequence_length
        and int(artifact.get("feature_count", 0) or 0) == feature_count
        and _artifact_training_prefix_matches(artifact, x, y)
    ):
        previous_state = artifact.get("state_dict")
        previous_samples = int(artifact.get("training_samples", 0) or 0)
        if previous_state is not None:
            x_mean = np.asarray(artifact.get("x_mean", x_mean), dtype=float)
            x_std = np.asarray(artifact.get("x_std", x_std), dtype=float)
            x_std[x_std == 0] = 1.0
            y_mean = float(artifact.get("y_mean", y_mean))
            y_std = float(artifact.get("y_std", y_std) or 1.0)
            warm_start_mode = "torch_fine_tune" if len(y) > previous_samples else "reused_artifact"

    x_scaled = (x - x_mean) / x_std
    y_scaled = (y - y_mean) / y_std

    class LSTMRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(feature_count, hidden_size, num_layers=1, batch_first=True)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
            )

        def forward(self, features):
            output, _ = self.lstm(features)
            return self.head(output[:, -1, :]).squeeze(-1)

    class TinyTransformerRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            d_model = 18
            self.projection = nn.Linear(feature_count, d_model)
            self.position = nn.Parameter(torch.zeros(1, sequence_length, d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=2,
                dim_feedforward=36,
                dropout=0.05,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=1)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
            )

        def forward(self, features):
            encoded = self.encoder(self.projection(features) + self.position)
            return self.head(encoded[:, -1, :]).squeeze(-1)

    model = LSTMRegressor() if model_type == "lstm" else TinyTransformerRegressor()
    if previous_state is not None:
        model.load_state_dict(previous_state, strict=True)

    x_tensor = torch.as_tensor(x_scaled, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_scaled, dtype=torch.float32)
    if previous_state is not None and warm_start_mode == "torch_fine_tune":
        fit_start = max(0, previous_samples - 40)
        fit_x_tensor = x_tensor[fit_start:]
        fit_y_tensor = y_tensor[fit_start:]
        epochs = 24 if model_type == "lstm" else 20
    else:
        fit_x_tensor = x_tensor
        fit_y_tensor = y_tensor
        epochs = 80 if model_type == "lstm" else 65

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.003 if previous_state is not None else 0.01,
        weight_decay=max(float(alpha), 0.01) / 10000.0,
    )
    loss_fn = nn.MSELoss()
    best_loss = np.inf
    best_state = None
    patience = 8
    stale_epochs = 0
    epochs_run = 0

    if warm_start_mode != "reused_artifact":
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(fit_x_tensor)
            loss = loss_fn(prediction, fit_y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epochs_run += 1

            loss_value = float(loss.detach().cpu())
            if loss_value + 1e-5 < best_loss:
                best_loss = loss_value
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        fitted = model(x_tensor).detach().cpu().numpy() * y_std + y_mean
    residuals = y - fitted
    residual_std = residuals.std(ddof=1) if len(residuals) > 1 else residuals.std(ddof=0)
    return {
        "kind": model_type,
        "model": model,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "residual_std": residual_std,
        "sequence_length": int(sequence_length),
        "feature_count": int(feature_count),
        "training_epochs": int(epochs_run),
        "training_metadata": {
            "model_training_mode": warm_start_mode,
            "warm_start_used": previous_state is not None,
            "new_labeled_samples": int(max(len(y) - previous_samples, 0)) if previous_state is not None else int(len(y)),
            "fine_tune_epochs": int(epochs_run),
        },
    }


def _fit_model(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    model_type: str,
    symbol: str | None = None,
    horizon_days: int | None = None,
    lookback_window: int | None = None,
    optimize_model: bool = True,
    warm_start: bool = True,
    force_retrain: bool = False,
    artifact_dir: str | Path | None = None,
) -> dict:
    artifact_path = None
    artifact = None
    if warm_start and symbol:
        artifact_path = _model_weight_artifact_path(
            symbol=symbol,
            model_type=model_type,
            horizon_days=horizon_days,
            lookback_window=lookback_window,
            alpha=alpha,
            optimize_model=optimize_model,
            x_shape=x.shape,
            artifact_dir=artifact_dir,
        )
        if not force_retrain:
            artifact = _load_model_weight_artifact(artifact_path)

    if model_type == "xgboost":
        model = _fit_xgboost(x, y, alpha, artifact=artifact, force_full_fit=force_retrain)
    elif model_type == "neural_net":
        model = _fit_neural_net(x, y, alpha, artifact=artifact, force_full_fit=force_retrain)
    elif model_type in {"lstm", "transformer"}:
        model = _fit_sequence_model(x, y, alpha, model_type, artifact=artifact, force_full_fit=force_retrain)
    else:
        model = _fit_ridge(x, y, alpha, artifact=artifact, force_full_fit=force_retrain)

    if artifact_path is not None:
        _save_model_weight_artifact(artifact_path, model, x, y, symbol, model_type)
        metadata = dict(model.get("training_metadata", {}))
        metadata["model_artifact_path"] = str(artifact_path)
        model["training_metadata"] = metadata
    return model


def _predict_matrix(x: np.ndarray, model: dict) -> np.ndarray:
    if model.get("kind") == "xgboost":
        return model["model"].predict(x)

    if model.get("kind") == "neural_net":
        x_scaled = (x - model["x_mean"]) / model["x_std"]
        return model["model"].predict(x_scaled) * model["y_std"] + model["y_mean"]

    if model.get("kind") in {"lstm", "transformer"}:
        x_scaled = (x - model["x_mean"]) / model["x_std"]
        tensor = torch.as_tensor(x_scaled, dtype=torch.float32)
        model["model"].eval()
        with torch.no_grad():
            return model["model"](tensor).detach().cpu().numpy() * model["y_std"] + model["y_mean"]

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
    if model_type in {"lstm", "transformer"}:
        window_candidates = _candidate_values(
            [requested_lookback],
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
    x, y = _training_data_for_model(log_returns, lookback_window, horizon_days, model_type, extra_feature_arrays)
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


def _forecast_from_horizon_return(
    close: pd.Series,
    horizon_days: int,
    horizon_return: float,
    raw_horizon_return: float,
    residual_std: float,
    metrics: dict,
    model_name: str,
) -> ForecastResult:
    horizon_days = max(1, int(horizon_days))
    residual_std = max(float(residual_std), 1e-6)
    steps = np.arange(1, horizon_days + 1, dtype=float)
    cumulative_returns = float(horizon_return) * (steps / float(horizon_days))
    future_log_returns = np.diff(np.concatenate([[0.0], cumulative_returns]))
    last_log_price = float(np.log(close.iloc[-1]))
    interval_scale = np.sqrt(steps / float(horizon_days))

    forecast = pd.DataFrame(
        {
            "forecast_close": np.exp(last_log_price + cumulative_returns),
            "lower_estimate": np.exp(last_log_price + cumulative_returns - 1.645 * residual_std * interval_scale),
            "upper_estimate": np.exp(last_log_price + cumulative_returns + 1.645 * residual_std * interval_scale),
            "expected_daily_return_pct": np.expm1(future_log_returns) * 100.0,
        },
        index=_future_index(close.index, horizon_days),
    )
    forecast.index.name = "date"

    probability_up_pct = _normal_cdf(float(horizon_return) / residual_std) * 100.0
    confidence_pct = max(probability_up_pct, 100.0 - probability_up_pct)
    expected_error_pct = np.expm1(residual_std) * 100.0
    forecast_change_pct = np.expm1(float(horizon_return)) * 100.0
    metrics = dict(metrics)
    metrics.update(
        {
            "forecast_change_pct": forecast_change_pct,
            "probability_up_pct": probability_up_pct,
            "probability_down_pct": 100.0 - probability_up_pct,
            "confidence_pct": confidence_pct,
            "expected_error_pct": expected_error_pct,
            "forecast_score": _forecast_score(forecast_change_pct, expected_error_pct, confidence_pct),
            "raw_forecast_change_pct": np.expm1(float(raw_horizon_return)) * 100.0,
            "shrink_factor": 1.0,
        }
    )
    return ForecastResult(forecast=forecast, metrics=metrics, model_name=model_name)


def _rl_training_data(log_returns: np.ndarray, lookback_window: int, horizon_days: int) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    targets = []
    last_start = len(log_returns) - horizon_days
    for as_of_pos in range(lookback_window, last_start + 1):
        window = log_returns[as_of_pos - lookback_window : as_of_pos]
        target = log_returns[as_of_pos : as_of_pos + horizon_days].sum()
        state = _rl_state_from_window(window)
        if np.isfinite(state).all() and np.isfinite(target):
            rows.append(state)
            targets.append(target)
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def _rl_state_from_window(window: np.ndarray) -> np.ndarray:
    window = np.asarray(window, dtype=float)
    recent_5 = window[-min(5, len(window)) :]
    recent_20 = window[-min(20, len(window)) :]
    momentum_5 = recent_5.sum()
    momentum_20 = recent_20.sum()
    volatility = recent_20.std(ddof=0)
    downside = np.minimum(recent_20, 0.0).std(ddof=0)
    mean_reversion = 0.0 if volatility == 0 else recent_5.mean() / volatility
    return np.asarray(
        [
            _bin_three_way(momentum_5, 0.01),
            _bin_three_way(momentum_20, 0.025),
            _bin_volatility(volatility),
            _bin_three_way(mean_reversion, 0.35),
            _bin_volatility(downside),
        ],
        dtype=float,
    )


def _bin_three_way(value: float, threshold: float) -> int:
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def _bin_volatility(value: float) -> int:
    if value >= 0.035:
        return 2
    if value >= 0.015:
        return 1
    return 0


def _rl_state_key(state: np.ndarray) -> str:
    return "|".join(str(int(value)) for value in np.asarray(state, dtype=float))


def _update_rl_q_table(
    q_table: dict[str, np.ndarray],
    states: np.ndarray,
    targets: np.ndarray,
    start_index: int = 0,
    learning_rate: float = 0.08,
    gamma: float = 0.05,
    trade_cost: float = 0.001,
) -> None:
    start_index = max(0, int(start_index))
    for idx in range(start_index, len(targets)):
        state_key = _rl_state_key(states[idx])
        q_values = q_table.setdefault(state_key, np.zeros(3, dtype=float))
        if idx + 1 < len(states):
            next_values = q_table.setdefault(_rl_state_key(states[idx + 1]), np.zeros(3, dtype=float))
            next_best = float(np.max(next_values))
        else:
            next_best = 0.0
        realized_return = float(targets[idx])
        rewards = np.asarray(
            [
                -realized_return - trade_cost,
                0.0,
                realized_return - trade_cost,
            ],
            dtype=float,
        )
        q_values += learning_rate * (rewards + gamma * next_best - q_values)


def _rl_forecast_for_state(q_table: dict[str, np.ndarray], state: np.ndarray) -> float:
    q_values = q_table.get(_rl_state_key(state))
    if q_values is None:
        return 0.0
    action_index = int(np.argmax(q_values))
    if action_index == 2:
        return float(q_values[2])
    if action_index == 0:
        return float(-q_values[0])
    return 0.0


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
