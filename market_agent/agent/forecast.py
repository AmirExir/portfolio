from dataclasses import dataclass
import hashlib
from math import erf, sqrt
import os
from pathlib import Path
import pickle
from typing import Mapping
import warnings

import numpy as np
import pandas as pd

from .earnings import us_equity_trading_sessions
from .policy_features import build_shadow_policy_data
from .shadow_policy import (
    PolicyPosition,
    ShadowPolicyConfig,
    ShadowTargetWeightPolicy,
)

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
    metrics["horizon_days"] = int(horizon_days)
    metrics["live_eligible"] = True
    metrics["shadow_mode"] = False
    outlier_limit_pct = max(
        50.0,
        2.5 * max(expected_error_pct, 0.25),
        3.0 * max(float(metrics.get("holdout_mae_pct", 0.0) or 0.0), 0.25),
    )
    metrics["forecast_outlier"] = bool(abs(metrics["raw_forecast_change_pct"]) > outlier_limit_pct)
    metrics["forecast_outlier_limit_pct"] = float(outlier_limit_pct)
    for metadata_key, metadata_value in (model.get("training_metadata", {}) or {}).items():
        metrics.setdefault(metadata_key, metadata_value)

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
    earnings_context: Mapping[str, object] | None = None,
    portfolio_returns: pd.Series | None = None,
    policy_position: PolicyPosition | None = None,
    position_state_verified: bool = False,
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

    available_components = {
        name: result
        for name, result in results.items()
        if (
            result.forecast is not None
            and not result.forecast.empty
            and not bool(result.metrics.get("shadow_mode"))
            and result.metrics.get("live_eligible", True) is not False
        )
    }
    if available_components:
        results["Ensemble"] = _ensemble_result(available_components)

    if include_rl:
        try:
            context_name, context_result = _fixed_non_rl_policy_context(results)
            (
                forecast_context_frame,
                latest_forecast_context,
                forecast_context_metadata,
            ) = _build_policy_forecast_context(
                close=_clean_close(df),
                source_name=context_name,
                result=context_result,
                components=available_components,
            )
            results["RL Policy"] = reinforcement_policy_forecast(
                df,
                horizon_days=horizon_days,
                lookback_window=lookback_window,
                context_df=context_df,
                earnings_context=earnings_context,
                portfolio_returns=portfolio_returns,
                policy_position=policy_position,
                position_state_verified=position_state_verified,
                forecast_context_df=forecast_context_frame,
                latest_forecast_context=latest_forecast_context,
                forecast_context_metadata=forecast_context_metadata,
                symbol=symbol,
                warm_start=warm_start,
                force_retrain=force_retrain,
                artifact_dir=model_artifact_dir,
            )
        except Exception as exc:
            results["RL Policy"] = _unavailable_result("rl policy unavailable", exc)

    return results


def reinforcement_policy_forecast(
    df: pd.DataFrame,
    horizon_days: int = 30,
    lookback_window: int = 20,
    context_df: pd.DataFrame | None = None,
    earnings_context: Mapping[str, object] | None = None,
    portfolio_returns: pd.Series | None = None,
    policy_position: PolicyPosition | None = None,
    position_state_verified: bool = False,
    forecast_context_df: pd.DataFrame | None = None,
    latest_forecast_context: Mapping[str, object] | None = None,
    forecast_context_metadata: Mapping[str, object] | None = None,
    symbol: str | None = None,
    warm_start: bool = True,
    force_retrain: bool = False,
    artifact_dir: str | Path | None = None,
) -> ForecastResult:
    """Evaluate the execution-aligned target-weight policy in shadow mode.

    The returned flat price path exists only for snapshot compatibility. The
    actionable research output is ``rl_target_weight``; this result is marked
    non-live and cannot enter model selection, ensembles, reliability, or
    orders.
    """
    close = _clean_close(df)
    horizon_days = max(1, int(horizon_days))
    if (forecast_context_df is None) != (latest_forecast_context is None):
        raise ValueError(
            "forecast_context_df and latest_forecast_context must be supplied together"
        )
    if forecast_context_df is None:
        fixed_result = forecast_close_prices(
            df,
            horizon_days=horizon_days,
            lookback_window=lookback_window,
            ridge_alpha=10.0,
            optimize_model=True,
            model_type="ridge",
            context_df=context_df,
            symbol=symbol,
            warm_start=warm_start,
            force_retrain=force_retrain,
            model_artifact_dir=artifact_dir,
        )
        (
            forecast_context_df,
            latest_forecast_context,
            generated_metadata,
        ) = _build_policy_forecast_context(
            close=close,
            source_name="Ridge",
            result=fixed_result,
            components={"Ridge": fixed_result},
        )
        forecast_context_metadata = generated_metadata
    context_metadata = dict(forecast_context_metadata or {})
    declared_context_horizon = context_metadata.get("horizon_sessions")
    if (
        declared_context_horizon is not None
        and int(declared_context_horizon) != horizon_days
    ):
        raise ValueError(
            "forecast context horizon does not match the policy horizon"
        )
    context_source = str(
        context_metadata.get("source_model", "externally_supplied_non_rl")
    )
    context_version = str(
        context_metadata.get(
            "context_version",
            f"fixed-non-rl-oos-v1-h{horizon_days}",
        )
    )
    data = build_shadow_policy_data(
        df,
        context_df=context_df,
        horizon_days=horizon_days,
        symbol=symbol,
        latest_event_context=earnings_context,
        portfolio_returns=portfolio_returns,
        forecast_context_df=forecast_context_df,
        latest_forecast_context=latest_forecast_context,
        forecast_context_source=context_source,
        forecast_context_version=context_version,
    )
    policy_frame = data.training_frame
    purge_sessions = max(30, horizon_days)
    holdout_size = min(
        max(15, int(len(policy_frame) * 0.20)),
        len(policy_frame) - purge_sessions - 30,
    )
    if holdout_size < 10:
        raise ValueError("Not enough matured observations for purged RL shadow evaluation.")
    test_start_pos = len(policy_frame) - holdout_size
    train_end_pos = test_start_pos - purge_sessions - 1
    if train_end_pos < 29:
        raise ValueError("Not enough pre-purge observations for RL shadow training.")

    policy = ShadowTargetWeightPolicy(
        ShadowPolicyConfig(
            risk_budget_fraction=0.05,
            minimum_purge_sessions=purge_sessions,
            random_seed=1729 + horizon_days,
        )
    )
    evaluation = policy.fit_purged_holdout(
        policy_frame,
        train_end=policy_frame.index[train_end_pos],
        test_start=policy_frame.index[test_start_pos],
        purge_sessions=purge_sessions,
    )
    external_position_verified = bool(
        position_state_verified and policy_position is not None
    )
    if external_position_verified:
        decision_position = policy_position
        position_state_source = "verified_external"
        position_state_as_of = str(data.as_of)
    else:
        (
            decision_position,
            position_state_source,
            position_state_as_of,
        ) = _position_from_shadow_evaluation(evaluation)
    decision = policy.decide(
        data.latest_context,
        position=decision_position,
    )
    q_values = np.asarray(decision.q_values, dtype=float)
    action_name = "hold" if decision.action_fraction <= 0.0 else "long"
    active = evaluation.decisions["active_return"]
    nonzero = evaluation.decisions["target_exposure"] > 0.0
    if nonzero.any():
        direction_accuracy = float((active.loc[nonzero] > 0.0).mean() * 100.0)
    else:
        direction_accuracy = 50.0
    matured_daily = pd.to_numeric(
        policy_frame["forward_asset_return"],
        errors="coerce",
    ).dropna()
    residual_std = max(
        float(np.log1p(matured_daily.clip(lower=-0.999999)).std(ddof=1))
        * np.sqrt(horizon_days),
        1e-6,
    )
    policy_execution_values = _future_index(close.index, 2)
    policy_start_value = policy_execution_values[0]
    policy_target_value = policy_execution_values[1]
    if isinstance(policy_start_value, (pd.Timestamp, np.datetime64)):
        policy_execution_start = str(
            pd.Timestamp(policy_start_value).date()
        )
    else:
        policy_execution_start = str(policy_start_value)
    if isinstance(policy_target_value, (pd.Timestamp, np.datetime64)):
        policy_execution_target = str(
            pd.Timestamp(policy_target_value).date()
        )
    else:
        policy_execution_target = str(policy_target_value)

    metrics = {
        "holdout_direction_accuracy": direction_accuracy,
        "holdout_mae_pct": np.nan,
        "holdout_rmse_pct": np.nan,
        "holdout_bias_pct": np.nan,
        "holdout_samples": int(len(evaluation.decisions)),
        "selected_lookback_window": int(max(5, lookback_window)),
        "selected_ridge_alpha": 0.0,
        "training_samples": int(train_end_pos + 1),
        "model_training_mode": "purged_frozen_shadow_fit",
        "warm_start_used": False,
        "new_labeled_samples": int(train_end_pos + 1),
        "rl_action": action_name,
        "rl_action_label": f"{decision.action_fraction * 100.0:.0f}% risk budget",
        "rl_action_fraction": float(decision.action_fraction),
        "rl_q_short": 0.0,
        "rl_q_hold": float(q_values[0]),
        "rl_q_long": float(np.max(q_values[1:])),
        "rl_q_values": [float(value) for value in q_values],
        "rl_target_weight": float(decision.target_exposure),
        "rl_state_visits": int(decision.state_visits),
        "rl_action_visits": int(decision.action_visits),
        "rl_abstained": bool(decision.abstained),
        "rl_abstention_reason": decision.reason if decision.abstained else "",
        "rl_position_state_verified": external_position_verified,
        "rl_position_state_source": position_state_source,
        "rl_position_state_as_of": position_state_as_of,
        "rl_position_state_auditable": True,
        "rl_live_allocation_enabled": False,
        "rl_position_exposure": float(decision_position.exposure),
        "rl_position_entry_drawdown": float(
            decision_position.entry_drawdown
        ),
        "rl_position_holding_period": int(
            decision_position.holding_period
        ),
        "rl_position_previous_action_fraction": float(
            decision_position.previous_action_fraction
        ),
        "rl_oos_cumulative_reward": float(evaluation.cumulative_reward),
        "rl_oos_net_return_pct": float(evaluation.cumulative_net_return * 100.0),
        "rl_oos_max_drawdown_pct": float(evaluation.max_drawdown * 100.0),
        "rl_oos_turnover": float(evaluation.turnover),
        "rl_oos_invested_fraction": float(evaluation.invested_fraction),
        "horizon_days": int(horizon_days),
        "policy_version": f"rl-shadow-contextual-v3-h{horizon_days}",
        "model_version": (
            f"execution-aligned-contextual-shadow-v3-h{horizon_days}"
        ),
        "forecast_context_horizon_sessions": int(horizon_days),
        "policy_execution_horizon_sessions": 1,
        "policy_decision_refresh_sessions": 1,
        "policy_execution_start_session": policy_execution_start,
        "policy_execution_target_session": policy_execution_target,
        "policy_feature_set_version": (
            f"shadow-market-position-event-forecast-v3-h{horizon_days}"
        ),
        "shadow_mode": True,
        "live_eligible": False,
        "reliability_eligible": False,
        "validation_is_oos": True,
        "validation_scheme": "purged_frozen_holdout",
        "purge_sessions": int(evaluation.purge_sessions or purge_sessions),
        "validation_train_end": str(evaluation.training_end),
        "validation_test_start": str(evaluation.evaluation_start),
        "policy_as_of": str(data.as_of),
        "execution_lag_sessions": int(data.execution_lag_sessions),
        "reward_window": "close_t_plus_1_to_close_t_plus_2",
        "benchmark_source": data.benchmark_source,
        "sector_proxy": data.sector_proxy,
        "portfolio_correlation_source": data.correlation_source,
        "forecast_context_source": data.forecast_context_source,
        "forecast_context_version": data.forecast_context_version,
        "forecast_context_samples": int(data.forecast_context_samples),
        "forecast_context_validation_scheme": str(
            context_metadata.get("validation_scheme", "")
        ),
        "forecast_context_component_models": list(
            context_metadata.get("component_models", [])
        ),
        "forecast_context_actual_outcomes_used": False,
        "forecast_context_return": float(
            data.latest_context["forecast_return"]
        ),
        "forecast_context_probability_up": float(
            data.latest_context["forecast_probability_up"]
        ),
        "forecast_context_lower_bound": float(
            data.latest_context["forecast_lower_bound"]
        ),
        "forecast_context_model_agreement": float(
            data.latest_context["forecast_model_agreement"]
        ),
        "forecast_context_uncertainty": float(
            data.latest_context["forecast_uncertainty"]
        ),
        "earnings_event_fingerprint": (
            str(
                earnings_context.get(
                    "earnings_effective_session",
                    earnings_context.get("effective_session", ""),
                )
            )
            + "|"
            + str(
                earnings_context.get(
                    "earnings_reported_at",
                    earnings_context.get("reported_at", ""),
                )
            )
            if earnings_context
            else ""
        ),
        "promotion_status": "shadow_only",
        "validation_warning": (
            "RL is a target-weight policy, not a price forecaster. Its flat "
            "compatibility path is excluded from every live decision."
        ),
    }
    result = _forecast_from_horizon_return(
        close=close,
        horizon_days=horizon_days,
        horizon_return=0.0,
        raw_horizon_return=0.0,
        residual_std=residual_std,
        metrics=metrics,
        model_name="execution-aligned target-weight RL policy (shadow only)",
    )
    return result


def _unavailable_result(model_name: str, error: Exception | str) -> ForecastResult:
    return ForecastResult(
        forecast=pd.DataFrame(),
        metrics={"error": str(error), "forecast_score": np.nan},
        model_name=model_name,
    )


def _fixed_non_rl_policy_context(
    results: Mapping[str, ForecastResult],
) -> tuple[str, ForecastResult]:
    """Choose the fixed non-RL champion used once as policy context."""
    for preferred in ("Ensemble", "Ridge"):
        result = results.get(preferred)
        if result is not None and _has_oos_policy_context(result):
            return preferred, result
    for name, result in results.items():
        if name != "RL Policy" and _has_oos_policy_context(result):
            return name, result
    raise ValueError(
        "No fixed non-RL model has point-in-time OOS predictions for policy context."
    )


def _has_oos_policy_context(result: ForecastResult) -> bool:
    return bool(
        result.forecast is not None
        and not result.forecast.empty
        and result.metrics.get("live_eligible", True) is not False
        and not result.metrics.get("shadow_mode", False)
        and result.metrics.get("validation_is_oos", False)
        and len(result.metrics.get("holdout_predicted_return_pct") or []) > 0
        and len(result.metrics.get("holdout_probability_up") or []) > 0
    )


def _build_policy_forecast_context(
    *,
    close: pd.Series,
    source_name: str,
    result: ForecastResult,
    components: Mapping[str, ForecastResult],
) -> tuple[pd.DataFrame, dict[str, float], dict[str, object]]:
    """Create timestamped OOS and current forecast state without outcomes.

    Only prediction-time arrays are read. ``holdout_actual_return_pct`` is
    intentionally ignored so realized target returns cannot enter the state.
    """
    if not result.metrics.get("validation_is_oos", False):
        raise ValueError("policy forecast context requires OOS validation")

    metrics = result.metrics
    predicted_pct = _finite_metric_array(
        metrics,
        "holdout_predicted_return_pct",
    )
    probability_up = _finite_metric_array(metrics, "holdout_probability_up")
    expected_error_raw = metrics.get("holdout_expected_error_pct")
    if expected_error_raw is None:
        expected_error_pct = np.full(
            len(predicted_pct),
            max(float(metrics.get("expected_error_pct", 0.0) or 0.0), 0.0),
            dtype=float,
        )
    else:
        expected_error_pct = _finite_metric_array(
            metrics,
            "holdout_expected_error_pct",
        )
    agreement_raw = metrics.get("holdout_model_agreement")
    if agreement_raw is None:
        model_agreement = np.ones(len(predicted_pct), dtype=float)
    else:
        model_agreement = _finite_metric_array(
            metrics,
            "holdout_model_agreement",
        )

    declared_samples = int(metrics.get("holdout_samples", 0) or 0)
    lengths = [
        len(predicted_pct),
        len(probability_up),
        len(expected_error_pct),
        len(model_agreement),
    ]
    if declared_samples > 0:
        lengths.append(declared_samples)
    sample_count = min(lengths)
    if sample_count <= 0:
        raise ValueError("policy forecast context has no OOS samples")

    predicted_pct = predicted_pct[-sample_count:]
    probability_up = probability_up[-sample_count:]
    expected_error_pct = expected_error_pct[-sample_count:]
    model_agreement = model_agreement[-sample_count:]
    if ((probability_up < 0.0) | (probability_up > 1.0)).any():
        raise ValueError("holdout_probability_up must be between 0 and 1")
    if (expected_error_pct < 0.0).any():
        raise ValueError("holdout_expected_error_pct cannot be negative")
    if ((model_agreement < 0.0) | (model_agreement > 1.0)).any():
        raise ValueError("holdout_model_agreement must be between 0 and 1")

    horizon = max(int(metrics.get("horizon_days", 1) or 1), 1)
    end_position = len(close) - horizon
    start_position = end_position - sample_count
    if start_position < 0 or end_position <= start_position:
        raise ValueError(
            "OOS forecast samples cannot be aligned to available price dates"
        )
    context_index = close.index[start_position:end_position]
    forecast_return = predicted_pct / 100.0
    uncertainty = expected_error_pct / 100.0
    context_frame = pd.DataFrame(
        {
            "forecast_return": forecast_return,
            "forecast_probability_up": probability_up,
            "forecast_lower_bound": (
                forecast_return - 1.645 * uncertainty
            ),
            "forecast_model_agreement": model_agreement,
            "forecast_uncertainty": uncertainty,
        },
        index=context_index,
    )

    latest_return_pct = _finite_metric_value(metrics, "forecast_change_pct")
    latest_probability_pct = _finite_metric_value(
        metrics,
        "probability_up_pct",
    )
    latest_error_pct = max(
        _finite_metric_value(metrics, "expected_error_pct"),
        0.0,
    )
    latest_return = latest_return_pct / 100.0
    latest_uncertainty = latest_error_pct / 100.0
    component_names = [
        name
        for name, component in components.items()
        if (
            name != "RL Policy"
            and component.forecast is not None
            and not component.forecast.empty
            and component.metrics.get("live_eligible", True) is not False
            and not component.metrics.get("shadow_mode", False)
            and component.metrics.get("validation_is_oos", False)
            and int(component.metrics.get("horizon_days", horizon) or horizon)
            == horizon
        )
    ]
    latest_agreement = _latest_model_agreement(
        latest_return_pct,
        [components[name] for name in component_names],
    )
    latest_context = {
        "forecast_return": float(latest_return),
        "forecast_probability_up": float(
            np.clip(latest_probability_pct / 100.0, 0.0, 1.0)
        ),
        "forecast_lower_bound": float(
            latest_return - 1.645 * latest_uncertainty
        ),
        "forecast_model_agreement": float(latest_agreement),
        "forecast_uncertainty": float(latest_uncertainty),
    }

    context_version = f"fixed-non-rl-oos-v1-h{horizon}"
    fingerprint_payload = "|".join(
        (
            str(source_name),
            context_version,
            str(context_index[0]),
            str(context_index[-1]),
            str(sample_count),
            ",".join(component_names),
        )
    )
    metadata: dict[str, object] = {
        "source_model": str(source_name),
        "context_version": context_version,
        "context_fingerprint": hashlib.sha256(
            fingerprint_payload.encode("utf-8")
        ).hexdigest()[:16],
        "validation_scheme": str(metrics.get("validation_scheme", "")),
        "horizon_sessions": int(horizon),
        "sample_count": int(sample_count),
        "sample_start": str(context_index[0]),
        "sample_end": str(context_index[-1]),
        "component_models": component_names,
        "actual_outcomes_used": False,
    }
    return context_frame, latest_context, metadata


def _finite_metric_array(metrics: Mapping[str, object], key: str) -> np.ndarray:
    raw = metrics.get(key)
    if raw is None:
        raise ValueError(f"{key} is required for policy forecast context")
    try:
        values = np.asarray(raw, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a numeric array") from exc
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"{key} must contain finite values")
    return values


def _finite_metric_value(metrics: Mapping[str, object], key: str) -> float:
    try:
        value = float(metrics.get(key))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric") from exc
    if not np.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def _latest_model_agreement(
    selected_return_pct: float,
    components: list[ForecastResult],
) -> float:
    signs: list[float] = []
    for component in components:
        try:
            value = float(component.metrics.get("forecast_change_pct"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            signs.append(float(np.sign(value)))
    if not signs:
        return 1.0
    selected_sign = float(np.sign(selected_return_pct))
    return float(np.mean(np.asarray(signs, dtype=float) == selected_sign))


def _position_from_shadow_evaluation(
    evaluation: object,
) -> tuple[PolicyPosition, str, str]:
    """Continue the frozen OOS paper episode for the next shadow decision."""
    decisions = getattr(evaluation, "decisions", None)
    if not isinstance(decisions, pd.DataFrame) or decisions.empty:
        return PolicyPosition(), "flat_shadow_initialization", ""
    last = decisions.iloc[-1]
    position = PolicyPosition(
        exposure=float(last["target_exposure"]),
        entry_drawdown=float(last["entry_drawdown"]),
        holding_period=int(last["holding_period"]),
        previous_action_fraction=float(last["action_fraction"]),
    )
    return position, "frozen_oos_shadow_tail", str(decisions.index[-1])


def _ensemble_result(components: dict[str, ForecastResult]) -> ForecastResult:
    usable = {
        name: result
        for name, result in components.items()
        if (
            name != "RL Policy"
            and result.forecast is not None
            and not result.forecast.empty
            and not bool(result.metrics.get("shadow_mode"))
            and result.metrics.get("live_eligible", True) is not False
        )
    }
    if not usable:
        return _unavailable_result("ensemble unavailable", "No component forecasts available.")

    # Fixed weights avoid fitting an ensemble on the same outer observations
    # later reported as OOS. Learned weights belong inside a nested fold.
    equal_weight = 1.0 / len(usable)
    weights = {name: equal_weight for name in usable}
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
    holdout_samples = min(
        len(result.metrics.get("holdout_predicted_return_pct") or [])
        for result in usable.values()
    )
    validation_is_oos = all(
        result.metrics.get("validation_is_oos", False)
        for result in usable.values()
    )
    holdout_mae = np.nan
    holdout_rmse = np.nan
    holdout_bias = np.nan
    direction_accuracy = np.nan
    calibration_error = np.nan
    brier_score = np.nan
    holdout_predicted_pct: list[float] = []
    holdout_actual_pct: list[float] = []
    holdout_probability_up: list[float] = []
    holdout_expected_error_pct: list[float] = []
    holdout_model_agreement: list[float] = []
    if holdout_samples > 0:
        predicted_arrays = {
            name: np.asarray(
                result.metrics["holdout_predicted_return_pct"][-holdout_samples:],
                dtype=float,
            )
            for name, result in usable.items()
        }
        actual_arrays = [
            np.asarray(
                result.metrics["holdout_actual_return_pct"][-holdout_samples:],
                dtype=float,
            )
            for result in usable.values()
        ]
        probability_arrays = {
            name: np.asarray(
                result.metrics["holdout_probability_up"][-holdout_samples:],
                dtype=float,
            )
            for name, result in usable.items()
        }
        expected_error_arrays = {}
        for name, result in usable.items():
            raw = result.metrics.get("holdout_expected_error_pct")
            if raw is None or len(raw) < holdout_samples:
                expected_error_arrays[name] = np.full(
                    holdout_samples,
                    max(
                        float(
                            result.metrics.get("expected_error_pct", 0.0)
                            or 0.0
                        ),
                        0.0,
                    ),
                    dtype=float,
                )
            else:
                expected_error_arrays[name] = np.asarray(
                    raw[-holdout_samples:],
                    dtype=float,
                )
        aligned_actuals = all(
            np.allclose(actual_arrays[0], actual, rtol=1e-10, atol=1e-10)
            for actual in actual_arrays[1:]
        )
        if aligned_actuals:
            ensemble_predicted = sum(
                weights[name] * predicted
                for name, predicted in predicted_arrays.items()
            )
            ensemble_probability = sum(
                weights[name] * probability
                for name, probability in probability_arrays.items()
            )
            ensemble_expected_error = sum(
                weights[name] * errors
                for name, errors in expected_error_arrays.items()
            )
            component_signs = np.vstack(
                [
                    np.sign(predicted)
                    for predicted in predicted_arrays.values()
                ]
            )
            ensemble_agreement = np.mean(
                component_signs
                == np.sign(ensemble_predicted)[np.newaxis, :],
                axis=0,
            )
            actual = actual_arrays[0]
            errors = ensemble_predicted - actual
            actual_up = (actual > 0.0).astype(float)
            holdout_mae = float(np.abs(errors).mean())
            holdout_rmse = float(np.sqrt(np.square(errors).mean()))
            holdout_bias = float(errors.mean())
            direction_accuracy = float(
                (np.sign(ensemble_predicted) == np.sign(actual)).mean() * 100.0
            )
            brier_score = float(
                np.square(ensemble_probability - actual_up).mean()
            )
            calibration_error = _expected_calibration_error_pct(
                ensemble_probability,
                actual_up,
            )
            holdout_predicted_pct = ensemble_predicted.tolist()
            holdout_actual_pct = actual.tolist()
            holdout_probability_up = ensemble_probability.tolist()
            holdout_expected_error_pct = (
                ensemble_expected_error.tolist()
            )
            holdout_model_agreement = ensemble_agreement.tolist()
        else:
            validation_is_oos = False
            holdout_samples = 0
    else:
        validation_is_oos = False

    metrics = {
        "forecast_change_pct": float(forecast_change),
        "probability_up_pct": float(probability_up),
        "probability_down_pct": float(100.0 - probability_up),
        "confidence_pct": float(confidence),
        "expected_error_pct": float(expected_error),
        "forecast_score": _forecast_score(forecast_change, expected_error, confidence),
        "holdout_mae_pct": float(holdout_mae),
        "holdout_rmse_pct": float(holdout_rmse),
        "holdout_bias_pct": float(holdout_bias),
        "holdout_direction_accuracy": float(direction_accuracy),
        "holdout_samples": int(holdout_samples),
        "calibration_error_pct": float(calibration_error),
        "brier_score": float(brier_score),
        "holdout_predicted_return_pct": holdout_predicted_pct,
        "holdout_actual_return_pct": holdout_actual_pct,
        "holdout_probability_up": holdout_probability_up,
        "holdout_expected_error_pct": holdout_expected_error_pct,
        "holdout_model_agreement": holdout_model_agreement,
        "forecast_outlier": any(
            bool(result.metrics.get("forecast_outlier", False))
            for result in usable.values()
        ),
        "component_weights": {name: round(weight, 3) for name, weight in weights.items()},
        "ensemble_weighting": "fixed_equal_non_rl",
        "shadow_components_excluded": sorted(set(components) - set(usable)),
        "live_eligible": True,
        "shadow_mode": False,
        "validation_is_oos": bool(validation_is_oos),
        "validation_scheme": "fixed_ensemble_on_nested_outer_holdout",
        "horizon_days": int(
            next(
                (
                    result.metrics.get("horizon_days")
                    for result in usable.values()
                    if result.metrics.get("horizon_days") is not None
                ),
                0,
            )
        ),
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

    # Candidate choice uses an earlier, purged validation block. The final
    # tail remains untouched until one candidate has been selected.
    approximate_samples = max(
        len(log_returns) - horizon_days - max(window_candidates, default=requested_lookback),
        0,
    )
    outer_test_size = max(8, int(approximate_samples * 0.20))
    reserved_outer_tail = outer_test_size + max(int(horizon_days), 1)

    for window in window_candidates:
        for alpha in alpha_candidates:
            metrics = _direct_holdout_metrics(
                log_returns,
                window,
                horizon_days,
                alpha,
                model_type,
                extra_feature_arrays,
                reserve_tail=reserved_outer_tail,
                validation_scheme="purged_inner_selection",
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

    selected = min(
        candidates,
        key=lambda item: (
            item["metrics"].get("holdout_mae_pct", np.inf),
            item["metrics"].get("holdout_rmse_pct", np.inf),
            -item["metrics"].get("holdout_direction_accuracy", 0.0),
        ),
    )
    outer_metrics = _direct_holdout_metrics(
        log_returns,
        selected["lookback_window"],
        horizon_days,
        selected["ridge_alpha"],
        model_type,
        extra_feature_arrays,
        reserve_tail=0,
        fixed_holdout_size=outer_test_size,
        validation_scheme="nested_purged_outer_holdout",
    )
    if not outer_metrics:
        outer_metrics = dict(selected["metrics"])
        outer_metrics["validation_is_oos"] = False
        outer_metrics["validation_warning"] = (
            "No untouched outer holdout was available after model selection."
        )
    outer_metrics["hyperparameter_selection_is_nested"] = bool(
        outer_metrics.get("validation_is_oos")
    )
    outer_metrics["selection_validation_mae_pct"] = float(
        selected["metrics"].get("holdout_mae_pct", np.nan)
    )
    outer_metrics["selection_validation_samples"] = int(
        selected["metrics"].get("holdout_samples", 0)
    )
    outer_metrics[
        "selection_validation_test_end_sample_exclusive"
    ] = int(
        selected["metrics"].get(
            "validation_test_end_sample_exclusive",
            0,
        )
    )
    outer_metrics["selection_reserved_outer_test_samples"] = int(
        outer_test_size
    )
    outer_metrics["selection_purge_before_outer"] = max(
        int(horizon_days),
        1,
    )
    return {
        "lookback_window": selected["lookback_window"],
        "ridge_alpha": selected["ridge_alpha"],
        "metrics": outer_metrics,
    }


def _direct_holdout_metrics(
    log_returns: np.ndarray,
    lookback_window: int,
    horizon_days: int,
    alpha: float,
    model_type: str = "ridge",
    extra_feature_arrays: list[np.ndarray] | None = None,
    reserve_tail: int = 0,
    fixed_holdout_size: int | None = None,
    validation_scheme: str = "purged_final_holdout",
) -> dict:
    x, y = _training_data_for_model(log_returns, lookback_window, horizon_days, model_type, extra_feature_arrays)
    if len(y) < 35:
        return {}

    reserve_tail = max(int(reserve_tail), 0)
    usable_end = len(y) - reserve_tail
    if usable_end < 35:
        return {}

    if fixed_holdout_size is None:
        requested_holdout = max(8, int(usable_end * 0.25))
    else:
        requested_holdout = max(int(fixed_holdout_size), 0)
    holdout_size = min(requested_holdout, usable_end - 20)
    if holdout_size < 8:
        return {}

    test_start = usable_end - holdout_size
    purge_sessions = max(int(horizon_days), 1)
    train_end = test_start - purge_sessions
    if train_end < 20:
        return {}

    model = _fit_model(x[:train_end], y[:train_end], alpha, model_type)
    predicted = np.asarray(
        [
            _clip_horizon_return(value, y[:train_end])
            for value in _predict_matrix(x[test_start:usable_end], model)
        ],
        dtype=float,
    )
    actual = y[test_start:usable_end]

    predicted_pct = np.expm1(predicted) * 100.0
    actual_pct = np.expm1(actual) * 100.0
    errors = predicted_pct - actual_pct
    directional_accuracy = float((np.sign(predicted) == np.sign(actual)).mean() * 100.0)
    residual_std = max(float(model.get("residual_std", 0.0) or 0.0), 1e-6)
    probability_up = np.asarray([_normal_cdf(float(value) / residual_std) for value in predicted])
    actual_up = (actual > 0.0).astype(float)
    expected_error_pct = float(np.expm1(residual_std) * 100.0)

    return {
        "holdout_direction_accuracy": directional_accuracy,
        "holdout_mae_pct": float(np.abs(errors).mean()),
        "holdout_rmse_pct": float(np.sqrt(np.square(errors).mean())),
        "holdout_bias_pct": float(errors.mean()),
        "holdout_samples": int(len(actual)),
        "validation_is_oos": True,
        "validation_scheme": str(validation_scheme),
        "purge_sessions": purge_sessions,
        "validation_train_samples": int(train_end),
        "validation_test_samples": int(len(actual)),
        "validation_test_start_sample": int(test_start),
        "validation_test_end_sample_exclusive": int(usable_end),
        "validation_reserved_tail_samples": int(reserve_tail),
        "brier_score": float(np.square(probability_up - actual_up).mean()),
        "calibration_error_pct": _expected_calibration_error_pct(
            probability_up,
            actual_up,
        ),
        "holdout_predicted_return_pct": predicted_pct.tolist(),
        "holdout_actual_return_pct": actual_pct.tolist(),
        "holdout_probability_up": probability_up.tolist(),
        "holdout_expected_error_pct": [
            expected_error_pct
            for _ in range(len(actual))
        ],
        "holdout_model_agreement": [1.0 for _ in range(len(actual))],
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


def _expected_calibration_error_pct(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    bins: int = 10,
) -> float:
    """Return weighted binary expected calibration error in percentage points."""
    probability = np.asarray(probabilities, dtype=float)
    actual = np.asarray(outcomes, dtype=float)
    valid = (
        np.isfinite(probability)
        & np.isfinite(actual)
        & (probability >= 0.0)
        & (probability <= 1.0)
    )
    probability = probability[valid]
    actual = actual[valid]
    if probability.size == 0:
        return np.nan
    edges = np.linspace(0.0, 1.0, max(int(bins), 2) + 1)
    bin_ids = np.clip(np.digitize(probability, edges[1:-1], right=False), 0, len(edges) - 2)
    error = 0.0
    for bin_id in range(len(edges) - 1):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        error += float(mask.mean()) * abs(
            float(probability[mask].mean()) - float(actual[mask].mean())
        )
    return float(error * 100.0)


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

        sessions = us_equity_trading_sessions(
            as_of=(
                pd.Timestamp(last_value.date(), tz="UTC")
                + pd.Timedelta(hours=12)
            ).to_pydatetime(),
            observed_sessions=tuple(
                timestamp.date() for timestamp in datetime_index
            ),
            future_session_count=horizon_days + 1,
        )
        future = [
            session for session in sessions
            if session > last_value.date()
        ][:horizon_days]
        return pd.DatetimeIndex(future)

    return pd.RangeIndex(start=len(index), stop=len(index) + horizon_days)
