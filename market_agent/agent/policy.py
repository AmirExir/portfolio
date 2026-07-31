from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SmartPolicyConfig:
    """Long-only adaptive policy parameters."""

    fast_window: int = 20
    slow_window: int = 50
    regime_window: int = 200
    min_model_edge_pct: float = 3.0
    min_forecast_error_ratio: float = 0.35
    buy_score_threshold: float = 0.75
    sell_score_threshold: float = -0.20
    rebalance_threshold_frac: float = 0.02
    target_volatility_pct: float = 18.0
    max_position_fraction: float = 0.05
    stop_loss_pct: float = 0.08
    trade_cost_bps: float = 5.0
    confidence_bound_scale: float = 1.0
    min_validation_samples: int = 8
    min_direction_accuracy_pct: float = 50.0
    max_calibration_error_pct: float = 20.0
    max_brier_score: float = 0.30
    min_model_agreement: int = 2
    require_oos_validation: bool = True
    minimum_earnings_confidence: float = 0.25
    adverse_earnings_block_threshold: float = -0.35
    maximum_earnings_component: float = 0.50
    risk_budget_actions: tuple[float, ...] = (0.0, 0.25, 0.50, 1.0)


@dataclass(frozen=True)
class SmartPolicyDecision:
    action: str
    side: str | None
    qty: int
    target_qty: int
    target_position_fraction: float
    score: float
    last_price: float
    reason: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


def smart_policy_report(
    df: pd.DataFrame,
    risk_fraction: float = 0.05,
    signal: pd.Series | None = None,
    forecast_metrics: dict[str, Any] | None = None,
    model_results: dict[str, Any] | None = None,
    equity: float = 100000.0,
    config: SmartPolicyConfig | None = None,
) -> dict[str, Any]:
    """Return report fields for ranking and publishing forecast recommendations."""
    config = config or SmartPolicyConfig()
    decision = smart_policy_decision(
        df=df,
        equity=equity,
        risk_fraction=risk_fraction,
        current_qty=0,
        avg_entry_price=None,
        signal=signal,
        forecast_metrics=forecast_metrics,
        model_results=model_results,
        config=config,
    )
    score = float(decision.score)
    has_target = decision.target_position_fraction > 0.0
    if has_target and score >= 1.75:
        policy_call = "Strong Buy"
    elif has_target and score >= config.buy_score_threshold:
        policy_call = "Buy"
    elif score <= -1.25:
        policy_call = "Strong Sell / Avoid"
    elif score <= config.sell_score_threshold:
        policy_call = "Sell / Avoid"
    else:
        policy_call = "Hold / Watch"

    diagnostics = decision.diagnostics or {}
    forecast_diag = diagnostics.get("forecast", {}) or {}
    trend_diag = diagnostics.get("trend", {}) or {}
    momentum_diag = diagnostics.get("momentum", {}) or {}
    earnings_diag = diagnostics.get("earnings", {}) or {}
    rl_diag = diagnostics.get("rl", {}) or {}

    return {
        "policy_call": policy_call,
        "policy_score": score,
        "policy_target_pct": float(decision.target_position_fraction * 100.0),
        "policy_reason": decision.reason,
        "forecast_component": float(forecast_diag.get("component", 0.0)),
        "trend_component": float(trend_diag.get("component", 0.0)),
        "momentum_component": float(momentum_diag.get("component", 0.0)),
        "earnings_component": float(earnings_diag.get("component", 0.0)),
        "earnings_event_flag": bool(earnings_diag.get("event_flag", False)),
        "earnings_event_score": float(earnings_diag.get("event_score", 0.0)),
        "earnings_confidence": float(earnings_diag.get("confidence", 0.0)),
        "earnings_summary": str(earnings_diag.get("summary", "") or ""),
        "earnings_reported_at": str(earnings_diag.get("reported_at", "") or ""),
        "earnings_effective_session": str(
            earnings_diag.get("effective_session", "") or ""
        ),
        "earnings_is_stale": bool(earnings_diag.get("is_stale", False)),
        "earnings_policy_eligible": bool(
            earnings_diag.get("policy_eligible", False)
        ),
        "earnings_blockers": list(earnings_diag.get("blockers", []) or []),
        "earnings_data_quality_flags": list(
            earnings_diag.get("data_quality_flags", []) or []
        ),
        "earnings_calendar_source": str(
            earnings_diag.get("calendar_source", "") or ""
        ),
        "earnings_error_code": str(
            earnings_diag.get("error_code", "") or ""
        ),
        "rl_component": float(rl_diag.get("component", 0.0)),
        "rl_action": rl_diag.get("action", ""),
        "rl_mode": "shadow",
        "allocation_eligible": bool(diagnostics.get("allocation_eligible", False)),
        "allocation_blockers": list(diagnostics.get("allocation_blockers", [])),
        "lower_confidence_bound_pct": float(diagnostics.get("lower_confidence_bound_pct", 0.0)),
        "agreeing_models": int(diagnostics.get("agreeing_models", 0)),
        "annual_volatility_pct": float(diagnostics.get("annual_volatility_pct", 0.0)),
    }


def smart_policy_decision(
    df: pd.DataFrame,
    equity: float,
    risk_fraction: float,
    current_qty: int = 0,
    avg_entry_price: float | None = None,
    signal: pd.Series | None = None,
    forecast_metrics: dict[str, Any] | None = None,
    model_results: dict[str, Any] | None = None,
    config: SmartPolicyConfig | None = None,
) -> SmartPolicyDecision:
    """
    Combine one selected forecast with independent trend and momentum evidence.

    This is deliberately long-only because the existing broker layer submits buy
    and sell orders, but does not manage borrow, margin, or short exposure. RL
    diagnostics are retained in shadow mode and never affect this score.
    """
    config = config or SmartPolicyConfig()
    close = _clean_close(df)
    if close.empty:
        return _hold("no clean close prices available")

    equity = max(_safe_float(equity, 0.0), 0.0)
    risk_fraction = max(_safe_float(risk_fraction, 0.0), 0.0)
    current_qty = max(int(current_qty or 0), 0)
    last_price = float(close.iloc[-1])
    if equity <= 0.0 or last_price <= 0.0:
        return _hold("invalid equity or price", last_price=last_price)

    avg_entry = _safe_float(avg_entry_price, np.nan)
    if current_qty > 0 and np.isfinite(avg_entry) and avg_entry > 0:
        drawdown = (last_price / avg_entry) - 1.0
        if drawdown <= -abs(config.stop_loss_pct):
            return SmartPolicyDecision(
                action="SELL",
                side="sell",
                qty=current_qty,
                target_qty=0,
                target_position_fraction=0.0,
                score=-3.0,
                last_price=last_price,
                reason=f"stop loss triggered at {drawdown * 100.0:.2f}%",
                diagnostics={"drawdown_pct": drawdown * 100.0},
            )

    primary_metrics = dict(forecast_metrics or {})
    rl_metrics = _result_metrics(model_results, "RL Policy")

    forecast_component, forecast_diag = _forecast_component(primary_metrics, config)
    trend_component, trend_diag = _trend_component(close, config)
    momentum_component, momentum_diag = _momentum_component(close)
    earnings_component, earnings_diag = _earnings_component(primary_metrics, config)
    rl_component, rl_diag = _rl_component(rl_metrics)

    score = (
        forecast_component
        + trend_component
        + momentum_component
        + earnings_component
    )
    score = float(np.clip(score, -3.0, 3.0))
    allocation_eligible, allocation_diag = _allocation_eligibility(
        primary_metrics,
        model_results,
        config,
    )

    annual_vol_pct = _annualized_volatility_pct(close)
    current_fraction = current_qty * last_price / equity
    max_fraction = min(max(risk_fraction, 0.0), config.max_position_fraction)
    target_fraction = 0.0

    if allocation_eligible and score >= config.buy_score_threshold and max_fraction > 0.0:
        target_fraction = _target_fraction(score, annual_vol_pct, max_fraction, config)
    elif current_qty > 0 and score > config.sell_score_threshold:
        target_fraction = min(current_fraction, max_fraction) if allocation_eligible else 0.0

    target_qty = int((equity * target_fraction) // last_price)
    target_qty = max(target_qty, 0)
    qty_delta = target_qty - current_qty

    diagnostics = {
        "score": score,
        "current_position_fraction": current_fraction,
        "target_position_fraction": target_fraction,
        "annual_volatility_pct": annual_vol_pct,
        "forecast": forecast_diag,
        "trend": trend_diag,
        "momentum": momentum_diag,
        "earnings": earnings_diag,
        "rl": rl_diag,
        "signal_component": 0.0,
        "allocation_eligible": allocation_eligible,
        **allocation_diag,
    }

    min_rebalance_qty = 1
    max_target_dollars = equity * max(max_fraction, 0.0)
    min_rebalance_dollars = min(
        equity * config.rebalance_threshold_frac,
        max(max_target_dollars * 0.25, last_price),
    )
    if qty_delta > 0 and qty_delta * last_price >= min_rebalance_dollars:
        return SmartPolicyDecision(
            action="BUY",
            side="buy",
            qty=max(qty_delta, min_rebalance_qty),
            target_qty=target_qty,
            target_position_fraction=target_fraction,
            score=score,
            last_price=last_price,
            reason=_decision_reason(
                "buy",
                score,
                forecast_diag,
                trend_diag,
                earnings_diag,
                annual_vol_pct,
            ),
            diagnostics=diagnostics,
        )

    if current_qty > 0 and (score <= config.sell_score_threshold or qty_delta < 0):
        sell_qty = current_qty if score <= config.sell_score_threshold else abs(qty_delta)
        return SmartPolicyDecision(
            action="SELL",
            side="sell",
            qty=max(int(sell_qty), min_rebalance_qty),
            target_qty=target_qty,
            target_position_fraction=target_fraction,
            score=score,
            last_price=last_price,
            reason=_decision_reason(
                "sell",
                score,
                forecast_diag,
                trend_diag,
                earnings_diag,
                annual_vol_pct,
            ),
            diagnostics=diagnostics,
        )

    return SmartPolicyDecision(
        action="HOLD",
        side=None,
        qty=0,
        target_qty=target_qty,
        target_position_fraction=target_fraction,
        score=score,
        last_price=last_price,
        reason=_decision_reason(
            "hold",
            score,
            forecast_diag,
            trend_diag,
            earnings_diag,
            annual_vol_pct,
        ),
        diagnostics=diagnostics,
    )


def _clean_close(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty or "close" not in df.columns:
        return pd.Series(dtype=float)
    close = df["close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return pd.to_numeric(close, errors="coerce").dropna().sort_index()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        if np.isfinite(result):
            return result
    except Exception:
        pass
    return default


def _hold(reason: str, last_price: float = 0.0) -> SmartPolicyDecision:
    return SmartPolicyDecision(
        action="HOLD",
        side=None,
        qty=0,
        target_qty=0,
        target_position_fraction=0.0,
        score=0.0,
        last_price=float(last_price or 0.0),
        reason=reason,
    )


def _result_metrics(model_results: dict[str, Any] | None, name: str) -> dict[str, Any]:
    if not model_results or name not in model_results:
        return {}
    result = model_results.get(name)
    metrics = getattr(result, "metrics", {})
    return dict(metrics or {}) if isinstance(metrics, dict) else {}


def _forecast_component(metrics: dict[str, Any], config: SmartPolicyConfig) -> tuple[float, dict[str, float]]:
    forecast_return = _safe_float(metrics.get("forecast_change_pct"), 0.0)
    expected_error = max(_safe_float(metrics.get("expected_error_pct"), 0.0), 0.25)
    confidence = _safe_float(metrics.get("confidence_pct"), 50.0)
    edge = max(confidence - 50.0, 0.0)
    cost_hurdle = config.trade_cost_bps / 100.0
    required_move = max(expected_error * config.min_forecast_error_ratio, cost_hurdle)
    lower_confidence_bound = (
        forecast_return
        - config.confidence_bound_scale * expected_error
        - cost_hurdle
    )

    if edge < config.min_model_edge_pct or abs(forecast_return) < required_move:
        component = 0.0
    else:
        risk_adjusted = abs(forecast_return) / expected_error
        component = np.sign(forecast_return) * risk_adjusted * min(edge / 10.0, 1.5)
        component = float(np.clip(component, -1.5, 1.5))

    return component, {
        "component": float(component),
        "forecast_return_pct": forecast_return,
        "expected_error_pct": expected_error,
        "confidence_pct": confidence,
        "edge_pct": edge,
        "required_move_pct": required_move,
        "lower_confidence_bound_pct": lower_confidence_bound,
    }


def _trend_component(close: pd.Series, config: SmartPolicyConfig) -> tuple[float, dict[str, float]]:
    fast = close.rolling(config.fast_window).mean()
    slow = close.rolling(config.slow_window).mean()
    regime = close.rolling(config.regime_window).mean()
    last_price = float(close.iloc[-1])
    fast_now = _safe_float(fast.iloc[-1], np.nan)
    slow_now = _safe_float(slow.iloc[-1], np.nan)
    regime_now = _safe_float(regime.iloc[-1], np.nan)

    component = 0.0
    if np.isfinite(fast_now) and np.isfinite(slow_now):
        if fast_now > slow_now and last_price > slow_now:
            component += 0.65
        elif fast_now < slow_now and last_price < slow_now:
            component -= 0.65

    if np.isfinite(regime_now):
        if last_price > regime_now and slow_now > regime_now:
            component += 0.20
        elif last_price < regime_now and slow_now < regime_now:
            component -= 0.20

    return float(np.clip(component, -0.85, 0.85)), {
        "component": float(np.clip(component, -0.85, 0.85)),
        "fast_sma": fast_now,
        "slow_sma": slow_now,
        "regime_sma": regime_now,
    }


def _momentum_component(close: pd.Series) -> tuple[float, dict[str, float]]:
    ret_5 = _lookback_return_pct(close, 5)
    ret_20 = _lookback_return_pct(close, 20)
    rsi = _rsi(close)

    component = 0.0
    if np.isfinite(ret_20):
        component += float(np.clip(ret_20 / 20.0, -0.35, 0.35))
    if np.isfinite(ret_5):
        component += float(np.clip(ret_5 / 12.0, -0.20, 0.20))

    if np.isfinite(rsi):
        if rsi >= 75.0:
            component -= 0.25
        elif rsi <= 30.0 and component >= 0.0:
            component += 0.10
        elif rsi <= 30.0:
            component -= 0.10

    return float(np.clip(component, -0.55, 0.55)), {
        "component": float(np.clip(component, -0.55, 0.55)),
        "return_5d_pct": ret_5,
        "return_20d_pct": ret_20,
        "rsi": rsi,
    }


def _earnings_component(
    metrics: dict[str, Any],
    config: SmartPolicyConfig,
) -> tuple[float, dict[str, Any]]:
    """Use one fresh point-in-time earnings interpretation exactly once."""
    event_flag = bool(
        metrics.get("earnings_event_flag", metrics.get("event_flag", False))
    )
    event_score = float(
        np.clip(
            _safe_float(
                metrics.get("earnings_event_score", metrics.get("event_score")),
                0.0,
            ),
            -1.0,
            1.0,
        )
    )
    confidence = float(
        np.clip(
            _safe_float(
                metrics.get(
                    "earnings_confidence",
                    metrics.get(
                        "event_confidence",
                        metrics.get("confidence"),
                    ),
                ),
                0.0,
            ),
            0.0,
            1.0,
        )
    )
    eligible = (
        event_flag
        and confidence >= config.minimum_earnings_confidence
        and metrics.get("earnings_policy_eligible", False) is True
    )
    component = (
        float(
            np.clip(
                event_score * confidence,
                -config.maximum_earnings_component,
                config.maximum_earnings_component,
            )
        )
        if eligible
        else 0.0
    )
    return component, {
        "component": component,
        "event_flag": event_flag,
        "event_score": event_score,
        "confidence": confidence,
        "eligible": eligible,
        "outcome": str(metrics.get("earnings_outcome", "") or ""),
        "summary": str(metrics.get("earnings_summary", "") or ""),
        "effective_session": str(
            metrics.get("earnings_effective_session", "") or ""
        ),
        "reported_at": str(
            metrics.get("earnings_reported_at", "") or ""
        ),
        "is_stale": bool(metrics.get("earnings_is_stale", False)),
        "policy_eligible": bool(
            metrics.get("earnings_policy_eligible", False)
        ),
        "blockers": list(metrics.get("earnings_blockers", []) or []),
        "data_quality_flags": list(
            metrics.get("earnings_data_quality_flags", []) or []
        ),
        "calendar_source": str(
            metrics.get("earnings_calendar_source", "") or ""
        ),
        "error_code": str(metrics.get("earnings_error_code", "") or ""),
    }


def _rl_component(metrics: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    action = str(metrics.get("rl_action", "hold") or "hold").lower()
    q_short = _safe_float(metrics.get("rl_q_short"), 0.0)
    q_hold = _safe_float(metrics.get("rl_q_hold"), 0.0)
    q_long = _safe_float(metrics.get("rl_q_long"), 0.0)
    visits = max(int(_safe_float(metrics.get("rl_state_visits"), 0.0)), 0)

    if action not in {"short", "hold", "long"}:
        action = "hold"

    q_values = np.asarray([q_short, q_hold, q_long], dtype=float)
    q_gap = float(np.max(q_values) - np.partition(q_values, -2)[-2]) if q_values.size >= 2 else 0.0
    if np.allclose(q_values, q_values[0]) or q_gap <= 0.0:
        action = "hold"

    return 0.0, {
        "component": 0.0,
        "action": action,
        "q_short": q_short,
        "q_hold": q_hold,
        "q_long": q_long,
        "q_gap": q_gap,
        "state_visits": visits,
        "shadow_mode": True,
    }


def _signal_component(signal: pd.Series | None) -> float:
    if signal is None or len(signal.dropna()) == 0:
        return 0.0
    latest = int(signal.dropna().iloc[-1])
    return 0.10 if latest == 1 else -0.10


def _lookback_return_pct(close: pd.Series, lookback: int) -> float:
    if len(close) <= lookback:
        return np.nan
    base = float(close.iloc[-lookback - 1])
    if base <= 0:
        return np.nan
    return (float(close.iloc[-1]) / base - 1.0) * 100.0


def _rsi(close: pd.Series, window: int = 14) -> float:
    if len(close) <= window:
        return np.nan
    changes = close.diff()
    gains = changes.clip(lower=0).rolling(window).mean()
    losses = (-changes.clip(upper=0)).rolling(window).mean()
    rs = gains / losses.replace(0, np.nan)
    value = 100.0 - (100.0 / (1.0 + rs.iloc[-1]))
    return _safe_float(value, np.nan)


def _annualized_volatility_pct(close: pd.Series) -> float:
    returns = close.pct_change().dropna()
    if returns.empty:
        return 0.0
    window = returns.tail(min(60, len(returns)))
    return float(window.std(ddof=0) * np.sqrt(252.0) * 100.0)


def _target_fraction(
    score: float,
    annual_vol_pct: float,
    max_fraction: float,
    config: SmartPolicyConfig,
) -> float:
    score_strength = np.clip((score - config.buy_score_threshold) / 1.75, 0.0, 1.0)
    vol_scale = 1.0
    if annual_vol_pct > 0.0:
        vol_scale = np.clip(config.target_volatility_pct / annual_vol_pct, 0.25, 1.25)
    raw_budget_fraction = float(np.clip((0.35 + 0.65 * score_strength) * vol_scale, 0.0, 1.0))
    actions = sorted(
        {
            float(np.clip(action, 0.0, 1.0))
            for action in config.risk_budget_actions
        }
    )
    eligible_actions = [action for action in actions if action <= raw_budget_fraction + 1e-12]
    selected_action = max(eligible_actions, default=0.0)
    return float(np.clip(max_fraction * selected_action, 0.0, max_fraction))


def _allocation_eligibility(
    metrics: dict[str, Any],
    model_results: dict[str, Any] | None,
    config: SmartPolicyConfig,
) -> tuple[bool, dict[str, Any]]:
    """Return whether a new long allocation clears every uncertainty gate."""
    blockers: list[str] = []
    forecast_return = _safe_float(metrics.get("forecast_change_pct"), 0.0)
    expected_error = max(_safe_float(metrics.get("expected_error_pct"), 0.0), 0.25)
    cost_pct = config.trade_cost_bps / 100.0
    lower_bound = forecast_return - config.confidence_bound_scale * expected_error - cost_pct
    reliability = str(metrics.get("reliability", "") or "").strip().lower()
    confidence = _safe_float(metrics.get("confidence_pct"), 50.0)
    model_edge = max(confidence - 50.0, 0.0)
    holdout_mae = _safe_float(metrics.get("holdout_mae_pct"), np.nan)
    validation_samples = int(max(_safe_float(metrics.get("holdout_samples"), 0.0), 0.0))
    direction_accuracy = _safe_float(metrics.get("holdout_direction_accuracy"), np.nan)
    calibration_error = _safe_float(metrics.get("calibration_error_pct"), np.nan)
    brier_score = _safe_float(metrics.get("brier_score"), np.nan)
    earnings_flag = bool(
        metrics.get("earnings_event_flag", metrics.get("event_flag", False))
    )
    earnings_score = _safe_float(
        metrics.get("earnings_event_score", metrics.get("event_score")),
        0.0,
    )
    earnings_confidence = _safe_float(
        metrics.get("earnings_confidence", metrics.get("event_confidence")),
        0.0,
    )

    derived_low_reliability = (
        reliability == "low"
        or model_edge < config.min_model_edge_pct
        or (
            np.isfinite(holdout_mae)
            and holdout_mae > 0.0
            and abs(forecast_return) < 0.5 * holdout_mae
        )
    )
    if derived_low_reliability:
        blockers.append("low_reliability")
    if forecast_return <= 0.0:
        blockers.append("non_positive_forecast")
    if lower_bound <= 0.0:
        blockers.append("non_positive_oos_lower_bound")
    if bool(metrics.get("forecast_outlier", False)):
        blockers.append("extreme_forecast")
    if config.require_oos_validation and metrics.get("validation_is_oos") is not True:
        blockers.append("missing_oos_validation")
    if validation_samples < config.min_validation_samples:
        blockers.append("insufficient_validation_samples")
    if not np.isfinite(direction_accuracy) or direction_accuracy < config.min_direction_accuracy_pct:
        blockers.append("weak_direction_calibration")
    if not np.isfinite(calibration_error) or calibration_error > config.max_calibration_error_pct:
        blockers.append("poor_probability_calibration")
    if not np.isfinite(brier_score) or brier_score > config.max_brier_score:
        blockers.append("poor_brier_score")
    if (
        earnings_flag
        and earnings_confidence >= config.minimum_earnings_confidence
        and earnings_score <= config.adverse_earnings_block_threshold
    ):
        blockers.append("adverse_earnings_event")

    agreeing_models, eligible_models = _model_agreement(
        metrics,
        model_results,
        config,
    )
    if agreeing_models < config.min_model_agreement:
        blockers.append("insufficient_model_agreement")

    return not blockers, {
        "allocation_blockers": blockers,
        "lower_confidence_bound_pct": float(lower_bound),
        "agreeing_models": int(agreeing_models),
        "eligible_models": int(eligible_models),
        "validation_samples": int(validation_samples),
        "calibration_error_pct": float(calibration_error) if np.isfinite(calibration_error) else np.nan,
        "brier_score": float(brier_score) if np.isfinite(brier_score) else np.nan,
    }


def _model_agreement(
    primary_metrics: dict[str, Any],
    model_results: dict[str, Any] | None,
    config: SmartPolicyConfig,
) -> tuple[int, int]:
    if not model_results:
        return 0, 0

    primary_horizon = int(_safe_float(primary_metrics.get("horizon_days"), 0.0))
    primary_direction = np.sign(_safe_float(primary_metrics.get("forecast_change_pct"), 0.0))
    agreeing = 0
    eligible = 0
    for name, result in model_results.items():
        if name in {"RL Policy", "Ensemble"}:
            continue
        model_metrics = getattr(result, "metrics", None)
        if not isinstance(model_metrics, dict):
            continue
        if model_metrics.get("shadow_mode") or model_metrics.get("live_eligible", True) is False:
            continue
        if model_metrics.get("validation_is_oos") is not True:
            continue
        if int(_safe_float(model_metrics.get("holdout_samples"), 0.0)) < config.min_validation_samples:
            continue
        model_calibration = _safe_float(
            model_metrics.get("calibration_error_pct"),
            np.nan,
        )
        model_brier = _safe_float(model_metrics.get("brier_score"), np.nan)
        if (
            not np.isfinite(model_calibration)
            or model_calibration > config.max_calibration_error_pct
            or not np.isfinite(model_brier)
            or model_brier > config.max_brier_score
        ):
            continue
        if model_metrics.get("horizon_days") is None:
            continue
        model_horizon = int(_safe_float(model_metrics.get("horizon_days"), 0.0))
        if not primary_horizon or model_horizon != primary_horizon:
            continue
        model_return = _safe_float(model_metrics.get("forecast_change_pct"), 0.0)
        if model_return == 0.0:
            continue
        eligible += 1
        if primary_direction != 0.0 and np.sign(model_return) == primary_direction:
            agreeing += 1
    return agreeing, eligible


def _decision_reason(
    action: str,
    score: float,
    forecast_diag: dict[str, float],
    trend_diag: dict[str, float],
    earnings_diag: dict[str, Any],
    annual_vol_pct: float,
) -> str:
    earnings_text = ""
    if earnings_diag.get("event_flag"):
        earnings_text = (
            f"; earnings {earnings_diag.get('event_score', 0.0):+.2f}"
            f" @ {earnings_diag.get('confidence', 0.0):.0%}"
        )
    return (
        f"{action} score {score:.2f}; "
        f"forecast {forecast_diag.get('forecast_return_pct', 0.0):+.2f}% "
        f"vs error {forecast_diag.get('expected_error_pct', 0.0):.2f}%; "
        f"trend {trend_diag.get('component', 0.0):+.2f}; "
        f"vol {annual_vol_pct:.1f}%"
        f"{earnings_text}"
    )
