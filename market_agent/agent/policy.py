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
    max_position_fraction: float = 0.25
    stop_loss_pct: float = 0.08
    trade_cost_bps: float = 5.0


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
    risk_fraction: float = 0.10,
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
    if score >= 1.75:
        policy_call = "Strong Buy"
    elif score >= config.buy_score_threshold:
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
    rl_diag = diagnostics.get("rl", {}) or {}

    return {
        "policy_call": policy_call,
        "policy_score": score,
        "policy_target_pct": float(decision.target_position_fraction * 100.0),
        "policy_reason": decision.reason,
        "forecast_component": float(forecast_diag.get("component", 0.0)),
        "trend_component": float(trend_diag.get("component", 0.0)),
        "momentum_component": float(momentum_diag.get("component", 0.0)),
        "rl_component": float(rl_diag.get("component", 0.0)),
        "rl_action": rl_diag.get("action", ""),
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
    Combine forecast edge, trend, tabular RL, volatility, and position state.

    This is deliberately long-only because the existing broker layer submits buy
    and sell orders, but does not manage borrow, margin, or short exposure.
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
    ensemble_metrics = _result_metrics(model_results, "Ensemble")
    rl_metrics = _result_metrics(model_results, "RL Policy")

    forecast_component, forecast_diag = _forecast_component(primary_metrics, config)
    ensemble_component, ensemble_diag = _forecast_component(ensemble_metrics, config)
    trend_component, trend_diag = _trend_component(close, config)
    momentum_component, momentum_diag = _momentum_component(close)
    rl_component, rl_diag = _rl_component(rl_metrics)
    signal_component = _signal_component(signal)

    score = (
        forecast_component
        + 0.35 * ensemble_component
        + trend_component
        + momentum_component
        + rl_component
        + signal_component
    )
    score = float(np.clip(score, -3.0, 3.0))

    annual_vol_pct = _annualized_volatility_pct(close)
    current_fraction = current_qty * last_price / equity
    max_fraction = min(max(risk_fraction, 0.0), config.max_position_fraction)
    target_fraction = 0.0

    if score >= config.buy_score_threshold and max_fraction > 0.0:
        target_fraction = _target_fraction(score, annual_vol_pct, max_fraction, config)
    elif current_qty > 0 and score > config.sell_score_threshold:
        target_fraction = min(current_fraction, max_fraction)

    target_qty = int((equity * target_fraction) // last_price)
    target_qty = max(target_qty, 0)
    qty_delta = target_qty - current_qty

    diagnostics = {
        "score": score,
        "current_position_fraction": current_fraction,
        "target_position_fraction": target_fraction,
        "annual_volatility_pct": annual_vol_pct,
        "forecast": forecast_diag,
        "ensemble": ensemble_diag,
        "trend": trend_diag,
        "momentum": momentum_diag,
        "rl": rl_diag,
        "signal_component": signal_component,
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
            reason=_decision_reason("buy", score, forecast_diag, trend_diag, annual_vol_pct),
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
            reason=_decision_reason("sell", score, forecast_diag, trend_diag, annual_vol_pct),
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
        reason=_decision_reason("hold", score, forecast_diag, trend_diag, annual_vol_pct),
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


def _rl_component(metrics: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    action = str(metrics.get("rl_action", "")).lower()
    q_short = _safe_float(metrics.get("rl_q_short"), 0.0)
    q_hold = _safe_float(metrics.get("rl_q_hold"), 0.0)
    q_long = _safe_float(metrics.get("rl_q_long"), 0.0)

    if action not in {"short", "hold", "long"}:
        return 0.0, {"component": 0.0, "action": ""}

    q_values = np.asarray([q_short, q_hold, q_long], dtype=float)
    q_gap = float(np.max(q_values) - np.partition(q_values, -2)[-2]) if q_values.size >= 2 else 0.0
    scale = min(max(abs(q_gap) / 0.03, 0.0), 1.0)
    if action == "long":
        component = 0.20 + 0.35 * scale
    elif action == "short":
        component = -0.20 - 0.35 * scale
    else:
        component = 0.0

    return float(component), {
        "component": float(component),
        "action": action,
        "q_short": q_short,
        "q_hold": q_hold,
        "q_long": q_long,
        "q_gap": q_gap,
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
    fraction = max_fraction * (0.35 + 0.65 * score_strength) * vol_scale
    return float(np.clip(fraction, 0.0, max_fraction))


def _decision_reason(
    action: str,
    score: float,
    forecast_diag: dict[str, float],
    trend_diag: dict[str, float],
    annual_vol_pct: float,
) -> str:
    return (
        f"{action} score {score:.2f}; "
        f"forecast {forecast_diag.get('forecast_return_pct', 0.0):+.2f}% "
        f"vs error {forecast_diag.get('expected_error_pct', 0.0):.2f}%; "
        f"trend {trend_diag.get('component', 0.0):+.2f}; "
        f"vol {annual_vol_pct:.1f}%"
    )
