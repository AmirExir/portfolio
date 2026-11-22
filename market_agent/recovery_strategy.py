"""Long-only US-equity research rule evaluated after each completed close."""

from __future__ import annotations

import numpy as np
import pandas as pd

if __package__:
    from .agent.policy import SmartPolicyConfig
    from .patterns import OVERSOLD_RSI, RSI_WINDOW, rsi
    from .risk_alerts import RANGE, SHOCK_FLOOR_PCT, SHOCK_MULTIPLIER
else:  # The scheduled runner executes daily_ml_forecast_report.py directly.
    from agent.policy import SmartPolicyConfig
    from patterns import OVERSOLD_RSI, RSI_WINDOW, rsi
    from risk_alerts import RANGE, SHOCK_FLOOR_PCT, SHOCK_MULTIPLIER


PROBE = 0.25
CONFIRM_SESSIONS = 5
MAX_PROBE_SESSIONS = 30
VERSION = "equity-recovery-v1"


def recovery_path(close: pd.Series, *, entry: str = "confirmed") -> pd.DataFrame:
    """Compute close-time research decisions; the earliest simulated fill is next open.

    'trend' disables the starter. 'immediate' buys a starter on an oversold
    close with the same breakdown veto, exit rules and size as 'confirmed'.
    The targets describe fractions of one research sleeve, not account weights.
    """
    if entry not in {"trend", "immediate", "confirmed"}:
        raise ValueError(f"Unknown entry policy: {entry}")
    if (
        not close.index.is_unique
        or not close.index.is_monotonic_increasing
        or not np.isfinite(close).all()
        or (close <= 0).any()
    ):
        raise ValueError("Expected unique, ordered, finite, positive closes")
    policy = SmartPolicyConfig()
    fast = close.rolling(policy.fast_window).mean()
    slow = close.rolling(policy.slow_window).mean()
    oscillator = rsi(close)
    changes = close.pct_change(fill_method=None) * 100
    threshold = (SHOCK_MULTIPLIER * changes.shift(1).rolling(RANGE).std()).clip(
        lower=SHOCK_FLOOR_PCT
    )
    shock = changes <= -threshold
    broken = close < close.shift(1).rolling(RANGE).min()
    oversold = oscillator < OVERSOLD_RSI
    rebound = (oscillator >= OVERSOLD_RSI) & (
        close > close.shift(1).rolling(CONFIRM_SESSIONS).max()
    )
    support = close.rolling(RANGE).min()
    trend_entry = (close > fast) & (fast > slow)
    trend_hold = (close > slow) & (fast > slow)

    weights = np.zeros(len(close))
    states = []
    confirmation_watch_next = []
    armed_until = -1
    probe_end = -1
    probe_floor = 0.0
    previous = 0.0
    inputs = zip(
        close, slow, oversold, shock | broken, trend_entry, trend_hold, rebound, support
    )
    for i, (price, slow_ma, low_rsi, risk, trend, hold, bounce, low) in enumerate(
        inputs
    ):
        weight, state = 0.0, "Flat"
        if low_rsi:
            armed_until = i + RANGE
        if pd.isna(slow_ma):
            state = "Warmup"
        elif risk:
            state = "Oversold / risk" if low_rsi else "Risk exit"
        elif trend or (previous == 1 and hold):
            weight, state = 1.0, "Trend"
            armed_until = -1
        elif previous == PROBE:
            if i < probe_end and price >= probe_floor:
                weight, state = PROBE, "Starter held"
            else:
                state = "Starter expired" if i >= probe_end else "Starter invalidated"
        elif i <= armed_until:
            if (entry == "confirmed" and bounce) or (entry == "immediate" and low_rsi):
                weight, state = PROBE, "Starter entry"
                probe_end, probe_floor, armed_until = i + MAX_PROBE_SESSIONS, low, -1
            else:
                state = "Watching for recovery"
        weights[i], previous = weight, weight
        states.append(state)
        confirmation_watch_next.append(weight == 0 and armed_until > i)
    return pd.DataFrame(
        {
            "state": states,
            "sleeve_fraction": weights,
            "rsi": oscillator,
            "confirmation_watch_next": confirmation_watch_next,
            "next_breakout_close": close.rolling(CONFIRM_SESSIONS).max(),
        },
        index=close.index,
    )


def recovery_snapshot(close: pd.Series) -> dict:
    """Return the latest research reading for the report, including the next closing hurdle."""
    path = recovery_path(close)
    if path.empty or path.iloc[-1]["state"] == "Warmup":
        return {}
    row = path.iloc[-1]
    early = recovery_path(close, entry="immediate").iloc[-1]
    # Tomorrow's simple RSI keeps these 13 changes and adds one unknown change.
    prior_changes = close.diff().tail(RSI_WINDOW - 1)
    gains = float(prior_changes.clip(lower=0).sum())
    losses = float(-prior_changes.clip(upper=0).sum())
    ratio = OVERSOLD_RSI / (100 - OVERSOLD_RSI)
    gain_needed = ratio * losses - gains
    move_for_rsi = gain_needed if gain_needed >= 0 else gain_needed / ratio
    return {
        "next_rsi_close": float(close.iloc[-1]) + move_for_rsi,
        "version": VERSION,
        "recently_oversold": bool((path["rsi"].tail(RANGE + 1) < OVERSOLD_RSI).any()),
        "confirmed_state": row["state"],
        "confirmed_sleeve_fraction": float(row["sleeve_fraction"]),
        "early_state": early["state"],
        "early_sleeve_fraction": float(early["sleeve_fraction"]),
        "rsi": float(row["rsi"]) if pd.notna(row["rsi"]) else None,
        "confirmation_watch_next": bool(row["confirmation_watch_next"]),
        "next_breakout_close": float(row["next_breakout_close"]),
        "recovery_rsi": OVERSOLD_RSI,
    }
