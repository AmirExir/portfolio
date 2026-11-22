"""Daily price observations for conditional entry and downside watches."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from functools import lru_cache
from math import isclose, isfinite

import numpy as np
import pandas as pd

from .earnings import us_equity_trading_sessions
from .technicals import CROSSOVER_LOOKBACK, TREND_CONFIRM_PERIOD, close_technicals, valid_trend_windows


RANGE_SESSIONS = 20
MOMENTUM_SESSIONS = 5
MIN_SHOCK_PCT = 2.0
SHOCK_VOL_MULTIPLIER = 2.0
ENTRY_PROXIMITY_PCT = 2.0
REQUIRED_CLOSES = RANGE_SESSIONS + 2


@lru_cache(maxsize=128)
def _equity_window(end: date, count: int) -> tuple[date, ...]:
    # Noon UTC is on the same New York civil date throughout the year.
    sessions = us_equity_trading_sessions(
        as_of=datetime.combine(end, time(12), tzinfo=timezone.utc),
        lookback_days=count * 2,
        future_session_count=1,
    )
    return tuple(session for session in sessions if session <= end)[-count:]


def price_watch_observation(
    symbol: str, close: pd.Series, *, short_window: int = 20, long_window: int = 50,
) -> dict | None:
    """Evaluate the last completed close against earlier completed sessions.

    Supply a provider symbol and a chronological, completed-bar series. Missing
    or invalid inputs in the required window yield no observation. The range
    uses prior closes and the volatility uses their prior one-session returns
    (sample standard deviation). The next session's reference range includes
    this completed close. Returns and thresholds are percentages.
    Technical indicators use the uninterrupted valid history ending on the
    same bar. A gap outside the price-watch window can shorten their history
    without suppressing a valid watch. Price levels use the source quote units.
    """
    if not isinstance(symbol, str) or not symbol.strip():
        return None
    symbol = symbol.strip().upper()
    if not isinstance(close, pd.Series) or len(close) < REQUIRED_CLOSES:
        return None
    if pd.api.types.is_numeric_dtype(close.index.dtype):
        return None
    try:
        sessions = pd.DatetimeIndex(pd.to_datetime(close.index, errors="coerce")).normalize()
    except (TypeError, ValueError, OverflowError):
        return None
    if sessions.hasnans or not sessions.is_monotonic_increasing or not sessions.is_unique:
        return None
    trend_history = long_window + CROSSOVER_LOOKBACK if valid_trend_windows(short_window, long_window) else 0
    history_count = min(len(close), max(TREND_CONFIRM_PERIOD, trend_history))
    observed_dates = sessions[-history_count:].date
    end = observed_dates[-1]
    if symbol.endswith("-USD"):
        expected_dates = pd.date_range(end=end, periods=history_count).date
    else:
        expected_dates = _equity_window(end, history_count)

    prices = pd.to_numeric(close.iloc[-history_count:], errors="coerce")
    if pd.api.types.is_complex_dtype(prices.dtype) or pd.api.types.is_bool_dtype(prices.dtype):
        return None
    values = prices.to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(values) & (values > 0.0) & (observed_dates == expected_dates)
    invalid = np.flatnonzero(~valid)
    start = int(invalid[-1] + 1) if len(invalid) else 0
    history = pd.Series(values[start:], index=sessions[-history_count:][start:])
    if len(history) < REQUIRED_CLOSES:
        return None
    values = values[-REQUIRED_CLOSES:]
    with np.errstate(over="ignore", invalid="ignore"):
        returns_pct = 100.0 * (np.diff(values) / values[:-1])
        daily_sd_pct = float(np.std(returns_pct[:-1], ddof=1))
    change_pct = float(returns_pct[-1])
    if not isfinite(change_pct) or not isfinite(daily_sd_pct):
        return None

    last_close = float(values[-1])
    prior_range = values[-RANGE_SESSIONS - 1:-1]
    next_range = values[-RANGE_SESSIONS:]
    prior_low = float(prior_range.min())
    prior_high = float(prior_range.max())
    next_high = float(next_range.max())
    shock_threshold_pct = max(MIN_SHOCK_PCT, SHOCK_VOL_MULTIPLIER * daily_sd_pct)
    # Return arithmetic on decimal quote prices can round at an inclusive boundary.
    shock = -change_pct >= shock_threshold_pct or isclose(
        -change_pct, shock_threshold_pct, rel_tol=1e-12
    )
    breakdown = last_close < prior_low
    entry_floor = next_high * (1.0 - ENTRY_PROXIMITY_PCT / 100.0)
    near_high = last_close >= entry_floor or isclose(last_close, entry_floor, rel_tol=1e-12)
    positive_momentum = last_close > float(values[-1 - MOMENTUM_SESSIONS])
    entry_watch = near_high and positive_momentum and not (shock or breakdown)
    return {
        "symbol": symbol,
        "as_of_session": end.isoformat(),
        "close": last_close,
        "change_pct": change_pct,
        "prior_low": prior_low,
        "prior_high": prior_high,
        "next_low": float(next_range.min()),
        "next_high": next_high,
        "shock_threshold_pct": shock_threshold_pct,
        "shock": shock,
        "breakdown": breakdown,
        "entry_watch": entry_watch,
        "entry_triggered": entry_watch and last_close > prior_high,
        "technical": close_technicals(history, short_window, long_window),
    }
