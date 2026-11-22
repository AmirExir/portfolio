"""Descriptive indicators and chart setups calculated from completed closes."""

from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd


RSI_PERIOD = 14
RSI_OVERSOLD = 35.0
RSI_OVERBOUGHT = 70.0
TREND_CONFIRM_PERIOD = 200
PATTERN_SESSIONS = 60
CROSSOVER_LOOKBACK = 5
PIVOT_NEIGHBORS = 2
DOUBLE_TOP_LABEL = "Confirmed Double Top"
DOUBLE_BOTTOM_LABEL = "Confirmed Double Bottom"
DOUBLE_MIN_SEPARATION = 5
DOUBLE_MAX_SEPARATION = 30
DOUBLE_MAX_BREAK_SESSIONS = 10
DOUBLE_MAX_TURN_DIFFERENCE = 0.02  # fraction of the lower turning price
DOUBLE_MIN_NECKLINE_DEPTH = 0.03  # fraction measured from the nearer turn


def rsi(close: pd.Series, window: int = RSI_PERIOD) -> pd.Series:
    """Rolling simple-mean RSI, with an all-flat window defined as neutral (50)."""
    changes = close.diff()
    gains = changes.clip(lower=0).rolling(window).mean()
    losses = (-changes.clip(upper=0)).rolling(window).mean()
    total_change = gains + losses
    return (100.0 * gains.div(total_change.where(total_change > 0))).mask(total_change == 0, 50.0)


def recent_cross(
    short_ma: pd.Series, long_ma: pd.Series, direction: str, lookback: int = CROSSOVER_LOOKBACK,
) -> bool:
    """Detect a moving-average crossover during the latest lookback sessions."""
    recent = (short_ma - long_ma).tail(lookback + 1)
    previous = recent.shift(1)
    if direction == "bullish":
        return bool(((previous <= 0) & (recent > 0)).any())
    return bool(((previous >= 0) & (recent < 0)).any())


def _confirmed_swings(prices: list[float]) -> tuple[list[int], list[int]]:
    """Locate strict local extremes that have two completed closes on each side."""
    swing_lows: list[int] = []
    swing_highs: list[int] = []
    for i in range(PIVOT_NEIGHBORS, len(prices) - PIVOT_NEIGHBORS):
        value = prices[i]
        neighbors = prices[i - PIVOT_NEIGHBORS:i] + prices[i + 1:i + 1 + PIVOT_NEIGHBORS]
        if value < min(neighbors):
            swing_lows.append(i)
        if value > max(neighbors):
            swing_highs.append(i)
    return swing_lows, swing_highs


def _confirmed_neckline(
    prices: list[float], turns: list[int], opposite_turns: list[int], *, top: bool,
) -> float | None:
    """Return the intervening pivot price on the first confirming close."""
    if len(turns) < 2:
        return None
    first, second = turns[-2:]
    if not DOUBLE_MIN_SEPARATION <= second - first <= DOUBLE_MAX_SEPARATION:
        return None
    if len(prices) - 1 - second > DOUBLE_MAX_BREAK_SESSIONS:
        return None
    lower, upper = sorted((prices[first], prices[second]))
    if upper > lower * (1 + DOUBLE_MAX_TURN_DIFFERENCE):
        return None

    between = prices[first + 1:second]
    neckline = min(between) if top else max(between)
    neckline_index = first + 1 + between.index(neckline)
    if neckline_index not in opposite_turns or between.count(neckline) != 1:
        return None

    before_today = prices[second:-1]
    if top:
        confirmed = (
            neckline <= lower * (1 - DOUBLE_MIN_NECKLINE_DEPTH)
            and max(before_today) <= upper and min(before_today) >= neckline
            and prices[-1] < neckline
        )
    else:
        confirmed = (
            neckline >= upper * (1 + DOUBLE_MIN_NECKLINE_DEPTH)
            and min(before_today) >= lower and max(before_today) <= neckline
            and prices[-1] > neckline
        )
    return float(neckline) if confirmed else None


def valid_trend_windows(short_window: int, long_window: int) -> bool:
    """Return whether the short/long settings describe two ordered SMA windows."""
    return (
        isinstance(short_window, Integral) and not isinstance(short_window, bool)
        and isinstance(long_window, Integral) and not isinstance(long_window, bool)
        and 0 < short_window < long_window
    )


def close_technicals(close: pd.Series, short_window: int = 20, long_window: int = 50) -> dict:
    """Describe the final bar of a nonempty, chronological series of valid closes.

    Trend compares the close and short SMA to the long SMA. A 200-session
    SMA confirms the long average when it is available and extends that
    horizon. RSI uses the scanner's simple 14-change averages and strict
    35/70 cutoffs. Each pivot requires two strictly higher or lower closes
    on both sides, so a plateau or an unconfirmed endpoint is excluded.
    """
    if close.empty or not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("Technical readings require a nonempty series of finite, positive closes.")
    last = float(close.iloc[-1])
    value = float(rsi(close).iloc[-1])
    if not np.isfinite(value):
        rsi_state = "Unavailable"
    elif value < RSI_OVERSOLD:
        rsi_state = "Oversold"
    elif value > RSI_OVERBOUGHT:
        rsi_state = "Overbought"
    else:
        rsi_state = "Neutral"

    trend = "Unavailable"
    windows: list[int] = []
    patterns: list[str] = []
    pullback = False
    if valid_trend_windows(short_window, long_window) and len(close) >= long_window:
        short_ma = close.rolling(short_window).mean()
        long_ma = close.rolling(long_window).mean()
        short_now, long_now = float(short_ma.iloc[-1]), float(long_ma.iloc[-1])
        windows = [int(short_window), int(long_window)]
        uptrend = last > long_now and short_now > long_now
        downtrend = last < long_now and short_now < long_now
        if long_window < TREND_CONFIRM_PERIOD <= len(close):
            average_200 = float(close.tail(TREND_CONFIRM_PERIOD).mean())
            windows.append(TREND_CONFIRM_PERIOD)
            uptrend = uptrend and long_now > average_200
            downtrend = downtrend and long_now < average_200
        if uptrend:
            trend = "Uptrend"
        elif downtrend:
            trend = "Downtrend"
        else:
            trend = "Mixed"
        pullback = uptrend and long_now < last < short_now
        if recent_cross(short_ma, long_ma, "bullish"):
            patterns.append("Bullish MA Crossover")
        if recent_cross(short_ma, long_ma, "bearish"):
            patterns.append("Bearish MA Crossover")

    if len(close) >= PATTERN_SESSIONS:
        window = close.tail(PATTERN_SESSIONS)
        if last >= window.max() * 0.98 and last > close.iloc[-21]:
            patterns.append("Near 60-Day High")
        if last <= window.min() * 1.08 and last > close.iloc[-6]:
            patterns.append("Rebound From 60-Day Low")
    if pullback:
        patterns.append("Pullback In Uptrend")
    prices = close.tail(PATTERN_SESSIONS).tolist()
    swing_lows, swing_highs = _confirmed_swings(prices)
    shape_necklines: dict[str, float] = {}
    top_neckline = _confirmed_neckline(prices, swing_highs, swing_lows, top=True)
    if top_neckline is not None:
        shape_necklines[DOUBLE_TOP_LABEL] = top_neckline
    bottom_neckline = _confirmed_neckline(prices, swing_lows, swing_highs, top=False)
    if bottom_neckline is not None:
        shape_necklines[DOUBLE_BOTTOM_LABEL] = bottom_neckline
    patterns.extend(shape_necklines)
    support = max((prices[i] for i in swing_lows if prices[i] < last), default=None)
    resistance = min((prices[i] for i in swing_highs if prices[i] > last), default=None)
    return {
        "available_sessions": len(close),
        "trend": trend,
        "trend_windows": windows,
        "rsi": value if np.isfinite(value) else None,
        "rsi_period": RSI_PERIOD,
        "rsi_state": rsi_state,
        "patterns": patterns,
        "shape_necklines": shape_necklines,
        "pivot_window": PATTERN_SESSIONS,
        "pivot_radius": PIVOT_NEIGHBORS,
        "support": support,
        "resistance": resistance,
    }
