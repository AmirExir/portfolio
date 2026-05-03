from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd


def _column(frame: pd.DataFrame, name: str) -> str:
    if name in frame.columns:
        return name
    title_name = name.title()
    if title_name in frame.columns:
        return title_name
    raise KeyError(f"Missing {name} column.")


def clean_series(frame: pd.DataFrame, column_name: str) -> pd.Series:
    series = frame[_column(frame, column_name)]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return pd.to_numeric(series, errors="coerce").dropna()


def latest_value(series: pd.Series, default=np.nan) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return default
    return float(clean.iloc[-1])


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    changes = close.diff()
    gains = changes.clip(lower=0).rolling(window).mean()
    losses = (-changes.clip(upper=0)).rolling(window).mean()
    rs = gains / losses.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def recent_cross(short_ma: pd.Series, long_ma: pd.Series, direction: str, lookback: int = 5) -> bool:
    diff = (short_ma - long_ma).dropna()
    if len(diff) < 2:
        return False
    recent = diff.tail(max(2, lookback + 1))
    previous = recent.shift(1)
    if direction == "bullish":
        return bool(((previous <= 0) & (recent > 0)).any())
    return bool(((previous >= 0) & (recent < 0)).any())


def recognize_patterns(frame: pd.DataFrame, short_window: int = 20, long_window: int = 50) -> dict:
    close = clean_series(frame, "close")
    if close.shape[0] < max(60, short_window, long_window):
        raise ValueError("Not enough clean close prices for pattern scan.")

    try:
        volume = clean_series(frame, "volume")
    except Exception:
        volume = pd.Series(dtype=float)

    last_close = float(close.iloc[-1])
    sma_short = close.rolling(short_window).mean()
    sma_long = close.rolling(long_window).mean()
    sma_200 = close.rolling(200).mean()
    ret_5 = (last_close / float(close.iloc[-6]) - 1.0) * 100.0 if close.shape[0] > 5 else np.nan
    ret_20 = (last_close / float(close.iloc[-21]) - 1.0) * 100.0 if close.shape[0] > 20 else np.nan
    ret_60 = (last_close / float(close.iloc[-61]) - 1.0) * 100.0 if close.shape[0] > 60 else np.nan
    high_60 = float(close.tail(60).max())
    low_60 = float(close.tail(60).min())

    daily_returns = close.pct_change().dropna()
    rolling_vol = daily_returns.rolling(20).std() * np.sqrt(252) * 100.0
    latest_vol = latest_value(rolling_vol)
    vol_percentile = np.nan
    vol_window = rolling_vol.dropna().tail(252)
    if not vol_window.empty and np.isfinite(latest_vol):
        vol_percentile = float((vol_window <= latest_vol).mean() * 100.0)

    latest_rsi = latest_value(rsi(close))
    volume_ratio = np.nan
    if not volume.empty:
        average_volume = volume.tail(20).mean()
        if np.isfinite(average_volume) and average_volume > 0:
            volume_ratio = float(volume.iloc[-1] / average_volume)

    short_now = latest_value(sma_short)
    long_now = latest_value(sma_long)
    sma200_now = latest_value(sma_200)
    uptrend = (
        np.isfinite(short_now)
        and np.isfinite(long_now)
        and last_close > long_now
        and short_now > long_now
        and (not np.isfinite(sma200_now) or long_now > sma200_now)
    )
    downtrend = (
        np.isfinite(short_now)
        and np.isfinite(long_now)
        and last_close < long_now
        and short_now < long_now
        and (not np.isfinite(sma200_now) or long_now < sma200_now)
    )

    patterns = []
    if uptrend:
        patterns.append("Uptrend")
    if downtrend:
        patterns.append("Downtrend")
    if recent_cross(sma_short, sma_long, "bullish"):
        patterns.append("Bullish MA Crossover")
    if recent_cross(sma_short, sma_long, "bearish"):
        patterns.append("Bearish MA Crossover")
    if high_60 > 0 and last_close >= high_60 * 0.98 and ret_20 > 0:
        patterns.append("Near 60-Day High")
    if low_60 > 0 and last_close <= low_60 * 1.08 and ret_5 > 0:
        patterns.append("Rebound From 60-Day Low")
    if uptrend and np.isfinite(short_now) and np.isfinite(long_now) and short_now > last_close > long_now:
        patterns.append("Pullback In Uptrend")
    if np.isfinite(latest_rsi) and latest_rsi < 35:
        patterns.append("Oversold")
    if np.isfinite(latest_rsi) and latest_rsi > 70:
        patterns.append("Overbought")
    if np.isfinite(volume_ratio) and volume_ratio >= 1.8:
        patterns.append("Volume Spike")
    if np.isfinite(vol_percentile) and vol_percentile >= 80:
        patterns.append("High Volatility")
    if not patterns:
        patterns.append("Range / No Dominant Pattern")

    return {
        "Primary Pattern": patterns[0],
        "All Patterns": ", ".join(patterns),
        "Last Price": last_close,
        "5D Return %": ret_5,
        "20D Return %": ret_20,
        "60D Return %": ret_60,
        "RSI": latest_rsi,
        "20D Volatility %": latest_vol,
        "Volume Ratio": volume_ratio,
    }


def scan_patterns(
    symbols: Iterable[str],
    history_days: int,
    get_ohlcv: Callable[[str, int], pd.DataFrame],
    short_window: int = 20,
    long_window: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    detail_rows = []
    pattern_rows = []
    errors = []

    for symbol in symbols:
        try:
            frame = get_ohlcv(symbol, history_days)
            pattern_info = recognize_patterns(frame, short_window=short_window, long_window=long_window)
            detail_rows.append({"Symbol": symbol, **pattern_info})
            for pattern in str(pattern_info["All Patterns"]).split(", "):
                pattern_rows.append(
                    {
                        "Pattern": pattern,
                        "Symbol": symbol,
                        "20D Return %": pattern_info["20D Return %"],
                        "60D Return %": pattern_info["60D Return %"],
                    }
                )
        except Exception as pattern_error:
            errors.append(f"{symbol}: {pattern_error}")

    detail_df = pd.DataFrame(detail_rows)
    if not pattern_rows:
        return pd.DataFrame(), detail_df, errors

    raw_patterns = pd.DataFrame(pattern_rows)
    summary_rows = []
    for pattern, group in raw_patterns.groupby("Pattern"):
        symbols_list = sorted(group["Symbol"].unique().tolist())
        summary_rows.append(
            {
                "Pattern": pattern,
                "Count": int(group["Symbol"].nunique()),
                "Avg 20D Return %": float(group["20D Return %"].mean()),
                "Avg 60D Return %": float(group["60D Return %"].mean()),
                "Symbols": ", ".join(symbols_list),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["Count", "Avg 20D Return %"], ascending=[False, False])
    return summary_df, detail_df, errors
