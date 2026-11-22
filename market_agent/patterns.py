from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

if __package__:
    from .agent.technicals import close_technicals, recent_cross, rsi
else:
    from agent.technicals import close_technicals, recent_cross, rsi


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


def recognize_patterns(frame: pd.DataFrame, short_window: int = 20, long_window: int = 50) -> dict:
    close = clean_series(frame, "close")
    if close.shape[0] < max(60, short_window, long_window):
        raise ValueError("Not enough clean close prices for pattern scan.")

    try:
        volume = clean_series(frame, "volume")
    except Exception:
        volume = pd.Series(dtype=float)

    last_close = float(close.iloc[-1])
    ret_5 = (last_close / float(close.iloc[-6]) - 1.0) * 100.0 if close.shape[0] > 5 else np.nan
    ret_20 = (last_close / float(close.iloc[-21]) - 1.0) * 100.0 if close.shape[0] > 20 else np.nan
    ret_60 = (last_close / float(close.iloc[-61]) - 1.0) * 100.0 if close.shape[0] > 60 else np.nan

    daily_returns = close.pct_change().dropna()
    rolling_vol = daily_returns.rolling(20).std() * np.sqrt(252) * 100.0
    latest_vol = latest_value(rolling_vol)
    vol_percentile = np.nan
    vol_window = rolling_vol.dropna().tail(252)
    if not vol_window.empty and np.isfinite(latest_vol):
        vol_percentile = float((vol_window <= latest_vol).mean() * 100.0)

    technical = close_technicals(close, short_window, long_window)
    volume_ratio = np.nan
    if not volume.empty:
        average_volume = volume.tail(20).mean()
        if np.isfinite(average_volume) and average_volume > 0:
            volume_ratio = float(volume.iloc[-1] / average_volume)

    patterns = []
    if technical["trend"] in ("Uptrend", "Downtrend"):
        patterns.append(technical["trend"])
    patterns.extend(technical["patterns"])
    if technical["rsi_state"] in ("Oversold", "Overbought"):
        patterns.append(technical["rsi_state"])
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
        "RSI": technical["rsi"] if technical["rsi"] is not None else np.nan,
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
    errors = []

    for symbol in symbols:
        try:
            frame = get_ohlcv(symbol, history_days)
            pattern_info = recognize_patterns(frame, short_window=short_window, long_window=long_window)
            detail = {"Symbol": symbol, **pattern_info}
            if isinstance(frame.index, pd.DatetimeIndex):
                detail["As Of Session"] = clean_series(frame, "close").index[-1].date().isoformat()
            detail_rows.append(detail)
        except Exception as pattern_error:
            errors.append(f"{symbol}: {pattern_error}")

    detail_df = pd.DataFrame(detail_rows)
    return summarize_patterns(detail_df), detail_df, errors


def summarize_patterns(details: pd.DataFrame) -> pd.DataFrame:
    """Aggregate recorded pattern labels and whichever return horizons the scan retained."""
    if details.empty:
        return pd.DataFrame()
    matches = details.assign(Pattern=details["All Patterns"].str.split(", ")).explode("Pattern")
    matches = matches[matches["Pattern"].notna() & matches["Pattern"].ne("")].drop_duplicates(["Symbol", "Pattern"])
    if matches.empty:
        return pd.DataFrame()
    groups = matches.groupby("Pattern")
    summary = groups["Symbol"].agg(
        Count="nunique", Symbols=lambda group: ", ".join(sorted(group)),
    )
    sort_columns = ["Count"]
    for column in ("20D Return %", "60D Return %"):
        if column in matches:
            summary[f"Avg {column}"] = groups[column].mean()
            if column == "20D Return %":
                sort_columns.append(f"Avg {column}")
    return summary.sort_values(sort_columns, ascending=False, kind="stable").reset_index()
