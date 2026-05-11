import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import contextlib
import io
import logging
import os
import re


def _cache_root() -> Path:
    configured = os.getenv("MARKET_AGENT_DATA_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[1] / "reports" / "ohlcv_cache"


def _cache_path(symbol: str, interval: str) -> Path:
    safe_symbol = re.sub(r"[^A-Za-z0-9_.-]+", "_", symbol)
    safe_interval = re.sub(r"[^A-Za-z0-9_.-]+", "_", interval)
    return _cache_root() / f"{safe_symbol}_{safe_interval}.csv"


def _normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(column[0]).lower() for column in frame.columns]
    else:
        frame = frame.rename(columns=str.lower)

    available = [column for column in ["open", "high", "low", "close", "volume"] if column in frame.columns]
    if "close" not in available:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    frame = frame[available].copy()
    for column in ["open", "high", "low", "close", "volume"]:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame.index = pd.to_datetime(frame.index, errors="coerce")
    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_convert(None)
    frame = frame[~frame.index.isna()]
    frame = frame[["open", "high", "low", "close", "volume"]]
    frame = frame.dropna(subset=["close"]).sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def _read_cached_ohlcv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    try:
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
        return _normalize_ohlcv(cached)
    except Exception:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def _write_cached_ohlcv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index_label="date")


def _trim_lookback(frame: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    cutoff = pd.Timestamp(datetime.utcnow() - timedelta(days=max(lookback_days * 2, 365 * 7)))
    return frame[frame.index >= cutoff]


@contextlib.contextmanager
def _quiet_yfinance_output():
    sink = io.StringIO()
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield
    finally:
        logging.disable(previous_disable_level)


def _download_ohlcv(symbol: str, fetch_start: str, interval: str) -> pd.DataFrame:
    try:
        with _quiet_yfinance_output():
            return yf.download(
                symbol,
                start=fetch_start,
                interval=interval,
                progress=False,
                threads=False,
            )
    except Exception:
        return pd.DataFrame()


def get_ohlcv(symbol: str, lookback_days: int = 200, interval="1d") -> pd.DataFrame:
    path = _cache_path(symbol, interval)
    use_cache = os.getenv("MARKET_AGENT_DISABLE_OHLCV_CACHE", "").strip().lower() not in {"1", "true", "yes"}
    cached = _read_cached_ohlcv(path) if use_cache else pd.DataFrame()

    if use_cache and not cached.empty:
        fetch_start = (cached.index.max().to_pydatetime() - timedelta(days=7)).strftime("%Y-%m-%d")
    else:
        fetch_start = (datetime.utcnow() - timedelta(days=lookback_days * 2)).strftime("%Y-%m-%d")

    downloaded = _normalize_ohlcv(_download_ohlcv(symbol, fetch_start, interval))

    if use_cache:
        if not downloaded.empty:
            if cached.empty:
                merged = downloaded.copy()
            else:
                merged = pd.concat([cached, downloaded]).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
            merged = _trim_lookback(merged, lookback_days)
            _write_cached_ohlcv(path, merged)
            return merged
        if not cached.empty:
            return _trim_lookback(cached, lookback_days)

    if downloaded.empty:
        raise ValueError(f"No OHLCV data returned for {symbol}.")
    return downloaded
