"""Session-aware observed heatmap prices, independent of forecast data and model caches."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
import logging

import numpy as np
import pandas as pd
import yfinance as yf

from curl_cffi.requests.exceptions import RequestException
from yfinance.exceptions import YFException

from market_agent.agent.earnings import us_equity_trading_sessions
from market_agent.heatmaps import heatmap_returns

EASTERN = ZoneInfo("America/New_York")
MAX_INTRADAY_AGE = timedelta(minutes=20)


def close_price_frame(raw: pd.DataFrame, tickers: tuple[str, ...]) -> pd.DataFrame:
    """Extract per-ticker closes from either yfinance column layout."""
    ticker_list = tuple(tickers)
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw, pd.Series):
        close = raw.to_frame(name=ticker_list[0] if ticker_list else raw.name)
    elif isinstance(raw.columns, pd.MultiIndex):
        column_levels = raw.columns
        if "Close" in column_levels.get_level_values(0):
            close = raw["Close"]
        elif "Close" in column_levels.get_level_values(column_levels.nlevels - 1):
            close = raw.xs("Close", axis=1, level=column_levels.nlevels - 1)
        else:
            return pd.DataFrame()
    elif "Close" in raw.columns:
        close = raw[["Close"]].copy()
        if len(ticker_list) == 1:
            close.columns = [ticker_list[0]]
    else:
        close = raw.copy()

    if isinstance(close, pd.Series):
        close = close.to_frame(name=ticker_list[0] if ticker_list else close.name)

    close = close.copy()
    close.columns = [str(column[-1] if isinstance(column, tuple) else column) for column in close.columns]
    requested_columns = [ticker for ticker in ticker_list if ticker in close.columns]
    if requested_columns:
        close = close[requested_columns]

    close = close.apply(pd.to_numeric, errors="coerce")
    close = close.dropna(axis=1, how="all").dropna(how="all")
    return close.loc[:, ~close.columns.duplicated()]


def _download_closes(tickers: tuple[str, ...], **options: object) -> pd.DataFrame:
    raw = yf.download(
        list(tickers), auto_adjust=False, prepost=False, progress=False,
        threads=4, timeout=8, **options,
    )
    return close_price_frame(raw, tickers)


def download_daily_closes(tickers: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    """Read split-adjusted, dividend-unadjusted Yahoo daily closing prices; end is exclusive."""
    return _download_closes(tickers, interval="1d", start=start.isoformat(), end=end.isoformat(), ignore_tz=True)


def download_crypto_day_close(tickers: tuple[str, ...], session: date) -> pd.Series:
    """Read the final UTC minute's closing trade when the completed crypto daily bar is missing."""
    midnight = datetime.combine(session, time.min, tzinfo=timezone.utc)
    last_minute = midnight - timedelta(minutes=1)
    # Include the next day to avoid losing the terminal interval at the provider's exclusive end.
    bars = _download_closes(tickers, interval="1m", start=last_minute - timedelta(minutes=1),
                           end=midnight + timedelta(minutes=1), ignore_tz=False)
    if last_minute in bars.index:
        return pd.to_numeric(bars.loc[last_minute].reindex(tickers), errors="coerce")
    return pd.Series(np.nan, index=list(tickers))


def download_regular_quotes(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Read Yahoo's regular-market last price and its provider timestamp for each symbol."""
    def fetch(symbol: str) -> tuple[str, float, pd.Timestamp]:
        try:
            metadata = yf.Ticker(symbol).get_history_metadata()
            price = float(metadata["regularMarketPrice"])
            timestamp = pd.Timestamp(metadata["regularMarketTime"], unit="s", tz="UTC")
            if metadata.get("currency") == "USD" and np.isfinite(price) and price > 0:
                return symbol, price, timestamp
        except (YFException, RequestException, OSError, KeyError, TypeError, ValueError, OverflowError) as error:
            logging.getLogger(__name__).warning("Heatmap quote unavailable for %s (%s)", symbol, type(error).__name__)
        return symbol, np.nan, pd.NaT

    with ThreadPoolExecutor(max_workers=4) as pool:
        return pd.DataFrame(pool.map(fetch, tickers), columns=["Ticker", "Price", "Observed"]).set_index("Ticker")


def _session_close(session: date) -> datetime:
    # Regular NYSE early closes on trading days: July 3, Christmas Eve and Black Friday.
    early = (session.month, session.day) in ((7, 3), (12, 24)) or (
        session.month == 11 and session.weekday() == 4 and 23 <= session.day <= 29
    )
    return datetime.combine(session, time(13) if early else time(16), tzinfo=EASTERN)


def _active_session(now: datetime, crypto: bool) -> tuple[date, bool]:
    if crypto:
        return now.astimezone(timezone.utc).date(), True
    local = now.astimezone(EASTERN)
    today = local.date()
    sessions = us_equity_trading_sessions(as_of=now, lookback_days=14, future_session_count=1)
    if today not in sessions or local.time() < time(9, 30):
        return max(session for session in sessions if session < today), False
    return today, local < _session_close(today)


def fetch_heatmap_prices(
    tickers: tuple[str, ...], lookback_days: int = 1, *, crypto: bool = False,
    now: datetime | None = None,
    read_daily: Callable[[tuple[str, ...], date, date], pd.DataFrame] = download_daily_closes,
    read_quotes: Callable[[tuple[str, ...]], pd.DataFrame] = download_regular_quotes,
    read_crypto_close: Callable[[tuple[str, ...], date], pd.Series] = download_crypto_day_close,
) -> pd.DataFrame:
    """Compare current observations with the closing price at the selected calendar horizon.

    During stock trading (and all day for crypto), require a timestamped quote no
    older than 20 minutes. Outside the regular session, use the expected daily
    close or, while that is pending, its timestamped regular-market closing
    quote. A missing endpoint cannot shift the comparison to a prior session. Inject the readers to cache provider calls or replay data.
    """
    if not tickers or len(set(tickers)) != len(tickers):
        raise ValueError("Heatmap prices require at least one symbol and no duplicates.")
    checked = now or datetime.now(timezone.utc)
    if checked.tzinfo is None or checked.utcoffset() is None:
        raise ValueError("Heatmap observation time must include a timezone.")
    if lookback_days < 1:
        raise ValueError("The return horizon must be positive.")
    checked = checked.astimezone(timezone.utc)
    session, intraday = _active_session(checked, crypto)
    target = pd.Timestamp(session)
    start = session - timedelta(days=lookback_days + 14)
    # During trading only completed daily bars can supply the comparison price.
    end = session if intraday else session + timedelta(days=1)
    prices = read_daily(tickers, start, end).reindex(columns=tickers).copy()
    dates = pd.DatetimeIndex(prices.index)
    prices.index = dates.tz_localize(None).normalize()
    if prices.index.hasnans or prices.index.has_duplicates:
        raise ValueError("Heatmap history requires distinct, dated daily prices.")
    prices = prices.loc[prices.index <= target]
    if crypto and lookback_days == 1:
        baseline = target - pd.Timedelta(days=1)
        previous = prices.reindex([baseline]).iloc[0]
        unavailable = tuple(previous.index[~(np.isfinite(previous) & (previous > 0))])
        if unavailable:
            boundary = read_crypto_close(unavailable, session)
            prices.loc[baseline, list(unavailable)] = boundary.reindex(unavailable).to_numpy()
    if target not in prices.index or intraday:
        prices.loc[target] = np.nan
    stamps = pd.Series("", index=list(tickers), dtype=object)

    last = pd.to_numeric(prices.loc[target], errors="coerce")
    valid = np.isfinite(last) & (last > 0)
    stamps.loc[valid] = f"{session} close"
    needed = tuple(last.index[~valid])
    if needed:
        quotes = read_quotes(needed).reindex(needed)
        if now is None:
            checked = datetime.now(timezone.utc)
        exchange_tz = timezone.utc if crypto else EASTERN
        opened = datetime.combine(session, time.min if crypto else time(9, 30), tzinfo=exchange_tz)
        # After the bell, allow the regular-session auction print to arrive a few minutes later.
        close = checked if intraday else _session_close(session)
        earliest = max(opened, close - MAX_INTRADAY_AGE) if intraday else close
        latest = checked if intraday else min(checked, close + timedelta(minutes=15))
        for symbol, row in quotes.iterrows():
            stamp, value = row["Observed"], row["Price"]
            if pd.isna(stamp) or stamp.tzinfo is None or not earliest <= stamp <= latest:
                continue
            if stamp.tz_convert(exchange_tz).date() != session:
                continue
            if pd.notna(value) and np.isfinite(value) and value > 0:
                prices.loc[target, symbol] = value
                stamps[symbol] = stamp.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S UTC")

    data = heatmap_returns(prices, tickers, lookback_days, crypto=crypto)
    data["Price as of"] = data["Ticker"].map(stamps)
    data.attrs.update(
        intraday=intraday, crypto=crypto, checked_at=checked.isoformat(),
        missing_price_count=int((stamps == "").sum()),
        quote_count=int(stamps.str.endswith(" UTC").sum()),
    )
    return data
