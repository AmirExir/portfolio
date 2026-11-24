"""Provider-reported USD market capitalization for the dashboard treemaps."""

from __future__ import annotations

import logging
from math import isfinite

import requests
from curl_cffi.requests.exceptions import RequestException as CurlRequestError
from yfinance.const import _QUERY1_URL_
from yfinance.data import YfData
from yfinance.exceptions import YFException

from market_agent.nasdaq_data import get_nasdaq_data

_LOG = logging.getLogger(__name__)


def _positive_cap(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        cap = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return cap if isfinite(cap) and cap > 0 else None


def _provider_rows(group: object, field: str) -> list[dict]:
    rows = group.get(field) if isinstance(group, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"Missing market-cap rows in {field}")
    return [row for row in rows if isinstance(row, dict)]


def download_market_caps(tickers: tuple[str, ...]) -> dict[str, tuple[float, str]]:
    """Get USD caps by exact symbol, with the provider that reported each value.

    Use Yahoo's batch quote field for equities and cryptocurrencies. Fill missing
    U.S. stock caps from the Nasdaq stock screener snapshot.
    """
    if not tickers:
        return {}
    requested = set(tickers)
    stock_candidates = {symbol for symbol in requested if not symbol.endswith("-USD")}
    caps: dict[str, tuple[float, str]] = {}
    try:
        # Use the same cookie/crumb session and USD quote field as yfinance Ticker.info.
        payload = YfData().get_raw_json(
            f"{_QUERY1_URL_}/v7/finance/quote",
            params={"symbols": ",".join(tickers), "fields": "symbol,currency,quoteType,marketCap",
                    "formatted": "false"}, timeout=8,
        )
        quotes = payload.get("quoteResponse") if isinstance(payload, dict) else None
        for row in _provider_rows(quotes, "result"):
            symbol = row.get("symbol")
            if not isinstance(symbol, str) or symbol not in requested:
                continue
            expected = "CRYPTOCURRENCY" if symbol.endswith("-USD") else "EQUITY"
            if row.get("currency") != "USD" or row.get("quoteType") != expected:
                stock_candidates.discard(symbol)
                continue
            value = _positive_cap(row.get("marketCap"))
            if value is not None:
                caps[symbol] = (value, "Yahoo Finance")
    except (YFException, CurlRequestError, requests.RequestException, OSError, ValueError) as error:
        _LOG.warning("Yahoo Finance market-cap batch unavailable (%s)", type(error).__name__)

    missing_stocks = stock_candidates.difference(caps)
    if missing_stocks:
        try:
            screener = get_nasdaq_data("screener/stocks", {"tableonly": "true", "download": "true"})
            for row in _provider_rows(screener, "rows"):
                symbol = row.get("symbol")
                if isinstance(symbol, str) and symbol in missing_stocks:
                    value = _positive_cap(row.get("marketCap"))
                    if value is not None:
                        caps[symbol] = (value, "Nasdaq screener")
        except (requests.RequestException, ValueError) as error:
            _LOG.warning("Nasdaq stock screener market caps unavailable (%s)", type(error).__name__)
    return caps
