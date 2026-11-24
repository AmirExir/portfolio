"""Circulating USD market caps for the crypto instruments in the dashboard."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import os
from sys import float_info

import requests


_LOG = logging.getLogger(__name__)

# Yahoo identifiers identify the price series. CoinGecko IDs identify the cap.
# TON-USD is TON Community's Ethereum token (0x6a6c2ada3ce053561c2fbc3ee211f23d9b8c520a).
_COINGECKO_IDS = {
    "BTC-USD": "bitcoin", "ETH-USD": "ethereum", "BNB-USD": "binancecoin",
    "SOL-USD": "solana", "XRP-USD": "ripple", "ADA-USD": "cardano",
    "DOGE-USD": "dogecoin", "AVAX-USD": "avalanche-2", "TON-USD": "tontoken",
    "DOT-USD": "polkadot", "ZEC-USD": "zcash", "COMP5692-USD": "compound-governance-token",
    "HYPE32196-USD": "hyperliquid", "MNT27075-USD": "mantle", "UNI7083-USD": "uniswap",
    "ENA-USD": "ethena",
}
CRYPTO_HEATMAP_TICKERS = tuple(_COINGECKO_IDS)


def download_crypto_market_caps(tickers: tuple[str, ...]) -> dict[str, tuple[float, str, str]]:
    """Read one CoinGecko USD markets snapshot, returning cap, source and UTC observation per ticker.

    Crypto trades continuously. Accept positive circulating market caps from the
    last 24 hours, and keep fully diluted valuations out of tile sizing.
    """
    requested = {_COINGECKO_IDS[ticker]: ticker for ticker in tickers if ticker in _COINGECKO_IDS}
    if not requested:
        return {}
    base = os.environ.get("COINGECKO_API_BASE_URL", "https://api.coingecko.com/api/v3").rstrip("/")
    try:
        response = requests.get(
            f"{base}/coins/markets",
            params={"ids": ",".join(requested), "vs_currency": "usd"},
            headers={"Accept": "application/json", "User-Agent": "Market-Agent/1.0"}, timeout=(3, 8),
        )
        response.raise_for_status()
        rows = response.json()
        if not isinstance(rows, list):
            raise ValueError("Missing CoinGecko markets list")
    except (requests.RequestException, ValueError) as exc:
        _LOG.warning("CoinGecko market caps unavailable (%s)", type(exc).__name__)
        return {}

    caps = {}
    checked = datetime.now(timezone.utc)
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            continue
        ticker = requested.get(row["id"])
        value = row.get("market_cap")
        if ticker is None or type(value) not in (int, float) or not 0 < value <= float_info.max:
            continue
        stamp = row.get("last_updated")
        if not isinstance(stamp, str):
            continue
        try:
            observed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError:
            continue
        if observed.tzinfo is None or not checked - timedelta(days=1) <= observed <= checked:
            continue
        as_of = observed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        caps[ticker] = (float(value), "CoinGecko", as_of)
    return caps
