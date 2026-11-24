"""Circulating USD market caps for the crypto instruments in the dashboard."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import os
from sys import float_info

import requests

from market_agent.heatmap_ton_cap import download_ton_community_cap


_LOG = logging.getLogger(__name__)

# Yahoo price-series ticker -> (CoinGecko ID, CoinMarketCap ID).
# TON-USD is TON Community's Ethereum token (0x6a6c2ada3ce053561c2fbc3ee211f23d9b8c520a).
_ASSET_IDS = {
    "BTC-USD": ("bitcoin", 1),
    "ETH-USD": ("ethereum", 1027),
    "BNB-USD": ("binancecoin", 1839),
    "SOL-USD": ("solana", 5426),
    "XRP-USD": ("ripple", 52),
    "ADA-USD": ("cardano", 2010),
    "DOGE-USD": ("dogecoin", 74),
    "AVAX-USD": ("avalanche-2", 5805),
    "TON-USD": ("tontoken", 6890),
    "DOT-USD": ("polkadot", 6636),
    "ZEC-USD": ("zcash", 1437),
    "COMP5692-USD": ("compound-governance-token", 5692),
    "HYPE32196-USD": ("hyperliquid", 32196),
    "MNT27075-USD": ("mantle", 27075),
    "UNI7083-USD": ("uniswap", 7083),
    "ENA-USD": ("ethena", 30171),
}
CRYPTO_HEATMAP_TICKERS = tuple(_ASSET_IDS)


def _get_quotes(source: str, url: str, params: dict[str, str]) -> dict:
    try:
        response = requests.get(
            url, params=params,
            headers={"Accept": "application/json", "User-Agent": "Market-Agent/1.0"}, timeout=(3, 8),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Missing quote payload")
        return payload
    except (requests.RequestException, ValueError) as exc:
        _LOG.warning("%s market caps unavailable (%s)", source, type(exc).__name__)
        return {}


def _observation(value: object, stamp: object, source: str, checked: datetime) -> tuple[float, str, str] | None:
    if type(value) not in (int, float) or not 0 < value <= float_info.max:
        return None
    try:
        if type(stamp) is int:
            observed = datetime.fromtimestamp(stamp, timezone.utc)
        elif isinstance(stamp, str):
            observed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        else:
            return None
    except (ValueError, OverflowError, OSError):
        return None
    if observed.tzinfo is None or not checked - timedelta(days=1) <= observed <= checked:
        return None
    return float(value), source, observed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _coinmarketcap_caps(missing: dict[int, str]) -> dict[str, tuple[float, str, str]]:
    base = os.environ.get("COINMARKETCAP_API_BASE_URL", "https://pro-api.coinmarketcap.com/public-api").rstrip("/")
    params = {"id": ",".join(map(str, missing)), "convert": "USD"}
    payload = _get_quotes("CoinMarketCap", f"{base}/v3/cryptocurrency/quotes/latest", params)
    status, rows = payload.get("status"), payload.get("data")
    if not isinstance(status, dict) or status.get("error_code") not in (0, "0") or not isinstance(rows, list):
        return {}

    checked = datetime.now(timezone.utc)
    caps = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        asset_id, quotes = row.get("id"), row.get("quote")
        if type(asset_id) is not int or asset_id not in missing or not isinstance(quotes, list):
            continue
        usd = [q for q in quotes if isinstance(q, dict) and q.get("symbol") == "USD"]
        if len(usd) == 1:
            cap = _observation(usd[0].get("market_cap"), usd[0].get("last_updated"), "CoinMarketCap", checked)
            if cap:
                caps[missing[asset_id]] = cap
    return caps


def download_crypto_market_caps(tickers: tuple[str, ...]) -> dict[str, tuple[float, str, str]]:
    """Read USD circulating caps, their source and UTC observation time by exact ticker.

    Both batch providers must report a positive circulating market cap from the
    past day. TON Community also has a contract-specific on-chain fallback.
    """
    requested = {_ASSET_IDS[t][0]: t for t in tickers if t in _ASSET_IDS}
    caps = {}
    if not requested:
        return caps
    base = os.environ.get("COINGECKO_API_BASE_URL", "https://api.coingecko.com/api/v3").rstrip("/")
    params = {
        "ids": ",".join(requested), "vs_currencies": "usd",
        "include_market_cap": "true", "include_last_updated_at": "true",
    }
    payload = _get_quotes("CoinGecko", f"{base}/simple/price", params)
    checked = datetime.now(timezone.utc)
    for asset_id, ticker in requested.items():
        row = payload.get(asset_id)
        if isinstance(row, dict):
            cap = _observation(row.get("usd_market_cap"), row.get("last_updated_at"), "CoinGecko", checked)
            if cap:
                caps[ticker] = cap

    missing = {_ASSET_IDS[t][1]: t for t in requested.values() if t not in caps}
    if missing:
        caps.update(_coinmarketcap_caps(missing))
    if "TON-USD" in requested.values() and "TON-USD" not in caps:
        ton = download_ton_community_cap()
        if ton:
            caps["TON-USD"] = ton
    return caps
