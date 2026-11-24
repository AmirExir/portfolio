"""GeckoTerminal circulating market value for the TON Community Ethereum token."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import logging
import math
import os

import requests


_LOG = logging.getLogger(__name__)
_CONTRACT = "0x6a6c2ada3ce053561c2fbc3ee211f23d9b8c520a"


def download_ton_community_cap() -> tuple[float, str, str] | None:
    """Return the verified USD circulating cap and when this active-pool snapshot was retrieved.

    GeckoTerminal's public token response does not date the capitalization. Its
    documented cache lifetime is one minute. Require a recent HTTP snapshot and
    trading in the top pool within 24 hours before accepting its reported cap.
    """
    base = os.getenv("GECKOTERMINAL_API_BASE_URL", "https://api.geckoterminal.com/api/v2").rstrip("/")
    token_id = f"eth_{_CONTRACT}"
    try:
        response = requests.get(
            f"{base}/networks/eth/tokens/{_CONTRACT}",
            params={"include": "top_pools", "include_inactive_source": "false"},
            headers={"Accept": "application/json;version=20230203", "User-Agent": "Market-Agent/1.0"},
            timeout=(3, 8),
        )
        response.raise_for_status()
        retrieved = datetime.now(timezone.utc)
        generated = parsedate_to_datetime(response.headers["Date"])
        age = int(response.headers.get("Age", "0"))
        if not -timedelta(seconds=5) <= retrieved - generated <= timedelta(minutes=5) or not 0 <= age <= 300:
            raise ValueError("Stale GeckoTerminal response")

        payload = response.json()
        token = payload["data"]
        attr = token["attributes"]
        if (token["type"], token["id"], attr["address"].lower(), attr["coingecko_coin_id"]) != (
            "token", token_id, _CONTRACT, "tontoken",
        ):
            raise ValueError("Wrong Ethereum token")
        top_id = token["relationships"]["top_pools"]["data"][0]["id"]
        pool = next((p for p in payload["included"] if p["id"] == top_id and p["type"] == "pool"), None)
        if not pool or not any(pool["relationships"][side]["data"]["id"] == token_id
                               for side in ("base_token", "quote_token")):
            raise ValueError("Missing matching top pool")
        activity = pool["attributes"]["transactions"]["h24"]
        if not any(type(activity.get(side)) is int and activity[side] > 0 for side in ("buys", "sells")):
            raise ValueError("No recent trades in the top pool")

        # The token endpoint returns null for unverified capitalization; fdv_usd is a different metric.
        raw_cap = attr["market_cap_usd"]
        if type(raw_cap) not in (int, float, str):
            raise ValueError("Missing circulating market cap")
        cap = float(raw_cap)
        if not math.isfinite(cap) or cap <= 0:
            raise ValueError("Invalid circulating market cap")
        return cap, "GeckoTerminal", retrieved.strftime("Retrieved %Y-%m-%d %H:%M:%S UTC")
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError, OverflowError, AttributeError) as exc:
        _LOG.warning("TON Community on-chain market cap unavailable (%s)", type(exc).__name__)
        return None
