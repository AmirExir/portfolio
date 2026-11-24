"""Vanguard ETF share-class observations for heatmap market-value estimates."""

from __future__ import annotations

from datetime import date
import logging
import os
from sys import float_info

import requests


_LOG = logging.getLogger(__name__)


def download_vanguard_etf_shares(ticker: str) -> tuple[float, str, str] | None:
    """Read Vanguard's listed VOO share count, its issuer and its effective date."""
    if ticker != "VOO":
        return None
    base = os.environ.get(
        "VANGUARD_ADVISORS_API_BASE_URL", "https://advisors.vanguard.com/investments/products/api",
    ).rstrip("/")
    try:
        # Vanguard's VOO product profile identifies its ETF share class with portId 0968.
        response = requests.get(
            f"{base}/funds/0968/pricing/outstanding-shares",
            headers={"Accept": "application/json", "User-Agent": "Market-Agent/1.0"}, timeout=(3, 8),
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Missing Vanguard share class observation")
        shares, stamp = data.get("outstandingShares"), data.get("effectiveDate")
        if type(shares) not in (int, float) or not 0 < shares <= float_info.max or not isinstance(stamp, str):
            raise ValueError("Invalid Vanguard share class observation")
        return float(shares), "Vanguard", date.fromisoformat(stamp).isoformat()
    except (requests.RequestException, ValueError) as exc:
        _LOG.warning("Vanguard share count unavailable for %s (%s)", ticker, type(exc).__name__)
        return None
