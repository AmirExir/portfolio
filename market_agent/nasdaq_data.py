"""Transport for the public Nasdaq market-data endpoints used by heatmaps."""

from __future__ import annotations

import os

import requests


def get_nasdaq_data(path: str, params: dict[str, str | int]) -> dict:
    """Read a Nasdaq API data object and reject HTTP or provider errors."""
    base = os.environ.get("NASDAQ_API_BASE_URL", "https://api.nasdaq.com/api").rstrip("/")
    response = requests.get(
        f"{base}/{path.lstrip('/')}", params=params,
        headers={"User-Agent": "Market-Agent/1.0", "Accept": "application/json"}, timeout=(3, 8),
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        status, data = payload.get("status"), payload.get("data")
        if isinstance(status, dict) and status.get("rCode") == 200 and isinstance(data, dict):
            return data
    raise ValueError("Nasdaq returned an unsuccessful or malformed market-data response")
