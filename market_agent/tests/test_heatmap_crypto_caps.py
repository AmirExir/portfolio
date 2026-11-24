"""Coverage for ID-matched circulating crypto market caps."""

from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import Mock, patch

import requests

from market_agent.heatmap_crypto_caps import download_crypto_market_caps


class CryptoCapTests(unittest.TestCase):
    @patch("market_agent.heatmap_crypto_caps.requests.get")
    def test_one_batch_matches_the_selected_instrument_and_records_the_observation(self, get: Mock) -> None:
        observed = datetime.now(timezone.utc) - timedelta(minutes=1)
        stamp = observed.astimezone(timezone(timedelta(hours=3))).isoformat()
        get.return_value.json.return_value = [
            {"id": "tontoken", "symbol": "ton", "market_cap": 130, "last_updated": stamp},
            {"id": "the-open-network", "symbol": "ton", "market_cap": 9_000, "last_updated": stamp},
            {"id": "compound-governance-token", "symbol": "comp", "market_cap": 500, "last_updated": stamp},
        ]
        result = download_crypto_market_caps(("TON-USD", "COMP5692-USD", "TON-USD", "UNKNOWN-USD"))
        source = ("CoinGecko", observed.strftime("%Y-%m-%d %H:%M:%S UTC"))
        self.assertEqual(result, {"TON-USD": (130, *source), "COMP5692-USD": (500, *source)})
        get.assert_called_once()
        self.assertEqual(get.call_args.kwargs["params"], {
            "ids": "tontoken,compound-governance-token", "vs_currency": "usd",
        })

    @patch("market_agent.heatmap_crypto_caps.requests.get")
    def test_missing_invalid_or_stale_cap_never_uses_fully_diluted_value(self, get: Mock) -> None:
        now = datetime.now(timezone.utc)
        row = {"id": "bitcoin", "fully_diluted_valuation": 11, "last_updated": now.isoformat(), "market_cap": 10}
        for override in (
            {"market_cap": None}, {"market_cap": False}, {"market_cap": float("inf")}, {"market_cap": 0},
            {"market_cap": 10 ** 500}, {"market_cap": float("nan")},
            {"last_updated": None}, {"last_updated": "unknown"}, {"last_updated": now.replace(tzinfo=None).isoformat()},
            {"last_updated": (now + timedelta(hours=1)).isoformat()},
            {"last_updated": (now - timedelta(days=2)).isoformat()},
        ):
            with self.subTest(override=override):
                get.return_value.json.return_value = [{**row, **override}]
                self.assertEqual(download_crypto_market_caps(("BTC-USD",)), {})

    @patch("market_agent.heatmap_crypto_caps.requests.get")
    def test_unknown_assets_and_unavailable_provider(self, get: Mock) -> None:
        self.assertEqual(download_crypto_market_caps(("NOT-USD",)), {})
        get.assert_not_called()
        get.side_effect = requests.Timeout()
        self.assertEqual(download_crypto_market_caps(("BTC-USD",)), {})
