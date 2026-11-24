"""Coverage for ID-matched circulating crypto market caps."""

from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import Mock, patch

import requests

from market_agent.heatmap_crypto_caps import download_crypto_market_caps


def _response(payload: object) -> Mock:
    response = Mock()
    response.json.return_value = payload
    return response


class CryptoCapTests(unittest.TestCase):
    @patch("market_agent.heatmap_crypto_caps.requests.get")
    def test_one_batch_matches_the_selected_instrument_and_records_the_observation(self, get: Mock) -> None:
        observed = datetime.now(timezone.utc) - timedelta(minutes=1)
        stamp = int(observed.timestamp())
        get.return_value = _response({
            "tontoken": {"usd_market_cap": 130, "last_updated_at": stamp},
            "the-open-network": {"usd_market_cap": 9_000, "last_updated_at": stamp},
            "compound-governance-token": {"usd_market_cap": 500, "last_updated_at": stamp},
        })
        result = download_crypto_market_caps(("TON-USD", "COMP5692-USD", "TON-USD", "UNKNOWN-USD"))
        source = ("CoinGecko", observed.strftime("%Y-%m-%d %H:%M:%S UTC"))
        self.assertEqual(result, {"TON-USD": (130, *source), "COMP5692-USD": (500, *source)})
        get.assert_called_once()
        self.assertTrue(get.call_args.args[0].endswith("/simple/price"))
        self.assertEqual(get.call_args.kwargs["params"], {
            "ids": "tontoken,compound-governance-token", "vs_currencies": "usd",
            "include_market_cap": "true", "include_last_updated_at": "true",
        })

    @patch("market_agent.heatmap_crypto_caps.requests.get")
    def test_missing_invalid_or_stale_cap_never_uses_fully_diluted_value(self, get: Mock) -> None:
        now = datetime.now(timezone.utc)
        row = {"fully_diluted_valuation": 11, "last_updated_at": int(now.timestamp()), "usd_market_cap": 10}
        for override in (
            {"usd_market_cap": None}, {"usd_market_cap": False}, {"usd_market_cap": float("inf")},
            {"usd_market_cap": 0}, {"usd_market_cap": 10 ** 500}, {"usd_market_cap": float("nan")},
            {"last_updated_at": None}, {"last_updated_at": "unknown"}, {"last_updated_at": 10 ** 500},
            {"last_updated_at": now.replace(tzinfo=None).isoformat()},
            {"last_updated_at": int((now + timedelta(hours=1)).timestamp())},
            {"last_updated_at": int((now - timedelta(days=2)).timestamp())},
        ):
            with self.subTest(override=override):
                get.side_effect = [_response({"bitcoin": {**row, **override}}), _response({})]
                self.assertEqual(download_crypto_market_caps(("BTC-USD",)), {})

    @patch("market_agent.heatmap_crypto_caps.requests.get")
    def test_backup_only_requests_missing_ids_and_reads_the_dated_usd_quote(self, get: Mock) -> None:
        now = datetime.now(timezone.utc) - timedelta(minutes=1)
        other_zone = now.astimezone(timezone(timedelta(hours=3))).isoformat()
        quote = {"symbol": "USD", "last_updated": other_zone, "market_cap": 50}
        primary = {"bitcoin": {"usd_market_cap": 777, "last_updated_at": int(now.timestamp())}}
        backup = {"status": {"error_code": "0"}, "data": [
            {"id": 1, "quote": [quote]},
            {"id": 5692, "quote": [{**quote, "symbol": "EUR", "market_cap": 90}, quote]},
            {"id": 32196, "quote": [{**quote, "last_updated": (now - timedelta(days=2)).isoformat()}]},
            {"id": 30171, "quote": [quote, quote]},
        ]}
        get.side_effect = [_response(primary), _response(backup)]
        result = download_crypto_market_caps(("BTC-USD", "COMP5692-USD", "HYPE32196-USD", "ENA-USD"))
        stamp = now.strftime("%Y-%m-%d %H:%M:%S UTC")
        self.assertEqual(result, {"BTC-USD": (777, "CoinGecko", stamp), "COMP5692-USD": (50, "CoinMarketCap", stamp)})
        self.assertEqual(get.call_count, 2)
        self.assertTrue(get.call_args.args[0].endswith("/v3/cryptocurrency/quotes/latest"))
        self.assertEqual(get.call_args.kwargs["params"], {"id": "5692,32196,30171", "convert": "USD"})

    @patch("market_agent.heatmap_crypto_caps.download_ton_community_cap")
    @patch("market_agent.heatmap_crypto_caps.requests.get")
    def test_zero_verified_ton_cap_ignores_self_reported_and_other_ton_assets(self, get: Mock, ton: Mock) -> None:
        now = datetime.now(timezone.utc).isoformat()
        quote = {"symbol": "USD", "last_updated": now, "market_cap": 0, "fully_diluted_market_cap": 400}
        backup = {"status": {"error_code": 0}, "data": [
            {"id": 6890, "symbol": "TON", "quote": [quote], "self_reported_market_cap": 900},
            {"id": 11419, "symbol": "TON", "quote": [{**quote, "market_cap": 800}]},
        ]}
        for contract_cap in (None, (110, "GeckoTerminal", now)):
            with self.subTest(contract_cap=contract_cap):
                ton.return_value = contract_cap
                get.side_effect = [requests.Timeout(), _response(backup)]
                self.assertEqual(download_crypto_market_caps(("TON-USD",)), {"TON-USD": contract_cap} if contract_cap else {})
        self.assertEqual(ton.call_count, 2)

    @patch("market_agent.heatmap_crypto_caps.download_ton_community_cap")
    @patch("market_agent.heatmap_crypto_caps.requests.get")
    def test_valid_batch_ton_precedes_contract_lookup(self, get: Mock, ton: Mock) -> None:
        now = datetime.now(timezone.utc)
        quote = {"symbol": "USD", "last_updated": now.isoformat(), "market_cap": 50}
        get.side_effect = [requests.Timeout(), _response({
            "status": {"error_code": "0"}, "data": [{"id": 6890, "quote": [quote]}],
        })]
        self.assertEqual(download_crypto_market_caps(("TON-USD",)), {
            "TON-USD": (50, "CoinMarketCap", now.strftime("%Y-%m-%d %H:%M:%S UTC")),
        })
        ton.assert_not_called()

    @patch("market_agent.heatmap_crypto_caps.requests.get")
    def test_unknown_assets_and_unavailable_provider(self, get: Mock) -> None:
        self.assertEqual(download_crypto_market_caps(("NOT-USD",)), {})
        get.assert_not_called()
        get.side_effect = requests.Timeout()
        self.assertEqual(download_crypto_market_caps(("BTC-USD",)), {})
