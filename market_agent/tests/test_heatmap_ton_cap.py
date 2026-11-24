"""Regressions for the on-chain TON Community fallback."""

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import unittest
from unittest.mock import Mock, patch

import requests

from market_agent.heatmap_ton_cap import download_ton_community_cap


CONTRACT = "0x6a6c2ada3ce053561c2fbc3ee211f23d9b8c520a"
TOKEN_ID = f"eth_{CONTRACT}"


class TonCapTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch("market_agent.heatmap_ton_cap.requests.get")
        self.get = patcher.start()
        self.addCleanup(patcher.stop)
        stamp = datetime.now(timezone.utc)
        self.response = Mock(headers={"Date": format_datetime(stamp, usegmt=True), "Age": "8"})
        self.token = {"type": "token", "id": TOKEN_ID, "attributes": {
            "address": CONTRACT, "coingecko_coin_id": "tontoken", "market_cap_usd": "430000", "fdv_usd": "650000",
        }, "relationships": {"top_pools": {"data": [{"id": "eth_pool"}]}}}
        self.pool = {"type": "pool", "id": "eth_pool", "relationships": {
            "base_token": {"data": {"id": TOKEN_ID}}, "quote_token": {"data": {"id": "eth_other"}},
        }, "attributes": {"transactions": {"h24": {"buys": 1, "sells": 0}}}}
        self.response.json.return_value = {"data": self.token, "included": [self.pool]}
        self.get.return_value = self.response

    def test_verified_cap_from_matching_active_pool_and_retrieval_label(self) -> None:
        cap, source, stamp = download_ton_community_cap()
        self.assertEqual((cap, source), (430000, "GeckoTerminal"))
        self.assertRegex(stamp, r"^Retrieved \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC$")
        self.assertTrue(self.get.call_args.args[0].endswith("/networks/eth/tokens/" + CONTRACT))
        self.assertEqual(self.get.call_args.kwargs["params"]["include_inactive_source"], "false")

    def test_null_cap_and_invalid_sizes_never_fall_back_to_fdv(self) -> None:
        for cap in (None, 0, "nan", False, "inf"):
            with self.subTest(cap=cap):
                self.token["attributes"]["market_cap_usd"] = cap
                self.assertIsNone(download_ton_community_cap())

    def test_other_asset_and_inactive_or_mismatched_pool(self) -> None:
        self.token["attributes"]["coingecko_coin_id"] = "the-open-network"
        self.assertIsNone(download_ton_community_cap())
        self.token["attributes"]["coingecko_coin_id"] = "tontoken"
        self.pool["attributes"]["transactions"]["h24"] = {"buys": 0, "sells": 0}
        self.assertIsNone(download_ton_community_cap())
        self.pool["attributes"]["transactions"]["h24"]["sells"] = 2
        self.pool["relationships"]["base_token"]["data"]["id"] = "eth_other"
        self.assertIsNone(download_ton_community_cap())

    def test_reject_old_missing_or_invalid_responses(self) -> None:
        old = datetime.now(timezone.utc) - timedelta(minutes=6)
        for headers in ({}, {"Date": format_datetime(old)}, {**self.response.headers, "Age": "1000"}):
            with self.subTest(headers=headers):
                self.response.headers = headers
                self.assertIsNone(download_ton_community_cap())
        self.get.side_effect = requests.Timeout()
        self.assertIsNone(download_ton_community_cap())
