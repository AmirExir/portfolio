"""Regression coverage for the Vanguard ETF share-class size observation."""

import unittest
from unittest.mock import Mock, patch

import requests

from market_agent.vanguard_etf_shares import download_vanguard_etf_shares


class VanguardShareTests(unittest.TestCase):
    @patch("market_agent.vanguard_etf_shares.requests.get")
    def test_voo_uses_its_issuer_count_and_effective_date(self, get: Mock) -> None:
        get.return_value.json.return_value = {
            "effectiveDate": "2026-10-31", "outstandingShares": 12345.75, "assetsUnderManagement": 111111110,
        }
        self.assertEqual(download_vanguard_etf_shares("VOO"), (12345.75, "Vanguard", "2026-10-31"))
        self.assertTrue(get.call_args.args[0].endswith("/funds/0968/pricing/outstanding-shares"))

    @patch("market_agent.vanguard_etf_shares.requests.get")
    def test_other_classes_invalid_data_and_issuer_outages(self, get: Mock) -> None:
        self.assertIsNone(download_vanguard_etf_shares("VFIAX"))
        get.assert_not_called()
        for shares, stamp in [(0, "2026-10-31"), (True, "2026-10-31"), (10, None), (10, "bad date")]:
            with self.subTest(shares=shares, stamp=stamp):
                get.return_value.json.return_value = {"outstandingShares": shares, "effectiveDate": stamp}
                self.assertIsNone(download_vanguard_etf_shares("VOO"))
        get.side_effect = requests.Timeout()
        self.assertIsNone(download_vanguard_etf_shares("VOO"))
