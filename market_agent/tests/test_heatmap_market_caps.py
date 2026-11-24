"""Market-cap source selection and unit checks for heatmap tile areas."""

import unittest
from unittest.mock import Mock, patch

import requests

from market_agent.heatmap_market_caps import download_market_caps


class HeatmapMarketCapTests(unittest.TestCase):
    @patch("market_agent.nasdaq_data.requests.get")
    @patch("market_agent.heatmap_market_caps.YfData")
    def test_provider_caps_and_missing_stock_snapshot(self, yf_data: Mock, nasdaq: Mock) -> None:
        example = {"quoteType": "EQUITY", "currency": "USD", "marketCap": 125.0}
        yf_data.return_value.get_raw_json.return_value = {"quoteResponse": {"result": [
            {**example, "symbol": "SNDK"}, {**example, "symbol": "OTHER"},
            {**example, "symbol": "BTC-USD", "quoteType": "CRYPTOCURRENCY", "marketCap": 25},
            {**example, "symbol": "COIN-USD", "quoteType": "CRYPTOCURRENCY", "marketCap": 0},
            {**example, "symbol": "ETF", "quoteType": "ETF", "totalAssets": 500},
            {**example, "symbol": "FOREIGN", "currency": "CAD"},
            {**example, "symbol": "AAPL", "marketCap": float("nan")},
        ]}}
        nasdaq.return_value.json.return_value = {"status": {"rCode": 200}, "data": {"rows": [
            {"symbol": "AAPL", "marketCap": "300.5"}, {"symbol": "SNDK", "marketCap": "140"},
            {"symbol": "FOREIGN", "marketCap": "6"}, {"symbol": "ETF", "marketCap": "42"},
            {"symbol": "COIN-USD", "marketCap": "90"}, {"symbol": "BAD", "marketCap": "N/A"},
        ]}}
        requested = ("SNDK", "AAPL", "BTC-USD", "COIN-USD", "FOREIGN", "ETF", "BAD")
        self.assertEqual(download_market_caps(requested), {
            "SNDK": (125, "Yahoo Finance"), "BTC-USD": (25, "Yahoo Finance"),
            "AAPL": (300.5, "Nasdaq screener"),
        })
        self.assertEqual(yf_data.return_value.get_raw_json.call_args.kwargs["params"]["symbols"], ",".join(requested))
        nasdaq.assert_called_once()

    @patch("market_agent.nasdaq_data.requests.get")
    @patch("market_agent.heatmap_market_caps.YfData")
    def test_no_cross_asset_substitution_and_provider_errors(self, yf_data: Mock, nasdaq: Mock) -> None:
        self.assertEqual(download_market_caps(()), {})
        yf_data.assert_not_called()
        yf_data.return_value.get_raw_json.side_effect = ValueError("Malformed quote response")
        self.assertEqual(download_market_caps(("BTC-USD",)), {})
        nasdaq.assert_not_called()
        nasdaq.return_value.json.return_value = {"status": {"rCode": 400}, "data": {"rows": [
            {"symbol": "SNDK", "marketCap": "200"},
        ]}}
        self.assertEqual(download_market_caps(("SNDK",)), {})
        nasdaq.side_effect = requests.Timeout()
        self.assertEqual(download_market_caps(("SNDK",)), {})
