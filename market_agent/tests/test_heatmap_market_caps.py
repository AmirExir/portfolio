"""Market-cap source selection and unit checks for heatmap tile areas."""

import unittest
from unittest.mock import Mock, patch

import requests
import pandas as pd

from market_agent.heatmap_market_caps import attach_heatmap_caps, download_market_caps


class HeatmapMarketCapTests(unittest.TestCase):
    def test_fund_size_uses_the_listed_price_and_a_share_count_available_by_the_session(self) -> None:
        quotes = pd.DataFrame({"Ticker": ["STOCK", "FUND", "FUTURE", "NO_PRICE"],
                               "Price (USD)": [10., 25., 32., float("nan")],
                               "Percent Change": [2., 1., 3., float("nan")]})
        quotes.attrs["end"] = "2026-11-23"
        shares = Mock(return_value={"FUND": (100., "Issuer", "2026-11-23"),
                                    "FUTURE": (10., "Issuer", "2026-11-24"),
                                    "NO_PRICE": (50., "Issuer", "2026-11-23")})
        stock_caps = Mock(return_value={"STOCK": (950., "Yahoo Finance")})
        crypto_caps = Mock()
        sized = attach_heatmap_caps(
            quotes, crypto=False, fund_listings={"FUND", "FUTURE", "NO_PRICE"},
            read_caps=stock_caps, read_fund_shares=shares, read_crypto_caps=crypto_caps,
        ).set_index("Ticker")
        stock_caps.assert_called_once_with(("STOCK",))
        crypto_caps.assert_not_called()
        self.assertEqual(sized.loc["FUND", "Market Cap"], 2500.)
        self.assertEqual(sized.loc["STOCK", "Market Cap"], 950.)
        self.assertTrue(sized.loc[["FUTURE", "NO_PRICE"], "Market Cap"].isna().all())

    def test_crypto_uses_reported_caps_only_for_coins_missing_from_the_dated_snapshot(self) -> None:
        quotes = pd.DataFrame({"Ticker": ["BTC-USD", "ETH-USD"], "Percent Change": [3., -2.]})
        fallback = Mock(return_value={"ETH-USD": (100., "Yahoo Finance")})
        sized = attach_heatmap_caps(
            quotes, crypto=True, fund_listings=(), read_caps=fallback, read_fund_shares=Mock(),
            read_crypto_caps=lambda _: {"BTC-USD": (450., "CoinGecko", "2026-11-23 12:30:00 UTC")},
        ).set_index("Ticker")
        fallback.assert_called_once_with(("ETH-USD",))
        self.assertEqual(sized.loc["BTC-USD", "Cap source"], "CoinGecko")
        self.assertTrue(pd.isna(sized.loc["ETH-USD", "Size input as of"]))

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
