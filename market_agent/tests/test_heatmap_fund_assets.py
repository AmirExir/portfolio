"""Regressions for dated, issuer-specific listed ETF share counts."""

from datetime import date, timedelta
from io import BytesIO
import unittest
from unittest.mock import Mock, patch

from openpyxl import Workbook
import requests

from market_agent.heatmap_fund_assets import ETF_LISTINGS, download_etf_shares


CHECKED = date(2026, 11, 24)


def spdr_book(symbol: str, metric: str = "Shares Outstanding") -> bytes:
    book = Workbook()
    sheet = book.active
    sheet.append(("Fund Name:", "Example"))
    sheet.append(("Ticker Symbol:", symbol))
    sheet.append(())
    sheet.append(("Date", "NAV", metric, "Total Net Assets"))
    sheet.append(("23-Nov-2026", 50, 12_345, 617_250))
    data = BytesIO()
    book.save(data)
    book.close()
    return data.getvalue()


class FundShareTests(unittest.TestCase):
    @patch("market_agent.heatmap_fund_assets.requests.get")
    def test_state_street_checks_identity_units_and_uses_share_column(self, get: Mock) -> None:
        get.return_value.content = spdr_book("GLD®")
        self.assertEqual(download_etf_shares(("GLD", "GLD", "AAPL"), today=CHECKED),
                         {"GLD": (12_345, "State Street", "2026-11-23")})
        get.assert_called_once()
        self.assertTrue(get.call_args.args[0].endswith("/navhist-us-en-gld.xlsx"))
        for wrong_book in (spdr_book("GLDM"), spdr_book("GLD", "Total Net Assets"), b"bad archive"):
            with self.subTest(wrong_book_size=len(wrong_book)):
                get.return_value.content = wrong_book
                self.assertEqual(download_etf_shares(("GLD",), today=CHECKED), {})

    @patch("market_agent.heatmap_fund_assets.requests.get")
    def test_ishares_observation_can_have_duplicate_responsive_fields(self, get: Mock) -> None:
        markup = b"""<html><p data-id="fundHeader-externalTicker">SLV</p>
        <span data-id="keyFundFacts-sharesOutstanding-asOf">as of <i></i> Nov 23, 2026</span>
        <span data-id="keyFundFacts-sharesOutstanding-data">123,456</span>
        <span data-id="keyFundFacts-sharesOutstanding-data">123,456</span></html>"""
        get.return_value.content = markup
        self.assertEqual(download_etf_shares(("SLV",), today=CHECKED),
                         {"SLV": (123_456, "iShares", "2026-11-23")})
        get.return_value.content = markup.replace(b"123,456", b"987", 1)
        self.assertEqual(download_etf_shares(("SLV",), today=CHECKED), {})

    @patch("market_agent.heatmap_fund_assets.requests.get")
    def test_qqq_retries_transient_timeout_and_checks_listing(self, get: Mock) -> None:
        result = Mock()
        result.json.return_value = {"ticker": "QQQ", "currencyCode": "USD", "sharesOutstanding": 555,
                                    "effectiveDate": "2026-11-23", "shareclassTotalNetAssets": 999_000}
        get.side_effect = [requests.Timeout(), result]
        self.assertEqual(download_etf_shares(("QQQ",), today=CHECKED),
                         {"QQQ": (555, "Invesco", "2026-11-23")})
        self.assertEqual(get.call_count, 2)
        get.side_effect = None
        get.return_value = result
        result.json.return_value["ticker"] = "QQQM"
        self.assertEqual(download_etf_shares(("QQQ",), today=CHECKED), {})

    @patch("market_agent.heatmap_fund_assets.requests.get")
    def test_uso_uses_latest_filed_share_observation_known_as_of_check(self, get: Mock) -> None:
        data = {"cik": 1327068, "taxonomy": "dei", "tag": "EntityCommonStockSharesOutstanding", "units": {
            "shares": [
                {"val": 222, "end": "2026-11-03", "filed": "2026-11-09", "form": "10-Q"},
                {"val": 333, "end": "2026-11-21", "filed": "2026-12-03", "form": "10-Q"},
                {"val": 111, "end": "2026-08-03", "filed": "2026-08-07", "form": "10-Q"},
            ],
        }}
        get.return_value.json.return_value = data
        self.assertEqual(download_etf_shares(("USO",), today=CHECKED),
                         {"USO": (222, "USCF (SEC filing)", "2026-11-03")})
        data["cik"] = 1
        self.assertEqual(download_etf_shares(("USO",), today=CHECKED), {})

    @patch("market_agent.heatmap_fund_assets._spdr_shares")
    @patch("market_agent.heatmap_fund_assets.download_vanguard_etf_shares")
    def test_share_counts_expire_by_publication_schedule_and_fail_independently(self, vanguard: Mock, spdr: Mock) -> None:
        vanguard.return_value = (1_500, "Vanguard", "2026-10-31")
        spdr.return_value = (2_000, "State Street", "2026-10-31")
        self.assertEqual(download_etf_shares(("VOO", "SPY"), today=CHECKED),
                         {"VOO": (1_500, "Vanguard", "2026-10-31")})
        for stamp, shares in (("2026-12-01", 10), ((CHECKED - timedelta(days=63)).isoformat(), 10),
                              ("2026-11-23", 0), ("2026-11-23", float("inf")), ("2026-11-23", True)):
            with self.subTest(stamp=stamp, shares=shares):
                vanguard.return_value = (shares, "Vanguard", stamp)
                self.assertEqual(download_etf_shares(("VOO",), today=CHECKED), {})

    @patch("market_agent.heatmap_fund_assets.requests.get")
    def test_only_configured_listings_trigger_lookups(self, get: Mock) -> None:
        self.assertEqual(ETF_LISTINGS, set("SPY VOO GLD SLV USO IWM TLT DIA XLK XLF XLE XLV XLY QQQ".split()))
        self.assertEqual(download_etf_shares(("SNDK", "BTC-USD", "VFIFX")), {})
        get.assert_not_called()
