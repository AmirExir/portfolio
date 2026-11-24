"""Point-in-time heatmap regressions covering regular sessions, closed markets and crypto."""

import unittest
from datetime import date
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from market_agent.heatmap_data import (
    download_crypto_day_close, download_equity_prior_closes, download_regular_quotes, fetch_heatmap_prices,
)


def prices(checked: str, daily: dict, quotes: dict, *, crypto: bool = False, days: int = 1) -> pd.DataFrame:
    """Reproduce a provider returning prior daily sessions plus timestamped market quotes."""
    history = pd.DataFrame(daily, dtype=float)
    history.index = pd.to_datetime(history.index)
    updates = pd.DataFrame.from_dict(quotes, orient="index", columns=["Price", "Observed"])
    updates["Observed"] = pd.to_datetime(updates["Observed"], format="ISO8601", utc=True)
    return fetch_heatmap_prices(
        tuple(daily), days, crypto=crypto, now=pd.Timestamp(checked),
        read_daily=lambda *_: history, read_quotes=lambda symbols: updates.reindex(symbols),
    ).set_index("Ticker")


class LiveHeatmapTests(unittest.TestCase):
    def test_only_recover_a_dated_prior_close_for_symbols_with_valid_current_prices(self) -> None:
        now = pd.Timestamp("2026-11-24T16:30Z")
        targets = ("SNDK", "GOOD", "NO_QUOTE")
        history = pd.DataFrame({"SNDK": [80., np.nan], "GOOD": [100., 100.], "NO_QUOTE": [70., np.nan]},
                               index=pd.to_datetime(["2026-11-20", "2026-11-23"]))
        quotes = pd.DataFrame({"Price": [78.939, 90., np.nan], "Observed": [now, now, pd.NaT]}, index=targets)
        recover = Mock(return_value=pd.Series({"SNDK": 87.71}))
        data = fetch_heatmap_prices(targets, now=now, read_daily=lambda *_: history, read_quotes=lambda *_: quotes,
                                    read_equity_close=recover).set_index("Ticker")
        recover.assert_called_once_with(("SNDK",), date(2026, 11, 23))
        self.assertEqual(data.attrs["nasdaq_close_symbols"], ("SNDK",))
        self.assertAlmostEqual(data.loc["SNDK", "Percent Change"], -10)
        self.assertEqual(data.loc["GOOD", "Percent Change"], -10)
        self.assertTrue(np.isnan(data.loc["NO_QUOTE", "Percent Change"]))
        recover.reset_mock()
        fetch_heatmap_prices(targets, 7, now=now, read_daily=lambda *_: history, read_quotes=lambda *_: quotes,
                             read_equity_close=recover)
        recover.assert_not_called()

    def test_nasdaq_fallback_requires_the_matching_symbol_date_and_positive_usd_close(self) -> None:
        baseline = date(2026, 11, 23)
        correct = {"date": "11/23/2026", "close": "$1,234.50"}
        other_day = {"date": "11/20/2026", "close": "$90.00"}
        for ticker, rows, expected in (
            ("SNDK", [other_day, correct], 1234.5),
            ("WDC", [correct], None),
            ("SNDK", [other_day], None),
            ("SNDK", [correct, correct], None),
            ("SNDK", [{**correct, "close": "€100"}], None),
            ("SNDK", [{**correct, "close": "$-1"}], None),
        ):
            with self.subTest(ticker=ticker, rows=rows):
                response = {"symbol": ticker, "tradesTable": {"rows": rows}}
                with patch("market_agent.heatmap_data.get_nasdaq_data", return_value=response) as get:
                    value = download_equity_prior_closes(("SNDK",), baseline)["SNDK"]
                self.assertEqual(get.call_args.args[1]["fromdate"], "2026-11-23")
                self.assertEqual(get.call_args.args[1]["todate"], "2026-11-24")
                if expected is None:
                    self.assertTrue(np.isnan(value))
                else:
                    self.assertEqual(value, expected)

    def test_monday_move_uses_friday_close_even_when_the_last_daily_return_was_negative(self) -> None:
        falling = {"2026-11-19": 100, "2026-11-20": 86}
        symbols = ("CURRENT", "STALE", "MISSING", "FUTURE", "PREMARKET")
        data = prices("2026-11-23T15:04Z", {s: falling for s in symbols}, {
            "CURRENT": (107.5, "2026-11-23T15:02Z"), "STALE": (120, "2026-11-23T14:43Z"),
            "FUTURE": (130, "2026-11-23T15:05Z"), "PREMARKET": (101, "2026-11-23T14:29Z"),
        })
        self.assertEqual(data.attrs["start"], "2026-11-20")
        self.assertEqual(data.attrs["end"], "2026-11-23")
        self.assertEqual(data.loc["CURRENT", "Price (USD)"], 107.5)
        self.assertAlmostEqual(data.loc["CURRENT", "Percent Change"], 25.0)
        self.assertEqual(data.loc["CURRENT", "Price as of"], "2026-11-23 15:02:00 UTC")
        self.assertTrue(data.drop("CURRENT")["Percent Change"].isna().all())

    def test_use_dated_close_or_timed_closing_quote_after_the_bell(self) -> None:
        daily = {
            "A": {"2026-11-20": 100, "2026-11-23": 105}, "B": {"2026-11-20": 100},
            "C": {"2026-11-20": 100}, "D": {"2026-11-20": 100}, "E": {"2026-11-20": 100},
        }
        data = prices("2026-11-24T02:10Z", daily, {
            "A": (207.5, "2026-11-23T23:00Z"), "B": (106, "2026-11-23T21:00:01Z"),
            "C": (95, "2026-11-20T21:00Z"), "D": (116, "2026-11-23T23:00Z"),
            "E": (104, "2026-11-23T20:59:59Z"),
        })
        self.assertEqual(data.loc["A", "Price as of"], "2026-11-23 close")
        self.assertEqual(data.loc["B", "Price (USD)"], 106)
        self.assertEqual(data.loc["B", "Percent Change"], 6)
        self.assertTrue(data.loc[["C", "D", "E"], "Percent Change"].isna().all())
        self.assertEqual(data.attrs["quote_count"], 1)

    def test_overnight_holidays_half_days_and_crypto_utc_boundaries(self) -> None:
        for checked in ("2026-11-22T18:00Z", "2026-11-23T14:29Z"):
            data = prices(checked, {"A": {"2026-11-19": 100, "2026-11-20": 86}}, {})
            self.assertFalse(data.attrs["intraday"])
            self.assertAlmostEqual(data.loc["A", "Percent Change"], -14)
        holiday = prices("2026-11-26T18:30Z", {"A": {"2026-11-24": 100, "2026-11-25": 105}}, {})
        self.assertEqual(holiday.attrs["end"], "2026-11-25")
        half = prices("2026-11-27T19:00Z", {"A": {"2026-11-25": 100}}, {"A": (102, "2026-11-27T18:00:01Z")})
        self.assertEqual(half.loc["A", "Percent Change"], 2)
        crypto = prices("2026-11-24T00:05Z", {"COIN": {"2026-11-23": 15}}, {"COIN": (18, "2026-11-24T00:04Z")}, crypto=True)
        self.assertEqual(crypto.loc["COIN", "Percent Change"], 20)
        old = prices("2026-11-24T00:05Z", {"COIN": {"2026-11-23": 15}}, {"COIN": (18, "2026-11-23T23:59Z")}, crypto=True)
        self.assertTrue(np.isnan(old.loc["COIN", "Price (USD)"]))

    def test_crypto_baseline_requires_the_exact_terminal_minute(self) -> None:
        checked = pd.Timestamp("2026-11-24T00:05Z")
        ticker = ("COIN",)
        bars = pd.DataFrame({"COIN": [100, 102]}, index=pd.to_datetime(["2026-11-23T23:58Z", "2026-11-24T00:00Z"]))
        quotes = pd.DataFrame({"Price": [110], "Observed": [checked]}, index=["COIN"])
        with patch("market_agent.heatmap_data._download_closes", return_value=bars):
            def current() -> pd.DataFrame:
                return fetch_heatmap_prices(ticker, crypto=True, now=checked,
                                            read_daily=lambda *_: pd.DataFrame(), read_quotes=lambda *_: quotes,
                                            read_crypto_close=download_crypto_day_close)
            self.assertTrue(np.isnan(current().loc[0, "Percent Change"]))
            bars.loc[pd.Timestamp("2026-11-23T23:59Z")] = 100
            self.assertAlmostEqual(current().loc[0, "Percent Change"], 10)

    def test_multi_day_price_and_calendar_baseline_are_independent(self) -> None:
        data = prices("2026-11-23T17:00Z", {"A": {"2026-11-16": 80}, "B": {"2026-11-17": 90}}, {
            "A": (100, "2026-11-23T16:59Z"), "B": (100, "2026-11-23T16:58Z"),
        }, days=7)
        self.assertEqual(data.loc["A", "Percent Change"], 25)
        self.assertTrue(np.isnan(data.loc["B", "Percent Change"]))
        self.assertEqual(data.loc["B", "Price (USD)"], 100)

    def test_failed_quote_or_wrong_currency_leaves_other_symbol_intact(self) -> None:
        base = {"regularMarketTime": 1795467601, "regularMarketPrice": 165.05, "currency": "USD"}
        with patch("market_agent.heatmap_data.yf.Ticker", side_effect=lambda s: Mock(
            get_history_metadata=Mock(return_value={"A": base, "B": {**base, "currency": "EUR"}, "C": {}}[s])
        )):
            quotes = download_regular_quotes(("A", "B", "C"))
        self.assertEqual(quotes.loc["A", "Observed"], pd.Timestamp("2026-11-23T21:00:01Z"))
        self.assertEqual(quotes.loc["A", "Price"], 165.05)
        self.assertTrue(quotes.loc[["B", "C"], "Price"].isna().all())

    def test_yfinance_quote_times_preserve_the_observation_instant(self) -> None:
        checked = pd.Timestamp("2026-11-24T16:20Z")
        observed = {
            "EPOCH": int(checked.timestamp()),
            "EXCHANGE": checked.tz_convert("America/New_York"),
            "UTC": checked,
            "STALE": (checked - pd.Timedelta(minutes=21)).tz_convert("America/New_York"),
            "NAIVE": checked.tz_localize(None),
            "MISSING": None,
        }
        symbols = tuple(observed)
        daily = pd.DataFrame(100.0, index=[pd.Timestamp("2026-11-23")], columns=symbols)
        with patch("market_agent.heatmap_data.yf.Ticker", side_effect=lambda symbol: Mock(
            get_history_metadata=Mock(return_value={
                "currency": "USD", "regularMarketPrice": 105.0, "regularMarketTime": observed[symbol],
            })
        )):
            data = fetch_heatmap_prices(symbols, now=checked, read_daily=lambda *_: daily).set_index("Ticker")
        for symbol in ("EPOCH", "EXCHANGE", "UTC"):
            with self.subTest(symbol=symbol):
                self.assertEqual(data.loc[symbol, "Price as of"], "2026-11-24 16:20:00 UTC")
                self.assertEqual(data.loc[symbol, "Percent Change"], 5.0)
        self.assertTrue(data.loc[["STALE", "NAIVE", "MISSING"], "Price (USD)"].isna().all())
