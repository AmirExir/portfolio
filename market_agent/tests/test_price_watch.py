"""Point-in-time range, shock and calendar checks for the daily price watch."""

from __future__ import annotations

from statistics import stdev
import unittest

import numpy as np
import pandas as pd

from market_agent.agent.price_watch import price_watch_observation


def _daily(prices: list[float]) -> pd.Series:
    return pd.Series(prices, index=pd.date_range("2025-02-01", periods=len(prices)))


class PriceWatchTests(unittest.TestCase):
    def test_loss_at_floor_triggers_even_at_float_boundary(self) -> None:
        closes = _daily([10.1] * 21 + [9.898])
        result = price_watch_observation("BTC-USD", closes)
        self.assertEqual(result["symbol"], "BTC-USD")
        self.assertEqual(result["as_of_session"], "2025-02-22")
        self.assertEqual(result["close"], 9.898)
        self.assertAlmostEqual(result["change_pct"], -2.0)
        self.assertEqual(result["shock_threshold_pct"], 2.0)
        self.assertTrue(result["shock"])
        self.assertTrue(result["breakdown"])
        self.assertFalse(result["entry_watch"])
        closes.iloc[-1] = 9.899
        self.assertFalse(price_watch_observation("BTC-USD", closes)["shock"])

    def test_volatility_and_range_exclude_latest_observation(self) -> None:
        historical_returns = [0.04, -0.04] * 10
        history = [100.0]
        for change in historical_returns:
            history.append(history[-1] * (1 + change))
        result = price_watch_observation("BTC-USD", _daily(history + [50.0]))
        self.assertAlmostEqual(result["shock_threshold_pct"], 200.0 * stdev(historical_returns))
        self.assertTrue(result["shock"])
        self.assertEqual(result["prior_low"], min(history[1:]))
        self.assertEqual(result["prior_high"], max(history[1:]))
        self.assertTrue(result["breakdown"])

    def test_entry_watch_requires_positive_momentum_and_confirmation_is_strict(self) -> None:
        history = [120.0] + [97.0] * 17 + [100.0, 98.0, 98.0]
        for latest, watched, triggered in (
            (97.99, False, False),
            (98.00, True, False),
            (100.0, True, False),
            (101.0, True, True),
        ):
            with self.subTest(latest=latest):
                result = price_watch_observation("BTC-USD", _daily(history + [latest]))
                self.assertEqual(result["prior_high"], 100.0)
                self.assertEqual(result["entry_watch"], watched)
                self.assertEqual(result["entry_triggered"], triggered)
        flat = price_watch_observation("BTC-USD", _daily([100.0] * 22))
        self.assertFalse(flat["entry_watch"])
        self.assertFalse(flat["breakdown"])
        self.assertFalse(flat["shock"])

    def test_next_range_rolls_off_oldest_extreme_and_includes_current_close(self) -> None:
        for oldest, latest, prior, following in (
            (80.0, 110.0, (80.0, 100.0), (100.0, 110.0)),
            (120.0, 90.0, (100.0, 120.0), (90.0, 100.0)),
        ):
            with self.subTest(oldest=oldest, latest=latest):
                history = [100.0, oldest] + [100.0] * 19 + [latest]
                result = price_watch_observation("BTC-USD", _daily(history))
                self.assertEqual((result["prior_low"], result["prior_high"]), prior)
                self.assertEqual((result["next_low"], result["next_high"]), following)
                next_day = price_watch_observation("BTC-USD", _daily(history + [105.0]))
                self.assertEqual((next_day["prior_low"], next_day["prior_high"]), following)

    def test_shock_blocks_entry_even_when_price_is_near_high(self) -> None:
        history = [97.0] * 20 + [100.0]
        result = price_watch_observation("BTC-USD", _daily(history + [98.0]))
        self.assertTrue(result["shock"])
        self.assertFalse(result["breakdown"])
        self.assertFalse(result["entry_watch"])

    def test_pending_entry_uses_high_remaining_for_the_next_close(self) -> None:
        history = [100.0, 130.0] + [95.0] * 17 + [100.0, 98.0]
        for latest, pending in ((98.0, True), (97.99, False)):
            with self.subTest(latest=latest):
                result = price_watch_observation("BTC-USD", _daily(history + [latest]))
                self.assertEqual(result["prior_high"], 130.0)
                self.assertEqual(result["next_high"], 100.0)
                self.assertEqual(result["entry_watch"], pending)
                self.assertFalse(result["entry_triggered"])

    def test_equity_skips_weekends_and_presidents_day_crypto_requires_them(self) -> None:
        weekdays = pd.bdate_range("2025-02-03", "2025-03-07")
        sessions = weekdays.drop(pd.Timestamp("2025-02-17"))[-22:]
        closes = pd.Series(100.0, index=sessions)
        self.assertIsNotNone(price_watch_observation("SPY", closes))
        self.assertIsNone(price_watch_observation("BTC-USD", closes))
        missing_equity_session = pd.Series(
            100.0, index=weekdays.drop([pd.Timestamp("2025-02-17"), pd.Timestamp("2025-03-05")])[-22:]
        )
        self.assertIsNone(price_watch_observation("SPY", missing_equity_session))
        crypto = _daily([100.0] * 23)
        self.assertIsNone(price_watch_observation("BTC-USD", crypto.drop(crypto.index[10])))

    def test_incomplete_invalid_duplicate_or_unordered_window_is_unavailable(self) -> None:
        closes = _daily([100.0] * 22)
        bad_index = list(closes.index)
        bad_index[-1] = bad_index[-2] + pd.Timedelta(hours=1)
        for invalid in (
            closes.iloc[1:], closes.iloc[::-1], closes.set_axis(bad_index),
            closes.reset_index(drop=True),
        ):
            self.assertIsNone(price_watch_observation("BTC-USD", invalid))
        for value in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(value=value):
                invalid = closes.copy()
                invalid.iloc[5] = value
                self.assertIsNone(price_watch_observation("BTC-USD", invalid))
