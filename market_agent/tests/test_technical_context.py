"""Completed-session technical evidence shared by the scanner and price watch."""

import unittest

import numpy as np
import pandas as pd

from market_agent.agent.price_watch import price_watch_observation
from market_agent.agent.technicals import DOUBLE_BOTTOM_LABEL, DOUBLE_TOP_LABEL, close_technicals, rsi
from market_agent.patterns import recognize_patterns


def _closes(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2025-01-01", periods=len(values)), dtype=float)


def _double_top_before_break() -> list[float]:
    # The two peaks of 110 and 110.5 bracket a trough (the neckline) at 103.
    return [94, 95, 97, 100, 104, 107, 110, 107.5, 105.5, 103, 106, 108, 110.5, 108.3, 106, 104]


class TechnicalContextTests(unittest.TestCase):
    def test_double_top_and_bottom_require_first_confirming_close(self) -> None:
        prices = _double_top_before_break() + [103, 102]
        for path, label, neckline in (
            (prices, DOUBLE_TOP_LABEL, 103.0),
            ([220 - p for p in prices], DOUBLE_BOTTOM_LABEL, 117.0),
        ):
            for length in range(1, len(path)):
                with self.subTest(label=label, prefix=length):
                    early = close_technicals(_closes(path[:length]))
                    self.assertNotIn(label, early["patterns"])
                    self.assertEqual(early["shape_necklines"], {})
            for scale in (0.00001, 1, 1000):
                reading = close_technicals(_closes([scale * price for price in path]))
                self.assertIn(label, reading["patterns"])
                self.assertEqual(reading["shape_necklines"], {label: scale * neckline})
            history = _closes([path[0]] * 60 + path)
            self.assertIn(label, recognize_patterns(history.to_frame("close"))["All Patterns"])
            self.assertIn(label, price_watch_observation("BTC-USD", history)["technical"]["patterns"])

    def test_false_or_stale_double_shapes_are_excluded(self) -> None:
        before = _double_top_before_break()
        unequal = before.copy()
        unequal[12] = 116
        shallow = before.copy()
        shallow[7:12] = [109, 108, 107.5, 108, 109]
        tied_peak = before.copy()
        tied_peak[13] = tied_peak[12]
        tied_neckline = before.copy()
        tied_neckline[8] = tied_neckline[9]
        cases = {
            "unequal turns": unequal + [102],
            "shallow neckline": shallow + [102],
            "tied peak": tied_peak + [102],
            "tied neckline": tied_neckline + [102],
            "late break": before + [104] * 8 + [102],
            "already broken": before + [102, 101],
            "recross": before + [102, 105, 102],
            "invalidated turn": before + [113, 102],
            "monotonic": list(range(90, 120)),
        }
        for case, path in cases.items():
            with self.subTest(case=case):
                self.assertNotIn(DOUBLE_TOP_LABEL, close_technicals(_closes(path))["patterns"])
                self.assertNotIn(DOUBLE_BOTTOM_LABEL, close_technicals(_closes([220 - p for p in path]))["patterns"])

    def test_rsi_edges_warmup_and_strict_thresholds(self) -> None:
        for closes, expected, state in (
            (range(100, 161), 100.0, "Overbought"),
            (range(161, 100, -1), 0.0, "Oversold"),
            ([100] * 61, 50.0, "Neutral"),
        ):
            with self.subTest(state=state):
                series = _closes(list(closes))
                self.assertTrue(rsi(series).iloc[:14].isna().all())
                self.assertEqual(rsi(series).iloc[14], expected)
                reading = price_watch_observation("BTC-USD", series)["technical"]
                self.assertEqual((reading["rsi"], reading["rsi_state"]), (expected, state))
                self.assertEqual(recognize_patterns(series.to_frame("close"))["RSI"], expected)
        for rise, fall, expected in ((7.0, 13.0, 35.0), (7.0, 3.0, 70.0)):
            close = _closes([100, 100 + rise] + [100 + rise - fall] * 13)
            reading = close_technicals(close)
            self.assertAlmostEqual(reading["rsi"], expected)
            self.assertEqual(reading["rsi_state"], "Neutral")

    def test_rsi_uses_current_date_after_last_loss_rolls_out(self) -> None:
        prices = _closes([110.0] * 45 + [90.0] + list(range(91, 105)))
        self.assertLess(rsi(prices).iloc[-2], 100)
        self.assertEqual(recognize_patterns(prices.to_frame("close"))["RSI"], 100.0)

    def test_trend_filters_and_setups_match_the_existing_scanner(self) -> None:
        for prices, trend, windows in (
            (list(range(100, 151)), "Uptrend", [20, 50]),
            (list(range(301, 100, -1)), "Downtrend", [20, 50, 200]),
            ([100.0] * 155 + list(range(1, 51)), "Mixed", [20, 50, 200]),
        ):
            with self.subTest(trend=trend):
                reading = price_watch_observation("BTC-USD", _closes(prices))["technical"]
                self.assertEqual(reading["trend"], trend)
                self.assertEqual(reading["trend_windows"], windows)
        pullback = _closes(list(range(100, 160)) + [145])
        tech = price_watch_observation("BTC-USD", pullback)["technical"]
        self.assertIn("Pullback In Uptrend", tech["patterns"])
        self.assertIn("Pullback In Uptrend", recognize_patterns(pullback.to_frame("close"))["All Patterns"])
        rising = price_watch_observation("BTC-USD", _closes(list(range(100, 161))))["technical"]
        self.assertIn("Near 60-Day High", rising["patterns"])
        crossing = price_watch_observation(
            "BTC-USD", _closes([100] * 54 + [95, 90, 89, 92, 96, 98]),
            short_window=2, long_window=5,
        )["technical"]
        self.assertIn("Bullish MA Crossover", crossing["patterns"])
        self.assertEqual(crossing["trend_windows"], [2, 5])

    def test_history_stops_at_gap_and_preserves_price_watch_or_shorter_indicators(self) -> None:
        uninterrupted = _closes(list(range(100, 340)))
        for earlier_gap in (uninterrupted.drop(uninterrupted.index[-36]), uninterrupted.mask(uninterrupted == 304)):
            tech = price_watch_observation("BTC-USD", earlier_gap)["technical"]
            self.assertEqual(tech["available_sessions"], 35)
            self.assertEqual(tech["trend"], "Unavailable")
            self.assertEqual(tech["trend_windows"], [])
            self.assertEqual(tech["rsi"], 100)
            self.assertNotIn("Near 60-Day High", tech["patterns"])
        for windows in ((0, 50), (50, 20)):
            obs = price_watch_observation("BTC-USD", uninterrupted, short_window=windows[0], long_window=windows[1])
            self.assertTrue(obs["entry_triggered"])
            self.assertEqual(obs["technical"]["trend"], "Unavailable")
            self.assertEqual(obs["technical"]["rsi_state"], "Overbought")
        partial = close_technicals(_closes([100.0] * 14))
        self.assertIsNone(partial["rsi"])
        self.assertEqual(partial["rsi_state"], "Unavailable")

    def test_nearest_confirmed_pivot_is_selected_by_price(self) -> None:
        close = _closes([100] * 22 + [99, 98, 99, 100, 101, 102, 101, 100, 98, 95, 98, 100, 103, 105, 103, 102, 101, 100])
        tech = price_watch_observation("BTC-USD", close)["technical"]
        self.assertEqual((tech["support"], tech["resistance"]), (98.0, 102.0))
        for new_close, level in ((90.0, "support"), (110.0, "resistance")):
            latest = close.copy()
            latest.iloc[-1] = new_close
            self.assertIsNone(price_watch_observation("BTC-USD", latest)["technical"][level])

    def test_pivots_need_two_confirmation_closes_and_exclude_ties_and_old_swings(self) -> None:
        prefix = [100] * 22 + [99, 95, 98]
        self.assertIsNone(close_technicals(_closes(prefix))["support"])
        self.assertEqual(close_technicals(_closes(prefix + [100]))["support"], 95.0)
        self.assertIsNone(close_technicals(_closes(prefix + [100] * 60))["support"])
        self.assertIsNone(close_technicals(_closes([100, 100, 95, 95, 100, 100]))["support"])
        self.assertIsNone(close_technicals(_closes([100] * 4))["resistance"])
        prices = _closes(prefix + [100] * 25)
        prices.iloc[-24] = np.nan
        reading = price_watch_observation("BTC-USD", prices)["technical"]
        self.assertEqual(reading["available_sessions"], 23)
        self.assertIsNone(reading["support"])
