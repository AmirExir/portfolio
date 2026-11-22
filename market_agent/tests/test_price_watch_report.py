"""Regression cases for reporting observed price triggers on both sides."""

import unittest

from market_agent.price_watch_report import build_price_watch, format_price_watch


def _observation(symbol: str, **changes: object) -> dict:
    return {
        "symbol": symbol, "as_of_session": "2026-11-20", "close": 96.0,
        "change_pct": 0.25, "prior_low": 90.0, "prior_high": 100.0,
        "next_low": 92.0, "next_high": 99.5,
        "shock_threshold_pct": 3.0, "shock": False, "breakdown": False,
        "entry_watch": False, "entry_triggered": False, **changes,
    }


class PriceWatchReportTests(unittest.TestCase):
    def test_benchmark_survives_separate_watchlist_caps(self) -> None:
        spy = _observation("SPY", change_pct=-4.0, shock=True)
        breakdown = _observation("AAA", close=89.0, change_pct=-5.0, shock=True, breakdown=True)
        entries = [
            _observation("BBB", close=99.0, entry_watch=True),
            _observation("CCC", close=101.5, entry_watch=True, entry_triggered=True),
        ]
        plan = build_price_watch([spy, breakdown, *entries], universe_count=6, limit=1)
        self.assertEqual(plan["benchmark"], spy)
        self.assertEqual(plan["observations"], [spy, breakdown, *entries])
        self.assertEqual(plan["downside_watch"], [breakdown])
        self.assertEqual(plan["entry_watch"], [entries[1]])
        self.assertEqual(plan["observed_count"], 4)
        text = "\n".join(format_price_watch(plan))
        self.assertIn("4 of 6 symbols", text)
        self.assertIn("unavailable for 2", text)
        self.assertIn("SPY benchmark (Nov 20, 2026) -4.00% in one session", text)
        self.assertIn("Exit / avoid watch (1 of 2 shown)", text)
        self.assertIn("Entry watch (1 of 2 shown)", text)
        self.assertIn("Loss reached the 3.00% alert threshold", text)
        self.assertIn("Already below 90.00", text)
        self.assertIn("If held, review an exit", text)
        self.assertIn("Recovery needs a close back above that level", text)
        self.assertIn("For a new entry, watch the recovery or breakout level", text)
        self.assertIn("Already above the prior 20-session closing high of 100.00", text)
        self.assertIn("close back below 100.00 invalidates the breakout", text)

    def test_relative_loss_priority_and_conditional_action(self) -> None:
        a = _observation("AAA", shock=True, change_pct=-6.0, shock_threshold_pct=4.0)
        b = _observation("BBB", shock=True, change_pct=-5.0, shock_threshold_pct=2.0)
        plan = build_price_watch([a, b], universe_count=2, limit=0)
        self.assertEqual(plan["downside_watch"], [b, a])
        self.assertIsNone(plan["benchmark"])
        self.assertIn(
            "Watch for a close below 92.00. Recheck for a breakout on a close above 99.50.",
            "\n".join(format_price_watch(plan)),
        )

    def test_pending_entry_keeps_micro_price_and_symbol_identity(self) -> None:
        entry = _observation(
            "PEPE-USD", close=0.000002553, prior_high=0.000002559,
            prior_low=0.00000212, next_low=0.000002129, next_high=0.000002556, entry_watch=True,
        )
        plan = build_price_watch([entry], universe_count=1, limit=2)
        text = "\n".join(format_price_watch(plan))
        self.assertIn("PEPE-USD (Nov 20, 2026) closed at 0.000002553", text)
        self.assertIn("entry on a close above 0.000002556", text)
        self.assertIn("close below 0.000002129 invalidates", text)
        self.assertNotIn("Already above", text)
        self.assertIn("Exit / avoid watch — no triggers in this scan", text)

    def test_missing_inputs_distinguished_from_no_trigger(self) -> None:
        invalid = [
            _observation("AAA", as_of_session=None),
            _observation("BBB", close=float("nan")),
            _observation("CCC", prior_low=0.0),
            _observation("DDD", shock="false"),
        ]
        plan = build_price_watch(invalid, universe_count=5, limit=2)
        text = "\n".join(format_price_watch(plan))
        self.assertIn("Price watch unavailable", text)
        self.assertIn("0 of 5", text)
        self.assertNotIn("no triggers", text)
        quiet = _observation("SPY")
        earlier = _observation("AAA", as_of_session="2026-11-19")
        plan = build_price_watch([quiet, earlier], universe_count=3, limit=2)
        text = "\n".join(format_price_watch(plan))
        self.assertIn("Nov 19, 2026 to Nov 20, 2026", text)
        self.assertEqual(plan["observations"], [quiet, earlier])
        self.assertIn("Watch for a close below its latest 20-session closing low of 92.00", text)
        self.assertIn("Exit / avoid watch — no triggers in this scan", text)
        self.assertIn("Entry watch — no triggers in this scan", text)

    def test_benchmark_recovery_after_breakdown(self) -> None:
        spy = _observation("SPY", close=89.0, change_pct=-2.0, breakdown=True)
        plan = build_price_watch([spy], universe_count=1, limit=1)
        benchmark_line = next(line for line in format_price_watch(plan) if line.startswith("SPY benchmark"))
        self.assertIn("Already below its prior 20-session closing low of 90.00", benchmark_line)
        self.assertIn("Watch for a close back above it", benchmark_line)

    def test_quiet_single_symbol_request_has_levels_for_next_close(self) -> None:
        quiet = _observation("AAPL")
        plan = build_price_watch([quiet], universe_count=1, limit=3)
        text = "\n".join(format_price_watch(plan))
        self.assertEqual(plan["entry_count"], 0)
        self.assertEqual(plan["downside_count"], 0)
        self.assertIn("Next closing levels for symbols without a trigger", text)
        self.assertIn("AAPL (Nov 20, 2026) closed at 96.00", text)
        self.assertIn("below 92.00 or recheck for a breakout above 99.50", text)

    def test_legacy_and_invalid_next_levels_keep_only_historical_references(self) -> None:
        legacy = _observation("SPY", shock=True, change_pct=-4.0)
        del legacy["next_low"], legacy["next_high"]
        pending = _observation("BBB", entry_watch=True, next_high=None)
        neutral = _observation("CCC", next_low=float("nan"))
        plan = build_price_watch([legacy, pending, neutral], universe_count=3, limit=2)
        text = "\n".join(format_price_watch(plan))
        self.assertEqual(text.count("Next-close levels unavailable."), 4)
        self.assertIn("Prior 20-session closing range 90.00–100.00", text)
        self.assertNotIn("a close below 90.00", text)
        self.assertNotIn("a close above 100.00", text)

    def test_pending_entries_rank_against_their_next_closing_high(self) -> None:
        closer_to_old = _observation("AAA", close=99.3, next_high=100.0, entry_watch=True)
        closer_to_next = _observation("BBB", close=98.3, next_high=98.4, entry_watch=True)
        plan = build_price_watch([closer_to_old, closer_to_next], universe_count=2, limit=1)
        self.assertEqual(plan["entry_watch"], [closer_to_next])

    def test_technical_readings_follow_watch_forecast_and_rsi_extremes(self) -> None:
        def item(symbol: str, value: float, state: str, **kwargs: object) -> dict:
            return _observation(symbol, technical={
                "trend": "Mixed", "trend_windows": [20, 50], "rsi": value,
                "rsi_state": state, "patterns": ["Bearish MA Crossover"],
                "support": 90.5, "resistance": 98.0,
            }, **kwargs)

        observations = [
            item("SPY", 30, "Oversold", shock=True), item("AAA", 25, "Oversold"),
            item("BBB", 80, "Overbought"), item("CCC", 50, "Neutral"),
            item("DDD", 75, "Overbought"), item("EEE", 65, "Neutral", entry_watch=True),
            item("COMP5692-USD", 55, "Neutral", report_symbol="COMP", as_of_session="2026-11-21"),
        ]
        observations[3]["technical"]["patterns"] = ["Confirmed Double Bottom"]
        observations[3]["technical"]["shape_necklines"] = {"Confirmed Double Bottom": 95.99}
        plan = build_price_watch(observations, universe_count=7, limit=1)
        text = "\n".join(format_price_watch(plan, forecast_symbols=["COMP", "SPY"]))
        readings = text.split("TECHNICAL READINGS")[1]
        self.assertIn("across 7 symbols | 2 oversold (below 35) | 2 overbought (above 70)", readings)
        self.assertIn("COMP5692-USD (Nov 21, 2026) | Mixed trend (SMA 20/50)", readings)
        self.assertIn("RSI(14) 55.0 (neutral) | Bearish MA Crossover", readings)
        self.assertIn("Support 90.50 | Resistance 98.00", readings)
        self.assertIn("neutral) | Confirmed Double Bottom (closed above neckline 95.99)", readings)
        for symbol in ("SPY", "AAA", "BBB", "CCC", "EEE", "COMP5692-USD"):
            self.assertEqual(readings.count(f"{symbol} ("), 1)
        self.assertNotIn("DDD (", readings)

    def test_technical_display_handles_sub_cent_levels_and_threshold_rounding(self) -> None:
        observation = _observation("PEPE-USD", close=0.000002553, technical={
            "rsi_state": "Oversold", "rsi": 34.98, "patterns": [],
            "support": 0.000002552, "resistance": 0.000002559,
        })
        plan = build_price_watch([observation], universe_count=1, limit=1)
        text = "\n".join(format_price_watch(plan))
        self.assertIn("Trend unavailable | RSI(14) just below 35 (oversold)", text)
        self.assertIn("Support 0.000002552 | Resistance 0.000002559", text)
        observation["technical"].update(support=10, resistance=float("nan"), rsi=70.01, rsi_state="Overbought")
        text = "\n".join(format_price_watch(plan))
        self.assertIn("RSI(14) just above 70 (overbought)", text)
        self.assertIn("Support unconfirmed below | Resistance unconfirmed above", text)
        old = build_price_watch([_observation("SPY")], universe_count=1, limit=1)
        self.assertNotIn("TECHNICAL READINGS", format_price_watch(old))


if __name__ == "__main__":
    unittest.main()
