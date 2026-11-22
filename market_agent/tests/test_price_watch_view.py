"""Exercise the scheduled dashboard panel without starting its network-backed tabs."""

from __future__ import annotations

import unittest

from streamlit.testing.v1 import AppTest

from market_agent.price_watch_report import build_price_watch, has_scheduled_report_data


def _app(plan: dict | None) -> AppTest:
    return AppTest.from_string(
        "from market_agent.price_watch_view import render_scheduled_price_watch\n"
        f"render_scheduled_price_watch({plan!r})\n"
        "import streamlit as st\n"
        "st.subheader('Forecast and policy signals')"
    ).run(timeout=30)


class PriceWatchViewTests(unittest.TestCase):
    def test_search_reaches_symbols_beyond_the_summary_cap_and_keeps_micro_prices(self) -> None:
        item = {
            "as_of_session": "2026-11-20", "symbol": "AAA", "close": 88.0, "prior_low": 90.0,
            "prior_high": 100.0, "change_pct": -11.0, "shock_threshold_pct": 3.0,
            "shock": True, "breakdown": True, "entry_watch": False, "entry_triggered": False,
        }
        micro = {
            **item, "symbol": "BBB-USD", "close": 0.0000021, "prior_low": 0.0000022,
            "prior_high": 0.0000025, "change_pct": -5.0,
            "next_low": 0.0000021, "next_high": 0.00000249999,
        }
        neutral = {**item, "symbol": "CCC", "close": 95.0, "change_pct": 0.1, "shock": False, "breakdown": False}
        plan = build_price_watch([item, micro, neutral], universe_count=3, limit=1)
        app = _app(plan)
        self.assertFalse(app.exception)
        self.assertNotIn("BBB-USD", app.text[0].value)
        self.assertEqual(app.dataframe[0].value["Symbol"].tolist(), ["AAA", "BBB-USD", "CCC"])
        self.assertEqual(app.dataframe[0].value.iloc[2]["Price condition"], "No trigger")
        self.assertEqual(app.dataframe[0].value.iloc[2]["Next-close lower level"], "Unavailable")
        self.assertEqual(app.dataframe[0].value.iloc[2]["Next-close upper level"], "Unavailable")

        app.text_input[0].set_value(" bb ").run()
        row = app.dataframe[0].value.iloc[0]
        self.assertEqual(app.dataframe[0].value["Symbol"].tolist(), ["BBB-USD"])
        self.assertEqual(float(row["Close"]), micro["close"])
        self.assertIn("Below prior closing low", row["Price condition"])
        self.assertEqual(float(row["Prior 20-session closing low"]), micro["prior_low"])
        self.assertEqual(float(row["Next-close lower level"]), micro["next_low"])
        self.assertEqual(float(row["Prior 20-session closing high"]), micro["prior_high"])
        self.assertEqual(float(row["Next-close upper level"]), micro["next_high"])
        app.text_input[0].set_value("missing").run()
        self.assertFalse(app.dataframe)
        self.assertIn("No observed symbol", app.caption[0].value)

    def test_legacy_report_has_explicit_availability_message(self) -> None:
        app = _app(None)
        self.assertFalse(app.exception)
        self.assertIn("unavailable in this saved report", app.caption[0].value)
        self.assertFalse(app.dataframe)

    def test_incomplete_saved_watch_does_not_hide_the_forecast_section(self) -> None:
        app = _app({"observed_count": 5})
        self.assertFalse(app.exception)
        self.assertIn("incomplete or invalid", app.warning[0].value)
        self.assertEqual(app.subheader[0].value, "Forecast and policy signals")

    def test_price_only_payload_requires_a_complete_run_and_verified_session(self) -> None:
        item = {
            "symbol": "SPY", "as_of_session": "2026-11-20", "lag_sessions": 0,
            "close": 94.0, "prior_low": 95.0, "prior_high": 100.0, "change_pct": -4.0,
            "shock_threshold_pct": 3.0, "shock": True, "breakdown": True,
            "entry_watch": False, "entry_triggered": False,
        }
        watch = build_price_watch([item], universe_count=1, limit=5)
        payload = {
            "run_complete": True, "ranked_count": 0, "rows": [], "price_watch": watch,
            "symbols": ["SPY"], "universe_count": 1, "as_of_sessions": ["2026-11-20"],
        }
        self.assertTrue(has_scheduled_report_data(payload))
        self.assertTrue(has_scheduled_report_data({"rows": [{"Symbol": "SPY"}]}))
        for updates in (
            {"run_complete": False}, {"run_complete": 1}, {"ranked_count": False},
            {"rows": None}, {"symbols": ["QQQ"]}, {"universe_count": True},
            {"universe_count": 2}, {"price_watch": {}}, {"as_of_sessions": ["2026-11-19"]},
            {"price_watch": {**watch, "observations": [{**item, "lag_sessions": 1}]}},
            {"price_watch": {**watch, "observations": [{**item, "prior_low": 0.0}]}},
        ):
            with self.subTest(updates=updates):
                self.assertFalse(has_scheduled_report_data({**payload, **updates}))
