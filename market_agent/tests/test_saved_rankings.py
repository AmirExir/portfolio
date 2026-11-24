"""Verify saved results preserve scope and survive ordinary dashboard reruns."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from market_agent.agent.forecast import ForecastResult
from market_agent.patterns import summarize_patterns
from market_agent.saved_rankings import (
    dashboard_result_path, scheduled_pattern_rows, scheduled_rows_for_symbols, stored_ranking_rows,
)


class SavedRankingTests(unittest.TestCase):
    def test_scheduled_rows_and_patterns_match_the_requested_asset_and_horizon(self) -> None:
        row = {"Symbol": "COMP", "Forecast Return %": -1.0, "As Of Session": "2026-11-21",
               "Primary Pattern": "Oversold", "All Patterns": "Oversold, Downtrend"}
        payload = {
            "horizon_days": 30, "rows": [row, {**row, "Symbol": "AAPL"}],
            "short_horizon_reports": [{"horizon_days": 1, "rows": [{**row, "Forecast Return %": 0.5}]}],
            "price_watch": {"observations": [
                {"symbol": "COMP5692-USD", "as_of_session": "2026-11-20", "technical": {"rsi": 4.0, "patterns": []}},
            ]},
        }
        symbols = ["COMP5692-USD"]
        aliases = {symbols[0]: "COMP"}
        result = scheduled_rows_for_symbols(payload, symbols, 1, aliases)
        self.assertEqual(result["Forecast Return %"].tolist(), [0.5])
        self.assertEqual(result["Symbol"].tolist(), symbols)
        self.assertTrue(scheduled_rows_for_symbols(payload, symbols, 7, aliases).empty)
        self.assertTrue(scheduled_rows_for_symbols({**payload, "run_complete": False}, symbols, 1, aliases).empty)
        self.assertTrue(scheduled_rows_for_symbols({**payload, "symbols": ["COMP"]}, symbols, 30, aliases).empty)
        patterns = scheduled_pattern_rows(payload, symbols, aliases)
        self.assertNotIn("RSI", patterns)
        counts = summarize_patterns(patterns)
        self.assertEqual(counts["Count"].tolist(), [1, 1])
        self.assertNotIn("Avg 20D Return %", counts)
        self.assertEqual(counts["Symbols"].tolist(), symbols * 2)
        price_only = {**payload, "rows": [], "price_watch": {"observations": [
            {"symbol": "COMP5692-USD", "as_of_session": "2026-11-21", "close": 45.0,
             "technical": {"rsi": 12.0, "rsi_state": "Oversold", "trend": "Downtrend", "patterns": []}},
        ]}}
        prices = scheduled_pattern_rows(price_only, symbols, aliases)
        self.assertEqual(prices["All Patterns"].tolist(), ["Downtrend, Oversold"])
        self.assertEqual(prices["RSI"].tolist(), [12.0])

    def test_cached_rankings_keep_final_weights_and_reject_mismatched_config(self) -> None:
        payload = {"rows": [{"Symbol": "AAPL", "Forecast Return %": 1.5, "Policy Target %": 0.0}],
                   "symbols": ["AAPL"], "horizon_days": 30, "primary_model": "Best Validation"}
        rows = stored_ranking_rows(payload, ["AAPL"], 30, "Best Validation")
        self.assertEqual(rows["Policy Target %"].tolist(), [0.0])
        for symbols, horizon, model in ((["SPY"], 30, "Best Validation"), (["AAPL"], 1, "Best Validation"),
                                         (["AAPL"], 30, "Ridge")):
            self.assertTrue(stored_ranking_rows(payload, symbols, horizon, model).empty)
        self.assertNotEqual(dashboard_result_path(".", "patterns", (["AAPL"], 365, 10, 50)),
                            dashboard_result_path(".", "patterns", (["AAPL"], 365, 20, 50)))

    def test_dashboard_restores_published_rows_and_an_explicit_scan_after_reopening(self) -> None:
        # AppTest instances share Streamlit's process-global read cache even across temporary apps.
        st.cache_data.clear()
        self.addCleanup(st.cache_data.clear)
        source = Path(__file__).resolve().parents[1] / "market_agent_app.py"
        index = pd.bdate_range(end="2026-11-20", periods=65)
        prices = pd.Series(np.linspace(75.0, 95.0, len(index)), index=index)
        frame = pd.DataFrame({column: prices for column in ("open", "high", "low", "close")})
        frame["volume"] = 2500
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(sys, "path", [str(source.parent), *sys.path]),
            patch("agent.data.get_ohlcv", return_value=frame),
            patch("agent.earnings.load_earnings_payload_file", return_value=Mock(available=False, error_code="fixture")),
            patch("agent.forecast.compare_forecast_models") as fit,
            patch("requests.get", return_value=Mock(status_code=404)),
            patch("yfinance.download", return_value=pd.DataFrame()),
            patch("yfinance.Ticker.get_history_metadata", return_value={}),
        ):
            script = Path(directory) / source.name
            script.write_text(source.read_text())
            reports = script.parent / "reports"
            reports.mkdir()
            payload = {
                "generated_at": "2026-11-22T02-06-09Z", "horizon_days": 30,
                "rows": [{"Symbol": "AAPL", "Forecast Return %": -1, "As Of Session": "2026-11-20",
                          "Primary Pattern": "Oversold", "All Patterns": "Oversold"}],
            }
            (reports / "ml_forecast_rankings_latest.json").write_text(json.dumps(payload))
            app = AppTest.from_file(str(script)).run(timeout=60)
            tabs = {tab.label: tab for tab in app.tabs}
            self.assertFalse(app.exception)
            self.assertTrue(any("Scheduled ranking" in c.value for c in tabs["Universe rankings"].caption))
            self.assertTrue(any("Scheduled scan" in c.value for c in tabs["Most Common Patterns"].caption))

            app.multiselect(key="market_universe_patterns").set_value(["AAPL"]).run(timeout=60)
            next(button for button in app.button if button.label == "Run pattern scan").click().run(timeout=60)
            self.assertFalse(app.exception)
            files = list((reports / "dashboard_results").glob("patterns_*.json"))
            self.assertEqual(len(files), 1)
            saved = json.loads(files[0].read_text())
            self.assertEqual(saved["rows"][0]["Symbol"], "AAPL")
            self.assertEqual(saved["rows"][0]["As Of Session"], "2026-11-20")

            reopened = AppTest.from_file(str(script)).run(timeout=60)
            reopened.multiselect(key="market_universe_patterns").set_value(["AAPL"]).run(timeout=60)
            patterns_tab = next(tab for tab in reopened.tabs if tab.label == "Most Common Patterns")
            self.assertTrue(any("Saved scan" in c.value for c in patterns_tab.caption))
            self.assertFalse(reopened.exception)
            fit.assert_not_called()

            forecast = pd.DataFrame(
                {"forecast_close": [96.75]}, index=pd.DatetimeIndex(["2027-01-06"], name="date"),
            )
            fit.return_value = {"Ridge": ForecastResult(forecast, {"forecast_change_pct": 1.84}, "Ridge")}
            reopened.button(key="ranking_run").click().run(timeout=60)
            self.assertFalse(reopened.exception)
            self.assertEqual(fit.call_count, 1)
            rankings = list((reports / "dashboard_results").glob("rankings_*.json"))
            self.assertEqual(len(rankings), 1)
            archived_row = json.loads(rankings[0].read_text())["rows"][0]
            self.assertIn("Portfolio Gross %", archived_row)
            self.assertEqual(archived_row["Forecast Return %"], 1.84)

            reopened.run(timeout=60)
            rankings_tab = next(tab for tab in reopened.tabs if tab.label == "Universe rankings")
            self.assertTrue(any("Saved on-demand ranking" in c.value for c in rankings_tab.caption))
            display = next(table.value for table in rankings_tab.dataframe if "Forecast Return %" in table.value)
            self.assertEqual(display["Forecast Return %"].tolist(), [1.84])
            self.assertEqual(fit.call_count, 1)


if __name__ == "__main__":
    unittest.main()
