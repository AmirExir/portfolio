"""Render the public dashboard with deterministic data and no external services."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import pyarrow as pa
from streamlit.testing.v1 import AppTest


class DashboardPresentationTests(unittest.TestCase):
    def test_navigation_and_portfolio_chart_labels(self) -> None:
        source = Path(__file__).resolve().parents[1] / "market_agent_app.py"
        prices = pd.Series([98.0, 99.0, 100.0], index=pd.bdate_range("2026-02-02", periods=3))
        frame = pd.DataFrame({field: prices for field in ("open", "high", "low", "close")})
        frame["volume"] = 1000
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("os.environ", {"MARKET_REPORT_TIMEZONE": "America/Chicago"}),
            patch.object(sys, "path", [str(source.parent), *sys.path]),
            patch("agent.data.get_ohlcv", return_value=frame),
            patch("agent.forecast.compare_forecast_models") as fit,
            patch("agent.broker.submit_order") as submit,
            patch("agent.earnings.load_earnings_payload_file", return_value=Mock(available=False, error_code="fixture")),
            patch("requests.get", return_value=Mock(status_code=404)),
            patch("yfinance.download", return_value=pd.DataFrame()),
        ):
            script = Path(directory) / source.name
            script.write_text(source.read_text())
            reports = script.parent / "reports"
            reports.mkdir()
            # Disabled models emit entirely null columns in real scheduled reports.
            (reports / "ml_forecast_rankings_2026-11-22T02-06-09Z.json").write_text(json.dumps({
                "generated_at": "2026-11-22T02-06-09Z",
                "rows": [{"Symbol": "XYZ", "Forecast Return %": -1.25, "LSTM Return %": None}],
            }))
            earlier = "ml_forecast_rankings_2026-11-19T02-06-09Z.json"
            (reports / earlier).write_text(json.dumps({
                "generated_at": "2026-11-19T02-06-09Z",
                "rows": [{"Symbol": "ABC", "Forecast Return %": 0.6, "LSTM Return %": None}],
            }))
            app = AppTest.from_file(str(script))
            app.run(timeout=60)
            new_stocks = {"NBIS": "Nebius", "CRWV": "CoreWeave", "IREN": "IREN"}
            rankings = app.multiselect(key="market_universe_rankings")
            picker = app.selectbox(key="symbol_select")
            for ticker, company in new_stocks.items():
                with self.subTest(ticker=ticker):
                    self.assertEqual(rankings.value.count(ticker), 1)
                    label = picker.format_func(ticker)
                    self.assertIn(company, label)
                    self.assertIn(ticker, label)
                    self.assertIn(label, picker.options)
            app.selectbox(key="scheduled_archive").set_value(earlier).run(timeout=60)
            scheduled = next(tab for tab in app.tabs if tab.label == "Scheduled outlook")
            historical = next(table for table in scheduled.dataframe if "LSTM Return %" in table.value.columns)
            self.assertEqual(historical.value["Symbol"].tolist(), ["ABC"])
            app.selectbox(key="scheduled_archive").set_value(None).run(timeout=60)
            app.multiselect(key="market_universe_heatmap").set_value(list(new_stocks)).run(timeout=60)
            self.assertEqual(app.multiselect(key="market_universe_rankings").value, list(new_stocks))
            self.assertEqual(app.multiselect(key="market_universe_patterns").value, list(new_stocks))
            self.assertIn("Showing 3 selected non-crypto symbols.", [caption.value for caption in app.caption])
            fit.assert_not_called()
            submit.assert_not_called()

        self.assertFalse(app.exception)
        tabs = {tab.label: tab for tab in app.tabs}
        analysis = tabs["Scheduled outlook"]
        self.assertIn("Generated: Nov 21, 2026 at 8:06 PM CST (UTC-06:00)", [caption.value for caption in analysis.caption])
        self.assertFalse(any("Scheduled market outlook unavailable" in caption.value for caption in analysis.caption))
        ranking = next(table for table in analysis.dataframe if "LSTM Return %" in table.value.columns)
        displayed = pa.ipc.open_stream(ranking.proto.styler.display_values).read_all().to_pandas()
        self.assertEqual(displayed["LSTM Return %"].tolist(), ["Unavailable"])
        self.assertEqual(displayed["Forecast Return %"].tolist(), ["-1.25%"])
        self.assertEqual(
            [tab.label for tab in app.tabs],
            ["Portfolio Balance", "AI-Generated Market Summary", "Most Common Patterns", "Market Analysis",
             "Scheduled outlook", "Universe rankings", "Symbol forecast", "Heatmaps"],
        )
        self.assertEqual(len(app.sidebar.button) + len(app.sidebar.text_input) + len(app.sidebar.checkbox), 0)
        self.assertIn("Enter Owner Key", [w.label for w in tabs["Portfolio Balance"].text_input])
        self.assertIn("Run rankings", [w.label for w in tabs["Universe rankings"].button])
        self.assertIn("Run symbol forecast", [w.label for w in tabs["Symbol forecast"].button])
        self.assertIn("Enable automatic paper trades", [w.label for w in tabs["Symbol forecast"].checkbox])
        self.assertIn("Run pattern scan", [w.label for w in tabs["Most Common Patterns"].button])
        self.assertEqual(tabs["Universe rankings"].multiselect[0].value, tabs["Heatmaps"].multiselect[0].value)
        chart = app.tabs[0].get("plotly_chart")[0].proto
        figure = json.loads(chart.spec)
        self.assertEqual([trace["name"] for trace in figure["data"]], ["Equity", "Cash"])
        self.assertEqual(chart.theme, "")
        layout = figure["layout"]
        self.assertEqual(layout["paper_bgcolor"], "#ffffff")
        self.assertEqual(layout["legend"]["font"]["color"], "#0f172a")
        for axis in ("xaxis", "yaxis"):
            self.assertEqual(layout[axis]["title"]["font"]["color"], "#0f172a")
            self.assertEqual(layout[axis]["tickfont"]["color"], "#0f172a")

    def test_rebuild_requests_one_fit_and_keeps_its_cache_generation(self) -> None:
        app = AppTest.from_string(
            "import streamlit as st\n"
            "from market_agent.dashboard_controls import forecast_run_buttons\n"
            "run, force = forecast_run_buttons('Run', 'Rebuild', 'symbol', disabled=False)\n"
            "st.json({'run': run, 'force': force, 'generation': st.session_state.symbol_refresh_nonce})"
        ).run()
        self.assertEqual(json.loads(app.json[0].value), {"run": False, "force": False, "generation": 0})
        app.button(key="symbol_rebuild").click().run()
        state = json.loads(app.json[0].value)
        self.assertTrue(state["run"] and state["force"])
        self.assertGreater(state["generation"], 0)
        app.run()
        self.assertEqual(json.loads(app.json[0].value), {"run": False, "force": False, "generation": state["generation"]})
        app.button(key="symbol_run").click().run()
        self.assertEqual(json.loads(app.json[0].value), {"run": True, "force": False, "generation": state["generation"]})
