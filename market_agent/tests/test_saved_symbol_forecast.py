"""Regression coverage for display-only restoration of historical symbol outputs."""

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from market_agent.forecast_cache import forecast_cache_path
from market_agent.saved_rankings import dashboard_result_path
from market_agent.saved_symbol_forecast import attach_archived_models, choose_saved_forecast, historical_accuracy_path


def _report() -> dict:
    return {
        "generated_at": "2026-11-22T01-59-38Z", "horizon_days": 30, "run_complete": True,
        "rows": [
            {"Symbol": "BTC", "As Of Session": "2026-11-21", "Target Session": "2026-12-21",
             "Selected Model": "Ridge", "Forecast Price": 111.0, "Forecast Return %": 11.0},
        ],
        "short_horizon_reports": [
            {"horizon_days": 1, "rows": [
                {"Symbol": "BTC", "As Of Session": "2026-11-21", "Target Session": "2026-11-22",
                 "Selected Model": "Ridge", "Forecast Price": 99.0, "Forecast Return %": -1.0},
            ]},
        ],
    }


def _model_cache() -> dict:
    return {
        "generated_at": "2026-11-22T02:00:00+00:00", "horizon_days": 30, "symbols": ["BTC-USD"],
        "primary_model": "Ensemble", "snapshot": {
            "symbol": "BTC-USD", "last_price": 100.0, "as_of_session": "2026-11-21",
            "models": {"Ridge": {"forecast": [{"date": "2026-12-21", "forecast_close": 111.0}],
                                 "metrics": {"forecast_change_pct": 11.0, "confidence_pct": 70,
                                             "expected_error_pct": 5, "probability_up_pct": 70,
                                             "selected_lookback_window": 12}}},
        },
    }


class SavedSymbolForecastTests(unittest.TestCase):
    def test_historical_accuracy_key_identifies_the_model_and_input_history(self) -> None:
        config = {
            "history_days": 1825, "horizon_days": 30, "lookback_window": 20, "ridge_alpha": 10,
            "use_market_context": True, "data_fingerprint": {"end": "2026-11-21", "hash": "original"},
            "context_fingerprint": {"market": {"hash": "context"}},
        }
        def key(settings: dict, model: str = "Ridge", points: int = 50) -> Path | None:
            return historical_accuracy_path(".", settings, "BTC-USD", model, points)

        original = key(config)
        self.assertIsNotNone(original)
        self.assertEqual(original, key({**config, "ridge_alpha": 10.0, "context_fingerprint": {"hash": "context"}}))
        self.assertNotEqual(original, key(config, points=10))
        self.assertNotEqual(original, key(config, model="Ensemble"))
        for changed in [
            {"data_fingerprint": {"end": "2026-11-20", "hash": "original"}},
            {"data_fingerprint": {"end": "2026-11-21", "hash": "revised"}},
            {"context_fingerprint": {"hash": "revised context"}}, {"use_market_context": False},
            {"history_days": 730}, {"horizon_days": 1}, {"lookback_window": 10}, {"ridge_alpha": 2},
        ]:
            with self.subTest(changed=changed):
                self.assertNotEqual(original, key({**config, **changed}))
        self.assertIsNone(key({**config, "data_fingerprint": None}))

    def test_direct_horizons_and_exact_symbol(self) -> None:
        report = _report()
        latest = choose_saved_forecast({}, report, "BTC-USD", "BTC", 30, "Best Validation")
        short = choose_saved_forecast({}, report, "BTC-USD", "BTC", 1, "Best Validation")
        self.assertEqual((latest.row["Forecast Price"], short.row["Forecast Price"]), (111.0, 99.0))
        self.assertEqual(short.row["Target Session"], "2026-11-22")
        self.assertIsNone(choose_saved_forecast({}, report, "ETH-USD", "ETH", 30, "Best Validation"))
        self.assertIsNone(choose_saved_forecast({}, report, "BTC-USD", "BTC", 7, "Best Validation"))
        report["short_horizon_reports"] = []
        self.assertIsNone(choose_saved_forecast({}, report, "BTC-USD", "BTC", 1, "Best Validation"))
        report["run_complete"] = False
        self.assertIsNone(choose_saved_forecast({}, report, "BTC-USD", "BTC", 30, "Best Validation"))

    def test_interactive_snapshot_restoration_uses_latest_observed_session(self) -> None:
        local, report = _model_cache(), _report()
        result = choose_saved_forecast(local, report, "BTC-USD", "BTC", 30, "Best Validation")
        self.assertEqual(result.source, "Interactive run")
        self.assertEqual(result.row["Selected Model"], "Ridge")
        self.assertEqual(result.models["Ridge"]["metrics"]["selected_lookback_window"], 12)
        local["snapshot"]["as_of_session"] = "2026-11-20"
        self.assertEqual(choose_saved_forecast(local, report, "BTC-USD", "BTC", 30, "Ridge").source, "Scheduled report")
        self.assertIsNone(choose_saved_forecast(local, {}, "AAPL", "AAPL", 30, "Ridge"))
        local["snapshot"]["models"]["Ridge"]["metrics"] = {}
        partial = choose_saved_forecast(local, {}, "BTC-USD", "BTC", 30, "Ridge")
        self.assertIsNone(partial.row["Forecast Return %"])
        self.assertIsNone(partial.row["Model Call"])

    def test_local_report_path_attaches_only_a_matching_endpoint(self) -> None:
        report = _report()
        name = "ml_forecast_rankings_2026-11-22T01-59-38Z.json"
        report["paths"] = {"json": "/old/machine/" + name}
        archive = {**report, "snapshots": [_model_cache()["snapshot"]]}
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / name).write_text(json.dumps(archive))
            result = choose_saved_forecast({}, report, "BTC-USD", "BTC", 30, "Ridge")
            attach_archived_models(result, report, directory, "BTC-USD", 30)
            self.assertIn("Ridge", result.models)
            report["rows"][0]["Forecast Price"] = 100.0
            result = choose_saved_forecast({}, report, "BTC-USD", "BTC", 30, "Ridge")
            attach_archived_models(result, report, directory, "BTC-USD", 30)
            self.assertFalse(result.models)

    def test_view_renders_dates_path_and_fitted_settings(self) -> None:
        script = (
            "from market_agent.saved_symbol_forecast import choose_saved_forecast, render_saved_symbol_forecast\n"
            f"result = choose_saved_forecast({_model_cache()!r}, {{}}, 'BTC-USD', 'BTC', 30, 'Best Validation')\n"
            "render_saved_symbol_forecast(result, 'BTC-USD', 30)"
        )
        app = AppTest.from_string(script).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertIn("Saved 30-session forecast", app.subheader[0].value)
        self.assertIn("Nov 21, 2026", app.caption[0].value)
        self.assertIn("Dec 21, 2026", app.caption[0].value)
        self.assertTrue(any("Lag (sessions)" in table.value.columns for table in app.dataframe))
        chart = json.loads(app.get("arrow_vega_lite_chart")[0].proto.spec)
        self.assertEqual(chart["mark"]["type"], "circle")
        self.assertEqual(chart["encoding"]["y"]["title"], "Forecast price (USD)")

    def test_dashboard_restores_saved_runs_if_price_feed_fails_without_model_fits(self) -> None:
        source = Path(__file__).resolve().parents[1] / "market_agent_app.py"
        report = _report()
        report.update(symbols=["BTC-USD"], short_horizon_reports=[])
        short = _report()["short_horizon_reports"][0]["rows"][0]
        short["Symbol"] = "BTC-USD"
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(sys, "path", [str(source.parent), *sys.path]),
            patch("agent.data.get_ohlcv", side_effect=ValueError("fixture price feed offline")),
            patch("agent.forecast.compare_forecast_models") as fit,
            patch("agent.forecast.backtest_forecasts") as backtest,
            patch("requests.get", return_value=Mock(status_code=404)),
            patch("yfinance.download", return_value=pd.DataFrame()),
        ):
            st.cache_data.clear()
            self.addCleanup(st.cache_data.clear)
            path = Path(temporary) / source.name
            path.write_text(source.read_text())
            directory = path.parent / "reports"
            directory.mkdir()
            (directory / "ml_forecast_rankings_latest.json").write_text(json.dumps(report))
            cache = forecast_cache_path(directory, ("BTC-USD",), 1825, 1, 20, 10., True, True, "off", False)
            view = dashboard_result_path(directory, "rankings", (cache.stem, "Best Validation"))
            view.parent.mkdir()
            record = {"generated_at": "2026-11-22T01:57:00Z", "horizon_days": 1, "history_days": 1825,
                      "lookback_window": 20, "ridge_alpha": 10.0, "use_market_context": True,
                      "data_fingerprint": {"end": "2026-11-21", "hash": "original"},
                      "context_fingerprint": {"market": {"hash": "context"}},
                      "symbols": ["BTC-USD"], "primary_model": "Best Validation", "rows": [short]}
            cache.write_text(json.dumps(record))
            view.write_text(json.dumps(record))
            history = historical_accuracy_path(str(directory), record, "BTC-USD", "Ridge", 50)
            history.write_text(json.dumps({"generated_at": "2026-11-22T01:58:00Z", "rows": [
                {"forecast_date": "2026-11-01", "actual_close": 99.75},
            ]}))
            app = AppTest.from_file(str(path)).run(timeout=60)
            app.selectbox(key="symbol_select").select("BTC-USD").run(timeout=60)
            self.assertFalse(app.exception)
            self.assertTrue(any("Saved interactive ranking" in cap.value for cap in app.caption))
            for horizon in (30, 1):
                self.assertIn(f"BTC-USD · Saved {horizon}-session forecast", [h.value for h in app.subheader])
            self.assertTrue(any("Walk-forward test completed" in cap.value for cap in app.caption))
            history_tables = [table.value for table in app.dataframe if "Actual Close" in table.value]
            self.assertEqual(len(history_tables), 1)
            self.assertEqual(history_tables[0]["Actual Close"].tolist(), [99.75])
            record["data_fingerprint"]["hash"] = "revised"
            cache.write_text(json.dumps(record))
            app.button(key="symbol_run").click().run(timeout=60)
            self.assertIn("BTC-USD · Saved 30-session forecast", [h.value for h in app.subheader])
            self.assertFalse(any("Walk-forward test completed" in cap.value for cap in app.caption))
            fit.assert_not_called()
            backtest.assert_not_called()


if __name__ == "__main__":
    unittest.main()
