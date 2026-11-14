"""Chart semantics and offline dashboard flows for prospective comparisons."""

from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from ERCOTAPI import forecast_comparison_panel as panel
from ERCOTAPI.forecast_comparison import ForecastComparisonStore
from ERCOTAPI.tests.test_forecast_comparison import ISSUED, official_report, snapshot


def fixture() -> tuple[dict, dict, dict]:
    """Use a genuinely validated saved snapshot, with later-vintage chart data."""
    with tempfile.TemporaryDirectory() as directory:
        store = ForecastComparisonStore(Path(directory) / "ledger.sqlite", clock=lambda: ISSUED + timedelta(minutes=6))
        identifier = store.append_snapshot(snapshot())
        saved = store.list_snapshots()[0]
        evaluation = store.evaluate_snapshot(identifier)
    current = official_report()
    current["published_at"] = (ISSUED + timedelta(minutes=30)).isoformat()
    current["generated_at"] = (ISSUED + timedelta(minutes=40)).isoformat()
    current["source_product_url"] = "https://www.ercot.com/forecast-product"
    for row in current["hourly_series"]:
        row["load_mw"] += 1000
    return current, saved, evaluation


def store_stub(saved: list[dict], evaluation: dict) -> Mock:
    store = Mock()
    store.list_snapshots.side_effect = lambda **_: deepcopy(saved)
    store.evaluate_snapshot.return_value = evaluation
    return store


class ForecastPresentationTests(unittest.TestCase):
    def test_streamlit_direct_entrypoint_imports_without_repository_on_pythonpath(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "import forecast_comparison_panel; import actual_load_report; import forecast_comparison_service"],
            cwd=Path(__file__).parents[1], capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_step_chart_uses_real_interval_ends_and_keeps_actual_missing(self) -> None:
        _, saved, _ = fixture()
        rows = [{**row, "actual_mw": None} for row in saved["rows"]]
        rows[0]["actual_mw"] = 120
        chart = panel.hourly_figure(rows, [("model_mw", "Model", "purple"), ("actual_mw", "Actual", "orange")])
        self.assertEqual(list(chart.data[0].y), [110, 190, 190])
        self.assertEqual(pd.Timestamp(chart.data[0].x[-1]), pd.Timestamp(rows[-1]["interval_end_utc"]))
        self.assertEqual(list(chart.data[1].y), [120, None, None])
        self.assertFalse(chart.data[1].connectgaps)
        self.assertEqual(chart.data[0].line.shape, "hv")
        self.assertIn("MW", chart.layout.yaxis.title.text)

    def test_missing_hour_is_a_gap_and_input_order_does_not_change_it(self) -> None:
        _, saved, _ = fixture()
        rows = deepcopy(saved["rows"])
        for key in ("interval_start_utc", "interval_end_utc"):
            rows[1][key] = (pd.Timestamp(rows[1][key]) + pd.Timedelta(hours=1)).isoformat()
        chart = panel.hourly_figure(list(reversed(rows)), [("model_mw", "Model", "purple")])
        self.assertEqual(list(chart.data[0].y), [110, 110, None, 190, 190])
        self.assertEqual(pd.Timestamp(chart.data[0].x[1]), pd.Timestamp(rows[0]["interval_end_utc"]))
        self.assertFalse(chart.data[0].connectgaps)

    def test_repeated_local_hour_has_distinct_offsets_and_utc_positions(self) -> None:
        index = pd.date_range("2026-11-01T06:00Z", periods=2, freq="h")
        rows = [{"interval_start_utc": start.isoformat(), "interval_end_utc": (start + pd.Timedelta(hours=1)).isoformat(), "load_mw": value} for start, value in zip(index, [100, 200])]
        chart = panel.hourly_figure(rows, [("load_mw", "Load", "teal")])
        self.assertEqual(list(chart.data[0].y), [100, 200, 200])
        self.assertIn("01:00 AM CDT", chart.data[0].customdata[0])
        self.assertIn("01:00 AM CST", chart.data[0].customdata[1])
        self.assertEqual(pd.Timestamp(chart.data[0].x[1]) - pd.Timestamp(chart.data[0].x[0]), pd.Timedelta(hours=1))

    def test_accuracy_table_keeps_pending_scores_missing_and_runs_separate(self) -> None:
        _, _, pending = fixture()
        scored = deepcopy(pending)
        scored.update(matched_count=1, pending_count=0, missing_actual_count=1)
        scored["model_metrics"] = {"mae_mw": 10, "rmse_mw": 10, "bias_mw": -10}
        scored["ercot_metrics"] = {"mae_mw": 20, "rmse_mw": 20, "bias_mw": -20}
        table = panel.accuracy_table([pending, scored])
        self.assertEqual(len(table), 2)
        self.assertTrue(pd.isna(table.iloc[0]["Model MAE (MW)"]))
        self.assertTrue(pd.isna(table.iloc[0]["ERCOT RMSE (MW)"]))
        self.assertEqual(table.iloc[0]["Future hours"], 2)
        self.assertEqual(table.iloc[1]["Model bias (MW)"], -10)
        self.assertEqual(table.iloc[1]["Awaiting actuals"], 1)


class ForecastPanelAppTests(unittest.TestCase):
    def assert_no_exception(self, app: AppTest) -> None:
        self.assertFalse(app.exception, [item.message for item in app.exception])

    def assert_saved_vintage_and_pending_accuracy(self, app: AppTest) -> None:
        plots = [json.loads(element.proto.spec) for element in app.get("plotly_chart")]
        saved_plot = next(plot for plot in plots if plot["data"][0]["name"] == "Official ERCOT · saved vintage")
        current_plot = next(plot for plot in plots if plot["data"][0]["name"] == "Official ERCOT")
        self.assertEqual(saved_plot["data"][0]["y"], [100, 200, 200])
        self.assertEqual(saved_plot["data"][1]["y"], [110, 190, 190])
        self.assertEqual(current_plot["data"][0]["y"], [1100, 1200, 1200])
        self.assertEqual(saved_plot["data"][0]["x"], saved_plot["data"][1]["x"])
        self.assertFalse(any(trace["name"] == "Observed actual load" for trace in saved_plot["data"]))
        accuracy = next(item.value for item in app.dataframe if "Future hours" in item.value.columns)
        self.assertTrue(pd.isna(accuracy.iloc[0]["Model MAE (MW)"]))
        self.assertEqual(accuracy.iloc[0]["Scored hours"], 0)
        self.assertEqual(accuracy.iloc[0]["Future hours"], 2)

    def test_forecast_view_is_available_without_authenticated_api_or_news(self) -> None:
        from ERCOTAPI import ercotapi

        current, saved, evaluation = fixture()
        app = AppTest.from_string("from ERCOTAPI.ercotapi import main\nmain()\n", default_timeout=30)
        app.query_params["view"] = "forecasts"
        with patch.object(panel, "official_forecast", return_value=current), patch.object(panel, "ForecastComparisonStore", return_value=store_stub([saved], evaluation)), patch.object(ercotapi, "ErcotAPI", side_effect=AssertionError("Authenticated API should not be used")) as api, patch.object(ercotapi, "get_latest_news_by_prefix", side_effect=AssertionError("News should not be fetched")) as news:
            app.run()
        self.assert_no_exception(app)
        api.assert_not_called()
        news.assert_not_called()
        self.assertIn("Load Forecast Comparison", [item.value for item in app.subheader])
        self.assert_saved_vintage_and_pending_accuracy(app)

    def test_run_button_loads_persisted_comparison_and_keeps_latest_vintage_separate(self) -> None:
        current, saved, evaluation = fixture()
        saved_runs = []
        store = store_stub(saved_runs, evaluation)
        bundle = {"history": pd.DataFrame({"load": [100.0]}, index=pd.date_range(ISSUED - timedelta(hours=1), periods=1, tz="UTC")), "metadata": {"warnings": [], "retrieved_at": ISSUED.isoformat(), "latest_interval_end_utc": ISSUED.isoformat()}}

        def save(*args, **kwargs) -> dict:
            saved_runs.append(saved)
            return {"snapshot": saved, "evaluation": evaluation}

        app = AppTest.from_string("from ERCOTAPI.forecast_comparison_panel import render_forecast_comparison\nrender_forecast_comparison()\n", default_timeout=30)
        with patch.object(panel, "official_forecast", return_value=current), patch.object(panel, "ForecastComparisonStore", return_value=store), patch.object(panel, "actual_history", return_value=bundle), patch.object(panel, "run_and_save_comparison", side_effect=save) as run:
            app.run()
            self.assert_no_exception(app)
            next(button for button in app.button if button.label == "Run my model vs ERCOT").click().run()
        self.assert_no_exception(app)
        run.assert_called_once()
        self.assertIs(run.call_args.args[0], current)
        self.assertIs(run.call_args.kwargs["store"], store)
        self.assertTrue(any("Saved 2 matching future hours" in item.value for item in app.success))
        self.assert_saved_vintage_and_pending_accuracy(app)

    def test_failed_run_or_refresh_preserves_saved_comparison(self) -> None:
        for button_label in ("Run my model vs ERCOT", "Update accuracy scores"):
            with self.subTest(action=button_label):
                current, saved, evaluation = fixture()
                app = AppTest.from_string("from ERCOTAPI.forecast_comparison_panel import render_forecast_comparison\nrender_forecast_comparison()\n", default_timeout=30)
                with patch.object(panel, "official_forecast", return_value=current), patch.object(panel, "ForecastComparisonStore", return_value=store_stub([saved], evaluation)), patch.object(panel, "actual_history", side_effect=ValueError("Current actuals unavailable")):
                    app.run()
                    self.assert_no_exception(app)
                    next(button for button in app.button if button.label == button_label).click().run()
                self.assert_no_exception(app)
                self.assertTrue(any("Previously saved runs remain available" in item.value for item in app.error))
                self.assert_saved_vintage_and_pending_accuracy(app)


if __name__ == "__main__":
    unittest.main()
