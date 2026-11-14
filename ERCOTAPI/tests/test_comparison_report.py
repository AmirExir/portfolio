"""Named-method reports retain official evidence and bound Telegram payloads."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
import json
import sqlite3
import unittest
from unittest.mock import patch

import pandas as pd

from ERCOTAPI import comparison_report
from ERCOTAPI.forecast_comparison import compare_forecasts
from ERCOTAPI.tests.test_load_forecast_chart import fixture


def comparison_fixture(model_type: str = "XGBRegressor") -> tuple[dict, dict]:
    report = fixture()
    issued = datetime.fromisoformat(report["generated_at"]) + timedelta(minutes=1)
    rows = [row for row in report["hourly_series"] if datetime.fromisoformat(row["interval_start_utc"]) > issued][:24]
    frame = pd.DataFrame({"predicted_load_mw": [row["load_mw"] + 100 for row in rows]},
                         index=pd.to_datetime([row["interval_start_utc"] for row in rows], utc=True))
    snapshot = compare_forecasts(frame, report, model_issued_at=issued,
        model_data_cutoff_utc=issued.replace(minute=0, second=0), now=issued + timedelta(seconds=1),
        model_metadata={"model_type": model_type, "method_version": "causal-hourly-v1", "dataset_sha256": "a" * 64})
    evaluation = {"status": "awaiting_actuals", "matched_count": 0, "pending_count": 24,
                  "model_metrics": None, "ercot_metrics": None, "rows": [], "model_metadata": snapshot["model_metadata"]}
    return report, {"snapshot_id": "b" * 64, "snapshot": snapshot, "evaluation": evaluation, "recent_evaluations": []}


class ComparisonReportTests(unittest.TestCase):
    def test_peak_interval_labels_keep_midnight_and_repeated_clock_hours_clear(self) -> None:
        for start, end, expected in (
            ("2026-11-15T05:00:00+00:00", "2026-11-15T06:00:00+00:00", "11:00 PM CST–12:00 AM CST (next day)"),
            ("2026-11-01T06:00:00+00:00", "2026-11-01T07:00:00+00:00", "1:00 AM CDT–1:00 AM CST"),
            ("2026-11-14T19:00:00+00:00", "2026-11-14T20:00:00+00:00", "1:00 PM CST–2:00 PM CST"),
        ):
            with self.subTest(start=start):
                rows = [{"interval_start_utc": start, "interval_end_utc": end, "model_mw": 50000.0}]
                self.assertIn(expected, comparison_report._peak_text(rows, "model_mw"))

    def test_success_uses_recorded_method_and_one_same_hour_window(self) -> None:
        for model_type, name in [("XGBRegressor", "XGBoost"), ("RandomForestRegressor", "Random Forest")]:
            with self.subTest(method=name):
                report, comparison = comparison_fixture(model_type)
                original = deepcopy(report)
                with patch.object(comparison_report, "run_and_save_comparison", return_value=comparison):
                    result = comparison_report.attach_model_comparison(report)
                self.assertEqual(report, original)
                self.assertEqual(result["hourly_series"], report["hourly_series"])
                self.assertEqual(result["comparison"]["snapshot"], comparison["snapshot"])
                self.assertTrue(result["text"].startswith(f"ERCOT vs {name} Load Forecast"))
                self.assertIn("24 matching future hours", result["text"])
                self.assertIn(f"Average {name} − ERCOT: +100 MW", result["text"])
                self.assertIn("not accuracy", result["text"])
                self.assertIn("CST", result["text"])
                self.assertIn(report["source_url"], result["text"])
                self.assertNotIn("Daily peaks", result["text"])
                self.assertNotIn("my model", result["text"].lower())
                self.assertLess(len(result["text"]), 3900)
                self.assertLess(len(result["chart_caption"]), 900)
                self.assertIn(name, result["chart_caption"])

    def test_delivery_omits_large_actual_evidence_and_recent_runs(self) -> None:
        report, comparison = comparison_fixture()
        comparison["evaluation"]["rows"] = [{"actual_source": "x" * 1000000}]
        comparison["recent_evaluations"] = [deepcopy(comparison["evaluation"])] * 5
        with patch.object(comparison_report, "run_and_save_comparison", return_value=comparison):
            result = comparison_report.attach_model_comparison(report)
        self.assertNotIn("rows", result["comparison"]["evaluation"])
        self.assertNotIn("recent_evaluations", result["comparison"])
        self.assertLess(len(json.dumps(result)), 100000)

    def test_expected_model_failure_retains_official_report_without_any_image(self) -> None:
        for failure in (ValueError("private source diagnostics"), OSError("private path"), RuntimeError("private model detail"), sqlite3.OperationalError("private db")):
            with self.subTest(error=type(failure).__name__):
                report, _ = comparison_fixture()
                report["chart"] = {"base64": "old official chart"}
                original = deepcopy(report)
                with patch.object(comparison_report, "run_and_save_comparison", side_effect=failure), self.assertLogs(comparison_report.__name__, level="ERROR") as logs:
                    result = comparison_report.attach_model_comparison(report)
                self.assertEqual(report, original)
                self.assertEqual(result["hourly_series"], report["hourly_series"])
                self.assertTrue(result["text"].startswith(report["text"]))
                self.assertIn("no comparison image was substituted", result["text"])
                self.assertEqual(result["chart_error"], "comparison_unavailable")
                self.assertNotIn("chart", result)
                self.assertNotIn("comparison", result)
                self.assertNotIn("private", result["text"])
                self.assertNotIn("private", " ".join(logs.output))

    def test_failed_official_source_never_runs_model(self) -> None:
        report = {"ok": False, "text": "Official forecast unavailable"}
        with patch.object(comparison_report, "run_and_save_comparison") as run:
            self.assertEqual(comparison_report.attach_model_comparison(report), report)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
