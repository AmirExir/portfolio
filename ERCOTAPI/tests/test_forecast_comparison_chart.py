"""Single-axis chart checks for paired forecasts, vintage and DST semantics."""

from __future__ import annotations

import base64
from copy import deepcopy
from datetime import date, datetime, timedelta
from io import BytesIO
import json
import unittest

from PIL import Image
import pandas as pd

from ERCOTAPI import forecast_comparison_chart as chart
from ERCOTAPI.forecast_comparison import compare_forecasts
from ERCOTAPI.tests.test_load_forecast_chart import fixture as official_fixture


def fixture(day: date = date(2026, 11, 13), hour: int = 12, model_type: str = "XGBRegressor") -> dict:
    """Build real comparison semantics around deterministic synthetic source MW."""
    report = official_fixture(day, hour)
    retrieved = datetime.fromisoformat(report["generated_at"])
    issued = retrieved + timedelta(minutes=1)
    captured = issued + timedelta(seconds=1)
    start = captured.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    starts = pd.date_range(start, periods=24, freq="h")
    official = {pd.Timestamp(row["interval_start_utc"]): row["load_mw"] for row in report["hourly_series"]}
    model = pd.DataFrame({"predicted_load_mw": [official[stamp] + (index % 6 - 3) * 10 for index, stamp in enumerate(starts)]}, index=starts)
    snapshot = compare_forecasts(model, report, model_issued_at=issued,
        model_data_cutoff_utc=issued.replace(minute=0, second=0, microsecond=0),
        model_metadata={"model_type": model_type, "method_version": "chart-test-v1", "dataset_sha256": "a" * 64}, now=captured)
    report["comparison"] = {"snapshot": snapshot}
    return report


class ForecastComparisonChartTests(unittest.TestCase):
    def test_exactly_one_axis_has_two_24_hour_steps_and_method_name(self) -> None:
        report = fixture()
        data = chart._prepare_comparison_plot_data(report)
        figure = chart._build_comparison_figure(data)
        try:
            self.assertEqual(len(figure.axes), 1)
            axis = figure.axes[0]
            self.assertEqual(len(axis.patches), 2)
            official, model = [patch.get_data() for patch in axis.patches]
            self.assertEqual(list(official.values), [row["ercot_mw"] for row in report["comparison"]["snapshot"]["rows"]])
            self.assertEqual(list(model.values), [row["model_mw"] for row in report["comparison"]["snapshot"]["rows"]])
            self.assertEqual(len(official.values), 24)
            self.assertEqual(list(official.edges), list(model.edges))
            self.assertAlmostEqual(official.edges[-1] - official.edges[0], 1.0)
            self.assertEqual([text.get_text() for text in axis.get_legend().get_texts()], ["Official ERCOT", "XGBoost · experimental"])
            self.assertEqual(axis.get_ylabel(), "Load (MW)")
            labels = " ".join(text.get_text() for text in figure.texts)
            self.assertIn("not accuracy", labels)
            self.assertNotIn("Your model", labels)
            self.assertNotIn("Seven", labels)
        finally:
            figure.clear()

    def test_png_is_bounded_traceable_and_does_not_mutate_report(self) -> None:
        report = fixture()
        original = deepcopy(report)
        attachment = chart.render_forecast_comparison_chart(report)
        self.assertEqual(original, report)
        self.assertEqual(attachment["mime_type"], "image/png")
        self.assertEqual(attachment["file_name"], "ercot-model-comparison.png")
        self.assertLessEqual(len(attachment["base64"]), chart.MAX_IMAGE_BASE64_LENGTH)
        self.assertLess(len(json.dumps({**report, "chart": attachment})), 700_000)
        with Image.open(BytesIO(base64.b64decode(attachment["base64"], validate=True))) as image:
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.size, (1350, 1080))
            self.assertEqual(image.info["Source"], report["source_url"])
            self.assertEqual(image.info["SourceSHA256"], report["source_sha256"])
            self.assertEqual(image.info["Model"], "XGBoost")
            self.assertEqual(image.info["ModelIssued"], report["comparison"]["snapshot"]["model_issued_at"])

    def test_random_forest_is_named_truthfully(self) -> None:
        data = chart._prepare_comparison_plot_data(fixture(model_type="RandomForestRegressor"))
        self.assertEqual(data.model_name, "Random Forest")

    def test_dst_preserves_24_elapsed_hours_and_repeated_local_hour(self) -> None:
        for day in (date(2026, 3, 8), date(2026, 11, 1)):
            data = chart._prepare_comparison_plot_data(fixture(day, hour=0))
            self.assertEqual(len(data.official), 24)
            self.assertEqual(data.official[-1].end - data.official[0].start, timedelta(hours=24))
            self.assertTrue(all(hour.end - hour.start == timedelta(hours=1) for hour in data.official))
        fall = chart._prepare_comparison_plot_data(fixture(date(2026, 11, 1), hour=0))
        self.assertEqual(fall.official[0].start.astimezone(chart.CENTRAL).hour, 1)
        self.assertEqual(fall.official[1].start.astimezone(chart.CENTRAL).hour, 1)
        self.assertNotEqual(fall.official[0].start.astimezone(chart.CENTRAL).utcoffset(), fall.official[1].start.astimezone(chart.CENTRAL).utcoffset())
        self.assertIn("CDT", chart._interval_label(fall.official[0]))
        self.assertIn("CST", chart._interval_label(fall.official[0]))

    def test_mismatched_vintage_values_or_metadata_are_rejected(self) -> None:
        original = fixture()
        for key, value in (("source_sha256", "b" * 64), ("source_url", "https://www.ercot.com/other"), ("report_name", "other.csv")):
            invalid = deepcopy(original)
            invalid[key] = value
            with self.subTest(field=key), self.assertRaises(chart.ForecastComparisonChartError):
                chart.render_forecast_comparison_chart(invalid)
        invalid = deepcopy(original)
        invalid["comparison"]["snapshot"]["ercot_published_at"] = (datetime.fromisoformat(original["published_at"]) - timedelta(minutes=1)).isoformat()
        with self.assertRaises(chart.ForecastComparisonChartError):
            chart.render_forecast_comparison_chart(invalid)
        invalid = deepcopy(original)
        invalid["comparison"]["snapshot"]["rows"][0]["ercot_mw"] += 1
        with self.assertRaises(chart.ForecastComparisonChartError):
            chart.render_forecast_comparison_chart(invalid)
        # A self-consistent snapshot still cannot substitute different source MW.
        altered_source = deepcopy(original)
        saved = altered_source["comparison"]["snapshot"]
        target = saved["rows"][0]["interval_start_utc"]
        next(row for row in altered_source["hourly_series"] if row["interval_start_utc"] == target)["load_mw"] += 1
        model = pd.DataFrame({"predicted_load_mw": [row["model_mw"] for row in saved["rows"]]},
                             index=pd.to_datetime([row["interval_start_utc"] for row in saved["rows"]], utc=True))
        altered = compare_forecasts(model, altered_source, model_issued_at=saved["model_issued_at"],
            model_data_cutoff_utc=saved["model_data_cutoff_utc"], model_metadata=saved["model_metadata"], now=saved["captured_at"])
        original["comparison"]["snapshot"] = altered
        with self.assertRaisesRegex(chart.ForecastComparisonChartError, "disagree with"):
            chart.render_forecast_comparison_chart(original)

    def test_missing_duplicate_bad_model_or_elapsed_evidence_is_not_filled(self) -> None:
        original = fixture()
        variants = [{key: value for key, value in original.items() if key != "comparison"}]
        for value in (None, True, float("nan"), -1):
            invalid = deepcopy(original)
            invalid["comparison"]["snapshot"]["rows"][0]["model_mw"] = value
            variants.append(invalid)
        missing = deepcopy(original)
        missing["comparison"]["snapshot"]["rows"].pop()
        variants.append(missing)
        duplicate = deepcopy(original)
        duplicate["comparison"]["snapshot"]["rows"][1] = deepcopy(duplicate["comparison"]["snapshot"]["rows"][0])
        variants.append(duplicate)
        malformed = deepcopy(original)
        malformed["comparison"]["snapshot"]["summary"] = None
        variants.append(malformed)
        elapsed = deepcopy(original)
        elapsed["comparison"]["snapshot"]["captured_at"] = elapsed["comparison"]["snapshot"]["rows"][0]["interval_start_utc"]
        variants.append(elapsed)
        for invalid in variants:
            with self.assertRaises(chart.ForecastComparisonChartError):
                chart.render_forecast_comparison_chart(invalid)

    def test_old_actual_cutoff_and_noncontiguous_target_window_are_rejected(self) -> None:
        original = fixture()
        snapshot = original["comparison"]["snapshot"]
        issued = datetime.fromisoformat(snapshot["model_issued_at"])
        snapshot["model_data_cutoff_utc"] = (issued - timedelta(hours=5)).isoformat()
        with self.assertRaisesRegex(chart.ForecastComparisonChartError, "too old"):
            chart.render_forecast_comparison_chart(original)
        # Rebuild a valid domain snapshot over a deliberately skipped hour.
        original = fixture()
        saved = original["comparison"]["snapshot"]
        starts = pd.to_datetime([row["interval_start_utc"] for row in saved["rows"]], utc=True)
        starts = starts[:-1].append(pd.DatetimeIndex([starts[-1] + pd.Timedelta(hours=1)]))
        model = pd.DataFrame({"predicted_load_mw": [row["model_mw"] for row in saved["rows"]]}, index=starts)
        original["comparison"]["snapshot"] = compare_forecasts(model, original,
            model_issued_at=saved["model_issued_at"], model_data_cutoff_utc=saved["model_data_cutoff_utc"],
            model_metadata=saved["model_metadata"], now=saved["captured_at"])
        with self.assertRaisesRegex(chart.ForecastComparisonChartError, "cannot fill"):
            chart.render_forecast_comparison_chart(original)


if __name__ == "__main__":
    unittest.main()
