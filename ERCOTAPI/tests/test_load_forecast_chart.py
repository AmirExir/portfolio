"""Offline chart checks for interval semantics, evidence and bounded PNG output."""

from __future__ import annotations

import base64
import copy
from datetime import date, datetime, timedelta
from io import BytesIO
import json
import unittest

from PIL import Image

from ERCOTAPI import load_forecast_chart as chart
from ERCOTAPI import load_forecast_report as report
from ERCOTAPI.tests.test_load_forecast_report import forecast_csv, vintage


def fixture(day: date = date(2026, 11, 13), hour: int = 12) -> dict:
    """Build a known synthetic report through the same source-calendar parser."""
    now = datetime(day.year, day.month, day.day, hour, 15, tzinfo=report.CENTRAL)
    return report.build_forecast_report(vintage(now), forecast_csv(day), now=now)


class LoadForecastChartTests(unittest.TestCase):
    def test_png_contains_provenance_and_fits_workflow_budget_without_mutation(self) -> None:
        payload = fixture()
        original = copy.deepcopy(payload)
        attachment = chart.render_load_forecast_chart(payload)
        self.assertEqual(payload, original)
        self.assertEqual(attachment["mime_type"], "image/png")
        self.assertEqual(attachment["file_name"], "ercot-load-forecast.png")
        self.assertLessEqual(len(attachment["base64"]), chart.MAX_IMAGE_BASE64_LENGTH)
        self.assertLess(len(json.dumps({**payload, "chart": attachment})), 700_000)
        with Image.open(BytesIO(base64.b64decode(attachment["base64"], validate=True))) as image:
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.size, (1120, 1358))
            self.assertEqual(image.info["Source"], payload["source_url"])
            self.assertEqual(image.info["Report"], payload["report_name"])
            self.assertIn("forecast", image.info["Title"])

    def test_detail_excludes_past_and_partial_hours_and_preserves_each_step(self) -> None:
        data = chart._prepare_plot_data(fixture())
        self.assertEqual(data.upcoming[0].start.astimezone(report.CENTRAL).hour, 13)
        self.assertEqual(len(data.upcoming[:48]), 48)
        figure = chart._build_figure(data)
        try:
            self.assertEqual(len(figure.axes), 2)
            detail, week = figure.axes
            detail_steps = detail.patches[0].get_data()
            self.assertEqual(list(detail_steps.values), [hour.mw for hour in data.upcoming[:48]])
            self.assertEqual(len(detail_steps.edges), 49)
            self.assertAlmostEqual(detail_steps.edges[-1] - detail_steps.edges[0], 2.0)
            week_steps = [patch.get_data() for patch in week.patches if hasattr(patch, "get_data")]
            self.assertEqual([value for step in week_steps for value in step.values],
                             [hour.mw for hour in data.hours])
            self.assertEqual(detail.get_ylabel(), "Load (MW)")
            self.assertEqual(week.get_ylabel(), "Load (MW)")
            self.assertTrue(any("24/48-hour views" in text.get_text() for text in figure.texts))
        finally:
            figure.clear()

    def test_dst_windows_use_actual_hours_and_unambiguous_peak_labels(self) -> None:
        for day, count in ((date(2026, 3, 8), 167), (date(2026, 11, 1), 169)):
            with self.subTest(day=day):
                data = chart._prepare_plot_data(fixture(day, hour=0))
                self.assertEqual(len(data.hours), count)
                self.assertEqual(len(data.daily_peaks), 7)
                self.assertTrue(all(hour.end - hour.start == timedelta(hours=1) for hour in data.hours))
                self.assertEqual(len(data.upcoming[:48]), 48)
        fall = chart._prepare_plot_data(fixture(date(2026, 11, 1), hour=0))
        repeated = next(hour for hour in fall.hours if hour.start.astimezone(report.CENTRAL).hour == 1
                        and hour.start.astimezone(report.CENTRAL).utcoffset() != hour.end.astimezone(report.CENTRAL).utcoffset())
        label = chart._interval_label(repeated)
        self.assertIn("CDT", label)
        self.assertIn("CST", label)

    def test_no_missing_or_malformed_hours_are_filled(self) -> None:
        original = fixture()
        variants = []
        missing = copy.deepcopy(original)
        missing["hourly_series"].pop(10)
        variants.append(missing)
        duplicate = copy.deepcopy(original)
        duplicate["hourly_series"][10] = copy.deepcopy(duplicate["hourly_series"][9])
        variants.append(duplicate)
        naive = copy.deepcopy(original)
        naive["hourly_series"][0]["interval_start_utc"] = "2026-11-13T06:00:00"
        variants.append(naive)
        for bad_value in (None, True, float("nan"), -1):
            invalid = copy.deepcopy(original)
            invalid["hourly_series"][0]["load_mw"] = bad_value
            variants.append(invalid)
        for invalid in variants:
            with self.assertRaises(chart.LoadForecastChartError):
                chart.render_load_forecast_chart(invalid)

    def test_chart_refuses_inconsistent_peaks_units_or_source(self) -> None:
        original = fixture()
        for field, bad_value in (("units", "GW"), ("ok", False),
                                 ("forecast_kind", "model_output"), ("source_url", "https://example.com/forecast")):
            invalid = {**original, field: bad_value}
            with self.subTest(field=field), self.assertRaises(chart.LoadForecastChartError):
                chart.render_load_forecast_chart(invalid)
        for key in ("daily_peaks", "next_24_hour_peak", "week_peak"):
            invalid = copy.deepcopy(original)
            peak = invalid[key][0] if key == "daily_peaks" else invalid[key]
            peak["peak_load_mw"] += 1000
            with self.subTest(peak=key), self.assertRaises(chart.LoadForecastChartError):
                chart.render_load_forecast_chart(invalid)


if __name__ == "__main__":
    unittest.main()
