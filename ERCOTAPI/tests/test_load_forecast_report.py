"""Offline tests of official forecast provenance, calendar coverage and output."""

from __future__ import annotations

from contextlib import redirect_stdout
import csv
from datetime import date, datetime, timedelta, timezone
from io import BytesIO, StringIO
import json
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from ERCOTAPI import load_forecast_report as report


def forecast_csv(start: date = date(2026, 11, 13), days: int = 7) -> str:
    """Return known synthetic MW on explicit regular/spring/fall day labels."""

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["DeliveryDate", "HourEnding", "SystemTotal", "DSTFlag"])
    for offset in range(days):
        day = start + timedelta(days=offset)
        labels = [(hour, "N") for hour in range(1, 25)]
        if day == date(2026, 3, 8):
            labels.remove((3, "N"))
        if day == date(2026, 11, 1):
            labels.insert(2, (2, "Y"))
        for hour, flag in labels:
            mw = 10000 + offset * 100 + hour
            writer.writerow([day.strftime("%m/%d/%Y"), f"{hour}:00", mw, flag])
    return output.getvalue()


def vintage(published: datetime | None = None) -> report.ReportDownload:
    """Return source metadata without a network call."""

    return report.ReportDownload(
        "cdr.00012312.0000000000000000.20261113.120000123.LFCWEATHERNP3561_csv.zip",
        published or datetime(2026, 11, 13, 12, tzinfo=report.CENTRAL),
        "https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId=123",
    )


class LoadForecastReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 11, 13, 12, 15, tzinfo=report.CENTRAL)

    def test_known_peak_and_next_24_complete_intervals(self) -> None:
        payload = report.build_forecast_report(vintage(), forecast_csv(), now=self.now)
        self.assertTrue(payload["ok"])
        self.assertEqual(len(payload["daily_peaks"]), 7)
        self.assertEqual(payload["daily_peaks"][0]["peak_load_mw"], 10024)
        self.assertEqual(payload["daily_peaks"][0]["hour_ending"], 24)
        # Next full intervals start at13:00 today through12:00 tomorrow.
        self.assertEqual(payload["next_24_hour_peak"]["peak_load_mw"], 10113)
        self.assertEqual(payload["next_24_hour_peak"]["hour_ending"], 13)
        self.assertIn("Fri, Nov 13, 2026", payload["text"])
        self.assertIn("12:00 PM CST", payload["text"])
        self.assertIn("10,024 MW", payload["text"])
        self.assertIn("Official ERCOT forecast", payload["text"])
        self.assertIn(vintage().url, payload["text"])
        self.assertEqual(len(payload["source_sha256"]), 64)
        self.assertLess(len(payload["text"]), 3900)

    def test_latest_csv_archive_is_selected_independent_of_order(self) -> None:
        def row(clock: str, doc: int, suffix: str = "csv") -> str:
            return (
                f"<tr><td>cdr.00012312.0000.20261113.{clock}.LFCWEATHERNP3561_{suffix}.zip</td>"
                f"<td><a href='/misdownload/servlets/mirDownload?doclookupId={doc}'>zip</a></td></tr>"
            )
        selected = report.select_latest_report(row("120000001", 1) + row("130000001", 3, "xml") + row("123000001", 2))
        self.assertTrue(selected.url.endswith("=2"))
        self.assertEqual(selected.published_at.hour, 12)
        self.assertEqual(selected.published_at.minute, 30)

    def test_ambiguous_latest_publication_does_not_fall_back(self) -> None:
        html = "".join(
            f"<tr><td>cdr.00012312.0000.20261101.{clock}.LFCWEATHERNP3561_csv.zip</td>"
            f"<td><a href='/misdownload/servlets/mirDownload?doclookupId={index}'>zip</a></td></tr>"
            for index, clock in enumerate(("003000000", "013000000"))
        )
        with self.assertRaisesRegex(report.ForecastReportError, "ambiguous"):
            report.select_latest_report(html)

    def test_invalid_newest_link_does_not_fall_back_to_an_older_vintage(self) -> None:
        html = (
            "<tr><td>cdr.00012312.0000.20261113.120000000.LFCWEATHERNP3561_csv.zip</td>"
            "<td><a href='/misdownload/servlets/mirDownload?doclookupId=1'>zip</a></td></tr>"
            "<tr><td>cdr.00012312.0000.20261113.130000000.LFCWEATHERNP3561_csv.zip</td>"
            "<td><a href='https://example.com/forecast.zip'>zip</a></td></tr>"
        )
        with self.assertRaises(report.ForecastReportError) as caught:
            report.select_latest_report(html)
        self.assertEqual(caught.exception.code, "invalid_download_link")

    def test_stale_and_future_publications_fail_closed(self) -> None:
        for published, code in (
            (self.now - timedelta(hours=4), "stale_report"),
            (self.now + timedelta(minutes=6), "future_report"),
        ):
            with self.subTest(code=code), self.assertRaises(report.ForecastReportError) as caught:
                report.build_forecast_report(vintage(published), forecast_csv(), now=self.now)
            self.assertEqual(caught.exception.code, code)

    def test_missing_day_and_missing_hour_are_not_summarized(self) -> None:
        for text in (forecast_csv(days=6), forecast_csv().replace("11/14/2026,3:00,10103,N\r\n", "")):
            with self.assertRaises(report.ForecastReportError) as caught:
                report.build_forecast_report(vintage(), text, now=self.now)
            self.assertEqual(caught.exception.code, "incomplete_forecast")

    def test_invalid_numeric_duplicate_and_changed_schema_are_rejected(self) -> None:
        original = forecast_csv()
        variants = [
            original.replace("10001,N", "nan,N", 1),
            original.replace("10001,N", "-1,N", 1),
            original + original.splitlines()[1] + "\n",
            original.replace("SystemTotal", "OtherColumn", 1),
            original.replace("SystemTotal,DSTFlag", "SystemTotal,DSTFlag,SystemTotal", 1),
        ]
        for text in variants:
            with self.subTest(first_line=text.splitlines()[0]), self.assertRaises(report.ForecastReportError):
                report.parse_forecast_csv(text)

    def test_spring_and_fall_days_use_23_and_25_source_hours(self) -> None:
        for start, expected_rows in ((date(2026, 3, 8), 23), (date(2026, 11, 1), 25)):
            now = datetime.combine(start, datetime.min.time(), report.CENTRAL) + timedelta(minutes=15)
            csv_text = forecast_csv(start)
            rows = report.parse_forecast_csv(csv_text)[start]
            payload = report.build_forecast_report(vintage(now), csv_text, now=now)
            self.assertEqual(len(rows), expected_rows)
            self.assertTrue(payload["ok"])
            starts = sorted(row["interval_start"] for row in rows)
            self.assertTrue(all(second - first == timedelta(hours=1) for first, second in zip(starts, starts[1:])))

    def test_unsupported_dst_labels_are_not_silently_repaired(self) -> None:
        csv_text = forecast_csv(date(2026, 11, 1)).replace("11/01/2026,2:00,10002,Y", "11/01/2026,25:00,10002,Y")
        with self.assertRaises(report.ForecastReportError):
            report.parse_forecast_csv(csv_text)

    def test_fetch_reads_single_archive_and_emits_fresh_source_metadata(self) -> None:
        archive = BytesIO()
        with ZipFile(archive, "w") as zipped:
            zipped.writestr("forecast.csv", forecast_csv())
        with patch.object(report, "_download", side_effect=[b"index", archive.getvalue()]) as download, patch.object(report, "select_latest_report", return_value=vintage()):
            payload = report.fetch_forecast_report(now=self.now)
        self.assertEqual(download.call_count, 2)
        self.assertEqual(payload["generated_at"], self.now.astimezone(timezone.utc).isoformat())

    def test_expected_failure_prints_only_sanitized_json(self) -> None:
        stream = StringIO()
        with patch.object(report, "fetch_forecast_report", side_effect=report.ForecastReportError("stale_report", "Latest report is stale.")), redirect_stdout(stream):
            report.main()
        payload = json.loads(stream.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_code"], "stale_report")
        self.assertIn("No estimates were substituted", payload["text"])

    def test_external_source_override_is_rejected_before_network(self) -> None:
        with patch.object(report.requests, "get") as get, self.assertRaises(report.ForecastReportError):
            report._download("https://example.com/forecast.zip")
        get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
