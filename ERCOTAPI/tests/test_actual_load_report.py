"""Offline regressions for official actual-load provenance and hourly integrity."""

from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
from io import BytesIO, StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from ERCOTAPI import actual_load_report as report


def daily_csv(day: date) -> str:
    """Known source labels with independent spring/fall calendar expectations."""
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["OperDay", "HourEnding", "TOTAL", "DSTFlag"])
    labels = [(hour, "N") for hour in range(1, 25)]
    if day == date(2026, 3, 8):
        labels.remove((3, "N"))
    if day == date(2026, 11, 1):
        labels.insert(2, (2, "Y"))
    for index, (hour, flag) in enumerate(labels):
        writer.writerow([day.strftime("%m/%d/%Y"), f"{hour:02}:00", 40000 + index * 10, flag])
    return output.getvalue()


def archive(body: str, name: str = "actual.csv") -> bytes:
    output = BytesIO()
    with ZipFile(output, "w") as zipped:
        zipped.writestr(name, body)
    return output.getvalue()


def index_row(day: date, doc: int, clock: str = "055000123", link: str | None = None) -> str:
    link = link or f"/misdownload/servlets/mirDownload?doclookupId={doc}"
    return f"<tr><td>cdr.00013101.0000.{day:%Y%m%d}.{clock}.ACTUALSYSLOADWZNP6345_csv.zip</td><td><a href='{link}'>zip</a></td></tr>"


def display(day: date, *, hours: int = 11, published_hour: int | None = None) -> str:
    published = datetime.combine(day, datetime.min.time()).replace(hour=hours if published_hour is None else published_hour, minute=20)
    header = "<tr>" + "".join(f"<td>{name}</td>" for name in report.DISPLAY_COLUMNS) + "</tr>"
    rows = []
    for hour in range(1, hours + 1):
        cells = [f"{day:%m/%d/%Y}", f"{hour:02}00", *(["5000.00"] * 8), str(45000 + hour)]
        rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in cells) + "</tr>")
    return f"<div>Last Date and Time: {published:%b %d, %Y %H:%M}</div><table>{header}{''.join(rows)}</table>"


class ActualLoadReportTests(unittest.TestCase):
    def test_transient_source_failure_retries_then_uses_verified_download(self) -> None:
        response = report.requests.Response()
        response.status_code = 503
        failure = report.ForecastReportError("source_unavailable", "Source unavailable")
        failure.__cause__ = report.requests.HTTPError(response=response)
        with patch.object(report, "_download", side_effect=[failure, b"source bytes"]) as download, patch.object(report.time, "sleep") as sleep:
            self.assertEqual(report._fetch(report.DEFAULT_INDEX_URL), b"source bytes")
        self.assertEqual(download.call_count, 2)
        sleep.assert_called_once_with(0.5)

    def test_source_errors_are_bounded_and_identify_the_failed_source(self) -> None:
        for status, attempts, retry_after in [(403, 1, None), (503, 3, None), (429, 1, "120")]:
            with self.subTest(status=status):
                response = report.requests.Response()
                response.status_code = status
                if retry_after:
                    response.headers["Retry-After"] = retry_after
                failure = report.ForecastReportError("source_unavailable", "Source unavailable")
                failure.__cause__ = report.requests.HTTPError(response=response)
                with patch.object(report, "_download", side_effect=failure) as download, patch.object(report.time, "sleep"):
                    with self.assertRaisesRegex(report.ActualLoadReportError, f"actual-load report index.*HTTP {status}"):
                        report._fetch(report.DEFAULT_INDEX_URL)
                self.assertEqual(download.call_count, attempts)

    def setUp(self) -> None:
        self.now = datetime(2026, 11, 14, 11, 30, tzinfo=report.CENTRAL)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name)
        self.environment = patch.dict("os.environ", {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def sources(self, now: datetime | None = None) -> dict[str, bytes]:
        now = now or self.now
        today = now.date()
        return {
            report.DEFAULT_INDEX_URL: (index_row(today, 2) + index_row(today - timedelta(days=1), 1)).encode(),
            "https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId=1": archive(daily_csv(today - timedelta(days=2))),
            "https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId=2": archive(daily_csv(today - timedelta(days=1))),
            report.DEFAULT_DISPLAY_URL: display(today).encode(),
        }

    def fetch(self, sources: dict[str, bytes] | None = None, now: datetime | None = None) -> dict:
        with patch.object(report, "_fetch", side_effect=(sources or self.sources()).__getitem__):
            return report.fetch_actual_load_history(now=now or self.now, history_days=2, cache_dir=self.cache)

    def test_complete_daily_hours_have_interval_start_mw_and_dst_offsets(self) -> None:
        regular = report.parse_actual_load_csv(daily_csv(date(2026, 11, 13)))
        self.assertEqual(len(regular), 24)
        self.assertEqual(str(regular.index.tz), "America/Chicago")
        self.assertEqual(regular.index[0].isoformat(), "2026-11-13T00:00:00-06:00")
        self.assertEqual(regular.index[-1].isoformat(), "2026-11-13T23:00:00-06:00")
        self.assertEqual(regular["load"].iloc[0], 40000)
        for day, count in ((date(2026, 3, 8), 23), (date(2026, 11, 1), 25)):
            with self.subTest(day=day):
                frame = report.parse_actual_load_csv(daily_csv(day))
                self.assertEqual(len(frame), count)
                self.assertTrue((frame.index.to_series().diff().iloc[1:] == timedelta(hours=1)).all())
                self.assertFalse(frame.index.has_duplicates)
        fall = report.parse_actual_load_csv(daily_csv(date(2026, 11, 1)))
        self.assertEqual(fall.index[1].isoformat(), "2026-11-01T01:00:00-05:00")
        self.assertEqual(fall.index[2].isoformat(), "2026-11-01T01:00:00-06:00")
        self.assertNotEqual(fall["load"].iloc[1], fall["load"].iloc[2])

    def test_daily_missing_duplicate_nonfinite_and_changed_schema_fail(self) -> None:
        body = daily_csv(date(2026, 11, 13))
        lines = body.splitlines()
        invalid = [
            "\n".join([lines[0], *lines[2:]]), body + lines[1],
            body.replace("40000", "NaN", 1), body.replace("40000", "-1", 1),
            body.replace("40000", "inf", 1), body.replace("TOTAL", "OTHER", 1),
            body.replace("TOTAL,DSTFlag", "TOTAL,DSTFlag,TOTAL", 1),
            body.replace("01:00,40000,N", "01:00,40000,Y", 1),
        ]
        for text in invalid:
            with self.subTest(first_line=text[:80]), self.assertRaises(report.ActualLoadReportError):
                report.parse_actual_load_csv(text)
        with self.assertRaisesRegex(report.ActualLoadReportError, "expected operating day"):
            report.parse_actual_load_csv(body, expected_day=date(2026, 11, 12))

    def test_current_display_uses_real_complete_hours_and_publication(self) -> None:
        frame, meta = report.parse_current_day_display(display(self.now.date()), now=self.now)
        self.assertEqual(len(frame), 11)
        self.assertEqual(frame.index[-1].isoformat(), "2026-11-14T10:00:00-06:00")
        self.assertEqual(meta["published_at_utc"], "2026-11-14T17:20:00+00:00")
        self.assertEqual(frame["load"].iloc[-1], 45011)

    def test_current_display_gaps_duplicates_future_and_stale_are_rejected(self) -> None:
        body = display(self.now.date())
        invalid = [
            body.replace("<td>0200</td>", "<td>0100</td>"),
            body.replace("<td>0200</td>", "<td>0300</td>"),
            body.replace("<td>TOTAL</td>", "<td>Other</td>"),
            body.replace("45011", "NaN"), display(self.now.date(), hours=12, published_hour=11),
            display(self.now.date(), hours=10, published_hour=11),
            display(self.now.date(), hours=7), display(self.now.date() - timedelta(days=1)),
        ]
        for text in invalid:
            with self.subTest(text=text[:70]), self.assertRaises(report.ActualLoadReportError):
                report.parse_current_day_display(text, now=self.now)

    def test_current_display_does_not_guess_unflagged_fall_hours(self) -> None:
        now = self.now.replace(month=11, day=1)
        with self.assertRaises(report.ActualLoadReportError) as caught:
            report.parse_current_day_display(display(now.date()), now=now)
        self.assertEqual(caught.exception.code, "ambiguous_display_dst")

    def test_latest_index_revision_selected_and_bad_newest_link_not_ignored(self) -> None:
        day = self.now.date()
        sources = report.select_actual_load_archives(index_row(day, 2, "060000123") + index_row(day, 1))
        self.assertTrue(sources[day].url.endswith("=2"))
        with self.assertRaises(report.ActualLoadReportError):
            report.select_actual_load_archives(index_row(day, 1) + index_row(day, 2, "060000123", "https://example.com/archive.zip"))

    def test_history_caches_unchanged_archives_but_refreshes_live_index_and_display(self) -> None:
        sources = self.sources()
        with patch.object(report, "_fetch", side_effect=sources.__getitem__) as download:
            first = report.fetch_actual_load_history(now=self.now, history_days=2, cache_dir=self.cache)
            self.assertEqual(download.call_count, 4)
            download.reset_mock()
            second = report.fetch_actual_load_history(now=self.now, history_days=2, cache_dir=self.cache)
            self.assertEqual([call.args[0] for call in download.call_args_list], [report.DEFAULT_INDEX_URL, report.DEFAULT_DISPLAY_URL])
        self.assertTrue(first["history"].equals(second["history"]))
        self.assertEqual(len(second["history"]), 59)
        self.assertEqual(second["metadata"]["latest_interval_end_utc"], "2026-11-14T17:00:00+00:00")
        self.assertEqual(second["metadata"]["archive_count"], 2)
        self.assertTrue(all(item["cache_hit"] for item in second["metadata"]["archives"]))
        self.assertEqual(second["metadata"]["warnings"], [])
        self.assertEqual(first["metadata"]["source_sha256"], second["metadata"]["source_sha256"])

    def test_corrupt_cache_is_refetched_and_never_substituted_for_live_index(self) -> None:
        self.fetch()
        next(self.cache.glob("*.zip")).write_bytes(b"not a valid cached archive")
        with patch.object(report, "_fetch", side_effect=self.sources().__getitem__) as download:
            result = report.fetch_actual_load_history(now=self.now, history_days=2, cache_dir=self.cache)
        self.assertEqual(download.call_count, 3)
        self.assertEqual(sum(item["cache_hit"] for item in result["metadata"]["archives"]), 1)
        with patch.object(report, "_fetch", side_effect=report.ActualLoadReportError("offline", "Source unavailable")):
            with self.assertRaises(report.ActualLoadReportError):
                report.fetch_actual_load_history(now=self.now, history_days=2, cache_dir=self.cache)

    def test_unavailable_current_day_is_explicit_and_daily_origin_is_retained(self) -> None:
        sources = self.sources()
        sources[report.DEFAULT_DISPLAY_URL] = b"source changed"
        result = self.fetch(sources)
        self.assertEqual(len(result["history"]), 48)
        self.assertEqual(result["metadata"]["latest_interval_end_utc"], "2026-11-14T06:00:00+00:00")
        self.assertEqual(result["metadata"]["current_day"]["status"], "omitted")
        self.assertTrue(result["metadata"]["warnings"])

    def test_daylight_saving_bridge_is_omitted_without_losing_daily_history(self) -> None:
        now = self.now.replace(day=1)
        result = self.fetch(self.sources(now), now)
        self.assertEqual(len(result["history"]), 48)
        self.assertEqual(result["metadata"]["current_day"]["reason"], "ambiguous_display_dst")
        self.assertIn("repeated-hour", result["metadata"]["warnings"][0])

    def test_missing_daily_archive_and_stale_or_future_publication_fail(self) -> None:
        for change, expected in (("missing", "missing_history"), ("stale", "stale_history"), ("future", "future_publication")):
            sources = self.sources()
            if change == "missing":
                sources[report.DEFAULT_INDEX_URL] = index_row(self.now.date(), 2).encode()
            elif change == "stale":
                sources[report.DEFAULT_INDEX_URL] = index_row(self.now.date() - timedelta(days=1), 1).encode()
            else:
                sources[report.DEFAULT_INDEX_URL] = index_row(self.now.date(), 2, "160000123").encode()
            with self.subTest(change=change), self.assertRaises(report.ActualLoadReportError) as caught:
                self.fetch(sources)
            self.assertEqual(caught.exception.code, expected)

    def test_before_daily_publication_pending_gap_is_visible_not_filled(self) -> None:
        now = self.now.replace(hour=4)
        yesterday = now.date() - timedelta(days=1)
        sources = {
            report.DEFAULT_INDEX_URL: (index_row(yesterday, 2) + index_row(yesterday - timedelta(days=1), 1)).encode(),
            "https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId=1": archive(daily_csv(now.date() - timedelta(days=3))),
            "https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId=2": archive(daily_csv(now.date() - timedelta(days=2))),
        }
        result = self.fetch(sources, now)
        self.assertEqual(result["metadata"]["current_day"]["reason"], "pending_daily_publication")
        self.assertEqual(result["metadata"]["latest_interval_end_utc"], "2026-11-13T06:00:00+00:00")
        self.assertIn("not published yet", result["metadata"]["warnings"][0])

    def test_cache_size_is_bounded_and_unwritable_cache_does_not_fabricate(self) -> None:
        with patch.object(report, "MAX_CACHE_ARCHIVES", 1):
            result = self.fetch()
        self.assertEqual(len(list(self.cache.glob("*.zip"))), 1)
        self.assertEqual(len(list(self.cache.glob("*.json"))), 1)
        bad = self.cache / "not-a-directory"
        bad.write_text("placeholder")
        with patch.object(report, "_fetch", side_effect=self.sources().__getitem__):
            result = report.fetch_actual_load_history(now=self.now, history_days=2, cache_dir=bad)
        self.assertEqual(len(result["history"]), 59)
        self.assertIn("cache is unavailable", result["metadata"]["warnings"][0])

    def test_invalid_configuration_and_archive_are_rejected_before_use(self) -> None:
        for value in (True, 1, 31, 2.5):
            with self.assertRaises(report.ActualLoadReportError):
                report.fetch_actual_load_history(now=self.now, history_days=value, cache_dir=self.cache)
        with patch.dict("os.environ", {"ERCOT_ACTUAL_LOAD_INDEX_URL": "https://example.com/data"}), patch.object(report, "_fetch") as download:
            with self.assertRaises(report.ActualLoadReportError):
                report.fetch_actual_load_history(now=self.now, history_days=2, cache_dir=self.cache)
            download.assert_not_called()
        sources = self.sources()
        sources["https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId=1"] = archive("untrusted shape", "data.txt")
        with self.assertRaises(report.ActualLoadReportError) as caught:
            self.fetch(sources)
        self.assertEqual(caught.exception.code, "invalid_archive")


if __name__ == "__main__":
    unittest.main()
