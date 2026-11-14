"""Publishable summaries of ERCOT's official public load forecast, without ML.

Reads NP3-561-CD from ERCOT MIS. Source hour-ending labels and repeated-hour
flags are checked against the America/Chicago hourly calendar. No missing
loads are filled and no previous report substitutes for a stale latest report.
This module prints JSON only and never sends a message or writes a report.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
from html.parser import HTMLParser
from io import BytesIO, StringIO
import json
import math
import os
import re
from typing import Any
from urllib.parse import urljoin, urlparse
from zipfile import BadZipFile, ZipFile
from zoneinfo import ZoneInfo

import requests


CENTRAL = ZoneInfo("America/Chicago")
UTC = timezone.utc
DEFAULT_INDEX_URL = "https://www.ercot.com/misapp/GetReports.do?reportTypeId=12312"
PRODUCT_URL = "https://www.ercot.com/mp/data-products/data-product-details?id=NP3-561-CD"
MAX_DOWNLOAD_BYTES = 2_000_000
REPORT_NAME = re.compile(
    r"cdr\.00012312\.\d+\.(\d{8})\.(\d{6})\d*\.LFCWEATHERNP3561_csv\.zip"
)


class ForecastReportError(ValueError):
    """An expected source failure with a safe user-facing explanation."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ReportDownload:
    """The immutable source name, publication clock and public download URL."""

    name: str
    published_at: datetime
    url: str


class _IndexParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[tuple[str, list[str]]] = []
        self.text: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.text, self.links = [], []
        if tag == "a":
            self.links.extend(value for key, value in attrs if key == "href" and value)

    def handle_data(self, data: str) -> None:
        self.text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "tr":
            self.rows.append((" ".join(self.text), list(self.links)))


def select_latest_report(index_html: str, index_url: str = DEFAULT_INDEX_URL) -> ReportDownload:
    """Select the latest named CSV archive, independent of index ordering."""

    parser = _IndexParser()
    parser.feed(index_html)
    candidates: list[ReportDownload] = []
    for text, links in parser.rows:
        match = REPORT_NAME.search(text)
        if not match:
            continue
        try:
            local = datetime.strptime(match[1] + match[2], "%Y%m%d%H%M%S")
        except ValueError as exc:
            raise ForecastReportError("invalid_publication", "ERCOT's report publication date is invalid.") from exc
        published = local.replace(tzinfo=CENTRAL)
        download_url = ""
        for href in links:
            url = urljoin(index_url, href)
            parsed = urlparse(url)
            if (
                parsed.scheme == "https" and parsed.hostname == "www.ercot.com"
                and parsed.path == "/misdownload/servlets/mirDownload"
                and re.search(r"(?:^|&)doclookupId=\d+(?:&|$)", parsed.query)
            ):
                download_url = url
                break
        candidates.append(ReportDownload(match[0], published, download_url))
    if not candidates:
        raise ForecastReportError("report_missing", "No usable official ERCOT forecast archive was found.")
    selected = max(candidates, key=lambda item: (item.published_at, item.name))
    if not selected.url:
        raise ForecastReportError("invalid_download_link", "ERCOT's newest forecast archive has no valid official download link.")
    # A filename has no offset. Never replace an ambiguous newest vintage
    # silently with an older one or guess its repeated-hour fold.
    if selected.published_at.utcoffset() != selected.published_at.replace(fold=1).utcoffset():
        raise ForecastReportError("ambiguous_publication", "ERCOT's latest publication clock is ambiguous at the daylight-saving transition.")
    return selected


def _day_intervals(day: date) -> dict[tuple[int, str], datetime]:
    """Map ERCOT HE/repeated-hour labels to actual interval-start instants."""

    start = datetime.combine(day, datetime.min.time(), CENTRAL).astimezone(UTC)
    end = datetime.combine(day + timedelta(days=1), datetime.min.time(), CENTRAL).astimezone(UTC)
    intervals = {}
    while start < end:
        local = start.astimezone(CENTRAL)
        intervals[(local.hour + 1, "Y" if local.fold else "N")] = start
        start += timedelta(hours=1)
    return intervals


def parse_forecast_csv(csv_text: str) -> dict[date, list[dict[str, Any]]]:
    """Read finite system MW values while preserving and checking source labels."""

    reader = csv.DictReader(StringIO(csv_text.lstrip("\ufeff")))
    required = {"DeliveryDate", "HourEnding", "SystemTotal", "DSTFlag"}
    names = reader.fieldnames or []
    if not required.issubset(names) or len(names) != len(set(names)):
        raise ForecastReportError("invalid_schema", "ERCOT's forecast columns changed or are missing.")
    days: dict[date, list[dict[str, Any]]] = {}
    seen: set[tuple[date, int, str]] = set()
    for row in reader:
        try:
            day = datetime.strptime(row["DeliveryDate"], "%m/%d/%Y").date()
            hour_match = re.fullmatch(r"(\d{1,2}):00", row["HourEnding"].strip())
            if not hour_match:
                raise ValueError("invalid hour ending")
            hour = int(hour_match[1])
            flag = row["DSTFlag"].strip().upper()
            value = float(row["SystemTotal"])
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid MW")
            interval_start = _day_intervals(day)[(hour, flag)]
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise ForecastReportError("invalid_data", "ERCOT returned an invalid load value or unsupported hour/DST label.") from exc
        key = (day, hour, flag)
        if key in seen:
            raise ForecastReportError("duplicate_interval", "ERCOT returned duplicate forecast intervals.")
        seen.add(key)
        days.setdefault(day, []).append({
            "hour_ending": hour, "dst_flag": flag, "load_mw": value,
            "interval_start": interval_start,
        })
    if not days:
        raise ForecastReportError("empty_forecast", "ERCOT returned no forecast rows.")
    return days


def _date_label(day: date) -> str:
    return f"{day:%a, %b} {day.day}, {day.year}"


def _clock_label(timestamp: datetime) -> str:
    local = timestamp.astimezone(CENTRAL)
    return f"{_date_label(local.date())} at {local:%I:%M %p %Z}".replace(" at 0", " at ")


def _peak_record(day: date, row: dict[str, Any]) -> dict[str, Any]:
    hour = row["hour_ending"]
    clock = "midnight (end of day)" if hour == 24 else f"{hour % 12 or 12} {'AM' if hour < 12 else 'PM'} CT"
    if row["dst_flag"] == "Y":
        clock += " (repeated hour)"
    return {
        "delivery_date": day.isoformat(), "peak_load_mw": row["load_mw"],
        "hour_ending": hour, "dst_flag": row["dst_flag"],
        "interval_start_utc": row["interval_start"].isoformat(),
        "time_label": f"{_date_label(day)}, hour ending {clock}",
    }


def build_forecast_report(
    source: ReportDownload, csv_text: str, *, now: datetime,
    max_age_hours: float = 3.0,
) -> dict[str, Any]:
    """Summarize seven complete operating days from one fresh official vintage."""

    if now.tzinfo is None or not math.isfinite(max_age_hours) or max_age_hours <= 0:
        raise ForecastReportError("invalid_configuration", "Forecast time/freshness configuration is invalid.")
    age = (now.astimezone(UTC) - source.published_at.astimezone(UTC)).total_seconds() / 3600
    if age < -5 / 60:
        raise ForecastReportError("future_report", "ERCOT's report publication time is ahead of the current clock.")
    if age > max_age_hours:
        raise ForecastReportError("stale_report", f"The latest official ERCOT forecast is older than {max_age_hours:g} hours; no current forecast was published.")
    days = parse_forecast_csv(csv_text)
    today = now.astimezone(CENTRAL).date()
    peaks = []
    upcoming = []
    for offset in range(7):
        day = today + timedelta(days=offset)
        rows = days.get(day, [])
        expected = _day_intervals(day)
        if {(row["hour_ending"], row["dst_flag"]) for row in rows} != set(expected):
            raise ForecastReportError("incomplete_forecast", f"ERCOT's forecast is missing complete hourly coverage for {_date_label(day)}.")
        peak = max(sorted(rows, key=lambda row: row["interval_start"]), key=lambda row: row["load_mw"])
        peaks.append(_peak_record(day, peak))
        upcoming.extend(
            (day, row) for row in rows
            if now.astimezone(UTC) <= row["interval_start"] < now.astimezone(UTC) + timedelta(hours=24)
        )
    if len(upcoming) != 24:
        raise ForecastReportError("incomplete_forecast", "ERCOT's forecast does not cover the next 24 complete hourly intervals.")
    next_day, next_row = max(upcoming, key=lambda item: item[1]["load_mw"])
    next_peak = _peak_record(next_day, next_row)
    lines = [
        "ERCOT Load Forecast", "Official ERCOT forecast · system-wide demand · MW",
        f"Report published: {_clock_label(source.published_at)}",
        f"Retrieved: {_clock_label(now)}", "",
        f"Next 24 hourly intervals: peak {next_peak['peak_load_mw']:,.0f} MW",
        next_peak["time_label"], "",
        "Daily forecast peaks (whole operating days):",
    ]
    for peak in peaks:
        day = date.fromisoformat(peak["delivery_date"])
        clock = peak["time_label"].split(", hour ending ", 1)[1]
        lines.append(f"• {_date_label(day)}: {peak['peak_load_mw']:,.0f} MW · hour ending {clock}")
    lines.extend([
        "", "Forecasts are estimates, not actual load or a grid-emergency prediction.",
        "The next 24 intervals start at the next hourly boundary; a partial current hour is excluded.",
        f"Source: {PRODUCT_URL}", f"Source file: {source.url}",
    ])
    return {
        "ok": True, "text": "\n".join(lines), "units": "MW",
        "forecast_kind": "official_ercot_np3_561_cd", "source_url": source.url,
        "source_product_url": PRODUCT_URL, "report_name": source.name,
        "published_at": source.published_at.isoformat(),
        "generated_at": now.astimezone(UTC).isoformat(),
        "source_sha256": hashlib.sha256(csv_text.encode("utf-8")).hexdigest(),
        "daily_peaks": peaks, "next_24_hour_peak": next_peak,
    }


def _download(url: str) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "www.ercot.com":
        raise ForecastReportError("invalid_configuration", "The forecast source must be an official HTTPS ERCOT URL.")
    try:
        with requests.get(url, timeout=(5, 25), stream=True, allow_redirects=False) as response:
            response.raise_for_status()
            if response.is_redirect:
                raise ForecastReportError("source_redirect", "ERCOT redirected the forecast source; the source configuration needs review.")
            chunks, size = [], 0
            for chunk in response.iter_content(chunk_size=65536):
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    raise ForecastReportError("source_too_large", "ERCOT's forecast download exceeded the expected size.")
                chunks.append(chunk)
            return b"".join(chunks)
    except requests.RequestException as exc:
        raise ForecastReportError("source_unavailable", "The official ERCOT forecast source could not be reached. Try again later.") from exc


def fetch_forecast_report(*, now: datetime | None = None) -> dict[str, Any]:
    """Fetch the latest public archive and summarize it without credentials."""

    index_url = os.getenv("ERCOT_LOAD_FORECAST_INDEX_URL", DEFAULT_INDEX_URL)
    try:
        max_age = float(os.getenv("ERCOT_LOAD_FORECAST_MAX_AGE_HOURS", "3"))
        source = select_latest_report(_download(index_url).decode("utf-8"), index_url)
        archive = _download(source.url)
        with ZipFile(BytesIO(archive)) as zipped:
            members = zipped.infolist()
            if len(members) != 1 or not members[0].filename.lower().endswith(".csv") or members[0].file_size > MAX_DOWNLOAD_BYTES:
                raise ForecastReportError("invalid_archive", "ERCOT's forecast archive has an unexpected layout or size.")
            csv_text = zipped.read(members[0]).decode("utf-8-sig")
    except (BadZipFile, UnicodeError) as exc:
        raise ForecastReportError("invalid_archive", "ERCOT's forecast archive could not be read.") from exc
    except ValueError as exc:
        if isinstance(exc, ForecastReportError):
            raise
        raise ForecastReportError("invalid_configuration", "Forecast freshness configuration is invalid.") from exc
    return build_forecast_report(source, csv_text, now=now or datetime.now(UTC), max_age_hours=max_age)


def main() -> None:
    """Emit one JSON result for workflow delivery; expected failures remain data."""

    try:
        payload = fetch_forecast_report()
    except ForecastReportError as exc:
        payload = {
            "ok": False, "error_code": exc.code,
            "text": f"ERCOT Load Forecast unavailable\n\n{exc}\nNo estimates were substituted.\nSource: {PRODUCT_URL}",
            "source_product_url": PRODUCT_URL,
        }
    print(json.dumps(payload, allow_nan=False))


if __name__ == "__main__":
    main()
