"""Credential-free, traceable ERCOT hourly actual load for dashboard models.

NP6-345-CD supplies complete previous operating days. GEN-527-UI supplies
completed hours of the current day; ERCOT identifies it as the display of the
same report data. Values are ERCOT's published system TOTAL in MW. Hour-ending
labels become interval starts on an aware America/Chicago index, preserving
23/25-hour days. Missing observations are never interpolated or forward-filled.

Only immutable daily archives are cached, outside the repository by default.
The index and current-day display are always retrieved again. Publication
metadata describes availability; historical rows are not assumed available at
their interval start when constructing retrospective forecast comparisons.
"""

from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
import hashlib
from html.parser import HTMLParser
from io import BytesIO, StringIO
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse
from zipfile import BadZipFile, ZipFile

import pandas as pd

if __package__:
    from .load_forecast_report import (
        CENTRAL, UTC, MAX_DOWNLOAD_BYTES, ForecastReportError, ReportDownload,
        _IndexParser, _day_intervals, _download, parse_forecast_csv,
    )
else:  # Direct Streamlit entry-point imports.
    from load_forecast_report import (
        CENTRAL, UTC, MAX_DOWNLOAD_BYTES, ForecastReportError, ReportDownload,
        _IndexParser, _day_intervals, _download, parse_forecast_csv,
    )



DEFAULT_INDEX_URL = "https://www.ercot.com/misapp/GetReports.do?reportTypeId=13101"
DEFAULT_DISPLAY_URL = "https://www.ercot.com/content/cdr/html/actual_loads_of_weather_zones.html"
PRODUCT_URL = "https://www.ercot.com/mp/data-products/data-product-details?id=NP6-345-CD"
DISPLAY_PRODUCT_URL = "https://www.ercot.com/mp/data-products/data-product-details?id=GEN-527-UI"
MAX_CACHE_ARCHIVES = 64
REPORT_NAME = re.compile(r"cdr\.00013101\.\d+\.(\d{8})\.(\d{6})\d*\.ACTUALSYSLOADWZNP6345_csv\.zip")
DISPLAY_COLUMNS = ["Oper Day", "Hour Ending", "COAST", "EAST", "FAR_WEST", "NORTH", "NORTH_C", "SOUTHERN", "SOUTH_C", "WEST", "TOTAL"]


class ActualLoadReportError(ValueError):
    """An expected source failure with a safe dashboard explanation."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _fetch(url: str) -> bytes:
    try:
        return _download(url)
    except ForecastReportError as exc:
        raise ActualLoadReportError(exc.code, "ERCOT's official actual-load source could not be read. Try again later.") from exc


def _official_url(url: str, *, display: bool = False) -> str:
    parsed = urlparse(url)
    valid = (
        parsed.scheme == "https" and parsed.netloc == "www.ercot.com"
        and not parsed.fragment
    )
    if display:
        valid = valid and parsed.path in {
            "/content/cdr/html/actual_loads_of_weather_zones.html",
            "/content/cdr/html/actual_loads_of_weather_zones",
        } and not parsed.query
    else:
        valid = valid and parsed.path == "/misapp/GetReports.do" and parse_qs(parsed.query) == {"reportTypeId": ["13101"]}
    if not valid:
        raise ActualLoadReportError("invalid_configuration", "Actual-load sources must use ERCOT's official NP6-345 report index and weather-zone display URLs.")
    return url


def select_actual_load_archives(index_html: str, index_url: str = DEFAULT_INDEX_URL) -> dict[date, ReportDownload]:
    """Select the newest archive for each publication day, without guessing links."""

    _official_url(index_url)
    parser = _IndexParser()
    parser.feed(index_html)
    candidates: dict[date, list[ReportDownload]] = {}
    for text, links in parser.rows:
        match = REPORT_NAME.search(text)
        if not match:
            continue
        try:
            published = datetime.strptime(match[1] + match[2], "%Y%m%d%H%M%S").replace(tzinfo=CENTRAL)
        except ValueError as exc:
            raise ActualLoadReportError("invalid_publication", "ERCOT's actual-load archive has an invalid publication date.") from exc
        download = ""
        for link in links:
            candidate = urljoin(index_url, link)
            parsed = urlparse(candidate)
            if (parsed.scheme == "https" and parsed.netloc == "www.ercot.com"
                    and parsed.path == "/misdownload/servlets/mirDownload" and not parsed.fragment
                    and re.fullmatch(r"\d+", parse_qs(parsed.query).get("doclookupId", [""])[0])):
                download = candidate
                break
        candidates.setdefault(published.date(), []).append(ReportDownload(match[0], published, download))
    if not candidates:
        raise ActualLoadReportError("report_missing", "No daily actual-load archives were found in ERCOT's public report index.")
    selected: dict[date, ReportDownload] = {}
    for day, versions in candidates.items():
        source = max(versions, key=lambda item: (item.published_at, item.name))
        if not source.url:
            raise ActualLoadReportError("invalid_download_link", "An ERCOT actual-load archive has no valid official download link.")
        if source.published_at.utcoffset() != source.published_at.replace(fold=1).utcoffset():
            raise ActualLoadReportError("ambiguous_publication", "ERCOT's actual-load publication clock is ambiguous at the daylight-saving transition.")
        selected[day] = source
    return selected


def parse_actual_load_csv(csv_text: str, *, expected_day: date | None = None) -> pd.DataFrame:
    """Validate one complete operating day, retaining the repeated-hour flag."""

    reader = csv.DictReader(StringIO(csv_text.lstrip("\ufeff")))
    names = reader.fieldnames or []
    required = {"OperDay", "HourEnding", "TOTAL", "DSTFlag"}
    if not required.issubset(names) or len(names) != len(set(names)):
        raise ActualLoadReportError("invalid_schema", "ERCOT's actual-load CSV columns changed or are missing.")
    normalized = StringIO()
    writer = csv.writer(normalized)
    writer.writerow(["DeliveryDate", "HourEnding", "SystemTotal", "DSTFlag"])
    for row in reader:
        if None in row or any(row.get(key) is None for key in required):
            raise ActualLoadReportError("invalid_data", "ERCOT's actual-load CSV contains an incomplete row.")
        writer.writerow([row["OperDay"], row["HourEnding"], row["TOTAL"], row["DSTFlag"]])
    try:
        days = parse_forecast_csv(normalized.getvalue())
    except ForecastReportError as exc:
        raise ActualLoadReportError(exc.code, "ERCOT's actual-load CSV contains an invalid value, duplicate interval, or unsupported hour/DST label.") from exc
    if len(days) != 1 or (expected_day is not None and set(days) != {expected_day}):
        raise ActualLoadReportError("unexpected_operating_day", "ERCOT's actual-load archive does not contain its expected operating day.")
    day, rows = next(iter(days.items()))
    expected = set(_day_intervals(day).values())
    if {row["interval_start"] for row in rows} != expected:
        raise ActualLoadReportError("incomplete_day", "ERCOT's actual-load archive is missing hourly observations; no values were filled.")
    return _frame(rows)


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    index = pd.DatetimeIndex([row["interval_start"] for row in rows], tz=UTC).tz_convert(CENTRAL)
    return pd.DataFrame({"load": [row["load_mw"] for row in rows]}, index=index.rename("interval_start")).sort_index()


class _DisplayParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.row: list[str] = []
        self.cell: list[str] | None = None
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.row = []
        if tag in {"td", "th"}:
            self.cell = []

    def handle_data(self, data: str) -> None:
        self.text.append(data)
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        if tag == "tr" and self.row:
            self.rows.append(self.row)


def parse_current_day_display(html: str, *, now: datetime) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read completed current-day hours; reject ambiguous unflagged DST data."""

    if now.tzinfo is None:
        raise ActualLoadReportError("invalid_configuration", "The actual-load cutoff must include a timezone.")
    day = now.astimezone(CENTRAL).date()
    intervals = _day_intervals(day)
    if len(intervals) == 25:
        raise ActualLoadReportError("ambiguous_display_dst", "Current-day actuals were omitted: the public table has no repeated-hour flag on this daylight-saving transition day.")
    parser = _DisplayParser()
    parser.feed(html)
    match = re.search(r"Last Date and Time:\s*([A-Za-z]{3} \d{1,2}, \d{4} \d{1,2}:\d{2})", " ".join(parser.text))
    try:
        if match is None:
            raise ValueError("missing timestamp")
        published = datetime.strptime(match[1], "%b %d, %Y %H:%M").replace(tzinfo=CENTRAL)
    except ValueError as exc:
        raise ActualLoadReportError("invalid_display_publication", "ERCOT's current-day actual-load publication time could not be verified.") from exc
    age = now.astimezone(UTC) - published.astimezone(UTC)
    if published.date() != day or age > timedelta(hours=2) or age < -timedelta(minutes=5):
        raise ActualLoadReportError("stale_display", "Current-day actuals were omitted because ERCOT's display is stale or has an invalid publication time.")
    if not parser.rows or parser.rows[0] != DISPLAY_COLUMNS:
        raise ActualLoadReportError("invalid_display_schema", "ERCOT's current-day actual-load table columns changed or are missing.")
    rows, seen = [], set()
    for cells in parser.rows[1:]:
        try:
            if len(cells) != len(DISPLAY_COLUMNS) or datetime.strptime(cells[0], "%m/%d/%Y").date() != day:
                raise ValueError("wrong operating day or column count")
            if not re.fullmatch(r"\d{4}", cells[1]) or not cells[1].endswith("00"):
                raise ValueError("invalid hour ending")
            start = intervals[(int(cells[1][:2]), "N")]
            value = float(cells[-1])
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid MW")
            if start in seen or start + timedelta(hours=1) > min(now.astimezone(UTC), published.astimezone(UTC)):
                raise ValueError("duplicate or incomplete hour")
        except (ValueError, KeyError, TypeError) as exc:
            raise ActualLoadReportError("invalid_display_data", "ERCOT's current-day table contains an invalid, duplicate, or unfinished hourly observation.") from exc
        seen.add(start)
        rows.append({"interval_start": start, "load_mw": value})
    if not rows:
        raise ActualLoadReportError("empty_display", "No completed current-day actual-load hours are available yet.")
    starts = sorted(seen)
    expected = sorted(start for start in intervals.values()
                      if start + timedelta(hours=1) <= min(now.astimezone(UTC), published.astimezone(UTC)))
    if starts != expected:
        raise ActualLoadReportError("incomplete_display", "Current-day actuals were omitted because hourly observations are missing; no values were filled.")
    return _frame(rows), {"published_at_utc": published.astimezone(UTC).isoformat(), "rows": len(rows)}


def _archive_frame(raw: bytes, source: ReportDownload) -> pd.DataFrame:
    try:
        with ZipFile(BytesIO(raw)) as zipped:
            members = zipped.infolist()
            if (len(members) != 1 or not members[0].filename.lower().endswith(".csv")
                    or members[0].file_size > MAX_DOWNLOAD_BYTES):
                raise ActualLoadReportError("invalid_archive", "ERCOT's actual-load archive has an unexpected layout or size.")
            body = zipped.read(members[0]).decode("utf-8-sig")
    except (BadZipFile, UnicodeError, RuntimeError, OSError) as exc:
        raise ActualLoadReportError("invalid_archive", "ERCOT's actual-load archive could not be read.") from exc
    return parse_actual_load_csv(body, expected_day=source.published_at.date() - timedelta(days=1))


def _cache_root(cache_dir: str | Path | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    configured = os.getenv("ERCOT_ACTUAL_LOAD_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache"))).expanduser() / "ercot-grid-dashboard" / "actual-load"


def _cached_archive(source: ReportDownload, root: Path) -> tuple[pd.DataFrame, str, bool, bool]:
    """Validate cached bytes and bound this module's immutable archive cache."""

    key = hashlib.sha256(source.url.encode()).hexdigest()
    archive_path, manifest_path = root / f"{key}.zip", root / f"{key}.json"
    cache_available = True
    try:
        if archive_path.is_file() and manifest_path.is_file() and archive_path.stat().st_size <= MAX_DOWNLOAD_BYTES and manifest_path.stat().st_size < 8192:
            raw = archive_path.read_bytes()
            manifest = json.loads(manifest_path.read_text())
            digest = hashlib.sha256(raw).hexdigest()
            if isinstance(manifest, dict) and manifest.get("url") == source.url and manifest.get("name") == source.name and manifest.get("sha256") == digest:
                return _archive_frame(raw, source), digest, True, True
    except (OSError, ValueError, ActualLoadReportError):
        # A cache is an optimization, never a substitute for live verified data.
        pass
    raw = _fetch(source.url)
    frame = _archive_frame(raw, source)
    digest = hashlib.sha256(raw).hexdigest()
    try:
        root.mkdir(parents=True, exist_ok=True)
        for target, data in ((archive_path, raw), (manifest_path, json.dumps({"url": source.url, "name": source.name, "sha256": digest}).encode())):
            temporary: str | None = None
            try:
                with tempfile.NamedTemporaryFile(dir=root, prefix=".actual-load-", delete=False) as handle:
                    temporary = handle.name
                    handle.write(data)
                os.replace(temporary, target)
                temporary = None
            finally:
                if temporary:
                    Path(temporary).unlink(missing_ok=True)
        paths = [path for path in root.glob("*.zip") if re.fullmatch(r"[0-9a-f]{64}\.zip", path.name)]
        for expired in sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)[MAX_CACHE_ARCHIVES:]:
            expired.unlink(missing_ok=True)
            expired.with_suffix(".json").unlink(missing_ok=True)
    except OSError:
        cache_available = False
    return frame, digest, False, cache_available


def fetch_actual_load_history(*, now: datetime | None = None, history_days: int = 30, cache_dir: str | Path | None = None) -> dict[str, Any]:
    """Fetch complete daily actuals plus verified completed current-day hours.

    ``history_days`` counts complete archived operating days (2–30); current-day
    observations are additional. A failed current-day bridge leaves daily data
    intact with an explicit metadata warning. Missing/stale daily history fails
    clearly. The caller must use ``latest_interval_end_utc`` as forecast origin,
    forecast across any gap to now, and display only the requested future hours.
    """

    if isinstance(history_days, bool) or not isinstance(history_days, int) or not 2 <= history_days <= 30:
        raise ActualLoadReportError("invalid_configuration", "Actual-load history must contain between 2 and 30 complete days.")
    cutoff = now or datetime.now(UTC)
    if cutoff.tzinfo is None:
        raise ActualLoadReportError("invalid_configuration", "The actual-load cutoff must include a timezone.")
    local_now = cutoff.astimezone(CENTRAL)
    retrieved = datetime.now(UTC)
    index_url = _official_url(os.getenv("ERCOT_ACTUAL_LOAD_INDEX_URL", DEFAULT_INDEX_URL))
    display_url = _official_url(os.getenv("ERCOT_ACTUAL_LOAD_DISPLAY_URL", DEFAULT_DISPLAY_URL), display=True)
    try:
        sources = select_actual_load_archives(_fetch(index_url).decode("utf-8"), index_url)
    except UnicodeError as exc:
        raise ActualLoadReportError("invalid_index", "ERCOT's actual-load report index could not be read.") from exc
    latest_publication_day = max(sources)
    latest = sources[latest_publication_day]
    if latest.published_at.astimezone(UTC) > cutoff.astimezone(UTC) + timedelta(minutes=5):
        raise ActualLoadReportError("future_publication", "ERCOT's newest actual-load publication is later than the requested cutoff.")
    earliest_allowed = local_now.date() - timedelta(days=int(local_now.hour < 6))
    if latest_publication_day < earliest_allowed:
        raise ActualLoadReportError("stale_history", "ERCOT's daily actual-load archive is stale. A current report is required; cached history was not substituted.")
    publication_days = [latest_publication_day - timedelta(days=offset) for offset in reversed(range(history_days))]
    if any(day not in sources for day in publication_days):
        raise ActualLoadReportError("missing_history", "ERCOT's public index is missing a requested daily archive; reduce the history window or retry later.")
    frames, archives, warnings = [], [], []
    cache_available = True
    root = _cache_root(cache_dir)
    for day in publication_days:
        source = sources[day]
        frame, digest, hit, available = _cached_archive(source, root)
        frames.append(frame)
        cache_available = cache_available and available
        archives.append({"name": source.name, "url": source.url, "published_at_utc": source.published_at.astimezone(UTC).isoformat(),
                         "operating_day": (day - timedelta(days=1)).isoformat(), "sha256": digest, "rows": len(frame), "cache_hit": hit})
    bridge: dict[str, Any] = {"source_url": display_url, "status": "omitted", "rows": 0}
    if latest_publication_day != local_now.date():
        warnings.append("Yesterday's daily actual-load archive is not published yet. Current-day rows were omitted to preserve a continuous history; the forecast origin is shown explicitly.")
        bridge["reason"] = "pending_daily_publication"
    else:
        try:
            display = _fetch(display_url)
            current, detail = parse_current_day_display(display.decode("utf-8"), now=cutoff)
            frames.append(current)
            bridge.update(detail, status="included", source_sha256=hashlib.sha256(display).hexdigest())
        except (ActualLoadReportError, UnicodeError) as exc:
            message = str(exc) if isinstance(exc, ActualLoadReportError) else "ERCOT's current-day actual-load table could not be decoded."
            warnings.append(message + " The daily archive remains available with its earlier forecast origin.")
            bridge["reason"] = getattr(exc, "code", "invalid_display")
    if not cache_available:
        warnings.append("The local actual-load cache is unavailable; this result uses verified live downloads.")
    history = pd.concat(frames).sort_index()
    if history.index.has_duplicates or not (history.index.to_series().diff().iloc[1:] == pd.Timedelta(hours=1)).all():
        raise ActualLoadReportError("discontinuous_history", "ERCOT's actual-load history is not a continuous hourly series; no gaps or duplicates were filled.")
    last_end = history.index[-1].tz_convert(UTC) + pd.Timedelta(hours=1)
    return {"history": history, "metadata": {
        "source": "Official ERCOT actual system load by weather zone (system TOTAL)",
        "product": "NP6-345-CD", "display_product": "GEN-527-UI", "units": "MW",
        "source_url": PRODUCT_URL, "display_product_url": DISPLAY_PRODUCT_URL, "index_url": index_url,
        "source_sha256": hashlib.sha256(history.to_csv().encode()).hexdigest(),
        "generated_at": cutoff.astimezone(UTC).isoformat(), "retrieved_at": retrieved.isoformat(),
        "latest_interval_end_utc": last_end.isoformat(), "first_interval_start_utc": history.index[0].tz_convert(UTC).isoformat(),
        "interval_convention": "interval_start; one elapsed hour; America/Chicago",
        "history_days": history_days, "archive_count": len(archives), "rows": len(history),
        "archives": archives, "current_day": bridge, "warnings": warnings,
        "availability_note": "Daily actuals are published the following morning; current-day observations are published after the completed hour. Retrospective comparisons must respect source publication times.",
    }}
