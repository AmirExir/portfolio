"""Render ERCOT's supplied hourly forecast as a traceable in-memory PNG.

The renderer does not forecast, interpolate, smooth, fetch or replace missing
observations. Timestamps are aware instants; operating days use Central time.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from io import BytesIO
import math
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo


CENTRAL = ZoneInfo("America/Chicago")
UTC = timezone.utc
MAX_IMAGE_BASE64_LENGTH = 600_000


class LoadForecastChartError(ValueError):
    """A forecast chart could not be rendered from the supplied evidence."""


@dataclass(frozen=True)
class _Hour:
    start: datetime
    end: datetime
    mw: float


@dataclass(frozen=True)
class _PlotData:
    hours: tuple[_Hour, ...]
    upcoming: tuple[_Hour, ...]
    daily_peaks: tuple[_Hour, ...]
    next_peak: _Hour
    week_peak: _Hour
    generated: datetime
    published: datetime
    source_url: str
    report_name: str


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise LoadForecastChartError(f"Chart {label} must be an offset-aware timestamp.")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LoadForecastChartError(f"Chart {label} is not a valid timestamp.") from exc
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise LoadForecastChartError(f"Chart {label} must include a UTC offset.")
    return stamp.astimezone(UTC)


def _mw(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LoadForecastChartError("Chart load values must be numeric MW.")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise LoadForecastChartError("Chart load values must be finite, nonnegative MW.")
    return result


def _check_peak(record: Any, expected: _Hour, label: str) -> None:
    if not isinstance(record, dict):
        raise LoadForecastChartError(f"Chart {label} evidence is missing.")
    if (
        _timestamp(record.get("interval_start_utc"), label) != expected.start
        or not math.isclose(_mw(record.get("peak_load_mw")), expected.mw, abs_tol=1e-6)
    ):
        raise LoadForecastChartError(f"Chart {label} disagrees with the hourly forecast.")


def _prepare_plot_data(report: dict[str, Any]) -> _PlotData:
    if report.get("ok") is not True or report.get("units") != "MW":
        raise LoadForecastChartError("A successful forecast in MW is required for the chart.")
    if report.get("forecast_kind") != "official_ercot_np3_561_cd":
        raise LoadForecastChartError("The chart requires the official ERCOT forecast report.")
    generated = _timestamp(report.get("generated_at"), "retrieval time")
    published = _timestamp(report.get("published_at"), "publication time")
    source_url = report.get("source_url")
    if not isinstance(source_url, str) or len(source_url) > 512:
        raise LoadForecastChartError("Chart source URL is missing or invalid.")
    parsed_url = urlparse(source_url)
    if parsed_url.scheme != "https" or parsed_url.hostname != "www.ercot.com":
        raise LoadForecastChartError("Chart source must identify the official ERCOT report.")
    rows = report.get("hourly_series")
    if not isinstance(rows, list) or not 167 <= len(rows) <= 169:
        raise LoadForecastChartError("Chart requires seven complete operating days of hourly data.")
    hours = []
    for row in rows:
        if not isinstance(row, dict):
            raise LoadForecastChartError("Chart hourly data is malformed.")
        start = _timestamp(row.get("interval_start_utc"), "interval start")
        end = _timestamp(row.get("interval_end_utc"), "interval end")
        if end - start != timedelta(hours=1):
            raise LoadForecastChartError("Chart intervals must each represent one actual hour.")
        hours.append(_Hour(start, end, _mw(row.get("load_mw"))))
    hours.sort(key=lambda hour: hour.start)
    today = generated.astimezone(CENTRAL).date()
    first = datetime.combine(today, time(), CENTRAL).astimezone(UTC)
    last = datetime.combine(today + timedelta(days=7), time(), CENTRAL).astimezone(UTC)
    if hours[0].start != first or hours[-1].end != last or any(
        before.end != after.start for before, after in zip(hours, hours[1:])
    ):
        raise LoadForecastChartError("Chart hourly coverage has a gap, duplicate or incorrect operating-day boundary.")
    upcoming = tuple(hour for hour in hours if hour.start >= generated)
    if len(upcoming) < 48:
        raise LoadForecastChartError("Chart requires the next 48 complete hourly intervals.")
    peaks = tuple(
        max((hour for hour in hours if hour.start.astimezone(CENTRAL).date() == today + timedelta(days=offset)),
            key=lambda hour: hour.mw)
        for offset in range(7)
    )
    records = report.get("daily_peaks")
    if not isinstance(records, list) or len(records) != 7:
        raise LoadForecastChartError("Chart daily-peak evidence is incomplete.")
    for record, peak in zip(records, peaks):
        _check_peak(record, peak, "daily peak")
    next_peak = max(upcoming[:24], key=lambda hour: hour.mw)
    _check_peak(report.get("next_24_hour_peak"), next_peak, "next-24-hour peak")
    week_peak = max(hours, key=lambda hour: hour.mw)
    if "week_peak" in report:
        _check_peak(report["week_peak"], week_peak, "seven-day peak")
    return _PlotData(tuple(hours), upcoming, peaks, next_peak, week_peak,
                     generated, published, source_url, str(report.get("report_name", ""))[:200])


def _clock(stamp: datetime) -> str:
    local = stamp.astimezone(CENTRAL)
    return f"{local.hour % 12 or 12}:{local.minute:02d} {local:%p %Z}"


def _interval_label(hour: _Hour) -> str:
    start = hour.start.astimezone(CENTRAL)
    end = hour.end.astimezone(CENTRAL)
    # Include both zone labels at a DST transition; repeated 1 AM is unambiguous.
    if start.utcoffset() != end.utcoffset():
        clocks = f"{_clock(start)}–{_clock(end)}"
    else:
        clocks = f"{start.hour % 12 or 12} {start:%p}–{end.hour % 12 or 12} {end:%p %Z}"
    return f"{start:%a, %b} {start.day} · {clocks}"


def _build_figure(data: _PlotData) -> Any:
    from matplotlib import dates as mdates
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    navy, blue, teal, orange = "#162C43", "#245FD0", "#087F8C", "#BE6412"
    muted, grey, grid, paper = "#586B7D", "#AFBBC8", "#E4EAF1", "#F7F9FC"
    figure = Figure(figsize=(8, 9.7), dpi=140, facecolor=paper)
    figure.text(0.09, 0.962, "ERCOT load forecast", fontsize=23, fontweight="bold", color=navy)
    figure.text(0.09, 0.928, "Official forecast  ·  System-wide demand  ·  MW", fontsize=12, color=muted)
    published = data.published.astimezone(CENTRAL)
    figure.text(0.09, 0.897,
                f"Published {published:%a, %b} {published.day}, {published.year} · {_clock(published)}",
                fontsize=11.5, color=muted)
    for left, label, peak, color in (
        (0.09, "NEXT 24 COMPLETE HOURS", data.next_peak, blue),
        (0.54, "7 OPERATING-DAY PEAK", data.week_peak, teal),
    ):
        figure.text(left, 0.852, label, fontsize=10.5, fontweight="bold", color=muted)
        figure.text(left, 0.816, f"{peak.mw:,.0f} MW", fontsize=21, fontweight="bold", color=color)
        figure.text(left, 0.789, _interval_label(peak), fontsize=9.5, color=muted)

    detail = figure.add_axes((0.12, 0.495, 0.83, 0.233), facecolor="white")
    week = figure.add_axes((0.12, 0.143, 0.83, 0.222), facecolor="white")
    number = lambda value: mdates.date2num(value)

    def draw_steps(axis: Any, rows: tuple[_Hour, ...], color: str, width: float = 2.0) -> None:
        if rows:
            edges = [number(hour.start) for hour in rows] + [number(rows[-1].end)]
            axis.stairs([hour.mw for hour in rows], edges, baseline=None, color=color, linewidth=width)

    for axis in (detail, week):
        axis.set_axisbelow(True)
        axis.grid(axis="y", color=grid, linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(grid)
        axis.tick_params(axis="both", colors=muted, labelsize=10.5, length=0, pad=8)
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:,.0f}"))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        axis.set_ylabel("Load (MW)", fontsize=11, color=muted, labelpad=8)

    next_hours = data.upcoming[:48]
    draw_steps(detail, next_hours, blue, 2.4)
    detail.set_title("Next 48 complete hourly intervals", loc="left", fontsize=14, color=navy,
                     fontweight="bold", pad=17)
    detail.set_xlim(number(next_hours[0].start), number(next_hours[-1].end))
    detail.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12], tz=CENTRAL))
    detail.xaxis.set_major_formatter(mdates.DateFormatter("%a %d\n%I %p", tz=CENTRAL))
    next_values = [hour.mw for hour in next_hours]
    detail_pad = max((max(next_values) - min(next_values)) * 0.20, max(next_values) * 0.025, 1)
    detail.set_ylim(max(0, min(next_values) - detail_pad), max(next_values) + detail_pad)
    midpoint = data.next_peak.start + timedelta(minutes=30)
    detail.scatter([number(midpoint)], [data.next_peak.mw], color=orange, s=48, zorder=5,
                   edgecolors="white", linewidths=1.1)

    earlier = tuple(hour for hour in data.hours if hour.start < data.generated)
    draw_steps(week, earlier, grey, 1.8)
    draw_steps(week, data.upcoming, teal, 2.0)
    if earlier:
        week.axvspan(number(earlier[0].start), number(earlier[-1].end), color="#EEF1F5", zorder=0)
    week.set_title("Seven operating days · hourly forecast + daily peaks", loc="left", fontsize=13.5,
                   color=navy, fontweight="bold", pad=17)
    week.set_xlim(number(data.hours[0].start), number(data.hours[-1].end))
    dates = [peak.start.astimezone(CENTRAL).date() for peak in data.daily_peaks]
    middays = [datetime.combine(day, time(12), CENTRAL).astimezone(UTC) for day in dates]
    week.set_xticks([number(stamp) for stamp in middays],
                    [f"{day:%a}\n{day:%b} {day.day}" for day in dates])
    values = [hour.mw for hour in data.hours]
    week_pad = max((max(values) - min(values)) * 0.25, max(values) * 0.035, 1)
    week.set_ylim(max(0, min(values) - week_pad * 0.5), max(values) + week_pad)
    for peak in data.daily_peaks:
        middle = number(peak.start + timedelta(minutes=30))
        color = grey if peak.start < data.generated else teal
        week.scatter([middle], [peak.mw], color=color, edgecolors="white", linewidths=0.8, s=33, zorder=5)
        week.annotate(f"{peak.mw:,.0f}", (middle, peak.mw), xytext=(0, 9), textcoords="offset points",
                      ha="center", va="bottom", fontsize=9.5, color=color, fontweight="bold")
    figure.text(0.12, 0.420, "Central time · Orange dot marks the next-24-hour peak", fontsize=10.5, color=muted)
    figure.text(0.12, 0.087, "Earlier/partial hours are gray; excluded from 24/48-hour views.", fontsize=10, color=muted)
    figure.text(0.09, 0.051, "Source: ERCOT NP3-561-CD · SystemTotal · Forecast, not actual load", fontsize=10, color=muted)
    retrieved = data.generated.astimezone(CENTRAL)
    figure.text(0.09, 0.025,
                f"Retrieved {retrieved:%b} {retrieved.day}, {retrieved.year} · {_clock(retrieved)}  |  Hourly steps; no smoothing",
                fontsize=9.5, color=muted)
    return figure


def render_load_forecast_chart(report: dict[str, Any]) -> dict[str, str]:
    """Return a bounded PNG attachment without fetching data or writing a file.

    Earlier/current partial hours appear only in the full-week context panel.
    Source hourly values and actual UTC intervals are preserved across DST.
    Invalid or inconsistent evidence raises ``LoadForecastChartError``.
    """

    if not isinstance(report, dict):
        raise LoadForecastChartError("Chart input must be a forecast report object.")
    data = _prepare_plot_data(report)
    figure = None
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        figure = _build_figure(data)
        buffer = BytesIO()
        FigureCanvasAgg(figure).print_png(buffer, metadata={
            "Title": "ERCOT official hourly load forecast",
            "Source": data.source_url,
            "Report": data.report_name,
            "Published": data.published.isoformat(),
            "Retrieved": data.generated.isoformat(),
        }, pil_kwargs={"optimize": True, "compress_level": 9})
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        if len(encoded) > MAX_IMAGE_BASE64_LENGTH:
            raise LoadForecastChartError("The forecast chart exceeds the attachment size limit.")
        return {"base64": encoded, "mime_type": "image/png", "file_name": "ercot-load-forecast.png"}
    except LoadForecastChartError:
        raise
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise LoadForecastChartError("The forecast chart renderer is unavailable; the forecast text remains available.") from exc
    finally:
        if figure is not None:
            figure.clear()
