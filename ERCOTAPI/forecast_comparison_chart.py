"""One evidence-checked plot of ERCOT and a named model on matching hours."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import BytesIO
import math
from typing import Any

if __package__:
    from .forecast_comparison import validate_comparison_snapshot
    from .forecast_labels import model_display_name
    from .load_forecast_chart import CENTRAL, MAX_IMAGE_BASE64_LENGTH, _clock, _interval_label, _Hour, _prepare_plot_data, _timestamp
else:
    from forecast_comparison import validate_comparison_snapshot
    from forecast_labels import model_display_name
    from load_forecast_chart import CENTRAL, MAX_IMAGE_BASE64_LENGTH, _clock, _interval_label, _Hour, _prepare_plot_data, _timestamp


class ForecastComparisonChartError(ValueError):
    """The supplied evidence cannot support a current matched-hour chart."""


@dataclass(frozen=True)
class _ComparisonPlotData:
    official: tuple[_Hour, ...]
    model: tuple[_Hour, ...]
    model_name: str
    published: datetime
    retrieved: datetime
    issued: datetime
    cutoff: datetime
    source_url: str
    source_hash: str
    report_name: str
    mean_difference_mw: float


def _prepare_comparison_plot_data(report: dict[str, Any]) -> _ComparisonPlotData:
    """Require one fresh source vintage and all 24 unmodified paired hours."""
    if not isinstance(report, dict):
        raise ForecastComparisonChartError("Chart input must be a forecast report object.")
    try:
        source = _prepare_plot_data(report)
        comparison = report.get("comparison")
        if not isinstance(comparison, dict):
            raise ForecastComparisonChartError("A saved model comparison is required for the combined chart.")
        snapshot = validate_comparison_snapshot(comparison.get("snapshot"))
        rows = snapshot["rows"]
        summary = snapshot["summary"]
        if len(rows) != 24 or summary["matched_hours"] != 24 or any(summary[key] for key in ("excluded_elapsed_model_hours", "model_hours_without_ercot")):
            raise ForecastComparisonChartError("The combined chart requires exactly 24 fully matched future hours.")
        if (snapshot["ercot_published_at"] != source.published.isoformat()
                or snapshot["ercot_retrieved_at"] != source.generated.isoformat()
                or any(snapshot["ercot_source"][key] != report.get(key) for key in ("source_url", "source_sha256", "report_name"))):
            raise ForecastComparisonChartError("The comparison and official report identify different source vintages.")
        issued = _timestamp(snapshot["model_issued_at"], "model issue time")
        captured = _timestamp(snapshot["captured_at"], "comparison capture time")
        cutoff = _timestamp(snapshot["model_data_cutoff_utc"], "actual-load cutoff")
        if issued - source.published > timedelta(hours=3) or issued - cutoff > timedelta(hours=4):
            raise ForecastComparisonChartError("The model or official source vintage is too old for a current combined chart.")
        official_by_start = {hour.start: hour for hour in source.hours}
        official, model = [], []
        first_target = captured.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        for index, row in enumerate(rows):
            start = _timestamp(row["interval_start_utc"], "comparison interval start")
            end = _timestamp(row["interval_end_utc"], "comparison interval end")
            official_hour = official_by_start.get(start)
            if start != first_target + timedelta(hours=index) or official_hour is None or end != official_hour.end:
                raise ForecastComparisonChartError("The combined chart cannot fill missing, elapsed or shifted hourly intervals.")
            if not math.isclose(row["ercot_mw"], official_hour.mw, rel_tol=0, abs_tol=1e-6):
                raise ForecastComparisonChartError("The saved ERCOT values disagree with the identified official source.")
            official.append(official_hour)
            model.append(_Hour(start, end, float(row["model_mw"])))
        return _ComparisonPlotData(
            tuple(official), tuple(model), model_display_name(snapshot["model_metadata"]["model_type"]),
            source.published, source.generated, issued, cutoff, source.source_url,
            report["source_sha256"], source.report_name, float(summary["mean_difference_mw"]),
        )
    except ForecastComparisonChartError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ForecastComparisonChartError("The combined chart's forecast evidence is missing or inconsistent.") from exc


def _build_comparison_figure(data: _ComparisonPlotData) -> Any:
    """Draw only the paired horizon; both methods share one time and MW axis."""
    from matplotlib import dates as mdates
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    navy, teal, purple = "#162C43", "#087F8C", "#8652C8"
    muted, grid, paper = "#586B7D", "#E4EAF1", "#F7F9FC"
    figure = Figure(figsize=(9, 7.2), dpi=150, facecolor=paper)
    figure.text(0.08, 0.947, "ERCOT vs " + data.model_name, fontsize=23, fontweight="bold", color=navy)
    figure.text(0.08, 0.906, "24 matching future hours  ·  System-wide demand  ·  MW", fontsize=12, color=muted)
    published, issued = data.published.astimezone(CENTRAL), data.issued.astimezone(CENTRAL)
    figure.text(0.08, 0.866,
                f"ERCOT published {published:%a, %b} {published.day} · {_clock(published)}",
                fontsize=10.7, color=muted)
    figure.text(0.08, 0.835,
                f"{data.model_name} issued {issued:%a, %b} {issued.day} · {_clock(issued)}",
                fontsize=10.7, color=muted)
    for left, title, rows, color in ((0.08, "OFFICIAL ERCOT PEAK", data.official, teal),
                                    (0.55, data.model_name.upper() + " PEAK", data.model, purple)):
        peak = max(rows, key=lambda hour: hour.mw)
        figure.text(left, 0.785, title, fontsize=10.5, fontweight="bold", color=muted)
        figure.text(left, 0.742, f"{peak.mw:,.0f} MW", fontsize=21, fontweight="bold", color=color)
        figure.text(left, 0.713, _interval_label(peak), fontsize=9.5, color=muted)

    axis = figure.add_axes((0.12, 0.28, 0.82, 0.40), facecolor="white")
    axis.set_axisbelow(True)
    axis.grid(axis="y", color=grid, linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(grid)
    axis.tick_params(axis="both", colors=muted, labelsize=10, length=0, pad=8)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:,.0f}"))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    axis.set_ylabel("Load (MW)", fontsize=11, color=muted, labelpad=9)
    edges = [mdates.date2num(hour.start) for hour in data.official] + [mdates.date2num(data.official[-1].end)]
    for rows, label, color, style in ((data.official, "Official ERCOT", teal, "-"),
                                      (data.model, data.model_name + " · experimental", purple, "--")):
        axis.stairs([hour.mw for hour in rows], edges, baseline=None, color=color,
                    linewidth=2.5, linestyle=style, label=label)
    axis.set_xlim(edges[0], edges[-1])
    ticks = [data.official[index].start for index in range(0, 24, 4)] + [data.official[-1].end]
    axis.set_xticks([mdates.date2num(stamp) for stamp in ticks],
                    [f"{stamp.astimezone(CENTRAL):%b %d}\n{stamp.astimezone(CENTRAL):%I %p %Z}" for stamp in ticks])
    values = [hour.mw for hour in (*data.official, *data.model)]
    pad = max((max(values) - min(values)) * 0.20, max(values) * 0.02, 1)
    axis.set_ylim(max(0, min(values) - pad), max(values) + pad)
    axis.legend(loc="upper left", fontsize=10, framealpha=1, edgecolor=grid)
    figure.text(0.12, 0.174, f"Average {data.model_name} − ERCOT: {data.mean_difference_mw:+,.0f} MW",
                fontsize=11, fontweight="bold", color=navy)
    figure.text(0.12, 0.142, "Forecast differences are not accuracy scores; actual load is needed.", fontsize=10.5, color=muted)
    cutoff = data.cutoff.astimezone(CENTRAL)
    figure.text(0.08, 0.095, f"Actual inputs complete through {cutoff:%b} {cutoff.day}, {cutoff.year} · {_clock(cutoff)}",
                fontsize=9.5, color=muted)
    figure.text(0.08, 0.061, "Official source: ERCOT NP3-561-CD · SystemTotal · Hourly intervals in Central time",
                fontsize=9.5, color=muted)
    figure.text(0.08, 0.030, "Both curves cover the same 24 intervals. No smoothing or missing-hour filling.",
                fontsize=9.5, color=muted)
    return figure


def render_forecast_comparison_chart(report: dict[str, Any]) -> dict[str, str]:
    """Render one PNG chart of two matching forecasts, without fetching or saving.

    A comparison failure raises visibly. The caller must not replace this chart
    with a different horizon or quietly draw only one of its two forecasts.
    """
    data = _prepare_comparison_plot_data(report)
    figure = None
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        figure = _build_comparison_figure(data)
        buffer = BytesIO()
        FigureCanvasAgg(figure).print_png(buffer, metadata={
            "Title": f"ERCOT and {data.model_name} hourly load forecast comparison",
            "Source": data.source_url, "SourceSHA256": data.source_hash,
            "Report": data.report_name, "Published": data.published.isoformat(),
            "Retrieved": data.retrieved.isoformat(), "ModelIssued": data.issued.isoformat(),
            "Model": data.model_name, "ActualInputCutoff": data.cutoff.isoformat(),
        }, pil_kwargs={"optimize": True, "compress_level": 9})
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        if len(encoded) > MAX_IMAGE_BASE64_LENGTH:
            raise ForecastComparisonChartError("The combined forecast chart exceeds the attachment size limit.")
        return {"base64": encoded, "mime_type": "image/png", "file_name": "ercot-model-comparison.png"}
    except ForecastComparisonChartError:
        raise
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise ForecastComparisonChartError("The combined forecast chart renderer is unavailable.") from exc
    finally:
        if figure is not None:
            figure.clear()
