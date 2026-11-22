"""Render the public dashboard with deterministic data and no external services."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import pyarrow as pa
from streamlit.testing.v1 import AppTest


class DashboardPresentationTests(unittest.TestCase):
    def test_navigation_and_portfolio_chart_labels(self) -> None:
        source = Path(__file__).resolve().parents[1] / "market_agent_app.py"
        prices = pd.Series([98.0, 99.0, 100.0], index=pd.bdate_range("2026-02-02", periods=3))
        frame = pd.DataFrame({field: prices for field in ("open", "high", "low", "close")})
        frame["volume"] = 1000
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("os.environ", {"MARKET_REPORT_TIMEZONE": "America/Chicago"}),
            patch.object(sys, "path", [str(source.parent), *sys.path]),
            patch("agent.data.get_ohlcv", return_value=frame),
            patch("agent.earnings.load_earnings_payload_file", return_value=Mock(available=False, error_code="fixture")),
            patch("requests.get", return_value=Mock(status_code=404)),
            patch("yfinance.download", return_value=pd.DataFrame()),
        ):
            script = Path(directory) / source.name
            script.write_text(source.read_text())
            reports = script.parent / "reports"
            reports.mkdir()
            # Disabled models emit entirely null columns in real scheduled reports.
            (reports / "ml_forecast_rankings_latest.json").write_text(json.dumps({
                "generated_at": "2026-11-22T02-06-09Z",
                "rows": [{"Symbol": "XYZ", "Forecast Return %": -1.25, "LSTM Return %": None}],
            }))
            app = AppTest.from_file(str(script))
            app.run(timeout=60)

        self.assertFalse(app.exception)
        analysis = app.tabs[3]
        self.assertIn("Generated: Nov 21, 2026 at 8:06 PM CST (UTC-06:00)", [caption.value for caption in analysis.caption])
        self.assertFalse(any("Scheduled market outlook unavailable" in caption.value for caption in analysis.caption))
        ranking = next(table for table in analysis.dataframe if "LSTM Return %" in table.value.columns)
        displayed = pa.ipc.open_stream(ranking.proto.styler.display_values).read_all().to_pandas()
        self.assertEqual(displayed["LSTM Return %"].tolist(), ["Unavailable"])
        self.assertEqual(displayed["Forecast Return %"].tolist(), ["-1.25%"])
        self.assertEqual(
            [tab.label for tab in app.tabs],
            ["Portfolio Balance", "AI-Generated Market Summary", "Most Common Patterns", "Market Analysis"],
        )
        chart = app.tabs[0].get("plotly_chart")[0].proto
        figure = json.loads(chart.spec)
        self.assertEqual([trace["name"] for trace in figure["data"]], ["Equity", "Cash"])
        self.assertEqual(chart.theme, "")
        layout = figure["layout"]
        self.assertEqual(layout["paper_bgcolor"], "#ffffff")
        self.assertEqual(layout["legend"]["font"]["color"], "#0f172a")
        for axis in ("xaxis", "yaxis"):
            self.assertEqual(layout[axis]["title"]["font"]["color"], "#0f172a")
            self.assertEqual(layout[axis]["tickfont"]["color"], "#0f172a")
