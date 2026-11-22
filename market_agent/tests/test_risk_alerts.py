from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from market_agent import daily_ml_forecast_report as report
from market_agent import risk_alerts as alerts
from market_agent.agent import data
from market_agent.forecast_cache import frame_fingerprint


def row(symbol: str, day: str, price: float, sell: bool = True) -> dict:
    return {
        "Symbol": symbol,
        "As Of Session": day,
        "Last Price": price,
        "Smart Policy": "Sell / Avoid" if sell else "Hold / Watch",
        "Model Call": "Neutral / No Edge",
        "Forecast Return %": 2.14,
        "Reliability": "Low",
        "Policy Trend Score": -0.65,
        "Policy Momentum Score": -0.55,
    }


def evaluate(rows: list[dict], state: dict, observations: dict | None = None) -> dict:
    return alerts.evaluate_watch(
        rows, observations or {}, state, [r["Symbol"] for r in rows], {}
    )


class RiskAlertTests(unittest.TestCase):
    def test_new_ongoing_and_worsening_are_independent_of_model_and_row_cap(self):
        state = {}
        symbols = [f"TEST{i}" for i in range(7)]
        evaluate([row(s, "2026-11-06", 181, False) for s in symbols], state)
        rows = [row(s, "2026-11-09", 178.67) for s in symbols]
        first = evaluate(rows, state)
        self.assertEqual(["New"] * 7, [w["status"] for w in first["warnings"]])
        for symbol in symbols:
            self.assertIn(symbol, alerts.format_watch(first))
        self.assertEqual(
            {"Ongoing"}, {w["status"] for w in evaluate(rows, state)["warnings"]}
        )
        frame = pd.Series(
            [181 + i % 3 for i in range(21)] + [125.53],
            index=pd.date_range(end="2026-11-20", periods=22, freq="B"),
        )
        obs = alerts.observe_closes("TEST0", frame, {}, 0)
        last = [row("TEST0", "2026-11-20", 125.53)]
        self.assertEqual(
            "Escalated", evaluate(last, state, {"TEST0": obs})["warnings"][0]["status"]
        )
        self.assertEqual(
            "Ongoing", evaluate(last, state, {"TEST0": obs})["warnings"][0]["status"]
        )

    def test_legacy_archive_replay_uses_no_future_reports_and_withholds_shib(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for stamp, price in [
                ("2026-11-21T00-32-21Z", 0.00000370),
                ("2026-11-21T11-30-09Z", 0.00000400),
                ("2026-11-25T10-00-00Z", 0.00000999),
            ]:
                doc = {
                    "symbols": ["SHIB-USD"],
                    "run_complete": True,
                    "horizon_days": 30,
                    "rows": [row("SHIB", "2026-11-20", price)],
                }
                (root / f"ml_forecast_rankings_{stamp}.json").write_text(
                    json.dumps(doc)
                )
            state = alerts.publication_history(
                root, 30, "2026-11-22T00-35-09Z", {"SHIB-USD": "SHIB"}
            )
            prices = pd.Series(
                [0.000004] * 21 + [0.00000351],
                index=pd.date_range(end="2026-11-21", periods=22),
            )
            obs = alerts.observe_closes("SHIB-USD", prices, {}, 0)
            self.assertAlmostEqual(-12.25, obs["change_pct"])
            watch = alerts.evaluate_watch(
                [row("SHIB", "2026-11-21", 0.00000351)],
                {"SHIB-USD": obs},
                state,
                ["SHIB-USD"],
                {"SHIB-USD": "SHIB"},
            )
            self.assertEqual([], watch["warnings"])
            self.assertEqual(0, watch["downside_count"])
            self.assertIn("2026-11-20", alerts.format_watch(watch))
            self.assertNotIn("-12.25%", alerts.format_watch(watch))

    def test_unfinalized_crypto_is_not_reused_and_revised_closes_survive_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            cols = {"Close": [0.00000370, 0.00000450]}
            frame = pd.DataFrame(
                cols, index=pd.to_datetime(["2026-11-20", "2026-11-21"])
            )
            revised = frame.copy()
            revised.iloc[0, 0] = 0.00000400
            with (
                patch.object(data, "_cache_root", return_value=Path(tmp)),
                patch.object(
                    data,
                    "_download_ohlcv",
                    side_effect=[frame, revised, pd.DataFrame()],
                ),
            ):
                now = datetime(2026, 11, 21, 7, tzinfo=UTC)
                data.get_ohlcv("SHIB-USD", 30, now=now)
                downloaded = data.get_ohlcv("SHIB-USD", 30, now=now)
                fallback = data.get_ohlcv(
                    "SHIB-USD", 30, now=datetime(2026, 11, 22, 2, tzinfo=UTC)
                )
            self.assertEqual(0.00000400, downloaded.loc["2026-11-20", "close"])
            self.assertEqual(
                [0.00000370, 0.00000400],
                fallback.attrs["close_revisions"]["2026-11-20"],
            )
            self.assertNotIn(pd.Timestamp("2026-11-21"), fallback.index)

    def test_corroboration_requires_all_24_utc_hours_and_the_same_terminal_price(self):
        index = pd.date_range("2026-11-20", periods=24, freq="h", tz="UTC")
        hourly = pd.DataFrame({"Close": [0.0000037] * 24}, index=index)
        candidates = {"2026-11-20": 0.00000370}
        with patch.object(data, "_download_ohlcv", return_value=hourly):
            self.assertEqual(
                candidates, data.corroborate_daily_closes("SHIB-USD", candidates)
            )
            self.assertFalse(
                data.corroborate_daily_closes("SHIB-USD", {"2026-11-20": 0.000004})
            )
        with patch.object(data, "_download_ohlcv", return_value=hourly.iloc[1:]):
            self.assertFalse(data.corroborate_daily_closes("SHIB-USD", candidates))

    def test_hourly_confirmation_clears_only_matching_dispute(self):
        dates = pd.date_range(end="2026-11-21", periods=22)
        prices = pd.Series([0.0000037] * 21 + [0.00000351], index=dates)
        obs = alerts.observe_closes(
            "SHIB-USD", prices, {}, 0, {"2026-11-20": [0.000004, 0.0000037]}
        )
        self.assertEqual({"2026-11-20": 0.0000037}, alerts.disputed_candidates(obs, {}))
        state = {}
        last = [row("SHIB-USD", "2026-11-21", 0.00000351)]
        self.assertTrue(evaluate(last, state, {"SHIB-USD": obs})["data_issues"])
        obs["corroborated_closes"] = {"2026-11-20": 0.0000037}
        self.assertFalse(evaluate(last, state, {"SHIB-USD": obs})["data_issues"])
        prices.iloc[-2] = 0.000004
        revised = alerts.observe_closes("SHIB-USD", prices, {}, 0)
        self.assertTrue(evaluate(last, state, {"SHIB-USD": revised})["data_issues"])

    def test_sub_decimal_price_changes_invalidate_model_cache(self):
        p = pd.Series([0.00000351])
        self.assertNotEqual(
            frame_fingerprint(p)["hash"], frame_fingerprint(p + 1e-12)["hash"]
        )

    def test_archived_report_runs_through_publication_formatter(self):
        root = Path(__file__).resolve().parents[1] / "reports"
        for stamp, expected in [
            ("2026-11-10T00-32-14Z", "New"),
            ("2026-11-22T00-35-09Z", "Escalated"),
        ]:
            archived = json.loads(
                (root / f"ml_forecast_rankings_{stamp}.json").read_text()
            )
            args = report.build_parser().parse_args(
                ["--symbols", ",".join(archived["symbols"])]
            )
            state = alerts.publication_history(root, 30, stamp, report.REPORT_SYMBOLS)
            observations = {
                o["symbol"]: o
                for o in archived.get("price_watch", {}).get("observations", [])
            }
            watch = alerts.evaluate_watch(
                archived["rows"],
                observations,
                state,
                archived["symbols"],
                report.REPORT_SYMBOLS,
            )
            warning = next(w for w in watch["warnings"] if w["symbol"] == "NVDA")
            self.assertEqual(expected, warning["status"])
            message = report.build_telegram_text(
                archived["rows"], [], args, price_watch=watch
            )
            self.assertIn("NVDA at ", message)
            self.assertIn("Forecast: Neutral / No Edge", message)
            if stamp.startswith("2026-11-22"):
                self.assertIn("alert 6.28%", message)
                self.assertIn("TLT at 81.45", message)
                self.assertIn(
                    "SHIB-USD — Conflicting completed closes for 2026-11-20", message
                )
                self.assertNotIn("-12.25%", message)
