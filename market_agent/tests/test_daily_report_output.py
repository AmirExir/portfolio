from __future__ import annotations

import datetime as dt
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import pandas as pd

from market_agent import daily_ml_forecast_report as report


def _research_buy_row(**overrides) -> dict:
    row = {
        "Symbol": "SNDK",
        "Forecast Return %": 12.5,
        "Model Call": "Strong Buy",
        "Selected Model": "Ridge",
        "Reliability": "High",
        "Smart Policy": "Hold / Watch",
        "Policy Score": 1.25,
        "Policy Target %": 0.0,
        "Pre-Portfolio Target %": 3.0,
        "Policy Allocation Eligible": False,
        "Portfolio Allocation Blocked": True,
        "Portfolio State Verified": False,
        "Portfolio Covariance Verified": True,
        "Portfolio Classification Verified": True,
        "Portfolio Gross %": 0.0,
        "Portfolio Cash %": 100.0,
        "Portfolio Binding Constraints": "unverified_portfolio_state",
        "Probability Up %": 82.0,
        "Model Edge %": 32.0,
        "Expected Error %": 8.0,
        "Direction Hit Rate %": 60.0,
        "Validation MAE %": 5.0,
        "Validation Samples": 30,
        "Calibration Error %": 12.0,
        "Brier Score": 0.20,
        "Primary Pattern": "Uptrend",
        "As Of Session": "2026-08-05",
        "Target Session": "2026-09-17",
        "Signal Tier": "Model-Confirmed Buy",
    }
    row.update(overrides)
    return row


class DailyReportOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.args = report.build_parser().parse_args(
            ["--run-profile", "quick", "--no-rl-policy"]
        )

    def test_scheduled_profile_is_single_pass_and_bounded(self) -> None:
        raw_args = ["--run-profile", "scheduled"]
        args = report.apply_run_profile(
            report.build_parser().parse_args(raw_args),
            raw_args,
        )

        self.assertEqual(args.run_profile, "scheduled")
        self.assertEqual(args.sequence_model, "off")
        self.assertEqual(args.short_horizons, "")
        self.assertEqual(args.short_sequence_model, "off")
        self.assertTrue(args.no_optimize)

    def test_research_buy_remains_visible_when_execution_is_blocked(self) -> None:
        row = _research_buy_row()

        self.assertTrue(report.is_threshold_buy(row, self.args))
        rendered = report.build_market_report([row], [], self.args)

        self.assertEqual(rendered["top_buys"], ["SNDK"])
        self.assertIn("Model-Confirmed Buy Forecasts", rendered["report_text"])
        self.assertIn("SNDK: +12.50% through 2026-09-17", rendered["report_text"])
        self.assertIn("indicative 3.0%, executable 0.0%", rendered["report_text"])
        self.assertIn(
            "Research visibility is separate from execution authorization",
            rendered["report_text"],
        )

    def test_rl_policy_cannot_reenter_the_published_buy_list(self) -> None:
        row = _research_buy_row(**{"Selected Model": "RL Policy"})

        self.assertFalse(report.is_threshold_buy(row, self.args))
        rendered = report.build_market_report([row], [], self.args)

        self.assertEqual(rendered["top_buys"], [])

    def test_rl_primary_request_is_accepted_only_as_shadow(self) -> None:
        args = report.build_parser().parse_args(
            ["--primary-model", "RL Policy", "--include-rl-policy"]
        )

        self.assertEqual(report.primary_model_from_args(args), "Best Validation")
        self.assertTrue(report.include_rl_policy_from_args(args))

    def test_poorly_calibrated_forecast_is_not_promoted(self) -> None:
        row = _research_buy_row(
            **{
                "Direction Hit Rate %": 20.0,
                "Validation MAE %": 4.0,
                "Validation Samples": 30,
                "Calibration Error %": 75.0,
                "Brier Score": 0.70,
            }
        )

        enriched = report.enrich_rows_with_signal_metadata([row], self.args)[0]

        self.assertEqual(enriched["Reliability"], "Low")
        self.assertEqual(enriched["Policy Target %"], 0.0)
        self.assertEqual(enriched["Signal Tier"], "Neutral / Monitor")
        self.assertFalse(report.is_threshold_buy(enriched, self.args))

    def test_each_reliability_gate_fails_closed(self) -> None:
        invalid_metrics = {
            "too_few_samples": {"Validation Samples": 7},
            "weak_hit_rate": {"Direction Hit Rate %": 49.9},
            "poor_calibration": {"Calibration Error %": 20.1},
            "poor_brier": {"Brier Score": 0.301},
            "missing_metric": {"Calibration Error %": None},
            "malformed_metric": {"Brier Score": "not-a-number"},
        }

        for case, overrides in invalid_metrics.items():
            with self.subTest(case=case):
                row = _research_buy_row(**overrides)
                self.assertEqual(report.reliability_grade(row), "Low")
                self.assertFalse(report.is_threshold_buy(row, self.args))

    def test_low_reliability_cannot_leak_into_sell_or_policy_watch(self) -> None:
        low_quality = _research_buy_row(
            **{
                "Model Call": "Sell",
                "Forecast Return %": -12.5,
                "Smart Policy": "Sell / Avoid",
                "Policy Score": -2.0,
                "Reliability": "High",
                "Direction Hit Rate %": 40.0,
            }
        )

        self.assertEqual(report.reliability_grade(low_quality), "Low")
        self.assertFalse(report.is_threshold_sell(low_quality, self.args))
        self.assertFalse(report.is_policy_watch_sell(low_quality, self.args))

    def test_calibrated_forecast_remains_visible_when_execution_is_blocked(self) -> None:
        row = _research_buy_row(
            **{
                "Direction Hit Rate %": 60.0,
                "Validation MAE %": 5.0,
                "Validation Samples": 30,
                "Calibration Error %": 12.0,
                "Brier Score": 0.20,
            }
        )

        enriched = report.enrich_rows_with_signal_metadata([row], self.args)[0]

        self.assertEqual(enriched["Reliability"], "High")
        self.assertTrue(report.is_threshold_buy(enriched, self.args))

    def test_low_reliability_row_cannot_become_report_headline(self) -> None:
        rejected = _research_buy_row(
            **{
                "Symbol": "BAD",
                "Policy Score": 100.0,
                "Direction Hit Rate %": 20.0,
            }
        )
        qualified = _research_buy_row(
            **{
                "Symbol": "GOOD",
                "Policy Score": 1.0,
            }
        )
        rows = report.enrich_rows_with_signal_metadata(
            [rejected, qualified],
            self.args,
        )

        rendered = report.build_market_report(rows, [], self.args)

        headline = rendered["report_text"].split(
            "Highest-Conviction Qualified Research Forecast\n",
            1,
        )[1].split("\n\n", 1)[0]
        self.assertIn("GOOD:", headline)
        self.assertNotIn("BAD:", headline)

    def test_incomplete_daily_bar_is_removed_before_forecasting(self) -> None:
        frame = pd.DataFrame(
            {"close": [100.0, 101.0]},
            index=pd.to_datetime(["2026-08-05", "2026-08-06"]),
        )

        completed, latest, expected, lag = report.completed_daily_market_data(
            frame,
            "SNDK",
            now=dt.datetime(2026, 8, 6, 13, 45, tzinfo=dt.timezone.utc),
        )

        self.assertEqual(completed.index.max().date(), dt.date(2026, 8, 5))
        self.assertEqual(latest, dt.date(2026, 8, 5))
        self.assertEqual(expected, dt.date(2026, 8, 5))
        self.assertEqual(lag, 0)

    def test_market_context_excludes_an_incomplete_session(self) -> None:
        raw = pd.DataFrame(
            {"Close": [100.0, 101.0]},
            index=pd.to_datetime(["2026-08-05", "2026-08-06"]),
        )
        with patch.object(report, "_yf_download", return_value=raw):
            context = report.load_market_context(
                400,
                now=dt.datetime(
                    2026,
                    8,
                    6,
                    13,
                    45,
                    tzinfo=dt.timezone.utc,
                ),
            )

        self.assertEqual(context.index.max().date(), dt.date(2026, 8, 5))

    def test_stale_cached_market_data_is_rejected_explicitly(self) -> None:
        frame = pd.DataFrame(
            {"close": [100.0, 105.0]},
            index=pd.to_datetime(["2026-07-31", "2026-08-06"]),
        )

        with self.assertRaisesRegex(ValueError, "stale OHLCV data"):
            report.require_fresh_daily_market_data(
                frame,
                "SNDK",
                1,
                now=dt.datetime(
                    2026,
                    8,
                    6,
                    13,
                    45,
                    tzinfo=dt.timezone.utc,
                ),
            )

    def test_etfs_do_not_call_the_corporate_earnings_provider(self) -> None:
        with patch.object(
            report,
            "fetch_yfinance_earnings_payload",
            side_effect=AssertionError("ETF earnings provider should not run"),
        ):
            context = report.earnings_context_for_symbol(
                "SPY",
                self.args,
            )

        self.assertEqual(context, {})

    def test_json_only_entrypoint_always_emits_parseable_output(self) -> None:
        row = _research_buy_row()
        timings = {
            "total_seconds": 0.1,
            "context_seconds": 0.0,
            "symbols_ranked": 1,
            "symbols_attempted": 1,
            "symbols_cached": 1,
            "symbol_timings": [],
        }
        stdout = io.StringIO()

        with (
            patch.object(sys, "argv", ["daily_ml_forecast_report.py", "--json-only", "--run-profile", "quick", "--no-rl-policy"]),
            patch.object(report, "run_rankings", return_value=([row], [], [], timings)),
            patch.object(report, "run_short_horizon_reports", return_value=[]),
            patch.object(report, "write_outputs", return_value={"txt": "/tmp/report.txt", "json": "/tmp/report.json"}),
            redirect_stdout(stdout),
        ):
            exit_code = report.main()

        self.assertEqual(exit_code, 0)
        output_text = stdout.getvalue().strip()
        self.assertTrue(output_text)
        payload = json.loads(output_text)
        self.assertEqual(payload["top_buys"], ["SNDK"])
        self.assertIn("Generated:", payload["telegram_text"])
        self.assertTrue(payload["generated_at"])
        self.assertTrue(payload["generated_at"].endswith("Z"))
        self.assertEqual(payload["horizon_days"], 30)
        self.assertEqual(payload["universe"], "Stocks + crypto + commodities")
        self.assertEqual(payload["universe_count"], len(report.DEFAULT_SYMBOLS))
        self.assertEqual(payload["ranked_count"], 1)
        self.assertEqual(payload["as_of_sessions"], ["2026-08-05"])
        self.assertEqual(payload["primary_model"], "Best Validation")
        self.assertEqual(payload["rl_mode"], "off")
        self.assertEqual(
            payload["signal_threshold"]["minimum_validation_samples"],
            8,
        )
        self.assertEqual(
            payload["signal_threshold"]["minimum_direction_hit_rate_pct"],
            50.0,
        )
        self.assertIn(
            f"Universe: 1 ranked / {len(report.DEFAULT_SYMBOLS)} configured",
            payload["telegram_text"],
        )
        self.assertIn(
            "Data as of completed session: 2026-08-05",
            payload["telegram_text"],
        )


if __name__ == "__main__":
    unittest.main()
