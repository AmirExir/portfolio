from __future__ import annotations

import datetime as dt
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
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
        "Validation Is OOS": True,
        "Direction Hit Rate %": 60.0,
        "Direction Skill %": 10.0,
        "Validation MAE %": 5.0,
        "Zero-Return MAE %": 6.5,
        "MAE Skill Score": 0.23,
        "Validation Samples": 30,
        "Nonoverlapping Validation Samples": 10,
        "Effective Validation Samples": 9.5,
        "Calibration Error %": 12.0,
        "Brier Score": 0.20,
        "Brier Baseline Score": 0.25,
        "Brier Skill Score": 0.20,
        "Primary Pattern": "Uptrend",
        "As Of Session": "2026-08-05",
        "Target Session": "2026-09-17",
        "Signal Tier": "Model-Qualified Buy",
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

    def test_overnight_profile_is_one_optimized_pass_with_runtime_budget(self) -> None:
        raw_args = ["--run-profile", "overnight"]
        args = report.apply_run_profile(
            report.build_parser().parse_args(raw_args),
            raw_args,
        )

        self.assertEqual(args.run_profile, "overnight")
        self.assertEqual(args.history_days, 1825)
        self.assertEqual(args.sequence_model, "adaptive")
        self.assertEqual(args.short_horizons, "")
        self.assertEqual(args.short_sequence_model, "off")
        self.assertFalse(args.no_optimize)
        self.assertFalse(args.include_rl_policy)
        self.assertEqual(args.runtime_budget_minutes, 240.0)
        self.assertEqual(args.adaptive_sequence_exploration_quota, 2)

    def test_runtime_deadlines_reserve_30_minutes_for_overnight_finalization(
        self,
    ) -> None:
        workflow_start = 100.0
        args = report.build_parser().parse_args(
            ["--runtime-budget-minutes", "240"]
        )

        publication_deadline = report.runtime_deadline_from_args(
            args,
            workflow_start,
        )
        ranking_deadline = (
            report.ranking_deadline_from_publication_deadline(
                publication_deadline,
                workflow_start,
            )
        )

        self.assertEqual(publication_deadline, workflow_start + 240 * 60)
        self.assertEqual(ranking_deadline, workflow_start + 210 * 60)
        self.assertEqual(report.EXTERNAL_RUNTIME_CEILING_MINUTES, 270.0)

    def test_small_runtime_budget_keeps_proportional_finalization_window(
        self,
    ) -> None:
        workflow_start = 50.0
        args = report.build_parser().parse_args(
            ["--runtime-budget-minutes", "20"]
        )

        publication_deadline = report.runtime_deadline_from_args(
            args,
            workflow_start,
        )
        ranking_deadline = (
            report.ranking_deadline_from_publication_deadline(
                publication_deadline,
                workflow_start,
            )
        )

        self.assertEqual(publication_deadline, workflow_start + 20 * 60)
        self.assertEqual(ranking_deadline, workflow_start + 17.5 * 60)

    def test_adaptive_sequence_selector_deduplicates_and_selects_one_family(self) -> None:
        args = report.build_parser().parse_args(
            [
                "--run-profile",
                "overnight",
                "--adaptive-sequence-min-wins",
                "2",
                "--adaptive-sequence-min-share",
                "0.5",
            ]
        )

        def metrics(mae: float) -> dict:
            return {
                "holdout_mae_pct": mae,
                "validation_is_oos": True,
                "mae_skill_score": 0.10,
                "brier_skill_score": 0.10,
                "direction_skill_pct": 5.0,
                "holdout_nonoverlapping_samples": 2,
            }

        def snapshot(symbol: str, as_of: str, lstm_mae: float) -> dict:
            return {
                "symbol": symbol,
                "as_of_session": as_of,
                "models": {
                    "Ridge": {"metrics": metrics(5.0)},
                    "Ensemble": {"metrics": {"holdout_mae_pct": 3.0}},
                    "LSTM": {"metrics": metrics(lstm_mae)},
                    "Transformer": {"metrics": metrics(6.0)},
                },
            }

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            payloads = [
                {
                    "generated_at": "2026-06-02T09-15-00Z",
                    "run_complete": True,
                    "horizon_days": 30,
                    "snapshots": [snapshot("AAA", "2026-06-01", 4.0)],
                },
                {
                    "generated_at": "2026-06-02T09-16-00Z",
                    "run_complete": True,
                    "horizon_days": 30,
                    "snapshots": [snapshot("AAA", "2026-06-01", 4.0)],
                },
                {
                    "generated_at": "2026-07-16T09-15-00Z",
                    "run_complete": True,
                    "horizon_days": 30,
                    "snapshots": [snapshot("AAA", "2026-07-15", 4.5)],
                },
            ]
            for index, payload in enumerate(payloads):
                (output_dir / f"ml_forecast_rankings_2026-08-0{index + 1}.json").write_text(
                    json.dumps(payload)
                )

            selected = report.adaptive_sequence_models_from_reports(
                args,
                output_dir,
            )

        self.assertEqual(selected, {"AAA": "lstm"})

    def test_adaptive_report_timestamp_supports_filename_style_clock(self) -> None:
        parsed = report.parse_report_generated_at("2026-08-28T07-47-59Z")

        self.assertEqual(
            parsed,
            dt.datetime(2026, 8, 28, 7, 47, 59, tzinfo=dt.timezone.utc),
        )

    def test_adaptive_equity_spacing_respects_market_holiday(self) -> None:
        distance = report.session_distance_for_symbol(
            dt.date(2026, 1, 16),
            dt.date(2026, 1, 20),
            "AAPL",
        )

        self.assertEqual(distance, 1)

    def test_adaptive_manual_symbols_request_both_sequence_families(self) -> None:
        args = report.build_parser().parse_args(
            ["--adaptive-sequence-symbols", "AAPL,BTC-USD"]
        )

        selected = report.adaptive_sequence_models_from_reports(
            args,
            Path("does-not-need-to-exist"),
        )

        self.assertEqual(selected, {"AAPL": "both", "BTC-USD": "both"})

    def test_adaptive_exploration_is_deterministic_and_bounded(self) -> None:
        args = report.build_parser().parse_args(
            [
                "--symbols",
                "AAPL,MSFT,NVDA,AMD",
                "--adaptive-sequence-exploration-quota",
                "2",
                "--adaptive-sequence-exploration-seed",
                "2026-08-28",
            ]
        )

        first = report.deterministic_adaptive_exploration_models(args, {})
        second = report.deterministic_adaptive_exploration_models(args, {})

        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertTrue(set(first.values()) <= {"lstm", "transformer"})

    def test_adaptive_selector_does_not_count_invalid_reports_toward_limit(
        self,
    ) -> None:
        args = report.build_parser().parse_args(
            [
                "--run-profile",
                "overnight",
                "--adaptive-sequence-report-limit",
                "2",
                "--adaptive-sequence-min-wins",
                "2",
                "--adaptive-sequence-min-share",
                "1.0",
            ]
        )

        def valid_metrics(mae: float) -> dict:
            return {
                "holdout_mae_pct": mae,
                "validation_is_oos": True,
                "mae_skill_score": 0.2,
                "brier_skill_score": 0.1,
                "direction_skill_pct": 4.0,
                "holdout_nonoverlapping_samples": 2,
            }

        def valid_snapshot(as_of: str) -> dict:
            return {
                "symbol": "AAA",
                "as_of_session": as_of,
                "models": {
                    "Ridge": {"metrics": valid_metrics(5.0)},
                    "LSTM": {"metrics": valid_metrics(4.0)},
                },
            }

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            payloads = [
                {
                    "generated_at": "2026-08-20T09-15-00Z",
                    "run_complete": True,
                    "horizon_days": 30,
                    "snapshots": [
                        {
                            "symbol": "BBB",
                            "as_of_session": "2026-08-19",
                            "models": {"Ridge": {"metrics": valid_metrics(5.0)}},
                        }
                    ],
                },
                {
                    "generated_at": "2026-07-16T09-15-00Z",
                    "run_complete": True,
                    "horizon_days": 30,
                    "snapshots": [valid_snapshot("2026-07-15")],
                },
                {
                    "generated_at": "2026-06-02T09-15-00Z",
                    "run_complete": True,
                    "horizon_days": 30,
                    "snapshots": [valid_snapshot("2026-06-01")],
                },
            ]
            for index, payload in enumerate(payloads):
                (output_dir / f"ml_forecast_rankings_2026-0{index + 6}.json").write_text(
                    json.dumps(payload)
                )

            selected = report.adaptive_sequence_models_from_reports(
                args,
                output_dir,
            )

        self.assertEqual(selected, {"AAA": "lstm"})

    def test_adaptive_selector_rejects_overlapping_or_non_oos_wins(self) -> None:
        args = report.build_parser().parse_args(
            [
                "--run-profile",
                "overnight",
                "--adaptive-sequence-min-wins",
                "2",
            ]
        )

        def snapshot(as_of: str, *, validation_is_oos: bool = True) -> dict:
            base = {
                "validation_is_oos": validation_is_oos,
                "mae_skill_score": 0.2,
                "brier_skill_score": 0.2,
                "direction_skill_pct": 5.0,
                "holdout_nonoverlapping_samples": 2,
            }
            return {
                "symbol": "AAA",
                "as_of_session": as_of,
                "models": {
                    "Ridge": {"metrics": {**base, "holdout_mae_pct": 5.0}},
                    "LSTM": {"metrics": {**base, "holdout_mae_pct": 4.0}},
                },
            }

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            payloads = [
                {
                    "generated_at": "2026-08-21T09-15-00Z",
                    "run_complete": True,
                    "horizon_days": 30,
                    "snapshots": [snapshot("2026-08-20")],
                },
                {
                    "generated_at": "2026-08-20T09-15-00Z",
                    "run_complete": True,
                    "horizon_days": 30,
                    "snapshots": [snapshot("2026-08-19")],
                },
                {
                    "generated_at": "2026-06-02T09-15-00Z",
                    "run_complete": True,
                    "horizon_days": 30,
                    "snapshots": [
                        snapshot("2026-06-01", validation_is_oos=False)
                    ],
                },
            ]
            for index, payload in enumerate(payloads):
                (output_dir / f"ml_forecast_rankings_2026-test-{index}.json").write_text(
                    json.dumps(payload)
                )

            selected = report.adaptive_sequence_models_from_reports(
                args,
                output_dir,
            )

        self.assertEqual(selected, {})

    def test_research_buy_remains_visible_when_execution_is_blocked(self) -> None:
        row = _research_buy_row()

        self.assertTrue(report.is_threshold_buy(row, self.args))
        rendered = report.build_market_report([row], [], self.args)

        self.assertEqual(rendered["top_buys"], ["SNDK"])
        self.assertIn("Model-Qualified Buy Forecasts", rendered["report_text"])
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
        self.assertEqual(enriched["Signal Tier"], "Unqualified Candidate Buy")
        self.assertTrue(enriched["Unqualified Candidate"])
        self.assertFalse(report.is_threshold_buy(enriched, self.args))
        rendered = report.build_market_report([enriched], [], self.args)
        self.assertEqual(rendered["unqualified_candidate_buys"], ["SNDK"])
        self.assertIn("Unqualified Model Candidates", rendered["report_text"])
        self.assertIn("MAE skill", rendered["report_text"])
        self.assertIn("direction skill", rendered["report_text"])
        self.assertIn(
            "failed gates [direction_hit_rate_below_minimum, "
            "calibration_error_above_maximum, brier_score_above_maximum]",
            rendered["report_text"],
        )
        self.assertEqual(
            enriched["Qualification Gate Failures"],
            [
                "direction_hit_rate_below_minimum",
                "calibration_error_above_maximum",
                "brier_score_above_maximum",
            ],
        )

    def test_each_reliability_gate_fails_closed(self) -> None:
        invalid_metrics = {
            "too_few_samples": {"Validation Samples": 7},
            "missing_oos_flag": {"Validation Is OOS": None},
            "non_oos_validation": {"Validation Is OOS": False},
            "non_literal_oos_validation": {"Validation Is OOS": 1},
            "weak_hit_rate": {"Direction Hit Rate %": 49.9},
            "no_direction_skill": {"Direction Skill %": 0.0},
            "poor_calibration": {"Calibration Error %": 20.1},
            "poor_brier": {"Brier Score": 0.301},
            "too_few_nonoverlapping": {
                "Nonoverlapping Validation Samples": 7
            },
            "no_mae_skill": {"MAE Skill Score": 0.0},
            "no_brier_skill": {"Brier Skill Score": -0.01},
            "missing_metric": {"Calibration Error %": None},
            "malformed_metric": {"Brier Score": "not-a-number"},
        }

        for case, overrides in invalid_metrics.items():
            with self.subTest(case=case):
                row = _research_buy_row(**overrides)
                self.assertEqual(report.reliability_grade(row), "Low")
                self.assertFalse(report.is_threshold_buy(row, self.args))

        missing_oos = _research_buy_row()
        missing_oos.pop("Validation Is OOS")
        self.assertEqual(report.reliability_grade(missing_oos), "Low")

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
        with patch.object(
            report,
            "_yf_download",
            return_value=raw,
        ) as download:
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
        self.assertEqual(download.call_args.kwargs["start"], "2025-07-02")
        self.assertEqual(
            context.attrs["history_coverage"]["requested_calendar_days"],
            400,
        )

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
            "run_complete": True,
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
        self.assertIs(
            payload["signal_threshold"]["oos_validation_required_value"],
            True,
        )
        self.assertIs(
            payload["signal_threshold"][
                "oos_validation_must_be_literal_boolean"
            ],
            True,
        )
        self.assertEqual(
            payload["signal_threshold"][
                "minimum_direction_skill_pct_exclusive"
            ],
            0.0,
        )
        self.assertIn(
            f"Universe: 1 ranked / {len(report.DEFAULT_SYMBOLS)} configured",
            payload["telegram_text"],
        )
        self.assertIn(
            "Data as of completed session: 2026-08-05",
            payload["telegram_text"],
        )

    def test_incomplete_runtime_budget_run_is_not_published(self) -> None:
        row = _research_buy_row()
        timings = {
            "total_seconds": 13_800.0,
            "context_seconds": 1.0,
            "symbols_ranked": 70,
            "symbols_attempted": 80,
            "symbols_cached": 0,
            "symbol_timings": [],
            "runtime_budget_exceeded": True,
            "run_complete": False,
        }
        stdout = io.StringIO()

        with (
            patch.object(
                sys,
                "argv",
                [
                    "daily_ml_forecast_report.py",
                    "--json-only",
                    "--run-profile",
                    "overnight",
                    "--send-telegram",
                ],
            ),
            patch.object(
                report,
                "run_rankings",
                return_value=([row], ["runtime budget reached"], [], timings),
            ),
            patch.object(report, "run_short_horizon_reports") as short_run,
            patch.object(report, "write_outputs") as write_outputs,
            patch.object(report, "send_telegram") as send_telegram,
            redirect_stdout(stdout),
        ):
            exit_code = report.main()

        self.assertEqual(exit_code, 2)
        short_run.assert_not_called()
        write_outputs.assert_not_called()
        send_telegram.assert_not_called()
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["run_complete"])
        self.assertEqual(payload["paths"], {})

    def test_missing_literal_completeness_suppresses_publication(self) -> None:
        row = _research_buy_row()
        timings = {
            "total_seconds": 1.0,
            "context_seconds": 0.0,
            "symbols_ranked": 1,
            "symbols_attempted": 1,
            "symbol_timings": [],
        }
        stdout = io.StringIO()

        with (
            patch.object(
                sys,
                "argv",
                [
                    "daily_ml_forecast_report.py",
                    "--json-only",
                    "--run-profile",
                    "quick",
                    "--send-telegram",
                ],
            ),
            patch.object(
                report,
                "run_rankings",
                return_value=([row], [], [], timings),
            ),
            patch.object(report, "run_short_horizon_reports") as short_run,
            patch.object(report, "write_outputs") as write_outputs,
            patch.object(report, "send_telegram") as send_telegram,
            redirect_stdout(stdout),
        ):
            exit_code = report.main()

        self.assertEqual(exit_code, 2)
        short_run.assert_not_called()
        write_outputs.assert_not_called()
        send_telegram.assert_not_called()
        self.assertIs(json.loads(stdout.getvalue())["run_complete"], False)

    def test_incomplete_short_horizon_suppresses_all_publication(self) -> None:
        row = _research_buy_row()
        main_timings = {
            "total_seconds": 10.0,
            "context_seconds": 1.0,
            "symbols_ranked": 1,
            "symbols_attempted": 1,
            "symbols_cached": 0,
            "symbol_timings": [],
            "runtime_budget_exceeded": False,
            "run_complete": True,
        }
        short_report = {
            "horizon_days": 1,
            "sequence_model": "off",
            "rows": [row],
            "errors": ["runtime deadline reached"],
            "snapshots": [],
            "timings": {
                "runtime_budget_exceeded": True,
                "run_complete": False,
            },
        }
        stdout = io.StringIO()

        with (
            patch.object(
                sys,
                "argv",
                [
                    "daily_ml_forecast_report.py",
                    "--json-only",
                    "--run-profile",
                    "custom",
                    "--short-horizons",
                    "1",
                    "--send-telegram",
                ],
            ),
            patch.object(
                report,
                "run_rankings",
                return_value=([row], [], [], main_timings),
            ),
            patch.object(
                report,
                "run_short_horizon_reports",
                return_value=[short_report],
            ),
            patch.object(report, "write_outputs") as write_outputs,
            patch.object(report, "send_telegram") as send_telegram,
            redirect_stdout(stdout),
        ):
            exit_code = report.main()

        self.assertEqual(exit_code, 2)
        write_outputs.assert_not_called()
        send_telegram.assert_not_called()
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["run_complete"])
        self.assertEqual(payload["paths"], {})

    def test_deadline_reached_before_finalization_suppresses_mutations(self) -> None:
        row = _research_buy_row()
        timings = {
            "total_seconds": 10.0,
            "context_seconds": 1.0,
            "symbols_ranked": 1,
            "symbols_attempted": 1,
            "symbols_cached": 0,
            "symbol_timings": [],
            "runtime_budget_exceeded": False,
            "run_complete": True,
        }
        stdout = io.StringIO()

        with (
            patch.object(
                sys,
                "argv",
                [
                    "daily_ml_forecast_report.py",
                    "--json-only",
                    "--run-profile",
                    "overnight",
                    "--send-telegram",
                ],
            ),
            patch.object(
                report,
                "run_rankings",
                return_value=([row], [], [], timings),
            ),
            patch.object(
                report,
                "run_short_horizon_reports",
                return_value=[],
            ),
            patch.object(
                report,
                "runtime_deadline_reached",
                return_value=True,
            ),
            patch.object(report, "write_outputs") as write_outputs,
            patch.object(report, "send_telegram") as send_telegram,
            redirect_stdout(stdout),
        ):
            exit_code = report.main()

        self.assertEqual(exit_code, 2)
        write_outputs.assert_not_called()
        send_telegram.assert_not_called()
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["run_complete"])

    def test_publication_transaction_rolls_back_if_deadline_elapses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "reports"
            output_dir.mkdir()
            ledger_path = output_dir / "prediction_ledger.jsonl"
            latest_path = output_dir / "ml_forecast_rankings_latest.json"
            ledger_path.write_text("old-ledger\n")
            latest_path.write_text('{"run_complete": true, "old": true}')
            args = report.build_parser().parse_args(
                [
                    "--run-profile",
                    "quick",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            def stage_outputs(*_args, **kwargs) -> dict:
                staging_args = _args[3]
                staging_dir = Path(staging_args.output_dir)
                staged_ledger = staging_dir / "prediction_ledger.jsonl"
                staged_latest = (
                    staging_dir / "ml_forecast_rankings_latest.json"
                )
                staged_timestamp = (
                    staging_dir
                    / "ml_forecast_rankings_2026-08-28T00-00-00Z.json"
                )
                staged_ledger.write_text("old-ledger\nnew-ledger\n")
                staged_latest.write_text('{"run_complete": true}')
                staged_timestamp.write_text('{"run_complete": true}')
                return {
                    "json": str(staged_timestamp),
                    "latest_json": str(staged_latest),
                }

            with (
                patch.object(
                    report,
                    "_write_outputs_to_staging",
                    side_effect=stage_outputs,
                ),
                patch.object(
                    report,
                    "runtime_deadline_reached",
                    side_effect=[False, False, False, True],
                ),
            ):
                with self.assertRaisesRegex(
                    TimeoutError,
                    "elapsed while committing",
                ):
                    report.write_outputs(
                        [],
                        [],
                        "report",
                        args,
                        [],
                        {"run_complete": True},
                        [],
                        deadline_monotonic=100.0,
                    )

            self.assertEqual(ledger_path.read_text(), "old-ledger\n")
            self.assertEqual(
                latest_path.read_text(),
                '{"run_complete": true, "old": true}',
            )
            self.assertFalse(
                (
                    output_dir
                    / "ml_forecast_rankings_2026-08-28T00-00-00Z.json"
                ).exists()
            )

    def test_two_publishers_serialize_live_ledger_copy_and_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "reports"
            output_dir.mkdir()
            ledger_path = output_dir / "prediction_ledger.jsonl"
            ledger_path.write_text("initial\n")
            args = report.build_parser().parse_args(
                [
                    "--run-profile",
                    "quick",
                    "--output-dir",
                    str(output_dir),
                    "--publication-lock-timeout-seconds",
                    "2",
                ]
            )
            first_staged = threading.Event()
            release_first = threading.Event()
            second_staged = threading.Event()
            failures: list[BaseException] = []

            def stage_outputs(*positional, **_kwargs) -> dict:
                staging_dir = Path(positional[3].output_dir)
                staged_ledger = staging_dir / "prediction_ledger.jsonl"
                existing = (
                    staged_ledger.read_text()
                    if staged_ledger.exists()
                    else ""
                )
                publisher = threading.current_thread().name
                if publisher == "publisher-one":
                    first_staged.set()
                    self.assertTrue(release_first.wait(2.0))
                else:
                    second_staged.set()
                staged_ledger.write_text(existing + publisher + "\n")
                staged_json = staging_dir / "ml_forecast_rankings_latest.json"
                staged_json.write_text('{"run_complete": true}')
                return {"latest_json": str(staged_json)}

            def publish() -> None:
                try:
                    report.write_outputs(
                        [],
                        [],
                        "report",
                        args,
                        [],
                        {"run_complete": True},
                        [],
                    )
                except BaseException as exc:
                    failures.append(exc)

            with patch.object(
                report,
                "_write_outputs_to_staging",
                side_effect=stage_outputs,
            ):
                first = threading.Thread(
                    target=publish,
                    name="publisher-one",
                )
                second = threading.Thread(
                    target=publish,
                    name="publisher-two",
                )
                first.start()
                self.assertTrue(first_staged.wait(2.0))
                second.start()
                self.assertFalse(second_staged.wait(0.15))
                release_first.set()
                first.join(2.0)
                second.join(2.0)

            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
            self.assertEqual(failures, [])
            self.assertEqual(
                ledger_path.read_text(),
                "initial\npublisher-one\npublisher-two\n",
            )

    def test_short_mixed_market_horizon_uses_asset_sessions(self) -> None:
        row = _research_buy_row()
        rendered = report.build_market_report(
            [row],
            [],
            self.args,
            short_horizon_reports=[
                {
                    "horizon_days": 1,
                    "sequence_model": "off",
                    "rows": [row],
                    "errors": [],
                    "timings": {"run_complete": True},
                }
            ],
        )

        self.assertIn("1 asset session | sequence model: off", rendered["report_text"])
        self.assertNotIn("1 trading day |", rendered["report_text"])

    def test_successful_transaction_is_complete_at_its_commit_boundary(
        self,
    ) -> None:
        row = _research_buy_row()
        timings = {
            "total_seconds": 1.0,
            "context_seconds": 0.0,
            "symbols_ranked": 1,
            "symbols_attempted": 1,
            "symbols_cached": 0,
            "symbol_timings": [],
            "runtime_budget_exceeded": False,
            "run_complete": True,
        }
        stdout = io.StringIO()

        with (
            patch.object(
                sys,
                "argv",
                [
                    "daily_ml_forecast_report.py",
                    "--json-only",
                    "--run-profile",
                    "overnight",
                    "--send-telegram",
                ],
            ),
            patch.object(
                report,
                "run_rankings",
                return_value=([row], [], [], timings),
            ),
            patch.object(
                report,
                "run_short_horizon_reports",
                return_value=[],
            ),
            patch.object(
                report,
                "write_outputs",
                return_value={"json": "/tmp/committed.json"},
            ) as write_outputs,
            patch.object(
                report,
                "runtime_deadline_reached",
                side_effect=[False, True],
            ),
            patch.object(report, "send_telegram") as send_telegram,
            redirect_stdout(stdout),
        ):
            exit_code = report.main()

        self.assertEqual(exit_code, 0)
        write_outputs.assert_called_once()
        send_telegram.assert_not_called()
        payload = json.loads(stdout.getvalue())
        self.assertIs(payload["run_complete"], True)
        self.assertEqual(payload["paths"], {"json": "/tmp/committed.json"})


if __name__ == "__main__":
    unittest.main()
