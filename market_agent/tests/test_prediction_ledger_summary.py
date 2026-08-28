from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from market_agent.agent.earnings import us_equity_trading_sessions
from market_agent.agent.ledger import (
    OutcomeRecord,
    PredictionLedger,
    PredictionRecord,
    UTC_DAILY_24_7_SESSION_CALENDAR,
)
from market_agent.daily_ml_forecast_report import prediction_ledger_summary


UTC = timezone.utc


class PredictionLedgerSummaryTests(unittest.TestCase):
    def test_metrics_and_shadow_counts_use_exact_provenance_cohorts(
        self,
    ) -> None:
        cohorts = (
            ("base", "us_equity", "SPY", "model-v1", "features-v1", "post-v1", True),
            ("benchmark", "us_equity", "QQQ", "model-v1", "features-v1", "post-v1", True),
            ("model", "us_equity", "SPY", "model-v2", "features-v1", "post-v1", True),
            ("features", "us_equity", "SPY", "model-v1", "features-v2", "post-v1", True),
            ("post", "us_equity", "SPY", "model-v1", "features-v1", "post-v2", True),
            ("delayed", "us_equity", "SPY", "model-v1", "features-v1", "post-v1", False),
            (
                "crypto",
                UTC_DAILY_24_7_SESSION_CALENDAR,
                "BTC-USD",
                "model-v1",
                "features-v1",
                "post-v1",
                False,
            ),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            ledger = PredictionLedger(
                Path(temporary_directory) / "prediction_ledger.jsonl"
            )
            for cohort in cohorts:
                self._append_provenance_cohort(ledger, *cohort)

            summary = prediction_ledger_summary(ledger)

        matched_metrics = [
            item
            for item in summary["metrics"]
            if item["policy_version"] == "rl-shadow-cohort-v1"
        ]
        matched_promotion = [
            item
            for item in summary["promotion"]["horizons"]
            if item["policy_version"] == "rl-shadow-cohort-v1"
        ]
        self.assertEqual(len(matched_metrics), len(cohorts))
        self.assertEqual(len(matched_promotion), len(cohorts))
        self.assertTrue(
            all(item["sample_count"] == 1 for item in matched_metrics)
        )
        self.assertEqual(
            {
                (
                    item["session_calendar"],
                    item["benchmark_symbol"],
                    item["model_version"],
                    item["feature_set_version"],
                    item["postprocessor_version"],
                    item["strict_close_t_eligible"],
                )
                for item in matched_metrics
            },
            {
                (calendar, benchmark, model, features, postprocessor, strict)
                for _, calendar, benchmark, model, features, postprocessor, strict in cohorts
            },
        )
        self.assertTrue(
            all(
                not item["eligible_for_gate_evaluation"]
                for item in matched_promotion
                if not item["strict_close_t_eligible"]
            )
        )

    def test_matured_rl_sessions_are_separate_and_never_auto_promoted(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            ledger = PredictionLedger(
                Path(temporary_directory) / "prediction_ledger.jsonl"
            )
            self._append_matured_shadow_history(
                ledger,
                forecast_context_horizon_sessions=1,
                session_count=60,
                policy_version="rl-shadow-contextual-v3-h1",
            )
            self._append_matured_shadow_history(
                ledger,
                forecast_context_horizon_sessions=30,
                session_count=61,
                policy_version="rl-shadow-contextual-v3-h30",
            )

            summary = prediction_ledger_summary(ledger)

        promotion_by_horizon = {
            item["forecast_context_horizon_days"]: item
            for item in summary["promotion"]["horizons"]
        }
        self.assertEqual(set(promotion_by_horizon), {1, 30})
        self.assertEqual(promotion_by_horizon[1]["shadow_sessions"], 60)
        self.assertEqual(promotion_by_horizon[30]["shadow_sessions"], 61)
        self.assertTrue(
            promotion_by_horizon[1]["eligible_for_gate_evaluation"]
        )
        self.assertTrue(
            promotion_by_horizon[30]["eligible_for_gate_evaluation"]
        )
        self.assertEqual(
            promotion_by_horizon[1]["execution_horizon_days"],
            1,
        )
        self.assertEqual(
            promotion_by_horizon[30]["execution_horizon_days"],
            1,
        )
        self.assertEqual(summary["promotion"]["status"], "shadow")
        self.assertFalse(summary["promotion"]["automatic_promotion"])

        samples_by_horizon = {
            item["forecast_context_horizon_days"]: item["sample_count"]
            for item in summary["metrics"]
            if (
                item["policy_version"].startswith("rl-shadow")
                and item["model_name"] == "RL Policy"
            )
        }
        self.assertEqual(samples_by_horizon, {1: 60, 30: 61})

    def _append_matured_shadow_history(
        self,
        ledger: PredictionLedger,
        *,
        forecast_context_horizon_sessions: int,
        session_count: int,
        policy_version: str,
    ) -> None:
        start_session = pd.Timestamp("2025-01-02").date()
        calendar = us_equity_trading_sessions(
            as_of=datetime(
                2025,
                1,
                2,
                17,
                tzinfo=UTC,
            ),
            future_session_count=(
                session_count + 5
            ),
        )
        sessions = [
            session for session in calendar if session >= start_session
        ][: session_count + 2]

        for index, as_of_session in enumerate(sessions[:session_count]):
            return_start_session = sessions[index + 1]
            target_session = sessions[index + 2]
            prediction_id = f"{policy_version}-{index:03d}"
            forecast_return = 0.01 if index % 2 == 0 else -0.01
            created_at = datetime.combine(
                as_of_session,
                time(hour=12),
                tzinfo=UTC,
            )
            prediction = PredictionRecord(
                prediction_id=prediction_id,
                created_at_utc=created_at,
                data_cutoff_utc=created_at - timedelta(minutes=1),
                as_of_session=as_of_session,
                return_start_session=return_start_session,
                target_session=target_session,
                symbol=f"TEST{index:03d}",
                horizon_sessions=1,
                model_name="RL Policy",
                model_version="execution-aligned-shadow-v2",
                policy_version=policy_version,
                forecast_return=forecast_return,
                target_weight=0.0,
                probability_positive=0.70 if forecast_return > 0 else 0.30,
                feature_set_version="deterministic-test-features-v1",
                feature_hash=(
                    "feature-"
                    f"{forecast_context_horizon_sessions}-{index:03d}"
                ),
                metadata={
                    "shadow_mode": True,
                    "live_eligible": False,
                    "forecast_context_horizon_sessions": (
                        forecast_context_horizon_sessions
                    ),
                    "policy_execution_horizon_sessions": 1,
                },
            )
            outcome = OutcomeRecord(
                outcome_id=f"outcome-{prediction_id}",
                prediction_id=prediction_id,
                recorded_at_utc=datetime.combine(
                    target_session,
                    time(hour=23),
                    tzinfo=UTC,
                ),
                target_session=target_session,
                target_maturity_utc=prediction.target_maturity_utc,
                realized_return=forecast_return / 2.0,
                benchmark_return=0.0,
                data_source="deterministic-test",
            )
            ledger.append_prediction(prediction)
            ledger.append_outcome(outcome)

    def _append_provenance_cohort(
        self,
        ledger: PredictionLedger,
        label: str,
        session_calendar: str,
        benchmark_symbol: str,
        model_version: str,
        feature_set_version: str,
        postprocessor_version: str,
        strict_close_t_eligible: bool,
    ) -> None:
        as_of_session = date(2026, 1, 2)
        if session_calendar == UTC_DAILY_24_7_SESSION_CALENDAR:
            target_session = date(2026, 1, 3)
            data_cutoff = datetime(2026, 1, 3, 0, tzinfo=UTC)
            created_at = datetime(2026, 1, 3, 9, tzinfo=UTC)
        else:
            target_session = date(2026, 1, 5)
            created_at = datetime(2026, 1, 2, 12, tzinfo=UTC)
            data_cutoff = created_at - timedelta(minutes=1)
        prediction = PredictionRecord(
            prediction_id=f"cohort-{label}",
            created_at_utc=created_at,
            data_cutoff_utc=data_cutoff,
            as_of_session=as_of_session,
            return_start_session=as_of_session,
            target_session=target_session,
            session_calendar=session_calendar,
            symbol=("BTC-USD" if label == "crypto" else f"TEST-{label}"),
            horizon_sessions=1,
            model_name="RL Policy",
            model_version=model_version,
            feature_set_version=feature_set_version,
            policy_version="rl-shadow-cohort-v1",
            forecast_return=0.01,
            target_weight=0.0,
            benchmark_symbol=benchmark_symbol,
            probability_positive=0.70,
            metadata={
                "forecast_context_horizon_sessions": 1,
                "postprocessor_version": postprocessor_version,
                "strict_close_t_eligible": strict_close_t_eligible,
            },
        )
        outcome = OutcomeRecord(
            outcome_id=f"outcome-cohort-{label}",
            prediction_id=prediction.prediction_id,
            recorded_at_utc=prediction.target_maturity_utc
            + timedelta(hours=1),
            target_session=target_session,
            target_maturity_utc=prediction.target_maturity_utc,
            realized_return=0.005,
            benchmark_return=0.0,
            data_source="deterministic-test",
        )
        ledger.append_prediction(prediction)
        ledger.append_outcome(outcome)


if __name__ == "__main__":
    unittest.main()
