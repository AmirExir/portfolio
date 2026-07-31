from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from market_agent.agent.earnings import us_equity_trading_sessions
from market_agent.agent.ledger import (
    OutcomeRecord,
    PredictionLedger,
    PredictionRecord,
)
from market_agent.daily_ml_forecast_report import prediction_ledger_summary


UTC = timezone.utc


class PredictionLedgerSummaryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
