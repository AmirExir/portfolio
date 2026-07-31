from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from market_agent.agent.evaluation import (
    FoldPerformance,
    PromotionEvidence,
    PromotionGateConfig,
    build_ledger_backed_promotion_evidence,
    evaluate_policy_returns,
    evaluate_promotion_gates,
    forecast_observations_from_ledger,
    policy_performance_from_periods,
    promotion_evaluation_id,
)
from market_agent.agent.earnings import us_equity_trading_sessions
from market_agent.agent.ledger import (
    OutcomeRecord,
    PredictionLedger,
    PredictionRecord,
)


UTC = timezone.utc


class LedgerBackedPromotionTests(unittest.TestCase):
    candidate_model_name = "RL Policy"
    candidate_policy_version = "rl-shadow-ledger-v1"
    baseline_policy_version = "ridge-v1"

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.ledger = PredictionLedger(
            Path(cls.temporary_directory.name) / "prediction_ledger.jsonl"
        )
        calendar = us_equity_trading_sessions(
            as_of=datetime(2025, 1, 2, 17, tzinfo=UTC),
            future_session_count=100,
        )
        cls.sessions = [
            session
            for session in calendar
            if session >= date(2025, 1, 2)
        ][:8]
        cls.candidate_prediction_ids = tuple(
            f"candidate-{index:03d}" for index in range(len(cls.sessions))
        )

        for index, as_of_session in enumerate(cls.sessions):
            forecast_return = 0.01 if index % 2 else -0.01
            cls._append_matured_prediction(
                prediction_id=cls.candidate_prediction_ids[index],
                as_of_session=as_of_session,
                horizon_sessions=1,
                model_name=cls.candidate_model_name,
                policy_version=cls.candidate_policy_version,
                forecast_return=forecast_return,
                probability_positive=0.95 if forecast_return > 0 else 0.05,
            )

        cls._append_matured_prediction(
            prediction_id="noise-30-day",
            as_of_session=cls.sessions[0],
            horizon_sessions=30,
            model_name=cls.candidate_model_name,
            policy_version=cls.candidate_policy_version,
            forecast_return=0.02,
            probability_positive=0.70,
        )
        cls._append_matured_prediction(
            prediction_id="noise-other-policy",
            as_of_session=cls.sessions[0],
            horizon_sessions=1,
            model_name=cls.candidate_model_name,
            policy_version="rl-shadow-other-v1",
            forecast_return=0.02,
            probability_positive=0.70,
        )
        cls._append_matured_prediction(
            prediction_id="noise-other-model",
            as_of_session=cls.sessions[0],
            horizon_sessions=1,
            model_name="Ridge",
            policy_version=cls.candidate_policy_version,
            forecast_return=0.02,
            probability_positive=0.70,
        )

        cls.candidate = cls._performance(
            [0.0020, 0.0020, 0.0010, -0.0002],
            cost_bps=5.0,
        )
        cls.baseline = cls._performance(
            [0.0020, -0.0020, 0.0010, -0.0015],
            cost_bps=5.0,
        )
        cls.candidate_stress = cls._performance(
            [0.0020, 0.0020, 0.0010, -0.0002],
            cost_bps=10.0,
        )
        cls.baseline_stress = cls._performance(
            [0.0020, -0.0020, 0.0010, -0.0015],
            cost_bps=10.0,
        )
        cls.fixed_ensemble = cls._performance(
            [0.0015, -0.0020, 0.0005, -0.0015],
            cost_bps=5.0,
        )
        cls.baseline_candidates = {
            "ridge": cls.baseline,
            "fixed-ensemble": cls.fixed_ensemble,
        }
        cls.baseline_candidate_versions = {
            "ridge": cls.baseline_policy_version,
            "fixed-ensemble": "fixed-ensemble-v1",
        }
        cls.folds = tuple(
            cls._fold(index)
            for index in range(2)
        )
        cls.gate_config = PromotionGateConfig(
            minimum_shadow_sessions=8,
            minimum_probability_samples=8,
        )
        cls.forecast_cutoff = cls._target_session(
            cls.sessions[-1],
            1,
        )
        cls.evidence = build_ledger_backed_promotion_evidence(
            cls.ledger,
            forecast_as_of_session=cls.forecast_cutoff,
            horizon_sessions=1,
            candidate_model_name=cls.candidate_model_name,
            candidate_policy_version=cls.candidate_policy_version,
            candidate=cls.candidate,
            baseline=cls.baseline,
            candidate_doubled_cost=cls.candidate_stress,
            baseline_doubled_cost=cls.baseline_stress,
            folds=cls.folds,
            baseline_policy_version=cls.baseline_policy_version,
            baseline_name="ridge",
            baseline_candidates=cls.baseline_candidates,
            baseline_candidate_versions=cls.baseline_candidate_versions,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_directory.cleanup()

    @classmethod
    def _append_matured_prediction(
        cls,
        *,
        prediction_id: str,
        as_of_session: date,
        horizon_sessions: int,
        model_name: str,
        policy_version: str,
        forecast_return: float,
        probability_positive: float,
    ) -> None:
        target_session = cls._target_session(
            as_of_session,
            horizon_sessions,
        )
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
            target_session=target_session,
            symbol="TEST",
            horizon_sessions=horizon_sessions,
            model_name=model_name,
            model_version=f"{model_name}-model-v1",
            policy_version=policy_version,
            forecast_return=forecast_return,
            target_weight=0.0,
            probability_positive=probability_positive,
            feature_set_version="ledger-promotion-test-v1",
            feature_hash=f"feature-{prediction_id}",
            metadata={"shadow_mode": True, "live_eligible": False},
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
        cls.ledger.append_prediction(prediction)
        cls.ledger.append_outcome(outcome)

    @staticmethod
    def _target_session(
        as_of_session: date,
        horizon_sessions: int,
    ) -> date:
        calendar = us_equity_trading_sessions(
            as_of=datetime.combine(
                as_of_session,
                time(hour=17),
                tzinfo=UTC,
            ),
            future_session_count=horizon_sessions + 1,
        )
        future = [
            session for session in calendar if session > as_of_session
        ]
        return future[horizon_sessions - 1]

    @classmethod
    def _performance(cls, pattern: list[float], *, cost_bps: float):
        returns = [
            pattern[index % len(pattern)]
            for index in range(len(cls.sessions))
        ]
        return evaluate_policy_returns(
            pd.DataFrame(
                {
                    "session": cls.sessions,
                    "symbol": ["TEST"] * len(cls.sessions),
                    "target_weight": [1.0] * len(cls.sessions),
                    "asset_return": returns,
                    "benchmark_return": [0.0] * len(cls.sessions),
                }
            ),
            transaction_cost_bps=cost_bps,
        )

    @classmethod
    def _fold(cls, fold_id: int) -> FoldPerformance:
        start = fold_id * 4
        stop = start + 4
        return FoldPerformance(
            fold_id=fold_id,
            candidate=policy_performance_from_periods(
                cls.candidate.periods[start:stop],
                transaction_cost_bps=5.0,
            ),
            baseline=policy_performance_from_periods(
                cls.baseline.periods[start:stop],
                transaction_cost_bps=5.0,
            ),
            horizon_sessions=1,
            candidate_policy_version=cls.candidate_policy_version,
            baseline_policy_version=cls.baseline_policy_version,
        )

    def test_filter_is_exact_for_model_policy_and_horizon(self) -> None:
        one_day = forecast_observations_from_ledger(
            self.ledger,
            as_of_session=self.forecast_cutoff,
            horizon_sessions=1,
            candidate_model_name=self.candidate_model_name,
            candidate_policy_version=self.candidate_policy_version,
        )
        thirty_day = forecast_observations_from_ledger(
            self.ledger,
            as_of_session=self.forecast_cutoff + timedelta(days=60),
            horizon_sessions=30,
            candidate_model_name=self.candidate_model_name,
            candidate_policy_version=self.candidate_policy_version,
        )

        self.assertEqual(
            tuple(item.prediction_id for item in one_day),
            self.candidate_prediction_ids,
        )
        self.assertEqual(
            tuple(item.prediction_id for item in thirty_day),
            ("noise-30-day",),
        )

    def test_builder_derives_verified_ledger_provenance(self) -> None:
        self.assertEqual(
            self.evidence.candidate_forecast_prediction_ids,
            self.candidate_prediction_ids,
        )
        self.assertEqual(
            self.evidence.candidate_forecast_metrics.sample_count,
            8,
        )
        self.assertEqual(
            self.evidence.ledger_head_hash,
            self.ledger.read_entries()[-1].record_hash,
        )
        decision = evaluate_promotion_gates(
            self.evidence,
            self.gate_config,
            ledger=self.ledger,
        )
        self.assertTrue(
            decision.promoted,
            [check.name for check in decision.failed_checks],
        )

    def test_missing_ledger_fails_closed(self) -> None:
        decision = evaluate_promotion_gates(
            self.evidence,
            self.gate_config,
        )

        self.assertFalse(decision.promoted)
        self.assertIn(
            "ledger_backed_forecast_provenance",
            [check.name for check in decision.failed_checks],
        )

    def test_ids_hash_and_metrics_cannot_self_certify(self) -> None:
        forged_values = (
            {
                "candidate_forecast_prediction_ids": tuple(
                    f"forged-{index:03d}"
                    for index in range(len(self.sessions))
                )
            },
            {"ledger_head_hash": "f" * 64},
            {
                "candidate_forecast_metrics": replace(
                    self.evidence.candidate_forecast_metrics,
                    mae=self.evidence.candidate_forecast_metrics.mae + 1.0,
                )
            },
        )

        for changes in forged_values:
            with self.subTest(changes=tuple(changes)):
                forged = self._reidentify(self.evidence, **changes)
                decision = evaluate_promotion_gates(
                    forged,
                    self.gate_config,
                    ledger=self.ledger,
                )
                checks = {check.name: check for check in decision.checks}
                self.assertTrue(
                    checks["matched_evaluation_provenance"].passed
                )
                self.assertFalse(
                    checks["ledger_backed_forecast_provenance"].passed
                )
                self.assertFalse(decision.promoted)

    def test_thirty_day_slice_cannot_certify_one_day_policy_sessions(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "sessions must exactly match"):
            build_ledger_backed_promotion_evidence(
                self.ledger,
                forecast_as_of_session=self.forecast_cutoff
                + timedelta(days=60),
                horizon_sessions=30,
                candidate_model_name=self.candidate_model_name,
                candidate_policy_version=self.candidate_policy_version,
                candidate=self.candidate,
                baseline=self.baseline,
                candidate_doubled_cost=self.candidate_stress,
                baseline_doubled_cost=self.baseline_stress,
                folds=self.folds,
                baseline_policy_version=self.baseline_policy_version,
                baseline_name="ridge",
                baseline_candidates=self.baseline_candidates,
                baseline_candidate_versions=(
                    self.baseline_candidate_versions
                ),
            )

    @staticmethod
    def _reidentify(
        evidence: PromotionEvidence,
        **changes,
    ) -> PromotionEvidence:
        values = {
            field_name: changes.get(field_name, getattr(evidence, field_name))
            for field_name in PromotionEvidence.__dataclass_fields__
            if field_name != "evaluation_id"
        }
        evaluation_id = promotion_evaluation_id(
            horizon_sessions=values["horizon_sessions"],
            candidate_policy_version=values["candidate_policy_version"],
            baseline_policy_version=values["baseline_policy_version"],
            baseline_name=values["baseline_name"],
            candidate=values["candidate"],
            baseline=values["baseline"],
            candidate_doubled_cost=values["candidate_doubled_cost"],
            baseline_doubled_cost=values["baseline_doubled_cost"],
            folds=values["folds"],
            baseline_candidates=values["baseline_candidates"],
            baseline_candidate_versions=(
                values["baseline_candidate_versions"]
            ),
            candidate_forecast_metrics=values[
                "candidate_forecast_metrics"
            ],
            candidate_forecast_prediction_ids=values[
                "candidate_forecast_prediction_ids"
            ],
            candidate_model_name=values["candidate_model_name"],
            forecast_as_of_session=values["forecast_as_of_session"],
            ledger_head_hash=values["ledger_head_hash"],
        )
        return PromotionEvidence(
            evaluation_id=evaluation_id,
            **values,
        )


if __name__ == "__main__":
    unittest.main()
