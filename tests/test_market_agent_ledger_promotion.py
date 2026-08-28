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
    UTC_DAILY_24_7_SESSION_CALENDAR,
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
            metadata={
                "shadow_mode": True,
                "live_eligible": False,
                "postprocessor_version": "ledger-promotion-v1",
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

    def test_builder_persists_and_replays_exact_candidate_filters(self) -> None:
        evidence = build_ledger_backed_promotion_evidence(
            self.ledger,
            forecast_as_of_session=self.forecast_cutoff,
            horizon_sessions=1,
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
            baseline_candidate_versions=self.baseline_candidate_versions,
            candidate_session_calendar="us_equity",
            candidate_benchmark_symbol="SPY",
            candidate_model_version="RL Policy-model-v1",
            candidate_feature_set_version="ledger-promotion-test-v1",
            candidate_postprocessor_version="ledger-promotion-v1",
            candidate_strict_close_t_eligible=True,
        )

        self.assertEqual(evidence.candidate_session_calendar, "us_equity")
        self.assertEqual(evidence.candidate_benchmark_symbol, "SPY")
        self.assertEqual(
            evidence.candidate_model_version,
            "RL Policy-model-v1",
        )
        self.assertEqual(
            evidence.candidate_feature_set_version,
            "ledger-promotion-test-v1",
        )
        self.assertEqual(
            evidence.candidate_postprocessor_version,
            "ledger-promotion-v1",
        )
        decision = evaluate_promotion_gates(
            evidence,
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
            candidate_session_calendar=values[
                "candidate_session_calendar"
            ],
            candidate_benchmark_symbol=values[
                "candidate_benchmark_symbol"
            ],
            candidate_model_version=values["candidate_model_version"],
            candidate_feature_set_version=values[
                "candidate_feature_set_version"
            ],
            candidate_postprocessor_version=values[
                "candidate_postprocessor_version"
            ],
            candidate_strict_close_t_eligible=values[
                "candidate_strict_close_t_eligible"
            ],
        )
        return PromotionEvidence(
            evaluation_id=evaluation_id,
            **values,
        )


class LedgerPromotionCohortTests(unittest.TestCase):
    def test_unfiltered_mixed_provenance_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = PredictionLedger(Path(directory) / "ledger.jsonl")
            self._append_completed(
                ledger,
                prediction_id="base",
                symbol="BASE",
                session_calendar="us_equity",
                benchmark_symbol="SPY",
                model_version="model-v1",
                feature_set_version="features-v1",
                postprocessor_version="post-v1",
                strict=True,
            )
            self._append_completed(
                ledger,
                prediction_id="benchmark",
                symbol="BENCH",
                session_calendar="us_equity",
                benchmark_symbol="QQQ",
                model_version="model-v1",
                feature_set_version="features-v1",
                postprocessor_version="post-v1",
                strict=True,
            )
            self._append_completed(
                ledger,
                prediction_id="model",
                symbol="MODEL",
                session_calendar="us_equity",
                benchmark_symbol="SPY",
                model_version="model-v2",
                feature_set_version="features-v1",
                postprocessor_version="post-v1",
                strict=True,
            )
            self._append_completed(
                ledger,
                prediction_id="postprocessor",
                symbol="POST",
                session_calendar="us_equity",
                benchmark_symbol="SPY",
                model_version="model-v1",
                feature_set_version="features-v1",
                postprocessor_version="post-v2",
                strict=True,
            )
            self._append_completed(
                ledger,
                prediction_id="features",
                symbol="FEATURES",
                session_calendar="us_equity",
                benchmark_symbol="SPY",
                model_version="model-v1",
                feature_set_version="features-v2",
                postprocessor_version="post-v1",
                strict=True,
            )
            self._append_completed(
                ledger,
                prediction_id="crypto",
                symbol="BTC-USD",
                session_calendar=UTC_DAILY_24_7_SESSION_CALENDAR,
                benchmark_symbol="BTC-USD",
                model_version="model-v1",
                feature_set_version="features-v1",
                postprocessor_version="post-v1",
                strict=True,
            )

            with self.assertRaisesRegex(
                ValueError,
                "Mixed session_calendar cohort",
            ):
                self._observations(ledger)
            with self.assertRaisesRegex(
                ValueError,
                "Mixed benchmark_symbol cohort",
            ):
                self._observations(
                    ledger,
                    session_calendar="us_equity",
                )
            with self.assertRaisesRegex(
                ValueError,
                "Mixed model_version cohort",
            ):
                self._observations(
                    ledger,
                    session_calendar="us_equity",
                    benchmark_symbol="SPY",
                )
            with self.assertRaisesRegex(
                ValueError,
                "Mixed feature_set_version cohort",
            ):
                self._observations(
                    ledger,
                    session_calendar="us_equity",
                    benchmark_symbol="SPY",
                    model_version="model-v1",
                )
            with self.assertRaisesRegex(
                ValueError,
                "Mixed postprocessor_version cohort",
            ):
                self._observations(
                    ledger,
                    session_calendar="us_equity",
                    benchmark_symbol="SPY",
                    model_version="model-v1",
                    feature_set_version="features-v1",
                )
            exact = self._observations(
                ledger,
                session_calendar="us_equity",
                benchmark_symbol="SPY",
                model_version="model-v1",
                feature_set_version="features-v1",
                postprocessor_version="post-v1",
            )

        self.assertEqual(
            tuple(item.prediction_id for item in exact),
            ("base",),
        )

    def test_delayed_crypto_is_research_visible_but_excluded_by_default(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = PredictionLedger(Path(directory) / "ledger.jsonl")
            self._append_completed(
                ledger,
                prediction_id="crypto-strict",
                symbol="BTC-USD",
                session_calendar=UTC_DAILY_24_7_SESSION_CALENDAR,
                benchmark_symbol="BTC-USD",
                model_version="crypto-model-v1",
                feature_set_version="crypto-features-v1",
                postprocessor_version="crypto-post-v1",
                strict=True,
                as_of_session=date(2026, 1, 2),
                created_at_utc=datetime(2026, 1, 3, 0, 10, tzinfo=UTC),
            )
            self._append_completed(
                ledger,
                prediction_id="crypto-delayed",
                symbol="ETH-USD",
                session_calendar=UTC_DAILY_24_7_SESSION_CALENDAR,
                benchmark_symbol="BTC-USD",
                model_version="crypto-model-v1",
                feature_set_version="crypto-features-v1",
                postprocessor_version="crypto-post-v1",
                strict=False,
                as_of_session=date(2026, 1, 3),
                created_at_utc=datetime(2026, 1, 4, 9, tzinfo=UTC),
            )
            filters = {
                "session_calendar": UTC_DAILY_24_7_SESSION_CALENDAR,
                "benchmark_symbol": "BTC-USD",
                "model_version": "crypto-model-v1",
                "feature_set_version": "crypto-features-v1",
                "postprocessor_version": "crypto-post-v1",
            }
            strict = self._observations(ledger, **filters)
            delayed = self._observations(
                ledger,
                strict_close_t_eligible=False,
                **filters,
            )
            with self.assertRaisesRegex(
                ValueError,
                "Mixed strict_close_t_eligible cohort",
            ):
                self._observations(
                    ledger,
                    strict_close_t_eligible=None,
                    **filters,
                )

        self.assertEqual(
            tuple(item.prediction_id for item in strict),
            ("crypto-strict",),
        )
        self.assertEqual(
            tuple(item.prediction_id for item in delayed),
            ("crypto-delayed",),
        )
        self.assertFalse(delayed[0].strict_close_t_eligible)

    @staticmethod
    def _observations(
        ledger: PredictionLedger,
        **filters,
    ):
        return forecast_observations_from_ledger(
            ledger,
            as_of_session=date(2026, 1, 10),
            horizon_sessions=1,
            candidate_model_name="Ensemble",
            candidate_policy_version="component-shadow-v1",
            **filters,
        )

    @staticmethod
    def _append_completed(
        ledger: PredictionLedger,
        *,
        prediction_id: str,
        symbol: str,
        session_calendar: str,
        benchmark_symbol: str,
        model_version: str,
        feature_set_version: str,
        postprocessor_version: str,
        strict: bool,
        as_of_session: date = date(2026, 1, 2),
        created_at_utc: datetime | None = None,
    ) -> None:
        is_crypto = session_calendar == UTC_DAILY_24_7_SESSION_CALENDAR
        target_session = (
            as_of_session + timedelta(days=1)
            if is_crypto
            else date(2026, 1, 5)
        )
        data_cutoff = (
            datetime.combine(
                as_of_session + timedelta(days=1),
                time.min,
                tzinfo=UTC,
            )
            if is_crypto
            else datetime(2026, 1, 2, 11, 59, tzinfo=UTC)
        )
        created_at = created_at_utc or (
            data_cutoff + timedelta(minutes=10)
            if is_crypto
            else datetime(2026, 1, 2, 12, tzinfo=UTC)
        )
        prediction = PredictionRecord(
            prediction_id=prediction_id,
            created_at_utc=created_at,
            data_cutoff_utc=data_cutoff,
            as_of_session=as_of_session,
            target_session=target_session,
            session_calendar=session_calendar,
            symbol=symbol,
            horizon_sessions=1,
            model_name="Ensemble",
            model_version=model_version,
            feature_set_version=feature_set_version,
            policy_version="component-shadow-v1",
            forecast_return=0.01,
            target_weight=0.0,
            benchmark_symbol=benchmark_symbol,
            probability_positive=0.70,
            metadata={
                "postprocessor_version": postprocessor_version,
                "strict_close_t_eligible": strict,
            },
        )
        outcome = OutcomeRecord(
            outcome_id=f"outcome-{prediction_id}",
            prediction_id=prediction_id,
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
