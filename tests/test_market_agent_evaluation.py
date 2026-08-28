from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
import json
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from market_agent.agent.evaluation import (
    DataLeakageError,
    FittedPolicy,
    FoldPerformance,
    ForecastObservation,
    MixedForecastCohortError,
    MixedHorizonError,
    PromotionEvidence,
    PromotionGateConfig,
    WalkForwardConfig,
    build_purged_walk_forward_folds,
    evaluate_forecasts,
    evaluate_policy_returns,
    evaluate_promotion_gates,
    policy_performance_from_periods,
    promotion_evaluation_id,
    run_frozen_walk_forward,
)


def _business_dates(count: int) -> list[date]:
    return [
        timestamp.date()
        for timestamp in pd.bdate_range("2026-01-02", periods=count)
    ]


class PurgedWalkForwardTests(unittest.TestCase):
    def test_folds_have_full_horizon_purge_and_embargo(self) -> None:
        sessions = _business_dates(24)
        config = WalkForwardConfig(
            horizon_sessions=2,
            minimum_training_sessions=6,
            test_sessions=3,
        )

        folds = build_purged_walk_forward_folds(sessions, config)

        self.assertGreaterEqual(len(folds), 2)
        first, second = folds[:2]
        self.assertEqual(len(first.train_sessions), 6)
        self.assertEqual(len(first.purged_sessions), 2)
        self.assertEqual(len(first.test_sessions), 3)
        self.assertEqual(len(first.embargoed_sessions), 2)
        self.assertLess(
            max(first.train_sessions),
            min(first.purged_sessions),
        )
        self.assertLess(
            max(first.purged_sessions),
            min(first.test_sessions),
        )
        self.assertEqual(
            second.test_start_session,
            sessions[
                sessions.index(first.test_end_session)
                + 1
                + config.embargo_sessions
            ],
        )

    def test_shorter_than_horizon_purge_or_embargo_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "purge_sessions"):
            WalkForwardConfig(
                horizon_sessions=30,
                minimum_training_sessions=100,
                test_sessions=10,
                purge_sessions=29,
            )
        with self.assertRaisesRegex(ValueError, "embargo_sessions"):
            WalkForwardConfig(
                horizon_sessions=30,
                minimum_training_sessions=100,
                test_sessions=10,
                embargo_sessions=29,
            )

    def test_policy_is_fit_once_and_frozen_for_each_complete_fold(self) -> None:
        sessions = _business_dates(22)
        horizon = 2
        rows = []
        for index in range(len(sessions) - horizon):
            rows.append(
                {
                    "as_of_session": sessions[index],
                    "target_session": sessions[index + horizon],
                    "horizon_sessions": horizon,
                    "symbol": "TEST",
                    "feature": float(index),
                    "realized_return": 0.01 * ((index % 3) - 1),
                    "benchmark_return": 0.0,
                }
            )
        frame = pd.DataFrame(rows)
        config = WalkForwardConfig(
            horizon_sessions=horizon,
            minimum_training_sessions=5,
            test_sessions=2,
        )
        fit_calls: list[int] = []
        prediction_calls: list[int] = []

        def fit_policy(train: pd.DataFrame, fold):
            fit_calls.append(fold.fold_id)
            self.assertLess(
                train["target_session"].max(),
                fold.test_start_session,
            )
            return FittedPolicy(
                model={"mean": float(train["realized_return"].mean())},
                version=f"frozen-{fold.fold_id}",
            )

        def predict_policy(model, features: pd.DataFrame, fold):
            prediction_calls.append(fold.fold_id)
            self.assertEqual(
                list(features.columns),
                ["as_of_session", "symbol", "feature"],
            )
            return features[["as_of_session", "symbol"]].assign(
                predicted_return=[model["mean"]] * len(features)
            )

        run = run_frozen_walk_forward(
            frame,
            config,
            feature_columns=["feature"],
            fit_policy=fit_policy,
            predict_policy=predict_policy,
            evaluation_as_of_session=sessions[17],
        )

        self.assertEqual(fit_calls, [fold.fold_id for fold in run.folds])
        self.assertEqual(prediction_calls, fit_calls)
        self.assertEqual(run.skipped_immature_fold_ids, (2,))
        self.assertEqual(
            sorted(run.predictions["policy_version"].unique()),
            [f"frozen-{fold.fold_id}" for fold in run.folds],
        )
        self.assertTrue(
            (
                run.predictions["target_session"]
                <= sessions[17]
            ).all()
        )

    def test_predictor_is_keyed_and_cannot_mutate_fitted_state(self) -> None:
        sessions = _business_dates(16)
        rows = [
            {
                "as_of_session": sessions[index],
                "target_session": sessions[index + 1],
                "horizon_sessions": 1,
                "symbol": "TEST",
                "feature": float(index),
                "realized_return": 0.0,
                "benchmark_return": 0.0,
            }
            for index in range(len(sessions) - 1)
        ]
        frame = pd.DataFrame(rows)
        config = WalkForwardConfig(1, 5, 2)

        def keyed_predict(model, features: pd.DataFrame, fold):
            return (
                features[["as_of_session", "symbol", "feature"]]
                .iloc[::-1]
                .rename(columns={"feature": "predicted_return"})
            )

        keyed = run_frozen_walk_forward(
            frame,
            config,
            feature_columns=["feature"],
            fit_policy=lambda *_: FittedPolicy({"fixed": True}, "v1"),
            predict_policy=keyed_predict,
            evaluation_as_of_session=sessions[-1],
        )
        expected = {
            row["as_of_session"]: row["feature"]
            for row in rows
        }
        for _, row in keyed.predictions.iterrows():
            self.assertEqual(
                row["predicted_return"],
                expected[row["as_of_session"]],
            )

        def mutating_predict(model, features: pd.DataFrame, fold):
            model["updates"] = model.get("updates", 0) + 1
            return features[["as_of_session", "symbol"]].assign(
                predicted_return=0.0
            )

        with self.assertRaisesRegex(DataLeakageError, "mutated"):
            run_frozen_walk_forward(
                frame,
                config,
                feature_columns=["feature"],
                fit_policy=lambda *_: FittedPolicy({}, "mutable-v1"),
                predict_policy=mutating_predict,
                evaluation_as_of_session=sessions[-1],
            )

    def test_mixed_horizons_and_outcome_features_are_rejected(self) -> None:
        sessions = _business_dates(12)
        frame = pd.DataFrame(
            {
                "as_of_session": sessions[:8],
                "target_session": sessions[2:10],
                "horizon_sessions": [2] * 7 + [1],
                "symbol": [f"S{i}" for i in range(8)],
                "feature": np.arange(8),
                "realized_return": np.zeros(8),
                "benchmark_return": np.zeros(8),
            }
        )
        config = WalkForwardConfig(2, 3, 1)

        with self.assertRaises(MixedHorizonError):
            run_frozen_walk_forward(
                frame,
                config,
                feature_columns=["feature"],
                fit_policy=lambda *_: FittedPolicy({}, "v1"),
                predict_policy=lambda *_: pd.DataFrame(),
                evaluation_as_of_session=sessions[-1],
            )

        frame["horizon_sessions"] = 2
        with self.assertRaises(DataLeakageError):
            run_frozen_walk_forward(
                frame,
                config,
                feature_columns=["realized_return"],
                fit_policy=lambda *_: FittedPolicy({}, "v1"),
                predict_policy=lambda *_: pd.DataFrame(),
                evaluation_as_of_session=sessions[-1],
            )


class EvaluationMetricTests(unittest.TestCase):
    def test_forecast_metrics_require_one_horizon_and_measure_calibration(self) -> None:
        observations = []
        start = date(2026, 1, 2)
        for index in range(20):
            positive = index % 2 == 0
            observations.append(
                ForecastObservation(
                    prediction_id=f"p-{index}",
                    as_of_session=start + timedelta(days=index),
                    target_session=start + timedelta(days=index + 2),
                    symbol=f"S{index}",
                    horizon_sessions=2,
                    predicted_return=0.02 if positive else -0.02,
                    realized_return=0.01 if positive else -0.01,
                    benchmark_return=0.0,
                    probability_positive=0.95 if positive else 0.05,
                    lower_bound_return=-0.03,
                    upper_bound_return=0.03,
                )
            )

        metrics = evaluate_forecasts(observations, horizon_sessions=2)

        self.assertEqual(metrics.sample_count, 20)
        self.assertEqual(metrics.probability_count, 20)
        self.assertAlmostEqual(metrics.direction_accuracy, 1.0)
        self.assertAlmostEqual(metrics.brier_score, 0.0025)
        self.assertAlmostEqual(metrics.expected_calibration_error, 0.05)
        self.assertAlmostEqual(metrics.interval_coverage, 1.0)

        observations[-1] = ForecastObservation(
            **{
                **observations[-1].__dict__,
                "horizon_sessions": 30,
            }
        )
        with self.assertRaises(MixedHorizonError):
            evaluate_forecasts(observations, horizon_sessions=2)

    def test_forecast_direction_treats_zero_as_not_up(self) -> None:
        start = date(2026, 1, 2)
        observations = [
            ForecastObservation(
                prediction_id="negative-forecast-flat-outcome",
                as_of_session=start,
                target_session=start + timedelta(days=1),
                symbol="FIRST",
                horizon_sessions=1,
                predicted_return=-0.01,
                realized_return=0.0,
                benchmark_return=0.0,
                probability_positive=0.10,
            ),
            ForecastObservation(
                prediction_id="flat-forecast-negative-outcome",
                as_of_session=start + timedelta(days=1),
                target_session=start + timedelta(days=2),
                symbol="SECOND",
                horizon_sessions=1,
                predicted_return=0.0,
                realized_return=-0.01,
                benchmark_return=0.0,
                probability_positive=0.10,
            ),
        ]

        metrics = evaluate_forecasts(observations, horizon_sessions=1)

        self.assertEqual(metrics.direction_accuracy, 1.0)

    def test_forecast_metrics_reject_mixed_provenance_cohorts(self) -> None:
        base = ForecastObservation(
            prediction_id="base",
            as_of_session=date(2026, 1, 2),
            target_session=date(2026, 1, 5),
            symbol="TEST",
            horizon_sessions=1,
            predicted_return=0.01,
            realized_return=0.005,
            benchmark_return=0.0,
            model_version="model-v1",
            feature_set_version="features-v1",
            postprocessor_version="post-v1",
        )
        mixed_values = {
            "session_calendar": "utc_daily_24_7",
            "benchmark_symbol": "QQQ",
            "model_version": "model-v2",
            "feature_set_version": "features-v2",
            "postprocessor_version": "post-v2",
            "strict_close_t_eligible": False,
        }

        for field_name, value in mixed_values.items():
            with self.subTest(field_name=field_name):
                changed = replace(
                    base,
                    prediction_id=f"mixed-{field_name}",
                    **{field_name: value},
                )
                with self.assertRaisesRegex(
                    MixedForecastCohortError,
                    field_name,
                ):
                    evaluate_forecasts(
                        (base, changed),
                        horizon_sessions=1,
                    )

    def test_forecast_observation_rejects_unknown_calendar(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be one of"):
            ForecastObservation(
                prediction_id="bad-calendar",
                as_of_session=date(2026, 1, 2),
                target_session=date(2026, 1, 5),
                symbol="TEST",
                horizon_sessions=1,
                predicted_return=0.01,
                realized_return=0.005,
                benchmark_return=0.0,
                session_calendar="weekday_guess",
            )

    def test_transaction_cost_applies_only_when_exposure_changes(self) -> None:
        sessions = _business_dates(3)
        frame = pd.DataFrame(
            {
                "session": sessions,
                "symbol": ["TEST"] * 3,
                "target_weight": [0.5, 0.5, 0.0],
                "asset_return": [0.01, 0.01, 0.0],
                "benchmark_return": [0.0, 0.0, 0.0],
            }
        )

        performance = evaluate_policy_returns(
            frame,
            transaction_cost_bps=10.0,
        )

        self.assertAlmostEqual(performance.periods[0].transaction_cost, 0.0005)
        self.assertAlmostEqual(performance.periods[1].transaction_cost, 0.0)
        self.assertAlmostEqual(performance.periods[2].transaction_cost, 0.0005)
        self.assertAlmostEqual(performance.total_transaction_cost, 0.001)
        self.assertAlmostEqual(performance.total_turnover, 1.0)

    def test_portfolio_gross_exposure_is_enforced(self) -> None:
        session = _business_dates(1)[0]
        frame = pd.DataFrame(
            {
                "session": [session, session],
                "symbol": ["A", "B"],
                "target_weight": [0.6, 0.5],
                "asset_return": [0.0, 0.0],
                "benchmark_return": [0.0, 0.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "Gross exposure"):
            evaluate_policy_returns(frame, transaction_cost_bps=5.0)

    def test_promotion_requires_every_gate(self) -> None:
        sessions = _business_dates(80)

        def performance(
            pattern: list[float],
            cost_bps: float,
            selected_sessions: list[date] | None = None,
        ):
            evaluation_sessions = selected_sessions or sessions
            returns = [
                pattern[index % len(pattern)]
                for index in range(len(evaluation_sessions))
            ]
            frame = pd.DataFrame(
                {
                    "session": evaluation_sessions,
                    "symbol": ["TEST"] * len(evaluation_sessions),
                    "target_weight": [1.0] * len(evaluation_sessions),
                    "asset_return": returns,
                    "benchmark_return": [0.0] * len(evaluation_sessions),
                }
            )
            return evaluate_policy_returns(
                frame,
                transaction_cost_bps=cost_bps,
            )

        candidate_pattern = [0.0020, 0.0020, 0.0010, -0.0002]
        baseline_pattern = [0.0020, -0.0020, 0.0010, -0.0015]
        candidate = performance(candidate_pattern, 5.0)
        baseline = performance(baseline_pattern, 5.0)
        candidate_stress = performance(candidate_pattern, 10.0)
        baseline_stress = performance(baseline_pattern, 10.0)
        folds = []
        for index in range(10):
            fold_sessions = sessions[index * 8 : (index + 1) * 8]
            session_set = set(fold_sessions)
            folds.append(
                FoldPerformance(
                    fold_id=index,
                    candidate=policy_performance_from_periods(
                        [
                            period
                            for period in candidate.periods
                            if period.session in session_set
                        ],
                        transaction_cost_bps=5.0,
                    ),
                    baseline=policy_performance_from_periods(
                        [
                            period
                            for period in baseline.periods
                            if period.session in session_set
                        ],
                        transaction_cost_bps=5.0,
                    ),
                    horizon_sessions=2,
                    candidate_policy_version="rl-shadow-v2",
                    baseline_policy_version="ridge-v1",
                )
            )
        forecast_metrics = evaluate_forecasts(
            [
                ForecastObservation(
                    prediction_id=f"p-{index}",
                    as_of_session=sessions[index],
                    target_session=sessions[index + 2],
                    symbol="TEST",
                    horizon_sessions=2,
                    predicted_return=0.01 if index % 2 else -0.01,
                    realized_return=0.01 if index % 2 else -0.01,
                    benchmark_return=0.0,
                    probability_positive=0.95 if index % 2 else 0.05,
                )
                for index in range(60)
            ],
            horizon_sessions=2,
        )
        fixed_ensemble = performance(
            [0.0015, -0.0020, 0.0005, -0.0015],
            5.0,
        )
        baseline_candidates = {
            "ridge": baseline,
            "fixed-ensemble": fixed_ensemble,
        }
        baseline_candidate_versions = {
            "ridge": "ridge-v1",
            "fixed-ensemble": "fixed-ensemble-v1",
        }
        prediction_ids = tuple(f"p-{index}" for index in range(60))
        ledger_head_hash = "a" * 64
        with mock.patch(
            "market_agent.agent.evaluation.json.dumps",
            wraps=json.dumps,
        ) as dumps:
            evaluation_id = promotion_evaluation_id(
                horizon_sessions=2,
                candidate_policy_version="rl-shadow-v2",
                baseline_policy_version="ridge-v1",
                baseline_name="ridge",
                candidate=candidate,
                baseline=baseline,
                candidate_doubled_cost=candidate_stress,
                baseline_doubled_cost=baseline_stress,
                folds=tuple(folds),
                baseline_candidates=baseline_candidates,
                baseline_candidate_versions=baseline_candidate_versions,
                candidate_forecast_metrics=forecast_metrics,
                candidate_forecast_prediction_ids=prediction_ids,
                candidate_model_name="RL Policy",
                forecast_as_of_session=sessions[-1],
                ledger_head_hash=ledger_head_hash,
            )
        legacy_payload = dumps.call_args.args[0]
        for new_field in (
            "candidate_session_calendar",
            "candidate_benchmark_symbol",
            "candidate_model_version",
            "candidate_feature_set_version",
            "candidate_postprocessor_version",
            "candidate_strict_close_t_eligible",
        ):
            self.assertNotIn(new_field, legacy_payload)
        evidence = PromotionEvidence(
            shadow_sessions=80,
            candidate=candidate,
            baseline=baseline,
            candidate_doubled_cost=candidate_stress,
            baseline_doubled_cost=baseline_stress,
            folds=tuple(folds),
            candidate_forecast_metrics=forecast_metrics,
            horizon_sessions=2,
            evaluation_id=evaluation_id,
            candidate_policy_version="rl-shadow-v2",
            baseline_policy_version="ridge-v1",
            baseline_name="ridge",
            baseline_candidates=baseline_candidates,
            baseline_candidate_versions=baseline_candidate_versions,
            candidate_forecast_prediction_ids=prediction_ids,
            candidate_model_name="RL Policy",
            forecast_as_of_session=sessions[-1],
            ledger_head_hash=ledger_head_hash,
        )

        decision = evaluate_promotion_gates(
            evidence,
            PromotionGateConfig(
                minimum_shadow_sessions=60,
                minimum_sharpe_improvement=0.25,
                minimum_positive_fold_fraction=0.70,
                minimum_probability_samples=60,
            ),
        )

        self.assertFalse(decision.promoted)
        self.assertIn(
            "ledger_backed_forecast_provenance",
            [check.name for check in decision.failed_checks],
        )

        failed = evaluate_promotion_gates(
            evidence,
            PromotionGateConfig(minimum_shadow_sessions=81),
        )
        self.assertFalse(failed.promoted)
        self.assertIn(
            "shadow_sessions",
            [check.name for check in failed.failed_checks],
        )

        with self.assertRaisesRegex(
            ValueError,
            "immutable period path",
        ):
            replace(
                candidate_stress,
                cumulative_net_return=999.0,
                sharpe=999.0,
            )


if __name__ == "__main__":
    unittest.main()
