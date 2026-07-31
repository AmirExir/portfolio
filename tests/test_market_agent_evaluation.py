from __future__ import annotations

from datetime import date, timedelta
import unittest

import numpy as np
import pandas as pd

from market_agent.agent.evaluation import (
    DataLeakageError,
    FittedPolicy,
    FoldPerformance,
    ForecastObservation,
    MixedHorizonError,
    PromotionEvidence,
    PromotionGateConfig,
    WalkForwardConfig,
    build_purged_walk_forward_folds,
    evaluate_forecasts,
    evaluate_policy_returns,
    evaluate_promotion_gates,
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
            return pd.DataFrame(
                {"predicted_return": [model["mean"]] * len(features)}
            )

        run = run_frozen_walk_forward(
            frame,
            config,
            feature_columns=["feature"],
            fit_policy=fit_policy,
            predict_policy=predict_policy,
            evaluation_as_of_session=sessions[-1],
        )

        self.assertEqual(fit_calls, [fold.fold_id for fold in run.folds])
        self.assertEqual(prediction_calls, fit_calls)
        self.assertEqual(
            sorted(run.predictions["policy_version"].unique()),
            [f"frozen-{fold.fold_id}" for fold in run.folds],
        )
        self.assertTrue(
            (
                run.predictions["target_session"]
                <= sessions[-1]
            ).all()
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

        def performance(pattern: list[float], cost_bps: float):
            returns = [
                pattern[index % len(pattern)]
                for index in range(len(sessions))
            ]
            frame = pd.DataFrame(
                {
                    "session": sessions,
                    "symbol": ["TEST"] * len(sessions),
                    "target_weight": [1.0] * len(sessions),
                    "asset_return": returns,
                    "benchmark_return": [0.0] * len(sessions),
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
        folds = tuple(
            FoldPerformance(
                fold_id=index,
                candidate=candidate,
                baseline=baseline,
            )
            for index in range(10)
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
        evidence = PromotionEvidence(
            shadow_sessions=80,
            candidate=candidate,
            baseline=baseline,
            candidate_doubled_cost=candidate_stress,
            baseline_doubled_cost=baseline_stress,
            folds=folds,
            candidate_forecast_metrics=forecast_metrics,
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

        self.assertTrue(
            decision.promoted,
            [check.name for check in decision.failed_checks],
        )

        insufficient_shadow = PromotionEvidence(
            **{
                **evidence.__dict__,
                "shadow_sessions": 59,
            }
        )
        failed = evaluate_promotion_gates(insufficient_shadow)
        self.assertFalse(failed.promoted)
        self.assertIn(
            "shadow_sessions",
            [check.name for check in failed.failed_checks],
        )


if __name__ == "__main__":
    unittest.main()
