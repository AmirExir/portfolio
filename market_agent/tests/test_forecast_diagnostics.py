from __future__ import annotations

from dataclasses import replace
from datetime import date
import unittest

from market_agent.agent.evaluation import (
    ForecastObservation,
    MixedForecastCohortError,
)
from market_agent.agent.forecast_diagnostics import forecast_performance_diagnostics


class ForecastDiagnosticsTests(unittest.TestCase):
    def observation(self, **overrides) -> ForecastObservation:
        values = {
            "prediction_id": "original",
            "as_of_session": date(2026, 1, 1),
            "target_session": date(2026, 1, 3),
            "symbol": "TEST",
            "horizon_sessions": 2,
            "predicted_return": 0.10,
            "realized_return": 0.20,
            "benchmark_return": 0.0,
            "probability_positive": 0.80,
        }
        values.update(overrides)
        return ForecastObservation(**values)

    def test_fixed_baselines_and_coverage_have_hand_calculated_values(self) -> None:
        observations = [
            self.observation(),
            self.observation(
                prediction_id="second",
                as_of_session=date(2026, 1, 3),
                target_session=date(2026, 1, 5),
                predicted_return=-0.10,
                realized_return=0.0,
                probability_positive=0.20,
            ),
        ]

        result = forecast_performance_diagnostics(observations, horizon_sessions=2)

        self.assertAlmostEqual(result["mae_pct"], 10.0)
        self.assertAlmostEqual(result["no_change_mae_pct"], 10.0)
        self.assertAlmostEqual(result["mae_skill_score"], 0.0)
        self.assertAlmostEqual(result["direction_accuracy_pct"], 100.0)
        self.assertAlmostEqual(result["always_up_accuracy_pct"], 50.0)
        self.assertAlmostEqual(result["always_not_up_accuracy_pct"], 50.0)
        self.assertAlmostEqual(result["brier_score"], 0.04)
        self.assertAlmostEqual(result["fair_coin_brier_score"], 0.25)
        self.assertAlmostEqual(result["brier_skill_score_vs_fair_coin"], 0.84)
        self.assertEqual(result["origin_session_count"], 2)
        self.assertEqual(result["symbol_count"], 1)
        self.assertEqual(result["first_origin_session"], "2026-01-01")
        self.assertEqual(result["last_target_session"], "2026-01-05")
        self.assertEqual(result["non_overlapping"]["sample_count"], 2)

    def test_repeated_runs_keep_first_forecast_and_do_not_inflate_evidence(self) -> None:
        first = self.observation()
        revised = replace(first, prediction_id="revision", predicted_return=0.20)

        result = forecast_performance_diagnostics([first, revised], horizon_sessions=2)

        self.assertEqual(result["raw_observation_count"], 2)
        self.assertEqual(result["duplicate_forecast_count"], 1)
        self.assertEqual(result["sample_count"], 1)
        self.assertAlmostEqual(result["mae_pct"], 10.0)

    def test_non_overlap_is_per_asset_and_keeps_adjacent_return_windows(self) -> None:
        first = self.observation()
        overlapping = replace(
            first,
            prediction_id="overlapping",
            as_of_session=date(2026, 1, 2),
            target_session=date(2026, 1, 4),
            predicted_return=-0.20,
        )
        adjacent = replace(
            first,
            prediction_id="adjacent",
            as_of_session=date(2026, 1, 3),
            target_session=date(2026, 1, 5),
        )
        other_asset = replace(first, prediction_id="other", symbol="OTHER")

        result = forecast_performance_diagnostics(
            [overlapping, adjacent, first, other_asset],
            horizon_sessions=2,
        )

        self.assertEqual(result["sample_count"], 4)
        self.assertEqual(result["non_overlapping"]["sample_count"], 3)
        self.assertEqual(result["non_overlapping"]["symbol_count"], 2)
        self.assertAlmostEqual(result["non_overlapping"]["mae_pct"], 10.0)

    def test_zero_baseline_and_absent_probabilities_are_not_given_fake_skill(self) -> None:
        result = forecast_performance_diagnostics(
            [self.observation(realized_return=0.0, probability_positive=None)],
            horizon_sessions=2,
        )

        self.assertEqual(result["no_change_mae_pct"], 0.0)
        self.assertIsNone(result["mae_skill_score"])
        self.assertIsNone(result["brier_score"])
        self.assertIsNone(result["fair_coin_brier_score"])
        self.assertIsNone(result["brier_skill_score_vs_fair_coin"])
        self.assertEqual(result["always_not_up_accuracy_pct"], 100.0)

    def test_deduplication_does_not_hide_mixed_provenance(self) -> None:
        first = self.observation()
        different_version = replace(first, prediction_id="v2", model_version="v2")

        with self.assertRaises(MixedForecastCohortError):
            forecast_performance_diagnostics(
                [first, different_version],
                horizon_sessions=2,
            )

    def test_empty_observations_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one matured forecast"):
            forecast_performance_diagnostics([], horizon_sessions=2)


if __name__ == "__main__":
    unittest.main()
