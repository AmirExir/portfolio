from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from market_agent.agent.forecast import (
    ForecastResult,
    _ensemble_result,
    _expected_calibration_error_pct,
    forecast_close_prices,
)


def _price_frame(rows: int = 360) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=rows)
    step = np.arange(rows, dtype=float)
    close = 100.0 * np.exp(0.0006 * step + 0.02 * np.sin(step / 13.0))
    return pd.DataFrame(
        {
            "open": close * (1.0 + 0.001 * np.sin(step)),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000.0 + 100_000.0 * np.cos(step / 9.0),
        },
        index=index,
    )


def _result(
    name: str,
    *,
    forecast_value: float,
    predicted: list[float],
    actual: list[float],
    probabilities: list[float],
) -> ForecastResult:
    index = pd.bdate_range("2026-01-02", periods=2)
    forecast = pd.DataFrame(
        {
            "forecast_close": [forecast_value, forecast_value],
            "lower_estimate": [forecast_value * 0.95] * 2,
            "upper_estimate": [forecast_value * 1.05] * 2,
            "expected_daily_return_pct": [0.0, 0.0],
        },
        index=index,
    )
    return ForecastResult(
        forecast=forecast,
        metrics={
            "forecast_change_pct": forecast_value - 100.0,
            "probability_up_pct": 60.0,
            "confidence_pct": 60.0,
            "expected_error_pct": 2.0,
            "holdout_predicted_return_pct": predicted,
            "holdout_actual_return_pct": actual,
            "holdout_probability_up": probabilities,
            "holdout_samples": len(actual),
            "validation_is_oos": True,
            "horizon_days": 30,
            "live_eligible": True,
        },
        model_name=name,
    )


class ForecastValidationTests(unittest.TestCase):
    def test_optimized_model_uses_nested_untouched_outer_holdout(self) -> None:
        result = forecast_close_prices(
            _price_frame(),
            horizon_days=30,
            lookback_window=20,
            ridge_alpha=10.0,
            optimize_model=True,
            model_type="ridge",
            warm_start=False,
        )

        metrics = result.metrics
        self.assertTrue(metrics["validation_is_oos"])
        self.assertTrue(metrics["hyperparameter_selection_is_nested"])
        self.assertEqual(
            metrics["validation_scheme"],
            "nested_purged_outer_holdout",
        )
        self.assertGreaterEqual(metrics["purge_sessions"], 30)
        self.assertEqual(
            len(metrics["holdout_predicted_return_pct"]),
            metrics["holdout_samples"],
        )
        outer_start = metrics["validation_test_start_sample"]
        inner_end = metrics[
            "selection_validation_test_end_sample_exclusive"
        ]
        self.assertEqual(
            metrics["selection_reserved_outer_test_samples"],
            metrics["holdout_samples"],
        )
        self.assertGreaterEqual(
            outer_start - inner_end,
            30,
        )

    def test_ece_detects_offsetting_catastrophic_probabilities(self) -> None:
        error = _expected_calibration_error_pct(
            np.asarray([0.0, 1.0]),
            np.asarray([1.0, 0.0]),
        )
        self.assertEqual(error, 100.0)

    def test_ensemble_uses_fixed_weights_and_real_outer_predictions(self) -> None:
        first = _result(
            "Ridge",
            forecast_value=102.0,
            predicted=[1.0, -1.0],
            actual=[1.0, -1.0],
            probabilities=[0.8, 0.2],
        )
        second = _result(
            "XGBoost",
            forecast_value=104.0,
            predicted=[3.0, 1.0],
            actual=[1.0, -1.0],
            probabilities=[0.6, 0.4],
        )

        ensemble = _ensemble_result({"Ridge": first, "XGBoost": second})

        self.assertEqual(
            ensemble.metrics["component_weights"],
            {"Ridge": 0.5, "XGBoost": 0.5},
        )
        self.assertEqual(
            ensemble.metrics["ensemble_weighting"],
            "fixed_equal_non_rl",
        )
        self.assertTrue(ensemble.metrics["validation_is_oos"])
        self.assertEqual(
            ensemble.metrics["holdout_predicted_return_pct"],
            [2.0, 0.0],
        )
        self.assertAlmostEqual(ensemble.metrics["holdout_mae_pct"], 1.0)


if __name__ == "__main__":
    unittest.main()
