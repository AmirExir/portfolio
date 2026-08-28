from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from market_agent.agent import forecast as forecast_module
from market_agent.agent.forecast import (
    FORECAST_POSTPROCESSOR_VERSION,
    ForecastResult,
    ForecastPostprocessorState,
    _direct_holdout_metrics,
    _ensemble_result,
    _expected_calibration_error_pct,
    _forecast_evaluation_metrics,
    _future_index,
    _postprocess_horizon_return,
    backtest_forecasts,
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
            "holdout_raw_predicted_return_pct": predicted,
            "holdout_transformed_predicted_return_pct": predicted,
            "holdout_predicted_return_pct": predicted,
            "holdout_actual_return_pct": actual,
            "holdout_probability_up": probabilities,
            "holdout_samples": len(actual),
            "postprocessor_version": FORECAST_POSTPROCESSOR_VERSION,
            "training_positive_base_rate": 0.5,
            "validation_is_oos": True,
            "horizon_days": 30,
            "live_eligible": True,
        },
        model_name=name,
    )


class ForecastValidationTests(unittest.TestCase):
    def test_backtest_direction_treats_flat_actual_as_not_up(self) -> None:
        index = pd.bdate_range("2026-01-02", periods=30)
        frame = pd.DataFrame({"close": np.full(len(index), 100.0)}, index=index)
        mocked_result = ForecastResult(
            forecast=pd.DataFrame(
                {"forecast_close": [99.0]},
                index=pd.bdate_range("2026-03-02", periods=1),
            ),
            metrics={
                "probability_up_pct": 10.0,
                "confidence_pct": 90.0,
                "expected_error_pct": 2.0,
            },
            model_name="mocked direct model",
        )

        with patch.object(
            forecast_module,
            "forecast_close_prices",
            return_value=mocked_result,
        ):
            result = backtest_forecasts(
                frame,
                horizon_days=1,
                lookback_window=5,
                max_points=5,
                min_train_samples=20,
                optimize_model=False,
            )

        self.assertTrue(result.forecasts["direction_correct"].all())
        self.assertEqual(result.metrics["historical_direction_accuracy"], 100.0)

    def test_live_forecast_forces_cold_fit_when_warm_start_is_requested(self) -> None:
        with (
            patch.object(
                forecast_module,
                "_fit_model",
                wraps=forecast_module._fit_model,
            ) as fit_model,
            patch.object(
                forecast_module,
                "_load_model_weight_artifact",
            ) as load_artifact,
        ):
            result = forecast_close_prices(
                _price_frame(),
                horizon_days=30,
                lookback_window=20,
                ridge_alpha=10.0,
                optimize_model=False,
                model_type="ridge",
                symbol="AAPL",
                warm_start=True,
            )

        self.assertTrue(
            all(
                call.kwargs.get("warm_start") is False
                for call in fit_model.call_args_list
            )
        )
        load_artifact.assert_not_called()
        self.assertTrue(result.metrics["warm_start_requested"])
        self.assertFalse(result.metrics["warm_start_live_enabled"])
        self.assertEqual(
            result.metrics["warm_start_validation_status"],
            "disabled_until_equivalent_oos_evaluation",
        )

    def test_sequence_results_are_shadow_only_and_excluded_from_ensemble(self) -> None:
        component = _result(
            "direct model",
            forecast_value=105.0,
            predicted=[1.0, 2.0],
            actual=[1.5, 1.0],
            probabilities=[0.6, 0.7],
        )

        with (
            patch.object(forecast_module, "XGBRegressor", None),
            patch.object(forecast_module, "MLPRegressor", None),
            patch.object(
                forecast_module,
                "forecast_close_prices",
                return_value=component,
            ),
        ):
            results = forecast_module.compare_forecast_models(
                _price_frame(),
                sequence_model="lstm",
                warm_start=False,
            )

        sequence = results["LSTM"]
        self.assertFalse(sequence.metrics["live_eligible"])
        self.assertTrue(sequence.metrics["shadow_mode"])
        self.assertTrue(sequence.metrics["promotion_required"])
        self.assertEqual(
            sequence.metrics["promotion_status"],
            "prospective_component_shadow_required",
        )
        self.assertEqual(
            results["Ensemble"].metrics["component_weights"],
            {"Ridge": 1.0},
        )

    def test_zero_return_is_the_not_up_direction_class(self) -> None:
        metrics = _forecast_evaluation_metrics(
            np.asarray([0.0, 0.0, -1.0, 1.0]),
            np.asarray([0.0, -1.0, 1.0, 0.0]),
            np.asarray([0.4, 0.4, 0.4, 0.6]),
            positive_base_rate=0.25,
            horizon_days=1,
        )

        self.assertEqual(metrics["training_direction_baseline"], "not_up")
        self.assertAlmostEqual(metrics["holdout_direction_accuracy"], 50.0)
        self.assertAlmostEqual(
            metrics["direction_baseline_accuracy_pct"],
            75.0,
        )
        self.assertAlmostEqual(metrics["direction_skill_pct"], -25.0)
        self.assertAlmostEqual(metrics["direction_skill_score"], -1.0)

    def test_future_index_uses_explicit_crypto_symbol_calendar(self) -> None:
        weekday_only_history = pd.bdate_range(
            "2025-12-29",
            "2026-01-02",
        )

        crypto_future = _future_index(
            weekday_only_history,
            2,
            symbol="BTC-USD",
        )
        equity_future = _future_index(
            weekday_only_history,
            2,
            symbol="AAPL",
        )

        self.assertEqual(
            list(crypto_future.date),
            [
                pd.Timestamp("2026-01-03").date(),
                pd.Timestamp("2026-01-04").date(),
            ],
        )
        self.assertEqual(
            list(equity_future.date),
            [
                pd.Timestamp("2026-01-05").date(),
                pd.Timestamp("2026-01-06").date(),
            ],
        )

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

    def test_live_and_holdout_use_the_same_training_only_transform(self) -> None:
        result = forecast_close_prices(
            _price_frame(),
            horizon_days=30,
            lookback_window=20,
            ridge_alpha=10.0,
            optimize_model=False,
            model_type="ridge",
            warm_start=False,
        )
        metrics = result.metrics

        self.assertEqual(
            metrics["postprocessor_version"],
            FORECAST_POSTPROCESSOR_VERSION,
        )
        holdout_state = ForecastPostprocessorState(
            **metrics["holdout_postprocessor_state"]
        )
        holdout_raw_log = np.log1p(
            np.asarray(metrics["holdout_raw_predicted_return_pct"])
            / 100.0
        )
        expected_holdout = [
            _postprocess_horizon_return(value, holdout_state)
            for value in holdout_raw_log
        ]
        np.testing.assert_allclose(
            metrics["holdout_transformed_predicted_return_pct"],
            [
                np.expm1(item["transformed_horizon_return"]) * 100.0
                for item in expected_holdout
            ],
        )
        np.testing.assert_allclose(
            metrics["holdout_probability_up"],
            [item["probability_up"] for item in expected_holdout],
        )

        live_state = ForecastPostprocessorState(
            **metrics["live_postprocessor_state"]
        )
        live_raw_log = np.log1p(
            metrics["raw_forecast_change_pct"] / 100.0
        )
        expected_live = _postprocess_horizon_return(
            live_raw_log,
            live_state,
        )
        self.assertAlmostEqual(
            metrics["forecast_change_pct"],
            np.expm1(expected_live["transformed_horizon_return"]) * 100.0,
        )
        self.assertAlmostEqual(
            metrics["probability_up_pct"] / 100.0,
            expected_live["probability_up"],
        )

    def test_outer_actuals_do_not_change_postprocessor_or_predictions(self) -> None:
        sample_count = 80
        horizon_days = 4
        holdout_size = 12
        features = np.arange(sample_count * 3, dtype=float).reshape(
            sample_count,
            3,
        )
        targets = np.linspace(-0.04, 0.04, sample_count)
        changed_targets = targets.copy()
        changed_targets[-holdout_size:] = np.linspace(
            0.15,
            -0.15,
            holdout_size,
        )
        raw_predictions = np.linspace(-0.03, 0.03, holdout_size)
        fixed_model = {"residual_std": 0.02}

        def holdout_metrics(target_values: np.ndarray) -> dict:
            with (
                patch(
                    "market_agent.agent.forecast._training_data_for_model",
                    return_value=(features, target_values),
                ),
                patch(
                    "market_agent.agent.forecast._fit_model",
                    return_value=fixed_model,
                ),
                patch(
                    "market_agent.agent.forecast._predict_matrix",
                    return_value=raw_predictions,
                ),
            ):
                return _direct_holdout_metrics(
                    np.zeros(100),
                    lookback_window=10,
                    horizon_days=horizon_days,
                    alpha=1.0,
                    fixed_holdout_size=holdout_size,
                )

        original = holdout_metrics(targets)
        changed = holdout_metrics(changed_targets)

        self.assertEqual(
            original["holdout_postprocessor_state"],
            changed["holdout_postprocessor_state"],
        )
        self.assertEqual(
            original["holdout_raw_predicted_return_pct"],
            changed["holdout_raw_predicted_return_pct"],
        )
        self.assertEqual(
            original["holdout_transformed_predicted_return_pct"],
            changed["holdout_transformed_predicted_return_pct"],
        )
        self.assertEqual(
            original["holdout_probability_up"],
            changed["holdout_probability_up"],
        )
        self.assertNotEqual(
            original["holdout_actual_return_pct"],
            changed["holdout_actual_return_pct"],
        )

    def test_direct_metrics_include_baseline_skill_and_effective_n(self) -> None:
        sample_count = 60
        horizon_days = 4
        holdout_size = 8
        features = np.arange(sample_count * 2, dtype=float).reshape(
            sample_count,
            2,
        )
        targets = np.zeros(sample_count, dtype=float)
        targets[:48] = np.asarray([0.01] * 36 + [-0.01] * 12)
        actual_pct = np.asarray([1.0, -1.0, 2.0, -2.0] * 2)
        targets[-holdout_size:] = np.log1p(actual_pct / 100.0)

        with (
            patch(
                "market_agent.agent.forecast._training_data_for_model",
                return_value=(features, targets),
            ),
            patch(
                "market_agent.agent.forecast._fit_model",
                return_value={"residual_std": 0.02},
            ),
            patch(
                "market_agent.agent.forecast._predict_matrix",
                return_value=np.zeros(holdout_size),
            ),
        ):
            metrics = _direct_holdout_metrics(
                np.zeros(100),
                lookback_window=10,
                horizon_days=horizon_days,
                alpha=1.0,
                fixed_holdout_size=holdout_size,
            )

        self.assertAlmostEqual(metrics["training_positive_base_rate"], 0.75)
        self.assertEqual(metrics["training_direction_baseline"], "up")
        self.assertAlmostEqual(metrics["zero_return_mae_pct"], 1.5)
        self.assertAlmostEqual(
            metrics["zero_return_rmse_pct"],
            np.sqrt(2.5),
        )
        self.assertAlmostEqual(metrics["mae_skill_score"], 0.0)
        self.assertAlmostEqual(
            metrics["direction_baseline_accuracy_pct"],
            50.0,
        )
        self.assertAlmostEqual(metrics["direction_skill_pct"], 0.0)
        self.assertAlmostEqual(metrics["direction_skill_score"], 0.0)
        self.assertAlmostEqual(
            metrics["probability_baseline_brier_score"],
            0.3125,
        )
        self.assertAlmostEqual(metrics["brier_skill_score"], 0.2)
        self.assertEqual(metrics["holdout_nonoverlapping_samples"], 2)
        self.assertAlmostEqual(metrics["holdout_effective_samples"], 2.0)
        self.assertEqual(
            metrics["holdout_overlap_stride_sessions"],
            horizon_days,
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
        self.assertEqual(
            ensemble.metrics["holdout_raw_predicted_return_pct"],
            [2.0, 0.0],
        )
        self.assertEqual(
            ensemble.metrics["holdout_transformed_predicted_return_pct"],
            [2.0, 0.0],
        )
        self.assertAlmostEqual(ensemble.metrics["holdout_mae_pct"], 1.0)
        self.assertAlmostEqual(ensemble.metrics["zero_return_mae_pct"], 1.0)
        self.assertAlmostEqual(ensemble.metrics["mae_skill_score"], 0.0)
        self.assertAlmostEqual(
            ensemble.metrics["probability_baseline_brier_score"],
            0.25,
        )
        self.assertAlmostEqual(ensemble.metrics["brier_skill_score"], 0.64)
        self.assertEqual(
            ensemble.metrics["holdout_nonoverlapping_samples"],
            1,
        )
        self.assertAlmostEqual(
            ensemble.metrics["holdout_effective_samples"],
            1.0,
        )

    def test_ensemble_requires_literal_true_oos_from_every_component(self) -> None:
        for invalid_oos_value in ("false", 1):
            with self.subTest(validation_is_oos=invalid_oos_value):
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
                second.metrics["validation_is_oos"] = invalid_oos_value

                ensemble = _ensemble_result(
                    {"Ridge": first, "XGBoost": second}
                )

                self.assertFalse(ensemble.metrics["validation_is_oos"])


if __name__ == "__main__":
    unittest.main()
