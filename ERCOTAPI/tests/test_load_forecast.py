"""Deterministic regression checks for causal dashboard forecasts."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from ERCOTAPI import load_forecast as forecast


def hourly_frame(count: int = 120) -> pd.DataFrame:
    """Return known MW values with a strictly hourly time axis."""
    return pd.DataFrame({"load": np.arange(count, dtype=float) + 100},
                        index=pd.date_range("2025-01-01", periods=count, freq="h"))


class IdentityScaler:
    def transform(self, values: pd.DataFrame) -> pd.DataFrame:
        return values


class PreviousDayModel:
    def __init__(self) -> None:
        self.rows: list[pd.DataFrame] = []

    def predict(self, values: pd.DataFrame) -> np.ndarray:
        self.rows.append(values.copy())
        return values["load_lag_24h"].to_numpy()


class LoadForecastTests(unittest.TestCase):
    def test_features_exclude_current_and_future_load(self) -> None:
        original = hourly_frame()
        changed = original.copy()
        changed.iloc[40:, 0] += 10000
        before = forecast.build_load_features(original)
        after = forecast.build_load_features(changed)
        pd.testing.assert_frame_equal(before[forecast.FEATURES].iloc[:41], after[forecast.FEATURES].iloc[:41])
        self.assertAlmostEqual(before.iloc[40]["rolling_mean_24h"], original["load"].iloc[16:40].mean())
        self.assertEqual(before.iloc[40]["load_lag_24h"], original.iloc[16]["load"])

    def test_recursive_features_match_training_definition(self) -> None:
        history = hourly_frame(48)
        model = PreviousDayModel()
        result = forecast.forecast_load(model, IdentityScaler(), history, hours=30)
        np.testing.assert_array_equal(result.iloc[:24, 0], history.iloc[-24:, 0])
        np.testing.assert_array_equal(result.iloc[24:, 0], history.iloc[-24:-18, 0])
        extended = pd.concat([history, result.rename(columns={"predicted_load_mw": "load"})])
        expected = forecast.build_load_features(extended).loc[result.index, forecast.FEATURES]
        for position, row in enumerate(model.rows):
            np.testing.assert_allclose(row.iloc[0].to_numpy(), expected.iloc[position].to_numpy())
        pd.testing.assert_frame_equal(history, hourly_frame(48))

    def test_unsorted_source_is_sorted_without_mutation(self) -> None:
        source = hourly_frame().iloc[::-1]
        clean = forecast.validate_hourly_load(source)
        pd.testing.assert_frame_equal(clean, hourly_frame())
        self.assertFalse(source.index.is_monotonic_increasing)

    def test_ambiguous_or_missing_time_axis_is_rejected(self) -> None:
        base = hourly_frame()
        invalid = [base.reset_index(drop=True), base.drop(base.index[30]), pd.concat([base, base.iloc[:1]])]
        for frame in invalid:
            with self.subTest(index=type(frame.index), length=len(frame)), self.assertRaises(ValueError):
                forecast.validate_hourly_load(frame)

    def test_missing_infinite_negative_and_text_load_are_rejected(self) -> None:
        for value in (np.nan, np.inf, -1, "unknown"):
            source = hourly_frame().astype(object)
            source.iloc[3, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                forecast.validate_hourly_load(source)

    def test_aware_hour_axis_survives_daylight_saving_transition(self) -> None:
        source = hourly_frame(48)
        source.index = pd.date_range("2025-11-01 12:00", periods=48, freq="h", tz="America/Chicago")
        result = forecast.forecast_load(PreviousDayModel(), IdentityScaler(), source, 2)
        self.assertEqual(result.index[0], source.index[-1] + pd.Timedelta(hours=1))
        self.assertEqual(str(result.index.tz), "America/Chicago")

    def test_training_scales_each_cv_fold_and_refits_only_after_evaluation(self) -> None:
        fitted_indices = []

        class RecordingScaler(StandardScaler):
            def fit(self, X, y=None, sample_weight=None):
                fitted_indices.append(X.index.copy())
                return super().fit(X, y, sample_weight=sample_weight)

        source = hourly_frame()
        model = RandomForestRegressor(n_estimators=3, max_depth=2, random_state=42)
        with patch.object(forecast, "StandardScaler", RecordingScaler), patch.object(
            forecast, "_model_and_parameters", return_value=(model, {"n_estimators": [3]}, 1)
        ):
            fitted, scaler, frame, split, predictions, metrics = forecast.train_load_forecast_model(source)
        x_train, x_test, _, y_test = split
        self.assertLess(x_train.index[-1], x_test.index[0])
        self.assertGreater(len(fitted_indices), 3)
        for indices in fitted_indices[:-1]:
            self.assertLess(indices[-1], x_test.index[0])
        self.assertLess(len(fitted_indices[0]), len(x_train))
        self.assertEqual(len(fitted_indices[-1]), len(frame))
        np.testing.assert_allclose(scaler.mean_, frame[forecast.FEATURES].mean().to_numpy())
        self.assertAlmostEqual(metrics["persistence_mae"], 1.0)
        self.assertAlmostEqual(metrics["seasonal_mae"], 24.0)
        self.assertAlmostEqual(metrics["test_mae"], np.abs(y_test.to_numpy() - predictions[1]).mean())
        self.assertEqual(metrics["forecast_refit_rows"], len(frame))
        self.assertEqual(len(forecast.forecast_load(fitted, scaler, source)), 24)

    def test_short_history_has_no_model_and_bad_horizon_is_rejected(self) -> None:
        self.assertEqual(forecast.train_load_forecast_model(hourly_frame(47)), (None,) * 6)
        for hours in (0, -1, True, 1.5):
            with self.subTest(hours=hours), self.assertRaises(ValueError):
                forecast.forecast_load(PreviousDayModel(), IdentityScaler(), hourly_frame(), hours)

    def test_invalid_model_output_is_not_silently_clipped(self) -> None:
        class InvalidModel:
            def predict(self, values):
                return [-1]

        with self.assertRaisesRegex(ValueError, "invalid load"):
            forecast.forecast_load(InvalidModel(), IdentityScaler(), hourly_frame())


if __name__ == "__main__":
    unittest.main()
