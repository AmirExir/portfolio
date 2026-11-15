"""Regression checks for optional forecast-weather features and missing coverage."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from ERCOTAPI import load_forecast as forecast


def hourly_frame(count: int = 120) -> pd.DataFrame:
    """Return deterministic hourly MW on ERCOT's local, offset-aware clock."""
    return pd.DataFrame(
        {"load": np.arange(count, dtype=float) + 100},
        index=pd.date_range("2026-11-01 00:00", periods=count, freq="h", tz="America/Chicago"),
    )


def temperatures(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Return separately named site temperatures without geographic weighting."""
    return pd.DataFrame(
        {"temperature_c_houston": np.arange(len(index), dtype=float) / 10 + 15,
         "temperature_c_dallas": np.arange(len(index), dtype=float) / 10 + 10},
        index=index.tz_convert("UTC"),
    )


def small_model() -> tuple:
    """Keep tests deterministic while exercising the real sklearn pipeline."""
    return RandomForestRegressor(n_estimators=3, max_depth=2, random_state=42), {"n_estimators": [3]}, 1


class NamedIdentityScaler:
    """Expose the fitted feature contract while keeping values inspectable."""

    feature_names_in_ = np.array(forecast.FEATURES + ["temperature_c_dallas"])

    def transform(self, values: pd.DataFrame) -> pd.DataFrame:
        return values


class WeatherIncrementModel:
    """Use both recursive lag updates and the temperature for each target hour."""

    def __init__(self) -> None:
        self.rows: list[pd.DataFrame] = []

    def predict(self, values: pd.DataFrame) -> np.ndarray:
        self.rows.append(values.copy())
        return (values["load_lag_1h"] + values["temperature_c_dallas"]).to_numpy()


class WeatherLoadForecastTests(unittest.TestCase):
    def test_training_preserves_original_holdout_boundary_and_reports_coverage(self) -> None:
        source = hourly_frame()
        weather = temperatures(source.index)
        weather.iloc[24:34] = np.nan
        weather.iloc[100] = np.nan
        original_source, original_weather = source.copy(deep=True), weather.copy(deep=True)
        fitted_indices = []

        class RecordingScaler(StandardScaler):
            def fit(self, X, y=None, sample_weight=None):
                fitted_indices.append(X.index.copy())
                return super().fit(X, y, sample_weight=sample_weight)

        with patch.object(forecast, "StandardScaler", RecordingScaler), patch.object(
            forecast, "_model_and_parameters", return_value=small_model()
        ):
            model, scaler, frame, split, _, metadata = forecast.train_load_forecast_model(
                source, training_weather=weather,
            )
        x_train, x_test, _, _ = split
        expected_names = forecast.FEATURES + ["temperature_c_dallas", "temperature_c_houston"]
        self.assertEqual(list(scaler.feature_names_in_), expected_names)
        self.assertEqual(list(x_train.columns), expected_names)
        self.assertEqual(x_train.index[-1], source.index[99])
        self.assertEqual(x_test.index[0], source.index[101])
        self.assertEqual(metadata["weather_holdout_boundary"], str(source.index[100]))
        self.assertEqual(metadata["weather_missing_rows"], 11)
        self.assertEqual(metadata["weather_eligible_rows"], 85)
        self.assertEqual(metadata["weather_training_rows"], 66)
        self.assertEqual(metadata["weather_test_rows"], 19)
        self.assertEqual(metadata["method_version"], "causal-hourly-weather-v1")
        self.assertEqual(metadata["weather_units"], "degrees_Celsius")
        self.assertEqual(len(metadata["weather_features_sha256"]), 64)
        for index in fitted_indices[:-1]:
            self.assertLess(index[-1], source.index[100])
        pd.testing.assert_index_equal(fitted_indices[-1], frame.index)
        np.testing.assert_allclose(scaler.mean_, frame[expected_names].mean().to_numpy())
        self.assertEqual(metadata["persistence_mae"], 1.0)
        self.assertEqual(metadata["seasonal_mae"], 24.0)
        future = pd.date_range(source.index[-1] + pd.Timedelta(hours=1), periods=25, freq="h")
        result = forecast.forecast_load(model, scaler, source, 25, future_weather=temperatures(future))
        self.assertEqual(len(result), 25)
        pd.testing.assert_frame_equal(source, original_source)
        pd.testing.assert_frame_equal(weather, original_weather)

    def test_late_only_weather_cannot_move_holdout_into_training(self) -> None:
        source = hourly_frame(240)
        weather = temperatures(source.index)
        weather.iloc[:191] = np.nan
        # 49 complete rows exist, but fewer than 24 precede the original holdout.
        with patch.object(forecast, "_model_and_parameters") as factory:
            with self.assertRaisesRegex(ValueError, "Insufficient archived forecast weather"):
                forecast.train_load_forecast_model(source, training_weather=weather)
        factory.assert_not_called()

    def test_insufficient_weather_is_reported_without_load_only_fallback(self) -> None:
        source = hourly_frame()
        weather = temperatures(source.index)
        weather.iloc[:73] = np.nan
        with self.assertRaisesRegex(forecast.WeatherInputError, "47 total"):
            forecast.train_load_forecast_model(source, training_weather=weather)
        with self.assertRaisesRegex(forecast.WeatherInputError, "0 total"):
            forecast.train_load_forecast_model(source, training_weather=weather * np.nan)

    def test_inference_uses_target_weather_and_elapsed_hour_during_dst(self) -> None:
        source = hourly_frame(24)
        source.index = pd.date_range(end="2026-11-01 00:00", periods=24, freq="h", tz="America/Chicago")
        future = pd.date_range(source.index[-1] + pd.Timedelta(hours=1), periods=3, freq="h")
        weather = pd.DataFrame({"temperature_c_dallas": [10., 20., 30.]}, index=future.tz_convert("UTC"))
        model = WeatherIncrementModel()
        result = forecast.forecast_load(model, NamedIdentityScaler(), source, 3, future_weather=weather)
        np.testing.assert_allclose(result["predicted_load_mw"], [133., 153., 183.])
        self.assertEqual(result.index[0].isoformat(), "2026-11-01T01:00:00-05:00")
        self.assertEqual(result.index[1].isoformat(), "2026-11-01T01:00:00-06:00")
        self.assertEqual([row["temperature_c_dallas"].iloc[0] for row in model.rows], [10., 20., 30.])
        self.assertEqual([row["load_lag_1h"].iloc[0] for row in model.rows], [123., 133., 153.])

    def test_missing_bridge_or_horizon_weather_fails_before_any_prediction(self) -> None:
        source = hourly_frame()
        future = pd.date_range(source.index[-1] + pd.Timedelta(hours=1), periods=25, freq="h")
        weather = temperatures(future)[["temperature_c_dallas"]]
        cases = [None, weather.iloc[1:], weather.iloc[:-1], weather.assign(temperature_c_dallas=np.nan)]
        for invalid in cases:
            with self.subTest(weather_rows=None if invalid is None else len(invalid)):
                model = WeatherIncrementModel()
                with self.assertRaisesRegex(forecast.WeatherInputError, "requires forecast temperatures|missing"):
                    forecast.forecast_load(model, NamedIdentityScaler(), source, 25, future_weather=invalid)
                self.assertEqual(model.rows, [])

    def test_feature_schema_mismatch_is_explicit(self) -> None:
        source = hourly_frame()
        future = pd.date_range(source.index[-1] + pd.Timedelta(hours=1), periods=2, freq="h")
        weather = temperatures(future)
        with self.assertRaisesRegex(ValueError, "match the fitted model"):
            forecast.forecast_load(WeatherIncrementModel(), NamedIdentityScaler(), source, 2, future_weather=weather)
        scaler = NamedIdentityScaler()
        scaler.feature_names_in_ = np.array(forecast.FEATURES)
        with self.assertRaisesRegex(ValueError, "fitted without weather"):
            forecast.forecast_load(WeatherIncrementModel(), scaler, source, 2, future_weather=weather)
        scaler.feature_names_in_ = np.array(forecast.FEATURES + ["observed_weather"])
        with self.assertRaisesRegex(ValueError, "schema is unsupported"):
            forecast.forecast_load(WeatherIncrementModel(), scaler, source, 2)

    def test_invalid_weather_inputs_do_not_train(self) -> None:
        source = hourly_frame()
        weather = temperatures(source.index)
        invalid = [
            weather.rename(columns={"temperature_c_dallas": "temperature_f_dallas"}),
            weather.assign(temperature_c_dallas=True),
            weather.assign(temperature_c_dallas="20"),
            weather.assign(temperature_c_dallas=np.inf),
            pd.concat([weather, weather.iloc[:1]]),
            weather.tz_localize(None),
            weather.set_axis(weather.index + pd.Timedelta(minutes=15)),
            weather.iloc[:, :0],
        ]
        for case in invalid:
            with self.subTest(columns=list(case.columns), rows=len(case)), patch.object(
                forecast, "_model_and_parameters"
            ) as factory:
                with self.assertRaises(ValueError):
                    forecast.train_load_forecast_model(source, training_weather=case)
                factory.assert_not_called()
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            forecast.train_load_forecast_model(source.tz_convert("UTC").tz_localize(None), training_weather=weather)

    def test_load_only_metadata_and_six_value_contract_remain_compatible(self) -> None:
        with patch.object(forecast, "_model_and_parameters", return_value=small_model()):
            result = forecast.train_load_forecast_model(hourly_frame(), tuning_mode="manual")
        self.assertEqual(len(result), 6)
        self.assertEqual(result[-1]["method_version"], "causal-hourly-v1")
        self.assertEqual(result[-1]["features"], forecast.FEATURES)
        self.assertNotIn("weather_features", result[-1])


if __name__ == "__main__":
    unittest.main()
