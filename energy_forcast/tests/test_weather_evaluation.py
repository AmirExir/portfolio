"""Leakage boundaries, calibration arithmetic, and seasonal fold regression cases."""

import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from energy_forcast.evaluation import add_intervals, calibrate_intervals, evaluation_summary, season_labels
from energy_forcast.forecasting import (
    ForecastConfig, forecast_with_intervals, make_features, recursive_backtest, recursive_forecast, run_experiment,
)
from energy_forcast.weather import read_weather_csv


def archive(rows: list[list[object]]) -> bytes:
    """Serialize synthetic forecast vintages used only in deterministic tests."""
    return pd.DataFrame(rows, columns=["valid_time", "issued_at", "available_at", "temperature_c",
                                       "record_type", "source"]).to_csv(index=False).encode()


def load_series(hours: int = 2000) -> pd.Series:
    """Return a deterministic positive test load."""
    return pd.Series(np.full(hours, 100.0), index=pd.date_range("2020-01-01", periods=hours, freq="h"), name="MW")


class WeatherTests(unittest.TestCase):
    def test_latest_issued_and_available_vintage_not_late_observation(self) -> None:
        weather = read_weather_csv(archive([
            ["2020-01-02 00:00", "2020-01-01 00:00", "2020-01-01 01:00", 10, "forecast", "fixture"],
            ["2020-01-02 00:00", "2020-01-01 12:00", "2020-01-01 13:00", 20, "forecast", "fixture"],
            ["2020-01-02 00:00", "2020-01-01 06:00", "2020-01-01 14:00", 99, "forecast", "fixture"],
            ["2020-01-02 00:00", "2020-01-01 23:00", "2020-01-02 01:00", 999, "forecast", "fixture"],
        ]))
        targets = pd.DatetimeIndex(["2020-01-02"] * 4)
        origins = pd.DatetimeIndex(["2020-01-01", "2020-01-01 12:00", "2020-01-01 13:00", "2020-01-01 23:00"])
        values = weather.at_origins(targets, origins).temperature_c.to_numpy()
        self.assertTrue(np.isnan(values[0]))
        np.testing.assert_array_equal(values[1:], [10, 20, 20])

    def test_rejects_observed_weather_bad_provenance_and_ambiguous_vintages(self) -> None:
        valid = ["2020-01-02", "2020-01-01", "2020-01-01 01:00", 10, "forecast", "fixture"]
        cases = []
        for column, value in ((4, "observed"), (5, ""), (2, "2019-12-31"),
                              (1, "2020-01-02"), (3, float("nan")), (0, "2020-01-02 00:15"),
                              (1, "2020-01-01T00:00Z")):
            row = valid.copy()
            row[column] = value
            cases.append(archive([row]))
        cases.extend([archive([valid, valid]), b"Datetime,temperature_c\n2020-01-01,10\n"])
        for content in cases:
            with self.subTest(content=content), self.assertRaises(ValueError):
                read_weather_csv(content)

    def test_recursive_weather_is_frozen_at_origin_for_all_48_hours(self) -> None:
        history = load_series(200)
        origin = history.index[-1]
        targets = pd.date_range(origin + pd.Timedelta(hours=1), periods=48, freq="h")
        rows = []
        for target in targets:
            rows.append([target, origin - pd.Timedelta(hours=1), origin, 10, "forecast", "fixture"])
            if target > origin + pd.Timedelta(hours=1):
                rows.append([target, origin + pd.Timedelta(minutes=30), origin + pd.Timedelta(hours=1),
                             99, "forecast", "fixture"])
        weather = read_weather_csv(archive(rows))

        class TemperaturePredictor:
            def predict(self, features: pd.DataFrame) -> np.ndarray:
                return features.temperature_c.to_numpy()

        result = recursive_forecast(TemperaturePredictor(), history, 48, weather)
        self.assertTrue(result.eq(10).all())
        # Future issue changes never enter the 48h trajectory, even after their publication hour.
        partial = read_weather_csv(archive(rows[:-2]))
        with self.assertRaisesRegex(ValueError, "does not cover"):
            recursive_forecast(TemperaturePredictor(), history, 48, partial)

    def test_one_step_weather_cutoff_is_previous_hour(self) -> None:
        load = load_series(200)
        target = load.index[-1]
        weather = read_weather_csv(archive([
            [target, target - pd.Timedelta(hours=24), target - pd.Timedelta(hours=1), 10, "forecast", "fixture"],
            [target, target - pd.Timedelta(minutes=30), target - pd.Timedelta(minutes=15), 99, "forecast", "fixture"],
        ]))
        features = make_features(load, weather)
        self.assertEqual(features.temperature_c.iloc[-1], 10)
        self.assertEqual(features.temperature_c.isna().sum(), 199)


class IntervalTests(unittest.TestCase):
    def calibration_rows(self) -> pd.DataFrame:
        origins = pd.date_range("2020-01-01", periods=19, freq="2D")
        return pd.DataFrame({"Origin": origins, "Lead time (hours)": 1, "XGBoost": 100,
                             "Actual (MW)": 100 + np.arange(1, 20)})

    def test_known_finite_sample_quantile_and_unseen_target_coverage(self) -> None:
        calibration = calibrate_intervals(self.calibration_rows(), coverage=0.9)
        self.assertEqual(calibration.loc[1, "Radius (MW)"], 18)
        tests = pd.DataFrame({"XGBoost": [100, 100], "Actual (MW)": [118, 119],
                              "Lead time (hours)": 1, "Horizon (hours)": 1,
                              "Origin": pd.to_datetime(["2021-01-01", "2021-01-02"]),
                              "Persistence": [100, 100], "Previous day": [100, 100], "Previous week": [100, 100]},
                             index=pd.to_datetime(["2021-01-01 01:00", "2021-01-02 01:00"]))
        bounded = add_intervals(tests, calibration)
        summary = evaluation_summary(bounded, ["Horizon (hours)"])
        self.assertEqual(summary.loc[summary.Model == "XGBoost", "Empirical coverage"].iloc[0], 0.5)
        self.assertEqual(summary.loc[summary.Model == "XGBoost", "Mean width (MW)"].iloc[0], 36)
        altered = tests.copy()
        altered["Actual (MW)"] = 10000
        assert_frame_equal(add_intervals(altered, calibration)[["Lower (MW)", "Upper (MW)"]],
                           bounded[["Lower (MW)", "Upper (MW)"]])

    def test_insufficient_calibration_is_missing_and_duplicate_residuals_rejected(self) -> None:
        calibration = calibrate_intervals(self.calibration_rows().iloc[:3], coverage=0.99, min_origins=1)
        self.assertTrue(calibration["Radius (MW)"].isna().all())
        with self.assertRaisesRegex(ValueError, "unique"):
            calibrate_intervals(pd.concat([self.calibration_rows(), self.calibration_rows()]))
        bounded = add_intervals(pd.DataFrame({"XGBoost": [2], "Lead time (hours)": [1]}),
                                calibrate_intervals(self.calibration_rows()))
        self.assertEqual(bounded["Lower (MW)"].iloc[0], 0)

    def test_seasons_assign_december_to_following_winter(self) -> None:
        np.testing.assert_array_equal(season_labels(pd.DatetimeIndex(["2020-12-01", "2021-02-01", "2021-03-01"])),
                                      ["2021 Winter", "2021 Winter", "2021 Spring"])


class IntegrationTests(unittest.TestCase):
    def test_expanding_folds_and_calibration_have_no_future_fit_or_selection(self) -> None:
        fitted = []

        class ConstantModel:
            def __init__(self, **kwargs: object) -> None:
                pass

            def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
                self.fit_index = features.index.copy()
                fitted.append(self)

            def predict(self, features: pd.DataFrame) -> np.ndarray:
                return np.full(len(features), 100.0)

        load = load_series(24 * 730)
        # Source files can start at 01:00; season boundaries must still be midnight.
        load.index = load.index + pd.Timedelta(hours=1)
        config = ForecastConfig(candidate_depths=(2,), max_origins=10, rolling_seasons=4)
        with patch("xgboost.XGBRegressor", ConstantModel):
            result = run_experiment(load, {"source_sha256": "synthetic-test"}, config)
        self.assertEqual(set(result.horizon_predictions["Horizon (hours)"]), {24, 48})
        self.assertEqual(len(result.metadata["rolling_seasons"]), 4)
        self.assertTrue(result.seasonal_metrics.Fold.eq(result.seasonal_metrics.Season).all())
        for number, report in enumerate(result.metadata["rolling_seasons"]):
            tuned, model = fitted[2 + number * 2:4 + number * 2]
            self.assertLess(tuned.fit_index[-1], pd.Timestamp(report["selection"]["start"]))
            self.assertLess(model.fit_index[-1], pd.Timestamp(report["calibration"]["start"]))
            self.assertLess(pd.Timestamp(report["calibration"]["end"]), pd.Timestamp(report["test_start"]))
        main_calibration_start = pd.Timestamp(result.metadata["periods"]["interval_calibration"]["start"])
        self.assertLess(fitted[1].fit_index[-1], main_calibration_start)
        self.assertEqual(fitted[-1].fit_index[-1], load.index[-1])
        self.assertEqual(result.horizon_predictions["Lead time (hours)"].max(), 48)
        output = forecast_with_intervals(result, load, 48)
        self.assertEqual(len(output), 48)
        self.assertTrue(output["Lower (MW)"].notna().all())
        self.assertTrue(output["Lower (MW)"].eq(100).all())
        json.dumps(result.metadata, allow_nan=False)

    def test_optional_weather_experiment_and_future_forecast(self) -> None:
        load = load_series()
        targets = pd.date_range(load.index[0], periods=len(load) + 48, freq="h")
        weather = read_weather_csv(archive([
            [target, target - pd.Timedelta(hours=48), target - pd.Timedelta(hours=48), 10, "forecast", "fixture"]
            for target in targets]))

        class WeatherModel:
            def __init__(self, **kwargs: object) -> None:
                pass

            def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
                if "temperature_c" not in features:
                    raise AssertionError("Missing weather feature")

            def predict(self, features: pd.DataFrame) -> np.ndarray:
                return features.temperature_c.to_numpy() * 10

        with patch("xgboost.XGBRegressor", WeatherModel):
            result = run_experiment(load, {}, ForecastConfig(candidate_depths=(2,), max_origins=2), weather)
        self.assertTrue(result.metadata["weather"]["enabled"])
        self.assertTrue(forecast_with_intervals(result, load, 48, weather).XGBoost.eq(100).all())
        with self.assertRaisesRegex(ValueError, "feature set"):
            forecast_with_intervals(result, load, 48)


if __name__ == "__main__":
    unittest.main()
