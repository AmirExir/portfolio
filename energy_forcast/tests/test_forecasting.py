"""Deterministic regression checks for causal load forecasting."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from energy_forcast.forecasting import (
    FEATURES, ForecastConfig, make_features, read_load_csv, recursive_backtest,
    recursive_forecast, regression_metrics, run_experiment, temporal_split,
)


def hourly_load(count: int = 2000) -> pd.Series:
    """Create positive deterministic hourly measurements."""
    return pd.Series(np.arange(count, dtype=float) + 100,
                     index=pd.date_range("2020-01-01", periods=count, freq="h", name="Datetime"), name="MW")


class RecordingModel:
    """Predict the previous load plus one, retaining inputs for causality checks."""

    def __init__(self) -> None:
        self.rows: list[pd.DataFrame] = []

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return a deterministic recursive forecast."""
        self.rows.append(features.copy())
        return features.lag_1.to_numpy() + 1


class InputTests(unittest.TestCase):
    def test_missing_hours_remain_unknown_and_input_order_is_sorted(self) -> None:
        load, report = read_load_csv(b"Datetime,MW\n2020-01-01 02:00,20\n2020-01-01 00:00,10\n")
        self.assertEqual(report["inserted_missing_hours"], 1)
        self.assertTrue(np.isnan(load.iloc[1]))
        self.assertEqual(load.iloc[2], 20)
        self.assertEqual(load.index.name, "Datetime")

    def test_duplicates_require_explicit_policy(self) -> None:
        content = b"Datetime,MW\n2020-01-01,10\n2020-01-01,20\n"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_load_csv(content)
        load, report = read_load_csv(content, "mean")
        self.assertEqual(load.iloc[0], 15)
        self.assertEqual(report["duplicate_extra_rows"], 1)
        self.assertEqual(len(report["source_sha256"]), 64)

    def test_duplicate_missing_measurement_is_not_silently_discarded(self) -> None:
        load, _ = read_load_csv(b"Datetime,MW\n2020-01-01,10\n2020-01-01,\n", "mean")
        self.assertTrue(load.isna().all())

    def test_accepts_original_aep_column(self) -> None:
        load, _ = read_load_csv(b"Datetime,AEP_MW\n2020-01-01,123\n")
        self.assertEqual(load.iloc[0], 123)

    def test_rejects_invalid_input(self) -> None:
        invalid = [b"", b"date,MW\n2020-01-01,1", b"Datetime,MW\nbad,1",
                   b"Datetime,MW\n,1", b"Datetime,MW\n2020-01-01,abc",
                   b"Datetime,MW\n2020-01-01,-1", b"Datetime,MW\n2020-01-01,inf",
                   b"Datetime,MW\n2020-01-01 01:30,1", b"Datetime,MW\n2020-01-01T01:00Z,1"]
        for content in invalid:
            with self.subTest(content=content), self.assertRaises(ValueError):
                read_load_csv(content)

    def test_bundled_data_quality_is_reported(self) -> None:
        path = Path(__file__).resolve().parents[1] / "AEP_hourly.csv"
        load, report = read_load_csv(path.read_bytes(), "mean")
        self.assertEqual(report["duplicate_extra_rows"], 4)
        self.assertEqual(report["inserted_missing_hours"], 27)
        self.assertEqual(load.index[-1], pd.Timestamp("2018-08-03"))


class FeatureTests(unittest.TestCase):
    def test_rolling_features_exclude_the_prediction_target(self) -> None:
        load = hourly_load(300)
        frame = make_features(load)
        point = 200
        self.assertEqual(frame.iloc[point].rolling_24h_mean, load.iloc[point - 24:point].mean())
        self.assertEqual(frame.iloc[point].rolling_168h_mean, load.iloc[point - 168:point].mean())
        self.assertEqual(frame.iloc[point].lag_168, load.iloc[point - 168])
        changed = load.copy()
        changed.iloc[point:] = 999999
        assert_frame_equal(frame.iloc[:point + 1], make_features(changed).iloc[:point + 1])

    def test_missing_measurement_invalidates_history_window_without_imputation(self) -> None:
        load = hourly_load(400)
        load.iloc[200] = np.nan
        frame = make_features(load)
        self.assertTrue(np.isnan(frame.iloc[201].lag_1))
        self.assertTrue(np.isnan(frame.iloc[368].rolling_168h_mean))
        self.assertFalse(np.isnan(frame.iloc[369].rolling_168h_mean))
        self.assertEqual(frame.iloc[225].lag_24, load.iloc[201])

    def test_irregular_grid_is_rejected_instead_of_using_row_lags(self) -> None:
        with self.assertRaisesRegex(ValueError, "hourly grid"):
            make_features(hourly_load().drop(hourly_load().index[5]))

    def test_temporal_partitions_are_disjoint_and_ordered(self) -> None:
        load = hourly_load()
        rows = make_features(load).assign(MW=load).dropna()
        train, validation, test = temporal_split(rows, ForecastConfig())
        self.assertLess(train.index[-1], validation.index[0])
        self.assertLess(validation.index[-1], test.index[0])
        self.assertEqual(len(train) + len(validation) + len(test), len(rows))
        self.assertEqual(len(train), int(0.7 * len(rows)))

    def test_insufficient_history_and_bad_configuration_are_actionable(self) -> None:
        with self.assertRaisesRegex(ValueError, "168 eligible"):
            temporal_split(make_features(hourly_load(300)).dropna(), ForecastConfig())
        for settings in ({"train_fraction": 0.9, "validation_fraction": 0.2},
                         {"candidate_depths": ()}, {"recursive_horizon": 0}, {"max_origins": 0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                ForecastConfig(**settings)


class ForecastTests(unittest.TestCase):
    def test_recursive_features_match_training_and_do_not_mutate_history(self) -> None:
        history = hourly_load(300)
        original = history.copy()
        model = RecordingModel()
        forecast = recursive_forecast(model, history, 48)
        expected = make_features(pd.concat([history, forecast]))
        for stamp, actual in zip(forecast.index, model.rows):
            np.testing.assert_allclose(actual.to_numpy(), expected.loc[[stamp]].to_numpy())
        np.testing.assert_allclose(forecast.to_numpy(), np.arange(400, 448))
        assert_series_equal(history, original)
        self.assertEqual(list(model.rows[0].columns), list(FEATURES))

    def test_forecast_requires_complete_recent_week_and_valid_horizon(self) -> None:
        history = hourly_load(300)
        history.iloc[-2] = np.nan
        with self.assertRaisesRegex(ValueError, "168 consecutive"):
            recursive_forecast(RecordingModel(), history, 24)
        for hours in (0, 169, 1.5, True):
            with self.subTest(hours=hours), self.assertRaises(ValueError):
                recursive_forecast(RecordingModel(), hourly_load(), hours)

    def test_invalid_model_output_is_not_clipped_or_hidden(self) -> None:
        for value in (-1, float("nan"), float("inf")):
            model = RecordingModel()
            with patch.object(model, "predict", return_value=np.array([value])):
                with self.assertRaisesRegex(ValueError, "invalid or negative"):
                    recursive_forecast(model, hourly_load(), 1)

    def test_backtest_uses_no_observations_within_forecast_window(self) -> None:
        load = hourly_load(500)
        origin = load.index[240]
        baseline = recursive_backtest(RecordingModel(), load, origin, horizon=48, max_origins=1)
        modified = load.copy()
        modified.iloc[240:] += 10000
        changed = recursive_backtest(RecordingModel(), modified, origin, horizon=48, max_origins=1)
        assert_frame_equal(baseline.drop(columns="Actual (MW)"), changed.drop(columns="Actual (MW)"))
        np.testing.assert_array_equal(baseline["Previous day"], np.tile(load.iloc[216:240], 2))
        self.assertTrue((baseline["Persistence"] == load.iloc[239]).all())
        self.assertTrue((baseline.Origin < baseline.index).all())
        self.assertEqual(baseline["Lead time (hours)"].tolist(), list(range(1, 49)))

    def test_backtest_reports_missing_eligible_windows(self) -> None:
        load = hourly_load(200)
        with self.assertRaisesRegex(ValueError, "No complete"):
            recursive_backtest(RecordingModel(), load, load.index[-1])

    def test_metrics_have_known_units_and_reject_invalid_arrays(self) -> None:
        metrics = regression_metrics(np.array([0, 3]), np.array([0, 0]))
        self.assertEqual(metrics["MAE (MW)"], 1.5)
        self.assertAlmostEqual(metrics["RMSE (MW)"], np.sqrt(4.5))
        for actual, predicted in [([], []), ([1], [1, 2]), ([np.nan], [1]), ([[1]], [[1]])]:
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                regression_metrics(actual, predicted)

    def test_selection_and_refits_keep_test_data_out_of_evaluation_training(self) -> None:
        fitted = []

        class FakeRegressor:
            def __init__(self, max_depth: int, **parameters: object) -> None:
                self.depth = max_depth

            def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
                self.fit_index = features.index.copy()
                fitted.append(self)

            def predict(self, features: pd.DataFrame) -> np.ndarray:
                return np.full(len(features), self.depth, dtype=float)

        load = hourly_load(2000) * 0 + 6
        config = ForecastConfig(candidate_depths=(4, 6), recursive_horizon=4, max_origins=2)
        with patch("xgboost.XGBRegressor", FakeRegressor):
            result = run_experiment(load, {"source_sha256": "synthetic"}, config)
        self.assertEqual(len(fitted), 4)
        validation_start = pd.Timestamp(result.metadata["periods"]["validation"]["start"])
        test_start = pd.Timestamp(result.metadata["periods"]["test"]["start"])
        self.assertLess(fitted[0].fit_index[-1], validation_start)
        self.assertLess(fitted[1].fit_index[-1], validation_start)
        self.assertLess(fitted[2].fit_index[-1], test_start)
        self.assertEqual(fitted[3].fit_index[-1], load.index[-1])
        self.assertIs(result.model, fitted[3])
        self.assertEqual(result.metadata["selected_parameters"]["max_depth"], 6)
        self.assertEqual(result.test_metrics.loc["XGBoost", "MAE (MW)"], 0)
        self.assertEqual(len(result.recursive_predictions), 8)
        json.dumps(result.metadata, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
