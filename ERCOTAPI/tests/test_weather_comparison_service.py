"""Weather orchestration keeps publication prospective and failures explicit."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from ERCOTAPI import forecast_comparison_service as service
from ERCOTAPI.forecast_comparison import ForecastComparisonStore
from ERCOTAPI.weather_forecast import WeatherConfig, WeatherForecastStore
from ERCOTAPI.tests.test_weather_forecast import LOCATION, URL, payload, raw
from ERCOTAPI.tests.test_forecast_comparison_service import (
    NOW, actual_bundle, official_report, trainer_for,
)


class WeatherComparisonTests(unittest.TestCase):
    def test_real_archive_model_and_saved_comparison_work_together(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            weather_clock = NOW - timedelta(hours=96)
            archive_store = WeatherForecastStore(Path(folder) / "weather.sqlite", clock=lambda: weather_clock)
            for hours in range(0, 97, 12):
                weather_clock = NOW - timedelta(hours=96 - hours)
                data = payload(issued=weather_clock - timedelta(hours=1), start=weather_clock.replace(minute=0), temperature=68 + hours / 24)
                archive_store.append_snapshot(LOCATION, URL, raw(data), available_at=weather_clock)
            config = WeatherConfig("model", (LOCATION,), "test-app contact@example.test", archive_store.path)
            ledger = ForecastComparisonStore(Path(folder) / "comparisons.sqlite", clock=lambda: NOW)
            with patch.object(service, "utc_now", return_value=NOW), patch.object(service, "collect_weather_forecasts", return_value={"status": "collected"}):
                result = service.run_and_save_comparison(
                    official_report(), actual_bundle=actual_bundle(), store=ledger,
                    tuning_mode="manual", manual_params={"n_estimators": 3, "max_depth": 2},
                    weather_config=config,
                )
            meta = result["snapshot"]["model_metadata"]
            self.assertEqual(meta["method_version"], "causal-hourly-weather-v1")
            self.assertEqual(meta["weather"]["status"], "used")
            self.assertEqual(meta["weather_features"], ["temperature_c_dallas"])
            self.assertGreaterEqual(meta["weather_training_rows"], 24)
            self.assertEqual(result["snapshot"]["summary"]["matched_hours"], 24)
            self.assertEqual(result["evaluation"]["status"], "awaiting_actuals")
            self.assertEqual(ledger.list_snapshots()[0]["model_metadata"], meta)

    def test_training_and_recursive_prediction_use_distinct_frozen_origins(self) -> None:
        archive = Mock()
        archive.report = {"source": "test forecast vintages", "sha256": "f" * 64}
        archive.at_origins.side_effect = lambda targets, origins: pd.DataFrame(
            {"temperature_c_houston": [20.0] * len(targets)}, index=targets,
        )
        trainer = trainer_for()

        def predict(model, scaler, history, hours, *, future_weather):
            self.assertEqual(hours, 25)
            self.assertEqual(len(future_weather), hours)
            return pd.DataFrame({"predicted_load_mw": [1050.0] * hours}, index=future_weather.index)

        with patch.object(service, "utc_now", return_value=NOW), patch.object(service, "forecast_load", side_effect=predict):
            snapshot = service.build_current_comparison(
                official_report(), actual_bundle(), trainer=trainer, weather_archive=archive,
            )
        training, future = archive.at_origins.call_args_list
        pd.testing.assert_index_equal(training.args[1], training.args[0] - pd.Timedelta(hours=1))
        self.assertEqual(future.args[1], NOW)
        self.assertEqual(future.args[0][0], pd.Timestamp(NOW).floor("h"))
        self.assertIn("training_weather", trainer.call_args.kwargs)
        self.assertEqual(snapshot["model_metadata"]["weather"]["status"], "used")
        self.assertEqual(snapshot["summary"]["matched_hours"], 24)

    def test_weather_prediction_still_rejects_hour_boundary_crossing(self) -> None:
        archive = Mock(report={})
        archive.at_origins.side_effect = lambda targets, origins: pd.DataFrame(
            {"temperature_c_houston": 20.0}, index=targets,
        )
        before = NOW.replace(minute=59, second=59)
        after = before + timedelta(seconds=2)

        def predict(model, scaler, history, hours, *, future_weather):
            return pd.DataFrame({"predicted_load_mw": 1050.0}, index=future_weather.index)

        with patch.object(service, "utc_now", side_effect=[before, before, after]), patch.object(service, "forecast_load", side_effect=predict):
            with self.assertRaisesRegex(ValueError, "hourly boundary"):
                service.build_current_comparison(official_report(), actual_bundle(), trainer=trainer_for(), weather_archive=archive)

    def test_collection_failure_is_visible_and_model_mode_never_falls_back(self) -> None:
        ledger = Mock()
        ledger.append_snapshot.return_value = "snapshot"
        ledger.list_snapshots.return_value = []
        for mode in ("collect", "model"):
            with self.subTest(mode=mode):
                config = SimpleNamespace(mode=mode, store_path="unused", locations=())
                with patch.object(service, "WeatherForecastStore"), patch.object(service, "collect_weather_forecasts", side_effect=ValueError("source unavailable")), patch.object(service, "build_current_comparison", return_value={"saved": True}) as build:
                    if mode == "collect":
                        with self.assertLogs(service.__name__, "WARNING"):
                            service.run_and_save_comparison(official_report(), actual_bundle=actual_bundle(), store=ledger, weather_config=config)
                        self.assertIsNone(build.call_args.kwargs["weather_archive"])
                        self.assertEqual(build.call_args.kwargs["weather_status"]["status"], "collection_failed")
                    else:
                        with self.assertRaisesRegex(ValueError, "source unavailable"):
                            service.run_and_save_comparison(official_report(), actual_bundle=actual_bundle(), store=ledger, weather_config=config)
                        build.assert_not_called()

    def test_collection_only_does_not_add_weather_features(self) -> None:
        config = SimpleNamespace(mode="collect", store_path="unused", locations=())
        ledger = Mock()
        ledger.list_snapshots.return_value = []
        with patch.object(service, "WeatherForecastStore") as weather_store, patch.object(service, "collect_weather_forecasts", return_value={"locations": 4}), patch.object(service, "build_current_comparison", return_value={}) as build:
            service.run_and_save_comparison(official_report(), actual_bundle=actual_bundle(), store=ledger, weather_config=config)
        weather_store.return_value.load_archive.assert_not_called()
        self.assertIsNone(build.call_args.kwargs["weather_archive"])
        self.assertEqual(build.call_args.kwargs["weather_status"]["status"], "collecting")

    def test_disabled_weather_does_not_access_network_or_archive(self) -> None:
        ledger = Mock()
        ledger.list_snapshots.return_value = []
        with patch.object(service, "collect_weather_forecasts") as collect, patch.object(service, "WeatherForecastStore") as archive, patch.object(service, "build_current_comparison", return_value={}):
            service.run_and_save_comparison(official_report(), actual_bundle=actual_bundle(), store=ledger, weather_config=SimpleNamespace(mode="off"))
        collect.assert_not_called()
        archive.assert_not_called()


if __name__ == "__main__":
    unittest.main()
