"""Integration regressions for causal future windows and graceful delivery."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from ERCOTAPI import forecast_comparison_service as service
from ERCOTAPI.forecast_comparison import ForecastComparisonStore


UTC = timezone.utc
NOW = datetime(2026, 11, 14, 17, 30, tzinfo=UTC)


class IdentityScaler:
    def transform(self, values: pd.DataFrame) -> pd.DataFrame:
        return values


class IncrementModel:
    """Use each previous prediction, proving that bridge hours are recursive."""

    def __init__(self) -> None:
        self.rows: list[pd.DataFrame] = []

    def predict(self, values: pd.DataFrame):
        self.rows.append(values.copy(deep=True))
        return values["load_lag_1h"].to_numpy() + 1


def actual_bundle(cutoff: datetime | None = None, *, observed: datetime = NOW, count: int = 72) -> dict:
    cutoff = cutoff or NOW.replace(minute=0)
    index = pd.date_range(end=pd.Timestamp(cutoff) - pd.Timedelta(hours=1), periods=count, freq="h")
    frame = pd.DataFrame({"load": [1000.0 + index for index in range(count)]}, index=index)
    return {"history": frame, "metadata": {
        "units": "MW", "latest_interval_end_utc": cutoff.isoformat(),
        "retrieved_at": observed.isoformat(), "source_url": "https://www.ercot.com/actual",
        "source_sha256": "b" * 64, "warnings": [], "archive_count": 3,
    }}


def official_report(now: datetime = NOW) -> dict:
    index = pd.date_range(start=pd.Timestamp(now).floor("h") - pd.Timedelta(hours=12), periods=168, freq="h")
    return {
        "ok": True, "forecast_kind": "official_ercot_np3_561_cd", "units": "MW",
        "published_at": (now - timedelta(minutes=30)).isoformat(),
        "generated_at": (now - timedelta(minutes=15)).isoformat(),
        "source_url": "https://www.ercot.com/forecast", "source_sha256": "c" * 64,
        "report_name": "official.csv", "text": "Official ERCOT forecast remains available.",
        "chart": {"base64": "test-chart", "mime_type": "image/png"},
        "hourly_series": [{"interval_start_utc": start.isoformat(), "interval_end_utc": (start + pd.Timedelta(hours=1)).isoformat(), "load_mw": 1050.0} for start in index],
    }


def trainer_for(model: IncrementModel | None = None) -> Mock:
    metrics = {"model_type": "IncrementModel", "method_version": "test-causal-v1", "dataset_sha256": "a" * 64, "test_mae": 1.5}
    return Mock(return_value=(model or IncrementModel(), IdentityScaler(), None, None, None, metrics))


class ForecastComparisonServiceTests(unittest.TestCase):
    def test_real_recursive_bridge_is_excluded_from_24_future_targets(self) -> None:
        bundle, official, model = actual_bundle(), official_report(), IncrementModel()
        original_history, original_official = bundle["history"].copy(deep=True), deepcopy(official)
        clock = [NOW, NOW + timedelta(minutes=1), NOW + timedelta(minutes=1, seconds=1), NOW + timedelta(minutes=1, seconds=2)]
        trainer = trainer_for(model)
        with patch.object(service, "utc_now", side_effect=clock):
            result = service.build_current_comparison(official, bundle, tuning_mode="manual", manual_params={"n_estimators": 3}, trainer=trainer)
        self.assertEqual(len(model.rows), 25)
        self.assertEqual(model.rows[0]["load_lag_1h"].iloc[0], 1071)
        self.assertEqual(model.rows[1]["load_lag_1h"].iloc[0], 1072)
        self.assertEqual(result["summary"]["matched_hours"], 24)
        self.assertEqual(result["rows"][0]["interval_start_utc"], "2026-11-14T18:00:00+00:00")
        self.assertEqual(result["rows"][-1]["interval_end_utc"], "2026-11-15T18:00:00+00:00")
        self.assertEqual(result["rows"][0]["model_mw"], 1073)
        self.assertEqual(result["rows"][0]["difference_mw"], 23)
        self.assertEqual(result["model_issued_at"], clock[2].isoformat())
        meta = result["model_metadata"]
        self.assertEqual((meta["recursive_steps"], meta["unobserved_bridge_hours"], meta["history_hours"]), (25, 1, 72))
        self.assertEqual(meta["calendar_timezone"], "America/Chicago")
        self.assertEqual(meta["source_metadata"], bundle["metadata"])
        self.assertEqual(meta["test_mae"], 1.5)
        self.assertEqual(str(trainer.call_args.args[0].index.tz), "America/Chicago")
        self.assertEqual(trainer.call_args.kwargs, {"tuning_mode": "manual", "manual_params": {"n_estimators": 3}})
        pd.testing.assert_frame_equal(bundle["history"], original_history)
        self.assertEqual(official, original_official)
        bundle["metadata"]["warnings"].append("changed after issue")
        self.assertEqual(meta["source_metadata"]["warnings"], [])

    def test_exact_hour_is_skipped_so_all_24_targets_remain_prospective(self) -> None:
        now = NOW.replace(hour=18, minute=0)
        with patch.object(service, "utc_now", return_value=now):
            result = service.build_current_comparison(official_report(now), actual_bundle(observed=now), trainer=trainer_for())
        self.assertEqual(result["summary"]["matched_hours"], 24)
        self.assertEqual(result["rows"][0]["interval_start_utc"], "2026-11-14T19:00:00+00:00")
        self.assertEqual(result["model_metadata"]["unobserved_bridge_hours"], 2)

    def test_hour_boundary_during_prediction_or_capture_requires_retry(self) -> None:
        before = NOW.replace(minute=59, second=59)
        after = before + timedelta(seconds=2)
        for clock in ([before, before, after], [before, before, before, after]):
            with self.subTest(clock=clock), patch.object(service, "utc_now", side_effect=clock):
                with self.assertRaisesRegex(ValueError, "future hourly intervals|hourly boundary"):
                    service.build_current_comparison(official_report(), actual_bundle(), trainer=trainer_for())

    def test_small_observation_delay_is_predicted_not_filled(self) -> None:
        bundle = actual_bundle(NOW.replace(hour=15, minute=0))
        model = IncrementModel()
        with patch.object(service, "utc_now", return_value=NOW):
            result = service.build_current_comparison(official_report(), bundle, trainer=trainer_for(model))
        self.assertEqual(len(model.rows), 27)
        self.assertEqual(result["model_metadata"]["unobserved_bridge_hours"], 3)
        self.assertEqual(result["rows"][0]["model_mw"], 1075)
        self.assertEqual(result["summary"]["matched_hours"], 24)

    def test_no_missing_duplicate_naive_unfinished_or_invalid_actuals_are_trained(self) -> None:
        source = actual_bundle()
        broken = []
        for mutation in (
            lambda frame: frame.drop(frame.index[10]),
            lambda frame: pd.concat([frame, frame.iloc[:1]]),
            lambda frame: frame.tz_localize(None),
            lambda frame: frame.iloc[-47:],
            lambda frame: frame.set_axis(frame.index + pd.Timedelta(minutes=15)),
            lambda frame: frame.assign(load=float("nan")),
            lambda frame: frame.assign(load=True),
        ):
            bundle = deepcopy(source)
            bundle["history"] = mutation(bundle["history"])
            broken.append(bundle)
        unfinished = actual_bundle(NOW.replace(hour=18, minute=0))
        wrong_units = deepcopy(source); wrong_units["metadata"]["units"] = "GW"
        wrong_cutoff = deepcopy(source); wrong_cutoff["metadata"]["latest_interval_end_utc"] = "2026-11-14T16:00:00+00:00"
        naive_cutoff = deepcopy(source); naive_cutoff["metadata"]["latest_interval_end_utc"] = "2026-11-14T17:00:00"
        future_observed = deepcopy(source); future_observed["metadata"]["retrieved_at"] = (NOW + timedelta(hours=1)).isoformat()
        broken.extend([unfinished, wrong_units, wrong_cutoff, naive_cutoff, future_observed, {}])
        for bundle in broken:
            trainer = trainer_for()
            with self.subTest(bundle=str(bundle.get("metadata"))), patch.object(service, "utc_now", return_value=NOW):
                with self.assertRaises(ValueError):
                    service.build_current_comparison(official_report(), bundle, trainer=trainer)
            trainer.assert_not_called()

    def test_stale_actual_and_official_vintages_fail_before_training(self) -> None:
        stale_actual = actual_bundle(NOW.replace(hour=13, minute=0))
        stale_official = official_report(); stale_official["published_at"] = (NOW - timedelta(hours=3, seconds=1)).isoformat()
        future_official = official_report(); future_official["generated_at"] = (NOW + timedelta(minutes=1)).isoformat()
        bad_official = official_report(); bad_official.pop("published_at")
        for official, bundle in ((official_report(), stale_actual), (stale_official, actual_bundle()), (future_official, actual_bundle()), (bad_official, actual_bundle())):
            trainer = trainer_for()
            with self.subTest(official=official.get("published_at")), patch.object(service, "utc_now", return_value=NOW):
                with self.assertRaises(ValueError):
                    service.build_current_comparison(official, bundle, trainer=trainer)
            trainer.assert_not_called()

    def test_freshness_is_rechecked_after_expensive_training(self) -> None:
        official = official_report(); official["published_at"] = (NOW - timedelta(hours=2, minutes=59)).isoformat()
        trainer = trainer_for()
        with patch.object(service, "utc_now", side_effect=[NOW, NOW + timedelta(minutes=2)]):
            with self.assertRaisesRegex(ValueError, "no longer fresh"):
                service.build_current_comparison(official, actual_bundle(), trainer=trainer)
        trainer.assert_called_once()

    def test_missing_official_target_is_not_filled_or_called_24_hours(self) -> None:
        official = official_report()
        official["hourly_series"] = [row for row in official["hourly_series"] if row["interval_start_utc"] != "2026-11-14T19:00:00+00:00"]
        with patch.object(service, "utc_now", return_value=NOW):
            with self.assertRaisesRegex(ValueError, "all 24 future"):
                service.build_current_comparison(official, actual_bundle(), trainer=trainer_for())

    def test_fall_clock_change_has_24_elapsed_future_hours_and_two_local_ones(self) -> None:
        now = datetime(2026, 11, 1, 5, 30, tzinfo=UTC)
        bundle = actual_bundle(now.replace(minute=0), observed=now)
        with patch.object(service, "utc_now", return_value=now):
            result = service.build_current_comparison(official_report(now), bundle, trainer=trainer_for())
        starts = pd.DatetimeIndex([row["interval_start_utc"] for row in result["rows"]]).tz_convert("America/Chicago")
        self.assertEqual(len(starts), 24)
        self.assertEqual(starts[0].isoformat(), "2026-11-01T01:00:00-05:00")
        self.assertEqual(starts[1].isoformat(), "2026-11-01T01:00:00-06:00")
        self.assertTrue((starts.to_series().diff().iloc[1:] == pd.Timedelta(hours=1)).all())

    def test_saved_runs_are_pending_then_scored_on_same_later_actual_hour(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            clock = NOW
            store = ForecastComparisonStore(Path(folder) / "comparisons.sqlite", clock=lambda: clock)
            with patch.object(service, "utc_now", return_value=clock):
                first = service.run_and_save_comparison(official_report(clock), actual_bundle=actual_bundle(), trainer=trainer_for(), store=store)
            self.assertEqual(first["evaluation"]["status"], "awaiting_actuals")
            self.assertEqual(first["evaluation"]["pending_count"], 24)
            self.assertIsNone(first["evaluation"]["model_metrics"])
            clock = NOW.replace(hour=19, minute=10)
            later = actual_bundle()
            later["history"] = pd.concat([later["history"], pd.DataFrame({"load": [1055.0, 1060.0]}, index=pd.date_range("2026-11-14T17:00:00Z", periods=2, freq="h"))])
            later["metadata"].update(latest_interval_end_utc="2026-11-14T19:00:00+00:00", retrieved_at=clock.isoformat(), source_sha256="d" * 64)
            with patch.object(service, "utc_now", return_value=clock):
                second = service.run_and_save_comparison(official_report(clock), actual_bundle=later, trainer=trainer_for(), store=store)
            old = next(item for item in second["recent_evaluations"] if item["snapshot_id"] == first["snapshot_id"])
            self.assertEqual(old["matched_count"], 1)
            self.assertEqual(old["model_metrics"]["mae_mw"], 13)
            self.assertEqual(old["ercot_metrics"]["mae_mw"], 10)
            self.assertEqual(old["rows"][0]["actual_mw"], 1060)
            self.assertEqual(old["status"], "partial")
            self.assertEqual(len(store.list_snapshots()), 2)

    def test_store_location_is_configurable_outside_repository(self) -> None:
        with patch.dict("os.environ", {"XDG_CACHE_HOME": "/tmp/test-ercot-cache"}, clear=True):
            self.assertEqual(service.comparison_store_path(), Path("/tmp/test-ercot-cache/ercot-grid-dashboard/forecast-comparisons.sqlite"))
        with patch.dict("os.environ", {"ERCOT_COMPARISON_STORE": "/tmp/chosen-ercot-ledger.sqlite"}, clear=True):
            self.assertEqual(service.comparison_store_path(), Path("/tmp/chosen-ercot-ledger.sqlite"))


if __name__ == "__main__":
    unittest.main()
