"""Deterministic prospective forecast/actual matching and storage checks."""

from __future__ import annotations

from contextlib import closing
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import tempfile
import unittest

import pandas as pd

from ERCOTAPI.forecast_comparison import ComparisonError, ForecastComparisonStore, compare_forecasts


UTC = timezone.utc
ISSUED = datetime(2026, 11, 13, 12, tzinfo=UTC)
MODEL_METADATA = {"model_type": "TestModel", "method_version": "test-v1", "dataset_sha256": "a" * 64}
ACTUAL_METADATA = {"source_url": "https://www.ercot.com/actual", "source_sha256": "c" * 64}


def model_frame() -> pd.DataFrame:
    return pd.DataFrame({"predicted_load_mw": [110.0, 190.0]}, index=pd.date_range(ISSUED + timedelta(hours=1), periods=2, freq="h"))


def official_report() -> dict:
    return {
        "ok": True, "units": "MW", "forecast_kind": "official_ercot_np3_561_cd",
        "published_at": (ISSUED - timedelta(minutes=30)).isoformat(),
        "generated_at": (ISSUED - timedelta(minutes=10)).isoformat(),
        "source_url": "https://www.ercot.com/forecast", "source_sha256": "b" * 64, "report_name": "official.csv",
        "hourly_series": [{"interval_start_utc": start.isoformat(), "interval_end_utc": (start + timedelta(hours=1)).isoformat(), "load_mw": value} for start, value in zip(model_frame().index, (100.0, 200.0))],
    }


def snapshot(**overrides) -> dict:
    parameters = dict(model_issued_at=ISSUED, model_data_cutoff_utc=ISSUED, model_metadata=MODEL_METADATA, now=ISSUED + timedelta(minutes=5))
    parameters.update(overrides)
    return compare_forecasts(model_frame(), official_report(), **parameters)


class ForecastComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.clock = ISSUED + timedelta(minutes=6)
        self.path = Path(self.temp.name) / "evidence.sqlite3"
        self.store = ForecastComparisonStore(self.path, clock=lambda: self.clock)

    def actual_frame(self, values=(120.0, 180.0)) -> pd.DataFrame:
        return pd.DataFrame({"load": values}, index=model_frame().index)

    def test_same_intervals_are_aligned_and_difference_is_not_accuracy(self) -> None:
        local = model_frame().iloc[::-1].tz_convert("America/Chicago")
        original = local.copy(deep=True)
        result = compare_forecasts(local, official_report(), model_issued_at=ISSUED, model_data_cutoff_utc=ISSUED, model_metadata=MODEL_METADATA, now=self.clock)
        self.assertEqual([row["difference_mw"] for row in result["rows"]], [10, -10])
        self.assertEqual([row["model_lead_hours"] for row in result["rows"]], [1, 2])
        self.assertEqual([row["ercot_lead_hours"] for row in result["rows"]], [1.5, 2.5])
        self.assertEqual(result["summary"]["mean_difference_mw"], 0)
        self.assertEqual(result["summary"]["mean_absolute_difference_mw"], 10)
        self.assertIn("not accuracy", result["summary"]["interpretation"])
        self.assertNotIn("model_metrics", result)
        pd.testing.assert_frame_equal(local, original)

    def test_no_hindsight_forecast_issue_cutoff_or_vintage_is_admitted(self) -> None:
        for kwargs in (
            {"model_data_cutoff_utc": ISSUED + timedelta(seconds=1)},
            {"model_issued_at": ISSUED + timedelta(minutes=6)},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ComparisonError):
                snapshot(**kwargs)
        for field, timestamp in (("generated_at", ISSUED + timedelta(seconds=1)), ("published_at", ISSUED)):
            official = official_report()
            official[field] = timestamp.isoformat()
            with self.subTest(field=field), self.assertRaises(ComparisonError):
                compare_forecasts(model_frame(), official, model_issued_at=ISSUED, model_data_cutoff_utc=ISSUED, model_metadata=MODEL_METADATA, now=self.clock)

    def test_elapsed_targets_and_missing_matches_are_reported_without_filling(self) -> None:
        model = model_frame()
        model.loc[pd.Timestamp(ISSUED)] = 500
        official = official_report()
        official["hourly_series"] = official["hourly_series"][:1]
        result = compare_forecasts(model, official, model_issued_at=ISSUED, model_data_cutoff_utc=ISSUED, model_metadata=MODEL_METADATA, now=self.clock)
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(result["summary"]["excluded_elapsed_model_hours"], 1)
        self.assertEqual(result["summary"]["model_hours_without_ercot"], 1)

    def test_naive_duplicate_nonhourly_and_invalid_values_are_rejected(self) -> None:
        malformed = [model_frame().tz_localize(None), pd.concat([model_frame(), model_frame().iloc[:1]])]
        shifted = model_frame()
        shifted.index += pd.Timedelta(minutes=1)
        malformed.append(shifted)
        for value in (float("nan"), float("inf"), -1, True):
            frame = model_frame().astype(object)
            frame.iloc[0, 0] = value
            malformed.append(frame)
        for frame in malformed:
            with self.subTest(frame=str(frame.index)), self.assertRaises(ComparisonError):
                compare_forecasts(frame, official_report(), model_issued_at=ISSUED, model_data_cutoff_utc=ISSUED, model_metadata=MODEL_METADATA, now=self.clock)

    def test_official_duplicate_wrong_units_and_wrong_interval_end_are_rejected(self) -> None:
        variants = []
        duplicate = official_report()
        duplicate["hourly_series"].append(duplicate["hourly_series"][0])
        variants.append(duplicate)
        wrong_unit = official_report()
        wrong_unit["units"] = "GW"
        variants.append(wrong_unit)
        wrong_end = official_report()
        wrong_end["hourly_series"][0]["interval_end_utc"] = (ISSUED + timedelta(hours=3)).isoformat()
        variants.append(wrong_end)
        for official in variants:
            with self.assertRaises(ComparisonError):
                compare_forecasts(model_frame(), official, model_issued_at=ISSUED, model_data_cutoff_utc=ISSUED, model_metadata=MODEL_METADATA, now=self.clock)

    def test_dst_repeated_local_hour_remains_two_distinct_utc_intervals(self) -> None:
        index = pd.date_range("2026-11-01T00:00", periods=4, freq="h", tz="America/Chicago")
        issued = index[0].to_pydatetime() - timedelta(hours=1)
        local = pd.DataFrame({"predicted_load_mw": [1, 2, 3, 4]}, index=index)
        official = official_report()
        official.update(published_at=(issued - timedelta(minutes=30)).isoformat(), generated_at=(issued - timedelta(minutes=10)).isoformat())
        official["hourly_series"] = [{"interval_start_utc": start.isoformat(), "interval_end_utc": (start.tz_convert("UTC") + pd.Timedelta(hours=1)).isoformat(), "load_mw": value} for start, value in zip(index, (1, 2, 3, 4))]
        result = compare_forecasts(local, official, model_issued_at=issued, model_data_cutoff_utc=issued, model_metadata=MODEL_METADATA, now=issued + timedelta(minutes=5))
        self.assertEqual(len(result["rows"]), 4)
        self.assertEqual([row["difference_mw"] for row in result["rows"]], [0, 0, 0, 0])
        self.assertEqual(result["rows"][1]["interval_start_utc"], "2026-11-01T06:00:00+00:00")
        self.assertEqual(result["rows"][2]["interval_start_utc"], "2026-11-01T07:00:00+00:00")

    def test_snapshot_is_idempotent_persistent_and_immutable(self) -> None:
        evidence = snapshot()
        identifier = self.store.append_snapshot(evidence)
        self.assertEqual(self.store.append_snapshot(deepcopy(evidence)), identifier)
        reopened = ForecastComparisonStore(self.path, clock=lambda: self.clock)
        self.assertEqual(len(reopened.list_snapshots()), 1)
        self.assertEqual(reopened.list_snapshots()[0]["rows"], evidence["rows"])
        with closing(sqlite3.connect(self.path)) as connection:
            for statement in ("UPDATE snapshots SET recorded_at='2000-01-01'", "DELETE FROM snapshots"):
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(statement)

    def test_snapshot_cannot_be_backfilled_after_target_or_modify_derived_fields(self) -> None:
        self.clock = ISSUED + timedelta(hours=1)
        with self.assertRaisesRegex(ComparisonError, "strictly before"):
            self.store.append_snapshot(snapshot())
        self.assertEqual(self.store.list_snapshots(), [])
        self.clock = ISSUED + timedelta(minutes=6)
        for modify in (lambda value: value["rows"][0].update(difference_mw=999), lambda value: value["summary"].update(mean_difference_mw=999)):
            evidence = snapshot()
            modify(evidence)
            with self.assertRaises(ComparisonError):
                self.store.append_snapshot(evidence)

    def test_metrics_use_identical_matured_hours_and_known_mw_errors(self) -> None:
        identifier = self.store.append_snapshot(snapshot())
        empty = self.store.evaluate_snapshot(identifier)
        self.assertEqual(empty["pending_count"], 2)
        self.assertIsNone(empty["model_metrics"])
        self.clock = ISSUED + timedelta(hours=2, minutes=5)
        first_actual = self.actual_frame().iloc[:1]
        self.store.append_actuals(first_actual, observed_at=self.clock, source_metadata=ACTUAL_METADATA)
        partial = self.store.evaluate_snapshot(identifier)
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(partial["matched_count"], 1)
        self.assertEqual(partial["model_metrics"]["bias_mw"], -10)
        self.assertEqual(partial["ercot_metrics"]["bias_mw"], -20)
        self.clock = ISSUED + timedelta(hours=3, minutes=5)
        self.assertEqual(self.store.append_actuals(self.actual_frame(), observed_at=self.clock, source_metadata=ACTUAL_METADATA), 1)
        complete = self.store.evaluate_snapshot(identifier)
        self.assertEqual(complete["status"], "complete")
        self.assertEqual(complete["matched_count"], 2)
        self.assertEqual(complete["model_metrics"], {"sample_count": 2, "mae_mw": 10, "rmse_mw": 10, "bias_mw": 0})
        self.assertEqual(complete["ercot_metrics"], {"sample_count": 2, "mae_mw": 20, "rmse_mw": 20, "bias_mw": 0})
        self.assertEqual([row["actual_mw"] for row in complete["rows"]], [120, 180])

    def test_actuals_cannot_be_future_or_retroactively_known(self) -> None:
        identifier = self.store.append_snapshot(snapshot())
        with self.assertRaises(ComparisonError):
            self.store.append_actuals(self.actual_frame(), observed_at=self.clock, source_metadata=ACTUAL_METADATA)
        self.clock = ISSUED + timedelta(hours=3, minutes=5)
        with self.assertRaises(ComparisonError):
            self.store.append_actuals(self.actual_frame(), observed_at=self.clock + timedelta(seconds=1), source_metadata=ACTUAL_METADATA)
        self.store.append_actuals(self.actual_frame(), observed_at=self.clock - timedelta(minutes=1), source_metadata=ACTUAL_METADATA)
        earlier = self.store.evaluate_snapshot(identifier, as_of=self.clock - timedelta(minutes=2))
        self.assertEqual(earlier["missing_actual_count"], 2)
        self.assertEqual(earlier["matched_count"], 0)
        with self.assertRaises(ComparisonError):
            self.store.evaluate_snapshot(identifier, as_of=self.clock + timedelta(seconds=1))

    def test_unchanged_bundle_refresh_is_idempotent_and_revisions_do_not_rewrite_scores(self) -> None:
        identifier = self.store.append_snapshot(snapshot())
        self.clock = ISSUED + timedelta(hours=3, minutes=5)
        self.assertEqual(self.store.append_actuals(self.actual_frame(), observed_at=self.clock, source_metadata=ACTUAL_METADATA), 2)
        before = self.store.evaluate_snapshot(identifier)
        changed_bundle = {**ACTUAL_METADATA, "source_sha256": "d" * 64}
        self.clock += timedelta(minutes=5)
        self.assertEqual(self.store.append_actuals(self.actual_frame(), observed_at=self.clock, source_metadata=changed_bundle), 0)
        self.assertEqual(self.store.append_actuals(self.actual_frame((121, 180)), observed_at=self.clock, source_metadata=changed_bundle), 1)
        after = self.store.evaluate_snapshot(identifier)
        self.assertEqual(before["model_metrics"], after["model_metrics"])
        self.assertEqual(before["ercot_metrics"], after["ercot_metrics"])
        self.assertEqual(after["rows"][0]["actual_source"]["source_sha256"], "c" * 64)
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM actuals").fetchone()[0], 3)
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute("DELETE FROM actuals")

    def test_same_source_hash_with_changed_values_is_rejected_atomically(self) -> None:
        self.clock = ISSUED + timedelta(hours=3, minutes=5)
        self.store.append_actuals(self.actual_frame(), observed_at=self.clock, source_metadata=ACTUAL_METADATA)
        with self.assertRaises(ComparisonError):
            self.store.append_actuals(self.actual_frame((999, 180)), observed_at=self.clock, source_metadata=ACTUAL_METADATA)


if __name__ == "__main__":
    unittest.main()
