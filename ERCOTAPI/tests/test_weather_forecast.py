"""Deterministic NWS capture, provenance, availability and DST regression cases."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests

from ERCOTAPI import weather_forecast as weather


NOW = datetime(2026, 11, 1, 4, 10, tzinfo=timezone.utc)
LOCATION = weather.WeatherLocation("dallas", 32.7767, -96.797)
URL = "https://api.weather.gov/gridpoints/FWD/80,106/forecast/hourly"


def payload(*, issued: datetime = NOW - timedelta(hours=1), start: datetime | None = None,
            temperature: float = 68, unit: str = "F") -> dict:
    """A small forecast with physically checkable temperatures across fall DST."""
    start = start or NOW.replace(minute=0)
    return {"properties": {"updateTime": issued.isoformat(), "generatedAt": issued.isoformat(),
                           "periods": [
                               {"startTime": (start + timedelta(hours=number)).isoformat(),
                                "endTime": (start + timedelta(hours=number + 1)).isoformat(),
                                "temperature": temperature, "temperatureUnit": unit}
                               for number in range(50)]}}


def raw(data: dict | None = None) -> bytes:
    return json.dumps(data or payload()).encode()


class WeatherForecastTests(unittest.TestCase):
    def test_default_mode_is_off_and_enabled_requires_identified_user_agent(self) -> None:
        config = weather.load_weather_config({"XDG_CACHE_HOME": "/tmp/weather-test-cache"})
        self.assertEqual(config.mode, "off")
        self.assertEqual(len(config.locations), 6)
        self.assertEqual(config.store_path, Path("/tmp/weather-test-cache/ercot-grid-dashboard/weather-forecasts.sqlite"))
        for env in ({"ERCOT_WEATHER_MODE": "auto"}, {"ERCOT_WEATHER_MODE": "model"},
                    {"ERCOT_WEATHER_MAX_AGE_HOURS": "nan"}, {"ERCOT_WEATHER_MAX_AGE_HOURS": "0"}):
            with self.subTest(env=env), self.assertRaises(weather.WeatherError):
                weather.load_weather_config(env)
        config = weather.load_weather_config({"ERCOT_WEATHER_MODE": "collect", "ERCOT_WEATHER_USER_AGENT": "test-app contact@example.test"})
        self.assertEqual(config.mode, "collect")

    def test_custom_locations_validate_unique_names_coordinates_and_precision(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "locations.json"
            for locations in ([], [{"id": "Austin", "latitude": 30, "longitude": -97}],
                              [{"id": "austin", "latitude": True, "longitude": -97}],
                              [{"id": "austin", "latitude": 30.12345, "longitude": -97}],
                              [{"id": "austin", "latitude": 99, "longitude": -97}],
                              [{"id": "austin", "latitude": 30, "longitude": -97}] * 2):
                path.write_text(json.dumps({"locations": locations}))
                with self.subTest(locations=locations), self.assertRaises(weather.WeatherError):
                    weather.load_weather_config({"ERCOT_WEATHER_LOCATIONS_FILE": str(path)})

    def test_future_only_capture_converts_units_and_preserves_provenance(self) -> None:
        parsed = weather.parse_hourly_forecast(raw(), available_at=NOW)
        self.assertEqual(len(parsed["records"]), 49)
        self.assertEqual(parsed["records"][0]["temperature_c"], 20)
        self.assertEqual(parsed["records"][0]["valid_time"], "2026-11-01T05:00:00+00:00")
        self.assertEqual(parsed["available_at"], NOW.isoformat())
        self.assertEqual(parsed["issued_at"], (NOW - timedelta(hours=1)).isoformat())
        data = payload(temperature=-5, unit="C")
        self.assertEqual(weather.parse_hourly_forecast(raw(data), available_at=NOW)["records"][0]["temperature_c"], -5)
        for item in data["properties"]["periods"]:
            item["temperature"] = {"value": 32, "unitCode": "wmoUnit:degF"}
        self.assertEqual(weather.parse_hourly_forecast(raw(data), available_at=NOW)["records"][0]["temperature_c"], 0)

    def test_invalid_period_units_values_gaps_duplicates_and_provenance_fail(self) -> None:
        mutations = [
            lambda p: p["properties"].update(updateTime=(NOW + timedelta(minutes=1)).isoformat()),
            lambda p: p["properties"].update(generatedAt=(NOW + timedelta(minutes=1)).isoformat()),
            lambda p: p["properties"].update(updateTime=(NOW - timedelta(hours=19)).isoformat()),
            lambda p: p["properties"]["periods"].pop(4),
            lambda p: p["properties"]["periods"].append(p["properties"]["periods"][3]),
            lambda p: p["properties"]["periods"][3].update(temperature=None),
            lambda p: p["properties"]["periods"][3].update(temperature=True),
            lambda p: p["properties"]["periods"][3].update(temperature=float("inf")),
            lambda p: p["properties"]["periods"][3].update(temperature=3000),
            lambda p: p["properties"]["periods"][3].update(temperatureUnit="K"),
            lambda p: p["properties"]["periods"][3].update(endTime=(NOW + timedelta(days=5)).isoformat()),
            lambda p: p["properties"]["periods"][3].update(startTime="2026-11-01T07:00:00"),
            lambda p: p["properties"].update(periods=p["properties"]["periods"][:20]),
        ]
        for change in mutations:
            data = payload()
            change(data)
            with self.subTest(change=change), self.assertRaises(weather.WeatherError):
                weather.parse_hourly_forecast(raw(data), available_at=NOW)

    def test_first_availability_is_persistent_and_database_is_insert_only(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weather.sqlite"
            store = weather.WeatherForecastStore(path, clock=lambda: NOW + timedelta(hours=1))
            identifier = store.append_snapshot(LOCATION, URL, raw(), available_at=NOW)
            self.assertEqual(store.append_snapshot(LOCATION, URL, raw(), available_at=NOW + timedelta(hours=1)), identifier)
            archive = weather.WeatherForecastStore(path).load_archive([LOCATION])
            self.assertEqual(archive.report["snapshot_count"], 1)
            self.assertEqual(archive.report["sources"][0]["available_at"], NOW.isoformat())
            with closing(sqlite3.connect(path)) as connection, connection:
                self.assertEqual(connection.execute("SELECT raw_payload FROM weather_snapshots").fetchone()[0], raw())
                for operation in ("DELETE FROM weather_snapshots", "UPDATE weather_snapshots SET available_at='2020-01-01'"):
                    with self.assertRaises(sqlite3.IntegrityError):
                        connection.execute(operation)

    def test_causal_join_excludes_later_capture_and_late_older_release(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = weather.WeatherForecastStore(Path(folder) / "weather.sqlite", clock=lambda: NOW + timedelta(hours=3))
            store.append_snapshot(LOCATION, URL, raw(payload(temperature=10, unit="C")), available_at=NOW)
            newer = payload(issued=NOW + timedelta(minutes=20), temperature=20, unit="C")
            store.append_snapshot(LOCATION, URL, raw(newer), available_at=NOW + timedelta(minutes=30))
            older = payload(issued=NOW - timedelta(hours=2), temperature=30, unit="C")
            store.append_snapshot(LOCATION, URL, raw(older), available_at=NOW + timedelta(minutes=40))
            archive = store.load_archive([LOCATION])
            targets = pd.date_range("2026-11-01T06:00:00Z", periods=3, freq="h")
            origins = pd.DatetimeIndex([NOW - timedelta(seconds=1), NOW, NOW + timedelta(minutes=45)])
            result = archive.at_origins(targets, origins)
            self.assertTrue(np.isnan(result.iloc[0, 0]))
            self.assertEqual(result.iloc[1, 0], 10)
            self.assertEqual(result.iloc[2, 0], 20)
            frozen = archive.at_origins(targets, NOW)
            self.assertTrue(frozen.iloc[:, 0].eq(10).all())

    def test_absent_points_changed_coordinates_and_stale_forecasts_stay_missing(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = weather.WeatherForecastStore(Path(folder) / "weather.sqlite", clock=lambda: NOW)
            store.append_snapshot(LOCATION, URL, raw(), available_at=NOW)
            absent = weather.WeatherLocation("austin", 30.2672, -97.7431)
            archive = store.load_archive([LOCATION, absent])
            targets = pd.date_range("2026-11-01T06:00:00Z", periods=2, freq="h")
            result = archive.at_origins(targets, NOW)
            self.assertTrue(result["temperature_c_austin"].isna().all())
            moved = weather.WeatherLocation("dallas", 32.7000, -96.7000)
            self.assertTrue(store.load_archive([moved]).at_origins(targets, NOW).isna().all().all())
            later = pd.date_range("2026-11-02T06:00:00Z", periods=2, freq="h")
            self.assertTrue(archive.at_origins(later, NOW + timedelta(hours=24)).isna().all().all())

    def test_utc_dst_targets_and_bridge_preserve_distinct_instants(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = weather.WeatherForecastStore(Path(folder) / "weather.sqlite", clock=lambda: NOW)
            store.append_snapshot(LOCATION, URL, raw(), available_at=NOW)
            archive = store.load_archive([LOCATION])
            targets = pd.date_range("2026-11-01T06:00:00Z", periods=2, freq="h").tz_convert("America/Chicago")
            result = archive.at_origins(targets, NOW + timedelta(hours=4))
            self.assertEqual(list(result.index.hour), [1, 1])
            self.assertNotEqual(result.index[0].utcoffset(), result.index[1].utcoffset())
            self.assertTrue(result.iloc[:, 0].eq(20).all())
            with self.assertRaises(weather.WeatherError):
                archive.at_origins(targets.tz_localize(None), NOW)
            with self.assertRaises(weather.WeatherError):
                archive.at_origins(targets, NOW.replace(tzinfo=None))

    def test_collector_uses_actual_retrieval_clock_and_fails_partial_collection(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            config = weather.WeatherConfig("collect", (LOCATION,), "ercot-test contact@example.test", Path(folder) / "weather.sqlite")
            store = weather.WeatherForecastStore(config.store_path, clock=lambda: NOW)
            point = json.dumps({"properties": {"forecastHourly": URL}}).encode()
            client = Mock()
            with patch.object(weather, "_fetch", side_effect=[point, raw()]) as fetch:
                status = weather.collect_weather_forecasts(config, store=store, session=client, clock=lambda: NOW)
            self.assertEqual(status["location_count"], 1)
            self.assertFalse(status["historical_backfill"])
            self.assertEqual(fetch.call_args_list[0].kwargs["user_agent"], config.user_agent)
            self.assertEqual(store.load_archive(config.locations).report["sources"][0]["available_at"], NOW.isoformat())
            with patch.object(weather, "_fetch", side_effect=weather.WeatherError("unavailable")):
                with self.assertRaisesRegex(weather.WeatherError, "unavailable"):
                    weather.collect_weather_forecasts(config, store=store, session=client)

    def test_network_source_validation_and_errors_never_leak_request_content(self) -> None:
        client = Mock()
        for url in ("https://evil.example/forecast", URL.replace("https:", "http:"),
                    URL + "?token=secret", URL.replace("api.weather.gov", "api.weather.gov.evil.example")):
            with self.subTest(url=url), self.assertRaises(weather.WeatherError):
                weather._fetch(client, url, user_agent="test", hourly=True)
        client.get.assert_not_called()
        client.get.side_effect = requests.ConnectionError("private request contents")
        with self.assertRaisesRegex(weather.WeatherError, "request failed") as failure:
            weather._fetch(client, URL, user_agent="test", hourly=True)
        self.assertNotIn("private", str(failure.exception))

    def test_requested_range_and_provenance_summary_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = weather.WeatherForecastStore(Path(folder) / "weather.sqlite", clock=lambda: NOW + timedelta(days=10))
            for delta in (timedelta(0), timedelta(minutes=30), timedelta(days=10)):
                source_time = NOW + delta
                data = payload(issued=source_time - timedelta(hours=1), start=source_time.replace(minute=0))
                store.append_snapshot(LOCATION, URL, raw(data), available_at=source_time)
            targets = pd.date_range("2026-11-01T06:00:00Z", periods=2, freq="h")
            archive = store.load_archive([LOCATION], target_start=targets[0], target_end=targets[-1])
            self.assertEqual(archive.report["snapshot_count"], 2)
            self.assertEqual(archive.report["record_count"], 4)
            self.assertEqual(len(archive.report["sources"]), 1)
            self.assertEqual(set(archive.records.valid_time), set(targets))
            with self.assertRaises(weather.WeatherError):
                store.load_archive([LOCATION], target_start=targets[-1], target_end=targets[0])

    def test_archive_read_rejects_derived_record_and_availability_corruption(self) -> None:
        for column in ("records_json", "available_at", "issued_at"):
            with self.subTest(column=column), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "weather.sqlite"
                store = weather.WeatherForecastStore(path, clock=lambda: NOW)
                store.append_snapshot(LOCATION, URL, raw(), available_at=NOW)
                with closing(sqlite3.connect(path)) as connection, connection:
                    connection.execute("DROP TRIGGER weather_no_update")
                    if column == "records_json":
                        records = json.loads(connection.execute("SELECT records_json FROM weather_snapshots").fetchone()[0])
                        records[0]["temperature_c"] = 999
                        value = json.dumps(records)
                    else:
                        value = (NOW - timedelta(minutes=1)).isoformat()
                    connection.execute(f"UPDATE weather_snapshots SET {column}=?", (value,))
                with self.assertRaises(weather.WeatherError):
                    store.load_archive([LOCATION])

    def test_config_direct_construction_rejects_invalid_modes_points_and_agent(self) -> None:
        valid = {"mode": "collect", "locations": (LOCATION,), "user_agent": "test-agent", "store_path": Path("/tmp/test-weather.sqlite")}
        for field, value in (("mode", "auto"), ("locations", ()), ("locations", (LOCATION, LOCATION)),
                             ("user_agent", "test\r\nInjected: header"), ("user_agent", ""),
                             ("max_age_hours", True), ("max_age_hours", float("nan"))):
            with self.subTest(field=field), self.assertRaises(weather.WeatherError):
                weather.WeatherConfig(**{**valid, field: value})


if __name__ == "__main__":
    unittest.main()
