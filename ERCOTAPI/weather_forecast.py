"""Capture NWS forecast vintages and select temperature available at an origin.

This is a prospective archive, not a historical-weather backfill. First actual
retrieval establishes availability. All joins use UTC instants, preserving DST.
City predictors are separate; no regional load weights are assumed.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests


NWS_ROOT = "https://api.weather.gov"
MAX_RESPONSE_BYTES = 2_000_000


class WeatherError(ValueError):
    """A validated weather source/configuration failure with safe error text."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _instant(value: Any, label: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is None:
            raise WeatherError("missing timezone")
        return stamp.tz_convert("UTC")
    except (ValueError, TypeError, OverflowError) as exc:
        raise WeatherError(f"Weather {label} requires an explicit, valid timezone.") from exc


@dataclass(frozen=True)
class WeatherLocation:
    """A named point predictor, not an inferred ERCOT load-zone weight."""

    id: str
    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", self.id):
            raise WeatherError("Weather location IDs must use lowercase letters, digits and underscores.")
        for value, lower, upper in ((self.latitude, -90, 90), (self.longitude, -180, 180)):
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not lower <= value <= upper:
                raise WeatherError("Weather coordinates must be finite latitude/longitude degrees.")
            if round(value, 4) != value:
                raise WeatherError("NWS coordinates support at most four decimal places.")
        object.__setattr__(self, "latitude", float(self.latitude))
        object.__setattr__(self, "longitude", float(self.longitude))


@dataclass(frozen=True)
class WeatherConfig:
    """Explicit mode and non-secret source/archive configuration."""

    mode: str
    locations: tuple[WeatherLocation, ...]
    user_agent: str
    store_path: Path
    max_age_hours: float = 18.0

    def __post_init__(self) -> None:
        if self.mode not in {"off", "collect", "model"}:
            raise WeatherError("ERCOT_WEATHER_MODE must be off, collect, or model.")
        if (not isinstance(self.user_agent, str) or "\r" in self.user_agent or "\n" in self.user_agent
                or (self.mode != "off" and not self.user_agent.strip())):
            raise WeatherError("Enabled weather requires a valid identifying User-Agent.")
        if (not isinstance(self.locations, (tuple, list)) or not 1 <= len(self.locations) <= 20
                or not all(isinstance(item, WeatherLocation) for item in self.locations)
                or len({item.id for item in self.locations}) != len(self.locations)):
            raise WeatherError("Weather configuration requires 1 to 20 distinct validated locations.")
        if (isinstance(self.max_age_hours, bool) or not isinstance(self.max_age_hours, (int, float))
                or not math.isfinite(self.max_age_hours) or not 0 < self.max_age_hours <= 48):
            raise WeatherError("Weather maximum forecast age must be positive and at most 48 hours.")
        try:
            object.__setattr__(self, "store_path", Path(self.store_path).expanduser())
        except TypeError as exc:
            raise WeatherError("Weather archive requires a valid filesystem path.") from exc
        object.__setattr__(self, "locations", tuple(self.locations))


def load_weather_config(environ: Mapping[str, str] | None = None) -> WeatherConfig:
    """Read optional weather mode; enabling collection/model requires a User-Agent."""
    env = os.environ if environ is None else environ
    mode = env.get("ERCOT_WEATHER_MODE", "off").strip().lower()
    if mode not in {"off", "collect", "model"}:
        raise WeatherError("ERCOT_WEATHER_MODE must be off, collect, or model.")
    agent = env.get("ERCOT_WEATHER_USER_AGENT", "").strip()
    if mode != "off" and (not agent or "\n" in agent or "\r" in agent):
        raise WeatherError("Enabled weather requires ERCOT_WEATHER_USER_AGENT identifying the application/contact.")
    filename = Path(env.get("ERCOT_WEATHER_LOCATIONS_FILE", str(Path(__file__).with_name("weather_locations.json")))).expanduser()
    try:
        payload = json.loads(filename.read_text(encoding="utf-8"))
        rows = payload["locations"]
        if not isinstance(rows, list) or not 1 <= len(rows) <= 20:
            raise WeatherError("Specify between 1 and 20 weather locations.")
        locations = tuple(WeatherLocation(**row) for row in rows)
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise WeatherError("Weather locations file must contain a readable locations list with id, latitude and longitude.") from exc
    if len({location.id for location in locations}) != len(locations):
        raise WeatherError("Weather location IDs must be unique.")
    try:
        age = float(env.get("ERCOT_WEATHER_MAX_AGE_HOURS", "18"))
    except ValueError as exc:
        raise WeatherError("Weather maximum forecast age must be positive hours.") from exc
    if not math.isfinite(age) or not 0 < age <= 48:
        raise WeatherError("Weather maximum forecast age must be positive and at most 48 hours.")
    root = Path(env.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    store = Path(env.get("ERCOT_WEATHER_STORE", str(root / "ercot-grid-dashboard" / "weather-forecasts.sqlite"))).expanduser()
    return WeatherConfig(mode, locations, agent, store, age)


def _nws_url(url: Any, *, hourly: bool) -> str:
    if not isinstance(url, str):
        raise WeatherError("NWS returned a missing forecast URL.")
    parsed = urlparse(url)
    pattern = r"/gridpoints/[A-Z]{3}/-?\d+,-?\d+/forecast/hourly" if hourly else r"/points/-?\d+(?:\.\d+)?,-?\d+(?:\.\d+)?"
    if (parsed.scheme != "https" or parsed.netloc != "api.weather.gov" or parsed.query
            or parsed.fragment or not re.fullmatch(pattern, parsed.path)):
        raise WeatherError("Weather sources must use official HTTPS NWS point/hourly forecast URLs.")
    return url


def _read_json(raw: bytes) -> dict[str, Any]:
    if not isinstance(raw, bytes) or len(raw) > MAX_RESPONSE_BYTES:
        raise WeatherError("NWS response must be bytes within the supported size limit.")
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise WeatherError("NWS returned invalid JSON.") from exc
    if not isinstance(data, dict):
        raise WeatherError("NWS returned an invalid forecast object.")
    return data


def parse_hourly_forecast(raw: bytes, *, available_at: datetime, max_age_hours: float = 18) -> dict[str, Any]:
    """Validate future hourly forecasts, preserving provider and acquisition times."""
    if not math.isfinite(max_age_hours) or not 0 < max_age_hours <= 48:
        raise WeatherError("Weather maximum forecast age must be positive and at most 48 hours.")
    available = _instant(available_at, "availability time")
    payload = _read_json(raw)
    try:
        properties = payload["properties"]
        issued = _instant(properties["updateTime"], "updateTime")
        generated = _instant(properties["generatedAt"], "generatedAt")
        periods = properties["periods"]
    except (KeyError, TypeError) as exc:
        raise WeatherError("NWS hourly forecast is missing provenance or periods.") from exc
    if not issued <= generated <= available:
        raise WeatherError("NWS forecast update/generation times cannot follow actual retrieval.")
    if (available - issued).total_seconds() > max_age_hours * 3600:
        raise WeatherError("NWS weather forecast is stale; no previous source is substituted.")
    if not isinstance(periods, list) or not periods:
        raise WeatherError("NWS returned no hourly forecast periods.")
    records, seen = [], set()
    for period in periods:
        if not isinstance(period, dict):
            raise WeatherError("NWS hourly forecast periods must be objects.")
        try:
            start = _instant(period["startTime"], "interval start")
            end = _instant(period["endTime"], "interval end")
        except (KeyError, TypeError) as exc:
            raise WeatherError("NWS hourly period is missing interval boundaries.") from exc
        if start != start.floor("h") or end - start != pd.Timedelta(hours=1) or start in seen:
            raise WeatherError("NWS hourly intervals must be unique, aligned, and exactly one elapsed hour.")
        seen.add(start)
        # Current/elapsed periods cannot establish a prospective training vintage.
        if start <= available:
            continue
        value = period.get("temperature")
        unit = period.get("temperatureUnit")
        if isinstance(value, dict):
            unit = {"wmoUnit:degC": "C", "wmoUnit:degF": "F"}.get(value.get("unitCode"))
            value = value.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or unit not in {"C", "F"}:
            raise WeatherError("NWS forecast requires finite temperatures with supported C/F units.")
        temperature = (float(value) - 32) * 5 / 9 if unit == "F" else float(value)
        if not -90 <= temperature <= 65:
            raise WeatherError("NWS temperature is outside the supported physical range (-90 to 65 C).")
        records.append({"valid_time": start.isoformat(), "temperature_c": temperature})
    records.sort(key=lambda row: row["valid_time"])
    if len(records) < 24:
        raise WeatherError("NWS forecast lacks at least 24 future hourly temperatures.")
    times = pd.DatetimeIndex([row["valid_time"] for row in records])
    if not (times.to_series().diff().iloc[1:] == pd.Timedelta(hours=1)).all():
        raise WeatherError("NWS forecast contains missing hourly periods; temperatures are not filled.")
    return {"issued_at": issued.isoformat(), "generated_at": generated.isoformat(),
            "available_at": available.isoformat(), "records": records,
            "source_sha256": hashlib.sha256(raw).hexdigest()}


def _snapshot_id(location: WeatherLocation, source_url: str, captured: Mapping[str, Any], recorded: str) -> str:
    identity = [location.id, location.latitude, location.longitude, source_url,
                captured["source_sha256"], captured["issued_at"], captured["generated_at"],
                captured["available_at"], recorded]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


@dataclass(frozen=True)
class WeatherArchive:
    """Immutable captured vintage data and causal temperature-selection policy."""

    records: pd.DataFrame
    locations: tuple[WeatherLocation, ...]
    report: dict[str, Any]
    max_age_hours: float = 18.0

    def at_origins(self, targets: pd.DatetimeIndex, origins: Any) -> pd.DataFrame:
        """Select each latest-issued available point forecast; gaps remain NaN.

        Bridge targets may precede the model issue time. Those targets use only
        vintages captured before the target began, never newly retrieved weather.
        """
        if not isinstance(targets, pd.DatetimeIndex) or targets.tz is None or targets.hasnans or targets.has_duplicates:
            raise WeatherError("Weather target times must be unique timezone-aware instants.")
        if not targets.tz_convert("UTC").equals(targets.tz_convert("UTC").floor("h")):
            raise WeatherError("Weather target times must identify hourly interval starts.")
        if isinstance(origins, (datetime, str, pd.Timestamp)):
            origin_values = [_instant(origins, "origin")] * len(targets)
        else:
            try:
                count = len(origins)
            except TypeError as exc:
                raise WeatherError("Weather origins must be aware timestamps or a matching sequence.") from exc
            if count != len(targets):
                raise WeatherError("Each weather target requires one origin.")
            origin_values = [_instant(value, "origin") for value in origins]
        columns = [f"temperature_c_{location.id}" for location in self.locations]
        result = pd.DataFrame(np.nan, index=targets, columns=columns)
        if self.records.empty:
            return result
        for number, (target, origin) in enumerate(zip(targets.tz_convert("UTC"), origin_values)):
            oldest_issue = min(origin, target) - pd.Timedelta(hours=self.max_age_hours)
            eligible = self.records.loc[(self.records.valid_time == target)
                                        & (self.records.issued_at >= oldest_issue)
                                        & (self.records.available_at <= origin)
                                        & (self.records.available_at < target)
                                        & (self.records.issued_at < target)]
            latest = eligible.sort_values(["issued_at", "generated_at", "available_at"]).drop_duplicates("location_id", keep="last")
            for record in latest.itertuples():
                result.iloc[number, columns.index(f"temperature_c_{record.location_id}")] = record.temperature_c
        return result


class WeatherForecastStore:
    """Append-only forecast evidence; repeated identical responses retain first availability."""

    def __init__(self, path: str | Path, *, clock: Callable[[], datetime] = _now) -> None:
        self.path, self.clock = Path(path), clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS weather_snapshots (
                id TEXT PRIMARY KEY, location_id TEXT NOT NULL, latitude REAL NOT NULL,
                longitude REAL NOT NULL, source_url TEXT NOT NULL, source_sha256 TEXT NOT NULL,
                issued_at TEXT NOT NULL, generated_at TEXT NOT NULL, available_at TEXT NOT NULL,
                recorded_at TEXT NOT NULL, raw_payload BLOB NOT NULL, records_json TEXT NOT NULL,
                valid_start TEXT NOT NULL, valid_end TEXT NOT NULL,
                UNIQUE(location_id, latitude, longitude, source_url, source_sha256))""")
            connection.execute("CREATE INDEX IF NOT EXISTS weather_target_idx ON weather_snapshots (location_id, latitude, longitude, valid_start, valid_end)")
            for operation in ("UPDATE", "DELETE"):
                connection.execute(f"CREATE TRIGGER IF NOT EXISTS weather_no_{operation.lower()} BEFORE {operation} ON weather_snapshots BEGIN SELECT RAISE(ABORT, 'weather evidence is append-only'); END")

    def append_snapshot(self, location: WeatherLocation, source_url: str, raw: bytes, *, available_at: datetime, max_age_hours: float = 18) -> str:
        """Save one validated source response with the actual completed-retrieval clock."""
        _nws_url(source_url, hourly=True)
        captured = parse_hourly_forecast(raw, available_at=available_at, max_age_hours=max_age_hours)
        recorded = _instant(self.clock(), "recorded time")
        if _instant(available_at, "availability") > recorded:
            raise WeatherError("Weather availability cannot follow the archive write clock.")
        identifier = _snapshot_id(location, source_url, captured, recorded.isoformat())
        with closing(sqlite3.connect(self.path, timeout=15)) as connection, connection:
            connection.execute("INSERT OR IGNORE INTO weather_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                identifier, location.id, location.latitude, location.longitude, source_url,
                captured["source_sha256"], captured["issued_at"], captured["generated_at"],
                captured["available_at"], recorded.isoformat(), raw, json.dumps(captured["records"]),
                captured["records"][0]["valid_time"], captured["records"][-1]["valid_time"]))
            saved = connection.execute("SELECT id FROM weather_snapshots WHERE location_id=? AND latitude=? AND longitude=? AND source_url=? AND source_sha256=?",
                                       (location.id, location.latitude, location.longitude, source_url, captured["source_sha256"])).fetchone()
        return saved[0]

    def load_archive(self, locations: Sequence[WeatherLocation], *, max_age_hours: float = 18,
                     target_start: Any | None = None, target_end: Any | None = None) -> WeatherArchive:
        """Load only the requested named coordinates; configuration changes do not mix sites."""
        if not math.isfinite(max_age_hours) or not 0 < max_age_hours <= 48:
            raise WeatherError("Weather maximum forecast age must be positive and at most 48 hours.")
        locations = tuple(locations)
        if not locations or len({item.id for item in locations}) != len(locations):
            raise WeatherError("Weather archive requires distinct configured locations.")
        records, snapshots = [], []
        lower = _instant(target_start, "requested start") if target_start is not None else None
        upper = _instant(target_end, "requested end") if target_end is not None else None
        if lower is not None and upper is not None and lower > upper:
            raise WeatherError("Weather requested start must precede end.")
        with closing(sqlite3.connect(self.path, timeout=15)) as connection:
            connection.row_factory = sqlite3.Row
            for location in locations:
                query = "SELECT * FROM weather_snapshots WHERE location_id=? AND latitude=? AND longitude=?"
                parameters: list[Any] = [location.id, location.latitude, location.longitude]
                if lower is not None:
                    query += " AND valid_end>=?"
                    parameters.append(lower.isoformat())
                if upper is not None:
                    query += " AND valid_start<=?"
                    parameters.append(upper.isoformat())
                query += " ORDER BY available_at, id"
                for row in connection.execute(query, parameters):
                    if hashlib.sha256(row["raw_payload"]).hexdigest() != row["source_sha256"]:
                        raise WeatherError("Weather archive source hash mismatch; evidence is corrupted.")
                    _nws_url(row["source_url"], hourly=True)
                    # Reconstruct derived values from source bytes, not mutable derived JSON.
                    captured = parse_hourly_forecast(row["raw_payload"], available_at=row["available_at"], max_age_hours=48)
                    try:
                        saved_records = json.loads(row["records_json"])
                    except (ValueError, TypeError) as exc:
                        raise WeatherError("Weather archive derived records are invalid.") from exc
                    if (any(captured[key] != row[key] for key in ("source_sha256", "issued_at", "generated_at", "available_at"))
                            or captured["records"] != saved_records
                            or captured["records"][0]["valid_time"] != row["valid_start"]
                            or captured["records"][-1]["valid_time"] != row["valid_end"]
                            or _instant(row["available_at"], "archive availability") > _instant(row["recorded_at"], "archive write")
                            or _snapshot_id(location, row["source_url"], captured, row["recorded_at"]) != row["id"]):
                        raise WeatherError("Weather archive provenance or derived records disagree with source evidence.")
                    snapshots.append({key: row[key] for key in ("id", "location_id", "source_url", "source_sha256", "issued_at", "generated_at", "available_at")})
                    records.extend({**record, "location_id": location.id, "issued_at": row["issued_at"],
                                    "generated_at": row["generated_at"], "available_at": row["available_at"]}
                                   for record in captured["records"]
                                   if (lower is None or pd.Timestamp(record["valid_time"]) >= lower)
                                   and (upper is None or pd.Timestamp(record["valid_time"]) <= upper))
        frame = pd.DataFrame(records, columns=["valid_time", "temperature_c", "location_id", "issued_at", "generated_at", "available_at"])
        for column in ("valid_time", "issued_at", "generated_at", "available_at"):
            frame[column] = pd.to_datetime(frame[column], utc=True)
        report = {"schema_version": "nws-hourly-v1", "provider": "NOAA National Weather Service", "units": "degrees Celsius",
                  "record_type": "forecast", "max_age_hours": max_age_hours, "snapshot_count": len(snapshots), "record_count": len(frame),
                  "features": [f"temperature_c_{item.id}" for item in locations],
                  "locations": [{"id": item.id, "latitude": item.latitude, "longitude": item.longitude} for item in locations],
                  "availability_policy": "First actual retrieval; latest issued forecast available by origin and strictly before target start",
                  "issued_at_semantics": "NWS updateTime: underlying grid-data update, not independently verified public publication time; availability is actual retrieval",
                  "geography_assumption": "Separate city point predictors; not ERCOT weather-zone centroids or load weights",
                  "archive_sha256": hashlib.sha256(json.dumps(snapshots, sort_keys=True).encode()).hexdigest(),
                  "target_start_utc": lower.isoformat() if lower is not None else None,
                  "target_end_utc": upper.isoformat() if upper is not None else None,
                  "first_available_at": min((item["available_at"] for item in snapshots), default=None),
                  "last_available_at": max((item["available_at"] for item in snapshots), default=None),
                  "sources_scope": "Most recently captured source per configured location; complete source manifest is hashed and retained in SQLite",
                  "sources": [next(item for item in reversed(snapshots) if item["location_id"] == location.id)
                              for location in locations if any(item["location_id"] == location.id for item in snapshots)]}
        return WeatherArchive(frame, locations, report, max_age_hours)


def _fetch(session: requests.Session, url: str, *, user_agent: str, hourly: bool) -> bytes:
    _nws_url(url, hourly=hourly)
    try:
        with session.get(url, headers={"User-Agent": user_agent, "Accept": "application/geo+json"},
                         timeout=(10, 30), stream=True, allow_redirects=False) as response:
            if response.status_code != 200:
                raise WeatherError(f"NWS weather request failed (HTTP {response.status_code}).")
            parts, size = [], 0
            for part in response.iter_content(65536):
                size += len(part)
                if size > MAX_RESPONSE_BYTES:
                    raise WeatherError("NWS weather response exceeds the supported size.")
                parts.append(part)
            return b"".join(parts)
    except requests.RequestException as exc:
        raise WeatherError("NWS weather request failed or timed out; no cached source was substituted.") from exc


def collect_weather_forecasts(config: WeatherConfig, *, store: WeatherForecastStore | None = None,
                              session: requests.Session | None = None,
                              clock: Callable[[], datetime] = _now) -> dict[str, Any]:
    """Capture all configured points; raise explicitly on any missing/stale source.

    Completed point captures remain available if a later request fails. This
    does not mark collection complete or synthesize absent point temperatures.
    """
    if config.mode not in {"collect", "model"} or not config.user_agent:
        raise WeatherError("Weather collection requires enabled mode and an explicit User-Agent.")
    ledger = store or WeatherForecastStore(config.store_path, clock=clock)
    client = session or requests.Session()
    identifiers = []
    try:
        for location in config.locations:
            point_url = f"{NWS_ROOT}/points/{location.latitude:.4f},{location.longitude:.4f}"
            point = _read_json(_fetch(client, point_url, user_agent=config.user_agent, hourly=False))
            try:
                url = point["properties"]["forecastHourly"]
            except (KeyError, TypeError) as exc:
                raise WeatherError(f"NWS does not provide an hourly forecast for {location.id}.") from exc
            raw = _fetch(client, url, user_agent=config.user_agent, hourly=True)
            available = clock()
            identifiers.append(ledger.append_snapshot(location, url, raw, available_at=available,
                                                       max_age_hours=config.max_age_hours))
    finally:
        if session is None:
            client.close()
    return {"status": "collected", "location_count": len(identifiers), "snapshot_ids": identifiers,
            "provider": "NOAA National Weather Service", "historical_backfill": False}
