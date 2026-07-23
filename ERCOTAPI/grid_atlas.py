"""Cached-source adapters for the public Texas infrastructure atlas.

These ArcGIS layers are geographic reference data.  They are deliberately kept
outside the engineering-document RAG and must not be described as an ERCOT
planning model or real-time topology.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import requests


TEXAS_BOUNDARY_QUERY_URL = (
    "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/"
    "USA_States_Generalized_Boundaries/FeatureServer/0/query"
)
TRANSMISSION_QUERY_URL = (
    "https://services2.arcgis.com/LYMgRMwHfrWWEg3s/arcgis/rest/services/"
    "HIFLD_US_Electric_Power_Transmission_Lines/FeatureServer/0/query"
)
SUBSTATION_QUERY_URL = (
    "https://services5.arcgis.com/HDRa0B57OVrv2E1q/ArcGIS/rest/services/"
    "Electric_Substations/FeatureServer/0/query"
)
POWER_PLANT_QUERY_URL = (
    "https://services2.arcgis.com/FiaPA4ga0iQKduv3/ArcGIS/rest/services/"
    "Power_Plants_in_the_US/FeatureServer/0/query"
)

TRANSMISSION_SOURCE_URL = TRANSMISSION_QUERY_URL.removesuffix("/query")
SUBSTATION_SOURCE_URL = SUBSTATION_QUERY_URL.removesuffix("/query")
POWER_PLANT_SOURCE_URL = POWER_PLANT_QUERY_URL.removesuffix("/query")

PAGE_SIZE = 1_000
REQUEST_TIMEOUT = (6, 45)
MISSING_NUMERIC_SENTINEL = -999_000
SNAPSHOT_SCHEMA_VERSION = 1
PACKAGED_SNAPSHOT_PATH = Path(__file__).with_name("grid_atlas_snapshot.json.gz")
SNAPSHOT_COLLECTIONS = ("transmission_lines", "substations", "power_plants")
MINIMUM_SNAPSHOT_COUNTS = {
    "transmission_lines": 5_000,
    "substations": 4_000,
    "power_plants": 800,
}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def clean_number(value: Any) -> float | None:
    """Return a displayable number and discard HIFLD missing-value sentinels."""

    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= MISSING_NUMERIC_SENTINEL:
        return None
    return number


def _clean_arcgis_date(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    try:
        numeric = float(text)
    except ValueError:
        numeric = 0
    if numeric > 100_000_000_000:
        try:
            return datetime.fromtimestamp(
                numeric / 1_000,
                tz=timezone.utc,
            ).date().isoformat()
        except (OverflowError, OSError, ValueError):
            return text
    if re_match := re.fullmatch(r"(20\d{2})(0[1-9]|1[0-2])", text):
        return f"{re_match.group(1)}-{re_match.group(2)}"
    return text


def _request_json(
    session: requests.Session,
    url: str,
    *,
    method: str = "GET",
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    attempts: int = 3,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            if method == "POST":
                response = session.post(url, data=data, timeout=REQUEST_TIMEOUT)
            else:
                response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("ArcGIS returned a non-object response")
            if payload.get("error"):
                error = payload["error"]
                if isinstance(error, dict):
                    message = str(error.get("message") or error)
                else:
                    message = str(error)
                raise RuntimeError(f"ArcGIS returned an error: {message}")
            return payload
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.25 * (2**attempt))
    raise RuntimeError(str(last_error or "ArcGIS request failed"))


def _feature_pages(
    session: requests.Session,
    url: str,
    *,
    base_params: dict[str, Any],
    max_records: int,
) -> list[dict[str, Any]]:
    """Read an ArcGIS feature layer without assuming one response is complete."""

    features: list[dict[str, Any]] = []
    offset = 0
    while offset < max_records:
        params = {
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": min(PAGE_SIZE, max_records - offset),
            "cacheHint": "true",
            **base_params,
        }
        payload = _request_json(session, url, params=params)
        page = [
            feature
            for feature in payload.get("features", [])
            if isinstance(feature, dict)
        ]
        features.extend(page)
        if not page:
            break
        if payload.get("exceededTransferLimit") is not True and (
            len(page) < params["resultRecordCount"]
            or payload.get("exceededTransferLimit") is False
        ):
            break
        offset += len(page)
    return features


def _chunks(values: list[int], size: int) -> Iterable[list[int]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _texas_boundary(session: requests.Session) -> dict[str, Any]:
    payload = _request_json(
        session,
        TEXAS_BOUNDARY_QUERY_URL,
        params={
            "f": "json",
            "where": "STATE_ABBR='TX'",
            "outFields": "STATE_ABBR",
            "returnGeometry": "true",
            "outSR": "4326",
            "geometryPrecision": "5",
            "cacheHint": "true",
        },
    )
    features = payload.get("features") or []
    if not features or not isinstance(features[0], dict):
        raise RuntimeError("Texas boundary was not returned")
    geometry = features[0].get("geometry")
    if not isinstance(geometry, dict) or not geometry.get("rings"):
        raise RuntimeError("Texas boundary geometry was invalid")
    return geometry


def fetch_transmission_lines(
    session: requests.Session,
    *,
    max_records: int = 8_000,
) -> list[dict[str, Any]]:
    """Fetch public lines that intersect the generalized Texas boundary."""

    boundary = _texas_boundary(session)
    ids_payload = _request_json(
        session,
        TRANSMISSION_QUERY_URL,
        method="POST",
        data={
            "f": "json",
            "where": "1=1",
            "geometry": json.dumps(boundary, separators=(",", ":")),
            "geometryType": "esriGeometryPolygon",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "returnIdsOnly": "true",
            "cacheHint": "true",
        },
    )
    object_ids = sorted(
        int(value)
        for value in ids_payload.get("objectIds", [])
        if str(value).lstrip("-").isdigit()
    )[:max_records]
    records: list[dict[str, Any]] = []
    for batch in _chunks(object_ids, PAGE_SIZE):
        payload = _request_json(
            session,
            TRANSMISSION_QUERY_URL,
            method="POST",
            data={
                "f": "json",
                "objectIds": ",".join(str(value) for value in batch),
                "outFields": (
                    "OBJECTID_1,ID,TYPE,STATUS,OWNER,VOLTAGE,VOLT_CLASS,"
                    "INFERRED,SUB_1,SUB_2,SOURCE,SOURCEDATE"
                ),
                "returnGeometry": "true",
                "outSR": "4326",
                "geometryPrecision": "5",
                "maxAllowableOffset": "0.01",
                "cacheHint": "true",
            },
        )
        for feature in payload.get("features", []):
            if not isinstance(feature, dict):
                continue
            attributes = feature.get("attributes") or {}
            geometry = feature.get("geometry") or {}
            paths = []
            for raw_path in geometry.get("paths", []):
                path = [
                    [float(pair[0]), float(pair[1])]
                    for pair in raw_path
                    if (
                        isinstance(pair, (list, tuple))
                        and len(pair) >= 2
                        and clean_number(pair[0]) is not None
                        and clean_number(pair[1]) is not None
                    )
                ]
                if len(path) > 1:
                    paths.append(path)
            if not paths:
                continue
            substation_1 = _clean_text(attributes.get("SUB_1"))
            substation_2 = _clean_text(attributes.get("SUB_2"))
            line_id = _clean_text(attributes.get("ID"))
            records.append(
                {
                    "object_id": attributes.get("OBJECTID_1"),
                    "id": line_id,
                    "name": (
                        f"{substation_1} – {substation_2}"
                        if substation_1 and substation_2
                        else substation_1 or substation_2 or line_id or "Transmission line"
                    ),
                    "type": _clean_text(attributes.get("TYPE")),
                    "status": _clean_text(attributes.get("STATUS")),
                    "owner": _clean_text(attributes.get("OWNER")),
                    "voltage": clean_number(attributes.get("VOLTAGE")),
                    "voltage_class": _clean_text(attributes.get("VOLT_CLASS")),
                    "inferred": _clean_text(attributes.get("INFERRED")),
                    "substation_1": substation_1,
                    "substation_2": substation_2,
                    "source": _clean_text(attributes.get("SOURCE")),
                    "source_date": _clean_arcgis_date(attributes.get("SOURCEDATE")),
                    "paths": paths,
                }
            )
    return records


def fetch_substations(
    session: requests.Session,
    *,
    max_records: int = 7_000,
) -> list[dict[str, Any]]:
    features = _feature_pages(
        session,
        SUBSTATION_QUERY_URL,
        base_params={
            "where": "STATE='TX'",
            "outFields": (
                "OBJECTID_1,ID,NAME,CITY,STATE,COUNTY,TYPE,STATUS,MAX_VOLT,"
                "MIN_VOLT,LINES,SOURCE,SOURCEDATE"
            ),
            "returnGeometry": "true",
            "outSR": "4326",
            "geometryPrecision": "5",
            "orderByFields": "OBJECTID_1 ASC",
        },
        max_records=max_records,
    )
    records: list[dict[str, Any]] = []
    for feature in features:
        attributes = feature.get("attributes") or {}
        geometry = feature.get("geometry") or {}
        latitude = clean_number(geometry.get("y"))
        longitude = clean_number(geometry.get("x"))
        if latitude is None or longitude is None:
            latitude = clean_number(attributes.get("LATITUDE"))
            longitude = clean_number(attributes.get("LONGITUDE"))
        if latitude is None or longitude is None:
            continue
        records.append(
            {
                "object_id": attributes.get("OBJECTID_1"),
                "id": _clean_text(attributes.get("ID")),
                "name": _clean_text(attributes.get("NAME")) or "Unnamed substation",
                "city": _clean_text(attributes.get("CITY")),
                "county": _clean_text(attributes.get("COUNTY")),
                "type": _clean_text(attributes.get("TYPE")),
                "status": _clean_text(attributes.get("STATUS")),
                "max_voltage": clean_number(attributes.get("MAX_VOLT")),
                "min_voltage": clean_number(attributes.get("MIN_VOLT")),
                "line_count": clean_number(attributes.get("LINES")),
                "source": _clean_text(attributes.get("SOURCE")),
                "source_date": _clean_arcgis_date(attributes.get("SOURCEDATE")),
                "lat": latitude,
                "lon": longitude,
            }
        )
    return records


def fetch_power_plants(
    session: requests.Session,
    *,
    max_records: int = 2_000,
) -> list[dict[str, Any]]:
    features = _feature_pages(
        session,
        POWER_PLANT_QUERY_URL,
        base_params={
            "where": "State='Texas'",
            "outFields": (
                "FID,Plant_Code,Plant_Name,Utility_Na,sector_nam,City,County,"
                "State,PrimSource,source_des,tech_desc,Install_MW,Total_MW,"
                "Bat_MW,Solar_MW,Wind_MW,Source,Period,Latitude,Longitude"
            ),
            "returnGeometry": "true",
            "outSR": "4326",
            "geometryPrecision": "5",
            "orderByFields": "FID ASC",
        },
        max_records=max_records,
    )
    records: list[dict[str, Any]] = []
    for feature in features:
        attributes = feature.get("attributes") or {}
        geometry = feature.get("geometry") or {}
        latitude = clean_number(attributes.get("Latitude"))
        longitude = clean_number(attributes.get("Longitude"))
        if latitude is None or longitude is None:
            latitude = clean_number(geometry.get("y"))
            longitude = clean_number(geometry.get("x"))
        if latitude is None or longitude is None:
            continue
        installed_mw = clean_number(attributes.get("Install_MW"))
        summer_mw = clean_number(attributes.get("Total_MW"))
        records.append(
            {
                "object_id": attributes.get("FID"),
                "plant_code": _clean_text(attributes.get("Plant_Code")),
                "name": _clean_text(attributes.get("Plant_Name")) or "Unnamed power plant",
                "utility": _clean_text(attributes.get("Utility_Na")),
                "sector": _clean_text(attributes.get("sector_nam")),
                "city": _clean_text(attributes.get("City")),
                "county": _clean_text(attributes.get("County")),
                "fuel": _clean_text(attributes.get("PrimSource")) or "unknown",
                "fuel_description": _clean_text(attributes.get("source_des")),
                "technology": _clean_text(attributes.get("tech_desc")),
                "installed_mw": installed_mw,
                "summer_mw": summer_mw,
                "capacity_mw": installed_mw if installed_mw is not None else summer_mw,
                "battery_mw": clean_number(attributes.get("Bat_MW")),
                "solar_mw": clean_number(attributes.get("Solar_MW")),
                "wind_mw": clean_number(attributes.get("Wind_MW")),
                "source": _clean_text(attributes.get("Source")),
                "period": _clean_arcgis_date(attributes.get("Period")),
                "lat": latitude,
                "lon": longitude,
            }
        )
    return records


def load_public_texas_grid(
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Load each public layer independently so one outage cannot blank the atlas."""

    owned_session = session is None
    active_session = session or requests.Session()
    active_session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "ERCOT-Grid-Intelligence-Dashboard/1.0",
        }
    )
    payload: dict[str, Any] = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "transmission_lines": [],
        "substations": [],
        "power_plants": [],
        "errors": {},
    }
    loaders = (
        ("transmission_lines", fetch_transmission_lines),
        ("substations", fetch_substations),
        ("power_plants", fetch_power_plants),
    )
    try:
        for key, loader in loaders:
            try:
                payload[key] = loader(active_session)
            except Exception as exc:  # Keep partial source availability visible.
                payload["errors"][key] = f"{type(exc).__name__}: {exc}"
    finally:
        if owned_session:
            active_session.close()
    return payload


def validate_grid_atlas_snapshot(
    payload: Any,
    *,
    minimum_counts: Mapping[str, int] = MINIMUM_SNAPSHOT_COUNTS,
) -> dict[str, Any]:
    """Validate the packaged atlas artifact before the dashboard trusts it."""

    if not isinstance(payload, dict):
        raise RuntimeError("Packaged Grid Atlas snapshot is not a JSON object")
    if payload.get("snapshot_schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise RuntimeError("Packaged Grid Atlas snapshot schema is unsupported")
    for collection in SNAPSHOT_COLLECTIONS:
        if not isinstance(payload.get(collection), list):
            raise RuntimeError(f"Packaged Grid Atlas snapshot is missing {collection}")
    for collection in SNAPSHOT_COLLECTIONS:
        minimum = int(minimum_counts.get(collection, 0))
        if len(payload[collection]) < minimum:
            raise RuntimeError(
                f"Packaged Grid Atlas snapshot has only {len(payload[collection]):,} "
                f"{collection.replace('_', ' ')}; expected at least {minimum:,}"
            )
    errors = payload.get("errors")
    if errors not in ({}, None):
        raise RuntimeError("Packaged Grid Atlas snapshot contains source errors")
    return payload


def load_packaged_texas_grid(
    path: Path = PACKAGED_SNAPSHOT_PATH,
    *,
    minimum_counts: Mapping[str, int] = MINIMUM_SNAPSHOT_COUNTS,
) -> dict[str, Any]:
    """Load the checked-in snapshot without any network or paid API request."""

    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to load packaged Grid Atlas snapshot: {exc}") from exc
    return validate_grid_atlas_snapshot(payload, minimum_counts=minimum_counts)


def write_grid_atlas_snapshot(
    payload: dict[str, Any],
    path: Path = PACKAGED_SNAPSHOT_PATH,
    *,
    minimum_counts: Mapping[str, int] = MINIMUM_SNAPSHOT_COUNTS,
) -> dict[str, Any]:
    """Write a deterministic compressed snapshot after a successful live refresh."""

    source_errors = payload.get("errors") or {}
    if source_errors:
        raise RuntimeError(
            "Refusing to package an incomplete Grid Atlas snapshot: "
            + ", ".join(sorted(source_errors))
        )
    snapshot = {
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "generated_at": str(payload.get("fetched_at") or datetime.now(timezone.utc).isoformat()),
        "source_urls": {
            "transmission_lines": TRANSMISSION_SOURCE_URL,
            "substations": SUBSTATION_SOURCE_URL,
            "power_plants": POWER_PLANT_SOURCE_URL,
        },
        "transmission_lines": list(payload.get("transmission_lines") or []),
        "substations": list(payload.get("substations") or []),
        "power_plants": list(payload.get("power_plants") or []),
        "errors": {},
    }
    validate_grid_atlas_snapshot(snapshot, minimum_counts=minimum_counts)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_handle,
            mtime=0,
        ) as compressed_handle:
            compressed_handle.write(
                (
                    json.dumps(
                        snapshot,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8")
            )
    temporary_path.replace(path)
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the checked-in public Texas Grid Atlas snapshot."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGED_SNAPSHOT_PATH,
    )
    args = parser.parse_args()
    snapshot = write_grid_atlas_snapshot(
        load_public_texas_grid(),
        args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "generated_at": snapshot["generated_at"],
                "counts": {
                    collection: len(snapshot[collection])
                    for collection in SNAPSHOT_COLLECTIONS
                },
            }
        )
    )


if __name__ == "__main__":
    main()
