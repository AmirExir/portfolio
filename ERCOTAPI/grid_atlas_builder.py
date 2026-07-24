"""Offline builder for the packaged U.S.–Canada public Grid Atlas.

This module is intentionally separate from the Streamlit runtime.  It turns
approved public GIS downloads into deterministic gzip shards so dashboard
sessions never need to bulk-download infrastructure data.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
import struct
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ERCOTAPI.grid_atlas_store import ATLAS_SCHEMA_VERSION


HIFLD_TRANSMISSION_URL = (
    "https://services2.arcgis.com/LYMgRMwHfrWWEg3s/arcgis/rest/services/"
    "HIFLD_US_Electric_Power_Transmission_Lines/FeatureServer/0"
)
HIFLD_SUBSTATION_URL = (
    "https://services5.arcgis.com/HDRa0B57OVrv2E1q/ArcGIS/rest/services/"
    "Electric_Substations/FeatureServer/0"
)
EIA_POWER_PLANT_URL = (
    "https://services2.arcgis.com/FiaPA4ga0iQKduv3/ArcGIS/rest/services/"
    "Power_Plants_in_the_US/FeatureServer/0"
)
CANVEC_DATASET_URL = (
    "https://open.canada.ca/data/en/dataset/"
    "92dbea79-f644-4a62-b25e-8eb993ca0264"
)
NACEI_MAP_URL = (
    "https://geoappext.nrcan.gc.ca/arcgis/rest/services/NACEI/"
    "energy_infrastructure_of_north_america_en/MapServer"
)
ISO_RTO_SOURCE_URL = (
    "https://catalog.data.gov/dataset/independent-system-operators"
)
NERC_REFERENCE_URL = "https://nerc.com/AboutNERC/keyplayers/Pages/default.aspx"
OPENSTREETMAP_URL = "https://www.openstreetmap.org/"

MARKET_ALIASES = {
    "CALIFORNIA INDEPENDENT SYSTEM OPERATOR": ("caiso", "CAISO"),
    "ELECTRIC RELIABILITY COUNCIL OF TEXAS": ("ercot", "ERCOT"),
    "ELECTRIC RELIABILITY COUNCIL OF TEXAS, INC.": ("ercot", "ERCOT"),
    "ISO NEW ENGLAND": ("iso-ne", "ISO-NE"),
    "ISO NEW ENGLAND INC.": ("iso-ne", "ISO-NE"),
    "MIDCONTINENT INDEPENDENT SYSTEM OPERATOR": ("miso", "MISO"),
    "MIDCONTINENT INDEPENDENT TRANSMISSION SYSTEM OPERATOR, INC..": (
        "miso",
        "MISO",
    ),
    "MIDCONTINENT INDEPENDENT TRANSMISSION SYSTEM OPERATOR, INC.": (
        "miso",
        "MISO",
    ),
    "MIDCONTINENT INDEPENDENT SYSTEM OPERATOR, INC.": ("miso", "MISO"),
    "NEW YORK INDEPENDENT SYSTEM OPERATOR": ("nyiso", "NYISO"),
    "PJM INTERCONNECTION": ("pjm", "PJM"),
    "PJM INTERCONNECTION, LLC": ("pjm", "PJM"),
    "SOUTHWEST POWER POOL": ("spp", "SPP"),
}
MARKET_ORDER = ("ercot", "miso", "pjm", "caiso", "spp", "nyiso", "iso-ne")
MARKET_LABELS = {market_id: label for market_id, label in MARKET_ALIASES.values()}
COLLECTIONS = ("transmission_lines", "substations", "power_plants")
MISSING_NUMERIC_SENTINEL = -999_000


class GridAtlasBuildError(RuntimeError):
    """Raised when source data is incomplete or cannot be normalized safely."""


def _text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= MISSING_NUMERIC_SENTINEL:
        return None
    return number


def _property(properties: Mapping[str, Any], *names: str) -> Any:
    if not properties:
        return None
    folded = {str(key).casefold(): value for key, value in properties.items()}
    for name in names:
        if name in properties:
            return properties[name]
        if name.casefold() in folded:
            return folded[name.casefold()]
    return None


def _date(value: Any) -> str:
    text = _text(value)
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
    if len(text) == 6 and text.isdigit() and text[:4].startswith("20"):
        return f"{text[:4]}-{text[4:]}"
    return text


def _load_feature_collection(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GridAtlasBuildError(f"Unable to read GIS source {path}: {exc}") from exc
    if isinstance(payload, dict) and isinstance(payload.get("features"), list):
        return [
            feature
            for feature in payload["features"]
            if isinstance(feature, dict)
        ]
    if isinstance(payload, list):
        return [feature for feature in payload if isinstance(feature, dict)]
    raise GridAtlasBuildError(f"GIS source {path} is not a feature collection")


def _geometry(feature: Mapping[str, Any]) -> Mapping[str, Any]:
    geometry = feature.get("geometry")
    return geometry if isinstance(geometry, Mapping) else {}


def _properties(feature: Mapping[str, Any]) -> Mapping[str, Any]:
    properties = feature.get("properties")
    if isinstance(properties, Mapping):
        return properties
    attributes = feature.get("attributes")
    return attributes if isinstance(attributes, Mapping) else {}


def _line_paths(geometry: Mapping[str, Any]) -> list[list[list[float]]]:
    geometry_type = _text(geometry.get("type"))
    coordinates = geometry.get("coordinates")
    if geometry_type == "LineString":
        candidates = [coordinates]
    elif geometry_type == "MultiLineString":
        candidates = coordinates
    elif isinstance(geometry.get("paths"), list):
        candidates = geometry["paths"]
    else:
        candidates = []
    paths: list[list[list[float]]] = []
    for raw_path in candidates or []:
        path: list[list[float]] = []
        for point in raw_path or []:
            if not isinstance(point, Sequence) or len(point) < 2:
                continue
            longitude = _number(point[0])
            latitude = _number(point[1])
            if longitude is None or latitude is None:
                continue
            if -180 <= longitude <= 180 and -90 <= latitude <= 90:
                path.append([round(longitude, 5), round(latitude, 5)])
        if len(path) >= 2:
            paths.append(path)
    return paths


def _point(geometry: Mapping[str, Any], properties: Mapping[str, Any]) -> tuple[float, float] | None:
    coordinates = geometry.get("coordinates")
    if geometry.get("type") == "Point" and isinstance(coordinates, Sequence):
        longitude = _number(coordinates[0] if len(coordinates) else None)
        latitude = _number(coordinates[1] if len(coordinates) > 1 else None)
    elif geometry.get("x") is not None and geometry.get("y") is not None:
        longitude = _number(geometry.get("x"))
        latitude = _number(geometry.get("y"))
    else:
        longitude = _number(_property(properties, "Longitude", "LONGITUDE", "lon"))
        latitude = _number(_property(properties, "Latitude", "LATITUDE", "lat"))
    if (
        longitude is None
        or latitude is None
        or not -180 <= longitude <= 180
        or not -90 <= latitude <= 90
    ):
        return None
    return round(longitude, 5), round(latitude, 5)


def normalize_us_lines(features: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for feature in features:
        properties = _properties(feature)
        paths = _line_paths(_geometry(feature))
        if not paths:
            continue
        object_id = _property(properties, "OBJECTID_1", "OBJECTID", "FID")
        source_id = _text(_property(properties, "ID"))
        substation_1 = _text(_property(properties, "SUB_1"))
        substation_2 = _text(_property(properties, "SUB_2"))
        stable_id = _text(object_id) or source_id
        if not stable_id:
            stable_id = hashlib.sha1(
                json.dumps(paths, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:16]
        records.append(
            {
                "asset_id": f"hifld-line:{stable_id}",
                "object_id": object_id,
                "id": source_id,
                "name": (
                    f"{substation_1} – {substation_2}"
                    if substation_1 and substation_2
                    else substation_1 or substation_2 or source_id or "Transmission line"
                ),
                "type": _text(_property(properties, "TYPE")),
                "status": _text(_property(properties, "STATUS")),
                "owner": _text(_property(properties, "OWNER")),
                "voltage": _number(_property(properties, "VOLTAGE")),
                "voltage_class": _text(_property(properties, "VOLT_CLASS")),
                "inferred": _text(_property(properties, "INFERRED")),
                "substation_1": substation_1,
                "substation_2": substation_2,
                "source": _text(_property(properties, "SOURCE")) or "HIFLD",
                "source_date": _date(_property(properties, "SOURCEDATE")),
                "source_url": HIFLD_TRANSMISSION_URL,
                "country": "US",
                "markets": [],
                "paths": paths,
            }
        )
    return _deduplicate(records)


def normalize_us_substations(
    features: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for feature in features:
        properties = _properties(feature)
        coordinate = _point(_geometry(feature), properties)
        if coordinate is None:
            continue
        longitude, latitude = coordinate
        object_id = _property(properties, "OBJECTID_1", "OBJECTID", "FID")
        source_id = _text(_property(properties, "ID"))
        stable_id = _text(object_id) or source_id or f"{longitude:.5f},{latitude:.5f}"
        records.append(
            {
                "asset_id": f"hifld-substation:{stable_id}",
                "object_id": object_id,
                "id": source_id,
                "name": _text(_property(properties, "NAME")) or "Unnamed substation",
                "city": _text(_property(properties, "CITY")),
                "state": _text(_property(properties, "STATE")),
                "county": _text(_property(properties, "COUNTY")),
                "type": _text(_property(properties, "TYPE")),
                "status": _text(_property(properties, "STATUS")),
                "max_voltage": _number(_property(properties, "MAX_VOLT")),
                "min_voltage": _number(_property(properties, "MIN_VOLT")),
                "line_count": _number(_property(properties, "LINES")),
                "source": _text(_property(properties, "SOURCE")) or "HIFLD",
                "source_date": _date(_property(properties, "SOURCEDATE")),
                "source_url": HIFLD_SUBSTATION_URL,
                "country": "US",
                "markets": [],
                "lat": latitude,
                "lon": longitude,
            }
        )
    return _deduplicate(records)


def normalize_us_plants(features: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for feature in features:
        properties = _properties(feature)
        coordinate = _point(_geometry(feature), properties)
        if coordinate is None:
            continue
        longitude, latitude = coordinate
        plant_code = _text(_property(properties, "Plant_Code", "Plant Code"))
        object_id = _property(properties, "FID", "OBJECTID")
        stable_id = plant_code or _text(object_id) or f"{longitude:.5f},{latitude:.5f}"
        installed_mw = _number(_property(properties, "Install_MW"))
        total_mw = _number(_property(properties, "Total_MW"))
        records.append(
            {
                "asset_id": f"eia-plant:{stable_id}",
                "object_id": object_id,
                "plant_code": plant_code,
                "name": _text(_property(properties, "Plant_Name")) or "Unnamed power plant",
                "utility": _text(_property(properties, "Utility_Na")),
                "sector": _text(_property(properties, "sector_nam")),
                "city": _text(_property(properties, "City")),
                "state": _text(_property(properties, "State")),
                "county": _text(_property(properties, "County")),
                "fuel": _text(_property(properties, "PrimSource")) or "unknown",
                "fuel_description": _text(_property(properties, "source_des")),
                "technology": _text(_property(properties, "tech_desc")),
                "installed_mw": installed_mw,
                "summer_mw": total_mw,
                "capacity_mw": installed_mw if installed_mw is not None else total_mw,
                "battery_mw": _number(_property(properties, "Bat_MW")),
                "solar_mw": _number(_property(properties, "Solar_MW")),
                "wind_mw": _number(_property(properties, "Wind_MW")),
                "source": _text(_property(properties, "Source")) or "U.S. EIA",
                "period": _date(_property(properties, "Period")),
                "source_url": EIA_POWER_PLANT_URL,
                "country": "US",
                "markets": [],
                "lat": latitude,
                "lon": longitude,
            }
        )
    return _deduplicate(records)


def _deduplicate(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        asset_id = str(record.get("asset_id") or "")
        if not asset_id:
            raise GridAtlasBuildError("Normalized Grid Atlas record has no asset id")
        by_id[asset_id] = record
    return [by_id[asset_id] for asset_id in sorted(by_id)]


def _osm_voltages(value: Any) -> list[float]:
    """Parse OSM's semicolon-delimited, volts-based voltage tag into kV."""

    voltages: set[float] = set()
    for item in str(value or "").replace(",", ";").split(";"):
        number = _number(item.strip())
        if number is None or number < 1_000:
            continue
        kilovolts = number / 1_000
        if 1 <= kilovolts <= 1_500:
            voltages.add(round(kilovolts, 3))
    return sorted(voltages, reverse=True)


def _polygon_center(geometry: Mapping[str, Any]) -> tuple[float, float] | None:
    coordinates = geometry.get("coordinates")
    geometry_type = _text(geometry.get("type"))
    if geometry_type == "Polygon":
        rings = coordinates
    elif geometry_type == "MultiPolygon":
        rings = (coordinates or [None])[0]
    else:
        return None
    outer = (rings or [None])[0] or []
    points = [
        (float(point[0]), float(point[1]))
        for point in outer
        if isinstance(point, Sequence)
        and len(point) >= 2
        and _number(point[0]) is not None
        and _number(point[1]) is not None
    ]
    if not points:
        return None
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def normalize_osm_power(
    features: Iterable[Mapping[str, Any]],
    *,
    retrieved_at: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize a verified OSM GeoJSON extract without treating it as authoritative."""

    lines: list[dict[str, Any]] = []
    substations: list[dict[str, Any]] = []
    for feature in features:
        properties = _properties(feature)
        power = _text(_property(properties, "power"))
        voltages = _osm_voltages(_property(properties, "voltage"))
        if not voltages:
            continue
        osm_id = _text(
            _property(properties, "@id", "id", "osm_id", "osm_way_id")
        )
        common = {
            "osm_asset_id": osm_id,
            "osm_voltages": voltages,
            "operator": _text(_property(properties, "operator")),
            "owner": _text(_property(properties, "owner")),
            "circuits": _text(_property(properties, "circuits")),
            "cables": _text(_property(properties, "cables")),
            "reference": _text(_property(properties, "ref")),
            "retrieved_at": retrieved_at,
            "osm_kind": "facility",
        }
        geometry = _geometry(feature)
        if power in {"line", "minor_line", "cable"}:
            paths = _line_paths(geometry)
            if paths:
                lines.append({**common, "paths": paths})
        elif power in {"substation", "station"}:
            coordinate = _point(geometry, properties) or _polygon_center(geometry)
            if coordinate is not None:
                substations.append(
                    {**common, "lon": coordinate[0], "lat": coordinate[1]}
                )
        elif (
            power == "transformer"
            and _text(_property(properties, "transformer")).casefold()
            in {"auto", "autotransformer"}
        ):
            coordinate = _point(geometry, properties) or _polygon_center(geometry)
            if coordinate is not None:
                substations.append(
                    {
                        **common,
                        "osm_kind": "autotransformer",
                        "lon": coordinate[0],
                        "lat": coordinate[1],
                    }
                )
    return lines, substations


def _distance_meters(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    latitude = math.radians((float(first[1]) + float(second[1])) / 2)
    dx = (float(first[0]) - float(second[0])) * 111_320 * math.cos(latitude)
    dy = (float(first[1]) - float(second[1])) * 110_540
    return math.hypot(dx, dy)


def _point_segment_distance_meters(
    point: Sequence[float],
    start: Sequence[float],
    end: Sequence[float],
) -> float:
    latitude = math.radians(float(point[1]))
    scale_x = 111_320 * math.cos(latitude)
    scale_y = 110_540
    px, py = 0.0, 0.0
    ax = (float(start[0]) - float(point[0])) * scale_x
    ay = (float(start[1]) - float(point[1])) * scale_y
    bx = (float(end[0]) - float(point[0])) * scale_x
    by = (float(end[1]) - float(point[1])) * scale_y
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    if denominator:
        position = max(0.0, min(1.0, -(ax * dx + ay * dy) / denominator))
        ax += position * dx
        ay += position * dy
    return math.hypot(ax - px, ay - py)


def _sample_paths(
    paths: Iterable[Sequence[Sequence[float]]],
    *,
    spacing_meters: float = 400,
    limit: int = 500,
) -> list[list[float]]:
    samples: list[list[float]] = []
    for path in paths:
        if len(path) < 2:
            continue
        for start, end in zip(path, path[1:]):
            distance = _distance_meters(start, end)
            steps = max(1, int(math.ceil(distance / spacing_meters)))
            for step in range(steps):
                fraction = step / steps
                samples.append(
                    [
                        float(start[0]) + (float(end[0]) - float(start[0])) * fraction,
                        float(start[1]) + (float(end[1]) - float(start[1])) * fraction,
                    ]
                )
                if len(samples) >= limit:
                    return samples
        samples.append([float(path[-1][0]), float(path[-1][1])])
        if len(samples) >= limit:
            return samples
    return samples


def _coverage(
    samples: Sequence[Sequence[float]],
    paths: Sequence[Sequence[Sequence[float]]],
    tolerance_meters: float,
) -> tuple[float, float]:
    if not samples:
        return 0.0, float("inf")
    distances = [
        min(
            _point_segment_distance_meters(point, start, end)
            for path in paths
            for start, end in zip(path, path[1:])
        )
        for point in samples
    ]
    return (
        sum(distance <= tolerance_meters for distance in distances) / len(distances),
        sum(distances) / len(distances),
    )


def _line_match_score(
    primary_paths: Sequence[Sequence[Sequence[float]]],
    osm_paths: Sequence[Sequence[Sequence[float]]],
    *,
    tolerance_meters: float,
) -> float:
    primary_samples = _sample_paths(primary_paths)
    osm_samples = _sample_paths(osm_paths)
    primary_coverage, primary_distance = _coverage(
        primary_samples, osm_paths, tolerance_meters
    )
    osm_coverage, osm_distance = _coverage(
        osm_samples, primary_paths, tolerance_meters
    )
    if primary_coverage < 0.8 or osm_coverage < 0.7:
        return 0.0
    proximity = max(
        0.0,
        1 - ((primary_distance + osm_distance) / 2) / tolerance_meters,
    )
    return 0.45 * primary_coverage + 0.35 * osm_coverage + 0.2 * proximity


def _apply_osm_fields(
    record: dict[str, Any],
    osm: Mapping[str, Any],
    *,
    voltage_field: str,
    score: float,
) -> tuple[bool, bool]:
    filled_voltage = record.get(voltage_field) is None
    if filled_voltage:
        record[voltage_field] = max(osm["osm_voltages"])
        record["voltage_source"] = "OpenStreetMap"
        record["voltage_source_url"] = OPENSTREETMAP_URL
        record["voltage_retrieved_at"] = osm["retrieved_at"]
        record["voltage_match_confidence"] = round(score, 3)
        record["voltage_match_status"] = "OSM-suggested"
    record["osm_metadata_source"] = "OpenStreetMap"
    record["osm_metadata_source_url"] = OPENSTREETMAP_URL
    record["osm_metadata_retrieved_at"] = osm["retrieved_at"]
    record["osm_match_confidence"] = round(score, 3)
    record["osm_voltages"] = list(osm["osm_voltages"])
    record["osm_asset_id"] = osm.get("osm_asset_id", "")
    filled_metadata = False
    for target, source in (
        ("owner", "owner"),
        ("operator", "operator"),
        ("circuits", "circuits"),
        ("cables", "cables"),
        ("line_reference", "reference"),
    ):
        if not record.get(target) and osm.get(source):
            record[target] = osm[source]
            filled_metadata = True
    return filled_voltage, filled_metadata


def enrich_unknown_voltage_from_osm(
    lines: list[dict[str, Any]],
    substations: list[dict[str, Any]],
    osm_lines: Sequence[Mapping[str, Any]],
    osm_substations: Sequence[Mapping[str, Any]],
    *,
    line_tolerance_meters: float = 250,
    substation_tolerance_meters: float = 200,
) -> dict[str, int]:
    """Fill only unknown voltage values using conservative, unambiguous OSM matches."""

    stats = {
        "line_matches": 0,
        "line_ambiguous": 0,
        "line_metadata_matches": 0,
        "substation_matches": 0,
        "substation_ambiguous": 0,
        "substation_metadata_matches": 0,
        "autotransformer_matches": 0,
    }
    cell_size = 0.1
    line_cells: dict[tuple[int, int], set[int]] = {}
    for index, osm in enumerate(osm_lines):
        for longitude, latitude in _sample_paths(
            osm.get("paths", []), spacing_meters=2_000, limit=250
        ):
            cell = (
                math.floor(longitude / cell_size),
                math.floor(latitude / cell_size),
            )
            line_cells.setdefault(cell, set()).add(index)
    substation_cells: dict[tuple[int, int], set[int]] = {}
    for index, osm in enumerate(osm_substations):
        cell = (
            math.floor(float(osm["lon"]) / cell_size),
            math.floor(float(osm["lat"]) / cell_size),
        )
        substation_cells.setdefault(cell, set()).add(index)

    for record in lines:
        if (
            record.get("voltage") is not None
            and record.get("owner")
            and record.get("operator")
            and record.get("circuits")
            and record.get("cables")
            and record.get("line_reference")
        ):
            continue
        candidate_indexes: set[int] = set()
        for longitude, latitude in _sample_paths(
            record.get("paths", []), spacing_meters=2_000, limit=250
        ):
            cell_x = math.floor(longitude / cell_size)
            cell_y = math.floor(latitude / cell_size)
            for offset_x in (-1, 0, 1):
                for offset_y in (-1, 0, 1):
                    candidate_indexes.update(
                        line_cells.get(
                            (cell_x + offset_x, cell_y + offset_y), ()
                        )
                    )
        grouped: dict[tuple[float, ...], list[Mapping[str, Any]]] = {}
        for osm in (osm_lines[index] for index in candidate_indexes):
            grouped.setdefault(tuple(osm.get("osm_voltages", [])), []).append(osm)
        scored: list[tuple[float, Mapping[str, Any]]] = []
        for voltages, candidates in grouped.items():
            combined_paths = [
                path
                for candidate in candidates
                for path in candidate.get("paths", [])
            ]
            common_fields: dict[str, Any] = {}
            for field in (
                "operator",
                "owner",
                "circuits",
                "cables",
                "reference",
                "retrieved_at",
            ):
                values = {
                    str(candidate.get(field) or "")
                    for candidate in candidates
                    if candidate.get(field)
                }
                common_fields[field] = values.pop() if len(values) == 1 else ""
            combined = {
                **common_fields,
                "osm_asset_id": ",".join(
                    sorted(
                        str(candidate.get("osm_asset_id") or "")
                        for candidate in candidates
                        if candidate.get("osm_asset_id")
                    )
                ),
                "osm_voltages": list(voltages),
                "paths": combined_paths,
            }
            scored.append(
                (
                    _line_match_score(
                        record.get("paths", []),
                        combined_paths,
                        tolerance_meters=line_tolerance_meters,
                    ),
                    combined,
                )
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        scored = [item for item in scored if item[0] >= 0.82]
        known_voltage = _number(record.get("voltage"))
        if known_voltage is not None:
            compatible = [
                item
                for item in scored
                if any(
                    math.isclose(known_voltage, voltage, abs_tol=0.1)
                    for voltage in item[1].get("osm_voltages", [])
                )
            ]
            scored = compatible or scored
        if not scored:
            continue
        best_score, best = scored[0]
        competing = [
            candidate
            for score, candidate in scored[1:]
            if best_score - score < 0.08
            and candidate.get("osm_voltages") != best.get("osm_voltages")
        ]
        if competing:
            stats["line_ambiguous"] += 1
            continue
        filled_voltage, filled_metadata = _apply_osm_fields(
            record, best, voltage_field="voltage", score=best_score
        )
        stats["line_matches"] += int(filled_voltage)
        stats["line_metadata_matches"] += int(filled_metadata)

    for record in substations:
        cell_x = math.floor(float(record["lon"]) / cell_size)
        cell_y = math.floor(float(record["lat"]) / cell_size)
        candidate_indexes: set[int] = set()
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                candidate_indexes.update(
                    substation_cells.get(
                        (cell_x + offset_x, cell_y + offset_y), ()
                    )
                )
        auto_candidates = sorted(
            (
                (
                    _distance_meters(
                        [record["lon"], record["lat"]],
                        [osm["lon"], osm["lat"]],
                    ),
                    osm,
                )
                for osm in (
                    osm_substations[index] for index in candidate_indexes
                )
                if osm.get("osm_kind") == "autotransformer"
            ),
            key=lambda item: item[0],
        )
        auto_candidates = [
            item for item in auto_candidates if item[0] <= 750
        ]
        known_voltage = _number(record.get("max_voltage"))
        if known_voltage is not None:
            compatible = [
                item
                for item in auto_candidates
                if any(
                    math.isclose(known_voltage, voltage, abs_tol=0.1)
                    for voltage in item[1].get("osm_voltages", [])
                )
            ]
            auto_candidates = compatible or auto_candidates
        if auto_candidates:
            nearest_distance = auto_candidates[0][0]
            nearby = [
                osm
                for distance, osm in auto_candidates
                if distance - nearest_distance <= 150
            ]
            record["autotransformer"] = True
            record["autotransformer_source"] = "OpenStreetMap"
            record["autotransformer_source_url"] = OPENSTREETMAP_URL
            record["autotransformer_retrieved_at"] = nearby[0]["retrieved_at"]
            record["autotransformer_match_status"] = "OSM-suggested"
            record["autotransformer_match_confidence"] = round(
                max(0.82, 1 - nearest_distance / 750), 3
            )
            record["autotransformer_osm_ids"] = sorted(
                {
                    str(osm.get("osm_asset_id") or "")
                    for osm in nearby
                    if osm.get("osm_asset_id")
                }
            )
            record["autotransformer_voltages"] = sorted(
                {
                    voltage
                    for osm in nearby
                    for voltage in osm.get("osm_voltages", [])
                },
                reverse=True,
            )
            stats["autotransformer_matches"] += 1

        if (
            record.get("max_voltage") is not None
            and record.get("owner")
            and record.get("operator")
        ):
            continue
        candidates = sorted(
            (
                (
                    _distance_meters(
                        [record["lon"], record["lat"]],
                        [osm["lon"], osm["lat"]],
                    ),
                    osm,
                )
                for osm in (
                    osm_substations[index] for index in candidate_indexes
                )
                if osm.get("osm_kind") != "autotransformer"
            ),
            key=lambda item: item[0],
        )
        candidates = [
            item for item in candidates if item[0] <= substation_tolerance_meters
        ]
        known_voltage = _number(record.get("max_voltage"))
        if known_voltage is not None:
            compatible = [
                item
                for item in candidates
                if any(
                    math.isclose(known_voltage, voltage, abs_tol=0.1)
                    for voltage in item[1].get("osm_voltages", [])
                )
            ]
            candidates = compatible or candidates
        if not candidates:
            continue
        distance, best = candidates[0]
        competing = [
            candidate
            for other_distance, candidate in candidates[1:]
            if other_distance - distance < 75
            and candidate.get("osm_voltages") != best.get("osm_voltages")
        ]
        if competing:
            stats["substation_ambiguous"] += 1
            continue
        score = max(0.82, 1 - distance / substation_tolerance_meters)
        filled_voltage, filled_metadata = _apply_osm_fields(
            record, best, voltage_field="max_voltage", score=score
        )
        stats["substation_matches"] += int(filled_voltage)
        stats["substation_metadata_matches"] += int(filled_metadata)
    return stats


def _read_dbf(path: Path) -> list[dict[str, Any]]:
    data = path.read_bytes()
    if len(data) < 33:
        raise GridAtlasBuildError(f"DBF file is truncated: {path}")
    record_count = struct.unpack_from("<I", data, 4)[0]
    header_length = struct.unpack_from("<H", data, 8)[0]
    record_length = struct.unpack_from("<H", data, 10)[0]
    fields: list[tuple[str, str, int, int]] = []
    offset = 32
    while offset + 32 <= header_length and data[offset] != 0x0D:
        descriptor = data[offset : offset + 32]
        name = descriptor[:11].split(b"\0", 1)[0].decode("ascii", errors="replace")
        fields.append((name, chr(descriptor[11]), descriptor[16], descriptor[17]))
        offset += 32
    records: list[dict[str, Any]] = []
    for index in range(record_count):
        start = header_length + index * record_length
        row = data[start : start + record_length]
        if len(row) < record_length:
            raise GridAtlasBuildError(f"DBF record is truncated: {path}")
        if row[:1] == b"*":
            records.append({})
            continue
        cursor = 1
        record: dict[str, Any] = {}
        for name, field_type, width, decimals in fields:
            raw = row[cursor : cursor + width].decode("utf-8", errors="replace").strip()
            cursor += width
            if field_type in {"N", "F"} and raw:
                try:
                    value: Any = float(raw) if decimals or "." in raw else int(raw)
                except ValueError:
                    value = raw
            elif field_type == "L":
                value = raw.upper() in {"Y", "T"}
            else:
                value = raw
            record[name] = value
        records.append(record)
    return records


def _read_shapefile(path: Path) -> list[dict[str, Any]]:
    """Read point/polyline/polygon geometry with only the standard library."""

    data = path.read_bytes()
    if len(data) < 100 or struct.unpack_from(">i", data, 0)[0] != 9994:
        raise GridAtlasBuildError(f"Invalid ESRI shapefile: {path}")
    records: list[dict[str, Any]] = []
    offset = 100
    while offset + 8 <= len(data):
        _, content_words = struct.unpack_from(">ii", data, offset)
        content_length = content_words * 2
        content = data[offset + 8 : offset + 8 + content_length]
        if len(content) != content_length or len(content) < 4:
            raise GridAtlasBuildError(f"Truncated shapefile record: {path}")
        shape_type = struct.unpack_from("<i", content, 0)[0]
        geometry: dict[str, Any] = {"shape_type": shape_type}
        if shape_type == 0:
            geometry["parts"] = []
        elif shape_type == 1:
            longitude, latitude = struct.unpack_from("<2d", content, 4)
            geometry["point"] = [longitude, latitude]
        elif shape_type in {3, 5}:
            if len(content) < 44:
                raise GridAtlasBuildError(f"Truncated poly geometry: {path}")
            part_count, point_count = struct.unpack_from("<2i", content, 36)
            parts_offset = 44
            points_offset = parts_offset + 4 * part_count
            part_starts = list(
                struct.unpack_from(f"<{part_count}i", content, parts_offset)
            )
            flat_points = [
                list(struct.unpack_from("<2d", content, points_offset + 16 * index))
                for index in range(point_count)
            ]
            part_starts.append(point_count)
            geometry["parts"] = [
                flat_points[part_starts[index] : part_starts[index + 1]]
                for index in range(part_count)
            ]
        else:
            raise GridAtlasBuildError(
                f"Unsupported shapefile geometry type {shape_type}: {path}"
            )
        records.append(geometry)
        offset += 8 + content_length
    return records


def _shapefile_records(path: Path) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    geometries = _read_shapefile(path)
    attributes = _read_dbf(path.with_suffix(".dbf"))
    if len(geometries) != len(attributes):
        raise GridAtlasBuildError(f"Shape/DBF record counts differ: {path}")
    yield from zip(geometries, attributes)


def normalize_canvec_lines(canvec_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    path = canvec_dir / "power_line_1.shp"
    for index, (geometry, attributes) in enumerate(_shapefile_records(path), start=1):
        paths = []
        for raw_path in geometry.get("parts", []):
            path_points = [
                [round(float(point[0]), 5), round(float(point[1]), 5)]
                for point in raw_path
                if len(point) >= 2
            ]
            if len(path_points) >= 2:
                paths.append(path_points)
        if not paths:
            continue
        feature_id = _text(_property(attributes, "feature_id")) or str(index)
        location = _text(
            _property(attributes, "liloc_en", "liloceng", "liloc")
        )
        records.append(
            {
                "asset_id": f"canvec-line:{feature_id}",
                "object_id": index,
                "id": feature_id,
                "name": "Canadian power line",
                "type": location or "Power line",
                "status": "",
                "owner": "",
                "voltage": None,
                "voltage_class": "",
                "inferred": "",
                "substation_1": "",
                "substation_2": "",
                "source": "Natural Resources Canada CanVec",
                "source_date": _date(_property(attributes, "datemax")),
                "source_url": CANVEC_DATASET_URL,
                "country": "CA",
                "markets": [],
                "paths": paths,
            }
        )
    return _deduplicate(records)


def _polygon_centroid(parts: Sequence[Sequence[Sequence[float]]]) -> tuple[float, float] | None:
    points = [point for part in parts for point in part if len(point) >= 2]
    if not points:
        return None
    return (
        sum(float(point[0]) for point in points) / len(points),
        sum(float(point[1]) for point in points) / len(points),
    )


def normalize_canvec_substations(canvec_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    sources = (
        ("transformer_station_0.shp", "point"),
        ("transformer_station_2.shp", "polygon"),
    )
    for filename, representation in sources:
        path = canvec_dir / filename
        for index, (geometry, attributes) in enumerate(_shapefile_records(path), start=1):
            if representation == "point":
                raw_point = geometry.get("point")
                coordinate = (
                    (float(raw_point[0]), float(raw_point[1]))
                    if isinstance(raw_point, Sequence) and len(raw_point) >= 2
                    else None
                )
            else:
                coordinate = _polygon_centroid(geometry.get("parts", []))
            if coordinate is None:
                continue
            longitude, latitude = coordinate
            feature_id = _text(_property(attributes, "feature_id")) or str(index)
            records.append(
                {
                    "asset_id": f"canvec-transformer-{representation}:{feature_id}",
                    "object_id": index,
                    "id": feature_id,
                    "name": "Canadian transformer station",
                    "city": "",
                    "province": "",
                    "county": "",
                    "type": f"Transformer station ({representation} source)",
                    "status": "",
                    "max_voltage": None,
                    "min_voltage": None,
                    "line_count": None,
                    "source": "Natural Resources Canada CanVec",
                    "source_date": _date(_property(attributes, "datemax")),
                    "source_url": CANVEC_DATASET_URL,
                    "country": "CA",
                    "markets": [],
                    "lat": round(latitude, 5),
                    "lon": round(longitude, 5),
                }
            )
    return _deduplicate(records)


def normalize_canada_plants(
    feature_groups: Iterable[Iterable[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for features in feature_groups:
        for feature in features:
            properties = _properties(feature)
            country = _text(_property(properties, "Country"))
            if country and country.casefold() != "canada":
                continue
            coordinate = _point(_geometry(feature), properties)
            if coordinate is None:
                continue
            longitude, latitude = coordinate
            name = _text(
                _property(properties, "Facility", "Plant_Name", "Station")
            ) or "Unnamed Canadian power plant"
            province = _text(_property(properties, "StateProv", "Province"))
            owner = _text(_property(properties, "Owner"))
            operator = _text(_property(properties, "Operator"))
            capacity = _number(
                _property(properties, "Total_MW", "TotalMW", "Capacity")
            )
            primary_source = _text(
                _property(properties, "PrimSource", "PrimRenew", "Primary")
            ) or "unknown"
            source_id = _text(
                _property(properties, "OBJECTID", "FID", "Facility_ID", "ID")
            )
            stable_key = "|".join(
                (
                    name.casefold(),
                    province.casefold(),
                    f"{longitude:.4f}",
                    f"{latitude:.4f}",
                )
            )
            stable_id = hashlib.sha1(stable_key.encode("utf-8")).hexdigest()[:16]
            records.append(
                {
                    "asset_id": f"nacei-plant:{stable_id}",
                    "object_id": source_id,
                    "plant_code": "",
                    "name": name,
                    "utility": owner or operator,
                    "sector": "",
                    "city": _text(_property(properties, "City")),
                    "province": province,
                    "county": "",
                    "fuel": primary_source,
                    "fuel_description": primary_source,
                    "technology": _text(_property(properties, "Technology")),
                    "installed_mw": capacity,
                    "summer_mw": None,
                    "capacity_mw": capacity,
                    "battery_mw": None,
                    "solar_mw": _number(_property(properties, "Solar_MW")),
                    "wind_mw": _number(_property(properties, "Wind_MW")),
                    "source": _text(_property(properties, "Source")) or "NRCan NACEI",
                    "period": _date(_property(properties, "Period")) or "2017-08",
                    "source_url": f"{NACEI_MAP_URL}",
                    "country": "CA",
                    "markets": [],
                    "lat": latitude,
                    "lon": longitude,
                }
            )
    # Layers 15 and 28 overlap.  Prefer the first record for identical facility/location.
    by_facility: dict[str, dict[str, Any]] = {}
    for record in records:
        key = "|".join(
            (
                record["name"].casefold(),
                _text(record.get("province")).casefold(),
                f"{record['lon']:.4f}",
                f"{record['lat']:.4f}",
            )
        )
        existing = by_facility.get(key)
        if existing is None or (
            existing.get("capacity_mw") is None
            and record.get("capacity_mw") is not None
        ):
            by_facility[key] = record
    return _deduplicate(by_facility.values())


def _geojson_polygons(geometry: Mapping[str, Any]) -> list[dict[str, Any]]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Polygon":
        candidates = [coordinates]
    elif geometry_type == "MultiPolygon":
        candidates = coordinates
    elif isinstance(geometry.get("rings"), list):
        # ArcGIS rings do not explicitly group holes.  The first/largest ring is
        # sufficient for these generalized display/filter footprints.
        candidates = [[ring] for ring in geometry["rings"]]
    else:
        candidates = []
    polygons: list[dict[str, Any]] = []
    for raw_polygon in candidates or []:
        rings: list[list[list[float]]] = []
        for raw_ring in raw_polygon or []:
            ring = [
                [round(float(point[0]), 5), round(float(point[1]), 5)]
                for point in raw_ring or []
                if isinstance(point, Sequence) and len(point) >= 2
            ]
            if len(ring) >= 4:
                if ring[0] != ring[-1]:
                    ring.append(list(ring[0]))
                rings.append(ring)
        if rings:
            largest_index = max(
                range(len(rings)),
                key=lambda index: abs(_signed_ring_area(rings[index])),
            )
            outer = rings.pop(largest_index)
            polygons.append({"outer": outer, "holes": rings})
    return polygons


def normalize_boundaries(
    features: Iterable[Mapping[str, Any]],
    *,
    kind: str,
) -> list[dict[str, Any]]:
    boundaries: list[dict[str, Any]] = []
    for feature in features:
        properties = _properties(feature)
        raw_name = _text(
            _property(
                properties,
                "NAME",
                "NERC_ERO_Region_Name",
                "ERO_Abbrev",
                "NERC_Label",
                "NERC",
                "SUBNAME",
            )
        )
        polygons = _geojson_polygons(_geometry(feature))
        if not raw_name or not polygons:
            continue
        if kind == "market":
            alias = MARKET_ALIASES.get(raw_name.upper())
            if alias is None:
                continue
            boundary_id, label = alias
        else:
            abbreviation = _text(
                _property(properties, "ERO_Abbrev", "NERC", "ID")
            )
            boundary_id = (abbreviation or raw_name).casefold().replace(" ", "-")
            label = abbreviation or raw_name
        boundaries.append(
            {
                "id": boundary_id,
                "label": label,
                "kind": kind,
                "source_name": raw_name,
                "source": (
                    "HIFLD public ISO/RTO footprint snapshot"
                    if kind == "market"
                    else "Public approximate NERC/ERO boundary layer"
                ),
                "source_url": ISO_RTO_SOURCE_URL if kind == "market" else NERC_REFERENCE_URL,
                "polygons": polygons,
            }
        )
    if kind == "market":
        found = {boundary["id"] for boundary in boundaries}
        missing = set(MARKET_ORDER) - found
        if missing:
            raise GridAtlasBuildError(
                "ISO/RTO boundary source is missing: " + ", ".join(sorted(missing))
            )
        boundaries.sort(key=lambda boundary: MARKET_ORDER.index(boundary["id"]))
    else:
        boundaries.sort(key=lambda boundary: boundary["label"])
    return boundaries


def _web_mercator_to_wgs84(x: float, y: float) -> list[float]:
    radius = 6_378_137.0
    longitude = math.degrees(float(x) / radius)
    latitude = math.degrees(
        2 * math.atan(math.exp(float(y) / radius)) - math.pi / 2
    )
    return [round(longitude, 5), round(latitude, 5)]


def load_market_boundaries(path: Path) -> list[dict[str, Any]]:
    """Load the original HIFLD shapefile or an already-converted GeoJSON file."""

    if path.suffix.casefold() != ".shp":
        return normalize_boundaries(
            _load_feature_collection(path),
            kind="market",
        )
    features: list[dict[str, Any]] = []
    for geometry, attributes in _shapefile_records(path):
        polygons = []
        for raw_part in geometry.get("parts", []):
            ring = [
                _web_mercator_to_wgs84(float(point[0]), float(point[1]))
                for point in raw_part
                if len(point) >= 2
            ]
            if len(ring) < 4:
                continue
            if ring[0] != ring[-1]:
                ring.append(list(ring[0]))
            polygons.append([ring])
        if polygons:
            features.append(
                {
                    "type": "Feature",
                    "properties": dict(attributes),
                    "geometry": {
                        "type": "MultiPolygon",
                        "coordinates": polygons,
                    },
                }
            )
    return normalize_boundaries(features, kind="market")


def _signed_ring_area(ring: Sequence[Sequence[float]]) -> float:
    return sum(
        float(ring[index][0]) * float(ring[index + 1][1])
        - float(ring[index + 1][0]) * float(ring[index][1])
        for index in range(len(ring) - 1)
    ) / 2


def _point_in_ring(longitude: float, latitude: float, ring: Sequence[Sequence[float]]) -> bool:
    inside = False
    previous = ring[-1]
    for current in ring:
        x1, y1 = float(previous[0]), float(previous[1])
        x2, y2 = float(current[0]), float(current[1])
        if (y1 > latitude) != (y2 > latitude):
            intersection = (x2 - x1) * (latitude - y1) / (y2 - y1) + x1
            if longitude < intersection:
                inside = not inside
        previous = current
    return inside


def _point_in_boundary(
    longitude: float,
    latitude: float,
    boundary: Mapping[str, Any],
) -> bool:
    for polygon in boundary.get("polygons", []):
        outer = polygon.get("outer", [])
        if not _point_in_ring(longitude, latitude, outer):
            continue
        if any(
            _point_in_ring(longitude, latitude, hole)
            for hole in polygon.get("holes", [])
        ):
            continue
        return True
    return False


def _orientation(
    a: Sequence[float],
    b: Sequence[float],
    c: Sequence[float],
) -> float:
    return (float(b[0]) - float(a[0])) * (float(c[1]) - float(a[1])) - (
        float(b[1]) - float(a[1])
    ) * (float(c[0]) - float(a[0]))


def _segments_intersect(
    a: Sequence[float],
    b: Sequence[float],
    c: Sequence[float],
    d: Sequence[float],
) -> bool:
    return (
        _orientation(a, b, c) * _orientation(a, b, d) <= 0
        and _orientation(c, d, a) * _orientation(c, d, b) <= 0
        and max(min(a[0], b[0]), min(c[0], d[0]))
        <= min(max(a[0], b[0]), max(c[0], d[0]))
        and max(min(a[1], b[1]), min(c[1], d[1]))
        <= min(max(a[1], b[1]), max(c[1], d[1]))
    )


def _bounds_from_points(points: Iterable[Sequence[float]]) -> list[float] | None:
    coordinates = [
        (float(point[0]), float(point[1]))
        for point in points
        if isinstance(point, Sequence) and len(point) >= 2
    ]
    if not coordinates:
        return None
    return [
        min(point[0] for point in coordinates),
        min(point[1] for point in coordinates),
        max(point[0] for point in coordinates),
        max(point[1] for point in coordinates),
    ]


def _bounds_overlap(first: Sequence[float], second: Sequence[float]) -> bool:
    return not (
        first[2] < second[0]
        or first[0] > second[2]
        or first[3] < second[1]
        or first[1] > second[3]
    )


def _boundary_bounds(boundary: Mapping[str, Any]) -> list[float]:
    supplied = boundary.get("bounds")
    if isinstance(supplied, Sequence) and len(supplied) == 4:
        return [float(value) for value in supplied]
    points = [
        point
        for polygon in boundary.get("polygons", [])
        for ring in [polygon.get("outer", []), *polygon.get("holes", [])]
        for point in ring
    ]
    bounds = _bounds_from_points(points)
    if bounds is None:
        raise GridAtlasBuildError(f"Boundary has no coordinates: {boundary.get('label')}")
    return bounds


def _line_intersects_boundary(
    paths: Sequence[Sequence[Sequence[float]]],
    boundary: Mapping[str, Any],
) -> bool:
    boundary_bounds = _boundary_bounds(boundary)
    line_points = [point for path in paths for point in path]
    line_bounds = _bounds_from_points(line_points)
    if line_bounds is None or not _bounds_overlap(line_bounds, boundary_bounds):
        return False
    if any(
        _point_in_boundary(float(point[0]), float(point[1]), boundary)
        for point in line_points
    ):
        return True
    line_segments = [
        (path[index], path[index + 1])
        for path in paths
        for index in range(len(path) - 1)
    ]
    for polygon in boundary.get("polygons", []):
        for ring in [polygon.get("outer", []), *polygon.get("holes", [])]:
            for ring_index in range(len(ring) - 1):
                c, d = ring[ring_index], ring[ring_index + 1]
                if any(_segments_intersect(a, b, c, d) for a, b in line_segments):
                    return True
    return False


def tag_market_membership(
    lines: list[dict[str, Any]],
    substations: list[dict[str, Any]],
    plants: list[dict[str, Any]],
    market_boundaries: Sequence[Mapping[str, Any]],
) -> None:
    """Mutate normalized U.S. records with approximate spatial market tags."""

    try:
        import numpy as np
    except ImportError:
        for record in substations:
            record["markets"] = [
                boundary["id"]
                for boundary in market_boundaries
                if _point_in_boundary(
                    float(record["lon"]),
                    float(record["lat"]),
                    boundary,
                )
            ]
        for record in plants:
            record["markets"] = [
                boundary["id"]
                for boundary in market_boundaries
                if _point_in_boundary(
                    float(record["lon"]),
                    float(record["lat"]),
                    boundary,
                )
            ]
        for record in lines:
            record["markets"] = [
                boundary["id"]
                for boundary in market_boundaries
                if _line_intersects_boundary(record.get("paths", []), boundary)
            ]
        return

    def make_boundary_masker(
        coordinates: "np.ndarray[Any, Any]",
    ) -> Any:
        x_order = np.argsort(coordinates[:, 0]) if len(coordinates) else np.asarray([], dtype=int)
        y_order = np.argsort(coordinates[:, 1]) if len(coordinates) else np.asarray([], dtype=int)
        sorted_x = coordinates[x_order, 0] if len(coordinates) else np.asarray([])
        sorted_y = coordinates[y_order, 1] if len(coordinates) else np.asarray([])

        def bbox_candidates(
            west: float,
            south: float,
            east: float,
            north: float,
        ) -> "np.ndarray[Any, Any]":
            x_start = int(np.searchsorted(sorted_x, west, side="left"))
            x_end = int(np.searchsorted(sorted_x, east, side="right"))
            y_start = int(np.searchsorted(sorted_y, south, side="left"))
            y_end = int(np.searchsorted(sorted_y, north, side="right"))
            x_candidates = x_order[x_start:x_end]
            y_candidates = y_order[y_start:y_end]
            if len(x_candidates) <= len(y_candidates):
                candidates = x_candidates
                return candidates[
                    (coordinates[candidates, 1] >= south)
                    & (coordinates[candidates, 1] <= north)
                ]
            candidates = y_candidates
            return candidates[
                (coordinates[candidates, 0] >= west)
                & (coordinates[candidates, 0] <= east)
            ]

        def ring_mask(
            candidate_coordinates: "np.ndarray[Any, Any]",
            ring: Sequence[Sequence[float]],
        ) -> "np.ndarray[Any, Any]":
            contained = np.zeros(len(candidate_coordinates), dtype=bool)
            if len(ring) < 4 or not len(candidate_coordinates):
                return contained
            xs = candidate_coordinates[:, 0]
            ys = candidate_coordinates[:, 1]
            previous = ring[-1]
            for current in ring:
                x1, y1 = float(previous[0]), float(previous[1])
                x2, y2 = float(current[0]), float(current[1])
                crosses = (y1 > ys) != (y2 > ys)
                if y2 != y1:
                    intersection = (x2 - x1) * (ys - y1) / (y2 - y1) + x1
                    contained ^= crosses & (xs < intersection)
                previous = current
            return contained

        def boundary_mask(
            boundary: Mapping[str, Any],
        ) -> "np.ndarray[Any, Any]":
            mask = np.zeros(len(coordinates), dtype=bool)
            for polygon in boundary.get("polygons", []):
                outer = np.asarray(polygon.get("outer", []), dtype=float)
                if len(outer) < 4:
                    continue
                west, south = np.min(outer, axis=0)
                east, north = np.max(outer, axis=0)
                candidates = bbox_candidates(west, south, east, north)
                if not len(candidates):
                    continue
                polygon_mask = ring_mask(coordinates[candidates], outer)
                for raw_hole in polygon.get("holes", []):
                    hole = np.asarray(raw_hole, dtype=float)
                    if len(hole) >= 4:
                        polygon_mask &= ~ring_mask(coordinates[candidates], hole)
                mask[candidates] |= polygon_mask
            return mask

        return boundary_mask

    point_records = [*substations, *plants]
    point_coordinates = np.asarray(
        [[record["lon"], record["lat"]] for record in point_records],
        dtype=float,
    )
    point_boundary_mask = make_boundary_masker(point_coordinates)
    point_membership = [set() for _ in point_records]
    for boundary in market_boundaries:
        for index in np.flatnonzero(point_boundary_mask(boundary)):
            point_membership[int(index)].add(str(boundary["id"]))
    for record, memberships in zip(point_records, point_membership):
        record["markets"] = [
            market_id for market_id in MARKET_ORDER if market_id in memberships
        ]

    representative_points: list[list[float]] = []
    representative_line_indexes: list[int] = []
    for line_index, record in enumerate(lines):
        for path in record.get("paths", []):
            for index, point in enumerate(path):
                representative_points.append([float(point[0]), float(point[1])])
                representative_line_indexes.append(line_index)
                if index + 1 < len(path):
                    following = path[index + 1]
                    representative_points.append(
                        [
                            (float(point[0]) + float(following[0])) / 2,
                            (float(point[1]) + float(following[1])) / 2,
                        ]
                    )
                    representative_line_indexes.append(line_index)
    line_membership = [set() for _ in lines]
    line_coordinates = np.asarray(representative_points, dtype=float)
    line_boundary_mask = make_boundary_masker(line_coordinates)
    for boundary in market_boundaries:
        matched_points = np.flatnonzero(
            line_boundary_mask(boundary)
        )
        for point_index in matched_points:
            line_membership[representative_line_indexes[int(point_index)]].add(
                str(boundary["id"])
            )
    for record, memberships in zip(lines, line_membership):
        record["markets"] = [
            market_id for market_id in MARKET_ORDER if market_id in memberships
        ]


def _distance_to_segment(
    point: Sequence[float],
    start: Sequence[float],
    end: Sequence[float],
) -> float:
    px, py = float(point[0]), float(point[1])
    x1, y1 = float(start[0]), float(start[1])
    x2, y2 = float(end[0]), float(end[1])
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    scale = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (x1 + scale * dx), py - (y1 + scale * dy))


def simplify_path(path: Sequence[Sequence[float]], tolerance: float) -> list[list[float]]:
    """Douglas–Peucker line simplification in geographic display coordinates."""

    if len(path) <= 2 or tolerance <= 0:
        return [list(point) for point in path]
    keep = {0, len(path) - 1}
    stack = [(0, len(path) - 1)]
    while stack:
        start, end = stack.pop()
        maximum_distance = -1.0
        split_index = start
        for index in range(start + 1, end):
            distance = _distance_to_segment(path[index], path[start], path[end])
            if distance > maximum_distance:
                maximum_distance = distance
                split_index = index
        if maximum_distance > tolerance and start < split_index < end:
            keep.add(split_index)
            stack.append((start, split_index))
            stack.append((split_index, end))
    return [list(path[index]) for index in sorted(keep)]


def simplify_ring(
    ring: Sequence[Sequence[float]],
    tolerance: float,
) -> list[list[float]]:
    """Simplify a closed polygon ring without collapsing its shared endpoint."""

    if len(ring) < 4:
        return [list(point) for point in ring]
    open_ring = list(ring[:-1] if ring[0] == ring[-1] else ring)
    if len(open_ring) < 3 or tolerance <= 0:
        result = [list(point) for point in open_ring]
    else:
        anchor = open_ring[0]
        split_index = max(
            range(1, len(open_ring)),
            key=lambda index: math.hypot(
                float(open_ring[index][0]) - float(anchor[0]),
                float(open_ring[index][1]) - float(anchor[1]),
            ),
        )
        first_arc = simplify_path(open_ring[: split_index + 1], tolerance)
        second_arc = simplify_path(
            [*open_ring[split_index:], open_ring[0]],
            tolerance,
        )
        result = [*first_arc[:-1], *second_arc[:-1]]
    if len(result) < 3:
        bounds = _bounds_from_points(open_ring)
        if bounds is None:
            return []
        west, south, east, north = bounds
        result = [
            [west, south],
            [east, south],
            [east, north],
            [west, north],
        ]
    result.append(list(result[0]))
    return result


def _simplify_line_record(record: Mapping[str, Any], tolerance: float) -> dict[str, Any]:
    simplified = dict(record)
    simplified["paths"] = [
        path
        for path in (
            simplify_path(raw_path, tolerance)
            for raw_path in record.get("paths", [])
        )
        if len(path) >= 2
    ]
    return simplified


def _simplify_boundary(boundary: Mapping[str, Any], tolerance: float) -> dict[str, Any]:
    simplified = dict(boundary)
    polygons = []
    for polygon in boundary.get("polygons", []):
        outer = simplify_ring(polygon.get("outer", []), tolerance)
        holes = []
        for raw_hole in polygon.get("holes", []):
            hole = simplify_ring(raw_hole, tolerance)
            if len(hole) >= 4:
                holes.append(hole)
        if len(outer) >= 4:
            polygons.append({"outer": outer, "holes": holes})
    simplified["polygons"] = polygons
    simplified["bounds"] = _boundary_bounds(simplified)
    return simplified


def _region_bounds(
    boundaries: Sequence[Mapping[str, Any]],
    records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[float]:
    points: list[Sequence[float]] = []
    for boundary in boundaries:
        points.extend(
            point
            for polygon in boundary.get("polygons", [])
            for ring in [polygon.get("outer", []), *polygon.get("holes", [])]
            for point in ring
        )
    points.extend(
        [record["lon"], record["lat"]]
        for collection in ("substations", "power_plants")
        for record in records[collection]
    )
    points.extend(
        point
        for record in records["transmission_lines"]
        for path in record.get("paths", [])
        for point in path
    )
    bounds = _bounds_from_points(points)
    if bounds is None:
        raise GridAtlasBuildError("Unable to determine Grid Atlas region bounds")
    return [round(value, 4) for value in bounds]


def _map_view(bounds: Sequence[float]) -> tuple[dict[str, float], float]:
    west, south, east, north = bounds
    width = max(1.0, east - west)
    height = max(1.0, north - south)
    zoom = min(6.2, max(1.6, math.log2(360 / max(width, height * 1.6))))
    return (
        {"lon": round((west + east) / 2, 4), "lat": round((south + north) / 2, 4)},
        round(zoom, 2),
    )


def _line_vertex_count(lines: Iterable[Mapping[str, Any]]) -> int:
    return sum(
        len(path)
        for line in lines
        for path in line.get("paths", [])
    )


def _sort_payload_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(record) for record in records),
        key=lambda record: str(record.get("asset_id") or ""),
    )


def _write_gzip_json(path: Path, payload: Mapping[str, Any]) -> tuple[str, int]:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    compressed = gzip.compress(encoded, compresslevel=9, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compressed)
    return hashlib.sha256(compressed).hexdigest(), len(compressed)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_region(
    *,
    region_id: str,
    label: str,
    kind: str,
    countries: list[str],
    lines: Iterable[Mapping[str, Any]],
    substations: Iterable[Mapping[str, Any]],
    plants: Iterable[Mapping[str, Any]],
    boundaries: Iterable[Mapping[str, Any]],
    generated_at: str,
    default_minimum_voltage: int,
    include_unknown_voltage: bool,
    detail: str,
    tolerance: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    region_lines = _sort_payload_records(
        _simplify_line_record(record, tolerance)
        for record in lines
    )
    region_substations = _sort_payload_records(substations)
    region_plants = _sort_payload_records(plants)
    region_boundaries = [
        _simplify_boundary(boundary, max(tolerance, 0.01))
        for boundary in boundaries
    ]
    records = {
        "transmission_lines": region_lines,
        "substations": region_substations,
        "power_plants": region_plants,
    }
    bounds = _region_bounds(region_boundaries, records)
    center, zoom = _map_view(bounds)
    payload = {
        "schema_version": ATLAS_SCHEMA_VERSION,
        "region_id": region_id,
        "generated_at": generated_at,
        **records,
        "boundaries": region_boundaries,
        "errors": {},
    }
    metadata = {
        "id": region_id,
        "label": label,
        "kind": kind,
        "countries": countries,
        "counts": {
            collection: len(payload[collection])
            for collection in COLLECTIONS
        },
        "line_vertices": _line_vertex_count(region_lines),
        "bounds": bounds,
        "center": center,
        "zoom": zoom,
        "default_minimum_voltage": default_minimum_voltage,
        "include_unknown_voltage": include_unknown_voltage,
        "detail": detail,
    }
    return payload, metadata


def build_packaged_atlas(
    *,
    us_lines_path: Path,
    us_substations_path: Path,
    us_plants_path: Path,
    market_boundaries_path: Path,
    canvec_dir: Path,
    canada_plant_paths: Sequence[Path],
    output_dir: Path,
    nerc_boundaries_path: Path | None = None,
    osm_power_path: Path | None = None,
    osm_retrieved_at: str = "",
) -> dict[str, Any]:
    """Normalize approved sources and atomically replace the packaged store."""

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    raw_us_lines = _load_feature_collection(us_lines_path)
    raw_us_substations = _load_feature_collection(us_substations_path)
    raw_us_plants = _load_feature_collection(us_plants_path)
    us_lines = normalize_us_lines(raw_us_lines)
    us_substations = normalize_us_substations(raw_us_substations)
    us_plants = normalize_us_plants(raw_us_plants)
    normalized_us_counts = {
        "lines": len(us_lines),
        "substations": len(us_substations),
        "plants": len(us_plants),
    }
    raw_us_counts = {
        "lines": len(raw_us_lines),
        "substations": len(raw_us_substations),
        "plants": len(raw_us_plants),
    }
    if normalized_us_counts != raw_us_counts:
        differences = [
            f"{name}: {normalized_us_counts[name]:,} normalized of "
            f"{raw_us_counts[name]:,} source records"
            for name in raw_us_counts
            if normalized_us_counts[name] != raw_us_counts[name]
        ]
        raise GridAtlasBuildError(
            "Refusing to package dropped U.S. source records: "
            + "; ".join(differences)
        )
    market_boundaries = load_market_boundaries(market_boundaries_path)
    market_boundaries = [
        _simplify_boundary(boundary, 0.018)
        for boundary in market_boundaries
    ]
    nerc_boundaries = (
        [
            _simplify_boundary(boundary, 0.035)
            for boundary in normalize_boundaries(
                _load_feature_collection(nerc_boundaries_path),
                kind="nerc",
            )
        ]
        if nerc_boundaries_path
        else []
    )
    canada_lines = normalize_canvec_lines(canvec_dir)
    canada_substations = normalize_canvec_substations(canvec_dir)
    canada_plants = normalize_canada_plants(
        _load_feature_collection(path)
        for path in canada_plant_paths
    )
    osm_stats: dict[str, int] | None = None
    raw_osm_power: list[dict[str, Any]] = []
    osm_lines: list[dict[str, Any]] = []
    osm_substations: list[dict[str, Any]] = []
    if osm_power_path:
        if not osm_retrieved_at:
            raise GridAtlasBuildError(
                "OSM enrichment requires an explicit retrieval date"
            )
        raw_osm_power = _load_feature_collection(osm_power_path)
        osm_lines, osm_substations = normalize_osm_power(
            raw_osm_power,
            retrieved_at=osm_retrieved_at,
        )
        osm_stats = enrich_unknown_voltage_from_osm(
            [*us_lines, *canada_lines],
            [*us_substations, *canada_substations],
            osm_lines,
            osm_substations,
        )
    minimum_source_counts = {
        "us_lines": 70_000,
        "us_substations": 70_000,
        "us_plants": 10_000,
        "canada_lines": 12_000,
        "canada_substations": 4_000,
        "canada_plants": 1_000,
    }
    actual_source_counts = {
        "us_lines": len(us_lines),
        "us_substations": len(us_substations),
        "us_plants": len(us_plants),
        "canada_lines": len(canada_lines),
        "canada_substations": len(canada_substations),
        "canada_plants": len(canada_plants),
    }
    collapsed = [
        f"{name} {actual_source_counts[name]:,} < {minimum:,}"
        for name, minimum in minimum_source_counts.items()
        if actual_source_counts[name] < minimum
    ]
    if collapsed:
        raise GridAtlasBuildError(
            "Refusing to package incomplete source data: " + "; ".join(collapsed)
        )

    tag_market_membership(
        us_lines,
        us_substations,
        us_plants,
        market_boundaries,
    )
    boundary_by_id = {boundary["id"]: boundary for boundary in market_boundaries}

    overview_lines = [
        *(
            record
            for record in us_lines
            if record.get("voltage") is not None
            and float(record["voltage"]) >= 230
        ),
        *canada_lines,
    ]
    overview_substations = [
        *(
            record
            for record in us_substations
            if record.get("max_voltage") is not None
            and float(record["max_voltage"]) >= 230
        ),
        *canada_substations,
    ]
    overview_plants = [
        record
        for record in [*us_plants, *canada_plants]
        if record.get("capacity_mw") is not None
        and float(record["capacity_mw"]) >= 100
    ]

    build_specs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    build_specs.append(
        _build_region(
            region_id="all",
            label="All U.S. & Canada",
            kind="overview",
            countries=["US", "CA"],
            lines=overview_lines,
            substations=overview_substations,
            plants=overview_plants,
            boundaries=[*nerc_boundaries, *market_boundaries],
            generated_at=generated_at,
            default_minimum_voltage=0,
            include_unknown_voltage=True,
            detail=(
                "Performance-safe overview: U.S. lines/substations at 230 kV and above, "
                "plants at 100 MW and above, plus Canada’s CanVec reference lines and "
                "transformer stations."
            ),
            tolerance=0.035,
        )
    )
    for market_id in MARKET_ORDER:
        build_specs.append(
            _build_region(
                region_id=market_id,
                label=MARKET_LABELS[market_id],
                kind="iso-rto",
                countries=["US"],
                lines=(
                    record
                    for record in us_lines
                    if market_id in record.get("markets", [])
                ),
                substations=(
                    record
                    for record in us_substations
                    if market_id in record.get("markets", [])
                ),
                plants=(
                    record
                    for record in us_plants
                    if market_id in record.get("markets", [])
                ),
                boundaries=[boundary_by_id[market_id]],
                generated_at=generated_at,
                default_minimum_voltage=0,
                include_unknown_voltage=True,
                detail=(
                    "Approximate public ISO/RTO footprint; asset location does not prove "
                    "membership, ownership, or electrical connectivity."
                ),
                tolerance=0.012,
            )
        )
    build_specs.append(
        _build_region(
            region_id="canada",
            label="Canada",
            kind="country",
            countries=["CA"],
            lines=canada_lines,
            substations=canada_substations,
            plants=canada_plants,
            boundaries=[],
            generated_at=generated_at,
            default_minimum_voltage=0,
            include_unknown_voltage=True,
            detail=(
                "NRCan public cartographic reference. CanVec line and transformer source "
                "dates extend through 2015 and do not include voltage, owner, or topology."
            ),
            tolerance=0.018,
        )
    )

    output_parent = output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_parent)
    )
    backup = output_dir.with_name(f".{output_dir.name}.previous")
    try:
        region_metadata: list[dict[str, Any]] = []
        for payload, metadata in build_specs:
            relative_path = Path("regions") / f"{metadata['id']}.json.gz"
            sha256, gzip_bytes = _write_gzip_json(staging / relative_path, payload)
            metadata["artifact"] = relative_path.as_posix()
            metadata["sha256"] = sha256
            metadata["gzip_bytes"] = gzip_bytes
            region_metadata.append(metadata)
        manifest = {
            "schema_version": ATLAS_SCHEMA_VERSION,
            "generated_at": generated_at,
            "title": "NERC U.S. & Canada Public Grid Atlas",
            "scope": "United States and Canada",
            "default_region": "ercot",
            "regions": region_metadata,
            "sources": {
                "us_transmission_lines": HIFLD_TRANSMISSION_URL,
                "us_substations": HIFLD_SUBSTATION_URL,
                "us_power_plants": EIA_POWER_PLANT_URL,
                "canada_lines_and_transformers": CANVEC_DATASET_URL,
                "canada_power_plants": NACEI_MAP_URL,
                "iso_rto_boundaries": ISO_RTO_SOURCE_URL,
                "nerc_reference": NERC_REFERENCE_URL,
                **(
                    {"osm_secondary_enrichment": OPENSTREETMAP_URL}
                    if osm_power_path
                    else {}
                ),
            },
            "source_notes": {
                "us_infrastructure": (
                    "HIFLD/EIA public reference layers; source dates are carried on "
                    "individual records."
                ),
                "canada_infrastructure": (
                    "CanVec cartographic line/transformer dates extend through 2015; "
                    "NACEI plant reference period is August 2017."
                ),
                "iso_rto_boundaries": (
                    "Approximate historical public footprint snapshot with feature source "
                    "date August 28, 2017."
                ),
                "nerc_boundaries": (
                    "Approximate contiguous-U.S. reference polygons only; Canadian NERC "
                    "regional boundaries are not included."
                ),
                **(
                    {
                        "osm_secondary_enrichment": (
                            "OSM may fill only missing voltage through conservative spatial "
                            "matching. Suggested values retain government geometry and carry "
                            "source, confidence, and retrieval-date fields."
                        )
                    }
                    if osm_power_path
                    else {}
                ),
            },
            "source_counts": actual_source_counts,
            "source_artifacts": {
                "us_transmission_lines": {
                    "sha256": _file_sha256(us_lines_path),
                    "records": len(raw_us_lines),
                    "source_data_last_edit": "2022-09-20",
                    "download_scope": "Complete U.S. public feature layer",
                },
                "us_substations": {
                    "sha256": _file_sha256(us_substations_path),
                    "records": len(raw_us_substations),
                    "source_item_modified": "2021-02-25",
                    "download_scope": "Complete U.S. public feature layer",
                },
                "us_power_plants": {
                    "sha256": _file_sha256(us_plants_path),
                    "records": len(raw_us_plants),
                    "reporting_period": "2025-02",
                    "download_scope": "Complete U.S. public feature layer",
                },
                "canvec_power_lines": {
                    "sha256": _file_sha256(canvec_dir / "power_line_1.shp"),
                    "records": len(canada_lines),
                    "source_dates_through": "2015-09",
                },
                "canvec_transformer_stations": {
                    "point_sha256": _file_sha256(
                        canvec_dir / "transformer_station_0.shp"
                    ),
                    "polygon_sha256": _file_sha256(
                        canvec_dir / "transformer_station_2.shp"
                    ),
                    "records": len(canada_substations),
                    "source_dates_through": "2015",
                },
                "canada_power_plants": {
                    "sha256": [
                        _file_sha256(path)
                        for path in canada_plant_paths
                    ],
                    "source_records": sum(
                        len(_load_feature_collection(path))
                        for path in canada_plant_paths
                    ),
                    "deduplicated_records": len(canada_plants),
                    "reference_period": "2017-08",
                },
                "iso_rto_boundaries": {
                    "sha256": _file_sha256(market_boundaries_path),
                    "features": len(market_boundaries),
                    "feature_source_date": "2017-08-28",
                    "validation_date": "2018-06-04",
                },
                "nerc_reference_boundaries": {
                    "sha256": (
                        _file_sha256(nerc_boundaries_path)
                        if nerc_boundaries_path
                        else ""
                    ),
                    "features": len(nerc_boundaries),
                    "coverage": "Contiguous United States only",
                    "layer_data_edit": "2022-07-12",
                },
                **(
                    {
                        "osm_secondary_enrichment": {
                            "sha256": _file_sha256(osm_power_path),
                            "source_features": len(raw_osm_power),
                            "normalized_lines_with_voltage": len(osm_lines),
                            "normalized_substations_with_voltage": len(
                                osm_substations
                            ),
                            "retrieved_at": osm_retrieved_at,
                            "matches": osm_stats,
                            "policy": "fill-unknown-only",
                        }
                    }
                    if osm_power_path and osm_stats is not None
                    else {}
                ),
            },
            "disclaimer": (
                "Public reference infrastructure and approximate regional footprints; "
                "not a NERC, ISO/RTO, ERCOT, planning, operating, or real-time topology model."
            ),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if backup.exists():
            shutil.rmtree(backup)
        if output_dir.exists():
            output_dir.replace(backup)
        staging.replace(output_dir)
        if backup.exists():
            shutil.rmtree(backup)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if backup.exists() and not output_dir.exists():
            backup.replace(output_dir)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build checked-in U.S.–Canada Grid Atlas region shards."
    )
    parser.add_argument("--us-lines", type=Path, required=True)
    parser.add_argument("--us-substations", type=Path, required=True)
    parser.add_argument("--us-plants", type=Path, required=True)
    parser.add_argument("--market-boundaries", type=Path, required=True)
    parser.add_argument("--nerc-boundaries", type=Path)
    parser.add_argument(
        "--osm-power",
        type=Path,
        help="Optional verified OSM GeoJSON extract used only to enrich unknown voltage",
    )
    parser.add_argument(
        "--osm-retrieved-at",
        default="",
        help="Required ISO date for an --osm-power extract, for example 2026-07-23",
    )
    parser.add_argument("--canvec-dir", type=Path, required=True)
    parser.add_argument("--canada-plants", type=Path, action="append", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("grid_atlas_data"),
    )
    args = parser.parse_args()
    manifest = build_packaged_atlas(
        us_lines_path=args.us_lines,
        us_substations_path=args.us_substations,
        us_plants_path=args.us_plants,
        market_boundaries_path=args.market_boundaries,
        nerc_boundaries_path=args.nerc_boundaries,
        canvec_dir=args.canvec_dir,
        canada_plant_paths=args.canada_plants,
        output_dir=args.output,
        osm_power_path=args.osm_power,
        osm_retrieved_at=args.osm_retrieved_at,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "generated_at": manifest["generated_at"],
                "regions": {
                    region["id"]: region["counts"]
                    for region in manifest["regions"]
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
