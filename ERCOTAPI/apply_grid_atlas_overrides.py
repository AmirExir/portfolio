"""Apply reviewed facility overrides to already-built Grid Atlas shards."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ERCOTAPI.grid_atlas_builder import (
    enrich_unknown_voltage_from_osm,
    normalize_osm_power,
)
from ERCOTAPI.grid_atlas_store import PACKAGED_ATLAS_ROOT


OVERRIDES_PATH = Path(__file__).with_name(
    "grid_atlas_autotransformer_overrides.json"
)
OSM_AUTOTRANSFORMERS_PATH = (
    Path(__file__).with_name("sources")
    / "official"
    / "osm_north_america_autotransformers_2026-07-24.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return payload


def _write_gzip(path: Path, payload: dict[str, Any]) -> tuple[str, int]:
    encoded = (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")
    compressed = gzip.compress(encoded, compresslevel=9, mtime=0)
    path.write_bytes(compressed)
    return hashlib.sha256(compressed).hexdigest(), len(compressed)


def _osm_features(payload: dict[str, Any]) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for element in payload.get("elements", []):
        if (
            not isinstance(element, dict)
            or element.get("type") != "node"
            or element.get("lat") is None
            or element.get("lon") is None
        ):
            continue
        properties = dict(element.get("tags") or {})
        properties["@id"] = f"node/{element['id']}"
        features.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": {
                    "type": "Point",
                    "coordinates": [element["lon"], element["lat"]],
                },
            }
        )
    return features


def apply_overrides(atlas_root: Path = PACKAGED_ATLAS_ROOT) -> dict[str, int]:
    override_payload = _read_json(OVERRIDES_PATH)
    overrides = {
        str(record["asset_id"]): record
        for record in override_payload.get("records", [])
    }
    osm_payload = _read_json(OSM_AUTOTRANSFORMERS_PATH)
    osm_timestamp = str(
        (osm_payload.get("osm3s") or {}).get("timestamp_osm_base")
        or "2026-07-24"
    )
    _, osm_autotransformers = normalize_osm_power(
        _osm_features(osm_payload),
        retrieved_at=osm_timestamp,
    )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{atlas_root.name}-", dir=atlas_root.parent)
    )
    backup = atlas_root.with_name(f".{atlas_root.name}.previous")
    matched_ids: set[str] = set()
    matched_osm_ids: set[str] = set()
    try:
        shutil.copytree(atlas_root, staging, dirs_exist_ok=True)
        manifest = _read_json(staging / "manifest.json")
        for region in manifest.get("regions", []):
            artifact = staging / str(region["artifact"])
            with gzip.open(artifact, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            enrich_unknown_voltage_from_osm(
                [],
                payload.get("substations", []),
                [],
                osm_autotransformers,
            )
            for substation in payload.get("substations", []):
                matched_osm_ids.update(
                    str(value)
                    for value in substation.get(
                        "autotransformer_osm_ids", []
                    )
                )
                override = overrides.get(str(substation.get("asset_id") or ""))
                if not override:
                    continue
                matched_ids.add(str(override["asset_id"]))
                substation.update(
                    {
                        "autotransformer": True,
                        "autotransformer_source": "ERCOT",
                        "autotransformer_source_url": override["source_url"],
                        "autotransformer_source_date": override["source_date"],
                        "autotransformer_match_status": "ERCOT-confirmed",
                        "autotransformer_voltages": override[
                            "autotransformer_voltages"
                        ],
                        "autotransformer_status": override["status"],
                        "autotransformer_reference": override["source"],
                    }
                )
            sha256, gzip_bytes = _write_gzip(artifact, payload)
            region["sha256"] = sha256
            region["gzip_bytes"] = gzip_bytes
        missing = sorted(set(overrides) - matched_ids)
        if missing:
            raise RuntimeError(
                "Reviewed autotransformer overrides matched no packaged asset: "
                + ", ".join(missing)
            )
        manifest.setdefault("sources", {})[
            "ercot_autotransformer_overrides"
        ] = "ERCOT public planning documents"
        manifest["sources"][
            "osm_autotransformers"
        ] = "https://wiki.openstreetmap.org/wiki/Tag:power%3Dtransformer"
        manifest.setdefault("source_artifacts", {})[
            "ercot_autotransformer_overrides"
        ] = {
            "sha256": hashlib.sha256(OVERRIDES_PATH.read_bytes()).hexdigest(),
            "records": len(overrides),
            "policy": "explicit ERCOT references only",
        }
        manifest["source_artifacts"]["osm_autotransformers"] = {
            "sha256": hashlib.sha256(
                OSM_AUTOTRANSFORMERS_PATH.read_bytes()
            ).hexdigest(),
            "source_records": len(osm_payload.get("elements", [])),
            "normalized_records": len(osm_autotransformers),
            "matched_osm_records": len(matched_osm_ids),
            "retrieved_at": osm_timestamp,
            "schemas": [
                "power=transformer + windings:auto=yes",
                "power=transformer + transformer=auto (deprecated)",
            ],
            "policy": "explicit autotransformer tags only",
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if backup.exists():
            shutil.rmtree(backup)
        atlas_root.replace(backup)
        staging.replace(atlas_root)
        shutil.rmtree(backup)
        return {
            "overrides": len(overrides),
            "matched_assets": len(matched_ids),
            "osm_records": len(osm_autotransformers),
            "matched_osm_records": len(matched_osm_ids),
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if backup.exists() and not atlas_root.exists():
            backup.replace(atlas_root)
        raise


if __name__ == "__main__":
    print(json.dumps(apply_overrides(), sort_keys=True))
