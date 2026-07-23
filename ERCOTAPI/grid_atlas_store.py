"""Validated local artifact store for the U.S.–Canada public Grid Atlas.

The dashboard imports this module at runtime.  It deliberately contains no
network client: bulk public GIS retrieval belongs to the offline builder, not
to a Streamlit session.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ATLAS_SCHEMA_VERSION = 1
PACKAGED_ATLAS_ROOT = Path(__file__).with_name("grid_atlas_data")
PACKAGED_ATLAS_MANIFEST = PACKAGED_ATLAS_ROOT / "manifest.json"
ATLAS_COLLECTIONS = ("transmission_lines", "substations", "power_plants")


class GridAtlasStoreError(RuntimeError):
    """Raised when a packaged Atlas artifact is absent or fails validation."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GridAtlasStoreError(f"Unable to read Grid Atlas manifest {path}: {exc}") from exc


def _safe_artifact_path(root: Path, relative_path: Any) -> Path:
    text = str(relative_path or "").strip()
    if not text:
        raise GridAtlasStoreError("Grid Atlas region is missing its artifact path")
    candidate = (root / text).resolve()
    resolved_root = root.resolve()
    if not candidate.is_relative_to(resolved_root):
        raise GridAtlasStoreError("Grid Atlas artifact path escapes its package directory")
    return candidate


def _region_index(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    regions = manifest.get("regions")
    if not isinstance(regions, list) or not regions:
        raise GridAtlasStoreError("Grid Atlas manifest has no regions")
    indexed: dict[str, dict[str, Any]] = {}
    for raw_region in regions:
        if not isinstance(raw_region, dict):
            raise GridAtlasStoreError("Grid Atlas manifest contains an invalid region")
        region_id = str(raw_region.get("id") or "").strip()
        label = str(raw_region.get("label") or "").strip()
        if not region_id or not label:
            raise GridAtlasStoreError("Grid Atlas region requires an id and label")
        if region_id in indexed:
            raise GridAtlasStoreError(f"Duplicate Grid Atlas region id: {region_id}")
        indexed[region_id] = raw_region
    return indexed


def load_grid_atlas_manifest(
    path: Path = PACKAGED_ATLAS_MANIFEST,
    *,
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    """Load and validate the small checked-in Atlas manifest."""

    manifest = _read_json(path)
    if not isinstance(manifest, dict):
        raise GridAtlasStoreError("Grid Atlas manifest is not a JSON object")
    if manifest.get("schema_version") != ATLAS_SCHEMA_VERSION:
        raise GridAtlasStoreError("Grid Atlas manifest schema is unsupported")
    indexed = _region_index(manifest)
    default_region = str(manifest.get("default_region") or "")
    if default_region not in indexed:
        raise GridAtlasStoreError("Grid Atlas default region is not defined")

    if verify_artifacts:
        root = path.parent
        for region in indexed.values():
            artifact_path = _safe_artifact_path(root, region.get("artifact"))
            if not artifact_path.is_file():
                raise GridAtlasStoreError(
                    f"Grid Atlas artifact is missing for {region['label']}: {artifact_path}"
                )
            expected_size = region.get("gzip_bytes")
            if expected_size is not None and artifact_path.stat().st_size != int(expected_size):
                raise GridAtlasStoreError(
                    f"Grid Atlas artifact size does not match {region['label']}"
                )
    return manifest


def grid_atlas_regions(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return regions in display order after validating their identities."""

    indexed = _region_index(manifest)
    return [indexed[str(region["id"])] for region in manifest["regions"]]


def grid_atlas_region(
    manifest: Mapping[str, Any],
    region_id: str,
) -> dict[str, Any]:
    """Return one region definition or raise a user-readable error."""

    indexed = _region_index(manifest)
    try:
        return indexed[region_id]
    except KeyError as exc:
        raise GridAtlasStoreError(f"Unknown Grid Atlas region: {region_id}") from exc


def _validate_region_payload(
    payload: Any,
    *,
    region: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise GridAtlasStoreError("Grid Atlas region artifact is not a JSON object")
    if payload.get("schema_version") != ATLAS_SCHEMA_VERSION:
        raise GridAtlasStoreError("Grid Atlas region artifact schema is unsupported")
    if payload.get("region_id") != region.get("id"):
        raise GridAtlasStoreError("Grid Atlas artifact belongs to a different region")
    expected_counts = region.get("counts")
    if not isinstance(expected_counts, dict):
        raise GridAtlasStoreError("Grid Atlas region is missing record counts")
    for collection in ATLAS_COLLECTIONS:
        records = payload.get(collection)
        if not isinstance(records, list):
            raise GridAtlasStoreError(f"Grid Atlas artifact is missing {collection}")
        if len(records) != int(expected_counts.get(collection, -1)):
            raise GridAtlasStoreError(
                f"Grid Atlas record count does not match {collection}"
            )
    boundaries = payload.get("boundaries")
    if boundaries is not None and not isinstance(boundaries, list):
        raise GridAtlasStoreError("Grid Atlas boundaries are invalid")
    if payload.get("errors") not in ({}, None):
        raise GridAtlasStoreError("Grid Atlas artifact contains source errors")
    return payload


def load_packaged_grid_region(
    region_id: str,
    manifest_path: Path = PACKAGED_ATLAS_MANIFEST,
    *,
    verify_hash: bool = True,
) -> dict[str, Any]:
    """Open one local gzip shard without any external request."""

    manifest = load_grid_atlas_manifest(manifest_path, verify_artifacts=False)
    region = grid_atlas_region(manifest, region_id)
    artifact_path = _safe_artifact_path(manifest_path.parent, region.get("artifact"))
    try:
        compressed = artifact_path.read_bytes()
    except OSError as exc:
        raise GridAtlasStoreError(
            f"Unable to read Grid Atlas artifact for {region['label']}: {exc}"
        ) from exc
    if verify_hash:
        expected_hash = str(region.get("sha256") or "")
        actual_hash = hashlib.sha256(compressed).hexdigest()
        if not expected_hash or actual_hash != expected_hash:
            raise GridAtlasStoreError(
                f"Grid Atlas artifact hash does not match {region['label']}"
            )
    try:
        payload = json.loads(gzip.decompress(compressed).decode("utf-8"))
    except (OSError, EOFError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GridAtlasStoreError(
            f"Unable to decompress Grid Atlas artifact for {region['label']}: {exc}"
        ) from exc
    return _validate_region_payload(payload, region=region)


def packaged_grid_region_bytes(
    region_id: str,
    manifest_path: Path = PACKAGED_ATLAS_MANIFEST,
) -> int:
    """Return compressed bytes without opening the artifact."""

    manifest = load_grid_atlas_manifest(manifest_path, verify_artifacts=False)
    region = grid_atlas_region(manifest, region_id)
    return int(region.get("gzip_bytes") or 0)
