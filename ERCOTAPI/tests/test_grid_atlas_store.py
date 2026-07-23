"""Tests for the offline-built U.S.–Canada Grid Atlas store."""

from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ERCOTAPI.grid_atlas_builder import (
    _point_in_boundary,
    normalize_boundaries,
    normalize_canada_plants,
    simplify_path,
)
from ERCOTAPI.grid_atlas_store import (
    ATLAS_SCHEMA_VERSION,
    PACKAGED_ATLAS_MANIFEST,
    GridAtlasStoreError,
    grid_atlas_regions,
    load_grid_atlas_manifest,
    load_packaged_grid_region,
)


def _write_test_store(root: Path) -> Path:
    payload = {
        "schema_version": ATLAS_SCHEMA_VERSION,
        "region_id": "all",
        "transmission_lines": [{"asset_id": "line:1", "paths": [[[-1, 0], [1, 0]]]}],
        "substations": [{"asset_id": "sub:1", "lat": 0, "lon": 0}],
        "power_plants": [{"asset_id": "plant:1", "lat": 0, "lon": 0}],
        "boundaries": [],
        "errors": {},
    }
    compressed = gzip.compress(
        (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"),
        mtime=0,
    )
    artifact = root / "regions" / "all.json.gz"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(compressed)
    manifest = {
        "schema_version": ATLAS_SCHEMA_VERSION,
        "default_region": "all",
        "regions": [
            {
                "id": "all",
                "label": "All U.S. & Canada",
                "artifact": "regions/all.json.gz",
                "gzip_bytes": len(compressed),
                "sha256": hashlib.sha256(compressed).hexdigest(),
                "counts": {
                    "transmission_lines": 1,
                    "substations": 1,
                    "power_plants": 1,
                },
            }
        ],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


class GridAtlasStoreTests(unittest.TestCase):
    def test_local_region_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = _write_test_store(Path(temporary))

            manifest = load_grid_atlas_manifest(manifest_path)
            payload = load_packaged_grid_region("all", manifest_path)

        self.assertEqual(manifest["default_region"], "all")
        self.assertEqual(payload["region_id"], "all")
        self.assertEqual(len(payload["transmission_lines"]), 1)

    def test_store_rejects_tampered_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = _write_test_store(root)
            with (root / "regions" / "all.json.gz").open("ab") as handle:
                handle.write(b"tampered")

            with self.assertRaisesRegex(GridAtlasStoreError, "hash"):
                load_packaged_grid_region("all", manifest_path)

    def test_manifest_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = _write_test_store(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["regions"][0]["artifact"] = "../outside.json.gz"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(GridAtlasStoreError, "escapes"):
                load_grid_atlas_manifest(manifest_path)

    def test_boundary_normalization_and_point_membership(self):
        names = [
            "ELECTRIC RELIABILITY COUNCIL OF TEXAS, INC.",
            "CALIFORNIA INDEPENDENT SYSTEM OPERATOR",
            "ISO NEW ENGLAND INC.",
            "MIDCONTINENT INDEPENDENT TRANSMISSION SYSTEM OPERATOR, INC.",
            "NEW YORK INDEPENDENT SYSTEM OPERATOR",
            "PJM INTERCONNECTION, LLC",
            "SOUTHWEST POWER POOL",
        ]
        features = [
            {
                "type": "Feature",
                "properties": {"NAME": name},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[-2, -1], [2, -1], [2, 1], [-2, 1], [-2, -1]]
                    ],
                },
            }
            for name in names
        ]

        boundaries = normalize_boundaries(features, kind="market")

        ercot = next(boundary for boundary in boundaries if boundary["id"] == "ercot")
        self.assertTrue(_point_in_boundary(0, 0, ercot))

    def test_point_membership_respects_hole(self):
        boundary = {
            "polygons": [
                {
                    "outer": [[-2, -2], [2, -2], [2, 2], [-2, 2], [-2, -2]],
                    "holes": [[[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]],
                }
            ]
        }
        self.assertTrue(_point_in_boundary(1.5, 0, boundary))
        self.assertFalse(_point_in_boundary(0, 0, boundary))

    def test_simplify_path_preserves_endpoints(self):
        simplified = simplify_path(
            [[0, 0], [0.25, 0.001], [0.5, 0], [1, 0]],
            tolerance=0.01,
        )
        self.assertEqual(simplified, [[0, 0], [1, 0]])

    def test_canadian_plant_layers_deduplicate_overlap(self):
        feature = {
            "type": "Feature",
            "properties": {
                "OBJECTID": 1,
                "Country": "Canada",
                "Facility": "Example Hydro",
                "StateProv": "Ontario",
                "Total_MW": 250,
                "PrimSource": "Hydro",
                "Period": 201708,
            },
            "geometry": {"type": "Point", "coordinates": [-80, 45]},
        }

        plants = normalize_canada_plants([[feature], [{**feature}]])

        self.assertEqual(len(plants), 1)
        self.assertEqual(plants[0]["capacity_mw"], 250)
        self.assertEqual(plants[0]["period"], "2017-08")

    def test_shipped_atlas_has_all_market_and_country_shards(self):
        manifest = load_grid_atlas_manifest(PACKAGED_ATLAS_MANIFEST)
        self.assertEqual(manifest["default_region"], "ercot")
        self.assertTrue(
            all(
                region["include_unknown_voltage"]
                for region in grid_atlas_regions(manifest)
            )
        )
        self.assertTrue(
            all(
                region["default_minimum_voltage"] == 0
                for region in grid_atlas_regions(manifest)
            )
        )
        region_ids = {
            region["id"]
            for region in grid_atlas_regions(manifest)
        }

        self.assertEqual(
            region_ids,
            {
                "all",
                "ercot",
                "miso",
                "pjm",
                "caiso",
                "spp",
                "nyiso",
                "iso-ne",
                "canada",
            },
        )
        self.assertEqual(
            manifest["source_counts"],
            {
                "us_lines": 74_553,
                "us_substations": 75_328,
                "us_plants": 13_446,
                "canada_lines": 13_009,
                "canada_substations": 4_538,
                "canada_plants": 1_125,
            },
        )

    def test_shipped_regions_validate_and_stay_within_render_budgets(self):
        manifest = load_grid_atlas_manifest(PACKAGED_ATLAS_MANIFEST)

        for region in grid_atlas_regions(manifest):
            with self.subTest(region=region["id"]):
                payload = load_packaged_grid_region(
                    region["id"],
                    PACKAGED_ATLAS_MANIFEST,
                )
                self.assertLess(region["gzip_bytes"], 2_500_000)
                self.assertLess(region["line_vertices"], 60_000)
                self.assertTrue(payload["transmission_lines"])
                self.assertTrue(payload["substations"])
                self.assertTrue(payload["power_plants"])
                self.assertFalse(
                    any(
                        record.get("voltage") is not None
                        and float(record["voltage"]) < 0
                        for record in payload["transmission_lines"]
                    )
                )


if __name__ == "__main__":
    unittest.main()
