"""Tests for the public ArcGIS adapters used by the ERCOT Grid Atlas."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ERCOTAPI.grid_atlas import (
    PACKAGED_SNAPSHOT_PATH,
    _feature_pages,
    clean_number,
    fetch_power_plants,
    fetch_substations,
    grid_atlas_change_summary,
    grid_atlas_content_hash,
    load_packaged_texas_grid,
    load_public_texas_grid,
    validate_grid_atlas_snapshot,
    write_grid_atlas_snapshot,
)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _Session:
    def __init__(self, get_payloads=None, post_payloads=None):
        self.get_payloads = list(get_payloads or [])
        self.post_payloads = list(post_payloads or [])
        self.headers = {}
        self.get_calls = []
        self.post_calls = []

    def get(self, url, params=None, timeout=None):
        self.get_calls.append((url, params, timeout))
        return _Response(self.get_payloads.pop(0))

    def post(self, url, data=None, timeout=None):
        self.post_calls.append((url, data, timeout))
        return _Response(self.post_payloads.pop(0))

    def close(self):
        return None


class GridAtlasTests(unittest.TestCase):
    def test_clean_number_discards_hifld_sentinel(self):
        self.assertIsNone(clean_number(-999999))
        self.assertIsNone(clean_number("-999999"))
        self.assertIsNone(clean_number("not available"))
        self.assertEqual(clean_number("345"), 345.0)

    def test_feature_pages_honors_transfer_limit(self):
        session = _Session(
            get_payloads=[
                {
                    "features": [{"attributes": {"id": 1}}],
                    "exceededTransferLimit": True,
                },
                {
                    "features": [{"attributes": {"id": 2}}],
                    "exceededTransferLimit": False,
                },
            ]
        )

        features = _feature_pages(
            session,
            "https://example.test/query",
            base_params={"where": "1=1"},
            max_records=2,
        )

        self.assertEqual(len(features), 2)
        self.assertEqual(session.get_calls[1][1]["resultOffset"], 1)

    def test_substation_parser_does_not_display_missing_voltage(self):
        session = _Session(
            get_payloads=[
                {
                    "features": [
                        {
                            "attributes": {
                                "NAME": "TAP161924",
                                "CITY": "Joaquin",
                                "MAX_VOLT": -999999,
                                "MIN_VOLT": -999999,
                            },
                            "geometry": {"x": -94.0384, "y": 31.9597},
                        }
                    ],
                    "exceededTransferLimit": False,
                }
            ]
        )

        records = fetch_substations(session)

        self.assertEqual(records[0]["name"], "TAP161924")
        self.assertIsNone(records[0]["max_voltage"])
        self.assertIsNone(records[0]["min_voltage"])

    def test_power_plant_parser_accepts_string_battery_capacity(self):
        session = _Session(
            get_payloads=[
                {
                    "features": [
                        {
                            "attributes": {
                                "Plant_Name": "Storage One",
                                "PrimSource": "batteries",
                                "Install_MW": 250,
                                "Bat_MW": "225.5",
                                "Period": "202502",
                                "Latitude": 31.1,
                                "Longitude": -98.2,
                            }
                        }
                    ],
                    "exceededTransferLimit": False,
                }
            ]
        )

        records = fetch_power_plants(session)

        self.assertEqual(records[0]["capacity_mw"], 250)
        self.assertEqual(records[0]["battery_mw"], 225.5)
        self.assertEqual(records[0]["period"], "2025-02")

    def test_partial_source_failure_is_preserved(self):
        class PartialSession(_Session):
            def get(self, url, params=None, timeout=None):
                if "USA_States" in url:
                    raise RuntimeError("boundary unavailable")
                return _Response({"features": [], "exceededTransferLimit": False})

        payload = load_public_texas_grid(PartialSession())

        self.assertIn("transmission_lines", payload["errors"])
        self.assertEqual(payload["substations"], [])
        self.assertEqual(payload["power_plants"], [])

    def test_snapshot_round_trip_never_needs_a_network_session(self):
        live_payload = {
            "fetched_at": "2026-07-23T12:00:00+00:00",
            "transmission_lines": [{"id": "1", "paths": [[[-99, 31], [-98, 32]]]}],
            "substations": [{"name": "A", "lat": 31, "lon": -99}],
            "power_plants": [{"name": "Plant", "lat": 31, "lon": -99}],
            "errors": {},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "atlas.json.gz"
            minimum_counts = {
                "transmission_lines": 1,
                "substations": 1,
                "power_plants": 1,
            }

            snapshot = write_grid_atlas_snapshot(
                live_payload,
                path,
                minimum_counts=minimum_counts,
            )
            loaded = load_packaged_texas_grid(
                path,
                minimum_counts=minimum_counts,
            )

        self.assertEqual(loaded, snapshot)
        self.assertEqual(loaded["generated_at"], live_payload["fetched_at"])

    def test_snapshot_validation_rejects_partial_payload(self):
        with self.assertRaisesRegex(RuntimeError, "missing substations"):
            validate_grid_atlas_snapshot(
                {
                    "snapshot_schema_version": 1,
                    "transmission_lines": [],
                    "power_plants": [],
                    "errors": {},
                }
            )

    def test_snapshot_writer_rejects_collapsed_source_counts(self):
        payload = {
            "fetched_at": "2026-07-23T12:00:00+00:00",
            "transmission_lines": [],
            "substations": [],
            "power_plants": [],
            "errors": {},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "atlas.json.gz"

            with self.assertRaisesRegex(RuntimeError, "expected at least"):
                write_grid_atlas_snapshot(payload, path)

            self.assertFalse(path.exists())

    def test_shipped_snapshot_has_all_validated_texas_layers(self):
        payload = load_packaged_texas_grid(PACKAGED_SNAPSHOT_PATH)

        self.assertEqual(len(payload["transmission_lines"]), 5_567)
        self.assertEqual(len(payload["substations"]), 4_939)
        self.assertEqual(len(payload["power_plants"]), 921)
        self.assertEqual(
            payload["content_sha256"],
            grid_atlas_content_hash(payload),
        )
        self.assertFalse(payload["errors"])
        self.assertFalse(
            any(
                record.get("max_voltage") is not None
                and record["max_voltage"] < 0
                for record in payload["substations"]
            )
        )

    def test_source_update_comparison_ignores_check_time_but_detects_content(self):
        packaged = load_packaged_texas_grid(PACKAGED_SNAPSHOT_PATH)
        same_records = dict(packaged)
        same_records["fetched_at"] = "2099-01-01T00:00:00+00:00"
        same_records["power_plants"] = list(
            reversed(packaged["power_plants"])
        )

        unchanged = grid_atlas_change_summary(packaged, same_records)

        self.assertFalse(unchanged["changed"])

        changed_records = dict(same_records)
        changed_plant = dict(packaged["power_plants"][0])
        changed_plant["name"] = f"{changed_plant['name']} updated"
        changed_records["power_plants"] = [
            changed_plant,
            *packaged["power_plants"][1:],
        ]

        changed = grid_atlas_change_summary(packaged, changed_records)

        self.assertTrue(changed["changed"])
        self.assertEqual(changed["changed_collections"], ["power_plants"])
        self.assertEqual(changed["counts"]["power_plants"]["delta"], 0)

    def test_source_update_comparison_rejects_moderate_count_collapse(self):
        packaged = load_packaged_texas_grid(PACKAGED_SNAPSHOT_PATH)
        incomplete = dict(packaged)
        incomplete["transmission_lines"] = packaged["transmission_lines"][:5_000]

        with self.assertRaisesRegex(RuntimeError, "expected at least 5,"):
            grid_atlas_change_summary(packaged, incomplete)

    def test_source_update_comparison_rejects_unrelated_identifiers(self):
        packaged = load_packaged_texas_grid(PACKAGED_SNAPSHOT_PATH)
        unrelated = dict(packaged)
        unrelated["power_plants"] = [
            {**record, "plant_code": f"new-{index}", "object_id": f"new-{index}"}
            for index, record in enumerate(packaged["power_plants"])
        ]

        with self.assertRaisesRegex(RuntimeError, "stable power plants identifiers"):
            grid_atlas_change_summary(packaged, unrelated)

    def test_source_update_comparison_requires_identifier_coverage(self):
        packaged = load_packaged_texas_grid(PACKAGED_SNAPSHOT_PATH)
        unidentified = dict(packaged)
        unidentified["power_plants"] = [
            {**record, "plant_code": None, "object_id": None}
            for record in packaged["power_plants"]
        ]

        with self.assertRaisesRegex(RuntimeError, "enough stable power plants"):
            grid_atlas_change_summary(packaged, unidentified)


if __name__ == "__main__":
    unittest.main()
