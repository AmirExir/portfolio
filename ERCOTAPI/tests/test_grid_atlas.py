"""Tests for the public ArcGIS adapters used by the ERCOT Grid Atlas."""

from __future__ import annotations

import unittest

from ERCOTAPI.grid_atlas import (
    _feature_pages,
    clean_number,
    fetch_power_plants,
    fetch_substations,
    load_public_texas_grid,
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


if __name__ == "__main__":
    unittest.main()
