"""Streamlit smoke tests for the packaged regional Grid Atlas view."""

from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).parents[1] / "ercotapi.py"


class GridAtlasAppTests(unittest.TestCase):
    def _run_region(self, region_id: str | None = None) -> AppTest:
        app = AppTest.from_file(str(APP_PATH), default_timeout=60)
        app.query_params["view"] = "atlas"
        if region_id is not None:
            app.query_params["grid_region"] = region_id
        app.run(timeout=60)
        self.assertFalse(
            app.exception,
            [exception.message for exception in app.exception],
        )
        return app

    def test_grid_atlas_defaults_to_ercot_without_hiding_other_regions(self):
        app = self._run_region()
        region_group = next(
            group
            for group in app.get("button_group")
            if group.label == "Grid area"
        )

        self.assertEqual(app.query_params["grid_region"], ["ercot"])
        self.assertEqual(region_group.value, "ERCOT")
        self.assertEqual(
            [option.content for option in region_group.options],
            [
                "All U.S. & Canada",
                "ERCOT",
                "MISO",
                "PJM",
                "CAISO",
                "SPP",
                "NYISO",
                "ISO-NE",
                "Canada",
            ],
        )
        self.assertTrue(
            any(
                caption.value.startswith("ERCOT loaded from")
                for caption in app.caption
            )
        )

    def test_overview_filter_lists_all_market_filters_and_uses_local_shard(self):
        app = self._run_region("all")
        region_group = next(
            group
            for group in app.get("button_group")
            if group.label == "Grid area"
        )
        options = [option.content for option in region_group.options]

        self.assertEqual(
            options,
            [
                "All U.S. & Canada",
                "ERCOT",
                "MISO",
                "PJM",
                "CAISO",
                "SPP",
                "NYISO",
                "ISO-NE",
                "Canada",
            ],
        )
        self.assertEqual(
            [(metric.label, metric.value) for metric in app.metric],
            [
                ("Transmission lines", "22,202"),
                ("Power plants", "2,785"),
                ("Substations / transformers", "11,230"),
            ],
        )
        self.assertTrue(
            any(
                "no bulk source, OpenAI, or embedding request" in caption.value
                for caption in app.caption
            )
        )
        self.assertNotIn(
            "Load public Texas grid",
            [button.label for button in app.button],
        )

    def test_canada_filter_preserves_unknown_voltage_reference_assets(self):
        app = self._run_region("canada")

        self.assertEqual(
            [(metric.label, metric.value) for metric in app.metric],
            [
                ("Transmission lines", "13,009"),
                ("Power plants", "1,125"),
                ("Substations / transformers", "4,538"),
            ],
        )
        unknown_voltage = next(
            checkbox
            for checkbox in app.checkbox
            if checkbox.label == "Include unknown voltage"
        )
        self.assertTrue(unknown_voltage.value)
        self.assertTrue(
            any("CanVec" in info.value and "through 2015" in info.value for info in app.info)
        )


if __name__ == "__main__":
    unittest.main()
