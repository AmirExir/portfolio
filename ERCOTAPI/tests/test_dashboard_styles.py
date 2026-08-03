"""Regression tests for dashboard accessibility styles."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from ERCOTAPI import ercotapi


class DashboardStyleTests(unittest.TestCase):
    def test_alert_text_has_explicit_light_theme_contrast(self) -> None:
        with patch.object(ercotapi.st, "markdown") as markdown:
            ercotapi.inject_dashboard_css()

        css = markdown.call_args.args[0]
        self.assertIn('[data-testid="stAlert"] *', css)
        self.assertIn('[data-testid="stAlertContainer"] *', css)
        self.assertIn("--ercot-alert-text: #172033", css)
        self.assertIn(
            "-webkit-text-fill-color: var(--ercot-alert-text) !important",
            css,
        )
        self.assertIn("--ercot-alert-link: #075985", css)


if __name__ == "__main__":
    unittest.main()
