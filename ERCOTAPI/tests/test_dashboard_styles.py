"""Regression tests for dashboard accessibility styles."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from ERCOTAPI import ercotapi


class DashboardStyleTests(unittest.TestCase):
    @staticmethod
    def _dashboard_css() -> str:
        with patch.object(ercotapi.st, "markdown") as markdown:
            ercotapi.inject_dashboard_css()

        return markdown.call_args.args[0]

    def test_alert_text_has_explicit_light_theme_contrast(self) -> None:
        css = self._dashboard_css()
        self.assertIn('[data-testid="stAlert"] *', css)
        self.assertIn('[data-testid="stAlertContainer"] *', css)
        self.assertIn("--ercot-alert-text: #172033", css)
        self.assertIn(
            "-webkit-text-fill-color: var(--ercot-alert-text) !important",
            css,
        )
        self.assertIn("--ercot-alert-link: #075985", css)

    def test_tabs_support_button_and_react_aria_markup(self) -> None:
        css = self._dashboard_css()

        self.assertIn('.stTabs [role="tab"]', css)
        self.assertIn('[data-testid="stTabs"] [role="tab"] *', css)
        self.assertIn('[data-testid="stTab"]', css)
        self.assertIn('[data-testid="stTab"][aria-selected="true"]', css)
        self.assertNotIn('button[role="tab"]', css)

    def test_pills_support_react_aria_selected_state(self) -> None:
        css = self._dashboard_css()

        self.assertIn(
            '[class*="st-key-ercot_revision_family_filter"] '
            'button[aria-checked="true"]',
            css,
        )
        self.assertIn(
            '[class*="st-key-ercot_revision_family_filter"] '
            'button[data-selected="true"]',
            css,
        )
        self.assertIn('button[aria-checked="true"] *', css)
        self.assertIn('button[data-selected="true"] *', css)


if __name__ == "__main__":
    unittest.main()
