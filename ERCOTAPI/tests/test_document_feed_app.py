"""Dashboard regressions for saved-feed availability and publication metadata."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from ERCOTAPI import ercotapi


class DocumentFeedAppTests(unittest.TestCase):
    def _render_feed(self, payload: dict) -> AppTest:
        app = AppTest.from_string(
            "from ERCOTAPI.ercotapi import render_latest_ercot_documents\n"
            "render_latest_ercot_documents()\n",
            default_timeout=30,
        )
        with patch.object(ercotapi, "load_latest_updates", return_value=payload):
            app.run()
        self.assertFalse(app.exception, [item.message for item in app.exception])
        return app

    def test_feed_publication_time_is_separate_from_monitor_health(self) -> None:
        app = self._render_feed(
            {"items": [], "generated_at": "2026-10-14T21:12:17+00:00"}
        )

        captions = " ".join(item.value for item in app.caption)
        self.assertIn("Saved feed generated: Oct 14, 2026 16:12 CDT", captions)
        self.assertIn("does not confirm document-monitor health", captions)
        self.assertIn("does not run document ingestion", captions)

    def test_empty_feed_does_not_claim_ercot_has_no_revision_requests(self) -> None:
        app = self._render_feed({"items": []})

        self.assertFalse(app.warning)
        self.assertEqual(app.tabs[0].label, "Revision Requests (0)")
        self.assertIn(
            "No revision-request issues are present in the saved document feed.",
            [item.value for item in app.info],
        )

    def test_unavailable_feed_does_not_present_zero_document_counts(self) -> None:
        for error in ("Feed file is missing.", "Feed JSON is invalid."):
            with self.subTest(error=error):
                app = self._render_feed({"items": [], "count": 0, "error": error})

                self.assertEqual(len(app.warning), 1)
                self.assertIn(error, app.warning[0].value)
                self.assertIn(
                    "does not establish that ERCOT has no revision requests",
                    app.warning[0].value,
                )
                self.assertEqual(app.tabs[0].label, "Revision Requests (unavailable)")
                self.assertTrue(
                    any("material count unavailable" in item.value for item in app.caption)
                )
                self.assertFalse(
                    any("No revision-request issues" in item.value for item in app.info)
                )
                self.assertEqual(
                    [tab.label for tab in app.tabs[-2:]],
                    ["Operational messages", "Public & market notices"],
                )

    def test_missing_or_ambiguous_publication_time_remains_unavailable(self) -> None:
        for generated_at in (None, "invalid", "2026-10-14T21:12:17"):
            with self.subTest(generated_at=generated_at):
                app = self._render_feed({"items": [], "generated_at": generated_at})

                self.assertTrue(
                    any("Saved feed generated: unavailable" in item.value for item in app.caption)
                )


if __name__ == "__main__":
    unittest.main()
