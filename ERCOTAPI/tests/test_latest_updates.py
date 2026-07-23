"""Tests for the public latest-ERCOT-document feed."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ERCOTAPI.latest_updates import build_latest_updates


class LatestUpdatesTests(unittest.TestCase):
    def test_feed_deduplicates_content_and_explains_revision_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for source in ("obdrr", "tac"):
                path = root / source / "2026" / "shared.html"
                path.parent.mkdir(parents=True)
                path.write_text("shared", encoding="utf-8")
                path.with_name(f"{path.name}.metadata.json").write_text(
                    json.dumps(
                        {
                            "content_sha256": "same-hash",
                            "source_label": source.upper(),
                            "title": "OBDRR050",
                            "document_number": "OBDRR050",
                            "published_date": "2026-07-22",
                            "original_url": "https://www.ercot.com/OBDRR050",
                        }
                    ),
                    encoding="utf-8",
                )
                paths.append(path)
            output = root / "feed.json"

            payload = build_latest_updates(paths, output_path=output)

            self.assertEqual(payload["count"], 1)
            self.assertEqual(payload["items"][0]["sources"], ["OBDRR", "TAC"])
            self.assertIn("Other Binding Document Revision Request", payload["items"][0]["explanation"])
            self.assertEqual(json.loads(output.read_text())["count"], 1)

    def test_feed_excludes_notices_and_pre_2026_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for source, year in (("MARKET NOTICES", "2026"), ("PGRR", "2025")):
                path = root / source / year / "item.html"
                path.parent.mkdir(parents=True)
                path.write_text(source, encoding="utf-8")
                path.with_name(f"{path.name}.metadata.json").write_text(
                    json.dumps(
                        {
                            "content_sha256": f"{source}-{year}",
                            "source_label": source,
                            "title": "item",
                            "published_date": f"{year}-01-01",
                        }
                    ),
                    encoding="utf-8",
                )
                paths.append(path)

            payload = build_latest_updates(paths, output_path=root / "feed.json")

            self.assertEqual(payload["count"], 0)

    def test_feed_excludes_notice_url_with_technical_source_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "dwg" / "2026" / "operations.html"
            path.parent.mkdir(parents=True)
            path.write_text("operations", encoding="utf-8")
            path.with_name(f"{path.name}.metadata.json").write_text(
                json.dumps(
                    {
                        "content_sha256": "operations-hash",
                        "source_label": "DWG",
                        "title": "Operational messages",
                        "published_date": "2026-07-22",
                        "original_url": "https://www.ercot.com/services/comm/mkt_notices/opsmessages",
                    }
                ),
                encoding="utf-8",
            )

            payload = build_latest_updates([path], output_path=root / "feed.json")

            self.assertEqual(payload["count"], 0)


if __name__ == "__main__":
    unittest.main()
