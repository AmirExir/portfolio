"""Tests for the public latest-ERCOT-document feed."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ERCOTAPI.latest_updates import (
    build_latest_updates,
    latest_archive_paths,
    load_latest_updates,
    merge_latest_updates,
    parse_revision_issue_text,
    revision_request_identity,
)


class LatestUpdatesTests(unittest.TestCase):
    def test_loader_distinguishes_unavailable_invalid_and_empty_feeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "feed.json"
            self.assertIn("error", load_latest_updates(path))
            for contents in ("not json", "[]", "{}", '{"items":[null]}', '{"items":[],"revision_issues":[]}'):
                with self.subTest(contents=contents):
                    path.write_text(contents, encoding="utf-8")
                    self.assertIn("error", load_latest_updates(path))
            path.write_text('{"items":[],"count":42}', encoding="utf-8")
            self.assertEqual(load_latest_updates(path), {"items": [], "count": 0})

    def test_archive_rebuild_uses_newest_url_snapshot_and_excludes_navigation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "archive"
            root.mkdir()
            for name, date, url in (
                ("new", "2026-10-14T01:00:00+00:00", "https://www.ercot.com/files/report.pdf"),
                ("old", "2026-10-13T23:00:00Z", "https://www.ercot.com/files/report.pdf"),
                ("index", "2026-10-14T01:00:00Z", "https://www.ercot.com/mktrules/issues/reports/nprr/pending"),
                ("library", "2026-10-14T01:00:00Z", "https://www.ercot.com/mktrules/guides/noperating/library/2025"),
            ):
                path = root / f"{name}.txt"
                path.write_text(name, encoding="utf-8")
                path.with_name(f"{path.name}.metadata.json").write_text(json.dumps({
                    "original_url": url,
                    "downloaded_at": date,
                    "published_date": "2026-10-01",
                    "source_label": "DWG",
                    "title": name,
                }), encoding="utf-8")
            paths = latest_archive_paths(root)
            self.assertNotIn(root / "old.txt", paths)
            payload = build_latest_updates(paths)
            self.assertEqual([item["title"] for item in payload["items"]], ["new"])
            (root / "new.txt").unlink()
            with self.assertRaisesRegex(ValueError, "Missing archived document"):
                latest_archive_paths(root)

    def test_out_of_order_batch_preserves_newer_material_and_issue_details(self) -> None:
        item = {
            "url": "https://www.ercot.com/mktrules/issues/PGRR147",
            "revision_id": "PGRR147", "source": "PGRR", "sources": ["PGRR"],
            "content_sha256": "new", "downloaded_at": "2026-10-14T01:00:00Z",
        }
        details = {
            "official_description": "New description", "source_downloaded_at": item["downloaded_at"],
        }
        previous = {"items": [item], "revision_issues": {"PGRR147": details}}
        batch = {
            "items": [{**item, "downloaded_at": "2026-10-13T00:00:00+00:00", "content_sha256": "old"}],
            "revision_issues": {"PGRR147": {"official_description": "Old description"}},
        }
        merged = merge_latest_updates(previous, batch)
        self.assertEqual(merged["count"], 1)
        self.assertEqual(merged["items"][0]["content_sha256"], "new")
        self.assertEqual(merged["items"][0]["explanation"], "New description")
        self.assertEqual(merged["revision_issues"]["PGRR147"], details)
        self.assertEqual(previous["items"][0], item)

    def test_merge_deduplicates_shared_content_and_preserves_source_labels(self) -> None:
        item = {"content_sha256": "same", "source": "TAC", "url": "https://www.ercot.com/one"}
        merged = merge_latest_updates({"items": [item]}, {"items": [{
            **item, "source": "NPRR", "url": "https://www.ercot.com/two",
        }]})
        self.assertEqual(merged["count"], 1)
        self.assertEqual(merged["items"][0]["sources"], ["NPRR", "TAC"])

    def test_revision_identity_handles_forward_reverse_and_issue_urls(self) -> None:
        self.assertEqual(
            revision_request_identity(title="145PGRR-72 PRS Report"),
            ("PGRR145", "PGRR"),
        )
        self.assertEqual(
            revision_request_identity(title="055OBDRR-08 Board Report"),
            ("OBDRR055", "OBDRR"),
        )
        self.assertEqual(
            revision_request_identity(
                url="https://www.ercot.com/mktrules/issues/NPRR1343"
            ),
            ("NPRR1343", "NPRR"),
        )
        self.assertIsNone(revision_request_identity(title="NPRR Submission Process"))

    def test_feed_corrects_incidental_metadata_revision_without_editing_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "proposal.txt"
            path.write_text("Archived proposal", encoding="utf-8")
            metadata_path = path.with_name(f"{path.name}.metadata.json")
            metadata_path.write_text(json.dumps({
                "title": "1345NPRR-01 Extend NPRR1279 Sunset Date",
                "document_number": "NPRR1279", "source_label": "NPRR",
                "original_url": "https://www.ercot.com/files/1345NPRR-01.docx",
                "published_date": "2026-10-01",
            }), encoding="utf-8")
            original = metadata_path.read_bytes()
            payload = build_latest_updates([path])
            self.assertEqual(payload["items"][0]["document_number"], "NPRR1345")
            self.assertEqual(payload["items"][0]["revision_id"], "NPRR1345")
            self.assertEqual(metadata_path.read_bytes(), original)

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

    def test_feed_excludes_navigation_pages_even_under_xrr_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "pgrr" / "2026" / "workshops.html"
            path.parent.mkdir(parents=True)
            path.write_text("navigation", encoding="utf-8")
            path.with_name(f"{path.name}.metadata.json").write_text(
                json.dumps(
                    {
                        "content_sha256": "navigation-hash",
                        "source_label": "PGRR",
                        "title": "Workshops",
                        "published_date": "2026-07-22",
                    }
                ),
                encoding="utf-8",
            )

            payload = build_latest_updates([path], output_path=root / "feed.json")

            self.assertEqual(payload["count"], 0)

    def test_feed_uses_official_issue_description_and_action_without_ai(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "NEWS" / "official" / "nogrr" / "2026" / "issue.html"
            path.parent.mkdir(parents=True)
            path.write_text(
                """
                <div id="tab-summary"><table>
                  <tr><th>Title</th><td>Addition of 765-kV Operational Voltage Limits</td></tr>
                  <tr><th>Next Group</th><td>TAC</td></tr>
                  <tr><th>Next Step</th><td>TAC for consideration</td></tr>
                  <tr><th>Status</th><td>Pending</td></tr>
                </table></div>
                <div id="tab-action"><table>
                  <tr><th>Date</th><th>Gov Body</th><th>Action Taken</th><th>Next Steps</th></tr>
                  <tr><td>07/09/2026</td><td>ROS</td><td>Recommended for Approval</td><td>TAC for consideration</td></tr>
                  <tr><td>06/04/2026</td><td>ROS</td><td>Deferred/Tabled</td><td>Impact Analysis</td></tr>
                </table></div>
                <div id="tab-background"><table>
                  <tr><th>Status:</th><td>Pending</td></tr>
                  <tr><th>Date Posted:</th><td>Mar 16, 2026</td></tr>
                  <tr><th>Sponsor:</th><td>Joint Sponsors</td></tr>
                  <tr><th>Urgent:</th><td>No</td></tr>
                  <tr><th>Sections:</th><td>2.7.3.1</td></tr>
                  <tr><th>Description:</th><td>Defines operational guidelines for reliable operation of 765-kV equipment.</td></tr>
                  <tr><th>Reason:</th><td>System improvement</td></tr>
                </table></div>
                """,
                encoding="utf-8",
            )
            path.with_name(f"{path.name}.metadata.json").write_text(
                json.dumps(
                    {
                        "content_sha256": "nogrr286-page",
                        "source_label": "NOGRR",
                        "title": "NOGRR286",
                        "document_number": "NOGRR286",
                        # Report indexes can expose an implementation date here;
                        # the issue page's Date Posted field must win for display.
                        "published_date": "January 1, 2029",
                        "downloaded_at": "2026-07-23T00:00:00Z",
                        "document_status": "Pending",
                        "original_url": "https://www.ercot.com/mktrules/issues/NOGRR286",
                    }
                ),
                encoding="utf-8",
            )

            payload = build_latest_updates([path], output_path=root / "feed.json")

            issue = payload["revision_issues"]["NOGRR286"]
            self.assertEqual(issue["issue_title"], "Addition of 765-kV Operational Voltage Limits")
            self.assertEqual(issue["affected_sections"], "2.7.3.1")
            self.assertEqual(issue["latest_action"]["date"], "07/09/2026")
            self.assertEqual(issue["latest_action"]["action"], "Recommended for Approval")
            self.assertEqual(issue["effective_state"], "pending_proposal")
            self.assertEqual(payload["items"][0]["explanation"], issue["official_description"])
            self.assertEqual(payload["items"][0]["published_date"], "Mar 16, 2026")

    def test_saved_chunk_text_parser_handles_description_across_chunks(self) -> None:
        details = parse_revision_issue_text(
            " ".join(
                (
                    "Summary Title Extension for LCL Cooling Load Ride-Through Compliance "
                    "Next Group ROS Next Step ROS for consideration Status Pending Action ",
                    "Background Status: Pending Date Posted: Jul 21, 2026 Sponsor: SB Energy "
                    "Urgent: Pending Sections: 2.6.4 and 2.15 Description: This NOGRR permits "
                    "an LCL to achieve ride-through compliance within 365 days after in-service. "
                    "Reason: General system improvement Key Documents 289NOGRR-01",
                )
            ),
            revision_id="NOGRR289",
            issue_url="https://www.ercot.com/mktrules/issues/NOGRR289",
        )

        self.assertEqual(
            details["issue_title"],
            "Extension for LCL Cooling Load Ride-Through Compliance",
        )
        self.assertEqual(details["date_posted"], "Jul 21, 2026")
        self.assertEqual(details["affected_sections"], "2.6.4 and 2.15")
        self.assertIn("within 365 days", details["official_description"])

    def test_shipped_feed_has_source_derived_details_for_every_revision_issue(self) -> None:
        feed_path = Path(__file__).resolve().parents[1] / "latest_ercot_updates.json"
        payload = json.loads(feed_path.read_text(encoding="utf-8"))
        revision_ids = {
            str(item.get("revision_id") or "")
            for item in payload["items"]
            if item.get("revision_id")
        }
        issues = payload.get("revision_issues", {})

        self.assertEqual(set(issues), revision_ids)
        self.assertTrue(revision_ids)
        # A single successful monitor batch must never replace the shipped
        # history of the core tracked revision families again.
        self.assertTrue(
            {"NPRR", "PGRR", "NOGRR", "OBDRR"}.issubset(
                {item.get("revision_family") for item in payload["items"]}
            )
        )
        for item in payload["items"]:
            if item.get("revision_id"):
                self.assertEqual(item["document_number"], item["revision_id"])
        for revision_id in revision_ids:
            issue = issues[revision_id]
            self.assertTrue(issue["issue_title"], revision_id)
            self.assertTrue(issue["official_description"], revision_id)
            self.assertTrue(issue["status"], revision_id)
            self.assertTrue(issue["date_posted"], revision_id)
            self.assertTrue(issue["affected_sections"], revision_id)
            self.assertTrue(issue["latest_action"], revision_id)
            self.assertNotIn(
                "search the assistant",
                issue["official_description"].casefold(),
                revision_id,
            )


if __name__ == "__main__":
    unittest.main()
