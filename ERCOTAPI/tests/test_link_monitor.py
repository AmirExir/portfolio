"""Downloader-to-ingestion integration tests with a fully fake HTTP session."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from ERCOTAPI import ercot_link_monitor as monitor
from ERCOTAPI.rag_ingestion.config import IngestionConfig, SourceRoot


class FakeResponse:
    def __init__(
        self,
        *,
        url: str,
        content: bytes = b"",
        text: str | None = None,
        headers: dict[str, str] | None = None,
        error: Exception | None = None,
        status_code: int = 200,
    ) -> None:
        self.url = url
        self.content = content
        self.text = text if text is not None else content.decode("utf-8", errors="replace")
        self.headers = headers or {}
        self.error = error
        self.status_code = status_code
        self.closed = False

    def raise_for_status(self) -> None:
        if self.error is not None:
            raise self.error

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self, responses: dict[str, FakeResponse]) -> None:
        self.responses = responses
        self.headers: dict[str, str] = {}
        self.calls: list[str] = []

    def get(self, url: str, **_kwargs) -> FakeResponse:
        self.calls.append(url)
        return self.responses[url]


class LinkMonitorTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.archive_root = self.root / "official"

    def item(self, *, url: str = "https://www.ercot.com/download?id=1234") -> monitor.DiscoveredItem:
        return monitor.DiscoveredItem(
            source_label="NPRR",
            source_url="https://www.ercot.com/mktrules/issues/nprr",
            title="NPRR 1234 Approval",
            url=url,
            item_type="page",
            published_hint="July 16, 2026",
        )

    def test_all_configured_revision_request_detail_links_are_discovered(self) -> None:
        source_url = "https://www.ercot.com/mktrules/issues/nprr"
        for prefix in ("NPRR", "PGRR", "NOGRR", "OBDRR", "SCR", "RRGRR", "VCMRR"):
            with self.subTest(prefix=prefix):
                self.assertTrue(
                    monitor.is_interesting_link(
                        source_url,
                        f"https://www.ercot.com/mktrules/issues/{prefix.lower()}1234",
                        f"{prefix} 1234",
                    )
                )
        self.assertTrue(
            monitor.is_interesting_link(
                "https://www.ercot.com/services/comm/mkt_notices/archives",
                "https://www.ercot.com/services/comm/mkt_notices/notices/1234",
                "Market notice 1234",
            )
        )

    def test_domain_and_redirect_validation_reject_lookalike_hosts(self) -> None:
        self.assertTrue(monitor.is_allowed_domain("https://www.ercot.com/path"))
        self.assertFalse(monitor.is_allowed_domain("https://evilercot.com/path"))
        self.assertFalse(monitor.is_allowed_domain("https://ercot.com.attacker.example/path"))
        self.assertFalse(monitor.is_allowed_domain("file://ercot.com/etc/passwd"))
        self.assertFalse(monitor.is_allowed_domain("ftp://ercot.com/archive"))
        self.assertFalse(monitor.is_allowed_domain("javascript://ercot.com/alert"))
        with self.assertRaisesRegex(ValueError, "non-ERCOT host"):
            monitor.fetch(FakeSession({}), "https://attacker.example/source")
        source_url = "https://www.ercot.com/source"
        redirected = FakeResponse(
            url="https://attacker.example/redirected",
            content=b"unexpected",
        )
        session = FakeSession({source_url: redirected})

        with self.assertRaisesRegex(ValueError, "disallowed host"):
            monitor.fetch(session, source_url)
        self.assertTrue(redirected.closed)

        failed = FakeResponse(
            url=source_url,
            error=RuntimeError("HTTP 500"),
        )
        with self.assertRaisesRegex(RuntimeError, "HTTP 500"):
            monitor.fetch(FakeSession({source_url: failed}), source_url)
        self.assertTrue(failed.closed)

    def test_manual_redirect_validation_never_requests_disallowed_target(self) -> None:
        source_url = "https://www.ercot.com/source"
        attacker_url = "https://attacker.example/payload"
        for caller in (
            lambda session: monitor.fetch(session, source_url),
            lambda session: monitor.fetch_archivable_content(session, source_url),
        ):
            with self.subTest(caller=caller):
                redirect = FakeResponse(
                    url=source_url,
                    status_code=302,
                    headers={"Location": attacker_url},
                )
                session = FakeSession({source_url: redirect})
                with self.assertRaisesRegex(ValueError, "disallowed host"):
                    caller(session)
                self.assertEqual(session.calls, [source_url])
                self.assertTrue(redirect.closed)

    def test_http_ercot_input_and_redirect_are_upgraded_before_requests(self) -> None:
        plaintext_source = "http://www.ercot.com/source"
        secure_source = "https://www.ercot.com/source"
        plaintext_target = "http://www.ercot.com/files/document.txt"
        secure_target = "https://www.ercot.com/files/document.txt"
        redirect = FakeResponse(
            url=secure_source,
            status_code=302,
            headers={"Location": plaintext_target},
        )
        final = FakeResponse(
            url=secure_target,
            content=b"official bytes",
            headers={"Content-Type": "text/plain"},
        )
        session = FakeSession(
            {
                secure_source: redirect,
                secure_target: final,
            }
        )

        response = monitor.fetch(session, plaintext_source)

        self.assertIs(response, final)
        self.assertEqual(session.calls, [secure_source, secure_target])
        self.assertNotIn(plaintext_source, session.calls)
        self.assertNotIn(plaintext_target, session.calls)
        self.assertTrue(all(url.startswith("https://") for url in session.calls))
        self.assertTrue(redirect.closed)
        response.close()

    def test_report_candidates_are_numeric_newest_first_and_status_changes_update_metadata(self) -> None:
        report_url = "https://www.ercot.com/mktrules/issues/reports/nprr"
        detail_urls = {
            number: f"https://www.ercot.com/mktrules/issues/NPRR{number}"
            for number in (999, 1000, 1340)
        }

        def report_html(status: str) -> str:
            return "".join(
                f'<tr><td>{status if number == 1340 else "Approved"}</td>'
                f'<td><a href="{detail_urls[number]}">NPRR{number}</a></td></tr>'
                for number in (999, 1000, 1340)
            )

        session = FakeSession(
            {
                report_url: FakeResponse(url=report_url, text=report_html("Pending")),
                **{
                    url: FakeResponse(
                        url=url,
                        content=f"NPRR {number} detail".encode(),
                        headers={"Content-Type": "text/plain"},
                    )
                    for number, url in detail_urls.items()
                },
            }
        )
        candidates = monitor.extract_anchor_candidates("NPRR", report_url, report_html("Pending"))
        self.assertEqual([item.title for item in candidates], ["NPRR1340", "NPRR1000", "NPRR999"])
        self.assertEqual(candidates[0].state_tag, "pending")

        with (
            mock.patch.object(monitor.requests, "Session", return_value=session),
            mock.patch.object(monitor, "REPORT_CANDIDATE_WINDOW", 2),
        ):
            first_changes, state = monitor.scan_sources(
                [monitor.SourceLink(label="NPRR", url=report_url)],
                {},
                self.archive_root,
            )
            session.responses[report_url] = FakeResponse(
                url=report_url,
                text=report_html("Approved"),
            )
            second_changes, _ = monitor.scan_sources(
                [monitor.SourceLink(label="NPRR", url=report_url)],
                state,
                self.archive_root,
            )

        self.assertEqual([item["title"] for item in first_changes], ["NPRR1340", "NPRR1000"])
        # The report window rotates, so the older third entry is considered on
        # the next scheduled pass instead of being hidden forever.
        self.assertIn(detail_urls[999], session.calls)
        changed = next(item for item in second_changes if item["title"] == "NPRR1340")
        self.assertEqual(changed["status"], "updated")
        self.assertEqual(changed["download_status"], "metadata_updated")
        metadata = json.loads(Path(changed["metadata_path"]).read_text(encoding="utf-8"))
        self.assertEqual(metadata["document_status"], "Approved")

    def test_report_cells_override_description_lifecycle_and_capture_effective_date(self) -> None:
        report_url = "https://www.ercot.com/mktrules/issues/reports/nprr"
        detail_url = "https://www.ercot.com/mktrules/issues/NPRR1234"
        report_html = (
            "<tr>"
            "<td>Description: approved January 1, 2025 under an older filing.</td>"
            f'<td><a href="{detail_url}">NPRR1234</a></td>'
            "<td>Pending</td>"
            "<td>Posted Date: 2026-07-01</td>"
            "<td>Effective Date: 2026-08-01</td>"
            "</tr>"
        )
        candidates = monitor.extract_anchor_candidates("NPRR", report_url, report_html)
        self.assertEqual(candidates[0].state_tag, "pending")
        self.assertEqual(candidates[0].published_hint, "2026-07-01")
        self.assertEqual(candidates[0].effective_date, "2026-08-01")

        session = FakeSession(
            {
                report_url: FakeResponse(url=report_url, text=report_html),
                detail_url: FakeResponse(
                    url=detail_url,
                    content=b"pending detail",
                    headers={"Content-Type": "text/plain"},
                ),
            }
        )
        with mock.patch.object(monitor.requests, "Session", return_value=session):
            changes, _ = monitor.scan_sources(
                [monitor.SourceLink(label="NPRR", url=report_url)],
                {},
                self.archive_root,
            )
        metadata = json.loads(Path(changes[0]["metadata_path"]).read_text(encoding="utf-8"))
        self.assertEqual(metadata["document_status"], "Pending")
        self.assertEqual(metadata["published_date"], "2026-07-01")
        self.assertEqual(metadata["effective_date"], "2026-08-01")

    def test_issue_detail_page_archives_key_document_attachments(self) -> None:
        source_url = "https://www.ercot.com/mktrules/issues/nprr"
        detail_url = "https://www.ercot.com/mktrules/issues/nprr1234"
        document_url = "https://www.ercot.com/files/key/NPRR1234.pdf"
        source_html = f'<a href="{detail_url}">NPRR1234</a>'
        detail_html = f'<a href="{document_url}">NPRR1234 Approval Document</a>'
        session = FakeSession(
            {
                source_url: FakeResponse(url=source_url, text=source_html),
                detail_url: FakeResponse(
                    url=detail_url,
                    content=detail_html.encode(),
                    text=detail_html,
                    headers={"Content-Type": "text/html"},
                ),
                document_url: FakeResponse(
                    url=document_url,
                    content=b"synthetic PDF",
                    headers={"Content-Type": "application/pdf"},
                ),
            }
        )

        with mock.patch.object(monitor.requests, "Session", return_value=session):
            changes, state = monitor.scan_sources(
                [monitor.SourceLink(label="NPRR", url=source_url)],
                {},
                self.archive_root,
            )

        self.assertEqual({item["final_url"] for item in changes}, {detail_url, document_url})
        self.assertTrue(all(Path(item["downloaded_path"]).is_file() for item in changes))
        self.assertEqual(len(monitor._document_state_entries(state["NPRR"])), 2)

    def test_nested_attachment_fanout_is_bounded_and_rotates_without_starvation(self) -> None:
        source_url = "https://www.ercot.com/mktrules/issues/nprr"
        detail_url = "https://www.ercot.com/mktrules/issues/NPRR1234"
        attachment_urls = [
            f"https://www.ercot.com/files/NPRR{number}.pdf"
            for number in range(2000, 2005)
        ]
        source_html = f'<a href="{detail_url}">NPRR1234</a>'
        detail_html = "".join(
            f'<a href="{url}">{Path(url).stem}</a>' for url in attachment_urls
        )
        responses = {
            source_url: FakeResponse(url=source_url, text=source_html),
            detail_url: FakeResponse(
                url=detail_url,
                content=detail_html.encode(),
                headers={"Content-Type": "text/html"},
            ),
        }
        responses.update(
            {
                url: FakeResponse(
                    url=url,
                    content=f"PDF {url}".encode(),
                    headers={"Content-Type": "application/pdf"},
                )
                for url in attachment_urls
            }
        )
        session = FakeSession(responses)
        links = [monitor.SourceLink(label="NPRR", url=source_url)]

        per_run_attachment_calls: list[int] = []
        all_changes: list[dict[str, object]] = []
        state: dict[str, list[str]] = {}
        with (
            mock.patch.object(monitor.requests, "Session", return_value=session),
            mock.patch.object(monitor, "MAX_ITEMS_PER_SOURCE", 1),
            mock.patch.object(monitor, "MAX_UNSEEN_ATTEMPTS_PER_SOURCE", 1),
            mock.patch.object(monitor, "MAX_NESTED_ITEMS_PER_SOURCE", 2),
        ):
            for _ in range(3):
                before = sum(session.calls.count(url) for url in attachment_urls)
                changes, state = monitor.scan_sources(links, state, self.archive_root)
                after = sum(session.calls.count(url) for url in attachment_urls)
                per_run_attachment_calls.append(after - before)
                all_changes.extend(changes)

        self.assertEqual(per_run_attachment_calls, [2, 2, 2])
        emitted_attachments = {
            str(item["final_url"])
            for item in all_changes
            if str(item.get("final_url")) in attachment_urls
        }
        self.assertEqual(emitted_attachments, set(attachment_urls))

    def test_cross_linked_attachment_preserves_both_parent_provenance_without_refetch(self) -> None:
        source_url = "https://www.ercot.com/mktrules/issues/nprr"
        detail_urls = [
            "https://www.ercot.com/mktrules/issues/NPRR1234",
            "https://www.ercot.com/mktrules/issues/NPRR1235",
        ]
        attachment_url = "https://www.ercot.com/files/shared-attachment.pdf"
        source_html = "".join(
            f'<a href="{url}">{Path(url).name}</a>' for url in detail_urls
        )
        responses = {source_url: FakeResponse(url=source_url, text=source_html)}
        for index, detail_url in enumerate(detail_urls):
            detail_html = (
                f'<a href="{attachment_url}">Shared NPRR Attachment</a>'
                f"<p>Parent {index}</p>"
            )
            responses[detail_url] = FakeResponse(
                url=detail_url,
                content=detail_html.encode(),
                headers={"Content-Type": "text/html"},
            )
        responses[attachment_url] = FakeResponse(
            url=attachment_url,
            content=b"shared attachment bytes",
            headers={"Content-Type": "application/pdf"},
        )
        session = FakeSession(responses)

        with mock.patch.object(monitor.requests, "Session", return_value=session):
            changes, _ = monitor.scan_sources(
                [monitor.SourceLink(label="NPRR", url=source_url)],
                {},
                self.archive_root,
            )

        attachment = next(
            item for item in changes if item.get("final_url") == attachment_url
        )
        metadata = json.loads(Path(attachment["metadata_path"]).read_text(encoding="utf-8"))
        self.assertEqual(metadata["source_page_urls"], sorted(detail_urls))
        attachment_observations = [
            observation
            for observation in metadata["provenance"]
            if observation["original_url"] == attachment_url
        ]
        self.assertEqual(
            sorted(observation["source_page_url"] for observation in attachment_observations),
            sorted(detail_urls),
        )
        self.assertEqual(session.calls.count(attachment_url), 1)

    def test_nested_budget_rotates_fairly_across_parent_pages(self) -> None:
        source_url = "https://www.ercot.com/mktrules/issues/nprr"
        detail_urls = [
            f"https://www.ercot.com/mktrules/issues/NPRR{number}"
            for number in range(1234, 1237)
        ]
        attachment_urls = [
            f"https://www.ercot.com/files/NPRR{number}.pdf"
            for number in range(1234, 1237)
        ]
        responses = {
            source_url: FakeResponse(
                url=source_url,
                text="".join(
                    f'<a href="{url}">{Path(url).name}</a>' for url in detail_urls
                ),
            )
        }
        for detail_url, attachment_url in zip(detail_urls, attachment_urls):
            detail_html = f'<a href="{attachment_url}">{Path(attachment_url).stem}</a>'
            responses[detail_url] = FakeResponse(
                url=detail_url,
                content=detail_html.encode(),
                headers={"Content-Type": "text/html"},
            )
            responses[attachment_url] = FakeResponse(
                url=attachment_url,
                content=attachment_url.encode(),
                headers={"Content-Type": "application/pdf"},
            )
        session = FakeSession(responses)
        links = [monitor.SourceLink(label="NPRR", url=source_url)]

        state: dict[str, list[str]] = {}
        emitted: set[str] = set()
        with (
            mock.patch.object(monitor.requests, "Session", return_value=session),
            mock.patch.object(monitor, "MAX_ITEMS_PER_SOURCE", 3),
            mock.patch.object(monitor, "MAX_NESTED_ITEMS_PER_SOURCE", 1),
        ):
            for _ in range(3):
                changes, state = monitor.scan_sources(links, state, self.archive_root)
                emitted.update(
                    str(item["final_url"])
                    for item in changes
                    if str(item.get("final_url")) in attachment_urls
                )

        self.assertEqual(emitted, set(attachment_urls))

    def test_report_lifecycle_status_is_inherited_by_nested_attachment(self) -> None:
        report_url = "https://www.ercot.com/mktrules/issues/reports/nprr"
        detail_url = "https://www.ercot.com/mktrules/issues/NPRR1234"
        document_url = "https://www.ercot.com/files/NPRR1234.pdf"
        report_html = (
            f'<tr><td>Approved</td><td><a href="{detail_url}">NPRR1234</a></td></tr>'
        )
        detail_html = f'<a href="{document_url}">NPRR1234 Approval Document</a>'
        session = FakeSession(
            {
                report_url: FakeResponse(url=report_url, text=report_html),
                detail_url: FakeResponse(
                    url=detail_url,
                    content=detail_html.encode(),
                    headers={"Content-Type": "text/html"},
                ),
                document_url: FakeResponse(
                    url=document_url,
                    content=b"approved PDF",
                    headers={"Content-Type": "application/pdf"},
                ),
            }
        )

        with mock.patch.object(monitor.requests, "Session", return_value=session):
            changes, _ = monitor.scan_sources(
                [monitor.SourceLink(label="NPRR", url=report_url)],
                {},
                self.archive_root,
            )

        attachment = next(item for item in changes if item["final_url"] == document_url)
        metadata = json.loads(Path(attachment["metadata_path"]).read_text(encoding="utf-8"))
        self.assertEqual(metadata["document_status"], "Approved")

    def test_archive_item_persists_hash_named_bytes_and_official_metadata(self) -> None:
        content = b"synthetic official PDF bytes"
        url = "https://www.ercot.com/download?id=1234"
        response = FakeResponse(
            url=url,
            content=content,
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Disposition": 'attachment; filename="NPRR1234.pdf"',
            },
        )
        session = FakeSession({url: response})

        archived = monitor.archive_item(session, self.item(url=url), self.archive_root)

        digest = hashlib.sha256(content).hexdigest()
        self.assertEqual(archived.path.name, f"{digest}.pdf")
        self.assertEqual(archived.path.read_bytes(), content)
        self.assertEqual(archived.archive_status, "archived")
        metadata = json.loads(archived.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["content_sha256"], digest)
        self.assertEqual(metadata["source_authority"], "ERCOT")
        self.assertFalse(metadata["is_generated"])
        self.assertEqual(metadata["source_kind"], "NPRR")
        self.assertEqual(metadata["original_url"], url)
        self.assertEqual(metadata["size"], len(content))
        self.assertEqual(metadata["schema_version"], 2)
        self.assertEqual(metadata["url_aliases"], [url])
        self.assertEqual(len(metadata["provenance"]), 1)

        marker_ns = 1_700_000_000_000_000_000
        os.utime(archived.path, ns=(marker_ns, marker_ns))
        os.utime(archived.metadata_path, ns=(marker_ns, marker_ns))
        repeated = monitor.archive_item(session, self.item(url=url), self.archive_root)
        self.assertEqual(repeated.path, archived.path)
        self.assertEqual(repeated.archive_status, "already_archived")
        self.assertEqual(repeated.path.stat().st_mtime_ns, marker_ns)
        self.assertEqual(repeated.metadata_path.stat().st_mtime_ns, marker_ns)

    def test_same_hash_alias_across_publication_years_reuses_one_archive_object(self) -> None:
        content = b"identical cross-year official bytes"
        first_url = "https://www.ercot.com/files/NPRR1234-2026.txt"
        second_url = "https://www.ercot.com/files/NPRR1234-2027.txt"
        session = FakeSession(
            {
                first_url: FakeResponse(
                    url=first_url,
                    content=content,
                    headers={"Content-Type": "text/plain"},
                ),
                second_url: FakeResponse(
                    url=second_url,
                    content=content,
                    headers={"Content-Type": "text/plain"},
                ),
            }
        )
        first = self.item(url=first_url)
        first.published_hint = "2026-12-31"
        second = self.item(url=second_url)
        second.published_hint = "2027-01-01"

        archived_first = monitor.archive_item(session, first, self.archive_root)
        archived_second = monitor.archive_item(session, second, self.archive_root)

        self.assertEqual(archived_second.path, archived_first.path)
        self.assertEqual(archived_second.archive_status, "metadata_updated")
        metadata = json.loads(archived_second.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["url_aliases"], sorted([first_url, second_url]))
        archived_files = [
            path
            for path in self.archive_root.rglob(f"{archived_first.content_hash}.*")
            if not path.name.endswith(".metadata.json")
        ]
        self.assertEqual(archived_files, [archived_first.path])

    def test_alias_metadata_aggregates_authoritative_status_and_newest_dates(self) -> None:
        content = b"shared lifecycle document"
        pending_url = "https://www.ercot.com/files/A-NPRR1234.txt"
        approved_url = "https://www.ercot.com/files/B-NPRR1234.txt"
        withdrawn_url = "https://www.ercot.com/files/C-NPRR1234.txt"
        session = FakeSession(
            {
                pending_url: FakeResponse(
                    url=pending_url,
                    content=content,
                    headers={"Content-Type": "text/plain"},
                ),
                approved_url: FakeResponse(
                    url=approved_url,
                    content=content,
                    headers={"Content-Type": "text/plain"},
                ),
                withdrawn_url: FakeResponse(
                    url=withdrawn_url,
                    content=content,
                    headers={"Content-Type": "text/plain"},
                ),
            }
        )
        pending = self.item(url=pending_url)
        pending.state_tag = "pending"
        pending.published_hint = "2026-01-01"
        approved = self.item(url=approved_url)
        approved.state_tag = "approved"
        approved.published_hint = "2027-01-01"
        approved.effective_date = "2027-02-01"

        monitor.archive_item(session, pending, self.archive_root)
        archived = monitor.archive_item(session, approved, self.archive_root)
        metadata = json.loads(archived.metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(metadata["original_url"], pending_url)
        self.assertEqual(metadata["document_status"], "Approved")
        self.assertEqual(metadata["published_date"], "2027-01-01")
        self.assertEqual(metadata["effective_date"], "2027-02-01")

        withdrawn = self.item(url=withdrawn_url)
        withdrawn.state_tag = "withdrawn"
        withdrawn.published_hint = "2028-01-01"
        archived = monitor.archive_item(session, withdrawn, self.archive_root)
        metadata = json.loads(archived.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["document_status"], "Withdrawn")
        self.assertEqual(metadata["published_date"], "2028-01-01")

    def test_validated_alias_promotes_ambiguous_bin_to_ingestible_extension(self) -> None:
        content = b"synthetic PDF bytes"
        ambiguous_url = "https://www.ercot.com/download?id=1234"
        pdf_url = "https://www.ercot.com/files/NPRR1234.pdf"
        session = FakeSession(
            {
                ambiguous_url: FakeResponse(
                    url=ambiguous_url,
                    content=content,
                    headers={"Content-Type": "application/octet-stream"},
                ),
                pdf_url: FakeResponse(
                    url=pdf_url,
                    content=content,
                    headers={"Content-Type": "application/pdf"},
                ),
            }
        )

        ambiguous = monitor.archive_item(
            session,
            self.item(url=ambiguous_url),
            self.archive_root,
        )
        promoted = monitor.archive_item(
            session,
            self.item(url=pdf_url),
            self.archive_root,
        )

        self.assertEqual(ambiguous.path.suffix, ".bin")
        self.assertEqual(promoted.path.suffix, ".pdf")
        self.assertFalse(ambiguous.path.exists())
        self.assertFalse(ambiguous.metadata_path.exists())
        self.assertEqual(promoted.path.read_bytes(), content)
        metadata = json.loads(promoted.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["url_aliases"], sorted([ambiguous_url, pdf_url]))

        source_url = "https://www.ercot.com/mktrules/issues/nprr"
        source_html = (
            f'<a href="{ambiguous_url}">NPRR1234 Report Z Ambiguous</a>'
            f'<a href="{pdf_url}">NPRR1234 A PDF</a>'
        )
        scan_session = FakeSession(
            {
                source_url: FakeResponse(url=source_url, text=source_html),
                ambiguous_url: FakeResponse(
                    url=ambiguous_url,
                    content=content,
                    headers={"Content-Type": "application/octet-stream"},
                ),
                pdf_url: FakeResponse(
                    url=pdf_url,
                    content=content,
                    headers={"Content-Type": "application/pdf"},
                ),
            }
        )
        scan_root = self.root / "scan-official"
        with mock.patch.object(monitor.requests, "Session", return_value=scan_session):
            changes, _ = monitor.scan_sources(
                [monitor.SourceLink(label="NPRR", url=source_url)],
                {},
                scan_root,
            )
        self.assertEqual(len(changes), 1)
        self.assertEqual(Path(changes[0]["downloaded_path"]).suffix, ".pdf")
        self.assertTrue(Path(changes[0]["downloaded_path"]).is_file())

    def test_scan_sources_archives_once_and_advances_state_only_after_success(self) -> None:
        source_url = "https://www.ercot.com/mktrules/issues/nprr"
        document_url = "https://www.ercot.com/files/NPRR1234.txt"
        html = f'<html><a href="{document_url}">NPRR1234 Approved</a></html>'
        session = FakeSession(
            {
                source_url: FakeResponse(url=source_url, text=html),
                document_url: FakeResponse(
                    url=document_url,
                    content=b"NPRR 1234 official requirement",
                    headers={"Content-Type": "text/plain"},
                ),
            }
        )
        links = [monitor.SourceLink(label="NPRR", url=source_url)]

        with mock.patch.object(monitor.requests, "Session", return_value=session):
            changes, state = monitor.scan_sources(links, {}, self.archive_root)
            repeated_changes, repeated_state = monitor.scan_sources(links, state, self.archive_root)

        self.assertEqual(len(changes), 1)
        change = changes[0]
        self.assertEqual(change["status"], "new")
        self.assertEqual(change["download_status"], "archived")
        self.assertTrue(Path(change["downloaded_path"]).is_file())
        self.assertTrue(Path(change["metadata_path"]).is_file())
        state_fingerprints = {
            monitor._decode_state_entry(entry)[0] for entry in state["NPRR"]
        }
        self.assertIn(self.item(url=document_url).fingerprint, state_fingerprints)
        self.assertEqual(repeated_changes, [])
        self.assertEqual(
            monitor._document_state_entries(repeated_state["NPRR"]),
            monitor._document_state_entries(state["NPRR"]),
        )
        # Known URLs are fetched again so a changed payload at a stable ERCOT
        # URL is detected by content hash, but unchanged bytes create no event.
        self.assertEqual(session.calls.count(document_url), 2)

    def test_legacy_plain_fingerprint_state_is_recognized_and_upgraded(self) -> None:
        source_url = "https://www.ercot.com/mktrules/issues/nprr"
        document_url = "https://www.ercot.com/files/NPRR1234.txt"
        html = f'<a href="{document_url}">NPRR1234 Approved</a>'
        session = FakeSession(
            {
                source_url: FakeResponse(url=source_url, text=html),
                document_url: FakeResponse(
                    url=document_url,
                    content=b"NPRR 1234 official requirement",
                    headers={"Content-Type": "text/plain"},
                ),
            }
        )
        legacy_fingerprint = self.item(url=document_url).fingerprint

        with mock.patch.object(monitor.requests, "Session", return_value=session):
            changes, state = monitor.scan_sources(
                [monitor.SourceLink(label="NPRR", url=source_url)],
                {"NPRR": [legacy_fingerprint]},
                self.archive_root,
            )

        self.assertEqual(changes[0]["status"], "updated")
        fingerprint, last_hash = monitor._decode_state_entry(state["NPRR"][0])
        self.assertEqual(fingerprint, legacy_fingerprint)
        self.assertRegex(str(last_hash), r"^[a-f0-9]{64}$")

    def test_unseen_top_level_backlog_is_bounded_and_drained_on_later_runs(self) -> None:
        source_url = "https://www.ercot.com/mktrules/issues/nprr"
        document_urls = [f"https://www.ercot.com/files/NPRR{number}.txt" for number in range(1000, 1007)]
        html = "".join(
            f'<a href="{url}">NPRR{number}</a>'
            for number, url in zip(range(1000, 1007), document_urls)
        )
        responses = {source_url: FakeResponse(url=source_url, text=html)}
        responses.update(
            {
                url: FakeResponse(
                    url=url,
                    content=f"NPRR {number} official text".encode(),
                    headers={"Content-Type": "text/plain"},
                )
                for number, url in zip(range(1000, 1007), document_urls)
            }
        )
        session = FakeSession(responses)
        links = [monitor.SourceLink(label="NPRR", url=source_url)]

        with (
            mock.patch.object(monitor.requests, "Session", return_value=session),
            mock.patch.object(monitor, "MAX_ITEMS_PER_SOURCE", 5),
        ):
            first_changes, state = monitor.scan_sources(links, {}, self.archive_root)
            second_changes, _ = monitor.scan_sources(links, state, self.archive_root)

        self.assertEqual(len(first_changes), 5)
        self.assertEqual(len(second_changes), 2)
        self.assertTrue(all(item["status"] == "new" for item in second_changes))

    def test_failed_download_is_reported_and_remains_retryable(self) -> None:
        source_url = "https://www.ercot.com/mktrules/issues/nprr"
        document_url = "https://www.ercot.com/files/NPRR4321.pdf"
        html = f'<a href="{document_url}">NPRR4321</a>'
        session = FakeSession(
            {
                source_url: FakeResponse(url=source_url, text=html),
                document_url: FakeResponse(
                    url=document_url,
                    error=RuntimeError("synthetic download failure"),
                ),
            }
        )

        with mock.patch.object(monitor.requests, "Session", return_value=session):
            changes, state = monitor.scan_sources(
                [monitor.SourceLink(label="NPRR", url=source_url)],
                {},
                self.archive_root,
            )

        self.assertEqual(changes[0]["status"], "error")
        self.assertEqual(changes[0]["download_status"], "failed")
        self.assertIn("synthetic download failure", changes[0]["summary"])
        self.assertNotIn(self.item(url=document_url).fingerprint, state["NPRR"])
        self.assertFalse(self.archive_root.exists())

    def test_known_stable_url_with_changed_bytes_is_archived_as_updated(self) -> None:
        source_url = "https://www.ercot.com/mktrules/issues/nprr"
        document_url = "https://www.ercot.com/files/current-NPRR.txt"
        html = f'<a href="{document_url}">NPRR1234 Current Version</a>'
        session = FakeSession(
            {
                source_url: FakeResponse(url=source_url, text=html),
                document_url: FakeResponse(
                    url=document_url,
                    content=b"NPRR version one",
                    headers={"Content-Type": "text/plain"},
                ),
            }
        )
        links = [monitor.SourceLink(label="NPRR", url=source_url)]

        with mock.patch.object(monitor.requests, "Session", return_value=session):
            first_changes, state = monitor.scan_sources(links, {}, self.archive_root)
            session.responses[document_url] = FakeResponse(
                url=document_url,
                content=b"NPRR version two",
                headers={"Content-Type": "text/plain"},
            )
            second_changes, next_state = monitor.scan_sources(links, state, self.archive_root)

        self.assertEqual(first_changes[0]["status"], "new")
        self.assertEqual(second_changes[0]["status"], "updated")
        self.assertNotEqual(first_changes[0]["content_sha256"], second_changes[0]["content_sha256"])
        self.assertNotEqual(first_changes[0]["downloaded_path"], second_changes[0]["downloaded_path"])
        self.assertTrue(Path(second_changes[0]["downloaded_path"]).is_file())
        first_fingerprint, first_hash = monitor._decode_state_entry(state["NPRR"][0])
        next_fingerprint, next_hash = monitor._decode_state_entry(next_state["NPRR"][0])
        self.assertEqual(first_fingerprint, next_fingerprint)
        self.assertNotEqual(first_hash, next_hash)

    def test_identical_bytes_from_multiple_urls_merge_provenance_and_then_stay_quiet(self) -> None:
        source_url = "https://www.ercot.com/mktrules/issues/nprr"
        first_url = "https://www.ercot.com/files/NPRR1234-A.txt"
        second_url = "https://www.ercot.com/files/NPRR1234-B.txt"
        html = (
            f'<a href="{first_url}">NPRR1234 Copy A</a>'
            f'<a href="{second_url}">NPRR1234 Copy B</a>'
        )
        shared_content = b"the same official NPRR 1234 document"
        session = FakeSession(
            {
                source_url: FakeResponse(url=source_url, text=html),
                first_url: FakeResponse(
                    url=first_url,
                    content=shared_content,
                    headers={"Content-Type": "text/plain"},
                ),
                second_url: FakeResponse(
                    url=second_url,
                    content=shared_content,
                    headers={"Content-Type": "application/octet-stream"},
                ),
            }
        )
        links = [monitor.SourceLink(label="NPRR", url=source_url)]

        with mock.patch.object(monitor.requests, "Session", return_value=session):
            first_changes, state = monitor.scan_sources(links, {}, self.archive_root)
            repeated_changes, repeated_state = monitor.scan_sources(
                links,
                state,
                self.archive_root,
            )

        self.assertEqual(len(first_changes), 1)
        self.assertEqual(first_changes[0]["alias_count"], 2)
        self.assertEqual(first_changes[0]["url_aliases"], sorted([first_url, second_url]))
        metadata_path = Path(first_changes[0]["metadata_path"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["original_url"], min(first_url, second_url))
        self.assertEqual(metadata["content_type"], "text/plain")
        self.assertEqual(metadata["url_aliases"], sorted([first_url, second_url]))
        self.assertEqual(
            [entry["original_url"] for entry in metadata["provenance"]],
            sorted([first_url, second_url]),
        )
        self.assertEqual(repeated_changes, [])
        self.assertEqual(
            monitor._document_state_entries(repeated_state["NPRR"]),
            monitor._document_state_entries(state["NPRR"]),
        )

    def test_stable_url_content_reversion_is_reported_even_when_hash_archive_exists(self) -> None:
        source_url = "https://www.ercot.com/mktrules/issues/nprr"
        document_url = "https://www.ercot.com/files/current-NPRR.txt"
        html = f'<a href="{document_url}">NPRR1234 Current Version</a>'
        session = FakeSession(
            {
                source_url: FakeResponse(url=source_url, text=html),
                document_url: FakeResponse(
                    url=document_url,
                    content=b"version A",
                    headers={"Content-Type": "text/plain"},
                ),
            }
        )
        links = [monitor.SourceLink(label="NPRR", url=source_url)]

        with mock.patch.object(monitor.requests, "Session", return_value=session):
            first_changes, state_a = monitor.scan_sources(links, {}, self.archive_root)
            session.responses[document_url] = FakeResponse(
                url=document_url,
                content=b"version B",
                headers={"Content-Type": "text/plain"},
            )
            _, state_b = monitor.scan_sources(links, state_a, self.archive_root)
            session.responses[document_url] = FakeResponse(
                url=document_url,
                content=b"version A",
                headers={"Content-Type": "text/plain"},
            )
            reverted_changes, reverted_state = monitor.scan_sources(
                links,
                state_b,
                self.archive_root,
            )
            quiet_changes, quiet_state = monitor.scan_sources(
                links,
                reverted_state,
                self.archive_root,
            )

        self.assertEqual(len(reverted_changes), 1)
        reverted = reverted_changes[0]
        self.assertEqual(reverted["status"], "updated")
        self.assertEqual(reverted["download_status"], "already_archived")
        self.assertTrue(reverted["content_changed_since_last_seen"])
        self.assertEqual(reverted["content_sha256"], first_changes[0]["content_sha256"])
        self.assertEqual(quiet_changes, [])
        self.assertEqual(quiet_state, reverted_state)

    def test_oversized_archive_response_is_rejected_before_persistence(self) -> None:
        document_url = "https://www.ercot.com/files/NPRR1234.txt"
        response = FakeResponse(
            url=document_url,
            content=b"0123456789",
            headers={"Content-Type": "text/plain"},
        )
        session = FakeSession({document_url: response})

        with (
            mock.patch.object(monitor, "MAX_RESPONSE_BYTES", 8),
            self.assertRaisesRegex(ValueError, "MAX_RESPONSE_BYTES"),
        ):
            monitor.archive_item(session, self.item(url=document_url), self.archive_root)

        self.assertFalse(self.archive_root.exists())
        self.assertTrue(response.closed)

    def test_oversized_source_page_is_rejected_before_link_discovery(self) -> None:
        source_url = "https://www.ercot.com/mktrules/issues/nprr"
        document_url = "https://www.ercot.com/files/NPRR1234.txt"
        oversized_html = f'<a href="{document_url}">NPRR1234</a>'.encode()
        source_response = FakeResponse(
            url=source_url,
            content=oversized_html,
            headers={"Content-Type": "text/html"},
        )
        session = FakeSession(
            {
                source_url: source_response,
                document_url: FakeResponse(
                    url=document_url,
                    content=b"must not be fetched",
                ),
            }
        )

        with (
            mock.patch.object(monitor.requests, "Session", return_value=session),
            mock.patch.object(monitor, "MAX_RESPONSE_BYTES", 16),
        ):
            changes, _ = monitor.scan_sources(
                [monitor.SourceLink(label="NPRR", url=source_url)],
                {},
                self.archive_root,
            )

        self.assertEqual(changes[0]["status"], "error")
        self.assertIn("MAX_RESPONSE_BYTES", changes[0]["summary"])
        self.assertNotIn(document_url, session.calls)
        self.assertTrue(source_response.closed)

    def test_inline_public_notices_page_is_archived_quietly_and_feeds_ingestion(self) -> None:
        source_url = "https://www.ercot.com/services/comm/mkt_notices/notices"
        inline_html = (
            "<html><body><section>"
            "<h2>Current Public Notices</h2>"
            "<div>July 16, 2026 — Reliability Unit Commitment notice inline only.</div>"
            "</section></body></html>"
        )
        session = FakeSession(
            {source_url: FakeResponse(url=source_url, text=inline_html)}
        )
        links = [monitor.SourceLink(label="PUBLIC NOTICES", url=source_url)]

        with mock.patch.object(monitor.requests, "Session", return_value=session):
            first_changes, state = monitor.scan_sources(links, {}, self.archive_root)
            repeated_changes, state = monitor.scan_sources(
                links,
                state,
                self.archive_root,
            )
            session.responses[source_url] = FakeResponse(
                url=source_url,
                text=inline_html.replace("notice inline only", "updated notice inline only"),
            )
            updated_changes, _ = monitor.scan_sources(
                links,
                state,
                self.archive_root,
            )

        self.assertEqual(len(first_changes), 1)
        snapshot = first_changes[0]
        self.assertEqual(snapshot["status"], "new")
        snapshot_path = Path(snapshot["downloaded_path"])
        snapshot_metadata_path = Path(snapshot["metadata_path"])
        self.assertEqual(
            snapshot_path,
            self.archive_root.resolve()
            / "public-notices"
            / "current"
            / "current.html",
        )
        self.assertIn("Reliability Unit Commitment", snapshot_path.read_text())
        metadata = json.loads(Path(snapshot["metadata_path"]).read_text(encoding="utf-8"))
        self.assertEqual(metadata["source_label"], "PUBLIC NOTICES")
        self.assertEqual(metadata["source_kind"], "PUBLIC NOTICES")
        self.assertEqual(metadata["source_authority"], "ERCOT")
        self.assertFalse(metadata["is_generated"])
        self.assertEqual(repeated_changes, [])
        self.assertEqual(len(updated_changes), 1)
        self.assertEqual(updated_changes[0]["status"], "updated")
        self.assertEqual(Path(updated_changes[0]["downloaded_path"]), snapshot_path)
        self.assertEqual(
            Path(updated_changes[0]["metadata_path"]),
            snapshot_metadata_path,
        )
        self.assertNotEqual(
            updated_changes[0]["content_sha256"],
            snapshot["content_sha256"],
        )
        self.assertIn("updated notice inline only", snapshot_path.read_text())
        self.assertEqual(
            sorted(path.name for path in snapshot_path.parent.iterdir()),
            ["current.html", "current.html.metadata.json"],
        )

        last_good_content = snapshot_path.read_bytes()
        last_good_metadata = snapshot_metadata_path.read_bytes()
        session.responses[source_url] = FakeResponse(
            url=source_url,
            text=inline_html.replace("notice inline only", "third notice version"),
        )
        real_atomic_write = monitor.atomic_write_bytes

        def fail_staged_metadata(path: Path, content: bytes) -> None:
            if path.name == "metadata.stage":
                raise OSError("injected staged metadata failure")
            real_atomic_write(path, content)

        with (
            mock.patch.object(monitor.requests, "Session", return_value=session),
            mock.patch.object(
                monitor,
                "atomic_write_bytes",
                side_effect=fail_staged_metadata,
            ),
        ):
            failed_changes, _ = monitor.scan_sources(
                links,
                state,
                self.archive_root,
            )

        self.assertEqual(len(failed_changes), 1)
        self.assertEqual(failed_changes[0]["status"], "error")
        self.assertEqual(failed_changes[0]["download_status"], "failed")
        self.assertIsNone(failed_changes[0]["downloaded_path"])
        self.assertEqual(snapshot_path.read_bytes(), last_good_content)
        self.assertEqual(snapshot_metadata_path.read_bytes(), last_good_metadata)
        self.assertEqual(
            sorted(path.name for path in snapshot_path.parent.iterdir()),
            ["current.html", "current.html.metadata.json"],
        )

        main_root = self.root / "main-official"
        main_session = FakeSession(
            {source_url: FakeResponse(url=source_url, text=inline_html)}
        )
        ingestion_result = {
            "enabled": True,
            "attempted": True,
            "status": "completed",
            "summary": {"changed": True},
            "error": None,
        }
        output = io.StringIO()
        with (
            mock.patch.object(monitor.requests, "Session", return_value=main_session),
            mock.patch.object(monitor, "load_links", return_value=links),
            mock.patch.object(monitor, "load_state", return_value={}),
            mock.patch.object(monitor, "save_state"),
            mock.patch.object(monitor, "official_document_dir", return_value=main_root),
            mock.patch.object(monitor, "env_flag", return_value=True),
            mock.patch.object(
                monitor,
                "invoke_incremental_ingestion",
                return_value=ingestion_result,
            ) as invoke,
            mock.patch.object(monitor, "SEND_TELEGRAM", False),
            redirect_stdout(output),
        ):
            monitor.main()

        # Public notices remain archived for reporting but are never embedded.
        invoke.assert_called_once_with([], main_root, enabled=True)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["downloaded_items"], 1)
        self.assertEqual(Path(payload["downloaded_paths"][0]).suffix, ".html")

    def test_docx_and_xlsx_zip_members_are_bounded_before_xml_parse(self) -> None:
        fixtures = (
            ("word/document.xml", monitor.extract_docx_summary),
            ("xl/workbook.xml", monitor.extract_xlsx_summary),
        )
        for member_name, extractor in fixtures:
            with self.subTest(member=member_name):
                payload = io.BytesIO()
                with zipfile.ZipFile(payload, "w") as archive:
                    archive.writestr(member_name, b"x" * 32)
                with mock.patch.object(monitor, "MAX_RESPONSE_BYTES", 16):
                    summary = extractor(payload.getvalue())
                self.assertIn("MAX_RESPONSE_BYTES", summary)

    def test_failed_top_candidates_do_not_starve_healthy_backlog(self) -> None:
        source_url = "https://www.ercot.com/mktrules/issues/nprr"
        numbers = list(range(1000, 1007))
        document_urls = {
            number: f"https://www.ercot.com/files/NPRR{number}.txt"
            for number in numbers
        }
        html = "".join(
            f'<a href="{document_urls[number]}">NPRR{number}</a>'
            for number in numbers
        )
        responses = {source_url: FakeResponse(url=source_url, text=html)}
        for number, url in document_urls.items():
            responses[url] = FakeResponse(
                url=url,
                content=f"NPRR {number}".encode(),
                headers={"Content-Type": "text/plain"},
                error=(RuntimeError("temporary failure") if number in {1006, 1005} else None),
            )
        session = FakeSession(responses)

        with (
            mock.patch.object(monitor.requests, "Session", return_value=session),
            mock.patch.object(monitor, "MAX_ITEMS_PER_SOURCE", 3),
            mock.patch.object(monitor, "MAX_UNSEEN_ATTEMPTS_PER_SOURCE", 6),
        ):
            changes, state = monitor.scan_sources(
                [monitor.SourceLink(label="NPRR", url=source_url)],
                {},
                self.archive_root,
            )

        self.assertEqual(len([item for item in changes if item["status"] == "error"]), 2)
        successful = [item for item in changes if item["status"] == "new"]
        self.assertEqual(len(successful), 3)
        self.assertEqual({item["title"] for item in successful}, {"NPRR1004", "NPRR1003", "NPRR1002"})
        self.assertEqual(len(monitor._document_state_entries(state["NPRR"])), 3)

    def test_unseen_retry_cursor_reaches_tail_after_more_failures_than_attempt_limit(self) -> None:
        source_url = "https://www.ercot.com/mktrules/issues/nprr"
        numbers = [1004, 1003, 1002, 1001]
        urls = {
            number: f"https://www.ercot.com/files/NPRR{number}.txt"
            for number in numbers
        }
        html = "".join(
            f'<a href="{urls[number]}">NPRR{number}</a>' for number in numbers
        )
        responses = {source_url: FakeResponse(url=source_url, text=html)}
        for number in numbers:
            responses[urls[number]] = FakeResponse(
                url=urls[number],
                content=f"NPRR {number}".encode(),
                headers={"Content-Type": "text/plain"},
                error=(RuntimeError("permanent failure") if number > 1001 else None),
            )
        session = FakeSession(responses)
        links = [monitor.SourceLink(label="NPRR", url=source_url)]

        with (
            mock.patch.object(monitor.requests, "Session", return_value=session),
            mock.patch.object(monitor, "MAX_ITEMS_PER_SOURCE", 1),
            mock.patch.object(monitor, "MAX_UNSEEN_ATTEMPTS_PER_SOURCE", 2),
        ):
            first_changes, state = monitor.scan_sources(links, {}, self.archive_root)
            second_changes, state = monitor.scan_sources(links, state, self.archive_root)

        self.assertEqual(len([item for item in first_changes if item["status"] == "error"]), 2)
        self.assertEqual(len([item for item in second_changes if item["status"] == "error"]), 1)
        healthy = [item for item in second_changes if item["status"] == "new"]
        self.assertEqual([item["title"] for item in healthy], ["NPRR1001"])
        self.assertIn(urls[1001], session.calls)

    def test_known_recheck_cursor_eventually_detects_changed_url_below_cap(self) -> None:
        source_url = "https://www.ercot.com/mktrules/issues/nprr"
        numbers = [1003, 1002, 1001, 1000]
        urls = {
            number: f"https://www.ercot.com/files/NPRR{number}.txt"
            for number in numbers
        }
        html = "".join(
            f'<a href="{urls[number]}">NPRR{number}</a>' for number in numbers
        )
        responses = {source_url: FakeResponse(url=source_url, text=html)}
        responses.update(
            {
                urls[number]: FakeResponse(
                    url=urls[number],
                    content=f"version one {number}".encode(),
                    headers={"Content-Type": "text/plain"},
                )
                for number in numbers
            }
        )
        session = FakeSession(responses)
        links = [monitor.SourceLink(label="NPRR", url=source_url)]

        with (
            mock.patch.object(monitor.requests, "Session", return_value=session),
            mock.patch.object(monitor, "MAX_ITEMS_PER_SOURCE", 4),
            mock.patch.object(monitor, "MAX_KNOWN_RECHECKS_PER_SOURCE", 1),
        ):
            _, state = monitor.scan_sources(links, {}, self.archive_root)
            session.responses[urls[1000]] = FakeResponse(
                url=urls[1000],
                content=b"version two 1000",
                headers={"Content-Type": "text/plain"},
            )
            observed: list[dict[str, object]] = []
            for _ in range(4):
                changes, state = monitor.scan_sources(links, state, self.archive_root)
                observed.extend(changes)

        changed = [
            item
            for item in observed
            if item.get("url") == urls[1000] and item.get("status") == "updated"
        ]
        self.assertEqual(len(changed), 1)
        self.assertGreaterEqual(session.calls.count(urls[1000]), 2)

    def test_bounded_state_retains_newest_candidates_without_reemitting_evictions(self) -> None:
        source_url = "https://www.ercot.com/mktrules/issues/nprr"
        numbers = list(range(1000, 1005))

        def page(values: list[int]) -> str:
            return "".join(
                f'<a href="https://www.ercot.com/files/NPRR{number}.txt">NPRR{number}</a>'
                for number in values
            )

        responses = {source_url: FakeResponse(url=source_url, text=page(numbers))}
        responses.update(
            {
                f"https://www.ercot.com/files/NPRR{number}.txt": FakeResponse(
                    url=f"https://www.ercot.com/files/NPRR{number}.txt",
                    content=f"NPRR {number}".encode(),
                    headers={"Content-Type": "text/plain"},
                )
                for number in [*numbers, 999]
            }
        )
        session = FakeSession(responses)
        links = [monitor.SourceLink(label="NPRR", url=source_url)]

        with (
            mock.patch.object(monitor.requests, "Session", return_value=session),
            mock.patch.object(monitor, "MAX_ITEMS_PER_SOURCE", 5),
            mock.patch.object(monitor, "MAX_STATE_ITEMS_PER_SOURCE", 3),
        ):
            first_changes, state = monitor.scan_sources(links, {}, self.archive_root)
            repeated_changes, repeated_state = monitor.scan_sources(
                links,
                state,
                self.archive_root,
            )
            session.responses[source_url] = FakeResponse(
                url=source_url,
                text=page([*numbers, 999]),
            )
            new_changes, final_state = monitor.scan_sources(
                links,
                repeated_state,
                self.archive_root,
            )

        self.assertEqual(len(first_changes), 5)
        self.assertEqual(len(monitor._document_state_entries(state["NPRR"])), 3)
        self.assertEqual(repeated_changes, [])
        self.assertEqual([item["title"] for item in new_changes], ["NPRR999"])
        self.assertEqual(len(monitor._document_state_entries(final_state["NPRR"])), 3)

    def test_incremental_ingestion_hook_uses_download_paths_and_archive_root(self) -> None:
        downloaded = self.archive_root / "nprr" / "2026" / "hash.txt"
        downloaded.parent.mkdir(parents=True)
        downloaded.write_text("NPRR content", encoding="utf-8")
        default_official = SourceRoot(
            name="official_downloads",
            path=self.root / "wrong-root",
            source_authority="ERCOT",
            is_generated=False,
            default_source_kind="Official Document",
            default_collections=("general",),
        )
        config = IngestionConfig(
            repo_root=self.root,
            index_dir=self.root / "index",
            source_roots=(default_official,),
        )
        observed: dict[str, object] = {}

        class FakePipeline:
            def __init__(self, selected: IngestionConfig) -> None:
                observed["config"] = selected

            def update(self, paths):
                observed["paths"] = paths
                return {"changed": True, "documents": 1}

        with (
            mock.patch("ERCOTAPI.rag_ingestion.default_config", return_value=config),
            mock.patch("ERCOTAPI.rag_ingestion.IngestionPipeline", FakePipeline),
        ):
            result = monitor.invoke_incremental_ingestion(
                [downloaded],
                self.archive_root,
                enabled=True,
            )

        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["attempted"])
        self.assertEqual(observed["paths"], [downloaded])
        selected = observed["config"]
        self.assertIsInstance(selected, IngestionConfig)
        official = next(root for root in selected.source_roots if root.name == "official_downloads")
        self.assertEqual(official.path, self.archive_root)

    def test_auto_ingest_selects_only_new_2026_technical_documents(self) -> None:
        candidates = [
            self.archive_root / "nprr" / "2026" / "new.pdf",
            self.archive_root / "dwg" / "2027" / "future.docx",
            self.archive_root / "nprr" / "2025" / "old.pdf",
            self.archive_root / "market-notices" / "2026" / "notice.html",
            self.archive_root / "public-notices" / "2026" / "notice.html",
            self.archive_root / "sswg" / "2026" / "source.pdf.metadata.json",
            self.archive_root / "sswg" / "2026" / "unsupported.zip",
        ]

        selected = monitor.auto_ingest_paths(candidates, self.archive_root)

        self.assertEqual(selected, candidates[:2])

    def test_historical_archive_candidates_are_rejected_before_download(self) -> None:
        old_url = monitor.DiscoveredItem(
            source_label="NPRR",
            source_url="source",
            title="NPRR439",
            url="https://www.ercot.com/files/docs/2012/07/01/NPRR439.doc",
            published_hint="",
            item_type="doc",
        )
        old_hint = monitor.DiscoveredItem(
            source_label="DWG",
            source_url="source",
            title="Old procedure",
            url="https://www.ercot.com/download?id=old",
            published_hint="2025-12-31",
            item_type="pdf",
        )
        current = monitor.DiscoveredItem(
            source_label="SSWG",
            source_url="source",
            title="Current procedure",
            url="https://www.ercot.com/download?id=current",
            published_hint="2026-07-21",
            item_type="pdf",
        )

        self.assertFalse(monitor.is_current_archive_candidate(old_url))
        self.assertFalse(monitor.is_current_archive_candidate(old_hint))
        self.assertTrue(monitor.is_current_archive_candidate(current))

    def test_main_bounds_command_output_without_skipping_archive_ingestion(self) -> None:
        self.archive_root.mkdir(parents=True)
        changes = [
            {
                "status": "new",
                "downloaded_path": str(self.archive_root / "nprr" / "2026" / f"document-{index}.txt"),
                "source": "NPRR",
                "title": f"NPRR {index}",
                "url": f"https://www.ercot.com/document-{index}",
                "summary": "x" * 1200,
            }
            for index in range(45)
        ]
        changes.extend(
            {
                "status": "error",
                "downloaded_path": None,
                "source": "PGRR",
                "title": f"PGRR {index}",
                "url": f"https://www.ercot.com/error-{index}",
                "summary": "failed",
            }
            for index in range(5)
        )
        output = io.StringIO()

        with (
            mock.patch.object(monitor, "load_links", return_value=[]),
            mock.patch.object(monitor, "load_state", return_value={}),
            mock.patch.object(monitor, "scan_sources", return_value=(changes, {})),
            mock.patch.object(monitor, "save_state"),
            mock.patch.object(monitor, "official_document_dir", return_value=self.archive_root),
            mock.patch.object(monitor, "env_flag", return_value=True),
            mock.patch.object(
                monitor,
                "invoke_incremental_ingestion",
                return_value={"status": "completed"},
            ) as invoke,
            mock.patch.object(monitor, "SEND_TELEGRAM", False),
            mock.patch.object(monitor, "MAX_OUTPUT_ITEMS", 40),
            mock.patch.object(monitor, "MAX_TELEGRAM_CHARS", 3900),
            mock.patch.object(monitor, "MAX_AUTO_INGEST_FILES", 100),
            redirect_stdout(output),
        ):
            monitor.main()

        payload = json.loads(output.getvalue())
        invoke.assert_called_once_with(
            [self.archive_root / "nprr" / "2026" / f"document-{index}.txt" for index in range(45)],
            self.archive_root,
            enabled=True,
        )
        self.assertEqual(payload["new_items"], 45)
        self.assertEqual(payload["downloaded_items"], 45)
        self.assertEqual(payload["total_changes"], 50)
        self.assertEqual(payload["reported_changes"], 40)
        self.assertEqual(payload["omitted_changes"], 10)
        self.assertEqual(len(payload["changes"]), 40)
        self.assertTrue(all(item["status"] == "new" for item in payload["changes"]))
        self.assertLessEqual(len(payload["telegram_text"]), 3900)
        self.assertLess(len(output.getvalue().encode("utf-8")), 200_000)

    def test_main_ingests_only_new_downloads(self) -> None:
        downloaded = self.archive_root / "nprr" / "2026" / "hash.txt"
        updated = self.archive_root / "nprr" / "2026" / "updated-hash.txt"
        self.archive_root.mkdir(parents=True)
        changes = [
            {"status": "new", "downloaded_path": str(downloaded), "source": "NPRR", "title": "new", "url": "u"},
            {
                "status": "updated",
                "downloaded_path": str(updated),
                "source": "NPRR",
                "title": "updated",
                "url": "u2",
            },
            {"status": "error", "downloaded_path": None, "source": "PGRR", "summary": "failed"},
        ]
        ingestion_result = {
            "enabled": True,
            "attempted": True,
            "status": "completed",
            "summary": {"changed": True},
            "error": None,
        }
        output = io.StringIO()

        with (
            mock.patch.object(monitor, "load_links", return_value=[monitor.SourceLink("NPRR", "source")]),
            mock.patch.object(monitor, "load_state", return_value={}),
            mock.patch.object(monitor, "scan_sources", return_value=(changes, {"NPRR": ["seen"]})),
            mock.patch.object(monitor, "save_state") as save_state,
            mock.patch.object(monitor, "official_document_dir", return_value=self.archive_root),
            mock.patch.object(monitor, "env_flag", return_value=True),
            mock.patch.object(monitor, "invoke_incremental_ingestion", return_value=ingestion_result) as invoke,
            mock.patch.object(monitor, "SEND_TELEGRAM", False),
            redirect_stdout(output),
        ):
            monitor.main()

        invoke.assert_called_once_with([downloaded, updated], self.archive_root, enabled=True)
        save_state.assert_called_once()
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["downloaded_items"], 2)
        self.assertEqual(payload["downloaded_paths"], [str(downloaded), str(updated)])
        self.assertEqual(payload["ingestion"]["status"], "completed")

    def test_main_does_not_rescan_archive_when_there_are_no_new_downloads(self) -> None:
        self.archive_root.mkdir(parents=True)
        ingestion_result = {
            "enabled": True,
            "attempted": True,
            "status": "completed",
            "summary": {"changed": False},
            "error": None,
        }
        output = io.StringIO()

        with (
            mock.patch.object(monitor, "load_links", return_value=[]),
            mock.patch.object(monitor, "load_state", return_value={}),
            mock.patch.object(monitor, "scan_sources", return_value=([], {})),
            mock.patch.object(monitor, "save_state"),
            mock.patch.object(monitor, "official_document_dir", return_value=self.archive_root),
            mock.patch.object(monitor, "env_flag", return_value=True),
            mock.patch.object(
                monitor,
                "invoke_incremental_ingestion",
                return_value=ingestion_result,
            ) as invoke,
            mock.patch.object(monitor, "SEND_TELEGRAM", False),
            redirect_stdout(output),
        ):
            monitor.main()

        invoke.assert_called_once_with([], self.archive_root, enabled=True)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["downloaded_items"], 0)
        self.assertEqual(payload["downloaded_paths"], [])
        self.assertEqual(payload["ingestion"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
