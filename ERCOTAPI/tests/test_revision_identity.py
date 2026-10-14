"""Regression tests for xRR attachments that reference other revision requests."""

from __future__ import annotations

import unittest

from ERCOTAPI.latest_updates import revision_request_identity


class RevisionIdentityTests(unittest.TestCase):
    def test_attachment_identifier_overrides_incidental_metadata_reference(self) -> None:
        cases = (
            ("NPRR1279", "1345NPRR-01 Extend NPRR1279 Sunset Date", "NPRR1345", "NPRR"),
            ("NPRR1188", "1336NPRR-01 Language Clean-up for NPRR1188 Implementation", "NPRR1336", "NPRR"),
            ("PGRR146", "1332NPRR-01 Related PGRR146", "NPRR1332", "NPRR"),
            ("NPRR1298", "035SMOGRR-08 PUCT Report Related NPRR1298", "SMOGRR035", "SMOGRR"),
            ("NPRR1340", "288NOGRR-01 Related NPRR1340", "NOGRR288", "NOGRR"),
        )
        for reference, title, expected_id, family in cases:
            with self.subTest(title=title):
                filename = title.replace(" ", "-") + ".docx"
                self.assertEqual(
                    revision_request_identity(
                        document_number=reference,
                        title=title,
                        url=f"https://www.ercot.com/files/docs/2026/10/14/{filename}",
                    ),
                    (expected_id, family),
                )

    def test_title_identifier_works_with_generic_download_urls(self) -> None:
        for title, reference, expected in (
            ("1345NPRR-01 Extend NPRR1279 Sunset Date", "NPRR1279", "NPRR1345"),
            ("1336NPRR-01 Language Clean-up for NPRR1188 Implementation", "NPRR1188", "NPRR1336"),
        ):
            with self.subTest(title=title):
                self.assertEqual(
                    revision_request_identity(
                        document_number=reference,
                        title=title,
                        url="https://www.ercot.com/download?id=12345",
                    ),
                    (expected, "NPRR"),
                )

    def test_filename_identifier_precedes_conflicting_title_and_metadata(self) -> None:
        self.assertEqual(
            revision_request_identity(
                document_number="NPRR1279",
                title="NPRR1279 sunset date",
                url="https://www.ercot.com/files/1345NPRR%2D01-Extend-NPRR1279.docx?download=1",
            ),
            ("NPRR1345", "NPRR"),
        )

    def test_explicit_issue_url_keeps_highest_priority(self) -> None:
        self.assertEqual(
            revision_request_identity(
                document_number="NPRR1279",
                title="1345NPRR-01 Extend NPRR1279 Sunset Date",
                url="https://www.ercot.com/mktrules/issues/NPRR1346",
            ),
            ("NPRR1346", "NPRR"),
        )

    def test_existing_fallback_to_document_number_is_preserved(self) -> None:
        self.assertEqual(
            revision_request_identity(
                document_number="NPRR1345",
                title="Extend NPRR1279 sunset date",
                url="https://www.ercot.com/download?id=12345",
            ),
            ("NPRR1345", "NPRR"),
        )


if __name__ == "__main__":
    unittest.main()
