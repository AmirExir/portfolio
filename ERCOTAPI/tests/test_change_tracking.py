"""Focused tests for pure section-level ERCOT change tracking."""

from __future__ import annotations

import json
import unittest

from ERCOTAPI.rag_ingestion.change_tracking import (
    compare_document_versions,
    logical_document_key,
    parse_numbered_sections,
)


class SectionParsingTests(unittest.TestCase):
    def test_parses_numbered_sections_with_page_ranges(self) -> None:
        text = """
[Page 4]
3.15 Voltage Support
(1) A Resource shall provide reactive capability.

[Page 5]
Additional requirements continue here.

3.15.1 Applicability
This subsection applies at the POI.

4 OPERATING REQUIREMENTS
The operator shall maintain the required voltage.
"""

        sections = parse_numbered_sections(text)

        self.assertEqual([section.number for section in sections], ["3.15", "3.15.1", "4"])
        self.assertEqual(sections[0].title, "Voltage Support")
        self.assertIn("(1) A Resource", sections[0].text)
        self.assertEqual(sections[0].page_start, 4)
        self.assertEqual(sections[0].page_end, 5)
        self.assertEqual(sections[0].locator, "Section 3.15, pages 4-5")
        self.assertEqual(sections[1].page_start, 5)

    def test_ignores_numbered_paragraphs_and_prefers_substantive_duplicate(self) -> None:
        text = """
TABLE OF CONTENTS
9.1 Introduction.............1
9.2 Applicability\t2

[Page 10]
9.1 Introduction
(1) This section defines the Batch Zero process.
(2) It applies to qualifying Large Loads.

9.2 Applicability
The requirements apply to the listed interconnections.
"""

        sections = parse_numbered_sections(text)

        self.assertEqual([section.number for section in sections], ["9.1", "9.2"])
        self.assertIn("Batch Zero", sections[0].text)
        self.assertNotIn("TABLE OF CONTENTS", sections[0].text)

    def test_parses_concatenated_docx_headings_after_a_table_of_contents(self) -> None:
        text = """
9.1Introduction1
9.2General Provisions2

9.1Introduction
(1) This section defines the Large Load process.

9.2General Provisions
(1) These are the governing provisions.

8ASettlement Process
The attachment explains settlement.

9LARGE LOAD INTERCONNECTION
The top-level section has substantive text.

10 General Requirements
This mixed-case top-level heading is also valid.
"""

        sections = parse_numbered_sections(text)

        self.assertEqual(
            [section.number for section in sections],
            ["9.1", "9.2", "8A", "9", "10"],
        )
        self.assertEqual(sections[0].title, "Introduction")
        self.assertIn("Large Load", sections[0].text)
        self.assertEqual(sections[2].title, "Settlement Process")
        self.assertEqual(sections[3].title, "LARGE LOAD INTERCONNECTION")
        self.assertEqual(sections[4].title, "General Requirements")


class VersionComparisonTests(unittest.TestCase):
    def test_reports_all_change_classes_and_version_specific_citations(self) -> None:
        old_text = """
[Page 1]
1.1 Purpose
This guide defines the study process.

[Page 2]
1.2 Applicability
This applies to Generation Resources.

1.3 Retired Requirement
The old submission is required.
"""
        new_text = """
[Page 3]
1.1 Purpose
This   guide
defines the study process.

[Page 4]
1.2 Applicability
This applies to Generation Resources and Energy Storage Resources.

[Page 5]
1.4 New Requirement
The model shall be submitted electronically.
"""
        old_metadata = {
            "title": "ERCOT Planning Guide — Section 1",
            "source_kind": "Planning Guide",
            "effective_date": "2026-01-01",
            "document_status": "Effective",
            "original_url": "https://www.ercot.com/old-guide",
        }
        new_metadata = {
            "title": "ERCOT Planning Guide — Section 1",
            "source_kind": "Planning Guide",
            "effective_date": "2026-08-01",
            "document_status": "Approved",
            "original_url": "https://www.ercot.com/new-guide",
        }

        report = compare_document_versions(
            old_text,
            new_text,
            old_metadata,
            new_metadata,
        )
        by_number = {change.section_number: change for change in report.changes}

        self.assertTrue(report.same_logical_document)
        self.assertEqual(
            report.counts,
            {"added": 1, "modified": 1, "removed": 1, "unchanged": 1},
        )
        self.assertEqual(by_number["1.1"].status, "unchanged")
        self.assertEqual(by_number["1.2"].status, "modified")
        self.assertEqual(by_number["1.3"].status, "removed")
        self.assertEqual(by_number["1.4"].status, "added")
        self.assertIn("OLD", by_number["1.2"].old_citation or "")
        self.assertIn("Section 1.2, page 2", by_number["1.2"].old_citation or "")
        self.assertIn("Effective 2026-01-01", by_number["1.2"].old_citation or "")
        self.assertIn("NEW", by_number["1.2"].new_citation or "")
        self.assertIn("Section 1.2, page 4", by_number["1.2"].new_citation or "")
        self.assertIsNone(by_number["1.3"].new_citation)
        self.assertIsNone(by_number["1.4"].old_citation)

        # The public report representation must be directly JSON serializable.
        encoded = json.dumps(report.to_dict())
        self.assertIn('"modified": 1', encoded)

    def test_heading_title_change_is_substantive(self) -> None:
        report = compare_document_versions(
            "2.1 Existing Title\nThe requirement is unchanged.",
            "2.1 Revised Title\nThe requirement is unchanged.",
        )

        self.assertEqual(report.changes[0].status, "modified")


class LogicalDocumentKeyTests(unittest.TestCase):
    def test_revision_request_variants_share_a_key(self) -> None:
        old = logical_document_key(
            {
                "document_number": "NPRR-1343",
                "title": "NPRR1343 Pending",
            }
        )
        new = logical_document_key(
            {
                "document_number": "NPRR 1343",
                "title": "NPRR1343 Board Approved",
            }
        )

        self.assertEqual(old, "revision-request:nprr1343")
        self.assertEqual(old, new)

    def test_reverse_xrr_filename_uses_issue_family(self) -> None:
        key = logical_document_key(
            {
                "title": "145PGRR-73 AEPSC Comments 050526",
                "source_kind": "PGRR",
                "source_page_url": "https://www.ercot.com/mktrules/issues/PGRR145",
            }
        )

        self.assertEqual(key, "revision-request:pgrr145")

    def test_artifact_url_corrects_inherited_combined_document_label(self) -> None:
        planning = logical_document_key(
            {
                "title": "Combined",
                "source_kind": "Planning Guide",
                "original_url": (
                    "https://www.ercot.com/files/docs/2026/06/18/"
                    "July-1-2026-Planning-Guide.pdf"
                ),
            }
        )
        mislabeled_protocol = logical_document_key(
            {
                "title": "Combined",
                "source_kind": "Planning Guide",
                "original_url": (
                    "https://www.ercot.com/files/docs/2026/06/18/"
                    "July-1-2026-Nodal-Protocols.pdf"
                ),
            }
        )

        self.assertEqual(planning, "planning-guide:full")
        self.assertEqual(mislabeled_protocol, "nodal-protocols:full")
        self.assertNotEqual(planning, mislabeled_protocol)

    def test_split_guide_versions_share_kind_and_section_key(self) -> None:
        old = logical_document_key(
            {
                "source_kind": "Planning Guide",
                "filename": "09-071126.docx",
                "title": "ERCOT Planning Guide Section 9",
            }
        )
        new = logical_document_key(
            {
                "source_kind": "Planning Guide",
                "filename": "09-080126.docx",
                "title": "ERCOT Planning Guide Section 9",
            }
        )

        self.assertEqual(old, "planning-guide:section-9")
        self.assertEqual(old, new)

    def test_explicit_family_id_has_precedence(self) -> None:
        key = logical_document_key(
            {
                "document_family_id": "ERCOT Reactive Capability Standard",
                "document_number": "Internal 42",
            }
        )

        self.assertEqual(key, "explicit:ercot-reactive-capability-standard")


if __name__ == "__main__":
    unittest.main()
