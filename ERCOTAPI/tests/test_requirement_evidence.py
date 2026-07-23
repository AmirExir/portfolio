"""Lifecycle, governing authority, and diversified evidence tests."""

from __future__ import annotations

import unittest

import numpy as np

from ERCOTAPI.rag_ingestion.requirements import (
    analyze_question,
    annotate_evidence,
    authority_class,
    diversify_evidence,
    lifecycle_metadata,
    validate_answer_citations,
)
from ERCOTAPI.rag_ingestion.retrieval import (
    LoadedIndex,
    _version_rank,
    format_citation,
    retrieve_chunks,
    retrieve_requirement_evidence,
)


def chunk(identifier: str, **values):
    record = {
        "chunk_id": identifier,
        "document_id": f"doc-{identifier}",
        "text": "Reactive power capability and voltage support requirement at the POI.",
        "title": identifier,
        "source_path": f"ERCOTAPI/sources/{identifier}.docx",
        "source_authority": "ERCOT",
        "source_kind": "Official Document",
        "is_generated": False,
        "collections": ["general"],
        "chunk_index": 0,
        "original_url": f"https://www.ercot.com/files/{identifier}",
    }
    record.update(values)
    return record


def index(records, vectors=None):
    vectors = vectors or [[1.0, 0.0] for _ in records]
    return LoadedIndex(
        chunks=list(records),
        embeddings=np.asarray(vectors, dtype="float32"),
        embedding_model="test-model",
        generation_id="test-generation",
        source="central",
        collections=(),
        state_token=("test-generation", 1),
    )


class LifecycleTests(unittest.TestCase):
    def test_natural_language_as_of_date_is_recognized(self) -> None:
        analysis = analyze_question(
            "As of July 22, 2026, is PGRR147 effective?"
        )
        self.assertEqual(analysis.as_of, "2026-07-22")

    def test_future_approved_xrr_does_not_displace_current_guide(self) -> None:
        current = chunk(
            "current-guide",
            source_kind="Planning Guide",
            source_category="planning_guide_uploads",
            document_status="Effective",
            effective_date="2026-01-01",
        )
        future = chunk(
            "future-pgrr",
            source_kind="PGRR",
            document_number="PGRR147",
            document_status="Board Approved",
            effective_date="2026-08-01",
        )

        results = retrieve_chunks(
            "As of 2026-07-22, what governs reactive power capability?",
            index([current, future], [[0.98, 0.1989975], [1.0, 0.0]]),
            top_k=2,
            query_embedder=lambda _question: [1.0, 0.0],
            as_of="2026-07-22",
        )

        self.assertEqual(results[0]["chunk_id"], "current-guide")
        self.assertEqual(results[0]["effective_state"], "effective")
        self.assertEqual(results[1]["effective_state"], "approved_not_effective")
        self.assertEqual(results[1]["evidence_role"], "related_change_record")

    def test_approved_xrr_without_implementation_is_not_governing(self) -> None:
        record = chunk(
            "approved-nprr",
            source_kind="NPRR",
            document_number="NPRR1343",
            document_status="Approved",
        )

        annotated = annotate_evidence(record, as_of="2026-07-22")

        self.assertEqual(annotated["authority_class"], "revision_request")
        self.assertEqual(annotated["effective_state"], "approved_effectiveness_unverified")
        self.assertEqual(annotated["evidence_role"], "related_change_record")

    def test_unknown_status_is_not_assumed_effective(self) -> None:
        metadata = lifecycle_metadata(
            chunk("unknown", source_kind="Planning Guide"),
            as_of="2026-07-22",
        )
        self.assertEqual(metadata["effective_state"], "effectiveness_unknown")

    def test_not_effective_status_is_not_mistaken_for_implemented(self) -> None:
        metadata = lifecycle_metadata(
            chunk(
                "approved-not-effective",
                source_kind="PGRR",
                document_number="PGRR147",
                document_status="Board Approved - Not Yet Effective",
            ),
            as_of="2026-07-22",
        )

        self.assertEqual(metadata["effective_state"], "not_effective")

    def test_controlled_split_date_is_resolved_without_reembedding(self) -> None:
        record = chunk(
            "current-section-nine",
            source_kind="Planning Guide",
            source_path="ERCOTAPI/sources/official/planning_guides/09-071126.docx",
            effective_date=None,
            chunk_index=4,
        )

        before = annotate_evidence(record, as_of="2026-07-10")
        current = annotate_evidence(record, as_of="2026-07-22")

        self.assertEqual(before["effective_state"], "approved_not_effective")
        self.assertEqual(current["effective_state"], "effective")
        self.assertEqual(current["effective_date"], "2026-07-11")
        self.assertTrue(current["effective_date_inferred"])

    def test_undated_current_copy_is_not_backdated_for_historical_question(self) -> None:
        metadata = lifecycle_metadata(
            chunk(
                "undated-current",
                source_kind="Planning Guide",
                source_category="planning_guide_uploads",
                effective_date=None,
            ),
            as_of="2025-01-01",
        )

        self.assertEqual(metadata["effective_state"], "effectiveness_unknown")

    def test_archived_guide_edition_date_does_not_prove_currentness(self) -> None:
        record = chunk(
            "archived-july-guide",
            source_kind="Planning Guide",
            source_path="ERCOTAPI/NEWS/official/planning-guide/2026/hash.pdf",
            original_url="https://www.ercot.com/July-1-2026-Planning-Guide.pdf",
            effective_date=None,
        )

        metadata = lifecycle_metadata(record, as_of="2026-07-22")
        annotated = annotate_evidence(record, as_of="2026-07-22")

        self.assertEqual(
            metadata["effective_state"],
            "effective_edition_currentness_unverified",
        )
        self.assertEqual(metadata["resolved_effective_date"], "2026-07-01")
        self.assertIsNone(annotated.get("effective_date"))

    def test_market_manual_is_related_procedure_not_governing_protocol(self) -> None:
        record = chunk("vcm", source_kind="Verifiable Cost Manual")

        self.assertEqual(authority_class(record), "procedure_or_criteria")

    def test_controlled_split_outranks_stale_published_metadata(self) -> None:
        current = chunk(
            "current-nine",
            source_kind="Planning Guide",
            source_path="ERCOTAPI/sources/official/planning_guides/09-071126.docx",
            published_date="Aug 1, 2025",
            effective_date=None,
        )
        archived = chunk(
            "archived-guide",
            source_kind="Planning Guide",
            source_path="ERCOTAPI/NEWS/official/planning-guide/2026/july.pdf",
            published_date="2026-06-18",
            effective_date=None,
        )

        self.assertGreater(_version_rank(current), _version_rank(archived))


class RequirementEvidenceTests(unittest.TestCase):
    def test_many_governing_documents_do_not_starve_procedure_evidence(self) -> None:
        governing = [
            chunk(
                f"guide-{number}",
                source_kind="Planning Guide",
                source_category="planning_guide_uploads",
                effective_date="2026-01-01",
                retrieval_score=1.0 - (number * 0.01),
            )
            for number in range(8)
        ]
        procedure = chunk(
            "resource-integration-handbook",
            source_kind="Resource Integration Handbook",
            retrieval_score=0.91,
        )

        selected = diversify_evidence(
            "What is the ERCOT generator interconnection process?",
            [*governing, procedure],
            top_k=4,
            as_of="2026-07-22",
        )

        self.assertEqual(len(selected), 4)
        self.assertIn(
            "procedure_or_engineering_criteria",
            {item["evidence_role"] for item in selected},
        )
        self.assertIn(
            "resource-integration-handbook",
            {item["chunk_id"] for item in selected},
        )

    def test_generic_interconnection_process_gets_generator_load_and_handbook_anchors(self) -> None:
        common_guide = {
            "source_kind": "Planning Guide",
            "source_category": "planning_guide_uploads",
            "effective_date": "2026-07-11",
            "collections": ["general", "planning"],
        }
        records = [
            chunk(
                "generation-initiation",
                **common_guide,
                document_id="planning-guide-section-5",
                source_path="ERCOTAPI/sources/official/planning_guides/05-071126.docx",
                chunk_index=1,
                text=(
                    "Any Entity subject to Section 5.2.1 must initiate a Generator "
                    "Interconnection or Modification by submitting a completed request in RIOO."
                ),
            ),
            chunk(
                "generation-studies",
                **common_guide,
                document_id="planning-guide-section-5",
                source_path="ERCOTAPI/sources/official/planning_guides/05-071126.docx",
                chunk_index=5,
                text=(
                    "The provisions in this Section establish the procedures for conducting "
                    "the Security Screening Study and Full Interconnection Study."
                ),
            ),
            chunk(
                "generation-commissioning",
                **common_guide,
                document_id="planning-guide-section-5",
                source_path="ERCOTAPI/sources/official/planning_guides/05-071126.docx",
                chunk_index=11,
                text=(
                    "5.5Generator Commissioning and Continuing Operations "
                    "(1)For each interconnecting Generation Resource, conditions apply."
                ),
            ),
            chunk(
                "resource-handbook",
                source_kind="Resource Integration",
                source_path="chatbot_ercot_all_in_one/ercot_sources/ercotRIhandbook.txt",
                filename="ercotRIhandbook.txt",
                document_id="resource-interconnection-handbook",
                collections=["general", "resource_integration"],
                text=(
                    "The resource interconnection process has been divided into the following "
                    "three stages for interactions among the developer, ERCOT, and TSPs."
                ),
            ),
            chunk(
                "load-introduction",
                **common_guide,
                document_id="planning-guide-section-9",
                source_path="ERCOTAPI/sources/official/planning_guides/09-071126.docx",
                chunk_index=0,
                text=(
                    "Section 9 defines the requirements and processes used to facilitate new "
                    "or modified Large Load interconnections with the ERCOT System."
                ),
            ),
            chunk(
                "load-batch-zero",
                **common_guide,
                document_id="planning-guide-section-9",
                source_path="ERCOTAPI/sources/official/planning_guides/09-071126.docx",
                chunk_index=8,
                text=(
                    "9.3.1Batch Zero Process Overview and Timelines "
                    "(1)The Batch Zero Interconnection Study is system-wide."
                ),
            ),
            chunk(
                "load-refinement",
                **common_guide,
                document_id="planning-guide-section-9",
                source_path="ERCOTAPI/sources/official/planning_guides/09-071126.docx",
                chunk_index=11,
                text=(
                    "9.5Batch Zero Study Refinement and Delivery of Transmission Plan "
                    "(1)ERCOT updates the Batch Zero Interconnection Study."
                ),
            ),
            chunk(
                "unrelated-planning",
                **common_guide,
                source_path="ERCOTAPI/sources/official/planning_guides/03-071126.docx",
                retrieval_score=0.5,
                text="Section 3 addresses Regional Planning Group project review.",
            ),
        ]
        bundle = retrieve_requirement_evidence(
            "What is the interconnection process in ERCOT?",
            index(records),
            top_k=7,
            candidate_chunks=[records[-1]],
            as_of="2026-07-23",
        )

        selected = {item["chunk_id"]: item for item in bundle["chunks"]}
        self.assertEqual(
            set(selected),
            {
                "generation-initiation",
                "generation-studies",
                "generation-commissioning",
                "resource-handbook",
                "load-introduction",
                "load-batch-zero",
                "load-refinement",
            },
        )
        self.assertEqual(selected["generation-initiation"]["section_number"], "5.2.2")
        self.assertEqual(selected["generation-studies"]["section_number"], "5.3")
        self.assertEqual(selected["generation-commissioning"]["section_number"], "5.5")
        self.assertEqual(selected["load-introduction"]["section_number"], "9.1")
        self.assertEqual(selected["load-batch-zero"]["section_number"], "9.3.1")
        self.assertEqual(selected["load-refinement"]["section_number"], "9.5")

    def test_specific_interconnection_requirement_does_not_force_process_anchors(self) -> None:
        common = {
            "source_kind": "Planning Guide",
            "source_category": "planning_guide_uploads",
            "source_path": "ERCOTAPI/sources/official/planning_guides/05-071126.docx",
            "effective_date": "2026-07-11",
            "document_id": "planning-guide-section-5",
            "collections": ["general", "planning"],
        }
        anchors = [
            chunk(
                "generation-initiation",
                **common,
                chunk_index=1,
                text="Entities must initiate a Generator Interconnection or Modification.",
            ),
            chunk(
                "generation-studies",
                **common,
                chunk_index=5,
                text=(
                    "The provisions in this Section establish the procedures for conducting "
                    "the Security Screening Study and Full Interconnection Study."
                ),
            ),
            chunk(
                "generation-commissioning",
                **common,
                chunk_index=11,
                text=(
                    "5.5Generator Commissioning and Continuing Operations "
                    "(1)For each interconnecting Generation Resource, conditions apply."
                ),
            ),
        ]
        cases = (
            (
                "What reactive capability is required for a new generator interconnection?",
                chunk(
                    "reactive-requirement",
                    **common,
                    chunk_index=9,
                    retrieval_score=1.0,
                    text="The reactive capability requirement applies at the POI.",
                ),
            ),
            (
                "What equipment is required at an ERCOT generator interconnection?",
                chunk(
                    "equipment-requirement",
                    **common,
                    chunk_index=4,
                    retrieval_score=1.0,
                    text="Section 5.2.11 specifies required interconnection equipment.",
                ),
            ),
        )
        for question, specific in cases:
            with self.subTest(question=question):
                bundle = retrieve_requirement_evidence(
                    question,
                    index([*anchors, specific]),
                    top_k=3,
                    candidate_chunks=[specific],
                    as_of="2026-07-23",
                )
                self.assertEqual(
                    [item["chunk_id"] for item in bundle["chunks"]],
                    [specific["chunk_id"]],
                )
                self.assertFalse(bundle["chunks"][0].get("retrieval_anchor"))

    def test_interconnection_anchors_do_not_inject_a_future_guide_for_as_of_date(self) -> None:
        current = chunk(
            "future-current-guide",
            source_kind="Planning Guide",
            source_category="planning_guide_uploads",
            source_path="ERCOTAPI/sources/official/planning_guides/05-071126.docx",
            effective_date="2026-07-11",
            chunk_index=1,
            text=(
                "An Entity must initiate a Generator Interconnection or Modification "
                "by submitting the current request."
            ),
        )
        historical = chunk(
            "historical-process",
            source_kind="Planning Guide",
            source_category="official_downloads",
            source_path="ERCOTAPI/NEWS/official/planning-guide/2025/section-5.pdf",
            effective_date="2025-01-01",
            retrieval_score=1.0,
            text="The 2025 generator interconnection process applied.",
        )
        undated_handbook = chunk(
            "current-undated-handbook",
            source_kind="Resource Integration",
            source_path="chatbot_ercot_all_in_one/ercot_sources/ercotRIhandbook.txt",
            filename="ercotRIhandbook.txt",
            text=(
                "The current process has been divided into the following three stages."
            ),
        )

        bundle = retrieve_requirement_evidence(
            "What is the ERCOT generator interconnection process?",
            index([current, historical, undated_handbook]),
            top_k=3,
            candidate_chunks=[historical],
            as_of="2025-07-01",
        )

        self.assertEqual(
            [item["chunk_id"] for item in bundle["chunks"]],
            ["historical-process"],
        )

    def test_citation_audit_rejects_invented_evidence_ids(self) -> None:
        records = [{"evidence_id": "E1"}, {"evidence_id": "E2"}]

        valid = validate_answer_citations("The rule applies at the POI [E1].", records)
        invalid = validate_answer_citations("The rule applies at the POI [E9].", records)

        self.assertTrue(valid["passed"])
        self.assertFalse(invalid["passed"])
        self.assertEqual(invalid["invalid_evidence_ids"], ["E9"])

        sparse = validate_answer_citations(
            "The first material requirement is supported [E1].\n"
            "A second material requirement has no supporting evidence citation.",
            records,
        )
        self.assertFalse(sparse["passed"])
        self.assertEqual(sparse["claim_line_coverage"], 0.5)

    def test_combines_governing_procedure_and_change_evidence(self) -> None:
        records = [
            chunk(
                "guide",
                source_kind="Planning Guide",
                source_category="planning_guide_uploads",
                effective_date="2026-01-01",
            ),
            chunk(
                "protocol",
                source_kind="Protocol",
                source_category="nodal_protocol_uploads",
                effective_date="2026-02-01",
            ),
            chunk("dwg", source_kind="DWG", document_status="Approved"),
            chunk(
                "pgrr",
                source_kind="PGRR",
                document_number="PGRR147",
                document_status="Pending",
            ),
        ]

        bundle = retrieve_requirement_evidence(
            "What are ERCOT reactive power capability requirements and related changes?",
            index(records),
            top_k=4,
            query_embedder=lambda _question: [1.0, 0.0],
            as_of="2026-07-22",
        )

        self.assertEqual({item["chunk_id"] for item in bundle["chunks"]}, {"guide", "protocol", "dwg", "pgrr"})
        roles = {item["evidence_role"] for item in bundle["chunks"]}
        self.assertIn("current_governing_requirement", roles)
        self.assertIn("procedure_or_engineering_criteria", roles)
        self.assertIn("related_change_record", roles)
        self.assertIn("Never present an xRR", bundle["answer_contract"])

    def test_notices_are_excluded_from_engineering_retrieval(self) -> None:
        notice = chunk(
            "notice",
            source_kind="Market Notice",
            source_path="ERCOTAPI/NEWS/official/market-notices/2026/notice.html",
            original_url="https://www.ercot.com/services/comm/mkt_notices/M-A072226-01",
        )
        guide = chunk(
            "guide",
            source_kind="Planning Guide",
            source_category="planning_guide_uploads",
        )

        results = retrieve_chunks(
            "reactive power capability",
            index([notice, guide]),
            top_k=2,
            query_embedder=lambda _question: [1.0, 0.0],
        )

        self.assertEqual([item["chunk_id"] for item in results], ["guide"])

    def test_section_and_page_are_in_citation(self) -> None:
        record = chunk(
            "located",
            source_kind="Protocol",
            section_number="3.15",
            section_title="Reactive Power Capability",
            page_start=12,
            page_end=13,
        )

        citation = format_citation(record)

        self.assertIn("Section 3.15 Reactive Power Capability", citation)
        self.assertIn("pages 12-13", citation)

    def test_change_report_is_limited_to_requested_section_family(self) -> None:
        old = chunk(
            "june-guide",
            source_kind="Planning Guide",
            title="Combined",
            published_date="2026-05-29",
            original_url="https://www.ercot.com/June-1-2026-Planning-Guide.pdf",
            text=(
                "8.1 Other Requirement\nOld unrelated language.\n\n"
                "9.1 Target Requirement\nOld target language."
            ),
        )
        new = chunk(
            "july-guide",
            source_kind="Planning Guide",
            title="Combined",
            published_date="2026-06-18",
            original_url="https://www.ercot.com/July-1-2026-Planning-Guide.pdf",
            text=(
                "8.1 Other Requirement\nNew unrelated language.\n\n"
                "9.1 Target Requirement\nNew target language."
            ),
        )

        bundle = retrieve_requirement_evidence(
            "What changed in Planning Guide Section 9?",
            index([old, new]),
            top_k=2,
            query_embedder=lambda _question: [1.0, 0.0],
            as_of="2026-07-22",
        )

        self.assertEqual(len(bundle["change_reports"]), 1)
        report = bundle["change_reports"][0]
        self.assertEqual(report["counts"]["modified"], 1)
        self.assertEqual(report["all_counts"]["modified"], 2)
        self.assertTrue(
            all(
                str(change["section_number"]) == "9"
                or str(change["section_number"]).startswith("9.")
                for change in report["changes"]
            )
        )

    def test_xrr_comments_are_not_auto_diffed_as_versions(self) -> None:
        first = chunk(
            "pgrr-comment-one",
            source_kind="PGRR",
            title="145PGRR-79 Mesquite Comments",
            source_page_url="https://www.ercot.com/mktrules/issues/PGRR145",
            published_date="2026-05-05",
            text="9.1 Requirement\nFirst stakeholder proposal.",
        )
        second = chunk(
            "pgrr-comment-two",
            source_kind="PGRR",
            title="145PGRR-80 AEPSC Comments",
            source_page_url="https://www.ercot.com/mktrules/issues/PGRR145",
            published_date="2026-05-06",
            text="9.1 Requirement\nDifferent stakeholder proposal.",
        )

        bundle = retrieve_requirement_evidence(
            "What changed in PGRR145?",
            index([first, second]),
            top_k=2,
            query_embedder=lambda _question: [1.0, 0.0],
            as_of="2026-07-22",
        )

        self.assertEqual(bundle["change_reports"], [])


if __name__ == "__main__":
    unittest.main()
