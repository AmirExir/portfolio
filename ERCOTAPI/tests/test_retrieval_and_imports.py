"""Authority-aware retrieval, citations, and import-safety tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

from ERCOTAPI.rag_ingestion.retrieval import (
    LoadedIndex,
    format_context,
    format_source_list,
    load_index,
    retrieve_chunks,
)


class RetrievalTests(unittest.TestCase):
    def make_index(self) -> LoadedIndex:
        common = {
            "text": "Capacity requirement applies to the ERCOT market.",
            "title": "Capacity Requirement",
            "chunk_index": 0,
        }
        generated = {
            **common,
            "chunk_id": "generated-chunk",
            "source_path": "ERCOTAPI/news_summaries/capacity.txt",
            "filename": "capacity.txt",
            "source_authority": "Generated",
            "source_kind": "News Summary",
            "is_generated": True,
            "collections": ["news", "market"],
        }
        official = {
            **common,
            "chunk_id": "official-chunk",
            "source_path": "ERCOTAPI/NEWS/official/NPRR1234.txt",
            "filename": "NPRR1234.txt",
            "source_authority": "ERCOT",
            "source_kind": "NPRR",
            "document_number": "NPRR1234",
            "document_status": "Approved",
            "effective_date": "2026-07-01",
            "published_date": "2026-06-15",
            "revision": "2",
            "is_generated": False,
            "collections": ["general", "protocols"],
            "original_url": "https://www.ercot.com/files/NPRR1234",
        }
        return LoadedIndex(
            chunks=[generated, official],
            embeddings=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype="float32"),
            embedding_model="test-model",
            generation_id="test-generation",
            source="central",
            collections=(),
            state_token=("test-generation", 1),
        )

    def test_authoritative_ercot_chunk_outranks_equivalent_generated_summary(self) -> None:
        results = retrieve_chunks(
            "What is the capacity requirement?",
            self.make_index(),
            top_k=2,
            query_embedder=lambda _question: [1.0, 0.0],
        )

        self.assertEqual([result["chunk_id"] for result in results], ["official-chunk", "generated-chunk"])
        self.assertGreater(results[0]["retrieval_score"], results[1]["retrieval_score"])

    def test_collection_filter_and_citation_preserve_provenance(self) -> None:
        results = retrieve_chunks(
            "capacity requirement",
            self.make_index(),
            top_k=5,
            collections="protocols",
            query_embedder=lambda _question: [1.0, 0.0],
        )

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result["chunk_id"], "official-chunk")
        citation = result["citation"]
        for expected in (
            "ERCOT",
            "NPRR",
            "NPRR1234",
            "Status Approved",
            "Effective 2026-07-01",
            "Published 2026-06-15",
            "Revision 2",
            "ERCOTAPI/NEWS/official/NPRR1234.txt",
            "chunk 1",
            "https://www.ercot.com/files/NPRR1234",
        ):
            self.assertIn(expected, citation)
        context = format_context(results)
        self.assertIn(f"Citation: {citation}", context)
        self.assertIn("Capacity requirement applies", context)
        footer = format_source_list([result, result])
        self.assertEqual(footer.count("- [ERCOT"), 1)
        self.assertIn("[open ERCOT source](<https://www.ercot.com/files/NPRR1234>)", footer)

    def test_newer_effective_revision_wins_an_equal_relevance_tie(self) -> None:
        index = self.make_index()
        older = {
            **index.chunks[1],
            "chunk_id": "aaa-older",
            "effective_date": "2020-01-01",
            "revision": "1",
        }
        newer = {
            **index.chunks[1],
            "chunk_id": "zzz-newer",
            "effective_date": "2026-01-01",
            "revision": "2",
        }
        versioned = LoadedIndex(
            chunks=[older, newer],
            embeddings=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype="float32"),
            embedding_model="test-model",
            generation_id="test-generation",
            source="central",
            collections=(),
            state_token=("test-generation", 1),
        )

        results = retrieve_chunks(
            "capacity requirement",
            versioned,
            top_k=2,
            query_embedder=lambda _question: [1.0, 0.0],
        )

        self.assertEqual([item["chunk_id"] for item in results], ["zzz-newer", "aaa-older"])

    def test_current_upload_bundle_outranks_equal_historical_static_text(self) -> None:
        index = self.make_index()
        historical = {
            **index.chunks[1],
            "chunk_id": "aaa-historical",
            "source_category": "nodal_protocols",
            "source_path": "chatbot_ercot_nodalprotocols/ercotnodals_part1.txt",
        }
        current = {
            **index.chunks[1],
            "chunk_id": "zzz-current",
            "source_category": "nodal_protocol_uploads",
            "source_path": "ERCOTAPI/sources/official/nodal_protocols/01-020126_Nodal.docx",
        }
        versioned = LoadedIndex(
            chunks=[historical, current],
            embeddings=np.asarray([[1.0, 0.0], [0.95, 0.31225]], dtype="float32"),
            embedding_model="test-model",
            generation_id="test-generation",
            source="central",
            collections=(),
            state_token=("test-generation", 1),
        )

        results = retrieve_chunks(
            "capacity requirement",
            versioned,
            top_k=2,
            query_embedder=lambda _question: [1.0, 0.0],
        )

        self.assertEqual(
            [item["chunk_id"] for item in results],
            ["zzz-current"],
        )

        historical_results = retrieve_chunks(
            "What did the 2023 Nodal Protocol require for capacity?",
            versioned,
            top_k=2,
            query_embedder=lambda _question: [1.0, 0.0],
        )
        self.assertEqual(
            [item["chunk_id"] for item in historical_results],
            ["aaa-historical", "zzz-current"],
        )

        current_year_results = retrieve_chunks(
            f"What does the {datetime.now().year} Nodal Protocol require for capacity?",
            versioned,
            top_k=2,
            query_embedder=lambda _question: [1.0, 0.0],
        )
        self.assertEqual(
            [item["chunk_id"] for item in current_year_results],
            ["zzz-current"],
        )

        current_as_of_results = retrieve_chunks(
            f"As of July {datetime.now().year}, what does the current Nodal Protocol require?",
            versioned,
            top_k=2,
            query_embedder=lambda _question: [1.0, 0.0],
        )
        self.assertEqual(
            [item["chunk_id"] for item in current_as_of_results],
            ["zzz-current"],
        )

    def test_current_upload_hides_archived_full_manual_but_not_revision_request(self) -> None:
        common = {
            "source_authority": "ERCOT",
            "is_generated": False,
            "collections": ["general", "planning"],
            "chunk_index": 0,
        }
        archived_manual = {
            **common,
            "chunk_id": "archived-2023-guide",
            "source_category": "official_downloads",
            "source_kind": "Planning Guide",
            "source_path": "ERCOTAPI/NEWS/official/planning-guide/2023/guide.pdf",
            "title": "June 2023 Planning Guide",
            "text": "Section 9 is reserved.",
        }
        revision_request = {
            **common,
            "chunk_id": "current-pgrr",
            "source_category": "official_downloads",
            "source_kind": "PGRR",
            "source_path": "ERCOTAPI/NEWS/official/pgrr/2026/PGRR143.docx",
            "title": "PGRR143",
            "text": "Current Planning Guide revision request details.",
        }
        current_manual = {
            **common,
            "chunk_id": "current-section-9",
            "source_category": "planning_guide_uploads",
            "source_kind": "Planning Guide",
            "source_path": "ERCOTAPI/sources/official/planning_guides/09-071126.docx",
            "title": "Current Planning Guide Section 9",
            "text": "Section 9 covers Large Load additions.",
        }
        index = LoadedIndex(
            chunks=[archived_manual, revision_request, current_manual],
            embeddings=np.asarray([[1.0, 0.0]] * 3, dtype="float32"),
            embedding_model="test-model",
            generation_id="test-generation",
            source="central",
            collections=(),
            state_token=("test-generation", 1),
        )

        current_results = retrieve_chunks(
            "What is Planning Guide Section 9?",
            index,
            top_k=3,
            query_embedder=lambda _question: [1.0, 0.0],
        )
        self.assertEqual(
            {item["chunk_id"] for item in current_results},
            {"current-pgrr", "current-section-9"},
        )
        self.assertEqual(current_results[0]["chunk_id"], "current-section-9")

        historical_results = retrieve_chunks(
            "What did the 2023 Planning Guide say about Section 9?",
            index,
            top_k=3,
            query_embedder=lambda _question: [1.0, 0.0],
        )
        self.assertEqual(
            {item["chunk_id"] for item in historical_results},
            {"archived-2023-guide", "current-pgrr", "current-section-9"},
        )

    def test_explicit_planning_section_routes_to_matching_current_split_file(self) -> None:
        common = {
            "source_authority": "ERCOT",
            "source_kind": "Planning Guide",
            "source_category": "planning_guide_uploads",
            "is_generated": False,
            "collections": ["general", "planning"],
            "chunk_index": 0,
        }
        section_one = {
            **common,
            "chunk_id": "section-one-toc",
            "source_path": "ERCOTAPI/sources/official/planning_guides/01-020126.docx",
            "title": "Planning Guide Section 1",
            "text": "Table of contents lists Section 9 Large Load additions.",
        }
        section_nine = {
            **common,
            "chunk_id": "section-nine-current",
            "source_category": "official_downloads",
            "source_path": "ERCOTAPI/NEWS/official/planning-guide/2026/content-hash.docx",
            "aliases": ["ERCOTAPI/sources/official/planning_guides/09-071126.docx"],
            "title": "Planning Guide Section 9",
            "text": "Current Section 9 governs Large Load additions.",
        }
        index = LoadedIndex(
            chunks=[section_one, section_nine],
            embeddings=np.asarray([[1.0, 0.0], [0.9, 0.43589]], dtype="float32"),
            embedding_model="test-model",
            generation_id="test-generation",
            source="central",
            collections=(),
            state_token=("test-generation", 1),
        )

        results = retrieve_chunks(
            "What is ERCOT Planning Guide Section 9?",
            index,
            top_k=2,
            query_embedder=lambda _question: [1.0, 0.0],
        )
        self.assertEqual([item["chunk_id"] for item in results], ["section-nine-current"])

    def test_current_dwg_upload_does_not_suppress_the_distinct_sswg_manual(self) -> None:
        common = {
            "source_authority": "ERCOT",
            "is_generated": False,
            "collections": ["general", "dwg_sswg"],
            "chunk_index": 0,
        }
        sswg = {
            **common,
            "chunk_id": "sswg-manual",
            "source_category": "authoritative_static",
            "source_kind": "SSWG",
            "source_path": "chatbot_ercot_all_in_one/ercot_sources/DWG_SSWG_Manuals.txt",
            "title": "SSWG Procedure Manual",
            "text": "SSWG steady-state case building requirement.",
        }
        dwg = {
            **common,
            "chunk_id": "dwg-manual",
            "source_category": "dwg_sswg_uploads",
            "source_kind": "DWG",
            "source_path": "ERCOTAPI/sources/official/dwg_sswg/current.docx",
            "title": "DWG Procedure Manual",
            "text": "DWG dynamics model requirement.",
        }
        index = LoadedIndex(
            chunks=[sswg, dwg],
            embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32"),
            embedding_model="test-model",
            generation_id="test-generation",
            source="central",
            collections=("dwg_sswg",),
            state_token=("test-generation", 1),
        )

        results = retrieve_chunks(
            "What does the SSWG manual require for case building?",
            index,
            top_k=2,
            query_embedder=lambda _question: [1.0, 0.0],
        )

        self.assertEqual(
            [item["chunk_id"] for item in results],
            ["sswg-manual", "dwg-manual"],
        )

    def test_current_dwg_upload_suppresses_only_dwg_half_of_combined_manual(self) -> None:
        common = {
            "source_authority": "ERCOT",
            "source_category": "authoritative_static",
            "source_kind": "SSWG",
            "source_path": "chatbot_ercot_all_in_one/ercot_sources/DWG_SSWG_Manuals.txt",
            "content_hash": "combined-manual-hash",
            "is_generated": False,
            "collections": ["general", "dwg_sswg"],
        }
        chunks = [
            {
                **common,
                "chunk_id": "sswg-section",
                "chunk_index": 0,
                "title": "SSWG Procedure Manual",
                "text": "SSWG steady-state case building requirement.",
            },
            {
                **common,
                "chunk_id": "old-dwg-heading",
                "chunk_index": 1,
                "title": "Combined procedure manuals",
                "text": "Dynamics Working Group\nProcedure Manual\nRevision 23",
            },
            {
                **common,
                "chunk_id": "old-dwg-continuation",
                "chunk_index": 2,
                "title": "Combined procedure manuals",
                "text": "Dynamic model quality test guideline requirements.",
            },
            {
                **common,
                "chunk_id": "current-dwg",
                "chunk_index": 0,
                "source_category": "dwg_sswg_uploads",
                "source_kind": "DWG",
                "source_path": "ERCOTAPI/sources/official/dwg_sswg/current.docx",
                "content_hash": "current-manual-hash",
                "title": "DWG Procedure Manual Revision 25",
                "text": "Current dynamic model requirement.",
            },
        ]
        index = LoadedIndex(
            chunks=chunks,
            embeddings=np.asarray([[1.0, 0.0]] * len(chunks), dtype="float32"),
            embedding_model="test-model",
            generation_id="test-generation",
            source="central",
            collections=("dwg_sswg",),
            state_token=("test-generation", 1),
        )

        current_results = retrieve_chunks(
            "What does the DWG manual require?",
            index,
            top_k=4,
            query_embedder=lambda _question: [1.0, 0.0],
        )
        self.assertEqual(
            {item["chunk_id"] for item in current_results},
            {"sswg-section", "current-dwg"},
        )

        collection_filtered_index = LoadedIndex(
            chunks=chunks[:3],
            embeddings=np.asarray([[1.0, 0.0]] * 3, dtype="float32"),
            embedding_model="test-model",
            generation_id="test-generation",
            source="central",
            collections=("planning",),
            state_token=("test-generation", 1),
        )
        collection_filtered_results = retrieve_chunks(
            "What does the current SSWG manual require?",
            collection_filtered_index,
            top_k=3,
            query_embedder=lambda _question: [1.0, 0.0],
        )
        self.assertEqual(
            [item["chunk_id"] for item in collection_filtered_results],
            ["sswg-section"],
        )

        historical_results = retrieve_chunks(
            "What did the 2025 DWG manual require?",
            index,
            top_k=4,
            query_embedder=lambda _question: [1.0, 0.0],
        )
        self.assertEqual(
            {item["chunk_id"] for item in historical_results},
            {item["chunk_id"] for item in chunks},
        )

        revision_results = retrieve_chunks(
            "What did DWG Procedure Manual Revision 23 require?",
            index,
            top_k=4,
            query_embedder=lambda _question: [1.0, 0.0],
        )
        self.assertEqual(
            {item["chunk_id"] for item in revision_results},
            {item["chunk_id"] for item in chunks},
        )

    def test_legacy_fallback_never_labels_protocol_text_as_operations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chunks_path = root / "chunks.json"
            embeddings_path = root / "embeddings.npy"
            chunks_path.write_text(
                json.dumps(
                    [
                        {
                            "text": "Nodal Protocol settlement rule.",
                            "source": "ercotnodals.txt",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            np.save(embeddings_path, np.asarray([[1.0, 0.0]], dtype="float32"))

            central_only = load_index(
                "protocols",
                index_dir=root / "missing-store",
                legacy_chunks_path=chunks_path,
                legacy_embeddings_path=embeddings_path,
            )
            protocols = load_index(
                "protocols",
                index_dir=root / "missing-store",
                allow_legacy=True,
                legacy_chunks_path=chunks_path,
                legacy_embeddings_path=embeddings_path,
            )
            operations = load_index(
                "operations",
                index_dir=root / "missing-store",
                allow_legacy=True,
                legacy_chunks_path=chunks_path,
                legacy_embeddings_path=embeddings_path,
            )

        self.assertEqual(central_only.source, "missing")
        self.assertEqual(central_only.chunks, [])
        self.assertEqual(len(protocols.chunks), 1)
        self.assertTrue(protocols.ready)
        self.assertEqual(operations.chunks, [])
        self.assertFalse(operations.ready)

    def test_exact_decimal_section_outweighs_a_prefix_section_near_tie(self) -> None:
        common = {
            "title": "ERCOT Planning Guide",
            "source_authority": "ERCOT",
            "source_kind": "Planning Guide",
            "is_generated": False,
            "collections": ["general", "planning"],
            "chunk_index": 0,
        }
        prefix_only = {
            **common,
            "chunk_id": "aaa-prefix-only",
            "source_path": "planning-guide-9-10.txt",
            "text": "Section 9.10 describes an unrelated planning requirement.",
        }
        exact = {
            **common,
            "chunk_id": "zzz-exact-section",
            "source_path": "planning-guide-9-1.txt",
            "text": "Section 9.1 describes the large-load planning requirement.",
        }
        index = LoadedIndex(
            chunks=[prefix_only, exact],
            embeddings=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype="float32"),
            embedding_model="test-model",
            generation_id="test-generation",
            source="central",
            collections=(),
            state_token=("test-generation", 1),
        )

        results = retrieve_chunks(
            "What does Planning Guide Section 9.1 require?",
            index,
            top_k=2,
            query_embedder=lambda _question: [1.0, 0.0],
        )

        self.assertEqual(
            [item["chunk_id"] for item in results],
            ["zzz-exact-section", "aaa-prefix-only"],
        )

    def test_ercot_domain_phrases_and_aliases_route_equal_vector_ties(self) -> None:
        cases = (
            (
                "What planning guide rules govern load additions?",
                "ERCOT Planning Guide load-addition rules.",
                "Planning notes from another guide discuss load-addition rules.",
            ),
            (
                "What does the nodal operating guide require for outages?",
                "The Nodal Operating Guide states the outage requirement.",
                "Nodal outage operating practices appear in another guide.",
            ),
            (
                "Which nodal protocols define this settlement rule?",
                "The ERCOT Nodal Protocols define this settlement rule.",
                "Nodal settlement notes cite a separate protocol rule.",
            ),
            (
                "Explain the generator interconnection timeline.",
                "The Generator Interconnection timeline has three phases.",
                "Generator schedules and the interconnection queue have timelines.",
            ),
            (
                "Explain the resource interconnection handbook process.",
                "The Resource Interconnection Handbook explains the process.",
                "Resource notes point to a separate interconnection process handbook.",
            ),
            (
                "What is a GINR?",
                "A Generation Interconnection or Change Request starts the process.",
                "This unrelated glossary entry has no interconnection definition.",
            ),
            (
                "What is an FIS?",
                "A Full Interconnection Study evaluates the proposed resource.",
                "This unrelated glossary entry has no study definition.",
            ),
        )
        for question, domain_text, distractor_text in cases:
            with self.subTest(question=question):
                common = {
                    "title": "ERCOT reference",
                    "source_authority": "ERCOT",
                    "source_kind": "ERCOT Reference",
                    "is_generated": False,
                    "collections": ["general"],
                    "chunk_index": 0,
                }
                distractor = {
                    **common,
                    "chunk_id": "aaa-distractor",
                    "source_path": "distractor.txt",
                    "text": distractor_text,
                }
                domain = {
                    **common,
                    "chunk_id": "zzz-domain",
                    "source_path": "domain.txt",
                    "text": domain_text,
                }
                index = LoadedIndex(
                    chunks=[distractor, domain],
                    embeddings=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype="float32"),
                    embedding_model="test-model",
                    generation_id="test-generation",
                    source="central",
                    collections=(),
                    state_token=("test-generation", 1),
                )

                results = retrieve_chunks(
                    question,
                    index,
                    top_k=2,
                    query_embedder=lambda _question: [1.0, 0.0],
                )

                self.assertEqual(results[0]["chunk_id"], "zzz-domain")
                self.assertGreater(
                    results[0]["retrieval_score"],
                    results[1]["retrieval_score"],
                )

    def test_lexical_routing_does_not_override_a_clear_vector_lead(self) -> None:
        common = {
            "title": "ERCOT reference",
            "source_authority": "ERCOT",
            "source_kind": "ERCOT Reference",
            "is_generated": False,
            "collections": ["general", "planning"],
            "chunk_index": 0,
        }
        vector_match = {
            **common,
            "chunk_id": "vector-match",
            "source_path": "semantic-match.txt",
            "text": "A semantically similar passage.",
        }
        lexical_match = {
            **common,
            "chunk_id": "lexical-match",
            "source_path": "planning-guide.txt",
            "text": "ERCOT Planning Guide Section 9.1 requirements.",
        }
        index = LoadedIndex(
            chunks=[vector_match, lexical_match],
            embeddings=np.asarray([[1.0, 0.0], [0.9, 0.4358899]], dtype="float32"),
            embedding_model="test-model",
            generation_id="test-generation",
            source="central",
            collections=(),
            state_token=("test-generation", 1),
        )

        results = retrieve_chunks(
            "What does Planning Guide Section 9.1 require?",
            index,
            top_k=2,
            query_embedder=lambda _question: [1.0, 0.0],
        )

        self.assertEqual(results[0]["chunk_id"], "vector-match")
        self.assertGreater(results[0]["vector_score"], results[1]["vector_score"])


class ImportSafetyTests(unittest.TestCase):
    def test_package_import_has_no_network_credentials_or_filesystem_side_effects(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            script = """
import pathlib
import sys

cwd = pathlib.Path.cwd()
assert not list(cwd.iterdir())
import ERCOTAPI.rag_ingestion
assert "openai" not in sys.modules
assert not list(cwd.iterdir())
print("import-safe")
"""
            environment = os.environ.copy()
            environment.pop("OPENAI_API_KEY", None)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            environment["PYTHONPATH"] = str(repo_root)
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=temporary,
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "import-safe")


if __name__ == "__main__":
    unittest.main()
