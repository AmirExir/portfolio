"""Authority-aware retrieval, citations, and import-safety tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
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

            protocols = load_index(
                "protocols",
                index_dir=root / "missing-store",
                legacy_chunks_path=chunks_path,
                legacy_embeddings_path=embeddings_path,
            )
            operations = load_index(
                "operations",
                index_dir=root / "missing-store",
                legacy_chunks_path=chunks_path,
                legacy_embeddings_path=embeddings_path,
            )

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
