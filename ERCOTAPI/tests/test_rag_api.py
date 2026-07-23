"""Endpoint-level behavior for the central ERCOT retrieval API."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
from fastapi import HTTPException

from chatbot_ercot_all_in_one import ercot_rag_api as api
from ERCOTAPI.rag_ingestion.retrieval import LoadedIndex
from ERCOTAPI.rag_ingestion.startup import CentralIndexUnavailable


def _index(
    chunks: list[dict],
    *,
    source: str = "central",
    generation: str = "test-generation",
) -> LoadedIndex:
    vectors = np.asarray([[1.0, 0.0] for _ in chunks], dtype="float32")
    return LoadedIndex(
        chunks=chunks,
        embeddings=vectors,
        embedding_model="test-model",
        generation_id=generation if source == "central" else None,
        source=source,
        collections=(),
        state_token=("test-generation", 1),
    )


def _chunk(identifier: str, text: str, vector_score: float) -> dict:
    return {
        "chunk_id": identifier,
        "text": text,
        "title": identifier,
        "source_path": f"ERCOTAPI/{identifier}.txt",
        "source_authority": "ERCOT",
        "source_kind": "ERCOT Reference",
        "is_generated": False,
        "collections": ["general"],
        "citation": f"[{identifier}]",
        "vector_score": vector_score,
        "retrieval_score": vector_score + 0.04,
    }


class RagApiTests(unittest.TestCase):
    def test_health_reports_central_startup_failure_without_legacy_degradation(self) -> None:
        with mock.patch.object(api, "_get_index", side_effect=RuntimeError("central missing")):
            result = api.health()

        self.assertFalse(result["ok"])
        self.assertTrue(result["degraded"])
        self.assertFalse(result["all_collections_ready"])
        self.assertFalse(result["collections"]["operations"]["ready"])
        self.assertIsNone(result["index_source"])
        self.assertIn("central missing", result["error"])
        self.assertIn("market", result["unavailable_collections"])

    def test_collection_loader_uses_central_only_startup_path(self) -> None:
        central = _index([_chunk("general", "ready", 1.0)], source="central")

        with mock.patch.object(api, "load_startup_index", return_value=central) as load:
            result = api._load_collection("general")

        self.assertIs(result, central)
        load.assert_called_once_with("general")

    def test_failed_generation_reload_never_serves_cached_old_snapshot(self) -> None:
        prior_indexes = api.INDEXES
        prior_state = api.DATA_STATE
        prior_error = api.LOAD_ERROR
        self.addCleanup(setattr, api, "INDEXES", prior_indexes)
        self.addCleanup(setattr, api, "DATA_STATE", prior_state)
        self.addCleanup(setattr, api, "LOAD_ERROR", prior_error)
        api.INDEXES = {"general": _index([_chunk("old", "old text", 1.0)])}
        api.DATA_STATE = ("old-generation", 1)
        api.LOAD_ERROR = ""

        with (
            mock.patch.object(api, "_file_state", return_value=("new-generation", 2)),
            mock.patch.object(
                api,
                "_load_collection",
                side_effect=CentralIndexUnavailable("new generation rejected"),
            ),
        ):
            with self.assertRaisesRegex(CentralIndexUnavailable, "rejected"):
                api._get_index("general")

        self.assertIn("new generation rejected", api.LOAD_ERROR)

    def test_state_switch_during_load_is_retried_not_cached_under_new_state(self) -> None:
        prior_indexes = api.INDEXES
        prior_state = api.DATA_STATE
        prior_error = api.LOAD_ERROR
        self.addCleanup(setattr, api, "INDEXES", prior_indexes)
        self.addCleanup(setattr, api, "DATA_STATE", prior_state)
        self.addCleanup(setattr, api, "LOAD_ERROR", prior_error)
        api.INDEXES = {}
        api.DATA_STATE = ()
        api.LOAD_ERROR = ""
        generation_a = _index(
            [_chunk("generation-a", "generation A", 1.0)],
            generation="generation-a",
        )

        with (
            mock.patch.object(
                api,
                "_file_state",
                side_effect=[("generation-a", 1), ("generation-b", 2)],
            ),
            mock.patch.object(api, "_load_collection", return_value=generation_a),
        ):
            with self.assertRaisesRegex(RuntimeError, "changed while it was loading"):
                api._get_index("general")

        self.assertEqual(api.INDEXES, {})
        self.assertIn("retry", api.LOAD_ERROR)

    def test_central_health_uses_manifest_counts_without_loading_every_matrix(self) -> None:
        general = _index([_chunk("general", "ready", 1.0)], source="central")
        counts = {name: 0 for name in api.COLLECTION_NAMES}
        counts.update({"general": 5, "protocols": 2, "market": 1})

        with (
            mock.patch.object(api, "_get_index", return_value=general) as get_index,
            mock.patch.object(
                api,
                "_manifest_collection_counts",
                return_value=("test-generation", counts),
            ) as manifest_counts,
        ):
            result = api.health()

        get_index.assert_called_once_with("general")
        manifest_counts.assert_called_once_with("test-generation")
        self.assertEqual(result["collections"]["protocols"]["chunks_loaded"], 2)
        self.assertTrue(result["collections"]["market"]["ready"])

    def test_empty_news_collection_is_optional_in_health(self) -> None:
        general = _index([_chunk("general", "ready", 1.0)], source="central")
        counts = {name: 1 for name in api.REQUIRED_COLLECTION_NAMES}
        counts["news"] = 0

        with (
            mock.patch.object(api, "_get_index", return_value=general),
            mock.patch.object(
                api,
                "_manifest_collection_counts",
                return_value=("test-generation", counts),
            ),
        ):
            result = api.health()

        self.assertTrue(result["ok"])
        self.assertFalse(result["degraded"])
        self.assertTrue(result["all_collections_ready"])
        self.assertEqual(result["unavailable_collections"], [])
        self.assertEqual(result["optional_unavailable_collections"], ["news"])

    def test_unready_collection_returns_service_unavailable(self) -> None:
        with mock.patch.object(api, "_get_index", return_value=_index([])):
            with self.assertRaises(HTTPException) as raised:
                api.retrieve(api.RetrieveRequest(question="market status", collection="market"))

        self.assertEqual(raised.exception.status_code, 503)

    def test_prefer_authoritative_false_considers_all_rows_and_reports_vector_score(self) -> None:
        authoritative = _chunk("official", "official text", 0.40)
        vector_winner = _chunk("vector-winner", "generated text", 0.99)
        index = _index([authoritative, vector_winner])
        ranked_by_authority = [authoritative, vector_winner]

        with (
            mock.patch.object(api, "_get_index", return_value=index),
            mock.patch.object(
                api,
                "retrieve_chunks",
                return_value=ranked_by_authority,
            ) as retrieve_chunks,
        ):
            result = api.retrieve(
                api.RetrieveRequest(
                    question="vector only",
                    top_k=1,
                    prefer_authoritative=False,
                )
            )

        self.assertEqual(retrieve_chunks.call_args.kwargs["top_k"], len(index.chunks))
        self.assertEqual(result.sources[0].chunk_id, "vector-winner")
        self.assertAlmostEqual(result.sources[0].score, 0.99)

    def test_sources_and_used_chunks_match_the_context_budget(self) -> None:
        first = _chunk("first", " ".join(["alpha"] * 1200), 0.9)
        second = _chunk("second", " ".join(["beta"] * 100), 0.8)

        with (
            mock.patch.object(api, "_get_index", return_value=_index([first, second])),
            mock.patch.object(api, "retrieve_chunks", return_value=[first, second]),
        ):
            result = api.retrieve(
                api.RetrieveRequest(
                    question="context budget",
                    top_k=2,
                    max_context_tokens=1000,
                )
            )

        self.assertEqual(result.used_chunks, 1)
        self.assertEqual([source.chunk_id for source in result.sources], ["first"])
        self.assertNotIn("second", result.context)

    def test_source_record_exposes_document_version_metadata(self) -> None:
        chunk = {
            **_chunk("versioned", "versioned text", 0.9),
            "document_status": "Approved",
            "effective_date": "2026-07-16",
            "published_date": "2026-07-09",
            "revision": "25",
            "document_number": "PGRR147",
            "authority_class": "revision_request",
            "effective_state": "approved_not_effective",
            "effectiveness_label": "Not effective as of 2026-07-22",
            "effectiveness_basis": "Future effective date.",
            "resolved_effective_date": "2026-07-16",
            "effective_date_inferred": True,
            "evidence_role": "related_change_record",
            "is_governing": False,
            "logical_document_id": "revision-request:pgrr147",
            "evidence_id": "E2",
            "section_number": "3.15",
            "section_title": "Reactive Power Capability",
            "page_start": 12,
            "page_end": 13,
        }

        source = api._source_record(chunk)

        self.assertEqual(source.document_status, "Approved")
        self.assertEqual(source.effective_date, "2026-07-16")
        self.assertEqual(source.published_date, "2026-07-09")
        self.assertEqual(source.revision, "25")
        self.assertEqual(source.document_number, "PGRR147")
        self.assertEqual(source.authority_class, "revision_request")
        self.assertEqual(source.effective_state, "approved_not_effective")
        self.assertEqual(source.resolved_effective_date, "2026-07-16")
        self.assertTrue(source.effective_date_inferred)
        self.assertEqual(source.evidence_role, "related_change_record")
        self.assertFalse(source.is_governing)
        self.assertEqual(source.logical_document_id, "revision-request:pgrr147")
        self.assertEqual(source.evidence_id, "E2")
        self.assertEqual(source.section_number, "3.15")
        self.assertEqual(source.section_title, "Reactive Power Capability")
        self.assertEqual((source.page_start, source.page_end), (12, 13))


if __name__ == "__main__":
    unittest.main()
