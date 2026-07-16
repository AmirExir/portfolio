"""Endpoint-level behavior for the central ERCOT retrieval API."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
from fastapi import HTTPException

from chatbot_ercot_all_in_one import ercot_rag_api as api
from ERCOTAPI.rag_ingestion.retrieval import LoadedIndex


def _index(chunks: list[dict], *, source: str = "central") -> LoadedIndex:
    vectors = np.asarray([[1.0, 0.0] for _ in chunks], dtype="float32")
    return LoadedIndex(
        chunks=chunks,
        embeddings=vectors,
        embedding_model="test-model",
        generation_id="test-generation" if source == "central" else None,
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
    def test_health_exposes_collection_readiness_without_breaking_general_ok(self) -> None:
        general = _index([_chunk("general", "ready", 1.0)], source="legacy")
        empty = _index([], source="legacy")

        with mock.patch.object(
            api,
            "_get_index",
            side_effect=lambda collection: general if collection == "general" else empty,
        ):
            result = api.health()

        self.assertTrue(result["ok"])
        self.assertTrue(result["degraded"])
        self.assertFalse(result["all_collections_ready"])
        self.assertFalse(result["collections"]["operations"]["ready"])
        self.assertIn("market", result["unavailable_collections"])

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


if __name__ == "__main__":
    unittest.main()
