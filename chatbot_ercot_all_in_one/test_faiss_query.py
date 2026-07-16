"""Manual probe for the inactive legacy FAISS index.

This file keeps its historical name for developer familiarity, but it is not an
automated test. Imports, pickle-backed FAISS loading, user input, and the paid
query embedding are intentionally isolated behind ``main`` so test discovery
and module import remain offline and non-interactive.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "ercot_combined_index"


def main() -> None:
    """Run one explicit query against the legacy local FAISS index."""

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required for the manual FAISS probe")
    if not (INDEX_DIR / "index.faiss").is_file() or not (INDEX_DIR / "index.pkl").is_file():
        raise SystemExit(f"Legacy FAISS index is incomplete: {INDEX_DIR}")

    # These optional legacy dependencies are imported only for an intentional
    # manual run, never by unittest/pytest collection or chatbot startup.
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS

    embedding_model = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=api_key,
    )
    faiss_index = FAISS.load_local(
        str(INDEX_DIR),
        embeddings=embedding_model,
        index_name="index",
        allow_dangerous_deserialization=True,
    )

    query = input("Enter your ERCOT question: ").strip()
    if not query:
        raise SystemExit("A non-empty question is required")
    results = faiss_index.similarity_search(query, k=5)
    print("\nTop 5 results\n" + "=" * 60)
    for index, document in enumerate(results, 1):
        source = document.metadata.get("source", "unknown")
        chunk_id = document.metadata.get("chunk_id", "N/A")
        print(f"\n[{index}] Source: {source} | Chunk ID: {chunk_id}")
        print(document.page_content[:1000])
        print("-" * 60)


class LegacyFaissProbeSafetyTests(unittest.TestCase):
    def test_legacy_probe_is_explicit_and_import_safe(self) -> None:
        self.assertTrue(callable(main))
        self.assertFalse("langchain_community" in globals())


if __name__ == "__main__":
    main()
