"""Deterministic section and page provenance for future RAG chunks."""

from __future__ import annotations

import unittest

from ERCOTAPI.rag_ingestion.chunking import (
    build_chunks,
    chunk_id,
    document_id,
    split_text,
)


class ChunkLocatorTests(unittest.TestCase):
    def build(
        self,
        text: str,
        *,
        chunk_size: int = 180,
        overlap: int = 30,
    ) -> list[dict]:
        return build_chunks(
            text,
            content_hash="a" * 64,
            metadata={"source_authority": "ERCOT"},
            chunk_size=chunk_size,
            overlap=overlap,
        )

    def test_locator_enrichment_preserves_chunk_text_and_ids(self) -> None:
        text = (
            "[Page 4]\n"
            "Section 3.15: Reactive Power Capability\n"
            + "An ERCOT requirement continues through this section. " * 12
        )
        expected_text = split_text(text, chunk_size=160, overlap=25)
        chunks = self.build(text, chunk_size=160, overlap=25)
        expected_document_id = document_id("a" * 64)

        self.assertEqual([chunk["text"] for chunk in chunks], expected_text)
        self.assertEqual(
            [chunk["chunk_id"] for chunk in chunks],
            [
                chunk_id(expected_document_id, index, piece)
                for index, piece in enumerate(expected_text)
            ],
        )

    def test_pdf_page_markers_produce_ranges_and_continuations_inherit(self) -> None:
        text = (
            "[Page 12]\n"
            "Section 3.15: Reactive Power Capability\n"
            + "Alpha requirement continues across the first page. " * 10
            + "\n[Page 13]\n"
            + "Beta continuation remains in the same section. " * 10
        )

        chunks = self.build(text)

        self.assertGreater(len(chunks), 3)
        self.assertTrue(
            all(chunk["section_number"] == "3.15" for chunk in chunks)
        )
        self.assertTrue(
            all(
                chunk["section_title"] == "Reactive Power Capability"
                for chunk in chunks
            )
        )
        continuation_on_page_12 = next(
            chunk
            for chunk in chunks
            if "[Page" not in chunk["text"] and chunk["page_start"] == 12
        )
        self.assertEqual(continuation_on_page_12["page_end"], 12)

        transition = next(
            chunk for chunk in chunks if "[Page 13]" in chunk["text"]
        )
        expected_transition_start = (
            13 if transition["text"].startswith("[Page 13]") else 12
        )
        self.assertEqual(transition["page_start"], expected_transition_start)
        self.assertEqual(transition["page_end"], 13)

        continuation_on_page_13 = next(
            chunk
            for chunk in chunks
            if "[Page 13]" not in chunk["text"] and chunk["page_start"] == 13
        )
        self.assertEqual(continuation_on_page_13["page_end"], 13)

    def test_nested_decimal_section_is_preserved_without_inventing_pages(self) -> None:
        text = (
            "9.1.2(3) Load Commissioning Plan\n"
            + "The plan must identify the requested energization sequence. " * 12
        )

        chunks = self.build(text, chunk_size=150, overlap=20)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk["section_number"], "9.1.2(3)")
            self.assertEqual(
                chunk["section_title"],
                "Load Commissioning Plan",
            )
            self.assertNotIn("page_start", chunk)
            self.assertNotIn("page_end", chunk)

    def test_continuation_chunks_inherit_the_most_recent_section(self) -> None:
        text = (
            "Section 9.1: Applicability\n"
            + "Applicability requirement. " * 12
            + "\n\n"
            + "9.2 Submission Requirements\n"
            + "Submission requirement. " * 18
        )

        chunks = self.build(text, chunk_size=150, overlap=20)
        section_nine_two = [
            chunk for chunk in chunks if chunk["section_number"] == "9.2"
        ]

        self.assertTrue(section_nine_two)
        self.assertTrue(
            any(
                "9.2 Submission Requirements" in chunk["text"]
                for chunk in section_nine_two
            )
        )
        inherited = next(
            chunk
            for chunk in section_nine_two
            if "9.2 Submission Requirements" not in chunk["text"]
        )
        self.assertEqual(inherited["section_title"], "Submission Requirements")

    def test_page_words_without_loader_markers_do_not_create_page_metadata(self) -> None:
        text = (
            "Section 5.2: Stability Assessment\n"
            "See page 19 of the application for supporting model information. "
            + "The stability criteria remain part of this section. " * 10
        )

        chunks = self.build(text, chunk_size=160, overlap=20)

        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertEqual(chunk["section_number"], "5.2")
            self.assertNotIn("page_start", chunk)
            self.assertNotIn("page_end", chunk)


if __name__ == "__main__":
    unittest.main()
