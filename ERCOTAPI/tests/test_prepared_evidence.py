"""Evidence sent to a generator must be identical to evidence checked afterward."""

import unittest

from ERCOTAPI.rag_ingestion.retrieval import (
    format_context,
    format_source_list,
    prepare_evidence,
)


class PreparedEvidenceTests(unittest.TestCase):
    def test_missing_ids_are_unique_without_mutating_index(self):
        chunks = [{"text": "First passage"}, {"text": "Second passage", "evidence_id": "E1"}]
        result = prepare_evidence(chunks)
        self.assertEqual([item["evidence_id"] for item in result], ["E2", "E1"])
        self.assertNotIn("evidence_id", chunks[0])
        self.assertIsNot(result[0], chunks[0])

    def test_retry_subset_preserves_original_ids(self):
        prepared = prepare_evidence([{"text": "first"}, {"text": "second"}, {"text": "third"}])
        retry = prepare_evidence(prepared[1:], max_words=1)
        self.assertEqual(retry[0]["evidence_id"], "E2")
        self.assertEqual(len(retry), 1)
        self.assertIn("Evidence [E2]", format_context(retry))

    def test_truncation_is_explicit_and_exact_for_both_consumers(self):
        chunks = [{"text": "One\n two   three four"}, {"text": "Hidden passage"}]
        result = prepare_evidence(chunks, max_words=3)
        self.assertEqual(result[0]["text"], "One\n two   three")
        self.assertTrue(result[0]["context_truncated"])
        self.assertEqual(len(result), 1)
        context = format_context(result)
        self.assertIn(result[0]["text"], context)
        self.assertNotIn("four", context)
        self.assertNotIn("Hidden passage", context)
        self.assertNotIn("context_truncated", chunks[0])

    def test_no_truncation_flag_when_full_passage_fits(self):
        self.assertNotIn("context_truncated", prepare_evidence([{"text": "one two"}], max_words=2)[0])

    def test_empty_evidence_is_not_assigned_a_reference(self):
        self.assertEqual(prepare_evidence([{"text": "   "}]), [])

    def test_ambiguous_ids_fail_before_generation(self):
        for identifiers in (("E1", "E1"), ("bad", "E2"), ("E0", "E1")):
            with self.subTest(identifiers=identifiers), self.assertRaises(ValueError):
                prepare_evidence([{"text": "passage", "evidence_id": identifier} for identifier in identifiers])

    def test_every_cited_section_in_one_document_has_a_footer_entry(self):
        evidence = prepare_evidence([
            {"text": "First section", "document_id": "same", "section_number": "1", "chunk_index": 0},
            {"text": "Second section", "document_id": "same", "section_number": "2", "chunk_index": 1},
        ])
        footer = format_source_list(evidence)
        self.assertIn("**[E1]**", footer)
        self.assertIn("**[E2]**", footer)
        self.assertIn("Section 1", footer)
        self.assertIn("Section 2", footer)
        self.assertEqual(format_source_list([evidence[0], evidence[0]]).count("**[E1]**"), 1)

    def test_invalid_budget_fails_explicitly(self):
        with self.assertRaises(ValueError):
            prepare_evidence([{"text": "passage"}], max_words=0)


if __name__ == "__main__":
    unittest.main()
