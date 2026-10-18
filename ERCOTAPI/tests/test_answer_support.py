"""Support-screening tests never equate literal overlap with semantic validation."""

import unittest

from ERCOTAPI.rag_ingestion.answer_support import ABSTENTION_TEXT, screen_answer_support, support_notice
from ERCOTAPI.rag_ingestion.requirements import validate_answer_citations
from ERCOTAPI.rag_ingestion.retrieval import format_citation, format_context


class AnswerSupportTests(unittest.TestCase):
    def test_existing_citation_can_still_have_an_unsupported_quantity(self):
        audit = validate_answer_citations(
            "The fixture limit is 200 MW [E1].",
            [{"evidence_id": "E1", "text": "The fixture limit is 100 MW."}],
        )
        self.assertTrue(audit["citation_format_passed"])
        self.assertTrue(audit["support_review_required"])
        self.assertEqual(audit["claim_checks"][0]["missing_quantities"], ["200 mw"])
        self.assertFalse(audit["semantic_support_verified"])
        self.assertTrue(validate_answer_citations(ABSTENTION_TEXT, [])["passed"])

    def test_matching_words_do_not_prove_qualifier_or_negation(self):
        audit = screen_answer_support(
            "The fixture is always required [E1].",
            [{"evidence_id": "E1", "text": "The fixture is not always required."}],
        )
        self.assertFalse(audit["semantic_support_verified"])
        self.assertEqual(audit["support_status"], "semantic_support_unverified")
        self.assertIn("remain unverified", support_notice(audit))

    def test_unmatched_quote_and_unrelated_text_are_review_signals(self):
        audit = screen_answer_support(
            '"Bananas magically repair transformers" [E1].',
            [{"evidence_id": "E1", "text": "Generator applications use the example system."}],
        )
        reasons = audit["claim_checks"][0]["review_reasons"]
        self.assertIn("quotation_not_found_in_cited_text", reasons)
        self.assertIn("low_lexical_overlap", reasons)

    def test_missing_text_and_extraction_flags_remain_visible(self):
        chunk = {"evidence_id": "E1", "extraction_report": {"review_required": True, "review_reasons": ["Page 2 yielded no text."]}}
        audit = screen_answer_support("The rule requires review [E1].", [chunk])
        self.assertIn("cited_text_unavailable", audit["claim_checks"][0]["review_reasons"])
        self.assertIn("EXTRACTION REVIEW REQUIRED", format_citation(chunk))
        self.assertIn("Page 2 yielded no text", format_context([chunk]))

    def test_canonical_abstention_is_distinct_from_support_verification(self):
        audit = screen_answer_support(ABSTENTION_TEXT, [])
        self.assertTrue(audit["abstention_detected"])
        self.assertFalse(audit["support_review_required"])
        self.assertFalse(audit["semantic_support_verified"])

    def test_units_are_not_interchangeable(self):
        audit = screen_answer_support("The fixture rating is 100 MW [E1].", [{"evidence_id": "E1", "text": "The fixture rating is 100 MVA."}])
        self.assertTrue(audit["support_review_required"])


if __name__ == "__main__":
    unittest.main()
