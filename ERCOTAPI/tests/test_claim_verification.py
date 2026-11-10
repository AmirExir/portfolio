"""Gate behavior with controlled judgments; these do not measure model accuracy."""

import json
import unittest

from ERCOTAPI.rag_ingestion.claim_verification import (
    FAILURE_TEXT, MAX_DRAFT_CHARACTERS, REQUEST_TIMEOUT_SECONDS, verify_answer,
)


def claim(text, *, identifier="C1", evidence_ids=None, span=None, kind="other"):
    identifiers = ["E1"] if evidence_ids is None else evidence_ids
    return {"claim_id": identifier, "text": text, "source_span": span or text + " [E1].", "evidence_ids": identifiers, "kind": kind}


def verdict(identifier="C1", *, status="supported", quote="The fixture limit is 100 MW.", evidence_id="E1", correction=None, correction_kind=None, correction_ids=None, kind="other", origin="faithful"):
    return {"claim_id": identifier, "status": status, "reason": "direct_support" if status == "supported" else "missing_evidence",
            "kind": kind, "origin": origin,
            "quotes": [{"evidence_id": evidence_id, "quote": quote}] if quote else [],
            "correction": correction, "correction_kind": correction_kind,
            "correction_evidence_ids": correction_ids or []}


class FakeClient:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []
        self.options = []
        self.responses = self

    def with_options(self, **kwargs):
        self.options.append(kwargs)
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        if isinstance(result, str):
            return {"status": "completed", "output_text": result}
        if "status" in result:
            return result
        return {"status": "completed", "output_text": json.dumps(result)}


class ClaimVerificationTests(unittest.TestCase):
    def setUp(self):
        self.sources = [{"evidence_id": "E1", "text": "The fixture limit is 100 MW."}]
        self.text = "The fixture limit is 100 MW"
        self.answer = self.text + " [E1]."

    def run_gate(self, client, *, answer=None, sources=None):
        return verify_answer(answer or self.answer, self.sources if sources is None else sources,
                             client=client, question="What is the fixture requirement?", as_of="2026-11-10")

    def test_accepted_claim_requires_separate_judgment_and_only_cited_sources(self):
        client = FakeClient({"claims": [claim(self.text)]}, {"verdicts": [verdict()]})
        result = self.run_gate(client, sources=self.sources + [{"evidence_id": "E2", "text": "Unrelated private evidence excluded from the judge."}])
        self.assertEqual(result.status, "verified")
        self.assertEqual(result.text, "- The fixture limit is 100 MW [E1]")
        self.assertEqual(result.cited_chunks, self.sources)
        self.assertEqual(len(client.calls), 2)
        payload = json.loads(client.calls[1]["input"][1]["content"])
        self.assertEqual([item["evidence_id"] for item in payload["claims"][0]["evidence"]], ["E1"])
        self.assertNotIn("Unrelated private", client.calls[1]["input"][1]["content"])
        self.assertEqual(client.options, [{"timeout": REQUEST_TIMEOUT_SECONDS, "max_retries": 0}])
        for call in client.calls:
            self.assertEqual(call["timeout"], REQUEST_TIMEOUT_SECONDS)
            self.assertTrue(call["text"]["format"]["strict"])
            self.assertFalse(call["store"])

    def test_negation_and_inequality_contradictions_are_removed(self):
        for answer_text, source_text in [
            ("The fixture is always required", "The fixture is not always required."),
            ("Resources above 75 MW must submit a study", "Resources below 75 MW must submit a study."),
        ]:
            with self.subTest(answer=answer_text):
                client = FakeClient({"claims": [claim(answer_text)]}, {"verdicts": [verdict(status="contradicted", quote=source_text)]})
                result = self.run_gate(client, answer=answer_text + " [E1].", sources=[{"evidence_id": "E1", "text": source_text}])
                self.assertEqual(result.status, "abstained")
                self.assertEqual(result.removed_claims, 1)
                self.assertNotIn(answer_text, result.text)

    def test_unknown_evidence_id_is_not_sent_to_judge(self):
        text = "The fixture limit is 200 MW"
        client = FakeClient({"claims": [claim(text, evidence_ids=["E9"], span=text + " [E9].")]})
        result = self.run_gate(client, answer=text + " [E9].")
        self.assertEqual(result.status, "abstained")
        self.assertEqual(result.checks[0]["reason"], "unknown_evidence_id")
        self.assertEqual(len(client.calls), 1)

    def test_extractor_cannot_invent_or_reassign_citation_ids(self):
        client = FakeClient({"claims": [claim(self.text, evidence_ids=["E2"])]})
        result = self.run_gate(client)
        self.assertEqual(result.status, "verification_failed")
        self.assertEqual(len(client.calls), 1)

    def test_omitted_or_duplicate_verdict_fails_entire_answer_closed(self):
        for values in [[], [verdict(), verdict()]]:
            client = FakeClient({"claims": [claim(self.text)]}, {"verdicts": values})
            result = self.run_gate(client)
            self.assertEqual(result.status, "verification_failed")
            self.assertEqual(result.text, FAILURE_TEXT)
            self.assertEqual(result.cited_chunks, [])

    def test_malformed_missing_keys_and_duplicate_json_keys_fail_closed(self):
        for output in ["{broken", '{"claims": [], "claims": []}', {}, {"claims": [{"claim_id": "C1"}]}]:
            client = FakeClient(output)
            self.assertEqual(self.run_gate(client).status, "verification_failed")

    def test_refusal_incomplete_and_timeout_never_publish_draft(self):
        for output in [
            {"status": "incomplete", "output_text": '{"claims": []}'},
            {"status": "completed", "output_text": '{"claims": []}', "output": [{"content": [{"type": "refusal", "refusal": "No"}]}]},
            TimeoutError("sensitive request text must not be surfaced"),
        ]:
            result = self.run_gate(FakeClient(output))
            self.assertEqual(result.status, "verification_failed")
            self.assertNotIn("sensitive", result.text + result.notice)
            self.assertNotIn(self.text, result.text)

    def test_corrected_claim_gets_independent_third_check(self):
        original = "The fixture limit is 200 MW"
        client = FakeClient(
            {"claims": [claim(original)]},
            {"verdicts": [verdict(status="contradicted", correction=self.text, correction_ids=["E1"], correction_kind="other")]},
            {"verdicts": [verdict(origin="not_applicable")]},
        )
        result = self.run_gate(client, answer=original + " [E1].")
        self.assertEqual(result.status, "partially_verified")
        self.assertEqual(result.corrected_claims, 1)
        self.assertEqual(result.removed_claims, 0)
        self.assertIn("100 MW", result.text)
        self.assertNotIn("200 MW", result.text)
        correction_payload = json.loads(client.calls[2]["input"][1]["content"])
        self.assertFalse(correction_payload["corrections_allowed"])
        self.assertNotIn("200 MW", client.calls[2]["input"][1]["content"])
        self.assertNotIn("verdict", correction_payload)
        self.assertEqual(len(client.calls), 3)

    def test_unverified_correction_is_removed(self):
        original = "The fixture limit is 200 MW"
        client = FakeClient(
            {"claims": [claim(original)]},
            {"verdicts": [verdict(status="contradicted", correction=self.text, correction_ids=["E1"], correction_kind="other")]},
            {"verdicts": [verdict(status="insufficient_evidence", quote=None, origin="not_applicable")]},
        )
        result = self.run_gate(client, answer=original + " [E1].")
        self.assertEqual(result.status, "abstained")
        self.assertEqual(result.corrected_claims, 0)
        self.assertEqual(result.removed_claims, 1)

    def test_quote_not_found_or_missing_fails_closed(self):
        for quote in ["The fixture limit is 200 MW.", None, "The"]:
            client = FakeClient({"claims": [claim(self.text)]}, {"verdicts": [verdict(quote=quote)]})
            self.assertEqual(self.run_gate(client).status, "verification_failed")

    def test_multiple_assertions_on_one_line_are_independently_gated(self):
        unsupported = "Annual inspections are required"
        second_span = unsupported + " [E1]."
        answer = self.answer + " " + second_span
        client = FakeClient(
            {"claims": [claim(self.text), claim(unsupported, identifier="C2", span=second_span)]},
            {"verdicts": [verdict(), verdict("C2", status="insufficient_evidence", quote=None)]},
        )
        result = self.run_gate(client, answer=answer)
        self.assertEqual(result.status, "partially_verified")
        self.assertNotIn(unsupported, result.text)
        self.assertEqual(result.removed_claims, 1)

    def test_source_quality_and_availability_gates(self):
        flags = [
            {"text": ""}, {"source_available": False}, {"is_stale": True},
            {"context_truncated": True}, {"is_generated": True},
            {"extraction_report": {"review_required": True}},
        ]
        for flag in flags:
            with self.subTest(flag=flag):
                client = FakeClient({"claims": [claim(self.text)]})
                result = self.run_gate(client, sources=[{**self.sources[0], **flag}])
                self.assertEqual(result.status, "abstained")
                self.assertEqual(len(client.calls), 1)

    def test_current_requirement_needs_effective_governing_source(self):
        for metadata in [
            {}, {"effective_state": "proposed_or_pending", "evidence_role": "related_change_record"},
            {"effective_state": "effective", "evidence_role": "current_governing_requirement", "effective_date": "2026-12-01"},
            {"effective_state": "effective", "evidence_role": "current_governing_requirement", "as_of": "2025-01-01"},
        ]:
            client = FakeClient({"claims": [claim(self.text, kind="current_requirement")]})
            self.assertEqual(self.run_gate(client, sources=[{**self.sources[0], **metadata}]).status, "abstained")

    def test_current_requirement_can_pass_with_matching_governing_metadata(self):
        client = FakeClient({"claims": [claim(self.text, kind="current_requirement")]}, {"verdicts": [verdict(kind="current_requirement")]})
        metadata = {"effective_state": "effective", "evidence_role": "current_governing_requirement", "effective_date": "2026-01-01", "as_of": "2026-11-10"}
        self.assertEqual(self.run_gate(client, sources=[{**self.sources[0], **metadata}]).status, "verified")

    def test_oversized_input_and_duplicate_source_ids_fail_before_calls(self):
        client = FakeClient()
        self.assertEqual(self.run_gate(client, answer="x" * (MAX_DRAFT_CHARACTERS + 1)).status, "verification_failed")
        self.assertEqual(self.run_gate(client, sources=self.sources * 2).status, "verification_failed")
        self.assertEqual(client.calls, [])

    def test_judge_error_after_extraction_withholds_draft(self):
        client = FakeClient({"claims": [claim(self.text)]}, RuntimeError("provider unavailable"))
        result = self.run_gate(client)
        self.assertEqual(result.status, "verification_failed")
        self.assertNotIn(self.text, result.text)

    def test_correction_cannot_introduce_a_new_citation(self):
        client = FakeClient({"claims": [claim(self.text)]}, {"verdicts": [verdict(status="contradicted", correction=self.text, correction_kind="other", correction_ids=["E2"])]})
        self.assertEqual(self.run_gate(client).status, "verification_failed")

    def test_rejected_claim_prose_not_exposed_in_checks(self):
        client = FakeClient({"claims": [claim(self.text)]}, {"verdicts": [verdict(status="contradicted")]})
        result = self.run_gate(client)
        self.assertNotIn(self.text, json.dumps(result.checks))

    def test_supported_verdict_cannot_override_absent_quantity_or_claimed_quote(self):
        for text in ["The fixture limit is 200 MW", 'The fixture requirement is "always connect without delay"', "Limit: 200 MW"]:
            client = FakeClient({"claims": [claim(text)]}, {"verdicts": [verdict()]})
            result = self.run_gate(client, answer=text + " [E1].")
            self.assertEqual(result.status, "abstained")
            self.assertEqual(result.removed_claims, 1)
            self.assertIn(result.checks[0]["reason"], ["quantity_not_found_in_cited_text", "quotation_not_found_in_cited_text"])

    def test_supported_correction_cannot_override_absent_quantity(self):
        for correction, reason in [("The fixture limit is 200 MW", "quantity_not_found_in_cited_text"),
                                   ('The fixture says "always connect without delay"', "quotation_not_found_in_cited_text")]:
            client = FakeClient(
                {"claims": [claim(self.text)]},
                {"verdicts": [verdict(status="contradicted", correction=correction, correction_kind="other", correction_ids=["E1"])]},
                {"verdicts": [verdict(origin="not_applicable")]},
            )
            result = self.run_gate(client)
            self.assertEqual(result.status, "abstained")
            self.assertEqual(result.corrected_claims, 0)
            self.assertEqual(result.checks[0]["correction_reason"], reason)

    def test_correction_verification_failure_never_restores_original(self):
        client = FakeClient({"claims": [claim(self.text)]},
                            {"verdicts": [verdict(status="contradicted", correction=self.text, correction_kind="other", correction_ids=["E1"])]},
                            TimeoutError("provider timed out"))
        result = self.run_gate(client)
        self.assertEqual(result.status, "verification_failed")
        self.assertEqual(result.text, FAILURE_TEXT)
        self.assertEqual(result.cited_chunks, [])

    def test_judge_independently_classifies_governing_requirement(self):
        client = FakeClient({"claims": [claim(self.text, kind="other")]}, {"verdicts": [verdict(kind="current_requirement")]})
        result = self.run_gate(client)
        self.assertEqual(result.status, "abstained")
        self.assertEqual(result.checks[0]["reason"], "current_governing_evidence_unavailable")

    def test_independent_fidelity_check_rejects_claim_swaps_and_citation_borrowing(self):
        answer = "Resources must submit annually [E1]. ERCOT publishes a guide [E2]."
        source_span = "ERCOT publishes a guide [E2]."
        sources = [{"evidence_id": "E1", "text": "Resources must submit biennially."},
                   {"evidence_id": "E2", "text": "ERCOT publishes a guide. Resources must submit annually."}]
        client = FakeClient({"claims": [claim("Resources must submit annually", evidence_ids=["E2"], span=source_span)]},
                            {"verdicts": [verdict(evidence_id="E2", quote="Resources must submit annually.", origin="changed")]})
        result = self.run_gate(client, answer=answer, sources=sources)
        self.assertEqual(result.status, "abstained")
        self.assertEqual(result.checks[0]["reason"], "extraction_changed_meaning_or_citation")
        payload = json.loads(client.calls[1]["input"][1]["content"])
        self.assertEqual(payload["claims"][0]["source_span"], source_span)


if __name__ == "__main__":
    unittest.main()
