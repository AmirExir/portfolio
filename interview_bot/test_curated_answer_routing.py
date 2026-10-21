"""Keep each curated planning answer attached to its own explicit question."""

from copy import deepcopy
import unittest
from unittest.mock import Mock

from interview_evidence import answer_markdown
from interview_retrieval import assigned_candidates, build_candidate, rank_candidates
from interview_services import InterviewServiceError, render_answer
from sync_manager_content import build_chunks, serialize_content


TOPICS = (
    "PROCESS", "VIOLATION", "NERC", "GENERATOR", "DATA CENTER", "IBR", "ECONOMICS",
    "DISCREPANCY", "STUDY TYPES", "SCENARIOS", "PRIORITIZE INVESTMENTS", "TOOLS",
    "LEADERSHIP", "UNCERTAINTY", "DISAGREEMENT", "DEADLINES", "BAD NEWS", "ERROR",
    "MENTORING", "STAKEHOLDERS",
)


def answer_set() -> list[dict]:
    """Create distinguishable scripts, including deliberately overlapping words."""
    return [{
        "id": f"example-{number}", "response_type": "direct", "category": "Planning",
        "question": f"How would you handle {topic.lower()} in this planning interview?",
        "aliases": [f"Explain your specific {topic.lower()} approach."],
        "answer": f"My complete answer {number}.\n\n1. **Check the basis.** Validate models.\n"
                  "2. **Compare alternatives.** Keep uncertainty explicit.",
        "answer_set": "ercot-planning-20", "question_key": f"{number} {topic}",
        "scenario_id": f"distinct-scenario-{number}",
    } for number, topic in enumerate(TOPICS, 1)]


def backup_record() -> dict:
    """Make an explicitly linked disagreement backup with its own script."""
    return {
        "id": "rating-backup", "response_type": "story", "principle": "Earn Trust",
        "question": "How did you resolve the rejected rating submission?",
        "aliases": ["Q15 backup", "another disagreement story"],
        "situation": "I rejected a rating based on an outdated planning model.",
        "task": "Resolve the disagreement using the correct topology.",
        "action": "I reconciled dates and acknowledged the outdated reference case.",
        "result": "The correct rating was approved.", "answer": "The complete saved rating story.",
        "backup_for": "example-15", "scenario_id": "rating-disagreement",
    }


class CuratedAnswerRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = answer_set()
        self.chunks, _, _ = build_chunks([], self.records)

    def test_every_explicit_question_alias_and_shortcut_has_only_its_own_answer(self) -> None:
        # Bad semantic neighbors and a different role must not override an
        # explicit request for one of the authored questions.
        for idx, record in enumerate(self.records):
            number = idx + 1
            topic = TOPICS[idx]
            queries = (
                record["question"], record["aliases"][0],
                f"{number}.\u2060 {record['question']}\n{number + 1}.",
                f"{number} {topic}", f"{number}. {topic}", f"#{number} {topic}",
                topic.lower(), f"Q{number}", f"Question {number}",
            )
            for query in queries:
                with self.subTest(number=number, query=query):
                    results = rank_candidates(
                        query, self.chunks, {(idx + 1) % 20: 100}, k=20,
                        focus="AI engineering",
                    )
                    self.assertEqual([result["index"] for result in results], [idx])
                    self.assertTrue(results[0]["assigned_match"])
                    self.assertEqual(results[0]["scenario_id"], record["scenario_id"])

    def test_generic_questions_and_bare_numbers_keep_ordinary_retrieval(self) -> None:
        for query in ("12", "Tell me about your work.", "How do you approach planning?"):
            with self.subTest(query=query):
                self.assertIsNone(assigned_candidates(query, self.chunks))
        results = rank_candidates("Tell me about your work.", self.chunks, {0: .9, 1: .8})
        self.assertGreater(len(results), 1)
        self.assertFalse(any(result.get("assigned_match") for result in results))

    def test_ambiguous_authored_alias_does_not_choose_or_mix_stories(self) -> None:
        duplicate = deepcopy(self.records[1])
        duplicate["aliases"] = self.records[0]["aliases"]
        chunks = [self.chunks[0], serialize_content(duplicate)]
        self.assertEqual(rank_candidates(self.records[0]["aliases"][0], chunks, {1: 1}), [])

    def test_conflicting_number_and_topic_do_not_select_the_other_question(self) -> None:
        for query in ("12 MENTORING", "12. MENTORING", "#12 MENTORING"):
            with self.subTest(query=query):
                self.assertEqual(rank_candidates(query, self.chunks, {18: 1}), [])

    def test_assigned_answer_preserves_full_script_without_model_call(self) -> None:
        client = Mock()
        candidate = assigned_candidates("12 TOOLS", self.chunks)[0]
        self.assertEqual(
            render_answer(client, "12 TOOLS", candidate, "Emphasize another subject."),
            self.records[11]["answer"],
        )
        client.chat.completions.create.assert_not_called()
        exported = answer_markdown({"question": "12 TOOLS", "text": self.records[11]["answer"], "candidate": candidate})
        self.assertIn("Saved answer wording", exported)
        self.assertNotIn("Generated wording", exported)
        self.assertIn(self.records[11]["answer"], exported)

    def test_ordinary_selection_of_managed_record_still_uses_normal_generation(self) -> None:
        client = Mock()
        client.chat.completions.create.return_value = {
            "choices": [{"finish_reason": "stop", "message": {"content": "Tailored response"}}],
        }
        candidate = build_candidate(1, 0, self.chunks[0])
        self.assertEqual(render_answer(client, "A related question", candidate, ""), "Tailored response")
        client.chat.completions.create.assert_called_once()

    def test_missing_assigned_script_fails_without_generating_a_substitute(self) -> None:
        client = Mock()
        with self.assertRaises(InterviewServiceError):
            render_answer(client, "TOOLS", {"assigned_match": True, "text": "Question: Tools?"}, "")
        client.chat.completions.create.assert_not_called()

    def test_assignment_validation_rejects_missing_questions_and_reused_scenarios(self) -> None:
        for field in ("scenario_id", "question_key"):
            records = deepcopy(self.records)
            records[1][field] = records[0][field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "unique one-to-one"):
                build_chunks([], records)
        with self.assertRaisesRegex(ValueError, "questions 1 through 20"):
            build_chunks([], self.records[:-1])
        missing_answer = deepcopy(self.records[0])
        missing_answer["response_type"] = "story"
        missing_answer.update({field: "Saved fact" for field in ("principle", "situation", "task", "action", "result")})
        missing_answer.pop("answer")
        with self.assertRaisesRegex(ValueError, "answer"):
            serialize_content(missing_answer)
        incomplete_metadata = deepcopy(self.records[0])
        incomplete_metadata.pop("scenario_id")
        with self.assertRaisesRegex(ValueError, "scenario_id"):
            serialize_content(incomplete_metadata)

    def test_backups_follow_only_their_primary_even_with_dominant_semantic_scores(self) -> None:
        chunks, _, _ = build_chunks([], [backup_record(), *self.records])
        results = rank_candidates("15 DISAGREEMENT", chunks, {0: 100, 2: 1000})
        self.assertEqual([item["source_id"] for item in results], ["example-15", "rating-backup"])
        self.assertEqual(results[1]["question_key"], "15 DISAGREEMENT")
        self.assertEqual(results[1]["answer_set"], "ercot-planning-20")
        self.assertTrue(all(item["assigned_match"] for item in results))
        self.assertEqual(rank_candidates("Q15", chunks, {0: 100}, k=1), results[:1])
        for record in self.records:
            if record["id"] != "example-15":
                with self.subTest(key=record["question_key"]):
                    unrelated = rank_candidates(record["question"], chunks, {0: 1000})
                    self.assertEqual([item["source_id"] for item in unrelated], [record["id"]])

    def test_explicit_backup_request_returns_only_its_intact_saved_script(self) -> None:
        backup = backup_record()
        chunks, _, _ = build_chunks([], [*self.records, backup])
        client = Mock()
        for query in [backup["question"], *backup["aliases"]]:
            with self.subTest(query=query):
                results = rank_candidates(query, chunks, {14: 100})
                self.assertEqual([item["source_id"] for item in results], [backup["id"]])
                self.assertEqual(render_answer(client, query, results[0], ""), backup["answer"])
        client.chat.completions.create.assert_not_called()
        generic = rank_candidates("Tell me about your engineering work.", chunks, {20: 100, 1: 1}, k=30)
        self.assertNotIn(backup["id"], [item["source_id"] for item in generic])

    def test_backup_validation_rejects_orphans_chains_reused_scenarios_and_double_assignment(self) -> None:
        backup = backup_record()
        for target in ("missing-primary", backup["id"]):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "primary assigned answer"):
                build_chunks([], [*self.records, {**backup, "backup_for": target}])
        with self.assertRaisesRegex(ValueError, "backup scenario_id must be unique"):
            build_chunks([], [*self.records, {**backup, "scenario_id": self.records[0]["scenario_id"]}])
        with self.assertRaisesRegex(ValueError, "backup scenario_id must be unique"):
            build_chunks([], [*self.records, backup, {**backup, "id": "second-backup"}])
        with self.assertRaisesRegex(ValueError, "cannot also be a primary"):
            serialize_content({**backup, "answer_set": "ercot-planning-20", "question_key": "15 DISAGREEMENT"})
        with self.assertRaisesRegex(ValueError, "answer"):
            serialize_content({key: value for key, value in backup.items() if key != "answer"})


if __name__ == "__main__":
    unittest.main()
