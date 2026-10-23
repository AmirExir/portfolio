"""Keep each curated planning answer attached to its own explicit question."""

from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from interview_evidence import answer_markdown
from interview_retrieval import assigned_candidates, build_candidate, rank_candidates
from interview_runtime_content import build_runtime_corpus
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

    def test_ordinary_selection_of_curated_record_preserves_saved_script(self) -> None:
        client = Mock()
        candidate = build_candidate(1, 0, self.chunks[0])
        self.assertEqual(render_answer(client, "A related question", candidate, ""), self.records[0]["answer"])
        client.chat.completions.create.assert_not_called()

    def test_ordinary_selection_of_noncurated_managed_record_uses_generation(self) -> None:
        client = Mock()
        client.chat.completions.create.return_value = {
            "choices": [{"finish_reason": "stop", "message": {"content": "Tailored response"}}],
        }
        record = {key: value for key, value in self.records[0].items()
                  if key not in {"answer_set", "question_key", "scenario_id"}}
        candidate = build_candidate(1, 0, serialize_content(record))
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


class UpdatedExperienceRoutingTests(unittest.TestCase):
    """Keep the October 21 experiences first while retaining distinct alternatives."""

    CASES = {
        14: {
            "scenario": "williamson-burnet-gtc-dynamic-study-conflict",
            "backup": "planning-backup-14-investment",
            "heading": "Question 14 (Uncertainty / Conflicting Study Results)",
            "natural_question": "Describe a time you made an investment recommendation with incomplete information.",
        },
        15: {
            "scenario": "highway29-liberty-hill-regional-disagreement",
            "backup": "planning-backup-15-ratings",
            "heading": "Question 15 (Disagreement with Stakeholders / Commercial Strategy)",
            "natural_question": "Tell me about a time you resolved a technical planning disagreement.",
        },
        18: {
            "scenario": "marshall-ford-mqt-swapped-reactances",
            "backup": "planning-backup-18-controls",
            "heading": "Question 18 (Identifying a Technical Error / Model Auditing)",
            "natural_question": "Tell me about a time you found a significant dynamic-model error.",
        },
    }

    @classmethod
    def setUpClass(cls) -> None:
        directory = Path(__file__).resolve().parent
        cls.content = json.loads((directory / "manager_content.json").read_text(encoding="utf-8"))
        cls.by_id = {item["id"]: item for item in cls.content}
        saved = json.loads((directory / "chunks_cleaned.json").read_text(encoding="utf-8"))
        cls.runtime = build_runtime_corpus(saved, cls.content)

    def test_numbered_headings_questions_and_natural_aliases_choose_new_primary(self) -> None:
        # Use current source against the independently saved index: a pending
        # embedding refresh must never restore an older assigned script.
        chunks = self.runtime.chunks
        scores = self.runtime.remap_scores(dict.fromkeys(range(len(chunks)), 1000.0))
        for number, case in self.CASES.items():
            primary = self.by_id[f"planning-interview-{number}"]
            queries = (
                f"Q{number}", f"Question {number}", primary["question_key"],
                primary["question"], case["heading"], case["natural_question"],
                f"{number}. {primary['question']}\n{number + 1}.",
                *primary.get("aliases", []),
            )
            for query in queries:
                with self.subTest(number=number, question=query):
                    selected = rank_candidates(query, chunks, scores, focus="Transmission planning")
                    self.assertEqual([item["source_id"] for item in selected], [primary["id"], case["backup"]])
                    self.assertEqual(selected[0]["scenario_id"], case["scenario"])
                    self.assertTrue(all(item["assigned_match"] for item in selected))

    def test_primary_scripts_render_as_saved_star_without_model_preface(self) -> None:
        client = Mock()
        for number, case in self.CASES.items():
            with self.subTest(number=number):
                primary = self.by_id[f"planning-interview-{number}"]
                candidate = assigned_candidates(case["natural_question"], self.runtime.chunks)[0]
                displayed = render_answer(client, case["natural_question"], candidate, "Emphasize leadership.")
                self.assertTrue(displayed.startswith("**Situation**"))
                self.assertEqual(displayed.split("\n\n**Follow-up preparation**\n\n")[0], primary["answer"])
                for heading in ("**Situation**", "**Task**", "**Action**", "**Result**"):
                    self.assertIn(heading, primary["answer"])
                self.assertNotRegex(primary["answer"], r"(?i)i (?:do not|don't|don’t) have a verified|hypothetical")
        client.chat.completions.create.assert_not_called()

    def test_prior_specific_scenarios_remain_exclusive_backup_requests(self) -> None:
        chunks = self.runtime.chunks
        for number, case in self.CASES.items():
            backup = self.by_id[case["backup"]]
            self.assertEqual(backup["backup_for"], f"planning-interview-{number}")
            self.assertNotEqual(backup["scenario_id"], case["scenario"])
            for query in (backup["question"], *backup.get("aliases", [])):
                with self.subTest(number=number, backup_question=query):
                    selected = assigned_candidates(query, chunks)
                    self.assertIsNotNone(selected)
                    self.assertEqual([item["source_id"] for item in selected], [backup["id"]])
        for number in (8, 13, 16, 17, 19):
            with self.subTest(unrelated_question=number):
                selected = assigned_candidates(f"Q{number}", chunks)
                self.assertEqual([item["source_id"] for item in selected], [f"planning-interview-{number:02d}"])


    def test_reported_paraphrases_select_current_saved_scripts_without_generation(self) -> None:
        cases = {
            "walk me through planning process": 1,
            '"How do you approach contingency analysis under NERC TPL-001 standards and ERCOT '
            'Planning Guides? Specifically, how do you evaluate dynamic stability, weak grid, and '
            'voltage recovery challenges associated with high Inverter-Based Resource (IBR) '
            'penetration in regions like West or South Texas?"': 6,
        }
        client = Mock()
        for question, number in cases.items():
            with self.subTest(question=question):
                selected = rank_candidates(question, self.runtime.chunks, {0: 1000}, focus="AI engineering")
                self.assertEqual([item["source_id"] for item in selected], [f"planning-interview-{number:02d}"])
                record = self.by_id[selected[0]["source_id"]]
                self.assertTrue(selected[0]["assigned_match"])
                rendered = render_answer(client, question, selected[0], "Emphasize uncertainty.")
                self.assertEqual(rendered.split("\n\n**Follow-up preparation**\n\n")[0], record["answer"])
        client.chat.completions.create.assert_not_called()

    def test_distinct_natural_intents_keep_primaries_and_only_linked_backups(self) -> None:
        cases = {
            "Could you outline the ERCOT transmission planning workflow from start to finish?": 1,
            "Please describe your planning process.": 1,
            "1. walk me through planning process": 1,
            "Q1: walk me through planning process": 1,
            "How should I organize NERC TPL-001 compliance for contingency analysis?": 3,
            "Explain weak-grid and voltage recovery issues with inverter-based resources.": 6,
            "What would you check when a consultant's results differ from ERCOT study findings?": 8,
            "Describe how you led a complex transmission planning project across teams.": 13,
            "Walk me through the Williamson-Burnet GTC study oscillations.": 14,
            "Explain the stakeholder objections on the Liberty Hill project.": 15,
            "How do you manage multiple transmission study deadlines with limited staffing?": 16,
            "Describe your Marshall Ford generator MQT investigation.": 18,
            "How do you coach and mentor transmission planning engineers?": 19,
        }
        for question, number in cases.items():
            with self.subTest(question=question):
                selected = assigned_candidates(question, self.runtime.chunks)
                primary_id = f"planning-interview-{number:02d}"
                backups = [record["id"] for record in self.content if record.get("backup_for") == primary_id]
                self.assertIsNotNone(selected)
                self.assertEqual([item["source_id"] for item in selected], [primary_id, *backups])

    def test_generic_and_unrelated_questions_do_not_force_curated_assignment(self) -> None:
        for question in (
            "How do you approach planning?", "Tell me about your work.",
            "Walk me through retirement planning process.",
            "Explain model validation.", "What is your experience with PSCAD?",
            "How do you examine resource behavior during disturbances?",
            "Tell me about a conflict.", "How do you coach software developers?",
            "What are the differences between steady-state and dynamic studies?",
            "How do you comply with NERC CIP cybersecurity standards?",
            "How do FAC-003 vegetation standards affect transmission planning?",
            "How do multiple inverter-based resources affect transmission planning?",
        ):
            with self.subTest(question=question):
                self.assertIsNone(assigned_candidates(question, self.runtime.chunks))

    def test_competing_intents_and_mismatched_backup_requests_fail_closed(self) -> None:
        for question in (
            "Explain NERC TPL compliance and IBR dynamic stability.",
            "How do NERC standards affect the ERCOT transmission planning process?",
            "Describe how you led a transmission planning project while managing multiple deadlines.",
            "Explain the consultant's conflicting ERCOT steady-state and dynamic study results.",
            "Q3 and Q6", "Q3 and 6", "Please answer questions 13 and 19",
            "Q18: walk me through planning process",
            "Q18 backup about the Marshall Ford model error",
            "Tell me about the Liberty Hill project and the Marshall Ford model.",
        ):
            with self.subTest(question=question):
                self.assertEqual(assigned_candidates(question, self.runtime.chunks), [])


if __name__ == "__main__":
    unittest.main()
