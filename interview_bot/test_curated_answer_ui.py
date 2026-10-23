"""Exercise strict answer assignment, full scripts, and correction gating."""

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from interview_evidence import answer_markdown, story_id
from interview_sessions import load_session, session_bytes
from sync_manager_content import serialize_content
import test_interview_ui as ui_fixtures


@unittest.skipIf(ui_fixtures.AppTest is None, "Streamlit and NumPy are required for UI tests")
class CuratedAnswerUITests(unittest.TestCase):
    setUp = ui_fixtures.JobFocusUITests.setUp
    click_button = ui_fixtures.JobFocusUITests.click_button

    def assert_saved_answer_and_reflection(self, text: str, record: dict) -> None:
        """Check the spoken script remains intact before its own follow-up notes."""
        heading = "\n\n**Follow-up preparation**\n\n"
        self.assertEqual(text.count(heading), 1)
        script, reflection = text.split(heading)
        self.assertEqual(script, record["answer"])
        self.assertEqual(reflection, record["reflection"])

    def records(self) -> list[dict]:
        """Return the actual twenty scripts used by the application."""
        content = json.loads(Path(__file__).with_name("manager_content.json").read_text())
        return [item for item in content if item.get("answer_set") == "ercot-planning-20"]

    def backups(self, source_id: str) -> list[dict]:
        """Return only explicitly authored backups for one primary answer."""
        content = json.loads(Path(__file__).with_name("manager_content.json").read_text())
        return [item for item in content if item.get("backup_for") == source_id]

    def select_semantic_neighbor(self, source_id: str) -> dict:
        """Choose a saved vector row while preserving runtime mapping and ranking."""
        chunks = json.loads(Path(__file__).with_name("chunks_cleaned.json").read_text())
        index, chunk = next(
            (index, chunk) for index, chunk in enumerate(chunks)
            if chunk.get("source_id") == source_id
        )
        self.app.session_state["index"].search = Mock(return_value=(
            ui_fixtures.np.array([[0.99]]), ui_fixtures.np.array([[index]]),
        ))
        return chunk

    def test_reported_paraphrases_return_saved_q1_and_q6_without_api_requests(self) -> None:
        questions = (
            ("planning-interview-01", "walk me through planning process"),
            ("planning-interview-06", (
                '"How do you approach contingency analysis under NERC TPL-001 standards and ERCOT '
                "Planning Guides? Specifically, how do you evaluate dynamic stability, weak grid, "
                "and voltage recovery challenges associated with high Inverter-Based Resource "
                '(IBR) penetration in regions like West or South Texas?"'
            )),
        )
        records = {record["id"]: record for record in self.records()}
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
            for source_id, question in questions:
                with self.subTest(source_id=source_id):
                    record = records[source_id]
                    self.app.chat_input[0].set_value(question).run()
                    self.assertFalse(self.app.exception)
                    active = self.app.session_state["active_answer"]
                    self.assertEqual(active["question"], question)
                    self.assertEqual(active["candidate"]["source_id"], source_id)
                    self.assertTrue(active["candidate"]["assigned_match"])
                    self.assertEqual(active["candidate"]["text"], serialize_content(record)["text"])
                    self.assert_saved_answer_and_reflection(active["text"], record)
                    self.assertTrue(any(
                        f"{record['question_key']} · Saved answer for this question" == caption.value
                        for caption in self.app.caption
                    ))
                    self.assertTrue(any(
                        f"**Saved question:** {record['question']}" == element.value
                        for element in self.app.markdown
                    ))
        self.client.embeddings.create.assert_not_called()
        self.client.chat.completions.create.assert_not_called()

    def test_semantic_curated_matches_preserve_scripts_reflections_and_saved_export(self) -> None:
        records = {record["id"]: record for record in self.records()}
        cases = (
            ("planning-interview-06", "How do you examine resource behavior during disturbances?"),
            ("planning-interview-18", "Tell me about your work."),
        )
        for source_id, question in cases:
            with self.subTest(source_id=source_id):
                record = records[source_id]
                self.select_semantic_neighbor(source_id)
                self.client.embeddings.create.reset_mock()
                self.app.chat_input[0].set_value(question).run()
                self.assertFalse(self.app.exception)
                active = self.app.session_state["active_answer"]
                candidate = active["candidate"]
                self.assertEqual(candidate["source_id"], source_id)
                self.assertFalse(candidate.get("assigned_match"))
                self.assertEqual(candidate["text"], serialize_content(record)["text"])
                self.assert_saved_answer_and_reflection(active["text"], record)
                self.assertTrue(any(
                    f"{record['question_key']} · Related saved answer · original wording" == caption.value
                    for caption in self.app.caption
                ))
                self.assertTrue(any(
                    f"**Saved question:** {record['question']}" == element.value
                    for element in self.app.markdown
                ))
                exported = answer_markdown(active)
                self.assertIn(f"Question: {question}\n", exported)
                self.assertIn(f"Saved question: {record['question']}\n", exported)
                self.assertIn(active["text"], exported)
                self.assertEqual(exported.count("**Follow-up preparation**"), 1)
                self.assertIn("Saved answer wording", exported)
                self.assertNotIn("Generated wording", exported)
                next(
                    checkbox for checkbox in self.app.checkbox
                    if checkbox.label == "Show retrieved context"
                ).set_value(True).run()
                self.assertFalse(self.app.exception)
                self.assertTrue(any(element.value == "**Saved answer source**" for element in self.app.markdown))
                self.assertTrue(any(element.value == candidate["text"] for element in self.app.code))
                self.assert_saved_answer_and_reflection(self.app.session_state["active_answer"]["text"], record)
                self.client.embeddings.create.assert_called_once()
                self.app.session_state["index"].search.assert_called_once()
        self.client.chat.completions.create.assert_not_called()

    def test_noncurated_semantic_match_still_generates_from_selected_evidence(self) -> None:
        source_id = "story-pec-annual-transmission-assessment"
        self.select_semantic_neighbor(source_id)
        question = "Tell me about your work."
        self.app.chat_input[0].set_value(question).run()
        self.assertFalse(self.app.exception)
        active = self.app.session_state["active_answer"]
        candidate = active["candidate"]
        self.assertEqual(candidate["source_id"], source_id)
        self.assertFalse(candidate.get("assigned_match"))
        self.assertFalse(candidate["answer_set"])
        self.assertEqual(active["text"], "Situation: Test.\nTask: Test.\nAction: Test.\nResult: Test.")
        self.client.embeddings.create.assert_called_once()
        self.client.chat.completions.create.assert_called_once()
        messages = self.client.chat.completions.create.call_args.kwargs["messages"]
        self.assertIn(f"Question: {question}", messages[1]["content"])
        self.assertIn(candidate["text"], messages[1]["content"])
        self.assertFalse(any("Related saved answer" in caption.value for caption in self.app.caption))
        exported = answer_markdown(active)
        self.assertIn("Generated wording", exported)
        self.assertNotIn("Saved answer wording", exported)
        self.assertNotIn("Saved question:", exported)

    def test_all_twenty_shortcuts_return_full_unique_scripts_without_generation(self) -> None:
        records = self.records()
        self.assertEqual(len(records), 20)
        selected_scenarios = []
        for record in records:
            with self.subTest(key=record["question_key"]):
                self.app.chat_input[0].set_value(record["question_key"]).run()
                self.assertFalse(self.app.exception)
                active = self.app.session_state["active_answer"]
                self.assert_saved_answer_and_reflection(active["text"], record)
                self.assertEqual(active["candidate"]["question"], record["question"])
                selected_scenarios.append(active["candidate"]["scenario_id"])
                backups = self.backups(record["id"])
                self.assertEqual(len(self.app.session_state["candidate_stories"]), 1 + len(backups))
                for position, backup in enumerate(backups, 1):
                    self.click_button("Give me another answer or story")
                    self.assertEqual(self.app.session_state["candidate_story_pos"], position)
                    self.assert_saved_answer_and_reflection(self.app.session_state["active_answer"]["text"], backup)
                    self.assertEqual(self.app.session_state["active_answer"]["candidate"]["backup_for"], record["id"])
                self.click_button("Give me another answer or story")
                self.assertEqual(self.app.session_state["candidate_story_pos"], len(backups))
                self.assert_saved_answer_and_reflection(
                    self.app.session_state["active_answer"]["text"], backups[-1] if backups else record,
                )
                self.assertTrue(any("end of these matching stories" in message.value for message in self.app.info))
        self.assertEqual(len(set(selected_scenarios)), 20)
        self.client.embeddings.create.assert_not_called()
        self.client.chat.completions.create.assert_not_called()

    def test_q15_backup_requires_explicit_selection_and_honors_individual_corrections(self) -> None:
        primary = next(item for item in self.records() if item["question_key"] == "15 DISAGREEMENT")
        backup = self.backups(primary["id"])[0]
        primary_id = story_id(serialize_content(primary)["text"])
        backup_id = story_id(serialize_content(backup)["text"])
        self.assertNotEqual(primary_id, backup_id)

        self.app.session_state["story_reviews"] = {primary_id: {"review_status": "Needs correction"}}
        self.app.chat_input[0].set_value("Q15").run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state["candidate_stories"], [])
        self.assertNotIn("active_answer", self.app.session_state)
        self.click_button("Give me another answer or story")
        self.assertNotIn("active_answer", self.app.session_state)

        # Naming the backup is explicit permission to use that alternative;
        # an excluded primary never makes it the default automatically.
        self.app.chat_input[0].set_value("Q15 backup").run()
        self.assertFalse(self.app.exception)
        self.assert_saved_answer_and_reflection(self.app.session_state["active_answer"]["text"], backup)
        self.assertTrue(any("Saved backup for this question" in caption.value for caption in self.app.caption))

        self.app.session_state["story_reviews"] = {backup_id: {"review_status": "Needs correction"}}
        self.app.chat_input[0].set_value("Q15").run()
        self.assertFalse(self.app.exception)
        self.assert_saved_answer_and_reflection(self.app.session_state["active_answer"]["text"], primary)
        self.assertEqual(len(self.app.session_state["candidate_stories"]), 1)
        self.click_button("Give me another answer or story")
        self.assert_saved_answer_and_reflection(self.app.session_state["active_answer"]["text"], primary)
        self.app.chat_input[0].set_value("Q15 backup").run()
        self.assertFalse(self.app.exception)
        self.assertNotIn("active_answer", self.app.session_state)
        self.assertEqual(self.app.session_state["candidate_stories"], [])
        self.client.embeddings.create.assert_not_called()
        self.client.chat.completions.create.assert_not_called()

    def test_follow_up_survives_audio_export_history_and_reruns_without_generation(self) -> None:
        record = next(item for item in self.records() if item["question_key"] == "18 ERROR")
        self.app.chat_input[0].set_value(record["question_key"]).run()
        self.assertFalse(self.app.exception)
        active = self.app.session_state["active_answer"]
        self.assert_saved_answer_and_reflection(active["text"], record)

        exported = answer_markdown(active)
        self.assertIn(active["text"], exported)
        self.assertEqual(exported.count("**Follow-up preparation**"), 1)
        self.assertIn("Saved answer wording", exported)
        restored = load_session(session_bytes(self.app.session_state.filtered_state, include_history=True))
        self.assertEqual(restored["messages"][-1], {"role": "assistant", "content": active["text"]})

        self.client.audio.speech.create.return_value = SimpleNamespace(content=b"ID3mock-audio")
        self.click_button("Make answer audio")
        self.assertEqual(self.client.audio.speech.create.call_args.kwargs["input"], active["text"])
        self.assertEqual(self.app.session_state["answer_audio"], b"ID3mock-audio")
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assert_saved_answer_and_reflection(self.app.session_state["active_answer"]["text"], record)
        self.assertEqual(self.client.audio.speech.create.call_count, 1)
        self.assertEqual(self.app.session_state["messages"], restored["messages"])
        self.client.embeddings.create.assert_not_called()
        self.client.chat.completions.create.assert_not_called()

    def test_revised_experiences_answer_natural_questions_before_backups_without_api_key(self) -> None:
        cases = {
            14: ("Describe a time you made an investment recommendation with incomplete information.", "williamson-burnet-gtc-dynamic-study-conflict"),
            15: ("Tell me about a time you resolved a technical planning disagreement.", "highway29-liberty-hill-regional-disagreement"),
            18: ("Tell me about a time you found a significant dynamic-model error.", "marshall-ford-mqt-swapped-reactances"),
        }
        by_id = {item["id"]: item for item in self.records()}
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
            for number, (question, scenario) in cases.items():
                with self.subTest(question=number):
                    primary = by_id[f"planning-interview-{number}"]
                    self.app.chat_input[0].set_value(question).run()
                    self.assertFalse(self.app.exception)
                    active = self.app.session_state["active_answer"]
                    self.assertEqual(active["candidate"]["scenario_id"], scenario)
                    self.assertTrue(active["text"].startswith("**Situation**"))
                    self.assert_saved_answer_and_reflection(active["text"], primary)
                    backup = self.backups(primary["id"])[0]
                    self.click_button("Give me another answer or story")
                    self.assertEqual(self.app.session_state["active_answer"]["candidate"]["source_id"], backup["id"])
                    self.assert_saved_answer_and_reflection(self.app.session_state["active_answer"]["text"], backup)
                    self.app.chat_input[0].set_value(f"Q{number}").run()
                    self.assertFalse(self.app.exception)
                    self.assert_saved_answer_and_reflection(self.app.session_state["active_answer"]["text"], primary)
                    self.assertEqual(self.app.session_state["candidate_story_pos"], 0)
        self.client.embeddings.create.assert_not_called()
        self.client.chat.completions.create.assert_not_called()

    def test_disabled_assignment_does_not_substitute_another_story_or_leave_prior_answer(self) -> None:
        source = next(item for item in self.records() if item["question_key"] == "19 MENTORING")
        self.assertEqual(source["response_type"], "story")
        self.app.chat_input[0].set_value("12 TOOLS").run()
        self.assertIn("active_answer", self.app.session_state)
        self.app.session_state["story_reviews"] = {
            story_id(serialize_content(source)["text"]): {"review_status": "Needs correction"},
        }
        self.app.chat_input[0].set_value(source["question"]).run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state["candidate_stories"], [])
        self.assertNotIn("active_answer", self.app.session_state)
        self.click_button("Give me another answer or story")
        self.assertNotIn("active_answer", self.app.session_state)
        self.client.embeddings.create.assert_not_called()
        self.client.chat.completions.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
