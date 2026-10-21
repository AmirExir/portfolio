"""Exercise strict answer assignment, full scripts, and correction gating."""

import json
from pathlib import Path
from types import SimpleNamespace
import unittest

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
