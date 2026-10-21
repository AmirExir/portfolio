"""Verify that deployed source edits reach answers before offline index refresh."""

from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import test_interview_ui as ui_fixtures
from interview_retrieval import extract_tag_value
from sync_manager_content import serialize_content


@unittest.skipIf(ui_fixtures.AppTest is None, "Streamlit and NumPy are required for UI tests")
class RuntimeContentUITests(unittest.TestCase):
    setUp = ui_fixtures.JobFocusUITests.setUp
    apply_focus = ui_fixtures.JobFocusUITests.apply_focus

    def revised_content(self) -> tuple[list[dict], dict]:
        """Change one managed story without altering the saved corpus or vectors."""
        content = json.loads(Path(__file__).with_name("manager_content.json").read_text())
        item = next(record for record in content if record["id"] == "planning-interview-19")
        item["action"] = "I used the latest mentoring framework to review assumptions and engineering decisions."
        item["answer"] = "\n\n".join(item[field] for field in ("situation", "task", "action", "result"))
        item["reflection"] = "**What would you do differently?**\nI would introduce the latest review checkpoints earlier."
        return content, item

    def test_numbered_mentoring_question_serves_latest_source_without_reembedding(self) -> None:
        content, item = self.revised_content()
        with patch("interview_runtime_content.load_json", return_value=content):
            self.app.run()
            self.assertFalse(self.app.exception)
            self.assertEqual(self.app.session_state["job_focus"], "All roles")
            question = f"19.\u2060 \u2060{item['question']}\n20."
            self.app.chat_input[0].set_value(question).run()
            self.assertFalse(self.app.exception)
            selected = self.app.session_state["active_answer"]["candidate"]
            self.assertEqual(selected["question"], item["question"])
            self.assertEqual(extract_tag_value(selected["text"], "Answer"), item["answer"])
            self.assertEqual(selected["text"], serialize_content(item)["text"])
            self.assertEqual(
                self.app.session_state["active_answer"]["text"],
                item["answer"] + "\n\n**Follow-up preparation**\n\n" + item["reflection"],
            )
            self.assertTrue(any("search index is refreshed" in element.value for element in self.app.info))
            self.client.embeddings.create.assert_not_called()
            self.client.chat.completions.create.assert_not_called()

    def test_actual_pasted_question_sends_current_engineering_mentoring_details(self) -> None:
        content = json.loads(Path(__file__).with_name("manager_content.json").read_text())
        item = next(record for record in content if record["id"] == "planning-interview-19")
        question = f"19. {item['question']} 20"
        self.app.chat_input[0].set_value(question).run()
        self.assertFalse(self.app.exception)
        selected = self.app.session_state["active_answer"]["candidate"]
        self.assertEqual(extract_tag_value(selected["text"], "Action"), item["action"])
        self.assertEqual(extract_tag_value(selected["text"], "Answer"), item["answer"])
        answer = self.app.session_state["active_answer"]["text"]
        script, reflection = answer.split("\n\n**Follow-up preparation**\n\n")
        self.assertEqual(script, item["answer"])
        self.assertEqual(reflection, item["reflection"])
        for detail in (
            "why a violation occurred", "redistributed power", "limiting facility rating",
            "reactive reserves", "PV/QV", "review points",
        ):
            self.assertIn(detail, selected["text"])
            self.assertIn(detail, answer)
        self.client.embeddings.create.assert_not_called()
        self.client.chat.completions.create.assert_not_called()

    def test_source_rerun_clears_active_answer_and_audio_without_repeating_requests(self) -> None:
        content, item = self.revised_content()
        with patch("interview_runtime_content.load_json", return_value=content):
            self.app.chat_input[0].set_value(item["question"]).run()
        self.assertFalse(self.app.exception)
        self.assertIn("active_answer", self.app.session_state)
        self.app.session_state["answer_audio"] = b"old answer audio"
        history = list(self.app.session_state["messages"])
        calls = self.client.chat.completions.create.call_count
        next_content = deepcopy(content)
        revised = next(record for record in next_content if record["id"] == item["id"])
        revised["action"] += " I reviewed the revised validation criteria."
        revised["answer"] = "\n\n".join(revised[field] for field in ("situation", "task", "action", "result"))
        with patch("interview_runtime_content.load_json", return_value=next_content):
            self.app.run()
        self.assertFalse(self.app.exception)
        self.assertNotIn("active_answer", self.app.session_state)
        self.assertNotIn("answer_audio", self.app.session_state)
        self.assertEqual(self.app.session_state["candidate_stories"], [])
        self.assertEqual(self.app.session_state["messages"], history)
        self.assertEqual(self.client.chat.completions.create.call_count, calls)

    def test_invalid_current_content_stops_before_answer_controls(self) -> None:
        with patch("interview_runtime_content.load_json", return_value=[{"id": "invalid"}]):
            self.app.run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("Current interview answers could not be loaded" in element.value for element in self.app.error))
        self.assertEqual(len(self.app.chat_input), 0)
        self.client.chat.completions.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
