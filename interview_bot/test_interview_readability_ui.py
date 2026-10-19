"""Keep long interview questions and saved evidence readable across UI reruns."""

import json
from pathlib import Path
import unittest

from interview_evidence import build_story_library
from interview_retrieval import extract_tag_value
from interview_sessions import load_session, session_bytes
import test_interview_ui as ui_fixtures
from test_interview_workflows import valid_feedback


@unittest.skipIf(ui_fixtures.AppTest is None, "Streamlit and NumPy are required for UI tests")
class InterviewReadabilityUITests(unittest.TestCase):
    setUp = ui_fixtures.JobFocusUITests.setUp
    click_button = ui_fixtures.JobFocusUITests.click_button
    apply_focus = ui_fixtures.JobFocusUITests.apply_focus

    def saved_story(self, source_id: str = "planning-interview-13") -> dict:
        """Find a real saved story whose long question previously clipped."""
        chunks = json.loads(Path(__file__).with_name("chunks_cleaned.json").read_text(encoding="utf-8"))
        return next(item for item in build_story_library(chunks) if any(
            source["source_id"] == source_id for source in item["sources"]
        ))

    def select_practice_story(self, record: dict) -> None:
        """Use the real selector so source-change state handling is exercised."""
        self.app.radio[0].set_value("Practice").run()
        next(field for field in self.app.selectbox if field.label == "Story to practice").select(record["id"]).run()
        self.assertFalse(self.app.exception)

    def test_practice_displays_complete_story_and_multiline_question(self) -> None:
        record = self.saved_story()
        self.assertGreater(len(record["title"]), 130)
        self.assertGreater(len(record["candidate"]["question"]), 180)
        self.select_practice_story(record)
        question = next(field for field in self.app.text_area if field.label == "Practice question")
        self.assertEqual(question.value, record["candidate"]["question"])
        self.assertFalse(any(field.label == "Practice question" for field in self.app.text_input))
        # The full title is outside the selector and its collapsed reference.
        self.assertTrue(any(
            element.type == "text" and element.value == record["title"]
            for element in self.app.main.children.values()
        ))
        rendered = "\n".join(field.value for field in self.app.markdown)
        self.assertIn("Saved leadership principle", rendered)
        self.assertTrue(any(
            element.type == "text" and element.value == record["candidate"]["principle"]
            for element in self.app.main.children.values()
        ))
        reference = next(expander for expander in self.app.expander if expander.label == "Reference story (optional)")
        self.assertFalse(any("Saved leadership principle" in field.value for field in reference.markdown))
        custom_question = record["candidate"]["question"] + "\nExplain the assumptions and the outcome in your own words."
        question.set_value(custom_question).run()
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(next(field for field in self.app.text_area if field.label == "Practice question").value, custom_question)
        self.client.chat.completions.create.assert_not_called()
        self.client.embeddings.create.assert_not_called()

    def test_evidence_renders_full_star_fields_and_wrapped_exact_source(self) -> None:
        record = self.saved_story()
        next(field for field in self.app.selectbox if field.label == "Review a saved story").select(record["id"]).run()
        self.assertFalse(self.app.exception)
        evidence = next(expander for expander in self.app.expander if expander.label == "Story evidence and job coverage")
        rendered_fields = [field.value for field in evidence.text]
        for tag in ("Question", "Situation", "Task", "Action", "Result"):
            with self.subTest(field=tag):
                self.assertIn(extract_tag_value(record["candidate"]["text"], tag), rendered_fields)
        exact_sources = [field for field in evidence.code if field.value == record["candidate"]["text"]]
        self.assertTrue(exact_sources)
        self.assertTrue(all(field.proto.wrap_lines for field in exact_sources))
        rendered = "\n".join(field.value for field in evidence.markdown)
        self.assertIn("Saved leadership principle", rendered)
        self.assertIn(record["candidate"]["principle"], rendered_fields)
        labels = "\n".join(field.value for field in evidence.caption)
        self.assertIn("employer", labels.lower())
        self.assertIn("framework", labels.lower())
        self.client.chat.completions.create.assert_not_called()
        self.client.embeddings.create.assert_not_called()

    def test_full_question_reaches_feedback_history_and_restored_session(self) -> None:
        record = self.saved_story()
        self.select_practice_story(record)
        question = record["candidate"]["question"] + "\nWhat uncertainty remained after the study?"
        next(field for field in self.app.text_area if field.label == "Practice question").set_value(question).run()
        attempt = "I checked the assumptions, studied alternatives, and explained the remaining uncertainty."
        next(field for field in self.app.text_area if field.label == "My practice answer").set_value(attempt).run()
        self.client.chat.completions.create.return_value = {
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(valid_feedback())}}]
        }
        self.click_button("Get practice feedback")
        messages = self.client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(json.loads(messages[1]["content"])["interview_question"], question)
        self.assertEqual(self.app.session_state["practice_history"][0]["question"], question)
        saved = load_session(session_bytes({"practice_history": self.app.session_state["practice_history"]}, include_history=True))
        self.app.session_state["pending_session_import"] = saved
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state["practice_history"][0]["question"], question)
        history = next(expander for expander in self.app.expander if expander.label == "Previous practice attempts")
        self.assertTrue(any(question in field.value for field in history.markdown))
        self.assertEqual(self.client.chat.completions.create.call_count, 1)
        self.client.embeddings.create.assert_not_called()

    def test_switching_story_loads_its_full_question_and_clears_previous_draft(self) -> None:
        self.select_practice_story(self.saved_story())
        next(field for field in self.app.text_area if field.label == "Practice question").set_value("My custom question\nwith a second line.").run()
        next(field for field in self.app.text_area if field.label == "My practice answer").set_value("An unfinished answer.").run()
        other = self.saved_story("planning-interview-18")
        next(field for field in self.app.selectbox if field.label == "Story to practice").select(other["id"]).run()
        self.assertFalse(self.app.exception)
        self.assertEqual(next(field for field in self.app.text_area if field.label == "Practice question").value, other["candidate"]["question"])
        self.assertEqual(next(field for field in self.app.text_area if field.label == "My practice answer").value, "")
        self.client.chat.completions.create.assert_not_called()
        self.client.embeddings.create.assert_not_called()

    def test_coverage_preserves_complete_requirements_context_and_source_excerpts(self) -> None:
        requirement = (
            "Mentor and develop transmission planners through recurring technical reviews, "
            "clear delegation, and detailed feedback while maintaining accountability "
            "for the quality of each engineering decision"
        )
        self.apply_focus("All roles", job_description=f"Requisition ID: 97852\n{requirement}\nMust speak Japanese")
        evidence = next(expander for expander in self.app.expander if expander.label == "Story evidence and job coverage")
        direct_text = [element.value for element in evidence.children.values() if element.type == "text"]
        self.assertIn(requirement, direct_text)
        self.assertIn("Must speak Japanese", direct_text)
        background = next(expander for expander in evidence.expander if expander.label == "Posting background, headings, and logistics")
        self.assertEqual([field.value for field in background.text], ["Requisition ID: 97852"])
        self.assertTrue(any("No related phrase found" in field.value for field in evidence.markdown))
        related = next(expander for expander in evidence.expander if expander.label == "Review related stories for requirement 1")
        self.assertTrue(any("Posting wording:" in field.value and "Saved wording:" in field.value for field in related.caption))
        chunks = json.loads(Path(__file__).with_name("chunks_cleaned.json").read_text(encoding="utf-8"))
        titles = {record["title"] for record in build_story_library(chunks)}
        snippets = [field.value for field in related.text if field.value not in titles]
        self.assertTrue(snippets)
        saved_fields = [extract_tag_value(chunk["text"], tag) for chunk in chunks for tag in ("Situation", "Task", "Action", "Result")]
        for snippet in snippets:
            with self.subTest(snippet=snippet):
                self.assertTrue(any(snippet in field for field in saved_fields), "Evidence excerpts must quote saved source text")
        self.client.chat.completions.create.assert_not_called()
        self.client.embeddings.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
