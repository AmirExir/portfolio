"""Exercise job-focus controls with local Streamlit and mocked external services."""

from pathlib import Path
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

try:
    import numpy as np
    from streamlit.testing.v1 import AppTest
except ImportError:
    np = None
    AppTest = None

from interview_retrieval import JOB_CONTEXT_INSTRUCTIONS, planning_evidence
from test_job_description import LATE_REQUIREMENT, LONG_JOB_DESCRIPTION


@unittest.skipIf(AppTest is None, "Streamlit and NumPy are required for UI tests")
class JobFocusUITests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = Mock()
        self.client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=[1.0] + [0.0] * 3071)]
        )
        self.client.chat.completions.create.return_value = {
            "choices": [{"finish_reason": "stop", "message": {
                "content": "Situation: Test.\nTask: Test.\nAction: Test.\nResult: Test."
            }}]
        }

        class TestIndex:
            """Supply deterministic semantic neighbors without a FAISS install."""

            def __init__(self, dimensions: int) -> None:
                self.dimensions = dimensions

            def add(self, embeddings: object) -> None:
                pass

            def search(self, query: object, count: int) -> tuple:
                return np.linspace(0.9, 0.1, count)[None, :], np.arange(count)[None, :]

        modules = {
            "openai": SimpleNamespace(OpenAI=Mock(return_value=self.client)),
            "faiss": SimpleNamespace(IndexFlatIP=TestIndex, normalize_L2=lambda array: None),
            "streamlit_mic_recorder": SimpleNamespace(mic_recorder=lambda **kwargs: None),
        }
        for patcher in (
            patch.dict(sys.modules, modules),
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-placeholder"}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.app = AppTest.from_file(str(Path(__file__).with_name("interview_bot.py"))).run()
        self.assertFalse(self.app.exception)

    def apply_focus(
        self, focus: str, background: str = "", job_description: str = ""
    ) -> None:
        """Commit role settings through the actual sidebar form."""
        self.app.sidebar.selectbox[0].select(focus)
        next(
            field for field in self.app.sidebar.text_area
            if field.label == "Job details (optional)"
        ).set_value(background)
        next(
            field for field in self.app.sidebar.text_area
            if field.label == "Job description"
        ).set_value(job_description)
        self.app.sidebar.button[0].click().run()
        self.assertFalse(self.app.exception)

    def click_button(self, label: str) -> None:
        """Click a visible button by label so layout order is irrelevant."""
        next(button for button in self.app.button if button.label == label).click().run()
        self.assertFalse(self.app.exception)

    def test_focus_guides_query_and_answer_and_change_clears_old_candidates(self) -> None:
        self.apply_focus("AI engineering", "Focus on model evaluation and deployment.")
        self.client.embeddings.create.assert_not_called()
        question = "Tell me about a time you solved a difficult problem."
        self.app.chat_input[0].set_value(question).run()
        self.assertFalse(self.app.exception)
        embedded = self.client.embeddings.create.call_args.kwargs["input"]
        self.assertIn(question, embedded)
        self.assertIn("AI engineering", embedded)
        self.assertIn("model evaluation and deployment", embedded)
        messages = self.client.chat.completions.create.call_args.kwargs["messages"]
        self.assertIn(JOB_CONTEXT_INSTRUCTIONS, messages[0]["content"])
        self.assertIn(f"Question: {question}", messages[1]["content"])
        self.assertIn("<job_background>", messages[1]["content"])
        self.assertTrue(all(
            candidate["response_type"] == "story" and candidate["topic_score"] > 0
            for candidate in self.app.session_state["candidate_stories"]
        ))
        self.click_button("Give me another answer or story")
        self.assertEqual(self.client.embeddings.create.call_count, 1)
        self.assertEqual(self.app.session_state["candidate_story_pos"], 1)

        history = list(self.app.session_state["messages"])
        calls = self.client.chat.completions.create.call_count
        self.apply_focus("Transmission planning")
        self.assertEqual(self.app.session_state["candidate_stories"], [])
        self.assertEqual(self.app.session_state["messages"], history)
        self.click_button("Give me another answer or story")
        self.assertEqual(self.client.chat.completions.create.call_count, calls)
        self.assertEqual(self.client.embeddings.create.call_count, 1)

    def test_browse_uses_applied_focus_without_a_typed_question(self) -> None:
        self.apply_focus("Resource integration")
        self.click_button("Show stories for this role")
        candidates = self.app.session_state["candidate_stories"]
        self.assertTrue(candidates)
        self.assertTrue(all(
            candidate["response_type"] == "story" and candidate["topic_score"] > 0
            for candidate in candidates
        ))
        self.assertIn("Resource integration", self.client.embeddings.create.call_args.kwargs["input"])
        self.assertEqual(self.app.session_state["candidate_story_pos"], 0)

    def test_planning_focus_rejects_top_semantic_ems_match_for_ownership_question(self) -> None:
        chunks = json.loads(Path(__file__).with_name("chunks_cleaned.json").read_text())
        ems_index = next(i for i, item in enumerate(chunks) if
                         "something significant outside your area of responsibility" in item["text"]
                         and "GE Energy Management System" in item["text"])
        annual_index = next(i for i, item in enumerate(chunks) if
                            item.get("source_id") == "story-pec-annual-transmission-assessment")
        self.apply_focus("Transmission planning", "Project Director, ERCOT Transmission Planning",
                         "Lead annual transmission planning assessments, studies and upgrades. Manage vendors and costs.")
        self.app.session_state["index"].search = Mock(return_value=(
            np.array([[0.99, 0.1]]), np.array([[ems_index, annual_index]])
        ))
        question = ("Tell me about a time when you took on something significant outside your area of "
                    "responsibility. Why was it important? What was the outcome?")
        self.app.chat_input[0].set_value(question).run()
        self.assertFalse(self.app.exception)
        candidates = self.app.session_state["candidate_stories"]
        self.assertTrue(candidates)
        self.assertNotIn(ems_index, [item["index"] for item in candidates])
        self.assertTrue(all(planning_evidence(item["text"]) for item in candidates))
        active = self.app.session_state["active_answer"]
        self.assertTrue(planning_evidence(active["candidate"]["text"]))
        sent = self.client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertIn(active["candidate"]["text"], sent)
        self.assertNotIn("GE Energy Management System", sent)
        self.click_button("Give me another answer or story")
        self.assertTrue(planning_evidence(self.app.session_state["active_answer"]["candidate"]["text"]))

    def test_full_job_description_guides_followup_questions_without_manual_focus(self) -> None:
        self.apply_focus("All roles", job_description=LONG_JOB_DESCRIPTION)
        self.assertEqual(self.app.session_state["job_description"], LONG_JOB_DESCRIPTION)
        self.client.embeddings.create.assert_not_called()
        self.client.chat.completions.create.assert_not_called()

        questions = (
            "Tell me about a time you solved a difficult problem.",
            "Tell me about a time you evaluated transmission alternatives.",
        )
        for question in questions:
            with self.subTest(question=question):
                self.app.chat_input[0].set_value(question).run()
                self.assertFalse(self.app.exception)
                embedded = self.client.embeddings.create.call_args.kwargs["input"]
                self.assertIn(question, embedded)
                self.assertIn("TARA", embedded)
                self.assertNotIn(LONG_JOB_DESCRIPTION, embedded)
                messages = self.client.chat.completions.create.call_args.kwargs["messages"]
                self.assertIn(JOB_CONTEXT_INSTRUCTIONS, messages[0]["content"])
                self.assertIn(LONG_JOB_DESCRIPTION, messages[1]["content"])
                self.assertIn(LATE_REQUIREMENT, messages[1]["content"])
                self.assertEqual(self.app.session_state["job_description"], LONG_JOB_DESCRIPTION)
        self.assertEqual(self.client.embeddings.create.call_count, len(questions))

    def test_changed_or_cleared_description_invalidates_pool_without_api_calls(self) -> None:
        original = "Responsibilities: Develop machine learning classifiers and evaluate models."
        changed = "Responsibilities: Evaluate transmission planning upgrades and contingencies."
        self.apply_focus("All roles", job_description=original)

        for description in (changed, ""):
            with self.subTest(description=description):
                self.app.chat_input[0].set_value("Tell me about a time you solved a difficult problem.").run()
                self.assertFalse(self.app.exception)
                self.assertTrue(self.app.session_state["candidate_stories"])
                history = list(self.app.session_state["messages"])
                embedding_calls = self.client.embeddings.create.call_count
                answer_calls = self.client.chat.completions.create.call_count
                self.apply_focus("All roles", job_description=description)
                self.assertEqual(self.app.session_state["job_description"], description)
                self.assertEqual(self.app.session_state["candidate_stories"], [])
                self.assertEqual(self.app.session_state["candidate_story_pos"], 0)
                self.assertEqual(self.app.session_state["active_query"], "")
                self.assertEqual(self.app.session_state["messages"], history)
                self.assertEqual(self.client.embeddings.create.call_count, embedding_calls)
                self.assertEqual(self.client.chat.completions.create.call_count, answer_calls)
                self.click_button("Give me another answer or story")
                self.assertEqual(self.client.chat.completions.create.call_count, answer_calls)

    def test_identical_job_description_apply_preserves_candidates(self) -> None:
        description = "Responsibilities: Evaluate transmission planning upgrades and contingencies."
        self.apply_focus("All roles", job_description=description)
        self.app.chat_input[0].set_value("Tell me about a time you solved a difficult problem.").run()
        self.click_button("Give me another answer or story")
        candidates = list(self.app.session_state["candidate_stories"])
        position = self.app.session_state["candidate_story_pos"]
        history = list(self.app.session_state["messages"])
        embedding_calls = self.client.embeddings.create.call_count
        answer_calls = self.client.chat.completions.create.call_count

        self.apply_focus("All roles", job_description=description)
        self.assertEqual(self.app.session_state["candidate_stories"], candidates)
        self.assertEqual(self.app.session_state["candidate_story_pos"], position)
        self.assertEqual(self.app.session_state["messages"], history)
        self.assertEqual(self.client.embeddings.create.call_count, embedding_calls)
        self.assertEqual(self.client.chat.completions.create.call_count, answer_calls)


if __name__ == "__main__":
    unittest.main()
