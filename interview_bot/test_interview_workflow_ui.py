"""Exercise persistent answer actions, retries, practice, and session restoration."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from interview_evidence import build_story_library, story_id
from interview_sessions import load_session, session_bytes
import test_interview_ui as ui_fixtures
from test_interview_workflows import valid_feedback, wav_recording


@unittest.skipIf(ui_fixtures.AppTest is None, "Streamlit and NumPy are required for UI tests")
class InterviewWorkflowUITests(unittest.TestCase):
    setUp = ui_fixtures.JobFocusUITests.setUp
    click_button = ui_fixtures.JobFocusUITests.click_button
    apply_focus = ui_fixtures.JobFocusUITests.apply_focus

    def ask(self) -> None:
        """Generate one answer with the mocked services."""
        self.app.chat_input[0].set_value("Tell me about a time you solved a difficult problem.").run()
        self.assertFalse(self.app.exception)

    def test_audio_and_debug_survive_reruns_without_regenerating_answer(self) -> None:
        self.client.audio.speech.create.return_value = SimpleNamespace(content=b"ID3mock-audio")
        self.ask()
        text = self.app.session_state["active_answer"]["text"]
        next(field for field in self.app.checkbox if field.label == "Show retrieved context").check().run()
        self.assertTrue(any("Selected context sent to GPT" in field.value for field in self.app.markdown))
        self.click_button("Make answer audio")
        self.assertEqual(self.client.audio.speech.create.call_count, 1)
        self.assertEqual(self.client.audio.speech.create.call_args.kwargs["input"], text)
        self.assertEqual(self.app.session_state["answer_audio"], b"ID3mock-audio")
        self.assertEqual(self.client.chat.completions.create.call_count, 1)
        self.assertEqual(self.client.embeddings.create.call_count, 1)
        self.assertTrue(any(button.label == "Make answer audio" for button in self.app.button))
        self.assertTrue(any(button.label == "Download this answer" for button in self.app.get("download_button")))

    def test_search_retry_preserves_question_without_duplicate_history(self) -> None:
        self.client.embeddings.create.side_effect = TimeoutError("private error payload")
        self.ask()
        self.assertEqual(len(self.app.session_state["messages"]), 1)
        self.assertTrue(any(button.label == "Retry story search" for button in self.app.button))
        self.client.embeddings.create.side_effect = None
        self.click_button("Retry story search")
        self.assertEqual(self.client.embeddings.create.call_count, 2)
        self.assertEqual([message["role"] for message in self.app.session_state["messages"]], ["user", "assistant"])
        self.assertNotIn("search_error", self.app.session_state)

    def test_invalid_query_vector_is_handled_without_generating_an_answer(self) -> None:
        self.client.embeddings.create.return_value = SimpleNamespace(data=[SimpleNamespace(embedding=["invalid"])])
        self.ask()
        self.assertIn("invalid vector", self.app.session_state["search_error"])
        self.assertEqual(self.app.session_state["candidate_stories"], [])
        self.client.chat.completions.create.assert_not_called()

    def test_answer_retry_does_not_repeat_retrieval(self) -> None:
        self.client.chat.completions.create.side_effect = TimeoutError("private")
        self.ask()
        self.assertTrue(any(button.label == "Retry answer" for button in self.app.button))
        self.client.chat.completions.create.side_effect = None
        self.click_button("Retry answer")
        self.assertEqual(self.client.embeddings.create.call_count, 1)
        self.assertEqual(self.client.chat.completions.create.call_count, 2)
        self.assertEqual(len(self.app.session_state["messages"]), 2)

    def test_typed_practice_uses_selected_source_and_produces_one_follow_up(self) -> None:
        self.app.radio[0].set_value("Practice").run()
        self.assertFalse(self.app.exception)
        self.client.chat.completions.create.assert_not_called()
        attempt = "I reviewed the study assumptions, evaluated alternatives, and presented my recommendation."
        next(field for field in self.app.text_area if field.label == "My practice answer").set_value(attempt).run()
        self.client.chat.completions.create.return_value = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(valid_feedback())}}]}
        self.click_button("Get practice feedback")
        request = self.client.chat.completions.create.call_args.kwargs
        payload = json.loads(request["messages"][1]["content"])
        self.assertEqual(payload["attempted_answer"], attempt)
        self.assertIn("Situation:", payload["selected_context"])
        self.assertEqual(self.app.session_state["practice_feedback"]["follow_up"].count("?"), 1)
        self.assertEqual(self.app.session_state["practice_history"][0]["answer"], attempt)
        self.client.embeddings.create.assert_not_called()

    def test_recorded_practice_transcribes_only_after_explicit_request(self) -> None:
        self.app.radio[0].set_value("Practice").run()
        with patch("interview_workspace_ui.mic_recorder", return_value={"bytes": wav_recording()}):
            self.app.run()
        self.client.audio.transcriptions.create.assert_not_called()
        self.client.audio.transcriptions.create.return_value = SimpleNamespace(text="I evaluated the case and recommended an upgrade.")
        self.click_button("Transcribe my practice answer")
        self.assertEqual(self.client.audio.transcriptions.create.call_count, 1)
        self.assertIn("recommended", self.app.session_state["practice_answer"])
        self.assertTrue(any("Recorded duration: 65 seconds" in field.value for field in self.app.caption))

    def test_invalid_practice_feedback_preserves_attempt_for_retry(self) -> None:
        self.app.radio[0].set_value("Practice").run()
        attempt = "I checked assumptions and reviewed the results with my team."
        next(field for field in self.app.text_area if field.label == "My practice answer").set_value(attempt).run()
        self.client.chat.completions.create.return_value = {"choices": [{"finish_reason": "stop", "message": {"content": "not JSON"}}]}
        self.click_button("Get practice feedback")
        self.assertEqual(self.app.session_state["practice_answer"], attempt)
        self.assertEqual(self.app.session_state["practice_history"], [])
        self.assertNotIn("practice_feedback", self.app.session_state)
        self.client.chat.completions.create.return_value = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(valid_feedback())}}]}
        self.click_button("Get practice feedback")
        self.assertEqual(len(self.app.session_state["practice_history"]), 1)

    def test_user_review_is_explicit_and_bound_to_source_hashes(self) -> None:
        identifier = self.app.session_state["evidence_story"]
        self.assertNotIn(identifier, self.app.session_state["story_reviews"])
        next(field for field in self.app.selectbox if field.label == "My review status").select("Reviewed by me")
        next(field for field in self.app.selectbox if field.label == "How I can support the result").select("Potential benefit")
        self.click_button("Save review in this session")
        review = self.app.session_state["story_reviews"][identifier]
        self.assertEqual(review["review_status"], "Reviewed by me")
        self.assertEqual(review["result_status"], "Potential benefit")
        self.assertTrue(review["source_hashes"])
        self.assertTrue(all(len(value) == 64 for value in review["source_hashes"]))
        self.client.chat.completions.create.assert_not_called()

    def test_session_restore_does_not_replay_imported_messages_as_instructions(self) -> None:
        self.ask()
        saved = load_session(session_bytes({
            "job_focus": "AI engineering", "job_description": "Build ML models.",
            "messages": [{"role": "assistant", "content": "Imported old answer"}],
        }, include_history=True))
        self.app.session_state["pending_session_import"] = saved
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state["job_focus"], "AI engineering")
        self.assertEqual(self.app.session_state["candidate_stories"], [])
        self.assertNotIn("active_answer", self.app.session_state)
        self.assertEqual(self.client.chat.completions.create.call_count, 1)
        self.assertEqual(self.client.embeddings.create.call_count, 1)

    def test_restored_session_resets_review_form_before_resaving(self) -> None:
        identifier = self.app.session_state["evidence_story"]
        next(field for field in self.app.selectbox if field.label == "My review status").select("Reviewed by me")
        next(field for field in self.app.selectbox if field.label == "How I can support the result").select("Potential benefit")
        self.click_button("Save review in this session")
        self.app.session_state["pending_session_import"] = load_session(session_bytes({}))
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state["story_reviews"], {})
        self.assertEqual(next(field for field in self.app.selectbox if field.label == "My review status").value, "Not reviewed")
        self.assertEqual(next(field for field in self.app.selectbox if field.label == "How I can support the result").value, "Not confirmed")
        self.click_button("Save review in this session")
        review = self.app.session_state["story_reviews"][identifier]
        self.assertEqual(review["review_status"], "Not reviewed")
        self.assertEqual(review["result_status"], "Not confirmed")
        self.client.chat.completions.create.assert_not_called()

    def test_changed_source_result_invalidates_practice_feedback(self) -> None:
        self.app.radio[0].set_value("Practice").run()
        identifier = self.app.session_state["practice_source"]
        attempt = "I reviewed the assumptions and presented the study results."
        next(field for field in self.app.text_area if field.label == "My practice answer").set_value(attempt).run()
        self.client.chat.completions.create.return_value = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(valid_feedback())}}]}
        self.click_button("Get practice feedback")
        previous_source = self.app.session_state["practice_loaded_source"]
        self.assertIn("practice_feedback", self.app.session_state)

        def updated_library(chunks: list[dict]) -> list[dict]:
            """Simulate a changed Result while preserving the canonical Situation."""
            updated = [dict(chunk) for chunk in chunks]
            for chunk in updated:
                if story_id(chunk["text"]) == identifier:
                    chunk["text"] = chunk["text"].replace("Result:", "Result: Updated saved outcome. ", 1)
            return build_story_library(updated)

        with patch("interview_evidence.build_story_library", side_effect=updated_library):
            self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state["practice_source"], identifier)
        self.assertNotIn("practice_feedback", self.app.session_state)
        self.assertNotEqual(self.app.session_state["practice_loaded_source"], previous_source)
        self.assertEqual(self.client.chat.completions.create.call_count, 1)
        self.client.embeddings.create.assert_not_called()

    def test_story_marked_needing_correction_is_excluded_from_next_search(self) -> None:
        self.ask()
        identifier = self.app.session_state["active_answer"]["story_id"]
        self.app.session_state["story_reviews"] = {identifier: {
            "review_status": "Needs correction", "result_status": "Not confirmed", "note": "Check the result.", "reviewed_at": "",
        }}
        self.ask()
        self.assertNotIn(identifier, [story_id(candidate["text"]) for candidate in self.app.session_state["candidate_stories"]])


if __name__ == "__main__":
    unittest.main()
