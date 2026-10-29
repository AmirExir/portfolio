"""Regression checks for rejected questions and reviewed microphone input."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from interview_sessions import load_session, session_bytes
import test_interview_ui as ui_fixtures
from test_interview_workflows import wav_recording
from test_interview_transcription import pcm_wav


@unittest.skipIf(ui_fixtures.AppTest is None, "Streamlit and NumPy are required for UI tests")
class QuestionInputUITests(unittest.TestCase):
    setUp = ui_fixtures.JobFocusUITests.setUp
    click_button = ui_fixtures.JobFocusUITests.click_button
    apply_focus = ui_fixtures.JobFocusUITests.apply_focus

    def test_test_message_clears_old_selection_without_search_or_answer_calls(self) -> None:
        self.apply_focus("All roles", job_description=(
            "Project Director, ERCOT Transmission Planning. Lead consultant studies, "
            "generator and large-load interconnections, planning and reliability assessments."
        ))
        self.app.chat_input[0].set_value("8 DISCREPANCY").run()
        self.assertIn("active_answer", self.app.session_state)
        self.client.embeddings.create.reset_mock()
        self.client.chat.completions.create.reset_mock()
        self.app.chat_input[0].set_value("Test.").run()
        self.assertFalse(self.app.exception)
        self.assertFalse("active_answer" in self.app.session_state)
        self.assertEqual(self.app.session_state["candidate_stories"], [])
        self.client.embeddings.create.assert_not_called()
        self.client.chat.completions.create.assert_not_called()
        self.assertTrue(self.app.info)

    def test_unrelated_question_does_not_receive_nearest_saved_answer(self) -> None:
        self.apply_focus("All roles", job_description="Lead ERCOT planning and consultant studies.")
        self.app.chat_input[0].set_value("How do I bake a sourdough loaf?").run()
        self.assertFalse(self.app.exception)
        self.assertFalse("active_answer" in self.app.session_state)
        self.assertEqual(self.app.session_state["candidate_stories"], [])
        self.client.chat.completions.create.assert_not_called()
        self.assertTrue(any("No saved answer clearly matches" in item.value for item in self.app.info))

    def test_transcript_is_edited_and_explicitly_submitted_once(self) -> None:
        self.client.audio.transcriptions.create.return_value = SimpleNamespace(text="MBC 뉴스 이덕영입니다.")
        recording = {"id": 123, "bytes": wav_recording(1)}
        with patch("interview_workspace_ui.mic_recorder", return_value=recording):
            self.app.run()
            self.assertFalse(self.app.exception)
            self.assertEqual(self.app.session_state["messages"], [])
            self.assertNotIn("active_answer", self.app.session_state)
            self.client.embeddings.create.assert_not_called()
            self.client.chat.completions.create.assert_not_called()
            draft = next(field for field in self.app.text_area if field.label == "Review your question")
            self.assertEqual(draft.value, "MBC 뉴스 이덕영입니다.")
            draft.set_value("Tell me about a time you solved a difficult problem.")
            self.click_button("Use this question")
            self.assertEqual(self.app.session_state["active_answer"]["question"],
                             "Tell me about a time you solved a difficult problem.")
            self.assertNotIn("question_transcript_draft", self.app.session_state)
            self.assertNotIn("question_recording", self.app.session_state)
            self.app.run()
        self.assertEqual(self.client.audio.transcriptions.create.call_count, 1)
        self.assertEqual(self.client.embeddings.create.call_count, 1)
        self.assertEqual([item["role"] for item in self.app.session_state["messages"]], ["user", "assistant"])
        self.assertNotIn("MBC", str(self.app.session_state["messages"]))

    def test_discard_does_not_replay_same_recording(self) -> None:
        self.client.audio.transcriptions.create.return_value = SimpleNamespace(text="Test.")
        with patch("interview_workspace_ui.mic_recorder", return_value={"id": 124, "bytes": wav_recording(1)}):
            self.app.run()
            self.click_button("Discard recording")
            self.app.run()
        self.assertFalse(self.app.exception)
        self.assertNotIn("question_transcript_draft", self.app.session_state)
        self.assertEqual(self.app.session_state["messages"], [])
        self.assertEqual(self.client.audio.transcriptions.create.call_count, 1)
        self.client.embeddings.create.assert_not_called()
        self.client.chat.completions.create.assert_not_called()

    def test_question_transcription_retry_preserves_audio_and_creates_only_a_draft(self) -> None:
        self.client.audio.transcriptions.create.side_effect = TimeoutError("private response")
        recording = {"id": 125, "bytes": wav_recording(1)}
        with patch("interview_workspace_ui.mic_recorder", return_value=recording):
            self.app.run()
            self.assertFalse(self.app.exception)
            self.assertEqual(self.app.session_state["question_recording"], recording["bytes"])
            self.assertNotIn("question_transcript_draft", self.app.session_state)
            self.app.run()
            self.assertEqual(self.client.audio.transcriptions.create.call_count, 1)
            self.client.audio.transcriptions.create.side_effect = None
            self.client.audio.transcriptions.create.return_value = SimpleNamespace(text="8 DISCREPANCY")
            self.click_button("Retry question transcription")
        self.assertEqual(self.client.audio.transcriptions.create.call_count, 2)
        self.assertEqual(self.app.session_state["question_transcript_draft"], "8 DISCREPANCY")
        self.assertEqual(self.app.session_state["messages"], [])
        self.client.embeddings.create.assert_not_called()

    def test_reviewed_test_or_blank_input_is_not_searched(self) -> None:
        for question in ("Test.", "   "):
            with self.subTest(question=question):
                self.app.session_state["question_transcript_draft"] = question
                self.app.run()
                self.click_button("Use this question")
                self.assertFalse(self.app.exception)
                self.assertNotIn("active_answer", self.app.session_state)
                self.client.embeddings.create.assert_not_called()
                self.client.chat.completions.create.assert_not_called()

    def test_restore_drops_draft_and_pending_submission_without_replaying_audio(self) -> None:
        self.client.audio.transcriptions.create.return_value = SimpleNamespace(text="8 DISCREPANCY")
        recording = {"id": 126, "bytes": wav_recording(1)}
        with patch("interview_workspace_ui.mic_recorder", return_value=recording):
            self.app.run()
            self.app.session_state["submitted_voice_question"] = "8 DISCREPANCY"
            self.app.session_state["pending_session_import"] = load_session(session_bytes({"job_focus": "All roles"}))
            self.app.run()
        self.assertFalse(self.app.exception)
        self.assertNotIn("question_transcript_draft", self.app.session_state)
        self.assertNotIn("submitted_voice_question", self.app.session_state)
        self.assertNotIn("active_answer", self.app.session_state)
        self.assertEqual(self.client.audio.transcriptions.create.call_count, 1)
        self.client.embeddings.create.assert_not_called()

    def test_silent_recording_shows_microphone_error_without_api_request(self) -> None:
        with patch("interview_workspace_ui.mic_recorder", return_value={
            "id": 127, "bytes": pcm_wav(b"\x00\x00" * 160),
        }):
            self.app.run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("silent" in error.value for error in self.app.error))
        self.assertNotIn("question_transcript_draft", self.app.session_state)
        self.client.audio.transcriptions.create.assert_not_called()
        self.client.embeddings.create.assert_not_called()

    def test_long_transcript_remains_editable_after_validation(self) -> None:
        transcript = "Long question " * 200
        self.app.session_state["question_transcript_draft"] = transcript
        self.app.run()
        self.click_button("Use this question")
        self.assertEqual(self.app.session_state["question_transcript_draft"], transcript)
        self.assertTrue(any("2,000 characters" in item.value for item in self.app.info))
        next(field for field in self.app.text_area if field.label == "Review your question").set_value("8 DISCREPANCY")
        self.click_button("Use this question")
        self.assertEqual(self.app.session_state["active_answer"]["question"], "8 DISCREPANCY")
        self.client.embeddings.create.assert_not_called()

    def test_edited_draft_survives_switching_to_practice_and_back(self) -> None:
        self.client.audio.transcriptions.create.return_value = SimpleNamespace(text="Test.")
        with patch("interview_workspace_ui.mic_recorder", return_value={"id": 128, "bytes": wav_recording(1)}):
            self.app.run()
        next(field for field in self.app.text_area if field.label == "Review your question").set_value("8 DISCREPANCY").run()
        self.app.radio[0].set_value("Practice").run()
        self.app.radio[0].set_value("Answer assistant").run()
        self.assertFalse(self.app.exception)
        draft = next(field for field in self.app.text_area if field.label == "Review your question")
        self.assertEqual(draft.value, "8 DISCREPANCY")
        self.assertEqual(self.client.audio.transcriptions.create.call_count, 1)
        self.client.embeddings.create.assert_not_called()
        self.client.chat.completions.create.assert_not_called()
        self.click_button("Use this question")
        self.assertEqual(self.app.session_state["active_answer"]["question"], "8 DISCREPANCY")

    def test_failed_transcription_can_be_discarded_without_replaying_audio(self) -> None:
        self.client.audio.transcriptions.create.side_effect = TimeoutError("private response")
        with patch("interview_workspace_ui.mic_recorder", return_value={"id": 129, "bytes": wav_recording(1)}):
            self.app.run()
            self.click_button("Discard recording")
            self.app.run()
        self.assertFalse(self.app.exception)
        self.assertNotIn("transcription_error", self.app.session_state)
        self.assertNotIn("question_recording", self.app.session_state)
        self.assertEqual(self.client.audio.transcriptions.create.call_count, 1)
        self.client.embeddings.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
