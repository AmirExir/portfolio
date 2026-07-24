"""Tests for DWG/SSWG Chat Completions output validation."""

from __future__ import annotations

import unittest

from DWG_SSWG_Chatbot.chat_completion_handling import assess_chat_completion


class AssessChatCompletionTests(unittest.TestCase):
    def test_completed_visible_answer_is_usable(self) -> None:
        result = assess_chat_completion(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Grounded answer."},
                    }
                ]
            }
        )
        self.assertTrue(result.usable)
        self.assertEqual(result.text, "Grounded answer.")

    def test_length_limited_answer_is_retried(self) -> None:
        result = assess_chat_completion(
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "Partial answer"},
                    }
                ]
            }
        )
        self.assertFalse(result.usable)
        self.assertTrue(result.retryable)
        self.assertEqual(result.incomplete_reason, "max_output_tokens")

    def test_empty_answer_is_retried(self) -> None:
        result = assess_chat_completion(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": None},
                    }
                ]
            }
        )
        self.assertFalse(result.usable)
        self.assertTrue(result.retryable)

    def test_content_filter_and_refusal_are_not_retried(self) -> None:
        filtered = assess_chat_completion(
            {
                "choices": [
                    {
                        "finish_reason": "content_filter",
                        "message": {"content": None},
                    }
                ]
            }
        )
        refused = assess_chat_completion(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": None,
                            "refusal": "I cannot answer that.",
                        },
                    }
                ]
            }
        )
        self.assertFalse(filtered.retryable)
        self.assertFalse(refused.retryable)
        self.assertIn("declined", refused.diagnostic)


if __name__ == "__main__":
    unittest.main()
