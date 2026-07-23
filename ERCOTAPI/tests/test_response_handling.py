from __future__ import annotations

import unittest
from types import SimpleNamespace

from ERCOTAPI.rag_ingestion.response_handling import (
    assess_response,
    compact_chat_messages,
)


class ResponseHandlingTests(unittest.TestCase):
    def test_completed_visible_text_is_usable(self):
        assessment = assess_response(
            SimpleNamespace(status="completed", output_text="Answer [E1].")
        )

        self.assertTrue(assessment.usable)
        self.assertFalse(assessment.retryable)
        self.assertEqual(assessment.text, "Answer [E1].")

    def test_incomplete_empty_response_is_retryable(self):
        assessment = assess_response(
            SimpleNamespace(
                status="incomplete",
                output_text="",
                incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            )
        )

        self.assertFalse(assessment.usable)
        self.assertTrue(assessment.retryable)
        self.assertIn("max_output_tokens", assessment.diagnostic)

    def test_refusal_is_not_retried_or_treated_as_an_answer(self):
        assessment = assess_response(
            {
                "status": "completed",
                "output_text": "",
                "output": [
                    {
                        "content": [
                            {"type": "refusal", "refusal": "Unable to answer."}
                        ]
                    }
                ],
            }
        )

        self.assertFalse(assessment.usable)
        self.assertFalse(assessment.retryable)
        self.assertIn("declined", assessment.diagnostic)

    def test_chat_compaction_removes_source_footers_and_old_turns(self):
        messages = [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "follow-up"},
            {
                "role": "assistant",
                "content": "grounded answer [E1]\n\n**Retrieved sources**\n\n- source",
            },
            {"role": "user", "content": "latest"},
        ]

        compacted = compact_chat_messages(messages, max_messages=3)

        self.assertEqual(
            compacted,
            [
                {"role": "user", "content": "follow-up"},
                {"role": "assistant", "content": "grounded answer [E1]"},
                {"role": "user", "content": "latest"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
