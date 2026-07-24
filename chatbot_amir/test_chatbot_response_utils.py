import unittest

from response_utils import assess_response, compact_messages


class ResponseUtilsTests(unittest.TestCase):
    def test_completed_response_with_text_is_usable(self):
        assessment = assess_response(
            {"status": "completed", "output_text": "A useful answer."}
        )
        self.assertTrue(assessment.usable)
        self.assertFalse(assessment.retryable)

    def test_empty_incomplete_response_is_retryable(self):
        assessment = assess_response(
            {
                "status": "incomplete",
                "output_text": "",
                "incomplete_details": {"reason": "max_output_tokens"},
            }
        )
        self.assertFalse(assessment.usable)
        self.assertTrue(assessment.retryable)
        self.assertIn("max_output_tokens", assessment.diagnostic)

    def test_refusal_is_not_retried(self):
        assessment = assess_response(
            {
                "status": "completed",
                "output_text": "",
                "output": [
                    {
                        "content": [
                            {"type": "refusal", "refusal": "Cannot answer."}
                        ]
                    }
                ],
            }
        )
        self.assertFalse(assessment.usable)
        self.assertFalse(assessment.retryable)

    def test_compaction_preserves_system_and_recent_turns(self):
        messages = [{"role": "system", "content": "resume"}]
        for index in range(10):
            messages.append({"role": "user", "content": f"question {index}"})
            messages.append({"role": "assistant", "content": f"answer {index}"})

        compacted = compact_messages(messages, max_recent_messages=4)

        self.assertEqual(compacted[0]["role"], "system")
        self.assertEqual(len(compacted), 5)
        self.assertEqual(compacted[-1]["content"], "answer 9")
        self.assertNotIn(
            "question 0",
            [message["content"] for message in compacted],
        )


if __name__ == "__main__":
    unittest.main()
