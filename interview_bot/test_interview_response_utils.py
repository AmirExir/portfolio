import unittest

from interview_response_utils import assess_chat_completion


class ChatCompletionResponseTests(unittest.TestCase):
    def test_stopped_answer_is_usable(self):
        assessment = assess_chat_completion(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Situation: A concise answer."},
                    }
                ]
            }
        )
        self.assertTrue(assessment.usable)
        self.assertFalse(assessment.retryable)

    def test_empty_answer_is_retryable(self):
        assessment = assess_chat_completion(
            {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": None}}
                ]
            }
        )
        self.assertFalse(assessment.usable)
        self.assertTrue(assessment.retryable)

    def test_length_limited_answer_is_retryable(self):
        assessment = assess_chat_completion(
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "Partial answer"},
                    }
                ]
            }
        )
        self.assertFalse(assessment.usable)
        self.assertTrue(assessment.retryable)
        self.assertIn("output limit", assessment.diagnostic)

    def test_refusal_is_not_retryable(self):
        assessment = assess_chat_completion(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": None,
                            "refusal": "Cannot help with that.",
                        },
                    }
                ]
            }
        )
        self.assertFalse(assessment.usable)
        self.assertFalse(assessment.retryable)


if __name__ == "__main__":
    unittest.main()
