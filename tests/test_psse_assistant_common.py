from __future__ import annotations

import unittest
from types import SimpleNamespace

from psse_assistant_common import (
    assess_response,
    compact_chat_messages,
    parse_planner_tasks,
    request_visible_answer,
    validate_saved_index,
)


class _Embeddings:
    def __init__(self, shape):
        self.shape = shape


class PsseResponseHandlingTests(unittest.TestCase):
    def test_nested_visible_text_is_accepted(self):
        response = {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Use psspy.fnsl()."}],
                }
            ],
        }

        assessment = assess_response(response)

        self.assertTrue(assessment.usable)
        self.assertEqual(assessment.text, "Use psspy.fnsl().")

    def test_incomplete_empty_response_gets_one_bounded_retry(self):
        responses = iter(
            [
                SimpleNamespace(
                    status="incomplete",
                    output_text="",
                    incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                ),
                SimpleNamespace(status="completed", output_text="Visible answer"),
            ]
        )
        requests = []

        def create(**kwargs):
            requests.append(kwargs)
            return next(responses)

        result = request_visible_answer(
            create,
            {"model": "test", "input": ["large"]},
            retry_request={"model": "test", "input": ["small"]},
        )

        self.assertTrue(result.usable)
        self.assertTrue(result.retried)
        self.assertEqual(result.text, "Visible answer")
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[-1]["input"], ["small"])

    def test_refusal_is_not_retried(self):
        calls = 0

        def create(**_kwargs):
            nonlocal calls
            calls += 1
            return {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "refusal", "refusal": "Cannot help."}],
                    }
                ],
            }

        result = request_visible_answer(
            create,
            {"model": "test"},
            retry_request={"model": "test"},
        )

        self.assertFalse(result.usable)
        self.assertEqual(calls, 1)

    def test_history_is_bounded(self):
        messages = [
            {"role": "user", "content": f"question {i}"}
            if i % 2 == 0
            else {"role": "assistant", "content": "x" * 100}
            for i in range(10)
        ]

        compacted = compact_chat_messages(
            messages,
            max_messages=4,
            max_characters_per_message=20,
        )

        self.assertEqual(len(compacted), 4)
        self.assertLessEqual(max(len(item["content"]) for item in compacted), 21)


class PsseSavedIndexTests(unittest.TestCase):
    def test_matching_saved_index_is_valid(self):
        validate_saved_index(
            [{"text": "one"}, {"text": "two"}],
            _Embeddings((2, 3072)),
            expected_dimension=3072,
        )

    def test_mismatched_saved_index_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "row mismatch"):
            validate_saved_index(
                [{"text": "one"}, {"text": "two"}],
                _Embeddings((1, 3072)),
            )


class PsseTaskParsingTests(unittest.TestCase):
    def test_task_list_is_deduplicated_and_bounded(self):
        tasks = parse_planner_tasks(
            "Tasks:\n1. Load the case\n2. Solve the case\n- Load the case\n"
            "3. Export results",
            max_tasks=2,
        )

        self.assertEqual(tasks, ["Load the case", "Solve the case"])

    def test_planner_error_never_becomes_an_executable_task(self):
        self.assertEqual(
            parse_planner_tasks("[Planner Error] no visible answer"),
            [],
        )


if __name__ == "__main__":
    unittest.main()
