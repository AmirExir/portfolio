"""Keep follow-up preparation grounded, separate, and available in both answer paths."""

from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from interview_evidence import build_story_library, review_for_story, story_id
from interview_retrieval import build_candidate, extract_tag_value
from interview_runtime_content import activate_runtime_content, build_runtime_corpus
from interview_services import answer_messages, render_answer
from sync_manager_content import serialize_content


def story_record() -> dict:
    """Make independent source facts and explicitly conditional follow-up wording."""
    return {
        "id": "model-review", "response_type": "story", "principle": "Ownership",
        "question": "Describe a modeling error.", "situation": "Two planning models differed.",
        "task": "Explain the difference.", "action": "I compared their assumptions.",
        "result": "The corrected comparison agreed.", "answer": "My complete original answer.",
        "reflection": "**What did you learn?**\nI learned to compare assumptions first.\n\n"
        "**What would you do differently?**\nIf I could go back, I would check both models earlier.",
    }


class AnswerReflectionTests(unittest.TestCase):
    def test_reflection_is_a_separate_optional_validated_source_field(self) -> None:
        story = story_record()
        direct = {**story, "response_type": "direct", "category": "Model comparison"}
        for record in (story, direct):
            with self.subTest(response_type=record["response_type"]):
                text = serialize_content(record)["text"]
                self.assertEqual(extract_tag_value(text, "Answer"), record["answer"])
                self.assertEqual(extract_tag_value(text, "Reflection"), record["reflection"])
                if record["response_type"] == "story":
                    self.assertEqual(extract_tag_value(text, "Result"), record["result"])
                legacy = {key: value for key, value in record.items() if key != "reflection"}
                self.assertEqual(extract_tag_value(serialize_content(legacy)["text"], "Reflection"), "")
                for invalid in (None, "", "  ", [], True, 123):
                    with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "reflection"):
                        serialize_content({**record, "reflection": invalid})

    def test_assigned_answer_appends_its_saved_reflection_without_rewriting_or_api_calls(self) -> None:
        record = story_record()
        candidate = build_candidate(1, 0, serialize_content(record))
        candidate["assigned_match"] = True
        client = Mock()
        text = render_answer(client, record["question"], candidate, "Other job requirements")
        main, reflection = text.split("\n\n**Follow-up preparation**\n\n")
        self.assertEqual(main, record["answer"])
        self.assertEqual(reflection, record["reflection"])
        client.chat.completions.create.assert_not_called()

    def test_generated_answers_request_followups_in_the_existing_single_call(self) -> None:
        for reflection in (None, "Looking back, I would compare both models earlier.", story_record()["reflection"]):
            with self.subTest(reflection=reflection):
                record = story_record()
                if reflection is None:
                    record.pop("reflection")
                else:
                    record["reflection"] = reflection
                candidate = build_candidate(1, 0, serialize_content(record))
                completed = "Generated main answer.\n\n**Follow-up preparation**\n\nGrounded follow-up answers."
                client = Mock()
                client.chat.completions.create.return_value = {
                    "choices": [{"finish_reason": "stop", "message": {"content": completed}}],
                }
                self.assertEqual(render_answer(client, "A related question", candidate, ""), completed)
                client.chat.completions.create.assert_called_once()
                instructions = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
                for constraint in (
                    "exactly one **Follow-up preparation**", "**What did you learn?**",
                    "**What would you do differently?**", "conditional improvement",
                    "Do not invent a", "saved Reflection field", "including legacy prose",
                    "Do not borrow another story", "not a second STAR story",
                ):
                    self.assertIn(constraint, instructions)

    def test_hypothetical_followups_cannot_be_presented_as_completed_experience(self) -> None:
        record = {**story_record(), "response_type": "direct", "category": "Proposed approach"}
        candidate = build_candidate(1, 0, serialize_content(record))
        instructions = answer_messages("How would you approach this?", candidate, "")[0]["content"]
        for constraint in (
            "**What is the key takeaway?**", "**What would you challenge first?**",
            "keeping both prospective", "never imply that I completed the scenario",
        ):
            self.assertIn(constraint, instructions)

    def test_every_assigned_record_has_distinct_followups_with_the_right_experience_basis(self) -> None:
        records = json.loads(Path(__file__).with_name("manager_content.json").read_text())
        records = [item for item in records if item.get("answer_set") or item.get("backup_for")]
        self.assertEqual(len(records), 23)
        self.assertEqual(len({item["reflection"] for item in records}), 23)
        for record in records:
            with self.subTest(source=record["id"]):
                reflection = record["reflection"]
                self.assertEqual(reflection.count("?**"), 2)
                if record["response_type"] == "story":
                    self.assertIn("**What did you learn?**", reflection)
                    self.assertIn("**What would you do differently?**", reflection)
                    self.assertIn("If I could go back, I would", reflection)
                else:
                    self.assertIn("**What is the key takeaway?**", reflection)
                    self.assertIn("**What would you challenge first?**", reflection)
                    self.assertNotIn("I learned", reflection)
                self.assertNotIn("**Follow-up preparation**", record["answer"])

    def test_reflection_revision_invalidates_cache_and_actions_but_preserves_story_identity(self) -> None:
        record = story_record()
        saved = serialize_content(record)
        original = deepcopy(saved)
        revised = {**record, "reflection": record["reflection"] + " I would retain the comparison basis."}
        runtime = build_runtime_corpus([saved], [revised])
        self.assertEqual(saved, original)
        self.assertEqual(runtime.semantic_row_map, {})
        self.assertEqual(runtime.pending_embedding_count, 1)
        self.assertEqual(story_id(saved["text"]), story_id(runtime.chunks[0]["text"]))
        old_library = build_story_library([saved])[0]
        review = {
            "review_status": "Reviewed by me",
            "source_hashes": [source["text_sha256"] for source in old_library["sources"]],
        }
        new_library = build_story_library(runtime.chunks)[0]
        self.assertTrue(review_for_story(new_library, review)["stale"])
        state = {"messages": [{"role": "assistant", "content": "Historical answer"}]}
        activate_runtime_content(state, [saved])
        state.update({"active_answer": {"text": "Old answer"}, "answer_audio": b"old", "practice_feedback": "old"})
        self.assertTrue(activate_runtime_content(state, runtime.chunks))
        for key in ("active_answer", "answer_audio", "practice_feedback"):
            self.assertNotIn(key, state)
        self.assertEqual(state["messages"][0]["content"], "Historical answer")


if __name__ == "__main__":
    unittest.main()
