"""Check current evidence, safe vector mapping, and revision invalidation offline."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from interview_evidence import build_story_library
from interview_retrieval import rank_candidates
from interview_runtime_content import (
    activate_runtime_content, build_runtime_corpus, load_runtime_corpus,
)
from sync_manager_content import serialize_content


def direct_record(identifier: str, answer: str) -> dict:
    """Make a small managed source with independently identifiable answer text."""
    return {
        "id": identifier, "response_type": "direct", "category": "Planning",
        "question": f"Question for {identifier}?", "answer": answer,
    }


class RuntimeContentTests(unittest.TestCase):
    def test_reordered_changed_new_and_removed_records_cannot_share_wrong_vectors(self) -> None:
        first = direct_record("first", "Original first answer")
        second = direct_record("second", "Unchanged second answer")
        removed = direct_record("removed", "Removed answer")
        base = {"text": "An original unmanaged source"}
        saved = [serialize_content(first), base, serialize_content(second), serialize_content(removed)]
        original_saved = deepcopy(saved)
        current = [second, {**first, "answer": "Revised first answer"}, direct_record("new", "New answer")]

        runtime = build_runtime_corpus(saved, current)

        self.assertEqual(runtime.chunks, [base, *map(serialize_content, current)])
        self.assertEqual(runtime.semantic_row_map, {1: 0, 2: 1})
        self.assertEqual(runtime.remap_scores({0: 0.99, 1: 0.5, 2: 0.7, 3: 0.9, 99: 1}), {0: 0.5, 1: 0.7})
        self.assertEqual(runtime.pending_embedding_count, 2)
        self.assertEqual(runtime.source_locations[0], {"file": "chunks_cleaned.json", "index": 1})
        self.assertEqual(runtime.source_locations[2], {"file": "manager_content.json", "index": 1})
        self.assertEqual(saved, original_saved)

    def test_text_identical_revisions_retain_only_exact_embedding_input_matches(self) -> None:
        item = direct_record("unchanged", "Same answer")
        saved = serialize_content(item)
        saved["text"] = " " + saved["text"] + "\n"
        runtime = build_runtime_corpus([saved], [item])
        self.assertEqual(runtime.semantic_row_map, {0: 0})
        self.assertEqual(runtime.pending_embedding_count, 0)
        changed = build_runtime_corpus([saved], [{**item, "answer": "same answer"}])
        self.assertEqual(changed.semantic_row_map, {})

    def test_exact_numbered_question_wins_without_semantics_or_job_focus(self) -> None:
        question = "How have you mentored transmission planning engineers?"
        current = {**direct_record("current", "Current mentoring approach"), "question": question}
        old = direct_record("other", " ".join([question] * 10))
        chunks = [serialize_content(old), serialize_content(current)]
        for query in (question, f"19.\u2060 \u2060{question}\n20.", f"19. {question} 20"):
            with self.subTest(query=query):
                results = rank_candidates(query, chunks, {0: 0.99}, focus="All roles")
                self.assertEqual(results[0]["index"], 1)
                self.assertTrue(results[0]["question_match"])

    def test_exact_alias_matching_preserves_technical_numbers(self) -> None:
        first = {**direct_record("first", "Small load"), "aliases": ["Evaluate a 1.0 MW load."]}
        second = {**direct_record("second", "Large load"), "aliases": ["Evaluate a 10 MW load."]}
        results = rank_candidates(
            "19. Evaluate a 1.0 MW load.\n20.",
            [serialize_content(first), serialize_content(second)], {1: 0.99},
        )
        self.assertEqual(results[0]["index"], 0)
        self.assertTrue(results[0]["question_match"])

    def test_duplicate_or_missing_saved_ids_fail_instead_of_guessing_identity(self) -> None:
        item = direct_record("one", "Answer")
        saved = serialize_content(item)
        for malformed in ([saved, saved], [{**saved, "source_id": ""}], [None], [{"text": ""}]):
            with self.subTest(saved=malformed):
                with self.assertRaises(ValueError):
                    build_runtime_corpus(malformed, [item])

    def test_invalid_current_source_is_not_replaced_with_saved_answers(self) -> None:
        item = direct_record("one", "Answer")
        saved = [serialize_content(item)]
        for malformed in (None, {}, [None], [item, item], [{**item, "answer": ""}]):
            with self.subTest(source=malformed):
                with self.assertRaises(ValueError):
                    build_runtime_corpus(saved, malformed)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "missing.json"
            with self.assertRaises(FileNotFoundError):
                load_runtime_corpus(saved, path)
            path.write_text("{", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_runtime_corpus(saved, path)

    def test_managed_story_evidence_points_to_current_source_record(self) -> None:
        story = {
            "id": "story", "response_type": "story", "principle": "Ownership",
            "question": "Describe a planning study.", "situation": "A transmission study.",
            "task": "Validate the result.", "action": "Checked the assumptions.",
            "result": "Corrected the model.",
        }
        old = serialize_content({**story, "action": "Old action."})
        runtime = build_runtime_corpus([old], [story])
        library = build_story_library(runtime.chunks, source_locations=runtime.source_locations)
        self.assertEqual(library[0]["sources"][0]["file"], "manager_content.json")
        self.assertEqual(library[0]["sources"][0]["index"], 0)
        self.assertIn("Checked the assumptions.", library[0]["candidate"]["text"])
        self.assertNotIn("Old action.", library[0]["candidate"]["text"])

    def test_content_revision_clears_actions_but_preserves_job_context_and_history(self) -> None:
        before = [{"text": "Original source"}]
        state = {
            "messages": [{"role": "assistant", "content": "Historical answer"}],
            "job_focus": "Transmission planning", "job_description": "Lead studies.",
            "story_reviews": {"prior": {"note": "My note"}},
            "practice_history": [{"answer": "Historical attempt"}],
            "index": "unchanged saved-vector index",
        }
        self.assertFalse(activate_runtime_content(state, before))
        state.update({
            "active_answer": {"text": "Old generated answer"}, "candidate_stories": ["old"],
            "pending_answer": "old", "answer_audio": b"old", "practice_feedback": "old",
            "practice_loaded_source": "old", "evidence_story": "old",
        })
        same_state = deepcopy(state)
        self.assertFalse(activate_runtime_content(state, before))
        self.assertEqual(state, same_state)
        self.assertTrue(activate_runtime_content(state, [{"text": "Current source"}]))
        for key in ("active_answer", "pending_answer", "answer_audio", "practice_feedback", "practice_loaded_source", "evidence_story"):
            self.assertNotIn(key, state)
        self.assertEqual(state["candidate_stories"], [])
        for key in ("messages", "job_focus", "job_description", "story_reviews", "practice_history", "index"):
            self.assertEqual(state[key], same_state[key])


if __name__ == "__main__":
    unittest.main()
