"""Keep wording corrections targeted, traceable, and safe to publish repeatedly."""

import copy
import hashlib
import json
from pathlib import Path
import unittest

from embedding_utils import plan_embedding_reuse
from interview_evidence import build_story_library, review_for_story, story_id
from interview_retrieval import extract_tag_value
from sync_manager_content import build_chunks, serialize_content


BASE_DIR = Path(__file__).resolve().parent
CANONICAL_ID = "story-e7747c6b09e0c270"
CORRECTION_ID = "story-lcra-hosting-capacity"


def text_hash(chunk: dict) -> str:
    """Hash exact saved text, including whitespace, for replacement matching."""
    return hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()


def correction_fixture() -> tuple[dict, dict]:
    """Build a small original and wording correction with unchanged facts."""
    original = {
        "text": "Response Type: Story | Principle: Ownership | Question: Describe a study. | "
        "Situation: A planning model needed review. | Task: Check its inputs. | "
        "Action: I checked the model data. | Result: The input error was corrected."
    }
    correction = {
        "id": "story-wording-correction", "response_type": "story", "principle": "Ownership",
        "question": "Describe a study.", "situation": "A planning model needed review.",
        "task": "Review the inputs.", "action": "I reviewed the model data.",
        "result": "The input error was corrected.", "replaces_text_sha256": text_hash(original),
    }
    return original, correction


class SavedHostingCapacityCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.active = json.loads((BASE_DIR / "chunks_cleaned.json").read_text(encoding="utf-8"))
        cls.content = json.loads((BASE_DIR / "manager_content.json").read_text(encoding="utf-8"))
        cls.correction = next(item for item in cls.content if item["id"] == CORRECTION_ID)
        cls.target, cls.base_count, cls.managed_count = build_chunks(cls.active, cls.content)

    def test_target_has_one_corrected_story_and_preserves_every_other_chunk(self) -> None:
        matches = [chunk for chunk in self.target if story_id(chunk["text"]) == CANONICAL_ID]
        self.assertEqual(matches, [serialize_content(self.correction)])
        self.assertEqual(matches[0]["source_id"], CORRECTION_ID)
        self.assertEqual(len(self.target), 166)
        self.assertEqual((self.base_count, self.managed_count), (116, 50))
        prior = next(chunk for chunk in self.active if story_id(chunk["text"]) == CANONICAL_ID)
        self.assertEqual(extract_tag_value(prior["text"], "Situation"), self.correction["situation"])
        self.assertEqual(extract_tag_value(prior["text"], "Question"), self.correction["question"])
        unrelated_before = [chunk for chunk in self.active if story_id(chunk["text"]) != CANONICAL_ID]
        unrelated_after = [chunk for chunk in self.target if chunk.get("source_id") != CORRECTION_ID]
        self.assertEqual(unrelated_after, unrelated_before)

    def test_embedding_refresh_requires_only_the_changed_story(self) -> None:
        indices, missing = plan_embedding_reuse(self.active, self.target)
        original_pending = any(text_hash(chunk) == self.correction["replaces_text_sha256"] for chunk in self.active)
        expected_missing = [serialize_content(self.correction)["text"]] if original_pending else []
        self.assertEqual(missing, expected_missing)
        for chunk, index in zip(self.target, indices):
            if chunk.get("source_id") != CORRECTION_ID:
                self.assertLess(index, len(self.active))
                self.assertEqual(self.active[index]["text"].strip(), chunk["text"].strip())
        synchronized, base_count, managed_count = build_chunks(self.target, self.content)
        self.assertEqual(synchronized, self.target)
        self.assertEqual((base_count, managed_count), (116, 50))
        self.assertEqual(plan_embedding_reuse(self.target, synchronized)[1], [])

    def test_old_review_is_invalidated_while_canonical_id_and_notes_survive(self) -> None:
        record = next(item for item in build_story_library(self.target) if item["id"] == CANONICAL_ID)
        review = {
            "review_status": "Reviewed by me", "result_status": "Reported actual outcome",
            "note": "Keep the underlying engineering facts unchanged.",
            "reviewed_at": "2026-10-19T12:00:00+00:00",
            "source_hashes": [self.correction["replaces_text_sha256"]],
        }
        revised_review = review_for_story(record, review)
        self.assertTrue(revised_review["stale"])
        self.assertEqual(revised_review["review_status"], "Not reviewed")
        self.assertEqual(revised_review["result_status"], "Not confirmed")
        self.assertEqual(revised_review["note"], review["note"])
        self.assertEqual(record["id"], CANONICAL_ID)
        self.assertEqual(len(record["sources"]), 1)
        self.assertNotEqual(record["sources"][0]["text_sha256"], review["source_hashes"][0])
        current_review = {**review, "source_hashes": [record["sources"][0]["text_sha256"]]}
        self.assertFalse(review_for_story(record, current_review)["stale"])


class TargetedReplacementTests(unittest.TestCase):
    def test_replaces_only_exact_original_and_does_not_mutate_inputs(self) -> None:
        original, correction = correction_fixture()
        similar = {"text": original["text"] + " "}
        existing = [original, similar]
        before = copy.deepcopy((existing, correction))
        target, base_count, managed_count = build_chunks(existing, [correction])
        self.assertEqual(target, [similar, serialize_content(correction)])
        self.assertEqual((base_count, managed_count), (1, 1))
        self.assertEqual((existing, correction), before)

    def test_rejects_malformed_replacement_hash(self) -> None:
        original, correction = correction_fixture()
        for invalid in ("", "not-a-hash", "a" * 63, "A" * 64, 123, True):
            with self.subTest(value=invalid):
                with self.assertRaises(ValueError):
                    build_chunks([original], [{**correction, "replaces_text_sha256": invalid}])

    def test_rejects_missing_original_without_published_replacement(self) -> None:
        _, correction = correction_fixture()
        with self.assertRaisesRegex(ValueError, "original story.*not found"):
            build_chunks([{"text": "Unrelated source text."}], [correction])

    def test_rejects_two_corrections_for_one_original(self) -> None:
        original, correction = correction_fixture()
        duplicate = {**correction, "id": "another-correction"}
        with self.assertRaisesRegex(ValueError, "same base text"):
            build_chunks([original], [correction, duplicate])

    def test_rejects_multiple_exact_original_matches(self) -> None:
        original, correction = correction_fixture()
        with self.assertRaises(ValueError):
            build_chunks([original, dict(original)], [correction])
        # A prior managed ID cannot justify deleting two ambiguous base records.
        with self.assertRaises(ValueError):
            build_chunks([original, dict(original), serialize_content(correction)], [correction])

    def test_published_replacement_is_idempotent_without_original(self) -> None:
        _, correction = correction_fixture()
        published = serialize_content(correction)
        existing = [{"text": "Unrelated source text."}, published]
        target, base_count, managed_count = build_chunks(existing, [correction])
        self.assertEqual(target, existing)
        self.assertEqual((base_count, managed_count), (1, 1))
        self.assertEqual(build_chunks(target, [correction])[0], target)

    def test_normalized_replacement_id_remains_idempotent(self) -> None:
        original, correction = correction_fixture()
        correction["id"] = f"  {correction['id']}  "
        target, _, _ = build_chunks([original], [correction])
        self.assertEqual(target[0]["source_id"], correction["id"].strip())
        self.assertEqual(build_chunks(target, [correction])[0], target)


if __name__ == "__main__":
    unittest.main()
