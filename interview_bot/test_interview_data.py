import unittest

from embedding_utils import chunk_texts, chunks_digest, load_valid_embeddings
from interview_runtime_content import build_runtime_corpus
from sync_manager_content import (
    CHUNKS_FILE,
    CONTENT_FILE,
    build_chunks,
    load_json,
    serialize_content,
)


class InterviewDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = load_json(CHUNKS_FILE)
        cls.manager_content = load_json(CONTENT_FILE)

    def test_manager_content_has_direct_answers_and_stories(self):
        response_types = [item["response_type"] for item in self.manager_content]
        self.assertEqual(response_types.count("direct"), 31)
        self.assertEqual(response_types.count("story"), 19)

    def test_sync_is_idempotent(self):
        updated, base_count, manager_count = build_chunks(
            self.chunks,
            self.manager_content,
        )
        self.assertEqual(base_count, 117)
        self.assertEqual(manager_count, 50)
        self.assertEqual(len(updated), 167)

        updated_again, _, _ = build_chunks(updated, self.manager_content)
        self.assertEqual(updated_again, updated)

    def test_runtime_content_uses_current_source_serialization(self):
        runtime = build_runtime_corpus(self.chunks, self.manager_content)
        manager_chunks = [
            chunk for chunk in runtime.chunks if chunk.get("source") == "manager_content"
        ]
        self.assertEqual(manager_chunks, [serialize_content(item) for item in self.manager_content])
        self.assertEqual(len({chunk["source_id"] for chunk in manager_chunks}), len(manager_chunks))
        for saved_row, current_row in runtime.semantic_row_map.items():
            self.assertEqual(
                self.chunks[saved_row]["text"].strip(), runtime.chunks[current_row]["text"].strip()
            )

    def test_saved_embedding_cache_matches_its_saved_corpus(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is required for saved-cache validation")
        vectors, status = load_valid_embeddings(
            self.chunks, CHUNKS_FILE.with_name("embeddings.npy"),
            CHUNKS_FILE.with_name("embeddings.meta.json"),
        )
        self.assertIsNotNone(vectors, status)
        self.assertEqual(status, "cache is current")

    def test_embedding_input_digest_covers_every_chunk(self):
        metadata = load_json(CHUNKS_FILE.with_name("embeddings.meta.json"))
        self.assertEqual(len(chunk_texts(self.chunks)), metadata["chunk_count"])
        self.assertEqual(chunks_digest(self.chunks), metadata["chunks_sha256"])
        self.assertEqual(len(chunks_digest(self.chunks)), 64)

    def test_annual_assessment_is_in_story_lists_and_next_bot_corpus(self) -> None:
        story_id = "story-pec-annual-transmission-assessment"
        source_stories = [item for item in self.manager_content if item["id"] == story_id]
        self.assertEqual(len(source_stories), 1)
        source_story = source_stories[0]

        stories = load_json(CHUNKS_FILE.with_name("stories.json"))
        matching = [item for item in stories if item.get("question") == source_story["question"]]
        self.assertEqual(len(matching), 1)
        for field in ("principle", "question", "situation", "task", "action", "result"):
            self.assertEqual(matching[0][field], source_story[field])

        summaries = load_json(CHUNKS_FILE.with_name("stories1.json"))
        matching_summaries = [
            item for item in summaries
            if item["situation"].startswith("PEC's annual transmission planning assessment")
        ]
        self.assertEqual(len(matching_summaries), 1)
        self.assertIn("potential", matching_summaries[0]["result"].lower())

        updated, _, _ = build_chunks(self.chunks, self.manager_content)
        bot_stories = [chunk for chunk in updated if chunk.get("source_id") == story_id]
        self.assertEqual(bot_stories, [serialize_content(source_story)])


if __name__ == "__main__":
    unittest.main()
