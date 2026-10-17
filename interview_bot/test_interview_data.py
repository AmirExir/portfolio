import unittest

from embedding_utils import chunk_texts, chunks_digest
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
        self.assertEqual(response_types.count("direct"), 18)
        self.assertEqual(response_types.count("story"), 11)

    def test_sync_is_idempotent(self):
        updated, base_count, manager_count = build_chunks(
            self.chunks,
            self.manager_content,
        )
        self.assertEqual(base_count, 117)
        self.assertEqual(manager_count, 29)
        self.assertEqual(len(updated), 146)

        updated_again, _, _ = build_chunks(updated, self.manager_content)
        self.assertEqual(updated_again, updated)

    def test_serialized_content_is_complete(self):
        manager_chunks = [
            chunk for chunk in self.chunks if chunk.get("source") == "manager_content"
        ]
        # New source stories can remain pending until their embeddings are ready.
        source_by_id = {item["id"]: item for item in self.manager_content}
        self.assertEqual(
            len({chunk["source_id"] for chunk in manager_chunks}), len(manager_chunks)
        )

        for chunk in manager_chunks:
            self.assertEqual(chunk, serialize_content(source_by_id[chunk["source_id"]]))
            text = chunk["text"]
            self.assertIn("Question:", text)
            if chunk["response_type"] == "direct":
                self.assertIn("Response Type: Direct", text)
                self.assertIn("Answer:", text)
            else:
                self.assertIn("Response Type: Story", text)
                for tag in ("Situation:", "Task:", "Action:", "Result:"):
                    self.assertIn(tag, text)

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
