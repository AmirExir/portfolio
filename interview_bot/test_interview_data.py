import unittest

from embedding_utils import chunk_texts, chunks_digest
from sync_manager_content import (
    CHUNKS_FILE,
    CONTENT_FILE,
    build_chunks,
    load_json,
)


class InterviewDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = load_json(CHUNKS_FILE)
        cls.manager_content = load_json(CONTENT_FILE)

    def test_manager_content_has_direct_answers_and_stories(self):
        response_types = [item["response_type"] for item in self.manager_content]
        self.assertEqual(response_types.count("direct"), 18)
        self.assertEqual(response_types.count("story"), 10)

    def test_sync_is_idempotent(self):
        updated, base_count, manager_count = build_chunks(
            self.chunks,
            self.manager_content,
        )
        self.assertEqual(base_count, 117)
        self.assertEqual(manager_count, 28)
        self.assertEqual(len(updated), 145)

        updated_again, _, _ = build_chunks(updated, self.manager_content)
        self.assertEqual(updated_again, updated)

    def test_serialized_content_is_complete(self):
        manager_chunks = [
            chunk for chunk in self.chunks if chunk.get("source") == "manager_content"
        ]
        self.assertEqual(len(manager_chunks), 28)

        for chunk in manager_chunks:
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
        self.assertEqual(len(chunk_texts(self.chunks)), 145)
        self.assertEqual(len(chunks_digest(self.chunks)), 64)


if __name__ == "__main__":
    unittest.main()
