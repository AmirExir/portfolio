import hashlib
import json
import unittest

from embedding_utils import chunk_texts, chunks_digest
from sync_manager_content import (
    CHUNKS_FILE,
    CONTENT_FILE,
    build_chunks,
    load_json,
    serialize_content,
)


# Validate the exact previously published chunks while source revisions await
# indexing, without requiring access to private drafts. Each digest covers the
# whole chunk serialized as sorted compact JSON with ensure_ascii=False.
# These hashes do not constrain current manager_content.json wording.
PUBLISHED_PLANNING_CHUNK_SHA256 = {
    "planning-interview-01": "24488d037ffa8b72fb3ae653ac430dcd6ee52b57728a2b639f1f3ac9a4a429c3",
    "planning-interview-02": "64b0ef3d1fc7fe58a3da20ab2cb99df23652ece502915129ee7e0076663fbcfc",
    "planning-interview-03": "b038fcf4b69fe5752158bc765aa38a566328ceb2521e0f4cfc1418487aaa7413",
    "planning-interview-04": "655ab2e71217dfe2633a6da601965df05703083e9363e87d76a64317d3ea5b9d",
    "planning-interview-05": "6cf799c139dda3a7f73b2e302480e270b737c5c9088dd11bb8151bad9ea8a8ba",
    "planning-interview-06": "07365a5e572b18362b9cf4efff89b4fdccc3dd19dc9c1a118596391d6d555c39",
    "planning-interview-07": "8a7aea3e988ef533833bc686ac57a486af3bfbb8d23c422b3ab8e286bd1ab952",
    "planning-interview-08": "bc79c1c0125cdc1a9e6cd2e66825e90ce49dd3db015d9d771ebab48eb28d8141",
    "planning-interview-09": "512f45fd2efe21acf59309e07395a4b8affefc245856594bdac2ccf2554edcc0",
    "planning-interview-10": "472838e4da750160d84bbb3551b6346061470050320aea51aa0c544d2e9de14e",
    "planning-interview-11": "3620f3edb5972a3f3c0fa564578ec62a32f9c383735cc4ad7793d807144e9023",
    "planning-interview-12": "2edd8439a9e62d148e35065d4b955a06b0272e5e3ee1a85ca429cc74bb428540",
    "planning-interview-13": "6d92296420edc119ac5b03eaf629e95daa57959a2a07da6a3d51c18eb1e988ab",
    "planning-interview-14": "7ef0ce19d6e548a9ac7c77db18d5ac9238f14ce8100c2b07f7e32985d36394a6",
    "planning-interview-15": "182d3684dbe4e2ddf347eb7e7c3b0a38cefde2a050be36ef3f0db71bf6b5c4cc",
    "planning-interview-16": "2e49df7fd776c9130c47caa3700a0cb9c78ed7df4dcdc344be2fd46f0bc70e61",
    "planning-interview-17": "a5e6dfe95b787e4a72b51442c69433b1a95152b53305d2c29efc749c8c13c862",
    "planning-interview-18": "17e63ace81082fd7937cea4ab9e401e01a5902b160979525e172f9b77e16f23d",
    "planning-interview-19": "7cfac2e9c88293644967c1c6e2f55936cbe1fb80d95f9ce1d1255793dd9cdbd8",
    "planning-interview-20": "7f643cf1c94914d66b11ddcbacc1a02951e37b422b00d328e8bc18b510b3f53a",
}


class InterviewDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = load_json(CHUNKS_FILE)
        cls.manager_content = load_json(CONTENT_FILE)

    def test_manager_content_has_direct_answers_and_stories(self):
        response_types = [item["response_type"] for item in self.manager_content]
        self.assertEqual(response_types.count("direct"), 31)
        self.assertEqual(response_types.count("story"), 18)

    def test_sync_is_idempotent(self):
        updated, base_count, manager_count = build_chunks(
            self.chunks,
            self.manager_content,
        )
        self.assertEqual(base_count, 117)
        self.assertEqual(manager_count, 49)
        self.assertEqual(len(updated), 166)

        updated_again, _, _ = build_chunks(updated, self.manager_content)
        self.assertEqual(updated_again, updated)

    def test_serialized_content_is_complete(self):
        manager_chunks = [
            chunk for chunk in self.chunks if chunk.get("source") == "manager_content"
        ]
        # New text and metadata can remain pending until embeddings are ready.
        # Only the known published version may differ from the current source.
        source_by_id = {item["id"]: item for item in self.manager_content}
        self.assertEqual(
            len({chunk["source_id"] for chunk in manager_chunks}), len(manager_chunks)
        )

        for chunk in manager_chunks:
            source = source_by_id[chunk["source_id"]]
            self.assertEqual(chunk["source_id"], source["id"])
            expected = serialize_content(source)
            if chunk != expected:
                self.assertIn(source["id"], PUBLISHED_PLANNING_CHUNK_SHA256)
                published_digest = hashlib.sha256(
                    json.dumps(chunk, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                self.assertEqual(published_digest, PUBLISHED_PLANNING_CHUNK_SHA256[source["id"]])
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
