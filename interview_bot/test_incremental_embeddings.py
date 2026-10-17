"""Validate safe publication and exact-text embedding reuse without API calls."""

import contextlib
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import generate_embeddings
from embedding_utils import (
    load_valid_embeddings,
    plan_embedding_reuse,
    save_embedding_cache,
)
from sync_manager_content import build_chunks

try:
    import numpy as np
except ImportError:
    np = None


class EmbeddingReusePlanTests(unittest.TestCase):
    def test_reordered_changed_and_duplicate_texts(self) -> None:
        existing = [{"text": "first"}, {"text": "second"}]
        target = [
            {"text": "second"},
            {"text": "new"},
            {"text": "first"},
            {"text": "new"},
            {"text": "changed"},
        ]
        indices, missing = plan_embedding_reuse(existing, target)
        self.assertEqual(indices, [1, 2, 0, 2, 3])
        self.assertEqual(missing, ["new", "changed"])

    def test_only_embedding_input_whitespace_is_normalized(self) -> None:
        indices, missing = plan_embedding_reuse(
            [{"text": " first "}], [{"text": "first"}, {"text": "First"}]
        )
        self.assertEqual(indices, [0, 1])
        self.assertEqual(missing, ["First"])


@unittest.skipIf(np is None, "NumPy is required for saved-cache integration tests")
class SynchronizedEmbeddingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        for name, value in {
            "BASE_DIR": self.base,
            "CHUNKS_FILE": self.base / "chunks_cleaned.json",
            "CONTENT_FILE": self.base / "manager_content.json",
            "EMBEDDING_FILE": self.base / "embeddings.npy",
            "METADATA_FILE": self.base / "embeddings.meta.json",
        }.items():
            patcher = patch.object(generate_embeddings, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.original = [{"text": "Existing planning story"}]
        self.original_vectors = np.array([[0.2, 0.4, 0.6]], dtype="float32")
        generate_embeddings.CHUNKS_FILE.write_text(json.dumps(self.original), encoding="utf-8")
        save_embedding_cache(
            self.original,
            self.original_vectors,
            generate_embeddings.EMBEDDING_FILE,
            generate_embeddings.METADATA_FILE,
        )
        self.content = [{
            "id": "test-new-story",
            "response_type": "story",
            "principle": "Ownership",
            "question": "Describe an annual assessment.",
            "situation": "A planning study needed internal ownership.",
            "task": "Perform the assessment.",
            "action": "Evaluate scenarios and alternatives.",
            "result": "Provide recommendations.",
        }]
        generate_embeddings.CONTENT_FILE.write_text(json.dumps(self.content), encoding="utf-8")

    def artifacts(self) -> tuple[bytes, bytes, bytes]:
        """Return active artifacts for byte-for-byte no-write assertions."""
        return tuple(path.read_bytes() for path in (
            generate_embeddings.CHUNKS_FILE,
            generate_embeddings.EMBEDDING_FILE,
            generate_embeddings.METADATA_FILE,
        ))

    def run_generator(self, *arguments: str) -> None:
        """Run the command entry point with output captured and no subprocess."""
        with patch.object(sys, "argv", ["generate_embeddings.py", *arguments]):
            with contextlib.redirect_stdout(io.StringIO()):
                generate_embeddings.main()

    def test_sync_embeds_only_new_text_and_preserves_existing_vector(self) -> None:
        target, _, _ = build_chunks(self.original, self.content)
        new_vectors = np.array([[0.7, 0.8, 0.9]], dtype="float32")
        fake_openai = SimpleNamespace(OpenAI=Mock())
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-placeholder"}):
            with patch.dict(sys.modules, {"openai": fake_openai}):
                with patch.object(generate_embeddings, "create_embeddings", return_value=new_vectors) as create:
                    self.run_generator("--sync-manager-content")
        self.assertEqual(create.call_args.args[1], [target[-1]["text"]])
        self.assertEqual(json.loads(generate_embeddings.CHUNKS_FILE.read_text()), target)
        vectors, status = load_valid_embeddings(
            target, generate_embeddings.EMBEDDING_FILE, generate_embeddings.METADATA_FILE
        )
        self.assertEqual(status, "cache is current")
        np.testing.assert_array_equal(vectors[0], self.original_vectors[0])
        np.testing.assert_array_equal(vectors[1], new_vectors[0])

        before = self.artifacts()
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
            with patch.object(generate_embeddings, "create_embeddings") as create:
                self.run_generator("--sync-manager-content")
        create.assert_not_called()
        self.assertEqual(self.artifacts(), before)

    def test_missing_key_preserves_active_artifacts(self) -> None:
        before = self.artifacts()
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
            with self.assertRaisesRegex(SystemExit, "OPENAI_API_KEY is not set"):
                self.run_generator("--sync-manager-content")
        self.assertEqual(self.artifacts(), before)

    def test_generation_failure_preserves_active_artifacts(self) -> None:
        before = self.artifacts()
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-placeholder"}):
            with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=Mock())}):
                with patch.object(generate_embeddings, "create_embeddings", side_effect=RuntimeError("API unavailable")):
                    with self.assertRaisesRegex(RuntimeError, "API unavailable"):
                        self.run_generator("--sync-manager-content")
        self.assertEqual(self.artifacts(), before)

    def test_staging_failure_preserves_active_artifacts(self) -> None:
        before = self.artifacts()
        with patch.object(generate_embeddings, "save_embedding_cache", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                generate_embeddings.save_synchronized_content(self.original, self.original_vectors)
        self.assertEqual(self.artifacts(), before)

    def test_dry_run_makes_no_cache_access_or_writes(self) -> None:
        before = self.artifacts()
        with patch.object(generate_embeddings, "load_valid_embeddings") as load_cache:
            with patch.object(generate_embeddings, "create_embeddings") as create:
                self.run_generator("--sync-manager-content", "--dry-run")
        load_cache.assert_not_called()
        create.assert_not_called()
        self.assertEqual(self.artifacts(), before)

    def test_default_mode_keeps_pending_manager_content_out_of_active_cache(self) -> None:
        before = self.artifacts()
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
            with patch.object(generate_embeddings, "create_embeddings") as create:
                self.run_generator()
        create.assert_not_called()
        self.assertEqual(self.artifacts(), before)


if __name__ == "__main__":
    unittest.main()
