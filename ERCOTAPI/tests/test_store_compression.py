"""Lossless packaged-generation loading for read-only deployments."""

from __future__ import annotations

import gzip
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ERCOTAPI.rag_ingestion.store import load_generation, write_generation


class CompressedStoreTests(unittest.TestCase):
    def test_load_generation_accepts_losslessly_compressed_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = Path(temporary)
            manifest = {
                "schema_version": 1,
                "generation_id": "packaged",
                "embedding_model": "test-model",
                "documents": {},
                "content": {},
                "summary": {"chunks": 1},
            }
            written = write_generation(
                store,
                "packaged",
                manifest,
                [{"chunk_id": "one", "text": "saved vector"}],
                np.asarray([[0.1, 0.2, 0.3]], dtype="float32"),
            )
            for source_name, compressed_name in (
                ("chunks.json", "chunks.json.gz"),
                ("embeddings.npy", "embeddings.npy.gz"),
            ):
                source = written.path / source_name
                with source.open("rb") as input_handle:
                    with gzip.open(written.path / compressed_name, "wb") as output_handle:
                        shutil.copyfileobj(input_handle, output_handle)
                source.unlink()

            loaded = load_generation(store)

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.chunks[0]["text"], "saved vector")
            np.testing.assert_allclose(
                loaded.embeddings,
                np.asarray([[0.1, 0.2, 0.3]], dtype="float32"),
            )


if __name__ == "__main__":
    unittest.main()
