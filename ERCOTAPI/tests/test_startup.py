"""Deployment startup behavior for the central-only ERCOT indexes."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from ERCOTAPI.rag_ingestion import startup
from ERCOTAPI.rag_ingestion.config import IngestionConfig, SourceRoot
from ERCOTAPI.rag_ingestion.pipeline import IngestionPipeline
from ERCOTAPI.rag_ingestion.retrieval import load_index
from ERCOTAPI.rag_ingestion.startup import (
    CentralIndexUnavailable,
    load_startup_index,
    startup_index_state,
    startup_source_roots,
)
from ERCOTAPI.rag_ingestion.store import current_generation_id, load_generation
from ERCOTAPI.tests.fakes import FakeEmbedder


class StartupTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repo_root = Path(temporary.name)
        self.checked_root = self.repo_root / "sources" / "checked"
        self.archive_root = self.repo_root / "NEWS" / "official"
        self.generated_root = self.repo_root / "NEWS" / "generated"
        self.checked_root.mkdir(parents=True)
        self.archive_root.mkdir(parents=True)
        self.generated_root.mkdir(parents=True)
        self.index_dir = self.repo_root / ".rag_store"
        self.legacy_chunks = self.repo_root / "legacy_chunks.json"
        self.legacy_embeddings = self.repo_root / "legacy_embeddings.npy"
        self.config = IngestionConfig(
            repo_root=self.repo_root,
            index_dir=self.index_dir,
            source_roots=(
                SourceRoot(
                    name="checked_in",
                    path=self.checked_root,
                    source_authority="ERCOT",
                    is_generated=False,
                    default_source_kind="Planning Guide",
                    default_collections=("general", "planning"),
                ),
                SourceRoot(
                    name="official_downloads",
                    path=self.archive_root,
                    source_authority="ERCOT",
                    is_generated=False,
                    default_source_kind="Official Document",
                    default_collections=("general",),
                ),
                SourceRoot(
                    name="generated_news",
                    path=self.generated_root,
                    source_authority="Generated",
                    is_generated=True,
                    default_source_kind="News Summary",
                    default_collections=("news", "market"),
                ),
            ),
            embedding_model="test-embedding-v1",
            chunk_size=200,
            chunk_overlap=20,
            legacy_chunks_path=self.legacy_chunks,
            legacy_embeddings_path=self.legacy_embeddings,
        )
        self.addCleanup(
            startup._VALIDATED_GENERATIONS.pop,
            self.index_dir.resolve(strict=False),
            None,
        )

    def write_checked(self, name: str, text: str) -> Path:
        path = self.checked_root / name
        path.write_text(text, encoding="utf-8")
        return path

    def write_legacy_cache(self) -> None:
        self.legacy_chunks.write_text(
            json.dumps([{"source": "old.txt", "text": "OLD LEGACY CONTENT"}]),
            encoding="utf-8",
        )
        np.save(
            self.legacy_embeddings,
            np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype="float32"),
        )

    def forget_process_validation(self) -> None:
        startup._VALIDATED_GENERATIONS.pop(
            self.index_dir.resolve(strict=False),
            None,
        )

    def test_missing_store_bootstraps_a_central_generation(self) -> None:
        self.write_checked("planning_guide.txt", "Current planning requirement for 2026.")
        embedder = FakeEmbedder()

        loaded = load_startup_index(
            "planning",
            config=self.config,
            embedder=embedder,
            bootstrap_on_missing=True,
            refresh=False,
        )

        self.assertEqual(loaded.source, "central")
        self.assertTrue(loaded.ready)
        self.assertTrue(loaded.generation_id)
        self.assertEqual(embedder.embedded_text_count, 1)
        self.assertIn("Current planning requirement", loaded.chunks[0]["text"])

    def test_dangling_current_pointer_bootstraps_a_central_generation(self) -> None:
        self.write_checked("planning_guide.txt", "Current deployment planning requirement.")
        self.index_dir.mkdir(parents=True)
        (self.index_dir / "CURRENT").write_text(
            "generation-not-shipped\n",
            encoding="utf-8",
        )
        embedder = FakeEmbedder()

        self.assertEqual(startup_index_state(self.config)[0], None)
        loaded = load_startup_index(
            "planning",
            config=self.config,
            embedder=embedder,
            bootstrap_on_missing=True,
            refresh=False,
        )

        self.assertEqual(loaded.source, "central")
        self.assertTrue(loaded.ready)
        self.assertNotEqual(loaded.generation_id, "generation-not-shipped")
        self.assertEqual(current_generation_id(self.index_dir), loaded.generation_id)
        self.assertEqual(embedder.embedded_text_count, 1)

    def test_disabled_bootstrap_never_returns_a_legacy_cache(self) -> None:
        self.write_checked("planning_guide.txt", "Current checked-in requirement.")
        self.write_legacy_cache()

        with self.assertRaisesRegex(CentralIndexUnavailable, "bootstrap is disabled"):
            load_startup_index(
                "general",
                config=self.config,
                embedder=FakeEmbedder(),
                bootstrap_on_missing=False,
                refresh=False,
            )

        self.assertIsNone(current_generation_id(self.index_dir))

    def test_missing_configured_checked_in_root_fails_closed(self) -> None:
        self.write_checked("old_protocol.txt", "Old 2023 nodal protocol requirement.")
        missing_current = SourceRoot(
            name="nodal_protocol_uploads",
            path=self.repo_root / "sources" / "missing-current-protocols",
            source_authority="ERCOT",
            is_generated=False,
            default_source_kind="Protocol",
            default_collections=("general", "protocols"),
        )
        config = replace(
            self.config,
            source_roots=(*self.config.source_roots, missing_current),
        )

        with self.assertRaisesRegex(CentralIndexUnavailable, "source roots are missing"):
            load_startup_index(
                "protocols",
                config=config,
                embedder=FakeEmbedder(),
                refresh=False,
            )

        self.assertIsNone(current_generation_id(self.index_dir))

    def test_same_process_validation_is_not_reused_for_a_changed_source_config(self) -> None:
        self.write_checked("old_protocol.txt", "Old static nodal protocol requirement.")
        initial = load_startup_index(
            "general",
            config=self.config,
            embedder=FakeEmbedder(),
            bootstrap_on_missing=True,
            refresh=False,
        )
        self.assertTrue(initial.ready)

        missing_current = SourceRoot(
            name="nodal_protocol_uploads",
            path=self.repo_root / "sources" / "missing-current-protocols",
            source_authority="ERCOT",
            is_generated=False,
            default_source_kind="Protocol",
            default_collections=("general", "protocols"),
        )
        changed_config = replace(
            self.config,
            source_roots=(*self.config.source_roots, missing_current),
        )

        with self.assertRaisesRegex(CentralIndexUnavailable, "source roots are missing"):
            load_startup_index(
                "general",
                config=changed_config,
                embedder=FakeEmbedder(),
                refresh=False,
            )

    def test_empty_required_root_is_retried_then_must_be_represented(self) -> None:
        self.write_checked("old_protocol.txt", "Old static nodal protocol requirement.")
        current_root = self.repo_root / "sources" / "current-protocols"
        current_root.mkdir()
        current_source = SourceRoot(
            name="nodal_protocol_uploads",
            path=current_root,
            source_authority="ERCOT",
            is_generated=False,
            default_source_kind="Protocol",
            default_collections=("general", "protocols"),
        )
        config = replace(
            self.config,
            source_roots=(*self.config.source_roots, current_source),
        )

        with self.assertRaisesRegex(CentralIndexUnavailable, "no retrievable documents"):
            load_startup_index(
                "protocols",
                config=config,
                embedder=FakeEmbedder(),
                bootstrap_on_missing=True,
                refresh=False,
            )

        (current_root / "current_protocol.txt").write_text(
            "Current nodal protocol requirement.",
            encoding="utf-8",
        )
        loaded = load_startup_index(
            "protocols",
            config=config,
            embedder=FakeEmbedder(),
            bootstrap_on_missing=True,
            refresh=False,
        )

        self.assertTrue(loaded.ready)
        self.assertTrue(any("Current nodal" in chunk["text"] for chunk in loaded.chunks))

    def test_corrupt_current_pointer_becomes_an_explicit_startup_error(self) -> None:
        self.index_dir.mkdir(parents=True)
        (self.index_dir / "CURRENT").write_text("../unsafe\n", encoding="utf-8")

        with self.assertRaisesRegex(CentralIndexUnavailable, "index state"):
            startup_index_state(self.config)

    def test_process_restart_loads_saved_generation_without_embedding(self) -> None:
        document = self.write_checked("planning_guide.txt", "Planning version one.")
        first = load_startup_index(
            "general",
            config=self.config,
            embedder=FakeEmbedder(),
            bootstrap_on_missing=True,
        )
        document.write_text("Planning version two for 2026.", encoding="utf-8")
        self.forget_process_validation()
        embedder = FakeEmbedder()

        embedder = FakeEmbedder()
        second = load_startup_index(
            "general",
            config=self.config,
            embedder=embedder,
        )

        self.assertEqual(second.generation_id, first.generation_id)
        self.assertEqual(embedder.calls, [])
        self.assertIn("version one", second.chunks[0]["text"])

    def test_explicit_refresh_rechecks_an_already_validated_generation(self) -> None:
        document = self.write_checked("planning_guide.txt", "Planning version one.")
        first = load_startup_index(
            "general",
            config=self.config,
            embedder=FakeEmbedder(),
            bootstrap_on_missing=True,
        )
        document.write_text("Planning version two.", encoding="utf-8")

        second = load_startup_index(
            "general",
            config=self.config,
            embedder=FakeEmbedder(),
            refresh=True,
        )

        self.assertNotEqual(second.generation_id, first.generation_id)
        self.assertIn("version two", second.chunks[0]["text"])

    def test_existing_complete_generation_can_skip_refresh_without_embedding(self) -> None:
        self.write_checked("planning_guide.txt", "Already indexed planning requirement.")
        initial = IngestionPipeline(self.config, embedder=FakeEmbedder()).update(
            [self.checked_root]
        )
        embedder = FakeEmbedder()

        loaded = load_startup_index(
            "general",
            config=self.config,
            embedder=embedder,
            bootstrap_on_missing=True,
            refresh=False,
        )

        self.assertEqual(loaded.generation_id, initial["generation"])
        self.assertEqual(embedder.calls, [])

    def test_download_archive_and_generated_summaries_are_not_startup_sources(self) -> None:
        self.write_checked("planning_guide.txt", "Safe checked-in planning requirement.")
        (self.archive_root / "NPRR_FAIL.txt").write_text(
            "FAIL archive should be monitor-owned",
            encoding="utf-8",
        )
        (self.generated_root / "summary_FAIL.txt").write_text(
            "FAIL generated summary should not bootstrap",
            encoding="utf-8",
        )
        embedder = FakeEmbedder()
        embedder.fail_on = "FAIL"

        loaded = load_startup_index(
            "general",
            config=self.config,
            embedder=embedder,
            bootstrap_on_missing=True,
            refresh=False,
        )

        self.assertTrue(loaded.ready)
        self.assertEqual([root.name for root in startup_source_roots(self.config)], ["checked_in"])
        generation = load_generation(self.index_dir)
        self.assertIsNotNone(generation)
        assert generation is not None
        self.assertNotIn("NEWS/official/NPRR_FAIL.txt", generation.manifest["documents"])
        self.assertNotIn("NEWS/generated/summary_FAIL.txt", generation.manifest["documents"])

    def test_partial_bootstrap_is_refused_then_retried_without_legacy_fallback(self) -> None:
        self.write_checked("good.txt", "Good current planning requirement.")
        broken = self.write_checked("broken.txt", "FAIL new planning requirement.")
        self.write_legacy_cache()
        failing = FakeEmbedder()
        failing.fail_on = "FAIL"

        with self.assertRaisesRegex(CentralIndexUnavailable, "refusing legacy or partial"):
            load_startup_index(
                "general",
                config=self.config,
                embedder=failing,
                bootstrap_on_missing=True,
                refresh=False,
            )

        partial = load_index("general", config=self.config, allow_legacy=True)
        self.assertEqual(partial.source, "central")
        self.assertNotIn("OLD LEGACY CONTENT", [chunk["text"] for chunk in partial.chunks])

        broken.write_text("Repaired current planning requirement.", encoding="utf-8")
        repaired = load_startup_index(
            "general",
            config=self.config,
            embedder=FakeEmbedder(),
            bootstrap_on_missing=True,
            refresh=False,
        )

        self.assertEqual(repaired.source, "central")
        self.assertEqual(len(repaired.chunks), 2)
        self.assertTrue(
            any("Repaired current" in chunk["text"] for chunk in repaired.chunks)
        )


if __name__ == "__main__":
    unittest.main()
