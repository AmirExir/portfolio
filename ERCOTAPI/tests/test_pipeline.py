"""Incremental-ingestion behavior using only temporary files and fake vectors."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest import mock
from xml.sax.saxutils import escape

import numpy as np

from ERCOTAPI.rag_ingestion.config import IngestionConfig, SourceRoot
from ERCOTAPI.rag_ingestion.pipeline import IngestionPipeline
from ERCOTAPI.rag_ingestion import store
from ERCOTAPI.rag_ingestion.store import current_generation_id, load_generation
from ERCOTAPI.tests.fakes import FakeEmbedder


class PipelineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repo_root = Path(temporary.name)
        self.official_root = self.repo_root / "ERCOTAPI" / "NEWS" / "official"
        self.generated_root = self.repo_root / "ERCOTAPI" / "NEWS" / "generated"
        self.official_root.mkdir(parents=True)
        self.generated_root.mkdir(parents=True)
        self.index_dir = self.repo_root / "ERCOTAPI" / ".rag_index"
        self.config = IngestionConfig(
            repo_root=self.repo_root,
            index_dir=self.index_dir,
            source_roots=(
                SourceRoot(
                    name="official_downloads",
                    path=self.official_root,
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
        )
        self.embedder = FakeEmbedder()
        self.pipeline = IngestionPipeline(self.config, embedder=self.embedder)

    def write(self, relative: str, content: str | bytes, *, generated: bool = False) -> Path:
        root = self.generated_root if generated else self.official_root
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        return path

    def relative(self, path: Path) -> str:
        return path.relative_to(self.repo_root).as_posix()

    def write_docx(self, relative: str, paragraphs: list[str]) -> Path:
        path = self.official_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        body = "".join(
            f"<w:p><w:r><w:t>{escape(paragraph)}</w:t></w:r></w:p>"
            for paragraph in paragraphs
        )
        document_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/'
            f'wordprocessingml/2006/main"><w:body>{body}</w:body></w:document>'
        )
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("word/document.xml", document_xml)
        return path

    def active(self):
        generation = load_generation(self.index_dir)
        self.assertIsNotNone(generation)
        return generation

    def legacy_pipeline(
        self,
        *,
        source_text: str,
        cached_text: str,
    ) -> tuple[IngestionPipeline, FakeEmbedder, Path]:
        legacy_root = self.repo_root / "legacy_sources"
        legacy_root.mkdir()
        document = legacy_root / "ercot_reference.txt"
        document.write_text(source_text, encoding="utf-8")
        chunks_path = self.repo_root / "legacy_chunks.json"
        chunks_path.write_text(
            json.dumps(
                [
                    {
                        "source": document.name,
                        "chunk_index": 0,
                        "text": cached_text,
                    }
                ]
            ),
            encoding="utf-8",
        )
        embeddings_path = self.repo_root / "legacy_embeddings.npy"
        np.save(embeddings_path, np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype="float32"))
        config = replace(
            self.config,
            source_roots=(
                SourceRoot(
                    name="authoritative_static",
                    path=legacy_root,
                    source_authority="ERCOT",
                    is_generated=False,
                    default_source_kind="ERCOT Reference",
                    default_collections=("general",),
                ),
            ),
            legacy_chunks_path=chunks_path,
            legacy_embeddings_path=embeddings_path,
            legacy_sources_dir=legacy_root,
            legacy_embedding_model=self.config.embedding_model,
            legacy_chunk_size=self.config.chunk_size,
            legacy_chunk_overlap=self.config.chunk_overlap,
        )
        embedder = FakeEmbedder()
        return IngestionPipeline(config, embedder=embedder), embedder, document

    def test_new_document_is_ingested_with_required_manifest_and_metadata(self) -> None:
        document = self.write(
            "protocols/NPRR1234_approved.txt",
            "NPRR 1234 establishes an authoritative protocol requirement.",
        )
        sidecar = document.with_name(f"{document.name}.metadata.json")
        sidecar.write_text(
            json.dumps(
                {
                    "title": "NPRR 1234 Approval",
                    "original_url": "https://www.ercot.com/files/NPRR1234",
                    "url_aliases": [
                        "https://www.ercot.com/files/NPRR1234",
                        "https://www.ercot.com/download/NPRR1234",
                    ],
                    "source_page_urls": ["https://www.ercot.com/mktrules/issues/nprr1234"],
                    "provenance": [
                        {
                            "original_url": "https://www.ercot.com/files/NPRR1234",
                            "document_status": "Approved",
                        }
                    ],
                    "downloaded_at": "2026-07-16T12:00:00Z",
                    "effective_date": "2026-08-01",
                }
            ),
            encoding="utf-8",
        )

        result = self.pipeline.update()

        self.assertTrue(result["changed"])
        self.assertEqual(result["embedded_chunks"], 1)
        self.assertEqual(result["errors"], 0)
        self.assertEqual(self.embedder.embedded_text_count, 1)
        generation = self.active()
        record = generation.manifest["documents"][self.relative(document)]
        required_fields = {
            "path",
            "source_path",
            "size",
            "mtime_ns",
            "modification_timestamp",
            "sha256",
            "document_type",
            "source_category",
            "ingestion_timestamp",
            "collections",
            "status",
            "error",
            "document_id",
            "chunk_ids",
        }
        self.assertTrue(required_fields.issubset(record))
        self.assertEqual(record["status"], "ingested")
        self.assertIsNone(record["error"])
        self.assertEqual(record["source_authority"], "ERCOT")
        self.assertFalse(record["is_generated"])
        self.assertEqual(record["source_kind"], "NPRR")
        self.assertEqual(record["document_number"], "NPRR1234")
        self.assertIn("general", record["collections"])
        self.assertIn("protocols", record["collections"])
        self.assertEqual(generation.chunks[0]["original_url"], "https://www.ercot.com/files/NPRR1234")
        self.assertEqual(len(record["url_aliases"]), 2)
        self.assertEqual(generation.chunks[0]["url_aliases"], record["url_aliases"])
        self.assertEqual(generation.chunks[0]["provenance"], record["provenance"])

    def test_docx_heading_metadata_reaches_manifest_and_retrieval_chunk(self) -> None:
        document = self.write_docx(
            "planning_guide_02-060126.docx",
            [
                "ERCOT Planning Guide",
                "Section 2: Definitions and Acronyms",
                "June 1, 2026",
                "Current authoritative planning definitions.",
            ],
        )

        result = self.pipeline.update()

        self.assertEqual(result["errors"], 0)
        generation = self.active()
        record = generation.manifest["documents"][self.relative(document)]
        self.assertEqual(
            record["title"],
            "ERCOT Planning Guide — Section 2: Definitions and Acronyms",
        )
        self.assertEqual(record["effective_date"], "2026-06-01")
        self.assertIn("planning", record["collections"])
        self.assertEqual(generation.chunks[0]["effective_date"], "2026-06-01")
        self.assertEqual(generation.chunks[0]["title"], record["title"])

    def test_incomplete_stale_legacy_cache_is_rejected_and_current_source_is_embedded(self) -> None:
        pipeline, embedder, document = self.legacy_pipeline(
            source_text="Cached ERCOT requirement. Current authoritative amendment.",
            cached_text="Cached ERCOT requirement.",
        )

        result = pipeline.update()

        self.assertEqual(result["embedded_chunks"], 1)
        self.assertEqual(result["reused_chunks"], 0)
        self.assertEqual(embedder.embedded_text_count, 1)
        generation = load_generation(pipeline.config.index_dir)
        self.assertIsNotNone(generation)
        assert generation is not None
        self.assertEqual(len(generation.chunks), 1)
        self.assertEqual(
            generation.chunks[0]["text"],
            "Cached ERCOT requirement. Current authoritative amendment.",
        )
        record = generation.manifest["documents"][self.relative(document)]
        self.assertEqual(record["indexed_sha256"], record["sha256"])

    def test_matching_normalized_legacy_cache_reuses_vectors_without_embedding(self) -> None:
        pipeline, embedder, _ = self.legacy_pipeline(
            source_text="Current authoritative\n\nERCOT requirement.",
            cached_text="Current authoritative ERCOT requirement.",
        )

        result = pipeline.update()

        self.assertEqual(result["embedded_chunks"], 0)
        self.assertEqual(result["reused_chunks"], 1)
        self.assertEqual(embedder.calls, [])
        generation = load_generation(pipeline.config.index_dir)
        self.assertIsNotNone(generation)
        assert generation is not None
        self.assertEqual(
            generation.chunks[0]["text"],
            "Current authoritative ERCOT requirement.",
        )
        np.testing.assert_array_equal(
            generation.embeddings,
            np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype="float32"),
        )

    def test_unchanged_document_does_not_embed_or_publish(self) -> None:
        self.write("planning/PGRR456.txt", "PGRR 456 planning requirement text.")
        with mock.patch("ERCOTAPI.rag_ingestion.pipeline._utc_now", return_value="2026-07-16T12:00:00Z"):
            first = self.pipeline.update()
        first_generation = first["generation"]
        first_call_count = self.embedder.embedded_text_count

        with mock.patch("ERCOTAPI.rag_ingestion.pipeline._utc_now", return_value="2026-07-16T12:05:00Z"):
            second = self.pipeline.update()

        self.assertFalse(second["changed"])
        self.assertEqual(second["generation"], first_generation)
        self.assertEqual(self.embedder.embedded_text_count, first_call_count)
        self.assertEqual(current_generation_id(self.index_dir), first_generation)

    def test_empty_store_accepts_an_injected_provider_model(self) -> None:
        self.write("NPRR610.txt", "NPRR 610 protocol requirement.")
        self.write("PGRR610.txt", "PGRR 610 planning requirement.")
        configured = replace(self.config, embedding_model="configured-production-model")
        embedder = FakeEmbedder()

        result = IngestionPipeline(configured, embedder=embedder).update()

        self.assertEqual(result["embedded_chunks"], 2)
        generation = self.active()
        self.assertEqual(generation.manifest["embedding_model"], embedder.model)
        self.assertEqual(generation.embeddings.shape, (2, embedder.dimension))

    def test_modified_same_size_and_mtime_is_reembedded_without_old_chunks(self) -> None:
        document = self.write("notices/market_notice.txt", "Alpha requirement v1.")
        original_stat = document.stat()
        first = self.pipeline.update()
        first_generation = self.active()
        old_hash = first_generation.chunks[0]["content_hash"]
        self.assertEqual(first["embedded_chunks"], 1)

        document.write_text("Bravo requirement v2.", encoding="utf-8")
        os.utime(document, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        scan = self.pipeline.scan()
        actions = {item["path"]: item["action"] for item in scan["files"]}
        self.assertEqual(actions[self.relative(document)], "modified")

        second = self.pipeline.update()

        self.assertEqual(second["embedded_chunks"], 1)
        self.assertEqual(self.embedder.embedded_text_count, 2)
        generation = self.active()
        self.assertEqual(len(generation.chunks), 1)
        self.assertIn("Bravo requirement", generation.chunks[0]["text"])
        self.assertNotIn(old_hash, generation.manifest["content"])

    def test_exact_duplicate_content_has_one_payload_and_manifest_alias(self) -> None:
        canonical = self.write("a/NPRR987.txt", "NPRR 987 exact duplicate body.")
        duplicate = self.write("b/NPRR987_copy.txt", "NPRR 987 exact duplicate body.")
        canonical.with_name(f"{canonical.name}.metadata.json").write_text(
            json.dumps(
                {
                    "original_url": "https://www.ercot.com/files/NPRR987-a",
                    "url_aliases": ["https://www.ercot.com/files/NPRR987-a"],
                    "source_page_urls": ["https://www.ercot.com/issues/NPRR987"],
                    "document_status": "Pending",
                    "published_date": "2025-01-01",
                    "provenance": [{"original_url": "https://www.ercot.com/files/NPRR987-a"}],
                }
            ),
            encoding="utf-8",
        )
        duplicate.with_name(f"{duplicate.name}.metadata.json").write_text(
            json.dumps(
                {
                    "original_url": "https://www.ercot.com/files/NPRR987-b",
                    "url_aliases": ["https://www.ercot.com/files/NPRR987-b"],
                    "source_page_urls": ["https://www.ercot.com/committees/NPRR987"],
                    "document_status": "Approved",
                    "published_date": "2026-01-01",
                    "provenance": [{"original_url": "https://www.ercot.com/files/NPRR987-b"}],
                }
            ),
            encoding="utf-8",
        )

        result = self.pipeline.update()

        generation = self.active()
        records = generation.manifest["documents"]
        canonical_record = records[self.relative(canonical)]
        duplicate_record = records[self.relative(duplicate)]
        self.assertEqual(result["embedded_chunks"], 1)
        self.assertEqual(self.embedder.embedded_text_count, 1)
        self.assertEqual(len(generation.chunks), 1)
        self.assertEqual(len(generation.manifest["content"]), 1)
        self.assertEqual(canonical_record["status"], "ingested")
        self.assertEqual(duplicate_record["status"], "duplicate")
        self.assertEqual(duplicate_record["duplicate_of"], self.relative(canonical))
        self.assertEqual(canonical_record["chunk_ids"], duplicate_record["chunk_ids"])
        self.assertEqual(
            generation.chunks[0]["aliases"],
            sorted((self.relative(canonical), self.relative(duplicate))),
        )
        self.assertEqual(
            generation.chunks[0]["url_aliases"],
            [
                "https://www.ercot.com/files/NPRR987-a",
                "https://www.ercot.com/files/NPRR987-b",
            ],
        )
        self.assertEqual(len(generation.chunks[0]["provenance"]), 2)
        self.assertEqual(generation.chunks[0]["document_status"], "Approved")
        self.assertEqual(generation.chunks[0]["published_date"], "2026-01-01")

    def test_path_scoped_metadata_update_preserves_duplicate_aliases(self) -> None:
        canonical = self.write("a/NPRR988.txt", "NPRR 988 exact duplicate body.")
        duplicate = self.write("b/NPRR988_copy.txt", "NPRR 988 exact duplicate body.")
        self.pipeline.update()
        embedded_before = self.embedder.embedded_text_count
        canonical.with_name(f"{canonical.name}.metadata.json").write_text(
            json.dumps({"title": "Updated canonical title"}),
            encoding="utf-8",
        )

        scan = self.pipeline.scan(paths=[canonical])
        self.assertEqual(scan["files"][0]["action"], "metadata_changed")
        self.pipeline.update(paths=[canonical])

        generation = self.active()
        self.assertEqual(self.embedder.embedded_text_count, embedded_before)
        self.assertEqual(
            generation.chunks[0]["aliases"],
            sorted((self.relative(canonical), self.relative(duplicate))),
        )
        self.assertIn("protocols", generation.chunks[0]["collections"])

    def test_metadata_rich_duplicate_becomes_canonical_regardless_of_path_order(self) -> None:
        generic = self.write("a/document.txt", "Exact shared ERCOT requirement body.")
        specific = self.write("z/document.txt", "Exact shared ERCOT requirement body.")
        specific.with_name(f"{specific.name}.metadata.json").write_text(
            json.dumps(
                {
                    "title": "NPRR 2048 Approved Requirement",
                    "source_kind": "NPRR",
                    "document_number": "NPRR2048",
                    "document_status": "Approved",
                    "effective_date": "2026-07-01",
                    "original_url": "https://www.ercot.com/NPRR2048",
                }
            ),
            encoding="utf-8",
        )

        self.pipeline.update()

        generation = self.active()
        records = generation.manifest["documents"]
        self.assertEqual(records[self.relative(specific)]["status"], "ingested")
        self.assertEqual(records[self.relative(generic)]["status"], "duplicate")
        self.assertEqual(generation.chunks[0]["source_path"], self.relative(specific))
        self.assertEqual(generation.chunks[0]["document_number"], "NPRR2048")

    def test_deleted_document_is_removed_atomically_and_prior_generation_survives(self) -> None:
        document = self.write("NOGRR321.txt", "NOGRR 321 operating guide requirement.")
        first = self.pipeline.update()
        previous_generation = first["generation"]
        self.assertEqual(len(self.active().chunks), 1)

        document.unlink()
        result = self.pipeline.update()

        self.assertTrue(result["changed"])
        current = self.active()
        self.assertEqual(current.manifest["documents"][self.relative(document)]["status"], "deleted")
        self.assertEqual(current.chunks, [])
        prior = load_generation(self.index_dir, previous_generation)
        self.assertIsNotNone(prior)
        self.assertEqual(len(prior.chunks), 1)

    def test_explicit_source_root_reconciliation_tombstones_deleted_file(self) -> None:
        document = self.write("NPRR322.txt", "NPRR 322 protocol requirement.")
        self.pipeline.update()
        embedded_before = self.embedder.embedded_text_count
        document.unlink()

        scan = self.pipeline.scan(paths=[self.official_root])
        deleted = next(item for item in scan["files"] if item["path"] == self.relative(document))
        self.assertEqual(deleted["action"], "deleted")

        result = self.pipeline.update(paths=[self.official_root])

        self.assertTrue(result["changed"])
        self.assertEqual(self.embedder.embedded_text_count, embedded_before)
        generation = self.active()
        self.assertEqual(
            generation.manifest["documents"][self.relative(document)]["status"],
            "deleted",
        )
        self.assertEqual(generation.chunks, [])

    def test_rename_reuses_vectors_and_stable_chunk_ids(self) -> None:
        old_path = self.write("old/NPRR765.txt", "NPRR 765 rename-stable text.")
        self.pipeline.update()
        old_generation = self.active()
        old_chunk_ids = [chunk["chunk_id"] for chunk in old_generation.chunks]
        embedded_before = self.embedder.embedded_text_count
        new_path = self.official_root / "new" / old_path.name
        new_path.parent.mkdir(parents=True)
        old_path.rename(new_path)

        scan = self.pipeline.scan()
        actions = {item["path"]: item["action"] for item in scan["files"]}
        self.assertEqual(actions[self.relative(new_path)], "renamed")
        self.assertEqual(actions[self.relative(old_path)], "deleted")
        result = self.pipeline.update()

        generation = self.active()
        self.assertEqual(result["embedded_chunks"], 0)
        self.assertEqual(self.embedder.embedded_text_count, embedded_before)
        self.assertEqual([chunk["chunk_id"] for chunk in generation.chunks], old_chunk_ids)
        self.assertEqual(
            generation.manifest["documents"][self.relative(old_path)]["status"], "deleted"
        )
        self.assertEqual(
            generation.manifest["documents"][self.relative(new_path)]["status"], "ingested"
        )

    def test_renamed_document_with_bad_sidecar_retains_prior_payload_as_stale(self) -> None:
        old_path = self.write("old/PGRR766.txt", "PGRR 766 rename-safe planning text.")
        self.pipeline.update()
        before = self.active()
        old_chunk_ids = [chunk["chunk_id"] for chunk in before.chunks]
        old_vectors = before.embeddings.copy()
        embedded_before = self.embedder.embedded_text_count
        new_path = self.official_root / "new" / old_path.name
        new_path.parent.mkdir(parents=True)
        old_path.rename(new_path)
        new_path.with_name(f"{new_path.name}.metadata.json").write_text(
            "{invalid JSON",
            encoding="utf-8",
        )

        result = self.pipeline.update()

        generation = self.active()
        record = generation.manifest["documents"][self.relative(new_path)]
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["embedded_chunks"], 0)
        self.assertEqual(self.embedder.embedded_text_count, embedded_before)
        self.assertEqual(record["status"], "error")
        self.assertTrue(record["stale"])
        self.assertIn("Invalid metadata sidecar", record["error"])
        self.assertEqual([chunk["chunk_id"] for chunk in generation.chunks], old_chunk_ids)
        self.assertTrue(generation.chunks[0]["stale"])
        np.testing.assert_array_equal(generation.embeddings, old_vectors)

    def test_malformed_document_records_error_while_valid_document_ingests(self) -> None:
        malformed = self.write("broken/NPRR999.docx", b"this is not a ZIP archive")
        valid = self.write("valid/PGRR999.txt", "PGRR 999 valid planning content.")

        result = self.pipeline.update()

        generation = self.active()
        records = generation.manifest["documents"]
        self.assertEqual(result["errors"], 1)
        self.assertEqual(records[self.relative(malformed)]["status"], "error")
        self.assertIn("Unable to open DOCX", records[self.relative(malformed)]["error"])
        self.assertIn("ingestion_timestamp", records[self.relative(malformed)])
        self.assertIn("last_attempted_at", records[self.relative(malformed)])
        self.assertEqual(records[self.relative(valid)]["status"], "ingested")
        self.assertEqual(len(generation.chunks), 1)
        self.assertIn("valid planning content", generation.chunks[0]["text"])

    def test_malformed_sidecar_retains_hash_and_deterministic_document_id(self) -> None:
        document = self.write("NPRR998.txt", "NPRR 998 source body.")
        document.with_name(f"{document.name}.metadata.json").write_text(
            "{not valid JSON",
            encoding="utf-8",
        )

        result = self.pipeline.update()

        record = self.active().manifest["documents"][self.relative(document)]
        self.assertEqual(result["errors"], 1)
        self.assertEqual(record["status"], "error")
        self.assertEqual(len(record["sha256"]), 64)
        self.assertEqual(len(record["document_id"]), 64)
        self.assertIn("Invalid metadata sidecar", record["error"])

    def test_unsupported_extension_is_skipped_with_reason(self) -> None:
        unsupported = self.write("payload.exe", b"not a document")
        unsupported.with_name(f"{unsupported.name}.metadata.json").write_text(
            json.dumps(
                {
                    "title": "Legacy Market Notice",
                    "source_kind": "Market Notice",
                    "original_url": "https://www.ercot.com/legacy.doc",
                    "downloaded_at": "2026-07-16T12:00:00Z",
                }
            ),
            encoding="utf-8",
        )

        scan = self.pipeline.scan()
        item = next(entry for entry in scan["files"] if entry["path"] == self.relative(unsupported))
        self.assertEqual(item["action"], "skipped")
        self.assertIn("Unsupported document extension: .exe", item["error"])

        result = self.pipeline.update()
        record = self.active().manifest["documents"][self.relative(unsupported)]
        self.assertEqual(record["status"], "skipped")
        self.assertIn("Unsupported document extension: .exe", record["error"])
        self.assertEqual(record["title"], "Legacy Market Notice")
        self.assertEqual(record["source_kind"], "Market Notice")
        self.assertIn("market", record["collections"])
        self.assertEqual(record["original_url"], "https://www.ercot.com/legacy.doc")
        self.assertIn("ingestion_timestamp", record)
        self.assertEqual(result["embedded_chunks"], 0)
        self.assertEqual(self.embedder.calls, [])

    def test_directory_discovery_skips_symlinked_files_outside_source_root(self) -> None:
        outside = self.repo_root / "private-local-note.txt"
        outside.write_text("must never be embedded", encoding="utf-8")
        linked = self.official_root / "NPRR_symlink.txt"
        try:
            linked.symlink_to(outside)
        except OSError as exc:  # pragma: no cover - platform permission fallback
            self.skipTest(f"symlinks are unavailable: {exc}")

        scan = self.pipeline.scan()

        self.assertNotIn(self.relative(linked), {item["path"] for item in scan["files"]})
        self.assertEqual(self.embedder.calls, [])

    def test_sidecar_symlink_is_rejected_without_reading_external_metadata(self) -> None:
        document = self.write("NPRR_sidecar.txt", "Safe official document body.")
        outside = self.repo_root / "private-metadata.json"
        outside.write_text('{"title":"must not be imported"}', encoding="utf-8")
        linked_sidecar = document.with_name(f"{document.name}.metadata.json")
        try:
            linked_sidecar.symlink_to(outside)
        except OSError as exc:  # pragma: no cover - platform permission fallback
            self.skipTest(f"symlinks are unavailable: {exc}")

        scan = self.pipeline.scan()

        record = next(item for item in scan["files"] if item["path"] == self.relative(document))
        self.assertEqual(record["action"], "error")
        self.assertIn("symbolic-link metadata sidecar", record["error"])
        self.assertNotIn("must not be imported", str(record))

    def test_dry_run_performs_no_manifest_index_or_embedding_mutation(self) -> None:
        self.write("NPRR222.txt", "NPRR 222 dry-run content.")

        result = self.pipeline.update(dry_run=True)

        self.assertEqual(result["command"], "update")
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["summary"], {"new": 1})
        self.assertFalse(self.index_dir.exists())
        self.assertEqual(self.embedder.calls, [])

    def test_force_reindex_reembeds_but_keeps_deterministic_chunk_ids(self) -> None:
        self.write("PGRR654.txt", "PGRR 654 force-reindex content.")
        first = self.pipeline.update()
        initial_ids = [chunk["chunk_id"] for chunk in self.active().chunks]
        initial_calls = self.embedder.embedded_text_count

        second = self.pipeline.update(force=True)

        self.assertTrue(second["changed"])
        self.assertNotEqual(second["generation"], first["generation"])
        self.assertEqual(self.embedder.embedded_text_count, initial_calls + len(initial_ids))
        self.assertEqual([chunk["chunk_id"] for chunk in self.active().chunks], initial_ids)

    def test_embedding_failure_keeps_prior_vectors_and_records_retryable_error(self) -> None:
        document = self.write("NPRR400.txt", "NPRR 400 prior working content.")
        sidecar = document.with_name(f"{document.name}.metadata.json")
        sidecar.write_text(
            json.dumps({"title": "Prior title", "original_url": "https://www.ercot.com/prior"}),
            encoding="utf-8",
        )
        self.pipeline.update()
        prior = self.active()
        prior_texts = [chunk["text"] for chunk in prior.chunks]
        prior_vectors = prior.embeddings.copy()

        document.write_text("NPRR 400 FAIL changed content.", encoding="utf-8")
        sidecar.write_text(
            json.dumps({"title": "Replacement title", "original_url": "https://www.ercot.com/new"}),
            encoding="utf-8",
        )
        self.embedder.fail_on = "FAIL"
        result = self.pipeline.update()

        current = self.active()
        record = current.manifest["documents"][self.relative(document)]
        self.assertEqual(result["errors"], 1)
        self.assertEqual(record["status"], "error")
        self.assertTrue(record["stale"])
        self.assertIn("synthetic embedding failure", record["error"])
        self.assertIn("last_attempted_at", record)
        self.assertEqual([chunk["text"] for chunk in current.chunks], prior_texts)
        self.assertEqual(current.chunks[0]["title"], "Prior title")
        self.assertEqual(current.chunks[0]["original_url"], "https://www.ercot.com/prior")
        self.assertTrue(current.chunks[0]["stale"])
        self.assertEqual(current.chunks[0]["current_content_hash"], record["sha256"])
        np.testing.assert_array_equal(current.embeddings, prior_vectors)

    def test_model_change_requires_atomic_full_rebuild(self) -> None:
        self.write("NPRR500.txt", "NPRR 500 original content.")
        first = self.pipeline.update()
        previous_id = first["generation"]
        self.write("PGRR500.txt", "PGRR 500 new content.")
        migrated_config = replace(self.config, embedding_model="test-embedding-v2")
        migrated_embedder = FakeEmbedder(dimension=6)
        migrated_embedder.model = "test-embedding-v2"
        migrated = IngestionPipeline(migrated_config, embedder=migrated_embedder)

        with self.assertRaisesRegex(RuntimeError, "Embedding model changed"):
            migrated.update()

        self.assertEqual(current_generation_id(self.index_dir), previous_id)
        self.assertEqual(self.active().manifest["embedding_model"], "test-embedding-v1")

        rebuilt = migrated.rebuild()

        self.assertTrue(rebuilt["changed"])
        generation = self.active()
        self.assertEqual(generation.manifest["embedding_model"], "test-embedding-v2")
        self.assertEqual(generation.embeddings.shape, (2, 6))
        self.assertEqual(len(generation.chunks), 2)

    def test_generation_retention_keeps_current_and_previous(self) -> None:
        document = self.write("NPRR501.txt", "NPRR 501 version 0.")
        retained_config = replace(self.config, generation_retention=3)
        pipeline = IngestionPipeline(retained_config, embedder=self.embedder)
        for version in range(5):
            document.write_text(f"NPRR 501 version {version}.", encoding="utf-8")
            pipeline.update()

        current = self.active()
        generation_root = self.index_dir / "generations"
        generations = sorted(
            candidate.name for candidate in generation_root.iterdir() if not candidate.name.startswith(".")
        )
        self.assertEqual(len(generations), 3)
        self.assertIn(current.generation_id, generations)
        self.assertIn(current.manifest["previous_generation"], generations)

    def test_failed_store_pointer_publish_leaves_previous_generation_active(self) -> None:
        document = self.write("PGRR401.txt", "PGRR 401 prior content.")
        first = self.pipeline.update()
        previous_id = first["generation"]
        previous = self.active()
        document.write_text("PGRR 401 replacement content.", encoding="utf-8")
        real_atomic_text = store._atomic_text

        def fail_only_current(path: Path, value: str) -> None:
            if path.name == store.CURRENT_FILE:
                raise OSError("synthetic CURRENT publish failure")
            real_atomic_text(path, value)

        with mock.patch.object(store, "_atomic_text", side_effect=fail_only_current):
            with self.assertRaisesRegex(OSError, "synthetic CURRENT publish failure"):
                self.pipeline.update()

        self.assertEqual(current_generation_id(self.index_dir), previous_id)
        current = self.active()
        self.assertEqual([chunk["text"] for chunk in current.chunks], [chunk["text"] for chunk in previous.chunks])
        np.testing.assert_array_equal(current.embeddings, previous.embeddings)


if __name__ == "__main__":
    unittest.main()
