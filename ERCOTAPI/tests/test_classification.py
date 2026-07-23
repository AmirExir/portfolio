"""Classification, routing, trust metadata, and identifier tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ERCOTAPI.rag_ingestion.chunking import build_chunks, sha256_bytes
from ERCOTAPI.rag_ingestion.classify import (
    authority_rank,
    classify_document,
    enrich_metadata_from_text,
)
from ERCOTAPI.rag_ingestion.config import SourceRoot, default_config


class ClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repo_root = Path(temporary.name)
        self.official_path = self.repo_root / "official"
        self.generated_path = self.repo_root / "generated"
        self.official_path.mkdir()
        self.generated_path.mkdir()
        self.official = SourceRoot(
            name="official_downloads",
            path=self.official_path,
            source_authority="ERCOT",
            is_generated=False,
            default_source_kind="Official Document",
            default_collections=("general",),
        )
        self.generated = SourceRoot(
            name="generated_news",
            path=self.generated_path,
            source_authority="Generated",
            is_generated=True,
            default_source_kind="News Summary",
            default_collections=("news", "market"),
        )

    def test_supported_categories_and_collection_routing(self) -> None:
        cases = (
            ("NPRR1234_approved.txt", "NPRR", "NPRR1234", {"general", "protocols"}),
            ("PGRR567_clean.txt", "PGRR", "PGRR567", {"general", "planning"}),
            ("NOGRR890_effective.txt", "NOGRR", "NOGRR890", {"general", "operations"}),
            ("OBDRR432_redline.txt", "OBDRR", "OBDRR432", {"general", "protocols"}),
            ("SCR222.txt", "System Change Request", "SCR222", {"general"}),
            (
                "RRGRR333.txt",
                "Resource Registration Glossary Revision Request",
                "RRGRR333",
                {"general"},
            ),
            (
                "VCMRR444.txt",
                "Verifiable Cost Manual Revision Request",
                "VCMRR444",
                {"general"},
            ),
            (
                "COPMGRR053_approved.txt",
                "Commercial Operations Market Guide Revision Request",
                "COPMGRR053",
                {"general", "market"},
            ),
            ("LPGRR076_pending.txt", "Load Profiling Guide Revision Request", "LPGRR076", {"general", "market"}),
            ("RMGRR186_pending.txt", "Retail Market Guide Revision Request", "RMGRR186", {"general", "market"}),
            ("SMOGRR031_pending.txt", "Settlement Metering Operating Guide Revision Request", "SMOGRR031", {"general", "market"}),
            ("CMGRR010_approved.txt", "Competitive Metering Guide Revision Request", "CMGRR010", {"general", "market"}),
            ("ercot_nodal_protocols.txt", "Protocol", None, {"general", "protocols"}),
            ("planning_guide_2026.txt", "Planning Guide", None, {"general", "planning"}),
            ("nodal_operating_guide.txt", "Operating Guide", None, {"general", "operations"}),
            (
                "resource_integration_handbook.txt",
                "Resource Integration",
                None,
                {"general", "resource_integration"},
            ),
            ("SSWG_procedure_manual.txt", "SSWG", None, {"general", "dwg_sswg"}),
            ("DWG_procedure_manual.txt", "DWG", None, {"general", "dwg_sswg"}),
            ("market_notice_W-A123.txt", "Market Notice", None, {"general", "market"}),
            ("summer_assessment_report.txt", "ERCOT Report", None, {"general", "market"}),
        )

        for filename, expected_kind, expected_number, expected_collections in cases:
            with self.subTest(filename=filename):
                metadata = classify_document(
                    self.official_path / filename,
                    self.official,
                    self.repo_root,
                )
                self.assertEqual(metadata["source_kind"], expected_kind)
                self.assertEqual(metadata["document_number"], expected_number)
                self.assertTrue(expected_collections.issubset(metadata["collections"]))

    def test_directory_trust_cannot_be_overridden_by_sidecar(self) -> None:
        official = classify_document(
            self.official_path / "NPRR100.txt",
            self.official,
            self.repo_root,
            sidecar={
                "source_authority": "Generated",
                "is_generated": True,
                "title": "Official NPRR",
            },
        )
        generated = classify_document(
            self.generated_path / "NPRR100_summary.txt",
            self.generated,
            self.repo_root,
            sidecar={
                "source_authority": "ERCOT",
                "is_generated": False,
                "title": "Generated NPRR summary",
            },
        )

        self.assertEqual(official["source_authority"], "ERCOT")
        self.assertFalse(official["is_generated"])
        self.assertEqual(authority_rank(official), 2)
        self.assertEqual(generated["source_authority"], "Generated")
        self.assertTrue(generated["is_generated"])
        self.assertEqual(authority_rank(generated), 0)
        self.assertNotIn("general", generated["collections"])
        self.assertEqual(set(generated["collections"]), {"market", "news"})

        generated_revision = classify_document(
            self.generated_path / "NPRR-1234-generated-summary.txt",
            self.generated,
            self.repo_root,
        )
        self.assertEqual(generated_revision["source_kind"], "NPRR")
        self.assertEqual(set(generated_revision["collections"]), {"market", "news"})
        self.assertNotIn("protocols", generated_revision["collections"])

    def test_default_roots_exclude_generated_content_and_old_split_corpora(self) -> None:
        config = default_config(
            repo_root=self.repo_root,
            index_dir=self.repo_root / ".rag_store",
        )
        root_names = {root.name for root in config.source_roots}

        self.assertEqual(
            root_names,
            {
                "authoritative_static",
                "official_downloads",
                "planning_guide_uploads",
                "nodal_protocol_uploads",
                "dwg_sswg_uploads",
                "market_document_uploads",
            },
        )
        self.assertIn("requirements.txt", config.ignored_names)

    def test_default_config_uses_packaged_store_when_local_store_is_absent(self) -> None:
        packaged = self.repo_root / "ERCOTAPI" / "deployment_rag_store"
        generation = packaged / "generations" / "packaged"
        generation.mkdir(parents=True)
        (packaged / "CURRENT").write_text("packaged\n", encoding="utf-8")
        for filename in ("manifest.json", "chunks.json.gz", "embeddings.npy.gz"):
            (generation / filename).write_bytes(b"saved")

        config = default_config(repo_root=self.repo_root)

        self.assertEqual(config.index_dir, packaged.resolve())

    def test_relative_configured_store_resolves_from_repository_root(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {"ERCOT_RAG_STORE": "mounted/saved-index"},
            clear=False,
        ):
            config = default_config(repo_root=self.repo_root)

        self.assertEqual(
            config.index_dir,
            (self.repo_root / "mounted" / "saved-index").resolve(),
        )

    def test_uploaded_official_roots_route_to_explicit_collections(self) -> None:
        config = default_config(
            repo_root=self.repo_root,
            index_dir=self.repo_root / ".rag_store",
        )
        cases = (
            ("planning_guide_uploads", "02-060126.docx", "Planning Guide", "planning"),
            ("nodal_protocol_uploads", "01-020126_Nodal.docx", "Protocol", "protocols"),
            ("dwg_sswg_uploads", "DWG-Procedure-Manual.docx", "DWG", "dwg_sswg"),
            ("market_document_uploads", "ERCOT-Fee-Schedule.docx", "Fee Schedule", "market"),
        )
        for root_name, filename, expected_kind, expected_collection in cases:
            with self.subTest(root=root_name):
                source_root = next(root for root in config.source_roots if root.name == root_name)
                metadata = classify_document(
                    source_root.path / filename,
                    source_root,
                    self.repo_root,
                )
                self.assertEqual(metadata["source_kind"], expected_kind)
                self.assertIn("general", metadata["collections"])
                self.assertIn(expected_collection, metadata["collections"])

    def test_extracted_guide_heading_adds_version_metadata_without_overrides(self) -> None:
        metadata = {
            "filename": "03-071126_Nodal.docx",
            "title": "03 071126 Nodal",
            "source_kind": "Protocol",
            "effective_date": None,
            "published_date": None,
            "document_status": None,
            "revision": None,
        }
        enriched = enrich_metadata_from_text(
            metadata,
            "ERCOT Nodal Protocols\n\nSection 3: Management\n\nJuly 11, 2026\n",
        )

        self.assertEqual(enriched["title"], "ERCOT Nodal Protocols — Section 3: Management")
        self.assertEqual(enriched["effective_date"], "2026-07-11")

        approved = enrich_metadata_from_text(
            {
                **metadata,
                "filename": "DWG-Procedure-Manual.docx",
                "title": "DWG Procedure Manual",
                "source_kind": "DWG",
            },
            "Dynamics Working Group\nProcedure Manual\nRevision 25\nROS Approved: July 9, 2026",
        )
        self.assertEqual(approved["published_date"], "2026-07-09")
        self.assertEqual(approved["document_status"], "Approved")
        self.assertEqual(approved["revision"], "25")

    def test_sidecar_metadata_takes_precedence_over_filename_fallback(self) -> None:
        metadata = classify_document(
            self.official_path / "ambiguous_document.txt",
            self.official,
            self.repo_root,
            sidecar={
                "title": "Planning Change",
                "source_kind": "PGRR",
                "document_number": "PGRR2020",
                "document_status": "Board Approved",
                "revision": "3",
                "original_url": "https://www.ercot.com/example",
                "downloaded_at": "2026-07-16T10:00:00Z",
                "effective_date": "2026-08-01",
            },
        )

        self.assertEqual(metadata["source_kind"], "PGRR")
        self.assertEqual(metadata["document_number"], "PGRR2020")
        self.assertEqual(metadata["document_status"], "Board Approved")
        self.assertEqual(metadata["revision"], "3")
        self.assertEqual(metadata["original_url"], "https://www.ercot.com/example")
        self.assertIn("planning", metadata["collections"])

    def test_plural_market_notice_source_label_routes_to_market(self) -> None:
        metadata = classify_document(
            self.official_path / "notice.txt",
            self.official,
            self.repo_root,
            sidecar={"source_kind": "MARKET NOTICES", "title": "Operations notice"},
        )

        self.assertIn("general", metadata["collections"])
        self.assertIn("market", metadata["collections"])

    def test_generic_committee_kind_still_routes_numbered_revision_requests(self) -> None:
        cases = (
            ("TAC_cross_post.txt", "NPRR1234 approval document", {"protocols", "market"}),
            ("RPG_cross_post.txt", "PGRR567 planning change", {"planning"}),
        )
        for filename, title, expected in cases:
            with self.subTest(filename=filename):
                metadata = classify_document(
                    self.official_path / filename,
                    self.official,
                    self.repo_root,
                    sidecar={"title": title, "source_kind": "TAC"},
                )
                self.assertEqual(metadata["source_kind"], "TAC")
                self.assertTrue(expected.issubset(set(metadata["collections"])))

    def test_explicit_sidecar_number_routes_hash_named_committee_document(self) -> None:
        metadata = classify_document(
            self.official_path / ("a" * 64 + ".pdf"),
            self.official,
            self.repo_root,
            sidecar={
                "title": "Approval document",
                "source_kind": "TAC",
                "document_number": "NPRR1234",
            },
        )

        self.assertEqual(metadata["source_kind"], "TAC")
        self.assertEqual(metadata["document_number"], "NPRR1234")
        self.assertTrue({"general", "protocols", "market"}.issubset(metadata["collections"]))

    def test_reverse_xrr_attachment_name_gets_family_document_number(self) -> None:
        metadata = classify_document(
            self.official_path / "145PGRR-73-AEPSC-Comments.docx",
            self.official,
            self.repo_root,
            sidecar={
                "source_kind": "PGRR",
                "source_page_url": "https://www.ercot.com/mktrules/issues/PGRR145",
            },
        )

        self.assertEqual(metadata["document_number"], "PGRR145")
        self.assertIn("planning", metadata["collections"])

    def test_official_source_aliases_route_generic_hash_named_documents(self) -> None:
        cases = (
            ("RPG", "planning"),
            ("RTP", "planning"),
            ("RIWG", "resource_integration"),
        )
        for source_kind, expected_collection in cases:
            with self.subTest(source_kind=source_kind):
                metadata = classify_document(
                    self.official_path / ("b" * 64 + ".pdf"),
                    self.official,
                    self.repo_root,
                    sidecar={"title": "Meeting attachment", "source_kind": source_kind},
                )

                self.assertEqual(metadata["source_kind"], source_kind)
                self.assertIn("general", metadata["collections"])
                self.assertIn(expected_collection, metadata["collections"])

    def test_document_and_chunk_ids_are_stable_when_only_metadata_changes(self) -> None:
        text = "Stable ERCOT requirement text. " * 20
        content_hash = sha256_bytes(text.encode("utf-8"))
        first = build_chunks(
            text,
            content_hash=content_hash,
            metadata={"source_path": "old/NPRR1234.txt", "title": "Old title"},
            chunk_size=200,
            overlap=20,
        )
        second = build_chunks(
            text,
            content_hash=content_hash,
            metadata={"source_path": "renamed/NPRR1234.txt", "title": "New title"},
            chunk_size=200,
            overlap=20,
        )

        self.assertEqual(
            [chunk["document_id"] for chunk in first],
            [chunk["document_id"] for chunk in second],
        )
        self.assertEqual(
            [chunk["chunk_id"] for chunk in first],
            [chunk["chunk_id"] for chunk in second],
        )


if __name__ == "__main__":
    unittest.main()
