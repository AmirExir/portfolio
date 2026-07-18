"""Smoke checks for the checked-in authoritative ERCOT source bundle."""

from __future__ import annotations

import unittest
from pathlib import Path

from ERCOTAPI.rag_ingestion.loaders import load_document


class RepositorySourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.source_root = cls.repo_root / "ERCOTAPI" / "sources" / "official"

    def test_every_legacy_doc_has_a_portable_parseable_docx_counterpart(self) -> None:
        legacy_documents = sorted(self.source_root.rglob("*.doc"))
        self.assertTrue(legacy_documents, "expected the checked-in legacy Word sources")

        for legacy in legacy_documents:
            with self.subTest(source=legacy.name):
                portable = legacy.with_suffix(".docx")
                self.assertTrue(portable.is_file())
                self.assertTrue(load_document(portable).strip())

    def test_current_2026_guides_and_dwg_manual_are_parseable(self) -> None:
        expected = {
            "planning_guides/01-020126.docx": "February 1, 2026",
            "planning_guides/03-071126.docx": "July 11, 2026",
            "nodal_protocols/01-020126_Nodal.docx": "February 1, 2026",
            (
                "dwg_sswg/"
                "DWG-Procedure-Manual-Revision-25-ROS-Approved-7092026.docx"
            ): "Revision 25",
        }

        for relative, marker in expected.items():
            with self.subTest(source=relative):
                text = load_document(self.source_root / relative)
                self.assertIn(marker, text)


if __name__ == "__main__":
    unittest.main()
