"""Offline document-loader coverage for every supported dispatch path."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

from ERCOTAPI.rag_ingestion.loaders import (
    DocumentLoadError,
    UnsupportedDocumentError,
    load_document,
    load_document_with_report,
)


class DocumentLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write(self, name: str, content: str | bytes) -> Path:
        path = self.root / name
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        return path

    def test_txt_html_and_htm_loaders_extract_visible_text(self) -> None:
        text_path = self.write("notice.txt", "ERCOT\r\n\x00market notice")
        html = (
            "<html><style>hidden css</style><script>hidden script</script>"
            "<h1>ERCOT Notice</h1><p>Visible requirement</p></html>"
        )
        html_path = self.write("notice.html", html)
        htm_path = self.write("notice.htm", html)

        self.assertEqual(load_document(text_path), "ERCOT\n market notice")
        for path in (html_path, htm_path):
            with self.subTest(path=path.name):
                loaded = load_document(path)
                self.assertIn("ERCOT Notice", loaded)
                self.assertIn("Visible requirement", loaded)
                self.assertNotIn("hidden css", loaded)
                self.assertNotIn("hidden script", loaded)

    def test_docx_loader_reads_paragraph_text(self) -> None:
        path = self.root / "NPRR1234.docx"
        document_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>NPRR 1234 heading</w:t></w:r></w:p>
    <w:p><w:r><w:t>Official requirement text</w:t></w:r></w:p>
  </w:body>
</w:document>
"""
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("word/document.xml", document_xml)

        loaded = load_document(path)

        self.assertIn("NPRR 1234 heading", loaded)
        self.assertIn("Official requirement text", loaded)

    def test_csv_loader_preserves_headers_and_structured_rows(self) -> None:
        path = self.write("market.csv", "resource,mw,status\nSolar,100,Available\nWind,200,Outage\n")

        loaded = load_document(path)

        self.assertIn("Columns: resource | mw | status", loaded)
        self.assertIn("resource=Solar", loaded)
        self.assertIn("mw=100", loaded)
        self.assertIn("status=Outage", loaded)

    def test_xlsx_loader_reads_inline_string_and_numeric_cells(self) -> None:
        path = self.root / "market.xlsx"
        workbook = """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="ERCOT Data" sheetId="1" r:id="rId1"/></sheets>
</workbook>
"""
        relationships = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>
"""
        worksheet = """<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1"><c t="inlineStr"><is><t>Resource</t></is></c><c t="inlineStr"><is><t>MW</t></is></c></row>
    <row r="2"><c t="inlineStr"><is><t>Solar</t></is></c><c><v>125</v></c></row>
  </sheetData>
</worksheet>
"""
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("xl/workbook.xml", workbook)
            archive.writestr("xl/_rels/workbook.xml.rels", relationships)
            archive.writestr("xl/worksheets/sheet1.xml", worksheet)

        loaded = load_document(path)

        self.assertIn("[Sheet: ERCOT Data]", loaded)
        self.assertIn("Row 1: Resource | MW", loaded)
        self.assertIn("Row 2: Solar | 125", loaded)

    def test_pdf_dispatch_is_lazy_and_uses_pdf_loader(self) -> None:
        path = self.write("protocol.pdf", b"synthetic PDF fixture")

        with mock.patch(
            "ERCOTAPI.rag_ingestion.loaders._load_pdf",
            return_value="[Page 1]\nOfficial protocol text",
        ) as loader:
            loaded = load_document(path)

        loader.assert_called_once_with(path, report=mock.ANY)
        self.assertEqual(loaded, "[Page 1]\nOfficial protocol text")

    def test_partial_pdf_reports_unextracted_pages_without_inventing_text(self) -> None:
        path = self.write("partial.pdf", b"synthetic PDF fixture")
        pages = [
            SimpleNamespace(extract_text=lambda: "First page text"),
            SimpleNamespace(extract_text=lambda: ""),
            SimpleNamespace(extract_text=lambda: "Third page text"),
        ]
        with mock.patch.dict("sys.modules", {"pypdf": SimpleNamespace(PdfReader=lambda _: SimpleNamespace(pages=pages))}):
            loaded = load_document_with_report(path)
        self.assertEqual(loaded.extraction_report["pages_without_text"], [2])
        self.assertEqual(loaded.extraction_report["pages_with_text"], 2)
        self.assertEqual(loaded.extraction_report["total_pages"], 3)
        self.assertAlmostEqual(loaded.extraction_report["text_page_coverage"], 2 / 3)
        self.assertTrue(loaded.extraction_report["review_required"])
        self.assertNotIn("[Page 2]", loaded.text)
        self.assertIn("[Page 3]", loaded.text)

    def test_wholly_unreadable_pdf_still_fails(self) -> None:
        path = self.write("scan.pdf", b"synthetic PDF fixture")
        reader = SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda: "")])
        with mock.patch.dict("sys.modules", {"pypdf": SimpleNamespace(PdfReader=lambda _: reader)}):
            with self.assertRaisesRegex(DocumentLoadError, "No useful text"):
                load_document_with_report(path)

    def test_docx_keeps_table_cells_merges_and_referenced_footnotes(self) -> None:
        path = self.root / "table.docx"
        namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        document = f'''<w:document xmlns:w="{namespace}"><w:body>
        <w:p><w:r><w:t>Before table</w:t><w:footnoteReference w:id="2"/></w:r></w:p>
        <w:tbl><w:tr><w:tc><w:tcPr><w:gridSpan w:val="2"/></w:tcPr>
        <w:p><w:r><w:t>Voltage (kV)</w:t></w:r></w:p></w:tc></w:tr>
        <w:tr><w:tc><w:p><w:r><w:t>138</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>345</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
        <w:p><w:r><w:t>After table</w:t></w:r></w:p>
        </w:body></w:document>'''
        notes = f'''<w:footnotes xmlns:w="{namespace}"><w:footnote w:id="2">
        <w:p><w:r><w:t>Fixture values only.</w:t></w:r></w:p></w:footnote></w:footnotes>'''
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("word/document.xml", document)
            archive.writestr("word/footnotes.xml", notes)
        loaded = load_document_with_report(path)
        self.assertIn("Row 1: C1-C2=Voltage (kV)", loaded.text)
        self.assertIn("Row 2: C1=138 | C2=345", loaded.text)
        self.assertIn("[Footnote 2] Fixture values only.", loaded.text)
        self.assertLess(loaded.text.index("Fixture values only."), loaded.text.index("[Table 1]"))
        self.assertLess(loaded.text.index("Before table"), loaded.text.index("[Table 1]"))
        self.assertLess(loaded.text.index("[Table 1]"), loaded.text.index("After table"))
        self.assertEqual(loaded.extraction_report["table_count"], 1)
        self.assertEqual(loaded.extraction_report["footnotes_extracted"], ["2"])
        self.assertTrue(loaded.extraction_report["review_required"])

    def test_missing_docx_footnote_requires_review(self) -> None:
        path = self.root / "missing-note.docx"
        document = '''<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
        <w:body><w:p><w:r><w:t>Text</w:t><w:footnoteReference w:id="3"/></w:r></w:p></w:body></w:document>'''
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("word/document.xml", document)
        loaded = load_document_with_report(path)
        self.assertEqual(loaded.extraction_report["missing_footnotes"], ["3"])
        self.assertTrue(loaded.extraction_report["review_required"])

    def test_invalid_docx_cell_span_fails_without_silently_changing_geometry(self) -> None:
        path = self.root / "invalid-table.docx"
        document = '''<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
        <w:body><w:tbl><w:tr><w:tc><w:tcPr><w:gridSpan w:val="-2"/></w:tcPr>
        <w:p><w:r><w:t>Value</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>'''
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("word/document.xml", document)
        with self.assertRaisesRegex(DocumentLoadError, "outside its valid range"):
            load_document_with_report(path)

    def test_malformed_supported_file_raises_clear_error(self) -> None:
        path = self.write("broken.docx", b"not a valid archive")

        with self.assertRaisesRegex(DocumentLoadError, "Unable to open DOCX"):
            load_document(path)

    def test_unsupported_extension_and_size_limit_fail_before_parsing(self) -> None:
        unsupported = self.write("archive.zip", b"content")
        oversized = self.write("large.txt", b"12345")

        with self.assertRaisesRegex(UnsupportedDocumentError, "Unsupported document extension: .zip"):
            load_document(unsupported)
        with self.assertRaisesRegex(DocumentLoadError, "configured limit is 4"):
            load_document(oversized, max_file_bytes=4)


if __name__ == "__main__":
    unittest.main()
