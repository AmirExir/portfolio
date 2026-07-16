"""Offline document-loader coverage for every supported dispatch path."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from ERCOTAPI.rag_ingestion.loaders import (
    DocumentLoadError,
    UnsupportedDocumentError,
    load_document,
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

        loader.assert_called_once_with(path)
        self.assertEqual(loaded, "[Page 1]\nOfficial protocol text")

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

