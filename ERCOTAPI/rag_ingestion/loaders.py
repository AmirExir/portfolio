"""Safe, side-effect-free document text loaders."""

from __future__ import annotations

import csv
import io
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET

from .chunking import normalize_text
from .config import SUPPORTED_EXTENSIONS


MAX_ARCHIVE_EXPANDED_BYTES = 250 * 1024 * 1024
MAX_TABULAR_ROWS = 10_000
MAX_TABULAR_COLUMNS = 200

SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


class DocumentLoadError(ValueError):
    """Raised when a supported document cannot be parsed safely."""


class UnsupportedDocumentError(DocumentLoadError):
    """Raised for a file type outside the configured ingestion contract."""


class _VisibleHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style", "noscript", "svg"}:
            self._hidden_depth += 1
        elif lowered in {"p", "div", "section", "article", "br", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style", "noscript", "svg"}:
            self._hidden_depth = max(0, self._hidden_depth - 1)
        elif lowered in {"p", "div", "section", "article", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._hidden_depth == 0:
            self.parts.append(data)


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _validate_archive(archive: zipfile.ZipFile) -> None:
    total = sum(max(0, item.file_size) for item in archive.infolist())
    if total > MAX_ARCHIVE_EXPANDED_BYTES:
        raise DocumentLoadError(
            f"Archive expands to {total} bytes; limit is {MAX_ARCHIVE_EXPANDED_BYTES}"
        )
    for item in archive.infolist():
        member = PurePosixPath(item.filename)
        if member.is_absolute() or ".." in member.parts:
            raise DocumentLoadError(f"Unsafe archive member: {item.filename}")


def _load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise DocumentLoadError("PDF ingestion requires the `pypdf` package") from exc

    try:
        reader = PdfReader(str(path))
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            text = normalize_text(page.extract_text() or "")
            if text:
                pages.append(f"[Page {number}]\n{text}")
    except Exception as exc:
        raise DocumentLoadError(f"Unable to parse PDF {path.name}: {exc}") from exc
    return "\n\n".join(pages)


def _load_html(content: bytes) -> str:
    parser = _VisibleHTMLParser()
    try:
        parser.feed(_decode_text(content))
        parser.close()
    except Exception as exc:
        raise DocumentLoadError(f"Unable to parse HTML: {exc}") from exc
    return " ".join(parser.parts)


def _load_docx(content: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            _validate_archive(archive)
            document_xml = archive.read("word/document.xml")
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise DocumentLoadError(f"Unable to open DOCX: {exc}") from exc

    try:
        root = ET.fromstring(document_xml)
    except ET.ParseError as exc:
        raise DocumentLoadError(f"Invalid DOCX XML: {exc}") from exc
    namespace = {"w": WORD_NS}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        fragments = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        line = "".join(fragments).strip()
        if line:
            paragraphs.append(line)
    return "\n\n".join(paragraphs)


def _load_csv(content: bytes) -> str:
    decoded = _decode_text(content)
    try:
        dialect = csv.Sniffer().sniff(decoded[:8192], delimiters=",\t;|")
    except csv.Error:
        dialect = csv.excel
    rows: list[list[str]] = []
    try:
        for index, row in enumerate(csv.reader(io.StringIO(decoded), dialect)):
            if index >= MAX_TABULAR_ROWS:
                rows.append([f"[truncated after {MAX_TABULAR_ROWS} rows]"])
                break
            rows.append([normalize_text(value) for value in row[:MAX_TABULAR_COLUMNS]])
    except csv.Error as exc:
        raise DocumentLoadError(f"Unable to parse CSV: {exc}") from exc
    if not rows:
        return ""
    header = rows[0]
    lines = ["Columns: " + " | ".join(header)]
    for row_number, row in enumerate(rows[1:], start=1):
        values = []
        for column, value in zip(header, row):
            if value:
                values.append(f"{column or 'column'}={value}")
        if values:
            lines.append(f"Row {row_number}: " + "; ".join(values))
    return "\n".join(lines)


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    except ET.ParseError as exc:
        raise DocumentLoadError(f"Invalid XLSX shared strings: {exc}") from exc
    strings: list[str] = []
    for item in root.findall(f"{{{SPREADSHEET_NS}}}si"):
        value = "".join(node.text or "" for node in item.iter(f"{{{SPREADSHEET_NS}}}t"))
        strings.append(normalize_text(value))
    return strings


def _xlsx_sheets(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    try:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    except (KeyError, ET.ParseError) as exc:
        raise DocumentLoadError(f"Invalid XLSX workbook: {exc}") from exc

    target_by_id = {
        relationship.attrib.get("Id", ""): relationship.attrib.get("Target", "")
        for relationship in relationships.findall(f"{{{PACKAGE_REL_NS}}}Relationship")
    }
    sheets: list[tuple[str, str]] = []
    for sheet in workbook.iter(f"{{{SPREADSHEET_NS}}}sheet"):
        name = sheet.attrib.get("name", "Sheet")
        rel_id = sheet.attrib.get(f"{{{OFFICE_REL_NS}}}id", "")
        target = target_by_id.get(rel_id, "")
        if not target:
            continue
        normalized = PurePosixPath("xl") / PurePosixPath(target)
        parts: list[str] = []
        for part in normalized.parts:
            if part == "..":
                if parts:
                    parts.pop()
            elif part not in {"", "."}:
                parts.append(part)
        sheets.append((name, "/".join(parts)))
    return sheets


def _xlsx_cell_value(cell: ET.Element, shared: list[str]) -> str:
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        return normalize_text("".join(node.text or "" for node in cell.iter(f"{{{SPREADSHEET_NS}}}t")))
    value_node = cell.find(f"{{{SPREADSHEET_NS}}}v")
    value = value_node.text if value_node is not None and value_node.text is not None else ""
    if cell_type == "s" and value:
        try:
            return shared[int(value)]
        except (ValueError, IndexError):
            return value
    if cell_type == "b":
        return "TRUE" if value == "1" else "FALSE"
    return normalize_text(value)


def _load_xlsx(content: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            _validate_archive(archive)
            shared = _shared_strings(archive)
            sheet_specs = _xlsx_sheets(archive)
            sections: list[str] = []
            for sheet_name, member in sheet_specs:
                try:
                    root = ET.fromstring(archive.read(member))
                except (KeyError, ET.ParseError) as exc:
                    raise DocumentLoadError(f"Invalid XLSX sheet {sheet_name}: {exc}") from exc
                lines = [f"[Sheet: {sheet_name}]"]
                for row_number, row in enumerate(root.iter(f"{{{SPREADSHEET_NS}}}row"), start=1):
                    if row_number > MAX_TABULAR_ROWS:
                        lines.append(f"[truncated after {MAX_TABULAR_ROWS} rows]")
                        break
                    values = [
                        _xlsx_cell_value(cell, shared)
                        for cell in list(row.findall(f"{{{SPREADSHEET_NS}}}c"))[:MAX_TABULAR_COLUMNS]
                    ]
                    if any(values):
                        lines.append(f"Row {row.attrib.get('r', row_number)}: " + " | ".join(values))
                sections.append("\n".join(lines))
    except zipfile.BadZipFile as exc:
        raise DocumentLoadError(f"Unable to open XLSX: {exc}") from exc
    return "\n\n".join(sections)


def load_document(path: Path, *, max_file_bytes: int = 100 * 1024 * 1024) -> str:
    """Extract useful text from one supported file without network access."""

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentError(f"Unsupported document extension: {extension or '(none)'}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise DocumentLoadError(f"Unable to stat {path}: {exc}") from exc
    if size > max_file_bytes:
        raise DocumentLoadError(f"File is {size} bytes; configured limit is {max_file_bytes}")
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise DocumentLoadError(f"Unable to read {path}: {exc}") from exc

    if extension == ".pdf":
        text = _load_pdf(path)
    elif extension == ".txt":
        text = _decode_text(content)
    elif extension in {".html", ".htm"}:
        text = _load_html(content)
    elif extension == ".docx":
        text = _load_docx(content)
    elif extension == ".csv":
        text = _load_csv(content)
    elif extension == ".xlsx":
        text = _load_xlsx(content)
    else:  # pragma: no cover - guarded by SUPPORTED_EXTENSIONS
        raise UnsupportedDocumentError(f"Unsupported document extension: {extension}")

    cleaned = normalize_text(text)
    if not cleaned:
        raise DocumentLoadError(f"No useful text could be extracted from {path.name}")
    return cleaned
