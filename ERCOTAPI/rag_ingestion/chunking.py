"""Deterministic text chunking and content-addressed identifiers."""

from __future__ import annotations

import hashlib
import re
from bisect import bisect_left, bisect_right
from typing import Any, Mapping


_PAGE_MARKER_RE = re.compile(
    r"^[ \t]*\[Page[ \t]+(?P<number>\d+)\][ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_SECTION_NUMBER_PATTERN = (
    r"\d+[A-Za-z]?(?:\.\d+[A-Za-z]?)*(?:\([A-Za-z0-9]+\))*"
)
_EXPLICIT_SECTION_RE = re.compile(
    rf"^[ \t]*(?:Section|§)[ \t]+(?P<number>{_SECTION_NUMBER_PATTERN})"
    r"(?P<title>[^\n]*)$",
    re.IGNORECASE | re.MULTILINE,
)
_DECIMAL_SECTION_RE = re.compile(
    rf"^[ \t]*(?P<number>\d+[A-Za-z]?(?:\.\d+[A-Za-z]?)+"
    r"(?:\([A-Za-z0-9]+\))*)"
    r"(?:[ \t]+|[ \t]*[:.\-–—][ \t]*)(?P<title>[^\n]+)$",
    re.MULTILINE,
)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def document_id(content_hash: str) -> str:
    """Return a rename-stable ID based solely on exact file content."""

    return hashlib.sha256(f"ercot-document\0{content_hash}".encode("ascii")).hexdigest()


def chunk_id(doc_id: str, index: int, text: str) -> str:
    """Return a deterministic ID stable for an unchanged document."""

    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    value = f"ercot-chunk\0{doc_id}\0{index}\0{text_hash}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def _split_text_spans(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
) -> tuple[str, list[tuple[str, int, int]]]:
    """Return unchanged chunk text plus its offsets in normalized source text."""

    normalized = normalize_text(text)
    if not normalized:
        return normalized, []
    if chunk_size < 100:
        raise ValueError("chunk_size must be at least 100 characters")
    overlap = min(max(0, overlap), chunk_size // 2)

    chunks: list[tuple[str, int, int]] = []
    start = 0
    length = len(normalized)
    while start < length:
        target = min(length, start + chunk_size)
        end = target
        if target < length:
            lower_bound = start + max(100, chunk_size // 2)
            paragraph = normalized.rfind("\n\n", lower_bound, target)
            newline = normalized.rfind("\n", lower_bound, target)
            space = normalized.rfind(" ", lower_bound, target)
            end = max(paragraph, newline, space)
            if end < lower_bound:
                end = target
        raw_piece = normalized[start:end]
        piece = raw_piece.strip()
        if piece:
            leading = len(raw_piece) - len(raw_piece.lstrip())
            trailing = len(raw_piece) - len(raw_piece.rstrip())
            piece_start = start + leading
            piece_end = end - trailing
            chunks.append((piece, piece_start, piece_end))
        if end >= length:
            break
        next_start = max(start + 1, end - overlap)
        while next_start < end and not normalized[next_start].isspace():
            next_start += 1
        start = min(next_start + 1, end) if next_start < end else next_start
    return normalized, chunks


def split_text(text: str, *, chunk_size: int = 5_000, overlap: int = 500) -> list[str]:
    """Split text near paragraph/word boundaries with deterministic overlap."""

    _, spans = _split_text_spans(text, chunk_size=chunk_size, overlap=overlap)
    return [piece for piece, _, _ in spans]


def _clean_section_title(value: str) -> str | None:
    title = " ".join(value.strip().lstrip(":.-–—").split())
    return title or None


def _section_events(text: str) -> list[tuple[int, str, str | None]]:
    """Find conservative, line-oriented section headings in normalized text."""

    events: dict[int, tuple[int, str, str | None]] = {}
    for pattern in (_EXPLICIT_SECTION_RE, _DECIMAL_SECTION_RE):
        for match in pattern.finditer(text):
            events.setdefault(
                match.start(),
                (
                    match.start(),
                    match.group("number"),
                    _clean_section_title(match.group("title")),
                ),
            )
    return [events[position] for position in sorted(events)]


def _event_slice(
    positions: list[int],
    start: int,
    end: int,
) -> tuple[int, int]:
    return bisect_left(positions, start), bisect_left(positions, end)


def _locator_metadata(
    *,
    start: int,
    end: int,
    page_events: list[tuple[int, int]],
    section_events: list[tuple[int, str, str | None]],
) -> dict[str, Any]:
    """Return only locators that are explicitly observable from source text."""

    locators: dict[str, Any] = {}

    page_positions = [position for position, _ in page_events]
    page_left, page_right = _event_slice(page_positions, start, end)
    page_at_start = bisect_right(page_positions, start) - 1
    if page_at_start >= 0:
        page_start = page_events[page_at_start][1]
    elif page_left < page_right:
        page_start = page_events[page_left][1]
    else:
        page_start = None
    if page_start is not None:
        page_end = (
            page_events[page_right - 1][1]
            if page_left < page_right
            else page_start
        )
        locators.update({"page_start": page_start, "page_end": page_end})

    section_positions = [position for position, _, _ in section_events]
    section_left, section_right = _event_slice(section_positions, start, end)
    if section_left < section_right:
        _, section_number, section_title = section_events[section_left]
    else:
        section_at_start = bisect_right(section_positions, start) - 1
        if section_at_start < 0:
            return locators
        _, section_number, section_title = section_events[section_at_start]
    locators["section_number"] = section_number
    if section_title:
        locators["section_title"] = section_title
    return locators


def build_chunks(
    text: str,
    *,
    content_hash: str,
    metadata: Mapping[str, Any],
    chunk_size: int,
    overlap: int,
) -> list[dict[str, Any]]:
    """Build serializable chunk records for one exact document content."""

    doc_id = document_id(content_hash)
    normalized, spans = _split_text_spans(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    page_events = [
        (match.start(), int(match.group("number")))
        for match in _PAGE_MARKER_RE.finditer(normalized)
    ]
    section_events = _section_events(normalized)
    records: list[dict[str, Any]] = []
    for index, (piece, start, end) in enumerate(spans):
        identifier = chunk_id(doc_id, index, piece)
        record = dict(metadata)
        record.update(
            {
                "id": identifier,
                "chunk_id": identifier,
                "document_id": doc_id,
                "content_hash": content_hash,
                "chunk_index": index,
                "text": piece,
            }
        )
        record.update(
            _locator_metadata(
                start=start,
                end=end,
                page_events=page_events,
                section_events=section_events,
            )
        )
        records.append(record)
    return records
