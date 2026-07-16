"""Deterministic text chunking and content-addressed identifiers."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping


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


def split_text(text: str, *, chunk_size: int = 5_000, overlap: int = 500) -> list[str]:
    """Split text near paragraph/word boundaries with deterministic overlap."""

    normalized = normalize_text(text)
    if not normalized:
        return []
    if chunk_size < 100:
        raise ValueError("chunk_size must be at least 100 characters")
    overlap = min(max(0, overlap), chunk_size // 2)

    chunks: list[str] = []
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
        piece = normalized[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        next_start = max(start + 1, end - overlap)
        while next_start < end and not normalized[next_start].isspace():
            next_start += 1
        start = min(next_start + 1, end) if next_start < end else next_start
    return chunks


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
    records: list[dict[str, Any]] = []
    for index, piece in enumerate(split_text(text, chunk_size=chunk_size, overlap=overlap)):
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
        records.append(record)
    return records
