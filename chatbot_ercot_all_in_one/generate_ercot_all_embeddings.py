"""Offline generator for the inactive legacy all-in-one ERCOT cache.

The deployed assistants use the versioned central index.  This script is kept
to reproduce the historical cache, and requires ``--force`` before it can make
paid full-corpus embedding requests.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "ercot_sources"
CHUNK_OUTPUT_FILE = BASE_DIR / "ercot_chunks_cached.json"
EMBEDDING_OUTPUT_FILE = BASE_DIR / "ercot_embeddings.npy"
CHUNK_SIZE = 7_600
CHUNK_OVERLAP = 400
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSION = 3072


def split_long_paragraph(text: str, max_size: int) -> list[str]:
    if len(text) <= max_size:
        return [text]

    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_size)
        if end < len(text):
            boundary = max(
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
                text.rfind("; ", start, end),
            )
            if boundary > start + (max_size // 2):
                end = boundary + 1
        part = text[start:end].strip()
        if part:
            parts.append(part)
        start = end
    return parts


def chunk_paragraphs(text: str, chunk_size: int, overlap: int) -> list[str]:
    current_chunk = ""
    result_chunks: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for piece in split_long_paragraph(paragraph, chunk_size):
            if len(current_chunk) + len(piece) + 2 <= chunk_size:
                current_chunk += piece + "\n\n"
            else:
                if current_chunk.strip():
                    result_chunks.append(current_chunk.strip())
                current_chunk = piece + "\n\n"
    if current_chunk.strip():
        result_chunks.append(current_chunk.strip())

    final_chunks = []
    for index, chunk in enumerate(result_chunks):
        overlap_text = result_chunks[index - 1][-overlap:] if index > 0 else ""
        final_chunks.append((overlap_text + "\n" + chunk).strip())
    return final_chunks


def build_chunks() -> tuple[list[dict], int]:
    chunks: list[dict] = []
    source_files = sorted(SOURCE_DIR.glob("*.txt"))
    for filepath in source_files:
        text = filepath.read_text(encoding="utf-8")
        for index, chunk_text in enumerate(
            chunk_paragraphs(text, CHUNK_SIZE, CHUNK_OVERLAP)
        ):
            chunks.append(
                {
                    "text": chunk_text,
                    "source": filepath.name,
                    "chunk_index": index,
                }
            )
    return chunks, len(source_files)


def validate_chunks(chunks: list[dict]) -> None:
    oversized = [
        (index, chunk["source"], chunk["chunk_index"], len(chunk["text"]))
        for index, chunk in enumerate(chunks)
        if len(chunk["text"]) > 8_192
    ]
    if oversized:
        preview = ", ".join(str(item) for item in oversized[:5])
        raise RuntimeError(f"Found chunks over 8192 characters after splitting: {preview}")


def saved_cache_is_current(chunks: list[dict]) -> bool:
    try:
        saved_chunks = json.loads(CHUNK_OUTPUT_FILE.read_text(encoding="utf-8"))
        embeddings = np.load(
            EMBEDDING_OUTPUT_FILE,
            mmap_mode="r",
            allow_pickle=False,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        saved_chunks == chunks
        and embeddings.ndim == 2
        and embeddings.shape == (len(chunks), EMBEDDING_DIMENSION)
    )


def require_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key in {"your-key", "your-key-here"}:
        raise SystemExit("OPENAI_API_KEY is required only for an explicit --force rebuild.")
    return api_key


def create_embedding(client, text: str, max_retries: int = 5) -> list[float]:
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
            if not response.data:
                raise RuntimeError("OpenAI returned no embedding data")
            return response.data[0].embedding
        except Exception as exc:
            if getattr(exc, "status_code", None) in {401, 403}:
                raise RuntimeError("OpenAI authentication failed; not retrying.") from None
            if attempt + 1 == max_retries:
                raise RuntimeError(
                    f"Embedding failed after {max_retries} attempts: {type(exc).__name__}"
                ) from exc
            wait_time = 2**attempt
            print(f"Embedding request failed; retrying in {wait_time} seconds.")
            time.sleep(wait_time)
    raise AssertionError("unreachable")


def write_cache(chunks: list[dict], embeddings: list[list[float]]) -> None:
    if len(chunks) != len(embeddings):
        raise RuntimeError(
            f"Refusing a misaligned cache: {len(chunks)} chunks, {len(embeddings)} vectors"
        )
    chunk_tmp = CHUNK_OUTPUT_FILE.with_suffix(".json.tmp")
    embedding_tmp = EMBEDDING_OUTPUT_FILE.with_suffix(".npy.tmp")
    chunk_tmp.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    with embedding_tmp.open("wb") as handle:
        np.save(handle, np.asarray(embeddings, dtype=np.float32))
    chunk_tmp.replace(CHUNK_OUTPUT_FILE)
    embedding_tmp.replace(EMBEDDING_OUTPUT_FILE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate or explicitly regenerate the legacy ERCOT RAG cache."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate chunking without calling OpenAI or writing files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Acknowledge the paid full-corpus OpenAI embedding rebuild.",
    )
    args = parser.parse_args(argv)

    chunks, source_count = build_chunks()
    validate_chunks(chunks)
    max_chunk_len = max((len(chunk["text"]) for chunk in chunks), default=0)
    section_9_chunks = [
        index
        for index, chunk in enumerate(chunks)
        if chunk["source"] == "ercotaiassistant.txt"
        and "LARGE LOAD ADDITIONS AT NEW OR MODIFICATION" in chunk["text"]
    ]
    print(f"Loaded and chunked {len(chunks)} chunks from {source_count} files.")
    print(f"Max chunk length: {max_chunk_len} characters")
    print(f"ERCOT Planning Guide Section 9 chunks: {section_9_chunks}")

    if args.dry_run:
        print("Dry run complete. No files were written and no API calls were made.")
        return 0
    if saved_cache_is_current(chunks) and not args.force:
        print("Saved legacy cache is current; no API calls were made.")
        return 0
    if not args.force:
        print("Saved legacy cache is missing or stale; rerun with --force to rebuild offline.")
        return 2

    from openai import OpenAI

    client = OpenAI(api_key=require_api_key())
    embeddings = []
    for index, chunk in enumerate(chunks, 1):
        print(f"Embedding chunk {index}/{len(chunks)}")
        embeddings.append(create_embedding(client, chunk["text"]))
    write_cache(chunks, embeddings)
    print(f"Saved {len(embeddings)} aligned embeddings to {EMBEDDING_OUTPUT_FILE}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
