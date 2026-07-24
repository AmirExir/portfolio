"""Explicit offline maintenance command for the saved PSS/E index.

Running this file without flags only validates the deployed artifacts. A paid
corpus rebuild requires both ``--rebuild`` and
``--confirm-paid-embedding-request`` so app startup can never trigger it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = str(BASE_DIR.parent)
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from psse_assistant_common import validate_saved_index  # noqa: E402

SOURCE_PATH = BASE_DIR / "input_chunks.json"
CHUNKS_PATH = BASE_DIR / "psse_chunks_cached.json"
EMBEDDINGS_PATH = BASE_DIR / "psse_embeddings.npy"
SINGLE_APP_DIR = BASE_DIR.parent / "chatbot_PSSEBOT"
EMBEDDING_MODEL = "text-embedding-3-large"


def validate_saved_index_command() -> None:
    with SOURCE_PATH.open("r", encoding="utf-8") as source_file:
        source_chunks = list(json.load(source_file))
    dimensions = None
    for app_dir in (BASE_DIR, SINGLE_APP_DIR):
        chunks_path = app_dir / "psse_chunks_cached.json"
        embeddings_path = app_dir / "psse_embeddings.npy"
        with chunks_path.open("r", encoding="utf-8") as chunks_file:
            chunks = list(json.load(chunks_file))
        embeddings = np.load(embeddings_path)
        validate_saved_index(chunks, embeddings, expected_dimension=3072)
        if chunks != source_chunks:
            raise RuntimeError(
                f"The saved PSS/E chunks in {app_dir.name} differ from "
                "input_chunks.json."
            )
        dimensions = embeddings.shape[1]
    print(
        "Both saved PSS/E indexes validated: "
        f"{len(source_chunks)} chunks, {dimensions} dimensions. "
        "No embedding API requests were made."
    )


def rebuild_saved_index(*, batch_size: int = 32) -> None:
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(f"Source chunks not found: {SOURCE_PATH}")
    with SOURCE_PATH.open("r", encoding="utf-8") as source_file:
        chunks = list(json.load(source_file))

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    vectors = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[str(chunk["text"]) for chunk in batch],
        )
        if len(response.data) != len(batch):
            raise RuntimeError(
                "Embedding response row count did not match the requested batch."
            )
        vectors.extend(item.embedding for item in response.data)

    embeddings = np.asarray(vectors, dtype=float)
    if embeddings.shape != (len(chunks), 3072):
        raise RuntimeError(
            f"Unexpected rebuilt embedding shape: {embeddings.shape}"
        )

    targets = (
        (CHUNKS_PATH, EMBEDDINGS_PATH),
        (
            SINGLE_APP_DIR / "psse_chunks_cached.json",
            SINGLE_APP_DIR / "psse_embeddings.npy",
        ),
    )
    temporary_targets = []
    for chunks_path, embeddings_path in targets:
        temporary_embeddings = embeddings_path.with_suffix(".npy.tmp")
        temporary_chunks = chunks_path.with_suffix(".json.tmp")
        with temporary_embeddings.open("wb") as embeddings_file:
            np.save(embeddings_file, embeddings)
        with temporary_chunks.open("w", encoding="utf-8") as chunks_file:
            json.dump(chunks, chunks_file, ensure_ascii=False, indent=2)
        temporary_targets.append(
            (
                temporary_chunks,
                chunks_path,
                temporary_embeddings,
                embeddings_path,
            )
        )

    for (
        temporary_chunks,
        chunks_path,
        temporary_embeddings,
        embeddings_path,
    ) in temporary_targets:
        os.replace(temporary_embeddings, embeddings_path)
        os.replace(temporary_chunks, chunks_path)
    print(
        f"Rebuilt both {len(chunks)}-chunk saved PSS/E indexes with "
        f"{EMBEDDING_MODEL}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild all saved corpus embeddings instead of validating them.",
    )
    parser.add_argument(
        "--confirm-paid-embedding-request",
        action="store_true",
        help="Acknowledge that --rebuild sends the full corpus to the API.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not args.rebuild:
        validate_saved_index_command()
        return
    if not args.confirm_paid_embedding_request:
        parser.error(
            "--rebuild also requires --confirm-paid-embedding-request"
        )
    if args.batch_size < 1 or args.batch_size > 100:
        parser.error("--batch-size must be between 1 and 100")
    rebuild_saved_index(batch_size=args.batch_size)


if __name__ == "__main__":
    main()
