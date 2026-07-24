"""Offline generator for the legacy Planning Guide embedding cache.

The deployed assistant reads the central ERCOT index and never imports this
module.  This utility remains for reproducibility of the historical standalone
cache, but a paid full-corpus rebuild requires an explicit ``--force``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

try:
    from .utils import split_text_into_chunks
except ImportError:  # Direct ``python chatbot_ercot/generate_embeddings.py`` run.
    from utils import split_text_into_chunks


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILES = tuple(BASE_DIR / f"ercot_planning_part{part}.txt" for part in range(1, 4))
OUTPUT_JSON = BASE_DIR / "ercot_planning_chunks.json"
OUTPUT_EMBEDDINGS = BASE_DIR / "ercot_planning_embeddings.npy"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSION = 3072


def build_chunks() -> list[dict]:
    chunks: list[dict] = []
    for path in INPUT_FILES:
        text = path.read_text(encoding="utf-8")
        chunks.extend(split_text_into_chunks(text, source=path.name))
    return chunks


def saved_cache_is_current(chunks: list[dict]) -> bool:
    try:
        saved_chunks = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
        embeddings = np.load(OUTPUT_EMBEDDINGS, mmap_mode="r", allow_pickle=False)
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


def get_embedding_with_retry(client, text: str, retries: int = 5, delay: int = 5):
    for attempt in range(retries):
        try:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
            return response.data[0].embedding
        except Exception as exc:
            if getattr(exc, "status_code", None) in {401, 403}:
                raise RuntimeError("OpenAI authentication failed; not retrying.") from None
            if attempt + 1 == retries:
                raise RuntimeError(
                    f"Embedding failed after {retries} attempts: {type(exc).__name__}"
                ) from exc
            time.sleep(delay * (2**attempt))
    raise AssertionError("unreachable")


def write_cache(chunks: list[dict], embeddings: list[list[float]]) -> None:
    if len(chunks) != len(embeddings):
        raise RuntimeError(
            f"Refusing a misaligned cache: {len(chunks)} chunks, {len(embeddings)} vectors"
        )
    chunk_tmp = OUTPUT_JSON.with_suffix(".json.tmp")
    embeddings_tmp = OUTPUT_EMBEDDINGS.with_suffix(".npy.tmp")
    chunk_tmp.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    with embeddings_tmp.open("wb") as handle:
        np.save(handle, np.asarray(embeddings, dtype=np.float32))
    chunk_tmp.replace(OUTPUT_JSON)
    embeddings_tmp.replace(OUTPUT_EMBEDDINGS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate or explicitly rebuild the legacy Planning Guide cache."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Acknowledge the paid full-corpus OpenAI embedding rebuild.",
    )
    args = parser.parse_args(argv)

    chunks = build_chunks()
    print(f"Prepared {len(chunks)} legacy Planning Guide chunks.")
    if saved_cache_is_current(chunks) and not args.force:
        print("Saved cache is current; no API calls were made.")
        return 0
    if not args.force:
        print("Saved cache is missing or stale; rerun with --force to rebuild it offline.")
        return 2

    from openai import OpenAI

    client = OpenAI(api_key=require_api_key())
    embeddings = []
    for index, chunk in enumerate(chunks, 1):
        print(f"Embedding chunk {index}/{len(chunks)}")
        embeddings.append(get_embedding_with_retry(client, chunk["text"]))
    write_cache(chunks, embeddings)
    print(f"Saved {len(embeddings)} aligned embeddings to {OUTPUT_EMBEDDINGS}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
