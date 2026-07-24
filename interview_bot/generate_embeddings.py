import argparse
import json
import os
from pathlib import Path

from embedding_utils import (
    EMBEDDING_MODEL,
    chunk_texts,
    chunks_digest,
    create_embeddings,
    load_valid_embeddings,
    save_embedding_cache,
)


BASE_DIR = Path(__file__).resolve().parent
CHUNKS_FILE = BASE_DIR / "chunks_cleaned.json"
EMBEDDING_FILE = BASE_DIR / "embeddings.npy"
METADATA_FILE = BASE_DIR / "embeddings.meta.json"


def main():
    parser = argparse.ArgumentParser(
        description="Generate a content-validated embedding cache for InterviewBot."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate chunks and print the content digest without calling OpenAI.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when the content-validated saved cache is already current.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)
    texts = chunk_texts(chunks)

    print(f"Validated {len(texts)} chunks")
    print(f"Chunk digest: {chunks_digest(chunks)}")
    if args.dry_run:
        print("Dry run complete; no API calls or files written.")
        return

    cached_embeddings, cache_status = load_valid_embeddings(
        chunks,
        EMBEDDING_FILE,
        METADATA_FILE,
        model=EMBEDDING_MODEL,
    )
    if cached_embeddings is not None and not args.force:
        print(
            f"Saved embedding cache is already current with "
            f"{cached_embeddings.shape[0]} vectors; no API call was made."
        )
        print("Use --force only when an intentional full rebuild is required.")
        return

    print(f"Embedding generation is required because {cache_status}.")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Export it and rerun: "
            "python interview_bot/generate_embeddings.py"
        )

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    def report_progress(completed, total):
        print(f"Embedded {completed}/{total} chunks")

    embeddings = create_embeddings(
        client,
        texts,
        model=EMBEDDING_MODEL,
        batch_size=args.batch_size,
        progress=report_progress,
    )
    save_embedding_cache(
        chunks,
        embeddings,
        EMBEDDING_FILE,
        METADATA_FILE,
        model=EMBEDDING_MODEL,
    )
    print(f"Saved embeddings with shape {embeddings.shape} to {EMBEDDING_FILE.name}")
    print(f"Saved cache metadata to {METADATA_FILE.name}")


if __name__ == "__main__":
    main()
