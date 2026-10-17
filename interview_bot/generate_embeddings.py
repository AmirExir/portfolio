import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from sync_manager_content import CONTENT_FILE, build_chunks, load_json

from embedding_utils import (
    EMBEDDING_MODEL,
    chunk_texts,
    chunks_digest,
    create_embeddings,
    load_valid_embeddings,
    plan_embedding_reuse,
    save_embedding_cache,
)


BASE_DIR = Path(__file__).resolve().parent
CHUNKS_FILE = BASE_DIR / "chunks_cleaned.json"
EMBEDDING_FILE = BASE_DIR / "embeddings.npy"
METADATA_FILE = BASE_DIR / "embeddings.meta.json"


def save_synchronized_content(chunks: list[dict], embeddings: object) -> None:
    """Stage a complete cache and corpus before publishing the corpus last.

    Generation and staging failures leave the active artifacts untouched.
    Readers retain the existing digest checks during the final replacements.
    """
    with TemporaryDirectory(prefix=".embedding-sync-", dir=BASE_DIR) as directory:
        staging_dir = Path(directory)
        staged_embeddings = staging_dir / EMBEDDING_FILE.name
        staged_metadata = staging_dir / METADATA_FILE.name
        staged_chunks = staging_dir / CHUNKS_FILE.name
        save_embedding_cache(
            chunks, embeddings, staged_embeddings, staged_metadata, model=EMBEDDING_MODEL
        )
        with staged_chunks.open("w", encoding="utf-8") as file:
            json.dump(chunks, file, indent=2, ensure_ascii=True)
            file.write("\n")
        staged_embeddings.replace(EMBEDDING_FILE)
        staged_metadata.replace(METADATA_FILE)
        staged_chunks.replace(CHUNKS_FILE)


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
    parser.add_argument(
        "--sync-manager-content",
        action="store_true",
        help=(
            "Sync manager_content.json after embedding succeeds, reusing validated "
            "vectors for unchanged texts."
        ),
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        existing_chunks = json.load(file)
    if args.sync_manager_content:
        chunks, _, _ = build_chunks(existing_chunks, load_json(CONTENT_FILE))
    else:
        chunks = existing_chunks
    texts = chunk_texts(chunks)
    if not texts:
        raise SystemExit("The interview corpus must contain at least one chunk")

    print(f"Validated {len(texts)} chunks")
    print(f"Chunk digest: {chunks_digest(chunks)}")
    if args.dry_run:
        print("Dry run complete; no API calls or files written.")
        return

    cached_embeddings, cache_status = load_valid_embeddings(
        existing_chunks,
        EMBEDDING_FILE,
        METADATA_FILE,
        model=EMBEDDING_MODEL,
    )
    if (
        cached_embeddings is not None
        and chunks == existing_chunks
        and not args.force
    ):
        print(
            f"Saved embedding cache is already current with "
            f"{cached_embeddings.shape[0]} vectors; no API call was made."
        )
        print("Use --force only when an intentional full rebuild is required.")
        return

    reuse_cache = args.sync_manager_content and cached_embeddings is not None and not args.force
    if args.sync_manager_content:
        row_indices, texts_to_embed = plan_embedding_reuse(
            existing_chunks if reuse_cache else [], chunks
        )
    else:
        row_indices, texts_to_embed = list(range(len(texts))), texts
    if reuse_cache:
        print(
            f"Reusing validated saved vectors; {len(texts_to_embed)} unique new or "
            "changed texts need embeddings."
        )
    else:
        if args.force:
            cache_status = "a full rebuild was requested"
        print(f"Embedding generation is required because {cache_status}.")

    generated_embeddings = None
    if texts_to_embed:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            command = "python interview_bot/generate_embeddings.py"
            if args.sync_manager_content:
                command += " --sync-manager-content"
            raise SystemExit(f"OPENAI_API_KEY is not set. Export it and rerun: {command}")

        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        def report_progress(completed: int, total: int) -> None:
            print(f"Embedded {completed}/{total} chunks")

        generated_embeddings = create_embeddings(
            client,
            texts_to_embed,
            model=EMBEDDING_MODEL,
            batch_size=args.batch_size,
            progress=report_progress,
        )

    import numpy as np

    if reuse_cache:
        combined = cached_embeddings
        if generated_embeddings is not None:
            combined = np.concatenate((cached_embeddings, generated_embeddings), axis=0)
    else:
        combined = generated_embeddings
    embeddings = np.asarray(combined[row_indices], dtype="float32")

    if args.sync_manager_content:
        save_synchronized_content(chunks, embeddings)
        print(f"Updated {CHUNKS_FILE.name} after successful embedding generation")
    else:
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
