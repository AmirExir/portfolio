import argparse
import json
import os
from pathlib import Path

import numpy as np
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
CHUNKS_PATH = BASE_DIR / "ercot_chunks_cached.json"
EMBEDDINGS_PATH = BASE_DIR / "ercot_embeddings.npy"
EMBEDDING_MODEL = "text-embedding-3-large"


def load_chunks() -> list[dict]:
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    if not isinstance(chunks, list):
        raise ValueError("ercot_chunks_cached.json must contain a JSON array.")

    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk {index} is not an object.")
        if sorted(chunk.keys()) != ["chunk_index", "source", "text"]:
            raise ValueError(f"Chunk {index} must contain exactly text, source, and chunk_index.")
        if not isinstance(chunk["text"], str) or not chunk["text"].strip():
            raise ValueError(f"Chunk {index} has empty or invalid text.")
        if not isinstance(chunk["source"], str) or not chunk["source"].strip():
            raise ValueError(f"Chunk {index} has invalid source.")
        if not isinstance(chunk["chunk_index"], int):
            raise ValueError(f"Chunk {index} has invalid chunk_index.")

    return chunks


def require_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key in {"your-key", "your-key-here"}:
        raise SystemExit("OPENAI_API_KEY is missing or still set to a placeholder.")
    return api_key


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed the existing ERCOT cached JSON chunks.")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    chunks = load_chunks()
    texts = [chunk["text"] for chunk in chunks]
    max_len = max((len(text) for text in texts), default=0)

    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")
    print(f"Max chunk length: {max_len} characters")

    client = OpenAI(api_key=require_api_key())
    embeddings = []
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start:start + args.batch_size]
        end = start + len(batch)
        print(f"Embedding chunks {start + 1}-{end}/{len(texts)}")
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        embeddings.extend(item.embedding for item in response.data)

    if len(embeddings) != len(chunks):
        raise RuntimeError(f"Embedding count mismatch: {len(embeddings)} vs {len(chunks)} chunks.")

    array = np.array(embeddings, dtype=np.float32)
    tmp_path = EMBEDDINGS_PATH.with_suffix(".npy.tmp")
    with open(tmp_path, "wb") as f:
        np.save(f, array)
    tmp_path.replace(EMBEDDINGS_PATH)

    print(f"Saved embeddings: {EMBEDDINGS_PATH}")
    print(f"Shape: {array.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
