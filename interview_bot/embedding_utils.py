import hashlib
import json
from pathlib import Path

EMBEDDING_MODEL = "text-embedding-3-large"


def chunk_texts(chunks):
    texts = []
    for index, chunk in enumerate(chunks):
        text = chunk.get("text", "") if isinstance(chunk, dict) else ""
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Chunk {index} is missing non-empty 'text'")
        texts.append(text.strip())
    return texts


def chunks_digest(chunks):
    payload = json.dumps(chunk_texts(chunks), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_metadata(chunks, embeddings, model=EMBEDDING_MODEL):
    return {
        "model": model,
        "chunk_count": len(chunks),
        "embedding_dimensions": int(embeddings.shape[1]),
        "chunks_sha256": chunks_digest(chunks),
    }


def load_valid_embeddings(chunks, embedding_file, metadata_file, model=EMBEDDING_MODEL):
    import numpy as np

    embedding_file = Path(embedding_file)
    metadata_file = Path(metadata_file)
    if not embedding_file.exists() or not metadata_file.exists():
        return None, "embedding cache or metadata is missing"

    try:
        embeddings = np.load(embedding_file)
        with metadata_file.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, f"embedding cache could not be read: {exc}"

    if embeddings.ndim != 2:
        return None, "embedding cache is not a two-dimensional matrix"
    if embeddings.shape[0] != len(chunks):
        return None, "embedding row count does not match chunks_cleaned.json"
    if metadata.get("model") != model:
        return None, "embedding model does not match the configured model"
    if metadata.get("chunk_count") != len(chunks):
        return None, "embedding metadata count does not match chunks_cleaned.json"
    if metadata.get("embedding_dimensions") != embeddings.shape[1]:
        return None, "embedding dimensions do not match metadata"
    if metadata.get("chunks_sha256") != chunks_digest(chunks):
        return None, "chunk content changed after embeddings were generated"

    return np.asarray(embeddings, dtype="float32"), "cache is current"


def create_embeddings(client, texts, model=EMBEDDING_MODEL, batch_size=64, progress=None):
    import numpy as np

    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(input=batch, model=model)
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(batch):
            raise RuntimeError(
                f"Embedding API returned {len(ordered)} vectors for a batch of {len(batch)} texts"
            )
        vectors.extend(item.embedding for item in ordered)
        if progress:
            progress(min(start + len(batch), len(texts)), len(texts))

    embeddings = np.asarray(vectors, dtype="float32")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise RuntimeError("Generated embedding matrix has an unexpected shape")
    return embeddings


def save_embedding_cache(chunks, embeddings, embedding_file, metadata_file, model=EMBEDDING_MODEL):
    import numpy as np

    embedding_file = Path(embedding_file)
    metadata_file = Path(metadata_file)
    embedding_tmp = embedding_file.with_suffix(embedding_file.suffix + ".tmp")
    metadata_tmp = metadata_file.with_suffix(metadata_file.suffix + ".tmp")

    with embedding_tmp.open("wb") as file:
        np.save(file, np.asarray(embeddings, dtype="float32"))
    with metadata_tmp.open("w", encoding="utf-8") as file:
        json.dump(build_metadata(chunks, embeddings, model=model), file, indent=2)
        file.write("\n")

    embedding_tmp.replace(embedding_file)
    metadata_tmp.replace(metadata_file)
