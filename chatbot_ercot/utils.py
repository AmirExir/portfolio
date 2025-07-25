import json
import numpy as np

def split_text_into_chunks(text, chunk_size=1000, overlap=200, source=""):
    words = text.split()
    chunks = []
    i = 0
    chunk_index = 0
    while i < len(words):
        chunk_words = words[i:i+chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "text": chunk_text,
            "source": source,
            "chunk_index": chunk_index
        })
        i += chunk_size - overlap
        chunk_index += 1
    return chunks

def save_embeddings(chunks, embeddings, json_path, embeddings_path):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    np.save(embeddings_path, np.array(embeddings))