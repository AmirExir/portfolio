# utils.py

import os
import json
import numpy as np
import re
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_chunks_and_embeddings(json_file="psse_chunks.json", embedding_model="text-embedding-3-small"):
    """Load documentation chunks and compute their embeddings."""
    with open(json_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = []
    for chunk in chunks:
        try:
            response = client.embeddings.create(
                model=embedding_model,
                input=chunk["text"][:8192]
            )
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            print(f"[Embedding Error] Chunk ID {chunk.get('id', '?')}: {e}")
            embeddings.append(None)

    # Filter out failed embeddings
    valid_pairs = [(c, e) for c, e in zip(chunks, embeddings) if e is not None]
    if not valid_pairs:
        raise ValueError("❌ No valid embeddings generated. Check input or OpenAI key.")

    chunks, embeddings = zip(*valid_pairs)
    return list(chunks), np.array(embeddings)

def embed_query(query, embedding_model="text-embedding-3-small"):
    """Embed a user query using OpenAI's embedding model."""
    response = client.embeddings.create(model=embedding_model, input=query)
    return response.data[0].embedding

def find_top_k_chunks(query, chunks, embeddings, k=50):
    """Find top-k semantically relevant chunks to a query."""
    query_emb = np.array(embed_query(query)).reshape(1, -1)
    scores = cosine_similarity(query_emb, embeddings).flatten()
    top_indices = scores.argsort()[-k:][::-1]
    return [chunks[i] for i in top_indices]

def limit_chunks_by_token_budget(chunks, max_tokens=8000):
    """Truncate context chunks to fit within token budget."""
    total = 0
    selected = []
    for chunk in chunks:
        tokens = len(chunk["text"].split())
        if total + tokens > max_tokens:
            break
        selected.append(chunk)
        total += tokens
    return selected

def extract_function_names(chunks):
    """Extract all PSSPY function names from chunks for validation."""
    pattern = r'\bpsspy\.(\w+)\b'
    func_names = set()
    for chunk in chunks:
        func_names.update(re.findall(pattern, chunk["text"]))
    return func_names