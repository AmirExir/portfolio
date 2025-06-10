# utils.py

import os
import json
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_chunks_and_embeddings(json_file="psse_examples_chunks.json", embedding_model="text-embedding-3-small"):
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
            print(f"Embedding failed for chunk {chunk['id']}: {e}")
            embeddings.append(None)

    valid_pairs = [(c, e) for c, e in zip(chunks, embeddings) if e is not None]
    if not valid_pairs:
        raise ValueError("No valid embeddings generated.")

    chunks, embeddings = zip(*valid_pairs)
    return list(chunks), np.array(embeddings)

def embed_query(query, embedding_model="text-embedding-3-small"):
    response = client.embeddings.create(model=embedding_model, input=query)
    return response.data[0].embedding

def find_top_k_chunks(query, chunks, embeddings, k=50):
    query_emb = np.array(embed_query(query)).reshape(1, -1)
    scores = cosine_similarity(query_emb, embeddings).flatten()
    top_indices = scores.argsort()[-k:][::-1]
    return [chunks[i] for i in top_indices]

def limit_chunks_by_token_budget(chunks, max_tokens=8000):
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
    import re
    pattern = r'\bpsspy\.(\w+)\b'
    func_names = set()
    for chunk in chunks:
        func_names.update(re.findall(pattern, chunk["text"]))
    return func_names