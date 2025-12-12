# retriever.py
import os
import json
import numpy as np
import streamlit as st

from utils import (
    embed_query,
    find_top_k_chunks,
    limit_chunks_by_token_budget,
)

# ---------------------------
# Load chunks + embeddings
# ---------------------------

@st.cache_data(show_spinner=False)
def load_chunks_and_embeddings(
    json_file="input_chunks.json",
    embedding_model="text-embedding-3-large"
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, json_file)

    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")

    cached_emb = os.path.join(base_dir, "psse_embeddings.npy")
    cached_chunks = os.path.join(base_dir, "psse_chunks_cached.json")

    # ---- Load cached ----
    if os.path.exists(cached_emb) and os.path.exists(cached_chunks):
        with open(cached_chunks, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        embeddings = np.load(cached_emb)
        return chunks, embeddings

    # ---- Compute embeddings ----
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = []
    valid_chunks = []

    for chunk in chunks:
        try:
            resp = client.embeddings.create(
                model=embedding_model,
                input=chunk["text"][:8192]
            )
            embeddings.append(resp.data[0].embedding)
            valid_chunks.append(chunk)
        except Exception as e:
            print(f"[Embedding failed] {e}")

    if not embeddings:
        raise ValueError("No valid embeddings generated")

    embeddings = np.array(embeddings)

    # ---- Cache results ----
    np.save(cached_emb, embeddings)
    with open(cached_chunks, "w", encoding="utf-8") as f:
        json.dump(valid_chunks, f, indent=2)

    return valid_chunks, embeddings


# ---------------------------
# Retrieval helpers
# ---------------------------

def find_relevant_chunks(query, chunks, embeddings, k=10, max_tokens=30000):
    top_chunks = find_top_k_chunks(query, chunks, embeddings, k=k)
    return limit_chunks_by_token_budget(top_chunks, max_tokens=max_tokens)