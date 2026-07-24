# retriever.py
import os
import json
import numpy as np
import streamlit as st

from utils import (
    find_top_k_chunks,
    limit_chunks_by_token_budget,
    validate_saved_index,
)

# ---------------------------
# Load chunks + embeddings
# ---------------------------

@st.cache_resource(show_spinner=False)
def load_chunks_and_embeddings(
    json_file="input_chunks.json",
    embedding_model="text-embedding-3-large"
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, json_file)

    cached_emb = os.path.join(base_dir, "psse_embeddings.npy")
    cached_chunks = os.path.join(base_dir, "psse_chunks_cached.json")

    missing = [
        os.path.basename(path)
        for path in (cached_emb, cached_chunks)
        if not os.path.isfile(path)
    ]
    if missing:
        raise RuntimeError(
            "Saved PSS/E RAG artifacts are missing: "
            f"{', '.join(missing)}. Runtime corpus embedding is disabled to "
            "prevent surprise API charges; build and deploy the index offline."
        )

    with open(cached_chunks, "r", encoding="utf-8") as f:
        chunks = list(json.load(f))
    embeddings = np.load(cached_emb)
    expected_dimension = 3072 if embedding_model == "text-embedding-3-large" else None
    validate_saved_index(
        chunks,
        embeddings,
        expected_dimension=expected_dimension,
    )

    # When a source chunk file is deployed, reject a stale cache rather than
    # silently answering from embeddings built for different text.
    if os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            source_chunks = list(json.load(f))
        if source_chunks != chunks:
            raise RuntimeError(
                "The saved PSS/E chunks do not match the source chunk file. "
                "Rebuild the saved index offline before deploying it."
            )

    return chunks, embeddings


# ---------------------------
# Retrieval helpers
# ---------------------------

def find_relevant_chunks(
    query,
    chunks,
    embeddings,
    k=10,
    max_tokens=12_000,
    query_cache=None,
):
    top_chunks = find_top_k_chunks(
        query,
        chunks,
        embeddings,
        k=k,
        query_cache=query_cache,
    )
    return limit_chunks_by_token_budget(top_chunks, max_tokens=max_tokens)
