# retriever.py (Refactored to use utils.py)

import os
import json
import streamlit as st
from utils import embed_query, find_top_k_chunks
from openai import OpenAI
import numpy as np

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@st.cache_data(show_spinner=False)
def load_chunks_and_embeddings(json_file="psse_examples_chunks.json", embedding_model="text-embedding-3-small"):
    with open(json_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Filter out irrelevant content (GIC, harmonics, etc.)
    excluded_keywords = ["gic", "harmonic", "emtp", "transient stability", "dynamic", "pmu"]
    filtered_chunks = [
        chunk for chunk in chunks
        if not any(keyword in chunk["text"].lower() for keyword in excluded_keywords)
    ]

    embeddings = []
    for chunk in filtered_chunks:
        try:
            response = client.embeddings.create(model=embedding_model, input=chunk["text"][:8192])
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            print(f"⚠️ Embedding failed for a chunk: {e}")
            embeddings.append(None)

    # Filter out chunks with failed embeddings
    final_pairs = [(c, e) for c, e in zip(filtered_chunks, embeddings) if e is not None]
    if not final_pairs:
        raise ValueError("No usable chunks found after filtering and embedding.")

    final_chunks, final_embeddings = zip(*final_pairs)
    return list(final_chunks), np.array(final_embeddings)

def find_relevant_chunks(query, chunks, embeddings, k=25):
    query_embed = embed_query(query)
    return find_top_k_chunks(query, chunks, embeddings, k)