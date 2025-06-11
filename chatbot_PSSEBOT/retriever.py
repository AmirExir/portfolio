# retriever.py (Refactored to use utils.py)

import os
import streamlit as st
from utils import load_chunks_and_embeddings, embed_query, find_top_k_chunks

@st.cache_data(show_spinner=False)
def load_chunks_and_embeddings(json_file="psse_examples_chunks.json", embedding_model="text-embedding-3-small"):
    with open(json_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Filter out irrelevant or misleading content (e.g., GIC, harmonic, EMT)
    excluded_keywords = ["gic", "harmonic", "emtp", "transient stability", "dynamic", "pmu"]
    filtered_chunks = [
        chunk for chunk in chunks
        if not any(keyword in chunk["text"].lower() for keyword in excluded_keywords)
    ]

def find_relevant_chunks(query, chunks, embeddings, k=25):
    query_embed = embed_query(query)
    return find_top_k_chunks(query, chunks, embeddings, k)
