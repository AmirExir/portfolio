# retriever.py (Refactored to use utils.py)

import os
import streamlit as st
from utils import load_chunks_and_embeddings, embed_query, find_top_k_chunks

@st.cache_data(show_spinner=False)
def load_chunks_and_embeddings_cached():
    return load_chunks_and_embeddings("psse_examples_chunks.json")

def find_relevant_chunks(query, chunks, embeddings, k=25):
    query_embed = embed_query(query)
    return find_top_k_chunks(query, chunks, embeddings, k)
