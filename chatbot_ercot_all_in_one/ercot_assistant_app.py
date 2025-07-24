# ercot_assistant_app.py

import streamlit as st

# ✅ THIS MUST BE FIRST Streamlit command
st.set_page_config(page_title="ERCOT Assistant", page_icon="⚡")

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# === Load cached chunks and embeddings ===
@st.cache_data(show_spinner=False)
def load_data():
    with open("ercot_chunks_cached.json", "r") as f:
        chunks = json.load(f)
    embeddings = np.load("ercot_embeddings.npy")
    return chunks, embeddings

chunks, embeddings = load_data()

# === Load same model to embed user query ===
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# === Embed query and find top matches ===
def get_top_k_matches(query, k=5):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_k_idx = np.argsort(similarities)[::-1][:k]
    return [(chunks[i]["text"], similarities[i]) for i in top_k_idx]

# === Streamlit UI ===
st.title("⚡ Ask ERCOT Assistant")

query = st.text_input("Ask your ERCOT-related question:")

if query:
    with st.spinner("Searching..."):
        results = get_top_k_matches(query, k=5)

    st.subheader("🔎 Top Answers:")
    for i, (text, score) in enumerate(results, 1):
        st.markdown(f"**[{i}]** (Score: {score:.4f})")
        st.write(text)
        st.markdown("---")