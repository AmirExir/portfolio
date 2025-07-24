import streamlit as st
from openai import OpenAI
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load OpenAI API key
api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Load saved embeddings and chunks
@st.cache_data(show_spinner=False)
def load_ercot_chunks_and_embeddings():
    with open("ercot_chunks_cached.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load("ercot_embeddings.npy")
    return chunks, embeddings

chunks, embeddings = load_ercot_chunks_and_embeddings()

# Embed query
def embed_query(query):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    )
    return np.array(response.data[0].embedding).reshape(1, -1)

# Search
def search_top_matches(query, chunks, embeddings, k=5):
    query_vec = embed_query(query)
    scores = cosine_similarity(query_vec, embeddings).flatten()
    top_k = scores.argsort()[-k:][::-1]
    return [chunks[i] for i in top_k], scores[top_k]

# Streamlit UI
st.title("ERCOT AI Assistant")
st.markdown("Ask me about **ERCOT Planning Guide**, **Protocols**, or **Interconnection Handbook** 📘")

query = st.text_input("🧠 Enter your question about ERCOT:")

if query:
    with st.spinner("Searching..."):
        top_chunks, scores = search_top_matches(query, chunks, embeddings, k=5)

    st.subheader("🔍 Top Matches")
    for i, chunk in enumerate(top_chunks):
        st.markdown(f"**[{i+1}]** — Similarity Score: `{scores[i]:.2f}`")
        st.write(chunk["text"])
        st.markdown("---")