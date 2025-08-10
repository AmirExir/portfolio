import streamlit as st
import os
import json
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import time

# Setup OpenAI client
api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Safe retry wrapper
def safe_openai_call(api_function, max_retries=5, backoff_factor=2, **kwargs):
    retries = 0
    while retries < max_retries:
        try:
            return api_function(**kwargs)
        except Exception as e:
            wait = backoff_factor ** retries
            st.warning(f"Retrying in {wait} sec due to: {e}")
            time.sleep(wait)
            retries += 1
    st.error("❌ Failed after retries.")
    return None

# Load ERCOT chunks and embeddings
@st.cache_data(show_spinner=False)
def load_chunks_and_embeddings():
    with open("ercot_chunks_cached.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load("ercot_embeddings.npy")
    return chunks, embeddings

chunks, embeddings = load_chunks_and_embeddings()

# Embed the query
def embed_query(query):
    response = safe_openai_call(
        client.embeddings.create,
        model="text-embedding-3-large",
        input=query
    )
    return np.array(response.data[0].embedding).reshape(1, -1) if response else None

# Find top matches
def find_top_matches(query, chunks, embeddings, k=5):
    query_vec = embed_query(query)
    if query_vec is None:
        return []
    scores = cosine_similarity(query_vec, embeddings).flatten()
    top_k = scores.argsort()[-k:][::-1]
    return [chunks[i] for i in top_k]

# Limit token budget (optional)
def limit_chunks_by_token_budget(chunks, max_tokens=100000):
    total = 0
    selected = []
    for chunk in chunks:
        tokens = len(chunk["text"].split())
        if total + tokens > max_tokens:
            break
        selected.append(chunk)
        total += tokens
    return selected

# Streamlit UI setup
st.set_page_config(page_title="ERCOT AI Assistant", page_icon="⚡")
st.title("⚡ Ask Amir Exir's ERCOT AI Assistant")

# Session memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Handle chat input
if prompt := st.chat_input("Ask about ERCOT planning, protocols, interconnection..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        top_chunks = find_top_matches(prompt, chunks, embeddings, k=25)
        trimmed = limit_chunks_by_token_budget(top_chunks, max_tokens=100000)
        context = "\n\n---\n\n".join(c["text"] for c in trimmed)

        system_prompt = {
            "role": "system",
            "content": f"""
You are an ERCOT interconnection, planning, and protocol expert assistant. Only answer using the following documents:

{context}

Do not make up anything. Only answer based on the given context.
"""
        }

        messages = [system_prompt] + st.session_state.messages

        response = safe_openai_call(
            client.chat.completions.create,
            model="gpt-5",
            messages=messages,
            max_tokens=2048
        )

    if response:
        reply = response.choices[0].message.content
        st.chat_message("assistant").markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})