import streamlit as st
import os
import openai
import json
import glob
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import numpy as np

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load or compute embeddings
@st.cache_data(show_spinner=False)
def load_ercot_chunks_and_embeddings():
    from openai import OpenAI
    embedding_model = "text-embedding-3-small"

    chunks = []
    embeddings = []

    for filepath in sorted(glob.glob("ercot_planning_part*.txt")):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            chunks.append({"filename": filepath, "text": text})

    for chunk in chunks:
        response = client.embeddings.create(
            model=embedding_model,
            input=chunk["text"][:8192]  # Embed first 8192 tokens max
        )
        embeddings.append(response.data[0].embedding)

    return chunks, np.array(embeddings)

# Embed the user query
def embed_query(query: str) -> List[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return response.data[0].embedding

# Find best matching chunk
def find_best_match(query: str, chunks, embeddings):
    query_embedding = np.array(embed_query(query)).reshape(1, -1)
    scores = cosine_similarity([query_embedding], embeddings).flatten()
    best_idx = int(np.argmax(scores))
    return chunks[best_idx]

# Streamlit UI
st.set_page_config(page_title="Amir Exir's ERCOT Planning Guides AI Assistant", page_icon="⚡")
st.title("⚡ Ask Amir Exir's ERCOT Planning Guides AI Assistant")

# Load data and embeddings once
with st.spinner("Loading planning guides and computing embeddings..."):
    chunks, embeddings = load_ercot_chunks_and_embeddings()

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show past messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about ERCOT planning guides..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        best_chunk = find_best_match(prompt, chunks, embeddings)

        system_prompt = {
            "role": "system",
            "content": f"""
You are an expert assistant on ERCOT's planning guides.
Only use the following documentation to answer the question:

---
Filename: {best_chunk['filename']}

{best_chunk['text']}
---
Stay factual. Do not guess beyond the information provided above.
"""
        }

        messages = [system_prompt] + st.session_state.messages

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1024
        )

        bot_msg = response.choices[0].message.content
        st.chat_message("assistant").markdown(bot_msg)
        st.session_state.messages.append({"role": "assistant", "content": bot_msg})