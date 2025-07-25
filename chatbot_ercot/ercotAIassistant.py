import streamlit as st
import os
import json
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load precomputed chunks and embeddings
@st.cache_data(show_spinner=False)
def load_ercot_chunks_and_embeddings():
    chunks_path = "chatbot_ercot/ercot_planning_chunks.json"
    embeddings_path = "chatbot_ercot/ercot_planning_embeddings.npy"

    if not os.path.exists(chunks_path) or not os.path.exists(embeddings_path):
        raise FileNotFoundError("Embeddings or chunks file not found.")

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load(embeddings_path)

    return chunks, embeddings

# Embed the user query
def embed_query(query: str) -> List[float]:
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    )
    return response.data[0].embedding

# Find best matching chunk
def find_best_match(query: str, chunks, embeddings):
    query_embedding = np.array(embed_query(query)).reshape(1, -1)
    scores = cosine_similarity(query_embedding, embeddings).flatten()
    best_idx = int(np.argmax(scores))
    return chunks[best_idx]

# Streamlit UI
st.set_page_config(page_title="Amir Exir's ERCOT Planning Guides AI Assistant", page_icon="⚡")
st.title("⚡ Ask Amir Exir's ERCOT Planning Guides AI Assistant")

# Load data and embeddings once
with st.spinner("Loading planning guide embeddings..."):
    chunks, embeddings = load_ercot_chunks_and_embeddings()

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
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
Filename: {best_chunk['source']}

{best_chunk['text']}
---
Instructions:
- Stay factual and grounded strictly in the provided content.
- If the answer is not explicitly found in the document, respond: "I couldn’t find that in the documentation."
- Do NOT guess, assume, or rely on outside knowledge.
"""
        }

        messages = [system_prompt] + st.session_state.messages

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000
        )

        bot_msg = response.choices[0].message.content
        st.chat_message("assistant").markdown(bot_msg)
        st.session_state.messages.append({"role": "assistant", "content": bot_msg})