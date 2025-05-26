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
def load_psse_chunks_and_embeddings():
    with open(os.path.join(os.path.dirname(__file__), "psse_examples_chunks.json"), "r", encoding="utf-8") as f:
    chunks = json.load(f)
st.write("Current working directory:", os.getcwd())
st.write("File absolute path:", os.path.join(os.path.dirname(__file__), "psse_examples_chunks.json"))

    embeddings = []
    embedding_model = "text-embedding-3-small"

    for chunk in chunks:
        try:
            response = client.embeddings.create(
                model=embedding_model,
                input=chunk["text"][:8192]
            )
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            st.warning(f"Embedding failed for chunk {chunk['id']}: {e}")
            embeddings.append(None)

    valid_pairs = [(c, e) for c, e in zip(chunks, embeddings) if e is not None]

    if not valid_pairs:
        st.warning("⚠️ No valid embeddings. Check your file or API key.")
        raise ValueError("No valid embeddings were generated.")

    chunks, embeddings = zip(*valid_pairs)
    embeddings = np.array(embeddings)
    return list(chunks), embeddings
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
    scores = cosine_similarity(query_embedding.reshape(1, -1), embeddings).flatten()
    best_idx = int(np.argmax(scores))
    return chunks[best_idx]

# Streamlit UI
st.set_page_config(page_title="Amir Exir's PSSE API AI Assistant", page_icon="⚡")
st.title("🧠 Ask Amir Exir's PSSE API AI Assistant")

# Load data and embeddings once
with st.spinner("Loading nodal protocols and computing embeddings..."):
    chunks, embeddings = load_psse_chunks_and_embeddings()

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show past messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about PSS/E automation, code generation, or API usage..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        best_chunk = find_best_match(prompt, chunks, embeddings)

        system_prompt = {
            "role": "system",
            "content": f"""
        You are an expert assistant on the PSS/E Python API and automation for power systems.

        Only use the following API chunk as your source:

        ---
        {best_chunk['text']}
        ---

        Answer clearly, with a focus on:
        - What the function(s) do
        - Example usage in Python
        - When they are typically used

        Do not hallucinate or invent details beyond this context.
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