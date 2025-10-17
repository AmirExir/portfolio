import streamlit as st
import os
import json
import glob
import openai 
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import numpy as np

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load or compute embeddings
@st.cache_data(show_spinner=False)
def load_Resource_Integration_chunks_and_embeddings():
    with open(os.path.join(os.path.dirname(__file__), "Resource_Integration_fully_structured.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)
        st.write("Current working directory:", os.getcwd())
        st.write("File absolute path:", os.path.join(os.path.dirname(__file__), "Resource_Integration_fully_structured.json"))

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
        st.warning(" No valid embeddings. Check your file or API key.")
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

# Find top K matches
def find_top_k_matches(query: str, chunks, embeddings, k=50):
    query_embedding = np.array(embed_query(query)).reshape(1, -1)
    scores = cosine_similarity(query_embedding, embeddings).flatten()
    top_indices = scores.argsort()[-k:][::-1]
    top_chunks = [chunks[i] for i in top_indices]
    return top_chunks

# Limit chunks by token budget
def limit_chunks_by_token_budget(chunks, max_input_tokens=100000):
    total = 0
    selected = []
    for chunk in chunks:
        token_count = len(chunk["text"].split())  # rough estimate
        if total + token_count > max_input_tokens:
            break
        selected.append(chunk)
        total += token_count
    return selected

# Streamlit UI
st.set_page_config(page_title="Amir Exir's Resource Integration  AI Assistant", page_icon="⚡")
st.title(" Ask Amir Exir's Resource Integration  AI Assistant")

# Load data and embeddings once
with st.spinner("Loading Resource Integration chunks and computing embeddings..."):
    chunks, embeddings = load_Resource_Integration_chunks_and_embeddings()

import re

def extract_function_names(chunks):
    pattern = r'\bpsspy\.(\w+)\b'
    func_names = set()
    for chunk in chunks:
        func_names.update(re.findall(pattern, chunk["text"]))
    return func_names

valid_funcs = extract_function_names(chunks)

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show past messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about ERCOT Resource Integration, QSA process..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        top_chunks = find_top_k_matches(prompt, chunks, embeddings, k=50)
        trimmed_chunks = limit_chunks_by_token_budget(top_chunks)
        combined_context = "\n\n---\n\n".join(chunk["text"] for chunk in trimmed_chunks)

        system_prompt = {
            "role": "system",
            "content": f"""
        You are an the most advanced ERCOT Resource Integration expert. When given a task, identify the relevant  and return a full explaination. Avoid made-up explaination. Cite the chunk you're using.

        Use only the following {len(trimmed_chunks)} reference chunks (from  handbook and examples):

        ---
        {combined_context}
        ---

        Respond with:
        - Clear descriptions of function usage
        - Real working Python code
        - Best practices and typical use cases

        Prioritize actual examples if available. Do not make up any function names not shown.
        """
        }

        messages = [system_prompt] + st.session_state.messages

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=8192,
            temperature = 0.2
            
        )

        bot_msg = response.choices[0].message.content

        def find_invalid_functions(response_text, valid_funcs):
            used = re.findall(r'\bpsspy\.(\w+)\b', response_text)
            return [f for f in used if f not in valid_funcs]

        invalid_funcs = find_invalid_functions(bot_msg, valid_funcs)

        if invalid_funcs:
            st.warning(f" Warning: These functions may not exist in the : {', '.join(invalid_funcs)}")
            bot_msg += f"\n\n *Caution: The following PSS/E  function(s) may be hallucinated or not found in the official documentation: {', '.join(invalid_funcs)}*"



        st.chat_message("assistant").markdown(bot_msg)
        st.session_state.messages.append({"role": "assistant", "content": bot_msg})