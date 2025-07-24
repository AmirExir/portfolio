
import streamlit as st
import os
import json
import glob
import openai 
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import numpy as np

import time
def safe_openai_call(api_function, max_retries=5, backoff_factor=2, **kwargs):
    retries = 0
    while retries < max_retries:
        try:
            return api_function(**kwargs)
        except openai.RateLimitError:
            wait_time = backoff_factor ** retries
            st.warning(f"⚠️ Rate limit hit. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
        except Exception as e:
            st.error(f"❌ API call failed: {e}")
            break
    return None

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load or compute embeddings
@st.cache_data(show_spinner=False)
def load_psse_chunks_and_embeddings():
    base_path = os.path.dirname(__file__)
    cached_emb = os.path.join(base_path, "psse_embeddings.npy")
    cached_chunks = os.path.join(base_path, "psse_chunks_cached.json")
    input_file = os.path.join(base_path, "input_chunks.json")

    if os.path.exists(cached_emb) and os.path.exists(cached_chunks):
        st.write("✅ Using precomputed embeddings from .npy and .json")
        with open(cached_chunks, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        embeddings = np.load(cached_emb)
        return list(chunks), embeddings

    st.write("⚠️ Precomputed files not found, computing new embeddings...")
    with open(input_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # 🔻 Fallback to compute embeddings
    with open(os.path.join(os.path.dirname(__file__), "input_chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)
        st.write("Current working directory:", os.getcwd())
        st.write("File absolute path:", os.path.join(os.path.dirname(__file__), "input_chunks.json"))

    embeddings = []
    embedding_model = "text-embedding-3-large"

    for chunk in chunks:
        try:
            response = safe_openai_call(
                client.embeddings.create,
                model=embedding_model,
                input=chunk["text"][:8192]
            )
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            st.warning(f"Embedding failed for chunk {chunk.get('id', 'unknown')}: {e}")
            embeddings.append(None)

    valid_pairs = [(c, e) for c, e in zip(chunks, embeddings) if e is not None]

    if not valid_pairs:
        st.warning("⚠️ No valid embeddings. Check your file or API key.")
        raise ValueError("No valid embeddings were generated.")

    chunks, embeddings = zip(*valid_pairs)
    embeddings = np.array(embeddings)

    # ✅ Save to disk for reuse
    np.save("psse_embeddings.npy", embeddings)
    with open("psse_chunks_cached.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    return list(chunks), embeddings

# Embed the user query
def embed_query(query: str) -> List[float]:
    response = safe_openai_call(
        client.embeddings.create,
        model="text-embedding-3-large",
        input=query
    )
    return response.data[0].embedding if response else []

# Find top K matches
def find_top_k_matches(query: str, chunks, embeddings, k=10):
    query_vec = embed_query(query)

    if not query_vec:
        st.error("❌ Failed to embed query — try rephrasing your question.")
        return []

    query_embedding = np.array(query_vec).reshape(1, -1)

    if query_embedding.shape[1] != embeddings.shape[1]:
        st.error(f"❌ Embedding dimension mismatch: {query_embedding.shape[1]} vs {embeddings.shape[1]}")
        return []

    scores = cosine_similarity(query_embedding, embeddings).flatten()
    top_indices = scores.argsort()[-k:][::-1]
    return [chunks[i] for i in top_indices]

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
st.set_page_config(page_title="Amir Exir's PSSE automation Assistant", page_icon="⚡")
st.title("🧠 Ask Amir Exir's PSSE automation Assistant")

# Load data and embeddings once
with st.spinner("Loading PSSE API examples and computing embeddings..."):
    chunks, embeddings = load_psse_chunks_and_embeddings()

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
if prompt := st.chat_input("Ask about PSS/E automation, code generation, or API usage..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        top_chunks = find_top_k_matches(prompt, chunks, embeddings, k=50)
        trimmed_chunks = limit_chunks_by_token_budget(top_chunks)
        combined_context = "\n\n---\n\n".join(chunk["text"] for chunk in trimmed_chunks)

        system_prompt = {
            "role": "system",
            "content": f"""
        You are an the most advanced PSS/E python API and automation expert for power systems. When given a task, identify the relevant API and return a full code sample. Avoid made-up functions. Cite the chunk you're using.
        
        Use only the following {len(trimmed_chunks)} reference chunks (from API manual and examples):

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
            model="gpt-4o",
            messages=messages,
            max_tokens=2048
            temperature=0.0
        )

        bot_msg = response.choices[0].message.content

        
        import re

        def find_invalid_functions(response_text, valid_funcs):
            used = re.findall(r'\bpsspy\.(\w+)\b', response_text)
            return [f for f in used if f not in valid_funcs]


        invalid_funcs = find_invalid_functions(bot_msg, valid_funcs)

        # Auto-correct loop if invalid functions found
        if invalid_funcs:
            st.warning(f"⚠️ Warning: These functions may not exist in the API: {', '.join(invalid_funcs)}")

            correction_prompt = {
                "role": "user",
                "content": (
                    f"⚠️ You used invalid function(s): {', '.join(invalid_funcs)}. "
                    "Please revise your answer using only valid PSS/E API functions from the reference chunks provided earlier. "
                    "Do not make up any function names."
                )
            }

            # Add original assistant message and correction request
            messages.append({"role": "assistant", "content": bot_msg})
            messages.append(correction_prompt)

            with st.spinner("Detected invalid functions. Requesting correction..."):
                correction_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=2048
                )
                bot_msg = correction_response.choices[0].message.content
                st.success("✅ Self-correction applied.")



        st.chat_message("assistant").markdown(bot_msg)
        st.session_state.messages.append({"role": "assistant", "content": bot_msg})