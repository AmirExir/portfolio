import streamlit as st
import os
import json
import openai
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

# === Safe OpenAI wrapper ===
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

# === Streamlit page config ===
st.set_page_config(page_title="ERCOT Assistant", page_icon="⚡")
st.title("⚡ Ask Amir Exir's DWG, SSWG, Nodal Protocols, Planning Guides, Resource Integration ERCOT AI Assistant")

# === Load API key ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Load ERCOT chunks and embeddings ===
@st.cache_data(show_spinner=False)
def load_ercot_chunks_and_embeddings():
    base_path = os.path.dirname(__file__)
    cached_emb = os.path.join(base_path, "ercot_embeddings.npy")
    cached_chunks = os.path.join(base_path, "ercot_chunks_cached.json")

    if os.path.exists(cached_emb) and os.path.exists(cached_chunks):
        with open(cached_chunks, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        embeddings = np.load(cached_emb)
        return list(chunks), embeddings
    else:
        st.error("❌ Missing cached ERCOT embeddings or chunks.")
        raise FileNotFoundError("Missing ERCOT files.")

chunks, embeddings = load_ercot_chunks_and_embeddings()

# Show file info
base_path = os.path.dirname(__file__)
json_path = os.path.join(base_path, "ercot_chunks_cached.json")
npy_path = os.path.join(base_path, "ercot_embeddings.npy")

json_size = os.path.getsize(json_path) / (1024 * 1024)  # in MB
npy_size = os.path.getsize(npy_path) / (1024 * 1024)    # in MB

st.success(f"✅ Loaded embeddings from: `{npy_path}` ({npy_size:.1f} MB)")
st.success(f"✅ Loaded chunks from: `{json_path}` ({json_size:.1f} MB)")
st.info(f"🔢 Embedding shape: `{embeddings.shape}`, Total chunks: `{len(chunks)}`")
# === Embed user query ===
def embed_query(query: str):
    response = safe_openai_call(
        client.embeddings.create,
        model="text-embedding-3-large",
        input=query
    )
    return response.data[0].embedding if response else []

# === Search top-k matching chunks ===
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

# === Limit by token budget ===
def limit_chunks_by_token_budget(chunks, max_input_tokens=100000):
    total = 0
    selected = []
    for chunk in chunks:
        token_count = len(chunk["text"].split())
        if total + token_count > max_input_tokens:
            break
        selected.append(chunk)
        total += token_count
    return selected

# === Streamlit chat state ===
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous chat
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# === Chat input ===
if prompt := st.chat_input("Ask a question about ERCOT DWG, SSWG,protocols, planning, or interconnection..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        top_chunks = find_top_k_matches(prompt, chunks, embeddings, k=5)
        trimmed_chunks = limit_chunks_by_token_budget(top_chunks)
        combined_context = "\n\n---\n\n".join(chunk["text"] for chunk in trimmed_chunks)

        system_prompt = {
            "role": "system",
            "content": f"""
        You are an ERCOT regulatory expert trained only on the following documents: ERCOT Nodal protocols, planning guides, interconnection handbook and QSA checklist, SSWG and DWG manuals and working group procedures.

        Answer the user's question **only using the text provided below**. 
        - Do **not make up any information**.
        - If the answer is **not explicitly stated**, say: "The documents do not contain that information."
        - **Do not guess** or generate hypothetical information.

        ---
        {combined_context}
        ---
        """
        }

        messages = [system_prompt] + st.session_state.messages

        response = client.chat.completions.create(
            model="gpt-5o",
            messages=messages,
            max_tokens=10000,
            temperature=0.2,
            
        )

        bot_msg = response.choices[0].message.content
        st.chat_message("assistant").markdown(bot_msg)
        st.session_state.messages.append({"role": "assistant", "content": bot_msg})