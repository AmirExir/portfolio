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
def load_ERCOT_Nodal_Protocols_chunks_and_embeddings(force_rebuild: bool = False):
    base_dir = os.path.dirname(__file__)
    chunks_path = os.path.join(base_dir, "ercot_nodal_protocols_fully_structured.json")
    emb_path   = os.path.join(base_dir, "ercot_embeddings.npy")
    meta_path  = os.path.join(base_dir, "ercot_nodal_protocols_embeddings.meta.json")

    def _sha1_texts(texts):
        h = hashlib.sha1()
        for t in texts:
            h.update(t.replace("\r\n", "\n").encode("utf-8"))
            h.update(b"\xff")
        return h.hexdigest()

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    st.write("Current working directory:", os.getcwd())
    st.write("Chunks JSON:", chunks_path)

    embedding_model = "text-embedding-3-large"
    texts = [c["text"] for c in chunks]
    payload_hash = _sha1_texts(texts) + f"|model={embedding_model}|n={len(texts)}"

    # Load cached embeddings if valid
    if not force_rebuild and os.path.exists(emb_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as mf:
                meta = json.load(mf)
            if meta.get("payload_hash") == payload_hash:
                embeddings = np.load(emb_path)
                if embeddings.shape[0] == len(chunks):
                    st.success("✅ Loaded cached embeddings from disk.")
                    return chunks, embeddings
        except Exception as e:
            st.warning(f"Cached embeddings exist but could not be used: {e}")

    # Build embeddings once
    st.info("🔄 Building embeddings… (first run only or after content changes)")
    embeddings = []
    for i, text in enumerate(texts):
        try:
            resp = client.embeddings.create(
                model=embedding_model,
                input=text[:8192]
            )
            embeddings.append(resp.data[0].embedding)
        except Exception as e:
            st.warning(f"Embedding failed for chunk {i}: {e}")
            embeddings.append(None)

    valid_pairs = [(c, e) for c, e in zip(chunks, embeddings) if e is not None]
    if not valid_pairs:
        st.error("❌ No valid embeddings. Check API key or input.")
        raise ValueError("No valid embeddings were generated.")

    chunks, embeddings = zip(*valid_pairs)
    embeddings = np.array(embeddings)

    # Save cache
    try:
        np.save(emb_path, embeddings)
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(
                {"payload_hash": payload_hash, "model": embedding_model,
                 "num_chunks": len(chunks), "saved_at": int(time.time())},
                mf, ensure_ascii=False, indent=2
            )
        st.success(f"💾 Saved embeddings to: {emb_path}")
    except Exception as e:
        st.warning(f"Could not save embeddings cache: {e}")

    return list(chunks), embeddings

# Embed the user query
def embed_query(query: str) -> List[float]:
    response = client.embeddings.create(
        model="text-embedding-3-large",
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
st.set_page_config(page_title="Amir Exir's ERCOT Nodal Protocols AI Assistant", page_icon="⚡")
st.title("🧠 Ask Amir Exir's ERCOT Nodal Protocols AI Assistant")

# Load data and embeddings once
with st.spinner("Loading ERCOT Nodal Protocols chunks and computing embeddings..."):
    chunks, embeddings = load_ERCOT_Nodal_Protocols_chunks_and_embeddings()

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
if prompt := st.chat_input("Ask about ERCOT Nodal Protocols manuals..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        top_chunks = find_top_k_matches(prompt, chunks, embeddings, k=50)
        trimmed_chunks = limit_chunks_by_token_budget(top_chunks)
        combined_context = "\n\n---\n\n".join(chunk["text"] for chunk in trimmed_chunks)

        system_prompt = {
            "role": "system",
            "content": f"""
        You are an the most advanced ERCOT Nodal Protocols manual expert. When given a task, identify the relevant  and return a full explaination. Avoid made-up explaination. Cite the chunk you're using.

        Use only the following {len(trimmed_chunks)} reference chunks (from  manual and examples):

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
            max_tokens=8192,
            temperature = 0.2
            
        )

        bot_msg = response.choices[0].message.content

        def find_invalid_functions(response_text, valid_funcs):
            used = re.findall(r'\bpsspy\.(\w+)\b', response_text)
            return [f for f in used if f not in valid_funcs]

        invalid_funcs = find_invalid_functions(bot_msg, valid_funcs)

        if invalid_funcs:
            st.warning(f"⚠️ Warning: These functions may not exist in the : {', '.join(invalid_funcs)}")
            bot_msg += f"\n\n⚠️ *Caution: The following PSS/E  function(s) may be hallucinated or not found in the official documentation: {', '.join(invalid_funcs)}*"



        st.chat_message("assistant").markdown(bot_msg)
        st.session_state.messages.append({"role": "assistant", "content": bot_msg})