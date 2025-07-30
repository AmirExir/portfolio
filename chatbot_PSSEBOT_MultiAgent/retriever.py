import os
import json
import streamlit as st
from utils import embed_query, find_top_k_chunks
import numpy as np
from openai import OpenAI
import tiktoken

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@st.cache_data(show_spinner=False)
def load_chunks_and_embeddings(json_file="input_chunks.json", embedding_model="text-embedding-3-large"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(current_dir, json_file)
    print(f"📂 Reading file from: {full_path}")

    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"❌ File not found: {full_path}")

    # Load cached if exists
    cached_emb = os.path.join(current_dir, "psse_embeddings.npy")
    cached_chunks = os.path.join(current_dir, "psse_chunks_cached.json")
    if os.path.exists(cached_emb) and os.path.exists(cached_chunks):
        st.write("✅ Using precomputed embeddings from .npy and .json")
        with open(cached_chunks, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        embeddings = np.load(cached_emb)
        print("✅ Loaded cached embeddings.")
        return list(chunks), embeddings

    # Else compute and save
    with open(full_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = []
    for chunk in chunks:
        try:
            response = client.embeddings.create(model=embedding_model, input=chunk["text"][:8192])
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            print(f"⚠️ Embedding failed for chunk: {e}")
            embeddings.append(None)
            temperature = 0.0,

    final_pairs = [(c, e) for c, e in zip(chunks, embeddings) if e is not None]
    if not final_pairs:
        raise ValueError("No usable chunks found after embedding.")

    final_chunks, final_embeddings = zip(*final_pairs)

    # Save embeddings for reuse
    with open(cached_chunks, "w", encoding="utf-8") as f:
        json.dump(list(final_chunks), f, indent=2)
    np.save(cached_emb, final_embeddings)
    print("💾 Embeddings saved for future use.")

    return list(final_chunks), np.array(final_embeddings)

def find_relevant_chunks(query, chunks, embeddings, k=50):
    query_embed = embed_query(query)
    return find_top_k_chunks(query, chunks, embeddings, k)

def limit_chunks_by_token_budget(chunks, max_tokens=30000, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    total = 0
    selected = []
    for chunk in chunks:
        tokens = len(encoding.encode(chunk["text"]))
        if tokens > 4000:
            continue
        if total + tokens > max_tokens:
            break
        selected.append(chunk)
        total += tokens
    return selected