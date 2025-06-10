# 2. retriever.py
import json, os
import numpy as np
import streamlit as st
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@st.cache_data(show_spinner=False)
def load_chunks_and_embeddings():
    with open("psse_examples_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = []
    for chunk in chunks:
        try:
            response = client.embeddings.create(model="text-embedding-3-small", input=chunk["text"][:8192])
            embeddings.append(response.data[0].embedding)
        except:
            embeddings.append(None)
    chunks, embeddings = zip(*[(c, e) for c, e in zip(chunks, embeddings) if e])
    return list(chunks), np.array(embeddings)

def find_relevant_chunks(query, chunks, embeddings, k=25):
    response = client.embeddings.create(model="text-embedding-3-small", input=query)
    query_embed = np.array(response.data[0].embedding).reshape(1, -1)
    scores = cosine_similarity(query_embed, embeddings).flatten()
    return [chunks[i] for i in scores.argsort()[-k:][::-1]]