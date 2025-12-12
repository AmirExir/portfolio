# utils.py
import re
import numpy as np
import tiktoken
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------
# Embeddings
# ---------------------------

def embed_text(text: str, model="text-embedding-3-large"):
    """Embed a single text string."""
    try:
        response = client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"[Embedding Error]: {e}")
        return None


def embed_query(query: str, model="text-embedding-3-large"):
    """Embed a user query."""
    return embed_text(query, model=model)


# ---------------------------
# Similarity Search
# ---------------------------

def find_top_k_chunks(query, chunks, embeddings, k=10):
    """Return top-k most relevant chunks based on cosine similarity."""
    query_vec = embed_query(query)
    if query_vec is None:
        return []

    query_emb = np.array(query_vec).reshape(1, -1)

    if query_emb.shape[1] != embeddings.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: {query_emb.shape[1]} vs {embeddings.shape[1]}"
        )

    scores = cosine_similarity(query_emb, embeddings).flatten()
    top_indices = scores.argsort()[-k:][::-1]
    return [chunks[i] for i in top_indices]


# ---------------------------
# Token Utilities
# ---------------------------

def count_tokens(text: str, encoding_name="cl100k_base"):
    """Count tokens without tying to a specific model."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def limit_chunks_by_token_budget(chunks, max_tokens=30000, encoding_name="cl100k_base"):
    """Trim chunks to fit within a token budget."""
    total = 0
    selected = []

    for chunk in chunks:
        tokens = count_tokens(chunk["text"], encoding_name)
        if total + tokens > max_tokens:
            break
        selected.append(chunk)
        total += tokens

    return selected


# ---------------------------
# Static Parsing / Validation
# ---------------------------

def extract_psspy_function_names(chunks):
    """Extract all psspy.* function names from documentation chunks."""
    pattern = r'\bpsspy\.(\w+)\b'
    funcs = set()

    for chunk in chunks:
        funcs.update(re.findall(pattern, chunk["text"]))

    return funcs