import streamlit as st
import os
import json
import time
import re
import numpy as np
from typing import List
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# OpenAI client
# ---------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------
# Retry-safe OpenAI call
# ---------------------------
def safe_openai_call(api_function, max_retries=5, backoff_factor=2, **kwargs):
    retries = 0
    while retries < max_retries:
        try:
            return api_function(**kwargs)
        except Exception as e:
            # Rate limit / transient errors often show up as generic Exception in some environments
            wait_time = backoff_factor ** retries
            if retries < max_retries - 1:
                st.warning(f"API call failed (attempt {retries+1}/{max_retries}): {e}\nRetrying in {wait_time}s...")
                time.sleep(wait_time)
                retries += 1
                continue
            st.error(f"API call failed (final): {e}")
            return None
    return None

# ---------------------------
# Robust Responses API extractor
# ---------------------------
def extract_response_text(response) -> str:
    # Fast path: some SDK versions provide this
    direct = getattr(response, "output_text", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    texts = []
    output_items = getattr(response, "output", None) or []
    for item in output_items:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", None) or []:
            t = getattr(content, "text", None)
            if isinstance(t, str) and t.strip():
                texts.append(t.strip())

    return "\n".join(texts).strip()

def response_debug_dict(response) -> dict:
    """
    Try to turn the response into something printable for debugging.
    """
    for fn in ("model_dump", "to_dict", "dict"):
        m = getattr(response, fn, None)
        if callable(m):
            try:
                return m()
            except Exception:
                pass
    # fallback: shallow introspection
    return {
        "has_output_text": hasattr(response, "output_text"),
        "output_text": getattr(response, "output_text", None),
        "has_output": hasattr(response, "output"),
        "output_len": len(getattr(response, "output", []) or []),
        "output_types": [getattr(x, "type", None) for x in (getattr(response, "output", []) or [])],
    }

# ---------------------------
# Load / cache embeddings
# ---------------------------
@st.cache_data(show_spinner=False)
def load_psse_chunks_and_embeddings():
    base_path = os.path.dirname(__file__)
    cached_emb = os.path.join(base_path, "psse_embeddings.npy")
    cached_chunks = os.path.join(base_path, "psse_chunks_cached.json")
    input_file = os.path.join(base_path, "input_chunks.json")

    if os.path.exists(cached_emb) and os.path.exists(cached_chunks):
        with open(cached_chunks, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        embeddings = np.load(cached_emb)
        return list(chunks), embeddings

    # compute new embeddings
    with open(input_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = []
    for i, chunk in enumerate(chunks):
        resp = safe_openai_call(
            client.embeddings.create,
            model="text-embedding-3-large",
            input=chunk["text"][:8192]
        )
        embeddings.append(resp.data[0].embedding if resp else None)

    valid_pairs = [(c, e) for c, e in zip(chunks, embeddings) if e is not None]
    if not valid_pairs:
        raise ValueError("No valid embeddings were generated.")

    chunks2, emb2 = zip(*valid_pairs)
    emb2 = np.array(emb2)

    # ✅ IMPORTANT FIX: save to the SAME directory (base_path)
    np.save(cached_emb, emb2)
    with open(cached_chunks, "w", encoding="utf-8") as f:
        json.dump(list(chunks2), f, indent=2)

    return list(chunks2), emb2

# ---------------------------
# Retrieval
# ---------------------------
def embed_query(query: str) -> List[float]:
    resp = safe_openai_call(
        client.embeddings.create,
        model="text-embedding-3-large",
        input=query
    )
    return resp.data[0].embedding if resp else []

def find_top_k_matches(query: str, chunks, embeddings, k=10):
    q = embed_query(query)
    if not q:
        return []
    scores = cosine_similarity(np.array(q).reshape(1, -1), embeddings).flatten()
    idx = scores.argsort()[-k:][::-1]
    return [chunks[i] for i in idx]

def limit_chunks_by_token_budget(chunks, max_words=50000):
    # keep your rough limiter (works), but make it deterministic
    total = 0
    selected = []
    for c in chunks:
        w = len(c["text"].split())
        if total + w > max_words:
            break
        selected.append(c)
        total += w
    return selected

def extract_function_names(chunks):
    funcs = set()
    for c in chunks:
        funcs.update(re.findall(r"\bpsspy\.(\w+)\b", c["text"]))
    return funcs

def find_invalid_functions(response_text, valid_funcs):
    used = re.findall(r"\bpsspy\.(\w+)\b", response_text)
    return [f for f in used if f not in valid_funcs]

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Amir Exir's PSSE Automation Assistant", page_icon="⚡")
st.title("Ask Amir Exir's PSSE Automation Assistant")

with st.spinner("Loading PSS/E documentation..."):
    chunks, embeddings = load_psse_chunks_and_embeddings()

valid_funcs = extract_function_names(chunks)

# Debug toggle
DEBUG = st.sidebar.checkbox("Debug mode", value=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    st.chat_message(m["role"]).markdown(m["content"])

# ---------------------------
# Chat
# ---------------------------
if prompt := st.chat_input("Ask about PSS/E automation, code generation, or API usage..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        top_chunks = find_top_k_matches(prompt, chunks, embeddings, k=10)
        trimmed = limit_chunks_by_token_budget(top_chunks, max_words=40000)
        context = "\n\n---\n\n".join(c["text"] for c in trimmed)

        system_prompt = {
            "role": "system",
            "content": f"""
You are a PSS/E Python automation expert.

You MUST always produce a final answer with working Python code.
Use only psspy functions that appear in the provided reference chunks.
If the reference chunks are insufficient, say so and return a best-practice skeleton.

Reference documentation chunks:
---
{context}
---
"""
        }

        messages = [system_prompt] + st.session_state.messages

        response = safe_openai_call(
            client.responses.create,
            model="gpt-5.2",                 # ✅ only model used for generation
            reasoning={"effort": "high"},
            input=messages,
            max_output_tokens=2048
        )

        if response is None:
            st.error("OpenAI call failed (response is None).")
            st.stop()

        bot_msg = extract_response_text(response)

        # ✅ If empty, show debug dump so we can see what the model returned
        if not bot_msg:
            st.error("Model returned no text output.")
            if DEBUG:
                st.subheader("Debug: raw response")
                st.json(response_debug_dict(response))
            st.stop()

        # Optional invalid-function correction
        invalid = find_invalid_functions(bot_msg, valid_funcs)
        if invalid:
            st.warning(f"Possible invalid psspy functions: {', '.join(invalid)}")
            correction_prompt = {
                "role": "user",
                "content": (
                    f"You used invalid function(s): {', '.join(invalid)}. "
                    "Revise using ONLY valid psspy functions found in the reference chunks."
                )
            }
            messages2 = messages + [{"role": "assistant", "content": bot_msg}] + [correction_prompt]
            corr = safe_openai_call(
                client.responses.create,
                model="gpt-5.2",
                reasoning={"effort": "high"},
                input=messages2,
                max_output_tokens=2048
            )
            if corr:
                corrected = extract_response_text(corr)
                if corrected:
                    bot_msg = corrected

        st.chat_message("assistant").markdown(bot_msg)
        st.session_state.messages.append({"role": "assistant", "content": bot_msg})