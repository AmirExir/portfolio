import streamlit as st
import os
import json
import re
import sys
from pathlib import Path
import numpy as np
from typing import List
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

_REPOSITORY_ROOT = str(Path(__file__).resolve().parents[1])
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

from psse_assistant_common import (
    compact_chat_messages,
    request_visible_answer,
    validate_saved_index,
)

# ---------------------------
# OpenAI client
# ---------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------
# Load / cache embeddings
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_psse_chunks_and_embeddings():
    base_path = os.path.dirname(__file__)
    cached_emb = os.path.join(base_path, "psse_embeddings.npy")
    cached_chunks = os.path.join(base_path, "psse_chunks_cached.json")

    missing = [
        os.path.basename(path)
        for path in (cached_emb, cached_chunks)
        if not os.path.isfile(path)
    ]
    if missing:
        raise RuntimeError(
            "Saved PSS/E RAG artifacts are missing: "
            f"{', '.join(missing)}. Runtime corpus embedding is disabled to "
            "prevent surprise API charges; build and deploy the saved index offline."
        )

    with open(cached_chunks, "r", encoding="utf-8") as f:
        chunks = list(json.load(f))
    embeddings = np.load(cached_emb)
    validate_saved_index(chunks, embeddings, expected_dimension=3072)
    return chunks, embeddings

# ---------------------------
# Retrieval
# ---------------------------
def embed_query(query: str) -> List[float]:
    try:
        resp = client.embeddings.create(
            model="text-embedding-3-large",
            input=query,
        )
    except Exception as exc:
        raise RuntimeError(
            f"query embedding request failed ({type(exc).__name__})"
        ) from exc
    return resp.data[0].embedding

def find_top_k_matches(query: str, chunks, embeddings, k=10):
    q = embed_query(query)
    if not q:
        return []
    scores = cosine_similarity(np.array(q).reshape(1, -1), embeddings).flatten()
    idx = scores.argsort()[-k:][::-1]
    return [chunks[i] for i in idx]

def limit_chunks_by_token_budget(chunks, max_words=50000):
    total = 0
    selected = []
    for c in chunks:
        w = len(c["text"].split())
        if total + w > max_words:
            break
        selected.append(c)
        total += w
    return selected


def format_reference_context(chunks):
    return "\n\n---\n\n".join(
        f"[{chunk.get('id', 'reference')}]\n{chunk['text']}"
        for chunk in chunks
    )


def build_system_prompt(context):
    return {
        "role": "system",
        "content": f"""
You are a PSS/E Python automation expert.

Always provide a direct final answer. Include working Python code when the
question asks for automation or code. Use only psspy functions that appear in
the provided reference chunks. If the reference chunks are insufficient, say
what is missing and provide a clearly labeled best-practice skeleton.

Reference documentation chunks:
---
{context}
---
""".strip(),
    }

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

try:
    with st.spinner("Loading saved PSS/E documentation index..."):
        chunks, embeddings = load_psse_chunks_and_embeddings()
except Exception as exc:
    st.error(f"PSS/E documentation index could not be loaded: {exc}")
    st.stop()

valid_funcs = extract_function_names(chunks)

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
        retrieval_error = ""
        try:
            top_chunks = find_top_k_matches(prompt, chunks, embeddings, k=10)
        except Exception as exc:
            top_chunks = []
            retrieval_error = f"{type(exc).__name__}: {exc}"
        trimmed = limit_chunks_by_token_budget(top_chunks, max_words=12_000)
        context = format_reference_context(trimmed)

        if retrieval_error or not context:
            generation = None
        else:
            conversation = compact_chat_messages(
                st.session_state.messages,
                max_messages=6,
                max_characters_per_message=4_000,
            )
            primary_request = {
                "model": "gpt-5.2",
                "reasoning": {"effort": "none"},
                "text": {"verbosity": "medium"},
                "input": [build_system_prompt(context), *conversation],
                "max_output_tokens": 6_000,
            }
            retry_chunks = limit_chunks_by_token_budget(top_chunks[:6], max_words=6_000)
            retry_context = format_reference_context(retry_chunks)
            retry_conversation = compact_chat_messages(
                st.session_state.messages,
                max_messages=2,
                max_characters_per_message=3_000,
            )
            retry_request = {
                **primary_request,
                "input": [build_system_prompt(retry_context), *retry_conversation],
            }
            generation = request_visible_answer(
                client.responses.create,
                primary_request,
                retry_request=retry_request,
            )

    if retrieval_error:
        st.error(f"PSS/E retrieval did not complete: {retrieval_error}.")
    elif not context:
        st.error("No matching PSS/E documentation was found in the saved index.")
    elif generation is None or not generation.usable:
        diagnostic = (
            generation.diagnostic
            if generation is not None
            else "the model request was not started"
        )
        st.error(
            "Answer generation did not complete, so no empty or partial answer "
            f"was shown. Details: {diagnostic}. Please retry the question."
        )
    else:
        bot_msg = generation.text
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
            correction_messages = [
                build_system_prompt(context),
                *conversation,
                {"role": "assistant", "content": bot_msg},
                correction_prompt,
            ]
            correction = request_visible_answer(
                client.responses.create,
                {
                    "model": "gpt-5.2",
                    "reasoning": {"effort": "none"},
                    "text": {"verbosity": "medium"},
                    "input": correction_messages,
                    "max_output_tokens": 6_000,
                },
            )
            if correction.usable:
                bot_msg = correction.text
            else:
                st.warning(
                    "The documented-function correction did not complete; "
                    "showing the original answer for review."
                )

        st.chat_message("assistant").markdown(bot_msg)
        st.session_state.messages.append({"role": "assistant", "content": bot_msg})
