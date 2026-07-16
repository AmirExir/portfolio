"""ERCOT DWG/SSWG assistant using the central RAG index."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import streamlit as st
from openai import OpenAI

try:
    from ERCOTAPI.rag_ingestion.retrieval import (
        format_context,
        format_source_list,
        index_state,
        load_collection,
        retrieve_chunks,
    )
except ModuleNotFoundError as exc:
    if exc.name != "ERCOTAPI":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ERCOTAPI.rag_ingestion.retrieval import (
        format_context,
        format_source_list,
        index_state,
        load_collection,
        retrieve_chunks,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_CHUNKS = REPO_ROOT / "chatbot_ercot_all_in_one" / "ercot_chunks_cached.json"
LEGACY_EMBEDDINGS = REPO_ROOT / "chatbot_ercot_all_in_one" / "ercot_embeddings.npy"


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return OpenAI(api_key=api_key)


def _legacy_state() -> tuple[int, int]:
    return (
        LEGACY_CHUNKS.stat().st_mtime_ns if LEGACY_CHUNKS.exists() else 0,
        LEGACY_EMBEDDINGS.stat().st_mtime_ns if LEGACY_EMBEDDINGS.exists() else 0,
    )


def _bootstrap_central_index() -> None:
    from ERCOTAPI.rag_ingestion.pipeline import IngestionPipeline

    IngestionPipeline().update()


@st.cache_resource(show_spinner=False, max_entries=1)
def load_dwg_sswg_index(cache_key: tuple[object, ...]):
    del cache_key
    try:
        return load_collection(
            "dwg_sswg",
            allow_legacy=False,
            legacy_chunks_path=LEGACY_CHUNKS,
            legacy_embeddings_path=LEGACY_EMBEDDINGS,
            legacy_embedding_model="text-embedding-3-large",
        )
    except FileNotFoundError:
        _bootstrap_central_index()
        return load_collection(
            "dwg_sswg",
            allow_legacy=False,
            legacy_chunks_path=LEGACY_CHUNKS,
            legacy_embeddings_path=LEGACY_EMBEDDINGS,
            legacy_embedding_model="text-embedding-3-large",
        )


def extract_function_names(chunks) -> set[str]:
    functions: set[str] = set()
    for chunk in chunks:
        functions.update(re.findall(r"\bpsspy\.(\w+)\b", str(chunk.get("text", ""))))
    return functions


def find_invalid_functions(response_text: str, valid_functions: set[str]) -> list[str]:
    return [
        function
        for function in re.findall(r"\bpsspy\.(\w+)\b", response_text)
        if function not in valid_functions
    ]


st.set_page_config(page_title="Amir Exir's DWG/SSWG AI Assistant", page_icon="⚡")
st.title("Ask Amir Exir's DWG/SSWG AI Assistant")

with st.spinner("Loading the DWG/SSWG index..."):
    rag_index = load_dwg_sswg_index((*index_state(), *_legacy_state()))
valid_functions = extract_function_names(rag_index.chunks)

with st.sidebar:
    st.caption(f"Loaded {len(rag_index.chunks)} DWG/SSWG chunks")
    st.caption(f"Generation: {rag_index.generation_id}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

if prompt := st.chat_input("Ask about ERCOT DWG and SSWG manuals..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        try:
            matches = retrieve_chunks(prompt, rag_index, top_k=50)
        except Exception as exc:
            st.error(f"Retrieval failed: {exc}")
            st.stop()
        context = format_context(matches, max_words=100000)
        system_prompt = {
            "role": "system",
            "content": f"""
You are an advanced ERCOT Dynamics Working Group and Steady-State Working Group
manual expert. Use only the cited official context below. Avoid made-up
explanations or function names, and retain the citations supporting your answer.

---
{context}
---
""",
        }
        response = get_openai_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[system_prompt] + st.session_state.messages,
            max_tokens=8192,
            temperature=0.2,
        )
        bot_message = response.choices[0].message.content or ""
        invalid_functions = find_invalid_functions(bot_message, valid_functions)
        if invalid_functions:
            warning = ", ".join(sorted(set(invalid_functions)))
            st.warning(f"These PSS/E functions were not found in the indexed documentation: {warning}")
            bot_message += f"\n\n*Caution: these functions were not found in the indexed sources: {warning}.*"

        bot_message = bot_message.rstrip() + format_source_list(matches)

        st.chat_message("assistant").markdown(bot_message)
        st.session_state.messages.append({"role": "assistant", "content": bot_message})
