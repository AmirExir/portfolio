"""ERCOT Nodal Protocols assistant using the central protocols collection."""

from __future__ import annotations

import os
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
def load_protocol_index(cache_key: tuple[object, ...]):
    del cache_key
    try:
        return load_collection(
            "protocols",
            allow_legacy=False,
            legacy_chunks_path=LEGACY_CHUNKS,
            legacy_embeddings_path=LEGACY_EMBEDDINGS,
            legacy_embedding_model="text-embedding-3-large",
        )
    except FileNotFoundError:
        _bootstrap_central_index()
        return load_collection(
            "protocols",
            allow_legacy=False,
            legacy_chunks_path=LEGACY_CHUNKS,
            legacy_embeddings_path=LEGACY_EMBEDDINGS,
            legacy_embedding_model="text-embedding-3-large",
        )


st.set_page_config(page_title="Amir Exir's ERCOT protocols AI Assistant", page_icon="⚡")
st.title("Ask Amir Exir's ERCOT Nodal protocols AI Assistant")

with st.spinner("Loading the Nodal Protocols index..."):
    rag_index = load_protocol_index((*index_state(), *_legacy_state()))

with st.sidebar:
    st.caption(f"Loaded {len(rag_index.chunks)} protocol chunks")
    st.caption(f"Generation: {rag_index.generation_id}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

if prompt := st.chat_input("Ask about ERCOT nodal protocols..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        try:
            matches = retrieve_chunks(prompt, rag_index, top_k=5)
        except Exception as exc:
            st.error(f"Retrieval failed: {exc}")
            st.stop()
        context = format_context(matches, max_words=20000)
        system_prompt = {
            "role": "system",
            "content": f"""
You are an expert assistant on ERCOT Nodal Protocols. Use only the cited
documentation below. Stay factual, do not guess beyond it, and retain relevant
citations in the answer.

---
{context}
---
""",
        }
        response = get_openai_client().responses.create(
            model="gpt-5.2",
            reasoning={"effort": "xhigh"},
            input=[system_prompt] + st.session_state.messages,
            max_output_tokens=10000,
        )
        bot_message = response.output_text.rstrip() + format_source_list(matches)
        st.chat_message("assistant").markdown(bot_message)
        st.session_state.messages.append({"role": "assistant", "content": bot_message})
