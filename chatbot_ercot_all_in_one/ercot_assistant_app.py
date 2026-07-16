"""Combined ERCOT Streamlit assistant using the central RAG collections."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import openai
import streamlit as st
from openai import OpenAI

try:
    from ERCOTAPI.rag_ingestion.retrieval import (
        format_context,
        format_source_list,
        index_state,
        load_index,
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
        load_index,
        retrieve_chunks,
    )


BASE_DIR = Path(__file__).resolve().parent
LEGACY_CHUNKS = BASE_DIR / "ercot_chunks_cached.json"
LEGACY_EMBEDDINGS = BASE_DIR / "ercot_embeddings.npy"


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return OpenAI(api_key=api_key)


def safe_openai_call(api_function, max_retries=5, backoff_factor=2, **kwargs):
    retries = 0
    while retries < max_retries:
        try:
            return api_function(**kwargs)
        except openai.RateLimitError:
            wait_time = backoff_factor**retries
            st.warning(f"Rate limit hit. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
        except Exception as exc:
            st.error(f"API call failed: {exc}")
            break
    return None


def _legacy_state() -> tuple[int, int]:
    return (
        LEGACY_CHUNKS.stat().st_mtime_ns if LEGACY_CHUNKS.exists() else 0,
        LEGACY_EMBEDDINGS.stat().st_mtime_ns if LEGACY_EMBEDDINGS.exists() else 0,
    )


def _bootstrap_central_index() -> None:
    from ERCOTAPI.rag_ingestion.pipeline import IngestionPipeline

    IngestionPipeline().update()


@st.cache_resource(show_spinner=False, max_entries=1)
def load_ercot_index(cache_key: tuple[object, ...]):
    del cache_key
    try:
        return load_index(
            "general",
            allow_legacy=False,
            legacy_chunks_path=LEGACY_CHUNKS,
            legacy_embeddings_path=LEGACY_EMBEDDINGS,
            legacy_embedding_model="text-embedding-3-large",
        )
    except FileNotFoundError:
        _bootstrap_central_index()
        return load_index(
            "general",
            allow_legacy=False,
            legacy_chunks_path=LEGACY_CHUNKS,
            legacy_embeddings_path=LEGACY_EMBEDDINGS,
            legacy_embedding_model="text-embedding-3-large",
        )


st.set_page_config(page_title="ERCOT Assistant", page_icon="⚡")
st.title("Ask Amir Exir's DWG, SSWG, Nodal Protocols, Planning Guides, Resource Integration ERCOT AI Assistant")

with st.spinner("Loading the ERCOT knowledge index..."):
    rag_index = load_ercot_index((*index_state(), *_legacy_state()))

with st.sidebar:
    st.caption(f"Loaded {len(rag_index.chunks)} ERCOT chunks")
    st.caption(f"Index: {rag_index.generation_id} ({rag_index.source})")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

if prompt := st.chat_input("Ask a question about ERCOT DWG, SSWG, protocols, planning, or interconnection..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        try:
            top_chunks = retrieve_chunks(prompt, rag_index, top_k=12)
        except Exception as exc:
            st.error(f"Retrieval failed: {exc}")
            st.stop()
        context = format_context(top_chunks, max_words=100000)
        if not context:
            st.error("No matching ERCOT documentation was found in the loaded index.")
            st.stop()

        system_prompt = {
            "role": "system",
            "content": f"""
You are an ERCOT regulatory expert trained only on the supplied ERCOT documentation.

Answer the user's question only using the cited context below. Do not make up
information. If the answer is not explicitly stated, say: "The documents do
not contain that information." Retain the supplied citations in your answer.

---
{context}
---
""",
        }
        client = get_openai_client()
        response = safe_openai_call(
            client.responses.create,
            model="gpt-5.2",
            reasoning={"effort": "xhigh"},
            input=[system_prompt] + st.session_state.messages,
            max_output_tokens=10000,
        )

    if response:
        bot_message = response.output_text.rstrip() + format_source_list(top_chunks)
        st.chat_message("assistant").markdown(bot_message)
        st.session_state.messages.append({"role": "assistant", "content": bot_message})
