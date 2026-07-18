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
        retrieve_chunks,
    )
    from ERCOTAPI.rag_ingestion.startup import (
        CentralIndexUnavailable,
        load_startup_index,
        startup_index_state,
    )
except ModuleNotFoundError as exc:
    if exc.name != "ERCOTAPI":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ERCOTAPI.rag_ingestion.retrieval import (
        format_context,
        format_source_list,
        retrieve_chunks,
    )
    from ERCOTAPI.rag_ingestion.startup import (
        CentralIndexUnavailable,
        load_startup_index,
        startup_index_state,
    )


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


@st.cache_resource(show_spinner=False, max_entries=1)
def load_ercot_index(cache_key: tuple[object, ...]):
    del cache_key
    return load_startup_index("general")


st.set_page_config(page_title="ERCOT Assistant", page_icon="⚡")
st.title("Ask Amir Exir's DWG, SSWG, Nodal Protocols, Planning Guides, Resource Integration ERCOT AI Assistant")

with st.spinner("Loading the saved ERCOT knowledge index..."):
    try:
        rag_index = load_ercot_index(startup_index_state())
    except CentralIndexUnavailable as exc:
        st.error(str(exc))
        st.stop()

with st.sidebar:
    st.caption(f"Loaded {len(rag_index.chunks)} ERCOT chunks")
    st.caption(f"Central generation: {rag_index.generation_id}")
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
not contain that information."

Lead with the direct answer, then explain it in practical, plain language. For
a broad question about an entire guide or section, do not stop at its title or
table of contents. Explain the section's purpose and scope, its main process or
requirements, the responsibilities of the affected entities, important timing
or decision points, and the practical takeaway whenever those details appear in
the context. Use short paragraphs and bullets when they make the explanation
easier to follow.

Cite the specific supplied chunks that support important statements, but do not
use citations as a substitute for the explanation. Prefer the current effective
document unless the user clearly asks for historical material. Do not create a
separate "Retrieved sources" section; the application adds a compact source
list after the answer.

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
        bot_message = response.output_text.rstrip() + format_source_list(
            top_chunks,
            max_sources=4,
        )
        st.chat_message("assistant").markdown(bot_message)
        st.session_state.messages.append({"role": "assistant", "content": bot_message})
