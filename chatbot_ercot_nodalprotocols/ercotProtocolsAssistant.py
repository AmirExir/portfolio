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
        load_index,
        retrieve_chunks,
    )
    from ERCOTAPI.rag_ingestion.response_handling import (
        assess_response,
        compact_chat_messages,
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
    from ERCOTAPI.rag_ingestion.response_handling import (
        assess_response,
        compact_chat_messages,
    )


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return OpenAI(api_key=api_key)


@st.cache_resource(show_spinner=False, max_entries=1)
def load_protocol_index(cache_key: tuple[object, ...]):
    del cache_key
    index = load_index("protocols", allow_legacy=False)
    if index.source != "central" or not index.ready:
        raise RuntimeError(
            "No saved central Nodal Protocols index is available. "
            "Run the separate ingestion job before starting this assistant."
        )
    return index


def build_system_prompt(context: str) -> dict[str, str]:
    return {
        "role": "system",
        "content": f"""
You are an expert assistant on ERCOT Nodal Protocols. Use only the cited
documentation below. Stay factual, do not guess beyond it, and retain relevant
citations in the answer. If the documentation does not support an answer, say
so directly. Do not create a separate source list; the application adds the
retrieved sources.

---
{context}
---
""",
    }


st.set_page_config(page_title="Amir Exir's ERCOT protocols AI Assistant", page_icon="⚡")
st.title("Ask Amir Exir's ERCOT Nodal protocols AI Assistant")

with st.spinner("Loading the saved central Nodal Protocols index..."):
    try:
        rag_index = load_protocol_index(index_state())
    except Exception as exc:
        st.error(str(exc))
        st.stop()

with st.sidebar:
    st.caption(f"Loaded {len(rag_index.chunks)} protocol chunks")
    st.caption(f"Central generation: {rag_index.generation_id}")
    st.caption("Read-only saved index; questions never rebuild document embeddings.")

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
        if not matches:
            st.error("No matching Nodal Protocol documentation was found.")
            st.stop()
        context = format_context(matches, max_words=14_000)
        conversation_messages = compact_chat_messages(
            st.session_state.messages,
            max_messages=6,
            max_characters_per_message=6_000,
        )
        client = get_openai_client()
        try:
            response = client.responses.create(
                model="gpt-5.2",
                reasoning={"effort": "none"},
                text={"verbosity": "medium"},
                input=[build_system_prompt(context)] + conversation_messages,
                max_output_tokens=6_000,
            )
            response_assessment = assess_response(response)
            if response_assessment.retryable:
                retry_context = format_context(matches[:3], max_words=5_000)
                retry_response = client.responses.create(
                    model="gpt-5.2",
                    reasoning={"effort": "none"},
                    text={"verbosity": "medium"},
                    input=[
                        build_system_prompt(retry_context),
                        *conversation_messages,
                    ],
                    max_output_tokens=6_000,
                )
                response_assessment = assess_response(retry_response)
        except Exception as exc:
            st.error(f"Answer generation failed: {exc}")
            st.stop()

        if not response_assessment.usable:
            st.error(
                "Answer generation did not complete, so no source-only result was shown. "
                f"Details: {response_assessment.diagnostic}. Please retry the question."
            )
            st.stop()

        bot_message = response_assessment.text + format_source_list(matches)
        st.chat_message("assistant").markdown(bot_message)
        st.session_state.messages.append({"role": "assistant", "content": bot_message})
