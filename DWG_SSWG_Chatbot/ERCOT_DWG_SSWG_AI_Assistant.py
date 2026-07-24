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
        load_index,
        retrieve_chunks,
    )
    from ERCOTAPI.rag_ingestion.response_handling import compact_chat_messages
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
    from ERCOTAPI.rag_ingestion.response_handling import compact_chat_messages

try:
    from DWG_SSWG_Chatbot.chat_completion_handling import assess_chat_completion
except ModuleNotFoundError:
    from chat_completion_handling import assess_chat_completion


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return OpenAI(api_key=api_key)


@st.cache_resource(show_spinner=False, max_entries=1)
def load_dwg_sswg_index(cache_key: tuple[object, ...]):
    del cache_key
    index = load_index("dwg_sswg", allow_legacy=False)
    if index.source != "central" or not index.ready:
        raise RuntimeError(
            "No saved central DWG/SSWG index is available. "
            "Run the separate ingestion job before starting this assistant."
        )
    return index


def build_system_prompt(context: str) -> dict[str, str]:
    return {
        "role": "system",
        "content": f"""
You are an advanced ERCOT Dynamics Working Group and Steady-State Working Group
manual expert. Use only the cited official context below. Avoid made-up
explanations or function names, and retain the citations supporting your answer.
If the documentation does not support an answer, say so directly. Do not create
a separate source list; the application adds the retrieved sources.

---
{context}
---
""",
    }


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

with st.spinner("Loading the saved central DWG/SSWG index..."):
    try:
        rag_index = load_dwg_sswg_index(index_state())
    except Exception as exc:
        st.error(str(exc))
        st.stop()
valid_functions = extract_function_names(rag_index.chunks)

with st.sidebar:
    st.caption(f"Loaded {len(rag_index.chunks)} DWG/SSWG chunks")
    st.caption(f"Central generation: {rag_index.generation_id}")
    st.caption("Read-only saved index; questions never rebuild document embeddings.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

if prompt := st.chat_input("Ask about ERCOT DWG and SSWG manuals..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        try:
            matches = retrieve_chunks(prompt, rag_index, top_k=16)
        except Exception as exc:
            st.error(f"Retrieval failed: {exc}")
            st.stop()
        if not matches:
            st.error("No matching DWG/SSWG documentation was found.")
            st.stop()
        context = format_context(matches, max_words=20_000)
        conversation_messages = compact_chat_messages(
            st.session_state.messages,
            max_messages=6,
            max_characters_per_message=6_000,
        )
        client = get_openai_client()
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[build_system_prompt(context)] + conversation_messages,
                max_tokens=4_096,
                temperature=0.2,
            )
            response_assessment = assess_chat_completion(response)
            if response_assessment.retryable:
                retry_context = format_context(matches[:6], max_words=6_000)
                retry_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        build_system_prompt(retry_context),
                        *conversation_messages,
                    ],
                    max_tokens=4_096,
                    temperature=0.2,
                )
                response_assessment = assess_chat_completion(retry_response)
        except Exception as exc:
            st.error(f"Answer generation failed: {exc}")
            st.stop()

        if not response_assessment.usable:
            st.error(
                "Answer generation did not complete, so no source-only result was shown. "
                f"Details: {response_assessment.diagnostic}. Please retry the question."
            )
            st.stop()

        bot_message = response_assessment.text
        invalid_functions = find_invalid_functions(bot_message, valid_functions)
        if invalid_functions:
            warning = ", ".join(sorted(set(invalid_functions)))
            st.warning(f"These PSS/E functions were not found in the indexed documentation: {warning}")
            bot_message += f"\n\n*Caution: these functions were not found in the indexed sources: {warning}.*"

        bot_message = bot_message.rstrip() + format_source_list(matches)

        st.chat_message("assistant").markdown(bot_message)
        st.session_state.messages.append({"role": "assistant", "content": bot_message})
