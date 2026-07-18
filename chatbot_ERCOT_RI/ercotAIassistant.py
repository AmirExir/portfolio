"""ERCOT Resource Integration assistant using the central RAG index."""

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


@st.cache_resource(show_spinner=False, max_entries=1)
def load_resource_integration_index(cache_key: tuple[object, ...]):
    del cache_key
    return load_startup_index("resource_integration")


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


st.set_page_config(page_title="Amir Exir's Resource Integration AI Assistant", page_icon="⚡")
st.title("Ask Amir Exir's Resource Integration AI Assistant")

with st.spinner("Loading or bootstrapping the central Resource Integration index..."):
    try:
        rag_index = load_resource_integration_index(startup_index_state())
    except CentralIndexUnavailable as exc:
        st.error(str(exc))
        st.stop()
valid_functions = extract_function_names(rag_index.chunks)

with st.sidebar:
    st.caption(f"Loaded {len(rag_index.chunks)} Resource Integration chunks")
    st.caption(f"Central generation: {rag_index.generation_id}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

if prompt := st.chat_input("Ask about ERCOT Resource Integration or the QSA process..."):
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
You are an advanced ERCOT Resource Integration expert. Use only the cited
handbook and official ERCOT context below. Avoid made-up explanations and retain
the citations supporting your answer.

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
        bot_message = response.output_text
        invalid_functions = find_invalid_functions(bot_message, valid_functions)
        if invalid_functions:
            warning = ", ".join(sorted(set(invalid_functions)))
            st.warning(f"These PSS/E functions were not found in the indexed documentation: {warning}")
            bot_message += f"\n\n*Caution: these functions were not found in the indexed sources: {warning}.*"

        bot_message = bot_message.rstrip() + format_source_list(matches)

        st.chat_message("assistant").markdown(bot_message)
        st.session_state.messages.append({"role": "assistant", "content": bot_message})
