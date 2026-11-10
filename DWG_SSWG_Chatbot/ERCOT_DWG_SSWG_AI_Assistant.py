"""ERCOT DWG/SSWG assistant using the central RAG index."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from openai import OpenAI

try:
    from ERCOTAPI.rag_ingestion.claim_verification import verify_answer
    from ERCOTAPI.rag_ingestion.retrieval import (
        format_context,
        format_source_list,
        index_state,
        load_index,
        prepare_evidence,
        retrieve_chunks,
    )
    from ERCOTAPI.rag_ingestion.response_handling import (
        ANSWER_VERIFICATION_POLICY,
        compact_chat_messages,
        retain_verified_history,
    )
except ModuleNotFoundError as exc:
    if exc.name != "ERCOTAPI":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ERCOTAPI.rag_ingestion.claim_verification import verify_answer
    from ERCOTAPI.rag_ingestion.retrieval import (
        format_context,
        format_source_list,
        index_state,
        load_index,
        prepare_evidence,
        retrieve_chunks,
    )
    from ERCOTAPI.rag_ingestion.response_handling import (
        ANSWER_VERIFICATION_POLICY,
        compact_chat_messages,
        retain_verified_history,
    )

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

Cite every material statement with its supplied evidence ID, such as [E1].

---
{context}
---
""",
    }


st.set_page_config(page_title="Amir Exir's DWG/SSWG AI Assistant", page_icon="⚡")
st.title("Ask Amir Exir's DWG/SSWG AI Assistant")

with st.spinner("Loading the saved central DWG/SSWG index..."):
    try:
        rag_index = load_dwg_sswg_index(index_state())
    except Exception as exc:
        st.error(str(exc))
        st.stop()

with st.sidebar:
    st.caption(f"Loaded {len(rag_index.chunks)} DWG/SSWG chunks")
    st.caption(f"Central generation: {rag_index.generation_id}")
    st.caption("Read-only saved index; questions never rebuild document embeddings.")

if "messages" not in st.session_state:
    st.session_state.messages = []
st.session_state.messages = retain_verified_history(st.session_state.messages)

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
        answer_evidence = prepare_evidence(matches, max_words=20_000)
        context = format_context(answer_evidence)
        conversation_messages = compact_chat_messages(
            st.session_state.messages,
            max_messages=6,
            max_characters_per_message=6_000,
        )
        client = get_openai_client()
        try:
            response = client.chat.completions.create(
                model="gpt-6-astra",
                messages=[build_system_prompt(context)] + conversation_messages,
                max_completion_tokens=4_096,
                reasoning_effort="low",
            )
            response_assessment = assess_chat_completion(response)
            if response_assessment.retryable:
                answer_evidence = prepare_evidence(matches[:6], max_words=6_000)
                retry_context = format_context(answer_evidence)
                retry_response = client.chat.completions.create(
                    model="gpt-6-astra",
                    messages=[
                        build_system_prompt(retry_context),
                        *conversation_messages,
                    ],
                    max_completion_tokens=4_096,
                    reasoning_effort="low",
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

        try:
            with st.spinner("Checking claims against cited sources..."):
                verified_answer = verify_answer(
                    response_assessment.text,
                    answer_evidence,
                    client=client,
                    question=prompt,
                    as_of=answer_evidence[0].get("as_of") if answer_evidence else None,
                )
        except Exception:
            st.error("Source verification could not complete. No generated answer was shown. Please retry.")
            st.stop()
        if not verified_answer.text.strip():
            st.error("Source verification returned no usable answer. Please retry.")
            st.stop()
        bot_message = verified_answer.text
        if verified_answer.notice:
            bot_message += "\n\n_" + verified_answer.notice + "_"
        bot_message += format_source_list(
            verified_answer.cited_chunks,
            max_sources=len(verified_answer.cited_chunks),
        )

        st.chat_message("assistant").markdown(bot_message)
        st.session_state.messages.append({
            "role": "assistant",
            "content": bot_message,
            "verification_policy": ANSWER_VERIFICATION_POLICY,
        })
