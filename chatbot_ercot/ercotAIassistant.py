"""ERCOT Planning Guide assistant backed by the central planning collection."""

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


REPO_ROOT = Path(__file__).resolve().parents[1]

HIDE_STREAMLIT_UI = """
<style>
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    [data-testid="stFooter"] { display: none; }
    [data-testid="stHeader"] { display: none; }
    [data-testid="stToolbar"] { display: none; }
</style>
"""


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return OpenAI(api_key=api_key)


@st.cache_resource(show_spinner=False, max_entries=1)
def load_planning_index(cache_key: tuple[object, ...]):
    del cache_key
    return load_startup_index("planning")


def get_loaded_sources(chunks) -> list[str]:
    sources = sorted(
        {
            str(chunk.get("source_path") or chunk.get("source") or "")
            for chunk in chunks
            if isinstance(chunk, dict)
        }
    )
    return [source for source in sources if source]


def is_meta_visibility_question(prompt: str) -> bool:
    lowered = (prompt or "").strip().lower()
    return any(
        trigger in lowered
        for trigger in ("can you see", "which files", "what files", "planning guides do you have", "what documents")
    )


def is_last_read_question(prompt: str) -> bool:
    lowered = (prompt or "").strip().lower()
    return any(
        trigger in lowered
        for trigger in (
            "what is the last thing you can read",
            "last thing you can read",
            "last paragraph you can read",
            "what are the last 3 words",
            "what are the last three words",
        )
    )


def _last_paragraph(text: str) -> str:
    parts = [part.strip() for part in (text or "").replace("\r\n", "\n").split("\n\n") if part.strip()]
    return parts[-1] if parts else (text or "").strip()


def _last_words(text: str, count: int) -> str:
    words = (text or "").split()
    return " ".join(words[-count:]) if words else ""


def build_last_read_answer(chunks, sources: list[str]) -> str:
    lines = ["Here’s the last indexed content I can read from each loaded source:", ""]
    chunks_by_source: dict[str, list[dict]] = {}
    for chunk in chunks:
        source = str(chunk.get("source_path") or chunk.get("source") or "")
        if source:
            chunks_by_source.setdefault(source, []).append(chunk)

    for source in sources:
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = REPO_ROOT / source_path
        raw_text = ""
        if source_path.exists() and source_path.is_file():
            try:
                raw_text = source_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                raw_text = ""
        if not raw_text:
            candidates = chunks_by_source.get(source, [])
            if candidates:
                last_chunk = max(candidates, key=lambda item: int(item.get("chunk_index", -1)))
                raw_text = str(last_chunk.get("text", ""))

        lines.extend(
            (
                f"File: {source}",
                f"Last paragraph: {_last_paragraph(raw_text) or '(no content found)'}",
                f"Last 10 words: {_last_words(raw_text, 10) or '(no content found)'}",
                "",
            )
        )
    return "\n".join(lines).strip()


st.set_page_config(page_title="Amir Exir's ERCOT Planning Guides AI Assistant", page_icon="⚡")
st.markdown(HIDE_STREAMLIT_UI, unsafe_allow_html=True)
st.title("Ask Amir Exir's ERCOT Planning Guides AI Assistant")

with st.spinner("Loading the saved central Planning Guide index..."):
    try:
        rag_index = load_planning_index(startup_index_state())
    except CentralIndexUnavailable as exc:
        st.error(str(exc))
        st.stop()
chunks = rag_index.chunks
sources = get_loaded_sources(chunks)

with st.sidebar:
    st.markdown("### Loaded planning sources")
    st.caption(f"Central generation: {rag_index.generation_id}")
    for source in sources:
        st.markdown(f"- {source}")
    if not sources:
        st.markdown("No planning sources are available.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

if prompt := st.chat_input("Ask about ERCOT planning guides..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if is_meta_visibility_question(prompt):
        answer = (
            "These planning sources are currently indexed:\n\n"
            + "\n".join(f"- {source}" for source in sources)
            if sources
            else "No planning sources are currently indexed."
        )
        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.stop()

    if is_last_read_question(prompt):
        answer = build_last_read_answer(chunks, sources)
        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.stop()

    with st.spinner("Thinking..."):
        try:
            matches = retrieve_chunks(prompt, rag_index, top_k=5)
        except Exception as exc:
            st.error(f"Retrieval failed: {exc}")
            st.stop()
        context = format_context(matches, max_words=12000)
        system_prompt = {
            "role": "system",
            "content": f"""
You are an expert assistant on ERCOT's Planning Guides. Answer only from the
cited documentation below. Do not guess or use outside knowledge. If the answer
is not explicitly present, respond exactly: "I couldn’t find that in the
documentation." Preserve relevant citations in the answer.

---
{context}
---
""",
        }
        response = get_openai_client().responses.create(
            model="gpt-5.2",
            reasoning={"effort": "high"},
            input=[system_prompt] + st.session_state.messages,
            max_output_tokens=1000,
        )
        bot_message = response.output_text.rstrip() + format_source_list(matches)
        st.chat_message("assistant").markdown(bot_message)
        st.session_state.messages.append({"role": "assistant", "content": bot_message})
