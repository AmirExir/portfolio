"""Combined ERCOT Streamlit assistant using the central RAG collections."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import openai
import streamlit as st
from openai import OpenAI

# Streamlit Cloud can launch the script with the app directory, rather than the
# repository root, at the front of sys.path. Resolve the checked-in ERCOTAPI
# package deterministically before importing any of its submodules.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
repository_root_text = str(REPOSITORY_ROOT)
if repository_root_text in sys.path:
    sys.path.remove(repository_root_text)
sys.path.insert(0, repository_root_text)

from ERCOTAPI.latest_updates import load_latest_updates
from ERCOTAPI.rag_ingestion.config import default_config
from ERCOTAPI.rag_ingestion.retrieval import (
    format_context,
    format_change_reports,
    format_source_list,
    retrieve_requirement_evidence,
)
from ERCOTAPI.rag_ingestion.startup import (
    CentralIndexUnavailable,
    load_startup_index,
    startup_index_state,
)
from ERCOTAPI.rag_ingestion.requirements import validate_answer_citations

PACKAGED_INDEX_DIR = REPOSITORY_ROOT / "ERCOTAPI" / "deployment_rag_store"


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
    try:
        return load_startup_index(
            "general",
            bootstrap_on_missing=False,
            refresh=False,
        )
    except CentralIndexUnavailable as configured_error:
        # An old Streamlit secret may still point ERCOT_RAG_STORE at an empty
        # ephemeral directory. The checked-in deployment snapshot is complete,
        # read-only, and already embedded, so it is a safe availability
        # fallback that cannot trigger document embedding.
        packaged_config = default_config(index_dir=PACKAGED_INDEX_DIR)
        try:
            return load_startup_index(
                "general",
                config=packaged_config,
                bootstrap_on_missing=False,
                refresh=False,
            )
        except CentralIndexUnavailable as packaged_error:
            raise CentralIndexUnavailable(
                f"{configured_error} Packaged saved-index fallback also failed: "
                f"{packaged_error}"
            ) from packaged_error


def ercot_index_cache_key() -> tuple[object, ...]:
    """Track both the configured store and immutable packaged fallback."""

    packaged_config = default_config(index_dir=PACKAGED_INDEX_DIR)
    try:
        configured_state: object = startup_index_state()
    except CentralIndexUnavailable as exc:
        configured_state = ("unavailable", str(exc))
    return (
        configured_state,
        startup_index_state(packaged_config),
    )


st.set_page_config(page_title="ERCOT Assistant", page_icon="⚡")
st.markdown(
    """
    <style>
    :root {
        --ercot-blue: #0b2f4f;
        --ercot-blue-hover: #123f67;
        --ercot-teal: #0b5e75;
        --ercot-focus: #0891b2;
        --ercot-soft: #eaf4f8;
    }
    .stButton > button,
    .stLinkButton > a,
    a[data-testid^="stBaseLinkButton-"] {
        background: var(--ercot-blue) !important;
        border: 1px solid var(--ercot-blue) !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        opacity: 1 !important;
        box-shadow: 0 4px 12px rgba(11, 47, 79, 0.16) !important;
        transition: background-color 150ms ease, border-color 150ms ease,
                    box-shadow 150ms ease, transform 150ms ease !important;
    }
    .stButton > button *,
    .stLinkButton > a *,
    a[data-testid^="stBaseLinkButton-"] * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        opacity: 1 !important;
    }
    .stButton > button:hover,
    .stLinkButton > a:hover,
    a[data-testid^="stBaseLinkButton-"]:hover {
        background: var(--ercot-blue-hover) !important;
        border-color: var(--ercot-focus) !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        box-shadow: 0 7px 18px rgba(11, 47, 79, 0.24) !important;
        transform: translateY(-1px);
    }
    .stButton > button:focus-visible,
    .stLinkButton > a:focus-visible,
    a[data-testid^="stBaseLinkButton-"]:focus-visible {
        outline: 3px solid rgba(8, 145, 178, 0.35) !important;
        outline-offset: 2px !important;
        box-shadow: 0 0 0 2px #ffffff, 0 0 0 5px var(--ercot-focus) !important;
    }
    .stButton > button:active,
    .stLinkButton > a:active,
    a[data-testid^="stBaseLinkButton-"]:active {
        background: #082a46 !important;
        transform: translateY(0);
    }
    .stButton > button:disabled {
        background: #e2e8f0 !important;
        border-color: #cbd5e1 !important;
        color: #475569 !important;
        -webkit-text-fill-color: #475569 !important;
        box-shadow: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Ask Amir Exir's ERCOT Engineering & Revision Request AI Assistant")
st.caption(
    "Covers Nodal Protocols, Planning and Operating Guides, Resource Integration, "
    "DWG/SSWG procedures, OBDRRs, and ERCOT revision requests including NPRR, PGRR, "
    "NOGRR, OBDRR, RRGRR, VCMRR, COPMGRR, LPGRR, RMGRR, SMOGRR, CMGRR, and SCR materials."
)
st.caption(
    "Answers separate current governing text from procedures and proposed changes, resolve status as of the question date, "
    "and use section/page evidence IDs with direct ERCOT source links."
)

with st.spinner("Loading the saved ERCOT knowledge index..."):
    try:
        rag_index = load_ercot_index(ercot_index_cache_key())
    except CentralIndexUnavailable as exc:
        st.error(str(exc))
        st.stop()

with st.sidebar:
    st.caption(f"Loaded {len(rag_index.chunks)} ERCOT chunks")
    index_label = (
        "Packaged saved snapshot"
        if str(rag_index.generation_id or "").startswith("deployment-")
        else "Persistent central store"
    )
    st.caption(f"Index source: {index_label}")
    st.caption(f"Generation: {rag_index.generation_id}")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    with st.expander("Evidence policy"):
        st.write(
            "Current Protocol/Guide/OBD text can be governing evidence. xRRs, ballots, and committee records are shown as "
            "change evidence until incorporated text establishes the effective requirement. Operational notices are excluded."
        )

updates = load_latest_updates(
    Path(__file__).resolve().parents[1] / "ERCOTAPI" / "latest_ercot_updates.json"
)
with st.expander(f"New ERCOT documents ({updates.get('count', 0)})", expanded=False):
    st.caption("New 2026+ technical documents recently added to the searchable ERCOT index.")
    for item in list(updates.get("items") or [])[:50]:
        label = item.get("document_number") or item.get("title") or "ERCOT document"
        sources = ", ".join(item.get("sources") or [item.get("source", "ERCOT")])
        st.markdown(f"**{label}** — {sources}")
        st.write(item.get("explanation", "New ERCOT technical material."))
        if item.get("url"):
            st.link_button("Open ERCOT source", item["url"])

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

if prompt := st.chat_input("Ask about ERCOT guides, xRRs, OBDRRs, studies, or interconnections..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        try:
            evidence_bundle = retrieve_requirement_evidence(prompt, rag_index, top_k=14)
            top_chunks = evidence_bundle["chunks"]
        except Exception as exc:
            st.error(f"Retrieval failed: {exc}")
            st.stop()
        context = format_context(top_chunks, max_words=100000)
        change_context = format_change_reports(evidence_bundle.get("change_reports") or [])
        if change_context:
            context += "\n\n=== SECTION-LEVEL CHANGE REPORTS ===\n\n" + change_context
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

Cite every material statement with the supplied evidence ID, such as [E1]. Do
not use citations as a substitute for the explanation. Do not describe a
revision request, ballot, committee record, approval, or redline as an effective
requirement unless the supplied evidence also identifies incorporated governing
text. Distinguish binding/current text, related engineering procedures, pending
changes, historical material, and uncertainty. Do not create a separate
"Retrieved sources" section; the application adds the verified source list.

Answer contract for this question:
{evidence_bundle['answer_contract']}

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
        answer_text = response.output_text.rstrip()
        citation_audit = validate_answer_citations(answer_text, top_chunks)
        if not citation_audit["passed"]:
            answer_text += (
                "\n\n_Citation audit warning: the answer omitted required evidence IDs or "
                "used an ID that was not retrieved. Verify the source list before relying on it._"
            )
        cited_ids = set(citation_audit["cited_evidence_ids"])
        cited_chunks = [
            chunk for chunk in top_chunks
            if str(chunk.get("evidence_id") or "") in cited_ids
        ]
        footer_chunks = cited_chunks or top_chunks[:4]
        bot_message = answer_text + format_source_list(
            footer_chunks,
            max_sources=len(footer_chunks),
        )
        st.chat_message("assistant").markdown(bot_message)
        st.session_state.messages.append({"role": "assistant", "content": bot_message})
