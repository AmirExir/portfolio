import streamlit as st
import os
import json
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

# Optional: hide Streamlit Community Cloud footer/badges (UI hack; may vary by Streamlit version)
HIDE_STREAMLIT_UI = """
<style>
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    [data-testid="stFooter"] { display: none; }
    [data-testid="stHeader"] { display: none; }
    [data-testid="stToolbar"] { display: none; }
    .viewerBadge_container__1QSob { display: none; }
    .styles_viewerBadge__1yB5_ { display: none; }
</style>
"""

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load precomputed chunks and embeddings
@st.cache_data(show_spinner=False)
def load_ercot_chunks_and_embeddings(chunks_path: str, embeddings_path: str, chunks_mtime: float, embeddings_mtime: float):
    if not os.path.exists(chunks_path) or not os.path.exists(embeddings_path):
        raise FileNotFoundError("Embeddings or chunks file not found.")

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load(embeddings_path)

    return chunks, embeddings

# Embed the user query
def embed_query(query: str) -> List[float]:
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    )
    return response.data[0].embedding

# Find best matching chunk
def find_best_match(query: str, chunks, embeddings):
    query_embedding = np.array(embed_query(query)).reshape(1, -1)
    scores = cosine_similarity(query_embedding, embeddings).flatten()
    best_idx = int(np.argmax(scores))
    return chunks[best_idx]


def find_top_matches(query: str, chunks, embeddings, k: int = 5):
    query_embedding = np.array(embed_query(query)).reshape(1, -1)
    scores = cosine_similarity(query_embedding, embeddings).flatten()
    k = max(1, min(k, len(scores)))
    top_indices = np.argsort(scores)[-k:][::-1]
    return [chunks[int(i)] for i in top_indices]


def get_loaded_sources(chunks) -> list[str]:
    sources = sorted({c.get("source", "") for c in chunks if isinstance(c, dict)})
    return [s for s in sources if s]


def is_meta_visibility_question(prompt: str) -> bool:
    p = (prompt or "").strip().lower()
    triggers = [
        "can you see",
        "which files",
        "what files",
        "planning guides do you have",
        "what documents",
    ]
    return any(t in p for t in triggers)


def is_last_read_question(prompt: str) -> bool:
    p = (prompt or "").strip().lower()
    triggers = [
        "what is the last thing you can read",
        "last thing you can read",
        "last paragraph you can read",
        "what are the last 3 words",
        "what are the last three words",
    ]
    return any(t in p for t in triggers)


def _last_paragraph(text: str) -> str:
    if not text:
        return ""
    parts = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
    return parts[-1] if parts else text.strip()


def _last_words(text: str, n: int) -> str:
    words = (text or "").split()
    if not words:
        return ""
    return " ".join(words[-n:])


def build_last_read_answer(chunks, sources: list[str]) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    lines: list[str] = [
        "Here’s the last content I can read from each loaded part (deterministic, from the raw files when available):",
        "",
    ]

    # Group chunks by source as a fallback if raw files are missing.
    chunks_by_source: dict[str, list[dict]] = {}
    for c in chunks:
        if isinstance(c, dict) and c.get("source"):
            chunks_by_source.setdefault(c["source"], []).append(c)

    for src in sources:
        raw_path = os.path.join(base_dir, src)
        raw_text = ""
        if os.path.exists(raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
        else:
            # Fallback: last chunk by max chunk_index for that source
            candidates = chunks_by_source.get(src, [])
            if candidates:
                last_chunk = max(candidates, key=lambda d: int(d.get("chunk_index", -1)))
                raw_text = last_chunk.get("text", "")

        last_para = _last_paragraph(raw_text)
        last_10 = _last_words(raw_text, 10)
        lines.append(f"File: {src}")
        if last_para:
            lines.append(f"Last paragraph: {last_para}")
        else:
            lines.append("Last paragraph: (no content found)")
        if last_10:
            lines.append(f"Last 10 words: {last_10}")
        else:
            lines.append("Last 10 words: (no content found)")
        lines.append("")

    return "\n".join(lines).strip()

# Streamlit UI
st.set_page_config(page_title="Amir Exir's ERCOT Planning Guides AI Assistant", page_icon="⚡")
st.markdown(HIDE_STREAMLIT_UI, unsafe_allow_html=True)
st.title("Ask Amir Exir's ERCOT Planning Guides AI Assistant")

# Load data and embeddings once
with st.spinner("Loading planning guide embeddings..."):
    chunks_path = "chatbot_ercot/ercot_planning_chunks.json"
    embeddings_path = "chatbot_ercot/ercot_planning_embeddings.npy"
    chunks_mtime = os.path.getmtime(chunks_path) if os.path.exists(chunks_path) else 0.0
    embeddings_mtime = os.path.getmtime(embeddings_path) if os.path.exists(embeddings_path) else 0.0
    chunks, embeddings = load_ercot_chunks_and_embeddings(chunks_path, embeddings_path, chunks_mtime, embeddings_mtime)

sources = get_loaded_sources(chunks)
with st.sidebar:
    st.markdown("### Loaded planning guide parts")
    if sources:
        for s in sources:
            st.markdown(f"- {s}")
    else:
        st.markdown("No sources found in chunks file.")

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about ERCOT planning guides..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Handle visibility/meta questions without calling the LLM.
    if is_meta_visibility_question(prompt):
        if sources:
            meta_answer = (
                "I can access the planning guide content that has been embedded and loaded into this app.\n\n"
                "These files are currently loaded:\n"
                + "\n".join([f"- {s}" for s in sources])
            )
        else:
            meta_answer = "I can’t determine which planning guide files are loaded (no sources found in the chunks file)."

        st.chat_message("assistant").markdown(meta_answer)
        st.session_state.messages.append({"role": "assistant", "content": meta_answer})
        st.stop()

    # Handle "last thing you can read" questions deterministically.
    if is_last_read_question(prompt):
        last_answer = build_last_read_answer(chunks, sources)
        st.chat_message("assistant").markdown(last_answer)
        st.session_state.messages.append({"role": "assistant", "content": last_answer})
        st.stop()

    with st.spinner("Thinking..."):
        top_chunks = find_top_matches(prompt, chunks, embeddings, k=5)

        context_blocks = []
        for c in top_chunks:
            context_blocks.append(
                f"Filename: {c.get('source', 'unknown')}\n\n{c.get('text', '')}"
            )

        context_text = "\n\n---\n\n".join(context_blocks)

        system_prompt = {
            "role": "system",
            "content": f"""
You are an expert assistant on ERCOT's planning guides.
Only use the following documentation to answer the question:

---
{context_text}
---
Instructions:
- Stay factual and grounded strictly in the provided content.
- Do NOT guess, assume, or rely on outside knowledge.
- If the answer is not explicitly found in the document, respond exactly:
  "I couldn’t find that in the documentation."

Formatting requirements (MANDATORY):
- Write in clear, well-spaced paragraphs.
- Use a short introductory sentence before any list.
- Use bullet points or numbered lists when presenting multiple items.
- Do NOT combine headings and content into a single sentence.
- Ensure readability suitable for a technical ERCOT planning audience.
"""
        }

        messages = [system_prompt] + st.session_state.messages

        response = client.responses.create(
            model="gpt-5.2",
            reasoning={"effort": "high"},
            input=messages,
            max_output_tokens=1000
        )

        bot_msg = response.output_text
        st.chat_message("assistant").markdown(bot_msg)
        st.session_state.messages.append(
            {"role": "assistant", "content": bot_msg}
        )