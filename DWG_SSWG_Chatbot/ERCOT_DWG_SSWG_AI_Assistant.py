import streamlit as st
import os
import openai
import json
import glob
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import numpy as np

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load or compute embeddings
@st.cache_data(show_spinner=False)
def load_ercot_chunks_and_embeddings():
    from openai import OpenAI
    embedding_model = "text-embedding-3-small"

    chunks = []
    embeddings = []

    st.write("📄 Available DWG_SSW files:")
    st.write(sorted(glob.glob("DWG_SSWG_Chatbot/dwg_sswg_chunks/dwg_sswg_chunk_*.txt")))

    for filepath in sorted(glob.glob("DWG_SSWG_Chatbot/dwg_sswg_chunks/dwg_sswg_chunk_*.txt")):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            chunks.append({"filename": filepath, "text": text})
    for chunk in chunks:
        try:
            response = client.embeddings.create(
                model=embedding_model,
                input=chunk["text"]
            )
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            st.warning(f"Embedding failed for {chunk['filename']}: {e}")
            print(f"ERROR for {chunk['filename']}: {e}")
            embeddings.append(None)

    # Clean up bad embeddings
    valid_pairs = [(c, e) for c, e in zip(chunks, embeddings) if e is not None]

    if not valid_pairs:
        st.warning("⚠️ No embeddings succeeded. Check file contents or OpenAI key.")
        raise ValueError("No valid embeddings were generated. Please check the input files.")

    chunks, embeddings = zip(*valid_pairs)
    embeddings = np.array(embeddings)
    return list(chunks), embeddings

# Embed the user query
def embed_query(query: str) -> List[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return response.data[0].embedding

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_top_k_matches(query: str, chunks, embeddings, top_k: int = 3, score_threshold: float = 0.6, debug: bool = False):
    query_embedding = np.array(embed_query(query)).reshape(1, -1)
    scores = cosine_similarity(query_embedding, embeddings).flatten()

    # Score each chunk with its index
    scored_chunks = [(i, scores[i]) for i in range(len(chunks))]

    # Filter by threshold
    filtered = [item for item in scored_chunks if item[1] >= score_threshold]

    # Sort by descending score
    top_filtered = sorted(filtered, key=lambda x: x[1], reverse=True)[:top_k]

    if debug:
        print("\n[DEBUG] Top matching chunks:")
        for i, score in top_filtered:
            print(f"\nRank: {i+1}, Score: {score:.4f}, Filename: {chunks[i]['filename']}")
            print(chunks[i]['text'][:300])  # Show snippet

    if not top_filtered:
        return "No relevant chunks found above the similarity threshold."

    selected_chunks = [chunks[i] for i, _ in top_filtered]
    combined_text = "\n---\n".join([f"Filename: {c['filename']}\n\n{c['text']}" for c in selected_chunks])

    return combined_text

# Streamlit UI
st.set_page_config(page_title="Amir Exir's ERCOT DWG & SSWG AI Assistant", page_icon="⚡")
st.title("🤖 ERCOT DWG & SSWG AI Assistant – by Amir Exir 😎")

# Load data and embeddings once
with st.spinner("Loading  DWG & SSWG and computing embeddings..."):
    chunks, embeddings = load_ercot_chunks_and_embeddings()

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show past messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about ERCOT  DWG & SSWG..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        fallback_context = None

        # Try exact string match for ANY entity
        for c in chunks:
            lowered_chunk = c["text"].lower()
            lowered_prompt = prompt.lower()

            # Check for presence of any entity name in prompt
            if any(name in lowered_prompt and name in lowered_chunk for name in ["lcra", "oncor", "austin energy", "cps energy", "goldenspread", "guadalupe valley", "central texas", "new braunfels", "american electric power", "denton", "brazos", "centerpoint", "lubbock", "sharyland", "tri-county", "rayburn", "texas new mexico", "college station", "bryan", "cross texas", "bluebonnet", "bandera", "san bernard", "farmers", "trinity valley", "fannin", "coleman", "concho", "lone star", "wind energy", "goldsmith", "greenville", "garland", "grayson", "pedernales", "rio grande", "lamar", "cooperative", "municipal"]):
                fallback_context = f"Filename: {c['filename']}\n\n{c['text'][:2000]}"
                st.info(f"⚡ Matched by keyword fallback: {c['filename']}")
                break

        if fallback_context:
            context = fallback_context
        else:
            context = find_top_k_matches(prompt, chunks, embeddings, top_k=5, debug=True)

        system_prompt = {
            "role": "system",
            "content": f"""You are an expert assistant that answers questions about ERCOT's DWG and SSWG planning documentation.

        You are only allowed to use the context below to answer. However, if the answer is clearly and explicitly present in the context, you must extract it and respond precisely.

        Do NOT ignore relevant content. If the answer is not present, say: "I couldn’t find that in the documentation."

        --- START OF CONTEXT ---
        {context}
        --- END OF CONTEXT ---
        """
        }

    messages = [system_prompt] + st.session_state.messages

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=1024
    )

    bot_msg = response.choices[0].message.content
    st.chat_message("assistant").markdown(bot_msg)
    st.session_state.messages.append({"role": "assistant", "content": bot_msg})