import streamlit as st
import openai
import os
import glob
import difflib

st.set_page_config(page_title="ERCOT Nodal Protocols Assistant")
st.title("⚡ Ask Amir Exir's ERCOT Nodal Protocols Assistant")

# Load all ERCOT chunks at startup
ercot_chunks = {}
for filepath in sorted(glob.glob("ercotnodals_part*.txt")):
    part_number = filepath.replace("ercotnodals_part", "").replace(".txt", "")
    with open(filepath, "r", encoding="utf-8") as f:
        ercot_chunks[part_number] = f.read()

# Simple fuzzy match function (can later be upgraded to semantic search)
def find_best_chunk(user_question):
    scores = {
        part: difflib.SequenceMatcher(None, user_question.lower(), text.lower()).ratio()
        for part, text in ercot_chunks.items()
    }
    best_part = max(scores, key=scores.get)
    return ercot_chunks[best_part]

# Initialize chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about ERCOT's Nodal Protocols"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    relevant_text = find_best_chunk(prompt)
    system_prompt = {
        "role": "system",
        "content": f"""
You are an expert assistant on ERCOT's Nodal Protocols.
Only use the following section of documentation to answer questions:

{relevant_text}
"""
    }

    with st.spinner("Thinking..."):
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[system_prompt] + st.session_state.messages,
            max_tokens=1024,
        )

    reply = response.choices[0].message["content"]
    st.chat_message("assistant").markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})