import streamlit as st
from openai import OpenAI
import os
import glob
import difflib

# Set your OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Page config
st.set_page_config(page_title="Amir Exir's ERCOT Planning Guides AI Assistant", page_icon="⚡")
st.title("⚡ Ask Amir Exir's ERCOT Planning Guides AI Assistant")

# Load and chunk ERCOT planning guide text files
ercot_chunks = {}
for filepath in sorted(glob.glob("ercot_planning_part*.txt")):
    part_number = filepath.replace("ercot_planning_part", "").replace(".txt", "")
    with open(filepath, "r", encoding="utf-8") as f:
        ercot_chunks[part_number] = f.read()

# Fuzzy match to find best chunk
def find_best_chunk(user_question):
    scores = {}
    for part, text in ercot_chunks.items():
        score = difflib.SequenceMatcher(None, user_question.lower(), text.lower()).ratio()
        scores[part] = score
    if not scores:
        return "No planning guide files found."
    best_part = max(scores, key=scores.get)
    if scores[best_part] < 0.3:
        return "Sorry, I couldn’t find a relevant section."
    return ercot_chunks[best_part]

# Chat history init
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask Amir Exir's AI assistant about ERCOT planning guides..."):
    st.chat_message("user").markdown(prompt)
    relevant_text = find_best_chunk(prompt)

    system_prompt = {
        "role": "system",
        "content": f"You are an expert assistant on ERCOT's planning guides. Use only the section below to answer:\n\n{relevant_text}"
    }

    messages = [system_prompt] + st.session_state.messages + [{"role": "user", "content": prompt}]
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1024
        )

    bot_msg = response.choices[0].message.content
    st.chat_message("assistant").markdown(bot_msg)
    st.session_state.messages.append({"role": "assistant", "content": bot_msg})