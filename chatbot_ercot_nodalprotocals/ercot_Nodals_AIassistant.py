import streamlit as st
from openai import OpenAI
import os

# Set your OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# UI setup
st.set_page_config(page_title="Amir Exir's ERCOT AI Assistant – Nodal Protocols", page_icon="⚡")
st.title("⚡ Ask Amir Exir's ERCOT Nodal Protocols Assistant")

# Dropdown for selecting which chunk of the Nodal Protocols to load
part_number = st.selectbox("Select section of ERCOT Nodal Protocols (split into 28 parts):", list(range(1,29)))

# Load selected chunk of ERCOT Nodal Protocols
filename = f"ercotnodals_part_{part_number}.txt"
try:
    with open(filename, "r", encoding="utf-8") as f:
        ercot_text = f.read()
except FileNotFoundError:
    st.error(f"File {filename} not found. Make sure it's in the same directory as this script.")
    st.stop()

# Use ERCOT info as system prompt
system_prompt = {
    "role": "system",
    "content": f"""
You are an expert assistant on ERCOT's Nodal Protocols.
Only use the following section of documentation to answer questions:\n\n{ercot_text}\n\nStay factual and cite based only on this info.
"""
}

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [system_prompt]

for msg in st.session_state.messages[1:]:
    st.chat_message(msg["role"]).markdown(msg["content"])

if prompt := st.chat_input("Ask a question about this section of the Nodal Protocols..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=st.session_state.messages,
            max_tokens=1024
        )

    bot_msg = response.choices[0].message.content
    st.chat_message("assistant").markdown(bot_msg)
    st.session_state.messages.append({"role": "assistant", "content": bot_msg})