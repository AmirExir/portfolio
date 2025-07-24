import streamlit as st
from openai import OpenAI
import os

# Set your OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load ERCOT protocols and planning text
with open("ercotRIhandbook.txt", "r", encoding="utf-8") as f:
    ercot_text = f.read()[:120000]

# Use ERCOT info as system prompt
system_prompt = {
    "role": "system",
    "content": f"""
You are an expert assistant on ERCOT's interconnection process.
Only use the following documentation to answer any questions:\n\n{ercot_text}\n\nStay factual and cite based only on this info.
"""
}

# Streamlit UI setup
st.set_page_config(page_title="Amir Exir's ERCOT Interconnection AI Assistant", page_icon="⚡")
st.title("⚡ Ask Amir Exir's ERCOT AI Assistant")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [system_prompt]

for msg in st.session_state.messages[1:]:
    st.chat_message(msg["role"]).markdown(msg["content"])

if prompt := st.chat_input("Ask Amir Exir's AI assistan about ERCOT interconnection..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=st.session_state.messages,
            max_tokens=20000,
        )

    bot_msg = response.choices[0].message.content
    st.chat_message("assistant").markdown(bot_msg)
    st.session_state.messages.append({"role": "assistant", "content": bot_msg})