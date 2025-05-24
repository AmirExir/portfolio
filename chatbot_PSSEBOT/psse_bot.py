import streamlit as st
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load PSS/E reference text
with open("psse_guide.txt", "r", encoding="utf-8") as f:
    psse_text = f.read()[:120000]  # Adjust based on context limit

system_prompt = {
    "role": "system",
    "content": f"""
You are a Python automation expert in PSS/E using the PSSPY library.
Only use the following as your knowledge base:\n\n{psse_text}\n\n
Always return well-commented and working Python code. Focus on clarity and reliability.
"""
}

st.set_page_config(page_title="Amir's PSS/E API Assistant", page_icon="🧠")
st.title("🧠 Amir Exir's PSS/E API Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [system_prompt]

for msg in st.session_state.messages[1:]:
    st.chat_message(msg["role"]).markdown(msg["content"])

if prompt := st.chat_input("Ask for PSS/E API Python code or automation help..."):
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