import streamlit as st
from openai import OpenAI
import os

# Set your OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load resume content
with open("amir_resume.txt", "r", encoding="utf-8") as f:
    resume_text = f.read()

# Use resume as system prompt
system_prompt = {
    "role": "system",
    "content": f"""
You are a helpful assistant who knows the professional background of Seyed Amirhossein Eksir Monfared (Amir Exir).
Here is his resume:\n\n{resume_text}\n\nOnly use this information to answer questions about Amir.
"""
}

# Streamlit setup
st.set_page_config(page_title="Ask Amir's AI Assistant", page_icon="🤖")
st.title("🤖 Ask Amir's Career Assistant")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [system_prompt]

# Show chat history
for msg in st.session_state.messages[1:]:
    st.chat_message(msg["role"]).markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask me anything about Amir..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):

        response = client.chat.completions.create(


        client = openai.OpenAI()  # add this at the top after setting the API key
        response = openai.ChatCompletion.create(

            model="gpt-4o",
            messages=st.session_state.messages,
            max_tokens=1024
        )

    bot_msg = response.choices[0].message.content
    st.chat_message("assistant").markdown(bot_msg)
    st.session_state.messages.append({"role": "assistant", "content": bot_msg})