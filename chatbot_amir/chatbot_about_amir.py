# chatbot_about_amir.py
import streamlit as st
import openai
import os

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define Amir's bio as the system prompt
system_prompt = {
    "role": "system",
    "content": """
You are a helpful assistant who knows the professional background of Seyed Amirhossein Eksir Monfared (Amir Exir), a U.S. citizen, licensed P.E., and experienced Power System Engineer. 
He worked at ERCOT and LCRA in planning, modeling, and operations. He holds a Master’s in Electrical Engineering from Lamar University and is pursuing a Master's in AI at UT Austin. 
He is proficient in tools like PSS/E, Python, GE EMS, and has certifications from AWS and IBM.
Answer only questions related to his background, work, education, certifications, and technical expertise.
"""
}

# Set up Streamlit UI
st.set_page_config(page_title="Ask Amir's AI Assistant", page_icon="🤖")
st.title("🤖 Ask Amir's Career Assistant")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [system_prompt]

# Display chat history
for msg in st.session_state.messages[1:]:
    st.chat_message(msg["role"]).markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask me anything about Amir..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=st.session_state.messages,
            max_tokens=1024
        )

    bot_msg = response['choices'][0]['message']['content']
    st.chat_message("assistant").markdown(bot_msg)
    st.session_state.messages.append({"role": "assistant", "content": bot_msg})