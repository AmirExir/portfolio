import streamlit as st
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder
import os
import io

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load resume
with open("amir_resume.txt", "r", encoding="utf-8") as f:
    resume_text = f.read()

system_prompt = {
    "role": "system",
    "content": f"""
You are a helpful assistant who knows the professional background of Seyed Amirhossein Eksir Monfared (Amir Exir).
Here is his resume:\n\n{resume_text}\n\nOnly use this information to answer questions about Amir.
"""
}

st.set_page_config(page_title="Amir's Career Assistant", page_icon="🎤")
st.title("🎤 Ask Amir's Career Assistant (Talk or Type)")

if "messages" not in st.session_state:
    st.session_state.messages = [system_prompt]

# Show chat history
for msg in st.session_state.messages[1:]:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Mic button
st.write("🎙️ Speak your question:")
audio = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop", just_once=True, use_container_width=True)

user_query = None

if audio:
    st.audio(audio["bytes"])  # playback user recording
    with st.spinner("Transcribing..."):
        audio_file = io.BytesIO(audio["bytes"])
        audio_file.name = "speech.wav"   # give it a proper name with extension!

        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        user_query = transcription.text
        st.chat_message("user").markdown(f"🎤 {user_query}")
        st.session_state.messages.append({"role": "user", "content": user_query})

# Fallback text input
prompt = st.chat_input("Or type your question here...")
if prompt:
    user_query = prompt
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

# Process assistant response
if user_query:
    with st.spinner("Thinking..."):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=st.session_state.messages,
            max_tokens=1024
        )

    bot_msg = response.choices[0].message.content
    st.chat_message("assistant").markdown(bot_msg)
    st.session_state.messages.append({"role": "assistant", "content": bot_msg})

    # Generate TTS audio
    with st.spinner("Speaking..."):
        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=bot_msg
        )
        audio_out = "assistant_reply.mp3"
        with open(audio_out, "wb") as f:
            f.write(speech.content)

    st.audio(audio_out, format="audio/mp3")