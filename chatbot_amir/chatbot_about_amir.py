import streamlit as st
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder
import os
import io

from response_utils import assess_response, compact_messages


api_key = os.getenv("OPENAI_API_KEY", "").strip()
client = OpenAI(api_key=api_key) if api_key else None

# Load resume
with open("amir_resume.txt", "r", encoding="utf-8") as f:
    resume_text = f.read()

system_prompt = {
    "role": "system",
    "content": f"""
You are a helpful assistant who knows the professional background of Amir Exir (Seyed Amirhossein Eksir Monfared).
Here is his resume:\n\n{resume_text}\n\nOnly use this information to answer questions about Amir.
"""
}

st.set_page_config(page_title="Amir's Career Assistant", page_icon="🎤")
st.title(" Ask Amir's Career Assistant")

if client is None:
    st.error("OPENAI_API_KEY is not set. It is required for questions and voice.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [system_prompt]

# Show chat history
for msg in st.session_state.messages[1:]:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Mic button
st.write(" Speak your question:")
speak_answers = st.toggle(
    "Read answers aloud",
    value=False,
    help="Speech generation uses a separate paid API request.",
)
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
        st.chat_message("user").markdown(f" {user_query}")
        st.session_state.messages.append({"role": "user", "content": user_query})

# Fallback text input
prompt = st.chat_input("Or type your question here...")
if prompt:
    user_query = prompt
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

# Process assistant response
if user_query:
    request_messages = compact_messages(
        st.session_state.messages,
        max_recent_messages=6,
    )
    with st.spinner("Thinking..."):
        try:
            response = client.responses.create(
                model="gpt-5.2",
                reasoning={"effort": "none"},
                input=request_messages,
                max_output_tokens=2048,
            )
            assessment = assess_response(response)

            # Retry only an empty or incomplete generation, with no old chat turns.
            if assessment.retryable:
                response = client.responses.create(
                    model="gpt-5.2",
                    reasoning={"effort": "none"},
                    input=[system_prompt, {"role": "user", "content": user_query}],
                    max_output_tokens=3072,
                )
                assessment = assess_response(response)
        except Exception:
            assessment = None

    if assessment is None:
        st.error("The answer request failed. Please retry in a moment.")
    elif not assessment.usable:
        st.error(
            f"No answer was returned because {assessment.diagnostic}. "
            "No speech request was made; please retry."
        )
    else:
        bot_msg = assessment.text
        st.chat_message("assistant").markdown(bot_msg)
        st.session_state.messages.append({"role": "assistant", "content": bot_msg})

        if speak_answers:
            with st.spinner("Speaking..."):
                try:
                    speech = client.audio.speech.create(
                        model="gpt-4o-mini-tts",
                        voice="alloy",
                        input=bot_msg,
                    )
                    st.audio(speech.content, format="audio/mp3")
                except Exception:
                    st.warning("The answer succeeded, but speech generation failed.")
