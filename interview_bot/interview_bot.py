import streamlit as st
import faiss, numpy as np, json, os, io
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# -------------------------
# Embeddings cache
# -------------------------
EMB_FILE = "embeddings.npy"
CHUNKS_FILE = "chunks.json"
# -------------------------
# Load resume + stories
# -------------------------
base_path = os.path.dirname(__file__)

with open(os.path.join(base_path, "amir_resume.txt"), "r", encoding="utf-8") as f:
    resume_text = f.read()

with open(os.path.join(base_path, "stories.json"), "r", encoding="utf-8") as f:
    stories = json.load(f)

chunks = []

# Split resume into chunks
for i, para in enumerate(resume_text.split("\n\n")):
    if para.strip():
        chunks.append({"text": para.strip(), "source": "resume"})

# Add STAR stories as chunks
for s in stories:
    # All stories now have the same format: principle, question, situation, task, action, result
    story_text = f"[{s['principle']}] Question: {s['question']} Situation: {s['situation']} Task: {s['task']} Action: {s['action']} Result: {s['result']}"
    chunks.append({"text": story_text, "source": "story", "principle": s["principle"]})

# Force rebuild button to clear cache
if st.button("🔄 Force Rebuild Embeddings"):
    if os.path.exists(EMB_FILE):
        os.remove(EMB_FILE)
        st.success("Deleted embeddings.npy")
    if os.path.exists(CHUNKS_FILE):
        os.remove(CHUNKS_FILE)
        st.success("Deleted chunks.json")
    st.warning("Cache cleared! Please restart the app to rebuild embeddings.")
    st.stop()



if not os.path.exists(EMB_FILE):
    embeddings = []
    for c in chunks:
        emb = client.embeddings.create(
            input=c["text"], 
            model="text-embedding-3-large"
        ).data[0].embedding
        embeddings.append(emb)
    embeddings = np.array(embeddings, dtype="float32")
    np.save(EMB_FILE, embeddings)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
else:
    embeddings = np.load(EMB_FILE)
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

def search(query, k=10):
    q_emb = client.embeddings.create(
        input=query, 
        model="text-embedding-3-large"
    ).data[0].embedding
    D, I = index.search(np.array([q_emb], dtype="float32"), k)
        # Print debug info
    print(f"Search query: {query}")
    print(f"Retrieved chunk indices: {I[0]}")
    print(f"Distances: {D[0]}")

    return [chunks[i] for i in I[0]]

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="InterviewBot", page_icon="🎤")
st.title("🎤 Amir's InterviewBot (Resume + STAR RAG)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Voice input
st.write("🎙️ Speak your question:")
audio = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop", just_once=True)

user_query = None
if audio:
    st.audio(audio["bytes"])
    with st.spinner("Transcribing..."):
        audio_file = io.BytesIO(audio["bytes"])
        audio_file.name = "speech.wav"
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
    retrieved = search(user_query)
    context = "\n\n".join(r["text"] for r in retrieved)

    # Add debugging checkbox to see what's being retrieved
    if st.checkbox("🔍 Show retrieved context (debugging)"):
        st.write(f"**Query:** {user_query}")
        st.write(f"**Number of chunks retrieved:** {len(retrieved)}")
        for i, chunk in enumerate(retrieved):
            st.write(f"**Chunk {i+1}:**")
            st.write(f"Source: {chunk['source']}")
            st.write(f"Principle: {chunk.get('principle', 'N/A')}")
            st.write(f"Text preview: {chunk['text'][:200]}...")
            if "Waterloo" in chunk['text']:
                st.write("✅ **CONTAINS WATERLOO**")
            st.write("---")



    messages = [
        {"role": "system", "content": (
            "You are Amir Exir in an interview. Answer using ONLY the provided context. If the context doesn't contain relevant information, say 'I don't have specific experience with that.' Never fabricate experiences."
            "Answer in first person using information from my resume and STAR stories. "
            "Sound confident, conversational, and authentic — like you're recalling the experience in real time. "
            "Organize your response into four short, clear paragraphs labeled: Situation, Task, Action, and Result. "
            "Each section should read naturally, not like a script, with smooth transitions and a storytelling tone."
        )},
        {"role": "user", "content": f"Question: {user_query}\n\nRelevant context:\n{context}"}
    ]
    with st.spinner("Answering..."):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1024,
            temperature=0.1
        )
    bot_msg = response.choices[0].message.content
    st.markdown(f"**Question:** {user_query}")    
    st.chat_message("assistant").markdown(bot_msg)
    st.session_state.messages.append({"role": "assistant", "content": bot_msg})

    # TTS response
    with st.spinner("Speaking..."):
        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=bot_msg
        )
        audio_out = "answer.mp3"
        with open(audio_out, "wb") as f:
            f.write(speech.content)
    st.audio(audio_out, format="audio/mp3")