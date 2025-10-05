import streamlit as st
import faiss, numpy as np, json, os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load resume and stories
with open("amir_resume.txt", "r", encoding="utf-8") as f:
    resume_text = f.read()

with open("stories.json", "r", encoding="utf-8") as f:
    stories = json.load(f)

# Build chunks (resume + stories)
chunks = []
for i, para in enumerate(resume_text.split("\n\n")):
    if para.strip():
        chunks.append({"text": para.strip(), "source": "resume"})
for s in stories:
    text = f"{s['principle']} story: {s['situation']} {s['task']} {s['action']} {s['result']}"
    chunks.append({"text": text, "source": "story", "principle": s["principle"]})

# Create or load embeddings
if not os.path.exists("embeddings.npy"):
    embeddings = []
    for c in chunks:
        emb = client.embeddings.create(input=c["text"], model="text-embedding-3-large").data[0].embedding
        embeddings.append(emb)
    embeddings = np.array(embeddings, dtype="float32")
    np.save("embeddings.npy", embeddings)
else:
    embeddings = np.load("embeddings.npy")

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

def search(query, k=4):
    q_emb = client.embeddings.create(input=query, model="text-embedding-3-large").data[0].embedding
    D, I = index.search(np.array([q_emb], dtype="float32"), k)
    return [chunks[i] for i in I[0]]

# Streamlit UI
st.set_page_config(page_title="Amir's RAG Assistant", page_icon="🤖")
st.title("🎤 Amir's Resume + STAR RAG Assistant")

prompt = st.text_input("Ask me an interview question:")

if st.button("🎙️ Record Voice"):
    audio = st.audio_input("Speak your question")
    if audio:
        with open("temp.wav", "wb") as f:
            f.write(audio.getbuffer())
        transcript = client.audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=open("temp.wav","rb"))
        prompt = transcript.text
        st.write(f"🗣️ You asked: {prompt}")

if prompt:
    retrieved = search(prompt)
    context = "\n\n".join([r["text"] for r in retrieved])
    
    messages = [
        {"role": "system", "content": "You are Amir's career assistant. Only answer using resume or STAR stories."},
        {"role": "user", "content": f"Question: {prompt}\n\nContext:\n{context}"}
    ]
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    answer = response.choices[0].message.content
    st.write(answer)

    # Optional: text-to-speech
    if st.button("🔊 Speak Answer"):
        speech = client.audio.speech.create(model="gpt-4o-mini-tts", voice="alloy", input=answer)
        with open("answer.mp3", "wb") as f:
            f.write(speech.read())
        st.audio("answer.mp3")