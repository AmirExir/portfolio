import streamlit as st
import numpy as np, json, os, io
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

# Add STAR stories as chunks (combine all possible fields into one chunk for each story)
for s in stories:
    principle = s.get("principle", "General")
    # Combine all relevant fields into one chunk for embedding
    fields = []
    if "principle" in s:
        fields.append(f"Principle: {s['principle']}")
    if "question" in s and s["question"]:
        fields.append(f"Question: {s['question']}")
    if "situation" in s and s["situation"]:
        fields.append(f"Situation: {s['situation']}")
    if "task" in s and s["task"]:
        fields.append(f"Task: {s['task']}")
    if "action" in s and s["action"]:
        fields.append(f"Action: {s['action']}")
    if "result" in s and s["result"]:
        fields.append(f"Result: {s['result']}")
    if fields:
        combined_text = "\n".join(fields)
        chunks.append({
            "text": combined_text,
            "source": "story-full",
            "principle": principle,
            "question": s.get("question", "")
        })
    else:
        # Fallback for other formats
        story_text = json.dumps(s)
        chunks.append({
            "text": story_text,
            "source": "story-generic",
            "principle": principle
        })

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
    for i, c in enumerate(chunks):
        print(f"Embedding chunk {i+1}/{len(chunks)}: {c['text'][:100]}...")
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

def search(query, k=10):
    query_lower = query.lower()
    query_words = [w for w in query_lower.split() if len(w) > 2]

    # 1️⃣ Keyword match
    keyword_hits = []
    for i, c in enumerate(chunks):
        text = c["text"].lower()
        if any(w in text for w in query_words):
            keyword_hits.append((i, 1.0))  # full weight for keyword hit

    # 2️⃣ Semantic FAISS search
    q_emb = client.embeddings.create(
        input=query,
        model="text-embedding-3-large"
    ).data[0].embedding
    D, I = index.search(np.array([q_emb], dtype="float32"), k)
    semantic_hits = [(int(i), float(1/(1+d))) for d, i in zip(D[0], I[0])]  # invert distance to weight

    # 3️⃣ Combine results
    combined = {}
    for i, score in keyword_hits + semantic_hits:
        combined[i] = combined.get(i, 0) + score

    # 4️⃣ Sort and select top k
    sorted_hits = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
    retrieved = [chunks[i] for i, _ in sorted_hits]

    return retrieved
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
st.markdown("### 🎤 Tap below to ask your question")
st.markdown(
    "<div style='text-align:center; padding:10px;'>"
    "<span style='font-size:18px;'>🎙️ Ready to listen...</span>"
    "</div>",
    unsafe_allow_html=True,
)

audio = mic_recorder(
    start_prompt="🎙️ Start Recording (Tap Once)",
    stop_prompt="⏹️ Stop Recording",
    use_container_width=True
)

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

    # ✅ Debugging: show retrieved chunks before calling GPT
    show_debug = st.checkbox("Show retrieved context (cosine search)")
    if show_debug:
        st.markdown(f"**Query:** `{user_query}`")
        st.markdown(f"**Retrieved {len(retrieved)} chunks**")
        for i, chunk in enumerate(retrieved):
            st.markdown(f"**Chunk {i+1}** — Source: `{chunk['source']}` | Principle: `{chunk.get('principle', 'N/A')}`")
            st.code(chunk["text"][:300] + "...", language="markdown")
            if "waterloo" in chunk["text"].lower():
                st.success("✅ Contains 'Waterloo'")
            st.write("---")

    # GPT response block
    messages = [
        {"role": "system", "content": (
            "You are Amir Exir in an interview. Answer using ONLY the provided context. "
            "If the context doesn't contain relevant information, say 'I don't have specific experience with that.' "
            "Never fabricate experiences. "
            "Answer in first person using information from my resume and STAR stories. "
            "Sound confident, conversational, and authentic — like you're recalling the experience in real time. "
            "Organize your response into four short, clear paragraphs labeled: Situation, Task, Action, and Result."
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