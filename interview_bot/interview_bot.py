import streamlit as st
import numpy as np, json, os, io
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder
import faiss

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# -------------------------
# Embeddings cache
# -------------------------
EMB_FILE = "embeddings.npy"
# -------------------------
# Load chunks_cleaned.json
# -------------------------
base_path = os.path.dirname(__file__)

with open(os.path.join(base_path, "chunks_cleaned.json"), "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Force rebuild button to clear cache
if st.button("🔄 Force Rebuild Embeddings"):
    if os.path.exists(EMB_FILE):
        os.remove(EMB_FILE)
        st.success("Deleted embeddings.npy")
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
else:
    embeddings = np.load(EMB_FILE)

# -------------------------
# Build FAISS index (robust version)
# -------------------------
if "index" not in st.session_state:
    if embeddings is not None and len(embeddings) > 0:
        # Normalize for cosine similarity (optional but better for semantic match)
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])  # IP = cosine similarity
        index.add(embeddings)
        st.session_state["index"] = index
        st.success("✅ FAISS index initialized successfully.")
    else:
        st.error("❌ No embeddings found! Please rebuild embeddings.")
        st.stop()
else:
    index = st.session_state["index"]
    
def search(query, index, chunks, embeddings, k=5):
    # Normalize query
    query_lower = query.lower().strip()
    keywords = [w for w in query_lower.split() if len(w) > 2]

    # --- 1️⃣ Exact keyword filtering ---
    keyword_matches = []
    for c in chunks:
        text_lower = c["text"].lower()
        if any(k in text_lower for k in keywords):
            keyword_matches.append(c["text"])

    # --- 2️⃣ If we found strong keyword matches, prioritize them ---
    if keyword_matches:
        print(f"✅ Keyword matches found for query: {query}")
        return keyword_matches[:k]

    # --- 3️⃣ Otherwise, fallback to semantic search ---
    print(f"⚠️ No keyword hits for '{query}', using semantic search...")
    q_emb = client.embeddings.create(
        input=query,
        model="text-embedding-3-large"
    ).data[0].embedding

    q_emb = np.array(q_emb, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q_emb)

    D, I = index.search(q_emb, k)
    top_texts = [chunks[i]["text"] for i in I[0]]

    return top_texts


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
    retrieved_texts = search(user_query, index, chunks, embeddings)
    context = "\n\n".join(retrieved_texts)

    # ✅ Debugging: show retrieved chunks before calling GPT
    show_debug = st.checkbox("Show retrieved context (cosine search)")
    if show_debug:
        st.markdown(f"**Query:** `{user_query}`")
        st.markdown(f"**Retrieved {len(retrieved_texts)} chunks**")
        for i, text in enumerate(retrieved_texts):
            st.markdown(f"**Chunk {i+1}**")
            st.code(text[:300] + "...", language="markdown")
            if "waterloo" in text.lower():
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