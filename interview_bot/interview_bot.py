import streamlit as st
import numpy as np, json, os, io
import re
from datetime import datetime
from uuid import uuid4
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
AUDIO_DIR = os.path.join(base_path, "saved_audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

with open(os.path.join(base_path, "chunks_cleaned.json"), "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Force rebuild button to clear cache
if st.button(" Force Rebuild Embeddings"):
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
        st.success(" FAISS index initialized successfully.")
    else:
        st.error(" No embeddings found! Please rebuild embeddings.")
        st.stop()
else:
    index = st.session_state["index"]

def search(query, index, chunks, embeddings, k=5):
    # Pure semantic retrieval with keyword-aware reranking.
    print(f"🔍 Semantic search for query: {query}")

    normalized_query = query.lower().strip()
    query_terms = [term for term in re.findall(r"[a-z0-9]+", normalized_query) if len(term) > 2]

    # Create embedding for the query
    q_emb = client.embeddings.create(
        input=query,
        model="text-embedding-3-large"
    ).data[0].embedding

    q_emb = np.array(q_emb, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q_emb)

    # FAISS similarity search
    D, I = index.search(q_emb, min(max(k * 4, k), len(chunks)))

    # Re-rank retrieved chunks so exact keyword matches surface before purely semantic neighbors.
    reranked = []
    seen_indices = set()
    for score, idx in zip(D[0], I[0]):
        text = chunks[idx]["text"]
        normalized_text = text.lower()
        keyword_hits = sum(1 for term in query_terms if term in normalized_text)
        exact_query_hit = 1 if normalized_query and normalized_query in normalized_text else 0
        rerank_score = float(score) + (0.15 * keyword_hits) + (0.5 * exact_query_hit)
        reranked.append((rerank_score, idx, text))
        seen_indices.add(idx)

    for idx, chunk in enumerate(chunks):
        if idx in seen_indices:
            continue
        normalized_text = chunk["text"].lower()
        keyword_hits = sum(1 for term in query_terms if term in normalized_text)
        exact_query_hit = 1 if normalized_query and normalized_query in normalized_text else 0
        if keyword_hits or exact_query_hit:
            rerank_score = (0.15 * keyword_hits) + (0.5 * exact_query_hit)
            reranked.append((rerank_score, idx, chunk["text"]))

    reranked.sort(key=lambda item: item[0], reverse=True)

    if not reranked:
        return []

    top_k = reranked[:k]
    top_texts = [text for _, _, text in top_k]
    print(f" Retrieved {len(top_texts)} ranked results from FAISS")
    for i, (score, _, text) in enumerate(top_k):
        print(f"{i+1}. score={score:.4f} | {text[:120]}...")

    return top_texts


def detect_principle_from_context(retrieved_texts):
    # Use the highest-ranked retrieved chunk and extract its single Principle tag.
    if not retrieved_texts:
        return "Unknown"

    match = re.search(r"Principle:\s*([^|]+)", retrieved_texts[0])
    if not match:
        return "Unknown"

    principle = match.group(1).strip()
    return principle if principle else "Unknown"


def build_unique_audio_path(prefix="answer", ext="mp3"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid4().hex[:8]
    filename = f"{prefix}_{timestamp}_{unique_id}.{ext}"
    return os.path.join(AUDIO_DIR, filename)


def enforce_single_star_story(text):
    # If the model outputs multiple STAR blocks, keep only the first one.
    if not text:
        return text

    first_idx = text.find("Situation:")
    if first_idx == -1:
        return text

    second_idx = text.find("\nSituation:", first_idx + 1)
    if second_idx == -1:
        return text

    trimmed = text[:second_idx].rstrip()
    return trimmed + "\n\nIf you'd like another example, click 'Give me another story.'"


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="InterviewBot")
st.title("Amir's InterviewBot")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "candidate_stories" not in st.session_state:
    st.session_state.candidate_stories = []

if "candidate_story_pos" not in st.session_state:
    st.session_state.candidate_story_pos = 0

if "active_query" not in st.session_state:
    st.session_state.active_query = ""

# Show chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Voice input
st.markdown("### Tap below to ask your question")
st.markdown(
    "<div style='text-align:center; padding:10px;'>"
    "<span style='font-size:18px;'> Ready to listen...</span>"
    "</div>",
    unsafe_allow_html=True,
)

audio = mic_recorder(
    start_prompt=" Start Recording (Tap Once)",
    stop_prompt=" Stop Recording",
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
        st.chat_message("user").markdown(f" {user_query}")
        st.session_state.messages.append({"role": "user", "content": user_query})

# Fallback text input
prompt = st.chat_input("Or type your question here...")
if prompt:
    user_query = prompt
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

another_story = st.button("Give me another story")

query_to_answer = None
retrieved_texts = []

if user_query:
    st.session_state.candidate_stories = search(user_query, index, chunks, embeddings, k=4)
    st.session_state.candidate_story_pos = 0
    st.session_state.active_query = user_query

    if st.session_state.candidate_stories:
        query_to_answer = user_query
        retrieved_texts = [st.session_state.candidate_stories[0]]
elif another_story:
    if not st.session_state.candidate_stories:
        st.warning("Ask a question first so I can find matching stories.")
    else:
        current_pos = st.session_state.candidate_story_pos
        if current_pos < len(st.session_state.candidate_stories) - 1:
            st.session_state.candidate_story_pos += 1
            query_to_answer = st.session_state.active_query
            retrieved_texts = [st.session_state.candidate_stories[st.session_state.candidate_story_pos]]
            st.chat_message("user").markdown("Give me another story for the same question.")
            st.session_state.messages.append(
                {"role": "user", "content": "Give me another story for the same question."}
            )
        else:
            st.info("You have reached story 4 for this question. Ask a new question to get a new top-4 set.")

# Process assistant response
if retrieved_texts and query_to_answer:
    context = "\n\n".join(retrieved_texts)
    detected_principle = detect_principle_from_context(retrieved_texts)
    st.markdown(f"**Detected Principle:** {detected_principle}")
    st.caption(
        f"Story {st.session_state.candidate_story_pos + 1} of {len(st.session_state.candidate_stories)} for this question"
    )

    #  Debugging: show retrieved chunks before calling GPT
    show_debug = st.checkbox("Show retrieved context (cosine search)")
    if show_debug:
        st.markdown(f"**Query:** `{query_to_answer}`")
        st.markdown(f"**Retrieved {len(st.session_state.candidate_stories)} chunks**")
        for i, text in enumerate(st.session_state.candidate_stories):
            st.markdown(f"**Chunk {i+1}**")
            st.code(text[:300] + "...", language="markdown")
            if "waterloo" in text.lower():
                st.success(" Contains 'Waterloo'")
            st.write("---")

    # GPT response block
    messages = [
        {"role": "system", "content": (
            "You are Amir Exir in an interview. Use the provided context to answer as best as possible. "
            "Even if the context is not structured, infer meaning from relevant sentences. "
            "If there is any mention of the topic or related tools (like AELAB, automation, PSS/E, ERCOT), explain it in a STAR-like manner. "
            "Only say 'I don't have specific experience with that' if the topic is clearly unrelated. "
            "Answer naturally in first person and organize into four short paragraphs labeled: Situation, Task, Action, and Result. "
            "Use exactly one story from the provided context for this response. "
            "Do not provide a second example in the same response, even if the question asks for multiple examples. "
            "If asked for more examples, provide only the best one now and wait for follow-up. "
            "prioritze transmission planning stories over operational stories slightly"
        )},
        {"role": "user", "content": f"Question: {query_to_answer}\n\nRelevant context:\n{context}"}
    ]

    with st.spinner("Answering..."):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=2048,
            temperature=0.2
        )

    bot_msg = enforce_single_star_story(response.choices[0].message.content)
    st.markdown(f"**Question:** {query_to_answer}")
    st.chat_message("assistant").markdown(bot_msg)
    st.session_state.messages.append({"role": "assistant", "content": bot_msg})

    # TTS response
    with st.spinner("Speaking..."):
        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=bot_msg
        )
        audio_out = build_unique_audio_path(prefix="answer", ext="mp3")
        with open(audio_out, "wb") as f:
            f.write(speech.content)
    st.audio(audio_out, format="audio/mp3")
    st.caption(f"Saved audio file: {os.path.basename(audio_out)}")
