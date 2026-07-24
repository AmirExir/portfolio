import streamlit as st
import numpy as np, json, os, io
import re
from datetime import datetime
from uuid import uuid4
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder
import faiss

from embedding_utils import (
    EMBEDDING_MODEL,
    chunks_digest,
    load_valid_embeddings,
)
from response_utils import assess_chat_completion

api_key = os.getenv("OPENAI_API_KEY", "").strip()
client = OpenAI(api_key=api_key) if api_key else None
base_path = os.path.dirname(__file__)
st.set_page_config(page_title="InterviewBot")
# -------------------------
# Embeddings cache
# -------------------------
EMB_FILE = os.path.join(base_path, "embeddings.npy")
EMB_META_FILE = os.path.join(base_path, "embeddings.meta.json")
# -------------------------
# Load chunks_cleaned.json
# -------------------------
AUDIO_DIR = os.path.join(base_path, "saved_audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

with open(os.path.join(base_path, "chunks_cleaned.json"), "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Recheck files without deleting or rebuilding the corpus in the live app.
if st.button("Reload saved embeddings"):
    st.rerun()

embeddings, cache_status = load_valid_embeddings(
    chunks,
    EMB_FILE,
    EMB_META_FILE,
    model=EMBEDDING_MODEL,
)

if embeddings is None:
    st.error(
        f"The saved interview embedding cache cannot be loaded because {cache_status}. "
        "The live app will not rebuild the full corpus. Run "
        "`python interview_bot/generate_embeddings.py` during deployment, "
        "then click **Reload saved embeddings**."
    )
    st.stop()

# -------------------------
# Build FAISS index (robust version)
# -------------------------
current_chunks_digest = chunks_digest(chunks)
if (
    "index" not in st.session_state
    or st.session_state.get("index_chunks_digest") != current_chunks_digest
):
    if embeddings is not None and len(embeddings) > 0:
        embeddings = np.ascontiguousarray(embeddings.copy(), dtype="float32")
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])  # IP = cosine similarity
        index.add(embeddings)
        st.session_state["index"] = index
        st.session_state["index_chunks_digest"] = current_chunks_digest
        st.success("FAISS index initialized successfully.")
    else:
        st.error("No embeddings found. Please rebuild embeddings.")
        st.stop()
else:
    index = st.session_state["index"]

if client is None:
    st.error("OPENAI_API_KEY is not set. It is required for questions, answers, voice, and audio.")
    st.stop()

STORY_TAGS = ("Situation", "Task", "Action", "Result")
CONTENT_TAGS = (
    "Response Type",
    "Principle",
    "Category",
    "Question",
    "Aliases",
    "Situation",
    "Task",
    "Action",
    "Result",
    "Reflection",
    "Answer",
)
TAG_LOOKAHEAD = "|".join(re.escape(tag) for tag in CONTENT_TAGS)


def extract_tag_value(text, tag):
    match = re.search(
        rf"{re.escape(tag)}:\s*(.*?)(?=\s*\|\s*(?:{TAG_LOOKAHEAD}):|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def is_star_story(text):
    return all(re.search(rf"{tag}:\s*", text, flags=re.IGNORECASE) for tag in STORY_TAGS)


def detect_response_type(text, chunk=None):
    if isinstance(chunk, dict) and chunk.get("response_type") in {"direct", "story"}:
        return chunk["response_type"]
    explicit_type = extract_tag_value(text, "Response Type").lower()
    if explicit_type in {"direct", "story"}:
        return explicit_type
    if is_star_story(text):
        return "story"
    if extract_tag_value(text, "Answer"):
        return "direct"
    return "context"


def content_identity(text, response_type):
    # STAR aliases share a situation, while direct answers are unique by question.
    if response_type == "story":
        identity_source = extract_tag_value(text, "Situation") or text[:500]
    elif response_type == "direct":
        identity_source = extract_tag_value(text, "Question") or text[:500]
    else:
        identity_source = text[:500]
    return re.sub(r"\W+", " ", identity_source.lower()).strip()


def build_candidate(score, idx, chunk):
    text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
    response_type = detect_response_type(text, chunk)
    return {
        "score": float(score),
        "index": int(idx),
        "text": text.strip(),
        "principle": extract_tag_value(text, "Principle") or "Unknown",
        "category": extract_tag_value(text, "Category") or "General",
        "question": extract_tag_value(text, "Question") or "Untitled context",
        "response_type": response_type,
        "is_structured": response_type in {"direct", "story"},
        "content_key": content_identity(text, response_type),
    }


def coerce_candidate(candidate):
    if isinstance(candidate, dict):
        return candidate
    return build_candidate(0, -1, str(candidate))


def search(query, index, chunks, embeddings, k=5):
    # Semantic retrieval with keyword-aware reranking. Return one candidate per
    # direct answer or underlying STAR story instead of a bag of raw chunks.
    print(f"Semantic search for query: {query}")

    normalized_query = query.lower().strip()
    query_terms = [term for term in re.findall(r"[a-z0-9]+", normalized_query) if len(term) > 2]
    behavioral_query = any(
        phrase in normalized_query
        for phrase in (
            "tell me about a time",
            "give me an example",
            "describe a time",
            "walk me through a time",
        )
    )

    # Create embedding for the query
    q_emb = client.embeddings.create(
        input=query,
        model=EMBEDDING_MODEL,
    ).data[0].embedding

    q_emb = np.array(q_emb, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q_emb)

    # Pull more than the final count because several chunks can describe the
    # same underlying story with different LP wording.
    search_window = min(max(k * 10, 40), len(chunks))
    D, I = index.search(q_emb, search_window)

    # Re-rank retrieved chunks so exact keyword matches surface before purely semantic neighbors.
    reranked = []
    seen_indices = set()
    for score, idx in zip(D[0], I[0]):
        chunk = chunks[idx]
        text = chunk["text"]
        normalized_text = text.lower()
        keyword_hits = sum(1 for term in query_terms if term in normalized_text)
        exact_query_hit = 1 if normalized_query and normalized_query in normalized_text else 0
        candidate = build_candidate(float(score), idx, chunk)
        type_bonus = 0.0
        if behavioral_query and candidate["response_type"] == "story":
            type_bonus = 0.35
        elif not behavioral_query and candidate["response_type"] == "direct":
            type_bonus = 0.35
        rerank_score = (
            float(score)
            + (0.15 * keyword_hits)
            + (0.5 * exact_query_hit)
            + type_bonus
        )
        candidate["score"] = rerank_score
        reranked.append(candidate)
        seen_indices.add(idx)

    for idx, chunk in enumerate(chunks):
        if idx in seen_indices:
            continue
        normalized_text = chunk["text"].lower()
        keyword_hits = sum(1 for term in query_terms if term in normalized_text)
        exact_query_hit = 1 if normalized_query and normalized_query in normalized_text else 0
        if keyword_hits or exact_query_hit:
            candidate = build_candidate(0, idx, chunk)
            type_bonus = 0.0
            if behavioral_query and candidate["response_type"] == "story":
                type_bonus = 0.35
            elif not behavioral_query and candidate["response_type"] == "direct":
                type_bonus = 0.35
            candidate["score"] = (
                (0.15 * keyword_hits) + (0.5 * exact_query_hit) + type_bonus
            )
            reranked.append(candidate)

    reranked.sort(key=lambda item: item["score"], reverse=True)

    if not reranked:
        return []

    unique_candidates = []
    seen_content_keys = set()
    for candidate in reranked:
        if not candidate["is_structured"]:
            continue
        content_key = candidate["content_key"]
        if content_key in seen_content_keys:
            continue
        seen_content_keys.add(content_key)
        unique_candidates.append(candidate)
        if len(unique_candidates) == k:
            break

    # If the query is about resume/study context and no structured answer is found,
    # still answer from a single best chunk rather than combining many chunks.
    if not unique_candidates:
        for candidate in reranked:
            content_key = candidate["content_key"]
            if content_key in seen_content_keys:
                continue
            seen_content_keys.add(content_key)
            unique_candidates.append(candidate)
            if len(unique_candidates) == k:
                break

    print(f"Retrieved {len(unique_candidates)} unique ranked answers from FAISS")
    for i, candidate in enumerate(unique_candidates):
        print(f"{i+1}. score={candidate['score']:.4f} | {candidate['text'][:120]}...")

    return unique_candidates


def display_category(candidate):
    if not candidate:
        return "Unknown"
    if candidate.get("response_type") == "story":
        return candidate.get("principle") or "Unknown"
    return candidate.get("category") or "General"


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
    return trimmed + "\n\nIf you'd like another example, click 'Give me another answer or story.'"


# -------------------------
# Streamlit UI
# -------------------------
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
    use_container_width=True,
    just_once=True,
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

another_story = st.button("Give me another answer or story")

query_to_answer = None
selected_story = None

if user_query:
    st.session_state.candidate_stories = search(user_query, index, chunks, embeddings, k=6)
    st.session_state.candidate_story_pos = 0
    st.session_state.active_query = user_query

    if st.session_state.candidate_stories:
        query_to_answer = user_query
        selected_story = coerce_candidate(st.session_state.candidate_stories[0])
elif another_story:
    if not st.session_state.candidate_stories:
        st.warning("Ask a question first so I can find matching answers.")
    else:
        current_pos = st.session_state.candidate_story_pos
        if current_pos < len(st.session_state.candidate_stories) - 1:
            st.session_state.candidate_story_pos += 1
            query_to_answer = st.session_state.active_query
            selected_story = coerce_candidate(
                st.session_state.candidate_stories[st.session_state.candidate_story_pos]
            )
            st.chat_message("user").markdown("Give me another answer for the same question.")
            st.session_state.messages.append(
                {"role": "user", "content": "Give me another answer for the same question."}
            )
        else:
            answer_count = len(st.session_state.candidate_stories)
            st.info(
                f"You have reached answer {answer_count} for this question. "
                f"Ask a new question to get a new top-{answer_count} set."
            )

# Process assistant response
if selected_story and query_to_answer:
    selected_story = coerce_candidate(selected_story)
    selected_context = selected_story["text"]
    response_type = selected_story["response_type"]
    detected_category = display_category(selected_story)
    category_label = "Detected Principle" if response_type == "story" else "Answer Category"
    st.markdown(f"**{category_label}:** {detected_category}")
    st.caption(
        f"Answer {st.session_state.candidate_story_pos + 1} of {len(st.session_state.candidate_stories)} for this question"
    )

    #  Debugging: show retrieved chunks before calling GPT
    show_debug = st.checkbox("Show retrieved context (cosine search)")
    if show_debug:
        st.markdown(f"**Query:** `{query_to_answer}`")
        st.markdown("**Selected context sent to GPT**")
        st.code(selected_context, language="markdown")
        st.markdown(f"**Candidate answer pool ({len(st.session_state.candidate_stories)} total, not sent together)**")
        for i, story in enumerate(st.session_state.candidate_stories):
            story = coerce_candidate(story)
            marker = " - selected" if i == st.session_state.candidate_story_pos else ""
            label = story["principle"] if story["response_type"] == "story" else story["category"]
            st.markdown(
                f"**Answer {i+1}{marker}: {story['response_type']} / {label}**"
            )
            st.code(story["text"][:300] + "...", language="markdown")
            if "waterloo" in story["text"].lower():
                st.success(" Contains 'Waterloo'")
            st.write("---")

    if response_type == "story":
        answer_instructions = (
            "Answer naturally in first person and organize the response into four short paragraphs "
            "labeled Situation, Task, Action, and Result. Use only the single story inside "
            "<selected_context>. Do not combine, borrow, or merge facts from another story or prior "
            "answer. Do not provide a second example."
        )
    elif response_type == "direct":
        answer_instructions = (
            "Answer naturally in first person in two to four concise paragraphs. Do not force the "
            "answer into STAR labels. Use the Question, Aliases, and Answer fields inside "
            "<selected_context> as the source. Adapt the wording to the interviewer's exact question "
            "without inventing company-specific facts, formal management experience, or results."
        )
    else:
        answer_instructions = (
            "Answer naturally in first person using only the facts inside <selected_context>. "
            "Do not invent details that are not supported by the context."
        )

    # GPT response block
    messages = [
        {"role": "system", "content": (
            "You are Amir Exir in an interview. Use the provided context to answer as best as possible. "
            "Even if the context is not structured, infer meaning from relevant sentences. "
            "Only say 'I don't have specific experience with that' if the topic is clearly unrelated. "
            f"{answer_instructions} "
            "Prioritize transmission planning examples over operational examples slightly when both are equally relevant."
        )},
        {"role": "user", "content": (
            f"Question: {query_to_answer}\n\n"
            f"Selected response type: {response_type}\n"
            "Selected context:\n"
            "<selected_context>\n"
            f"{selected_context}\n"
            "</selected_context>"
        )}
    ]

    with st.spinner("Answering..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=2048,
                temperature=0.2,
            )
            assessment = assess_chat_completion(response)

            # A second generation is used only when the first one is blank or
            # truncated. The same single retrieved context remains authoritative.
            if assessment.retryable:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=3072,
                    temperature=0.2,
                )
                assessment = assess_chat_completion(response)
        except Exception:
            assessment = None

    if assessment is None:
        st.error("The answer request failed. Please retry in a moment.")
    elif not assessment.usable:
        st.error(
            f"No interview answer was returned because {assessment.diagnostic}. "
            "No blank answer was added to the conversation."
        )
    else:
        bot_msg = assessment.text
        if response_type == "story":
            bot_msg = enforce_single_star_story(bot_msg)
        st.markdown(f"**Question:** {query_to_answer}")
        st.chat_message("assistant").markdown(bot_msg)
        st.session_state.messages.append({"role": "assistant", "content": bot_msg})

        # Optional TTS — generate only when user requests it
        make_audio = st.button("Make answer audio")
        if make_audio:
            with st.spinner("Generating audio..."):
                speech = client.audio.speech.create(
                    model="gpt-4o-mini-tts",
                    voice="alloy",
                    input=bot_msg,
                )
                audio_out = build_unique_audio_path(prefix="answer", ext="mp3")
                with open(audio_out, "wb") as f:
                    f.write(speech.content)
            st.audio(audio_out, format="audio/mp3")
            st.caption(f"Saved audio file: {os.path.basename(audio_out)}")
