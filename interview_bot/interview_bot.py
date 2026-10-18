import streamlit as st
import numpy as np, json, os, io
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
from interview_response_utils import assess_chat_completion
from interview_retrieval import (
    JOB_FOCUSES,
    JOB_CONTEXT_INSTRUCTIONS,
    MAX_JOB_DESCRIPTION_CHARS,
    build_job_context,
    build_search_query,
    clear_candidate_pool,
    coerce_candidate,
    display_category,
    rank_candidates,
)

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

def search(
    query: str, index, chunks: list[dict], embeddings, k: int = 5,
    focus: str = "All roles", background: str = "", stories_only: bool = False,
    job_description: str = "",
) -> list[dict]:
    """Retrieve answers using the question and selected job background."""
    q_emb = client.embeddings.create(
        input=build_search_query(query, focus, background, job_description),
        model=EMBEDDING_MODEL,
    ).data[0].embedding
    q_emb = np.array(q_emb, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q_emb)
    search_window = min(max(k * 10, 40), len(chunks))
    distances, indices = index.search(q_emb, search_window)
    scores = {
        int(idx): float(score)
        for score, idx in zip(distances[0], indices[0])
        if 0 <= idx < len(chunks)
    }
    return rank_candidates(
        query, chunks, scores, k=k, focus=focus, stories_only=stories_only,
        job_description=job_description,
    )


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

if "job_focus" not in st.session_state:
    st.session_state.job_focus = "All roles"
if "job_background" not in st.session_state:
    st.session_state.job_background = ""
if "job_description" not in st.session_state:
    st.session_state.job_description = ""

with st.sidebar.form("job_focus_form"):
    st.markdown("### Interview background")
    selected_focus = st.selectbox(
        "Job focus", list(JOB_FOCUSES),
        index=list(JOB_FOCUSES).index(st.session_state.job_focus),
        help="Choose a role to limit stories to that area, or leave All roles to use the job description across your experience.",
    )
    selected_background = st.text_area(
        "Job details (optional)", value=st.session_state.job_background,
        placeholder="Example: ERCOT transmission planning director; large loads, team leadership, and investment decisions.",
        max_chars=1500,
    )
    selected_description = st.text_area(
        "Job description", value=st.session_state.job_description,
        key="job_description_draft", height=260,
        placeholder="Paste the job posting here, including responsibilities and qualifications.",
        max_chars=MAX_JOB_DESCRIPTION_CHARS,
        help="Paste up to 20,000 characters. The description will guide every question in this session after you apply it.",
    )
    apply_focus = st.form_submit_button("Apply job focus")

if apply_focus:
    if (selected_focus, selected_background.strip(), selected_description.strip()) != (
        st.session_state.job_focus, st.session_state.job_background, st.session_state.job_description
    ):
        st.session_state.job_focus = selected_focus
        st.session_state.job_background = selected_background.strip()
        st.session_state.job_description = selected_description.strip()
        clear_candidate_pool(st.session_state)
    st.sidebar.success("Interview background applied. Ask a question or browse your stories.")

job_focus = st.session_state.job_focus
job_background = st.session_state.job_background
job_description = st.session_state.job_description
job_context = build_job_context(job_focus, job_background, job_description)
st.caption(f"Current job focus: {job_focus}")
if job_description:
    st.caption("Job description applied. Story selection and answers will be tailored to this posting.")
if job_background:
    st.caption("Your additional job details will guide story selection and answer emphasis.")
browse_stories = st.button("Show stories for this role")

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
stories_only = False
if browse_stories:
    role_label = "the pasted job description" if job_description else "my experience"
    if job_focus != "All roles":
        role_label = job_focus
    user_query = f"Show me a relevant story for {role_label}."
    stories_only = True
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
if audio:
    stories_only = False
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
    stories_only = False
    user_query = prompt
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

another_story = st.button("Give me another answer or story")

query_to_answer = None
selected_story = None

if user_query:
    with st.spinner("Finding relevant experience..."):
        st.session_state.candidate_stories = search(
            user_query, index, chunks, embeddings, k=6,
            focus=job_focus, background=job_background, stories_only=stories_only,
            job_description=job_description,
        )
    st.session_state.candidate_story_pos = 0
    st.session_state.active_query = user_query

    if st.session_state.candidate_stories:
        query_to_answer = user_query
        selected_story = coerce_candidate(st.session_state.candidate_stories[0])
    else:
        st.info(
            "No saved stories matched this question and job focus. "
            "Try a different question or choose All roles."
        )
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
    show_debug = st.checkbox("Show retrieved context")
    if show_debug:
        st.markdown(f"**Query:** `{query_to_answer}`")
        st.markdown(f"**Job focus:** {job_focus}")
        if job_context:
            st.text(job_context)
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
            f"{JOB_CONTEXT_INSTRUCTIONS}"
        )},
        {"role": "user", "content": (
            f"Question: {query_to_answer}\n\n"
            "<job_background>\n"
            f"{job_context or 'No specific job focus selected.'}\n"
            "</job_background>\n\n"
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
                model="gpt-6-astra",
                messages=messages,
                max_completion_tokens=2048,
                reasoning_effort="low",
            )
            assessment = assess_chat_completion(response)

            # A second generation is used only when the first one is blank or
            # truncated. The same single retrieved context remains authoritative.
            if assessment.retryable:
                response = client.chat.completions.create(
                    model="gpt-6-astra",
                    messages=messages,
                    max_completion_tokens=3072,
                    reasoning_effort="low",
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
