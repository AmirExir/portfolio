import streamlit as st
import numpy as np, json, os
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder
import faiss

from embedding_utils import (
    EMBEDDING_MODEL,
    chunks_digest,
    load_valid_embeddings,
)
from interview_evidence import answer_markdown, build_story_library, story_id
from interview_services import InterviewServiceError, render_answer, service_error, transcribe
from interview_sessions import restore_session
from interview_runtime_content import activate_runtime_content, load_runtime_corpus
from interview_workspace_ui import render_evidence, render_practice, render_session_controls
from interview_retrieval import (
    JOB_FOCUSES,
    MAX_JOB_DESCRIPTION_CHARS,
    apply_retrieval_policy,
    build_job_context,
    build_search_query,
    clear_candidate_pool,
    coerce_candidate,
    display_category,
    rank_candidates,
    assigned_candidates,
)

api_key = os.getenv("OPENAI_API_KEY", "").strip()
client = OpenAI(api_key=api_key, timeout=45, max_retries=1) if api_key else None
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
with open(os.path.join(base_path, "chunks_cleaned.json"), "r", encoding="utf-8") as f:
    saved_chunks = json.load(f)

# Recheck files without deleting or rebuilding the corpus in the live app.
if st.button("Reload saved embeddings"):
    st.rerun()

embeddings, cache_status = load_valid_embeddings(
    saved_chunks,
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

try:
    runtime_corpus = load_runtime_corpus(saved_chunks)
except (OSError, ValueError):
    st.error(
        "Current interview answers could not be loaded. Correct manager_content.json "
        "and reload the app; previous answer text will not be substituted."
    )
    st.stop()
chunks = runtime_corpus.chunks
content_changed = activate_runtime_content(st.session_state, chunks)
if runtime_corpus.pending_embedding_count:
    st.info(
        f"Latest answers are active. {runtime_corpus.pending_embedding_count} updated answers "
        "use question and text matching until their search index is refreshed."
    )
if content_changed:
    st.caption("Saved answers changed. Previous answer actions were cleared; conversation history is retained.")

# -------------------------
# Build FAISS index (robust version)
# -------------------------
current_chunks_digest = chunks_digest(saved_chunks)
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
    st.warning("OPENAI_API_KEY is not set. You can review stories and saved sessions; AI requests require server configuration.")

def search(
    query: str, index, chunks: list[dict], embeddings, k: int = 5,
    focus: str = "All roles", background: str = "", stories_only: bool = False,
    job_description: str = "",
) -> list[dict]:
    """Retrieve answers using the question and selected job background."""
    def reviewed_candidates(candidates: list[dict]) -> list[dict]:
        return [candidate for candidate in candidates if
                candidate["response_type"] != "story" or
                st.session_state.get("story_reviews", {}).get(story_id(candidate["text"]), {}).get("review_status") != "Needs correction"][:k]

    assigned = assigned_candidates(query, chunks)
    if assigned is not None:
        # An excluded primary must not make its backup the automatic answer.
        # An explicitly requested backup is its own first candidate instead.
        if assigned and not reviewed_candidates(assigned[:1]):
            return []
        return reviewed_candidates(assigned)
    if client is None:
        raise InterviewServiceError("Story search requires OPENAI_API_KEY to be configured on the server.")
    try:
        q_emb = client.embeddings.create(
            input=build_search_query(query, focus, background, job_description),
            model=EMBEDDING_MODEL,
        ).data[0].embedding
    except Exception as exc:
        raise service_error("Story search", exc) from exc
    try:
        q_emb = np.array(q_emb, dtype="float32").reshape(1, -1)
    except (TypeError, ValueError) as exc:
        raise InterviewServiceError("Story search returned an invalid vector. Retry the search.") from exc
    if q_emb.shape[1] != embeddings.shape[1] or not np.isfinite(q_emb).all() or not np.any(q_emb):
        raise InterviewServiceError("Story search returned an invalid vector. Retry the search.")
    faiss.normalize_L2(q_emb)
    search_window = min(max(k * 10, 40), embeddings.shape[0])
    distances, indices = index.search(q_emb, search_window)
    saved_scores = {
        int(idx): float(score)
        for score, idx in zip(distances[0], indices[0])
        if 0 <= idx < embeddings.shape[0]
    }
    scores = runtime_corpus.remap_scores(saved_scores)
    candidates = rank_candidates(
        query, chunks, scores, k=len(chunks), focus=focus, stories_only=stories_only,
        job_description=job_description,
    )
    return reviewed_candidates(candidates)


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

for key, default in {
    "messages": [], "candidate_stories": [], "candidate_story_pos": 0,
    "active_query": "", "job_focus": "All roles", "job_background": "",
    "job_description": "", "story_reviews": {}, "practice_history": [],
    "session_name": "Interview session",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.get("pending_session_import"):
    restore_session(st.session_state, st.session_state.pop("pending_session_import"))
    st.success("Session restored. Imported history and review notes are user data, not new source evidence.")

apply_retrieval_policy(st.session_state)
library = build_story_library(chunks, source_locations=runtime_corpus.source_locations)
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
        help="Paste up to 20,000 characters. Apply to use it throughout this session.",
    )
    apply_focus = st.form_submit_button("Apply job focus")

if apply_focus:
    try:
        build_job_context(selected_focus, selected_background, selected_description)
    except ValueError as exc:
        st.sidebar.error(str(exc))
    else:
        if (selected_focus, selected_background.strip(), selected_description.strip()) != (
            st.session_state.job_focus, st.session_state.job_background, st.session_state.job_description
        ):
            st.session_state.job_focus = selected_focus
            st.session_state.job_background = selected_background.strip()
            st.session_state.job_description = selected_description.strip()
            clear_candidate_pool(st.session_state)
            for key in ("pending_answer", "pending_search", "answer_error", "search_error", "practice_feedback"):
                st.session_state.pop(key, None)
        st.sidebar.success("Interview background applied. Ask a question or browse your stories.")

job_focus = st.session_state.job_focus
job_background = st.session_state.job_background
job_description = st.session_state.job_description
job_context = build_job_context(job_focus, job_background, job_description)
st.caption(f"Current job focus: {job_focus}")
if job_description:
    st.caption("Job description applied. Story selection and answers will be tailored to this posting.")
render_evidence(library, job_description)
mode = st.radio("Mode", ("Answer assistant", "Practice"), horizontal=True)
if mode == "Practice":
    render_practice(client, library, job_focus, job_context)
    render_session_controls()
    st.stop()

browse_stories = st.button("Show stories for this role")
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

st.markdown("### Ask your question")
audio = mic_recorder(
    start_prompt=" Start Recording (Tap Once)", stop_prompt=" Stop Recording",
    use_container_width=True, just_once=True, key="question_mic", format="wav",
)
user_query = None
stories_only = False
if browse_stories:
    role_label = "the pasted job description" if job_description else "my experience"
    if job_focus != "All roles":
        role_label = job_focus
    user_query = f"Show me a relevant story for {role_label}."
    stories_only = True
if audio:
    st.session_state.question_recording = audio["bytes"]
    st.session_state.transcribe_question = True
if st.session_state.get("transcription_error"):
    st.error(st.session_state.transcription_error)
    if st.button("Retry question transcription"):
        st.session_state.transcribe_question = True
if st.session_state.pop("transcribe_question", False):
    try:
        if client is None:
            raise InterviewServiceError("Transcription requires server API credentials.")
        user_query = transcribe(client, st.session_state.question_recording)
        st.session_state.pop("transcription_error", None)
    except InterviewServiceError as exc:
        st.session_state.transcription_error = str(exc)
        st.rerun()

prompt = st.chat_input("Or type your question here...")
if prompt:
    user_query = prompt
    stories_only = False
another_story = st.button("Give me another answer or story")
search_request = None
if user_query:
    if len(user_query) > 2000:
        st.error("Please shorten the question to 2,000 characters or fewer.")
    else:
        st.chat_message("user").markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        search_request = {"question": user_query, "stories_only": stories_only}
if st.session_state.get("search_error"):
    st.error(st.session_state.search_error)
    if st.button("Retry story search"):
        search_request = st.session_state.get("pending_search")

selected_story = None
query_to_answer = None
if search_request:
    clear_candidate_pool(st.session_state)
    for key in ("active_answer", "answer_audio", "audio_error"):
        st.session_state.pop(key, None)
    st.session_state.pending_search = search_request
    st.session_state.pop("pending_answer", None)
    st.session_state.pop("answer_error", None)
    try:
        with st.spinner("Finding relevant experience..."):
            candidates = search(
                search_request["question"], index, chunks, embeddings, k=6,
                focus=job_focus, background=job_background,
                stories_only=search_request["stories_only"], job_description=job_description,
            )
        st.session_state.candidate_stories = candidates
        st.session_state.active_query = search_request["question"]
        st.session_state.pop("pending_search", None)
        st.session_state.pop("search_error", None)
        if candidates:
            selected_story = coerce_candidate(candidates[0])
            query_to_answer = search_request["question"]
        else:
            st.info("No saved stories matched this question and job focus. Try a different question or All roles.")
    except InterviewServiceError as exc:
        st.session_state.search_error = str(exc)
        st.rerun()
elif another_story:
    if not st.session_state.candidate_stories:
        st.warning("Ask a question first so I can find matching answers.")
    elif st.session_state.candidate_story_pos + 1 < len(st.session_state.candidate_stories):
        st.session_state.candidate_story_pos += 1
        selected_story = coerce_candidate(st.session_state.candidate_stories[st.session_state.candidate_story_pos])
        query_to_answer = st.session_state.active_query
        st.session_state.messages.append({"role": "user", "content": "Give me another answer for the same question."})
    else:
        st.info("You have reached the end of these matching stories. Ask a new question to search again.")

request_answer = bool(selected_story and query_to_answer)
if request_answer:
    st.session_state.pending_answer = {
        "question": query_to_answer, "candidate": selected_story, "job_context": job_context,
    }
    st.session_state.pop("answer_error", None)
if st.session_state.get("answer_error"):
    st.error(st.session_state.answer_error)
    if st.button("Retry answer"):
        request_answer = True
if request_answer and st.session_state.get("pending_answer"):
    pending = st.session_state.pending_answer
    try:
        with st.spinner("Answering..."):
            text = render_answer(client, pending["question"], pending["candidate"], pending["job_context"])
        if pending["candidate"]["response_type"] == "story" and not pending["candidate"].get("assigned_match"):
            text = enforce_single_star_story(text)
        st.session_state.active_answer = {
            **pending, "text": text,
            "story_id": story_id(pending["candidate"]["text"]) if pending["candidate"]["response_type"] == "story" else "Direct answer",
        }
        st.session_state.pop("answer_audio", None)
        st.session_state.pop("audio_error", None)
        st.session_state.pop("answer_error", None)
        st.session_state.pop("pending_answer", None)
        st.session_state.messages.append({"role": "assistant", "content": text})
        st.chat_message("assistant").markdown(text)
    except InterviewServiceError as exc:
        st.session_state.answer_error = str(exc)
        st.rerun()

# Actions remain available on reruns without repeating retrieval or generation.
active = st.session_state.get("active_answer")
if active:
    candidate = active["candidate"]
    st.caption(f"Selected answer for: {active['question']}")
    st.markdown(f"**{'Detected Principle' if candidate['response_type'] == 'story' else 'Answer Category'}:** {display_category(candidate)}")
    st.caption(f"Saved evidence: {active['story_id']}. Claims are not independently verified.")
    if candidate.get("assigned_match"):
        saved_label = "Saved backup for this question" if candidate.get("backup_for") else "Saved answer for this question"
        st.caption(f"{candidate['question_key']} · {saved_label}")
    if st.session_state.story_reviews.get(active["story_id"], {}).get("review_status") == "Needs correction":
        st.warning("You marked this source as needing correction. Review the source before using this earlier answer.")
    if st.checkbox("Show retrieved context"):
        st.markdown("**Saved answer source**" if candidate.get("assigned_match") else "**Selected context sent to GPT**")
        st.code(candidate["text"], language="text")
        st.text(active["job_context"] or "No job background was applied to this answer.")
    st.download_button("Download this answer", answer_markdown(active), "interview-answer.md", "text/markdown")
    if st.button("Make answer audio", disabled=client is None):
        try:
            with st.spinner("Generating audio..."):
                speech = client.audio.speech.create(model="gpt-4o-mini-tts", voice="alloy", input=active["text"])
            if not isinstance(speech.content, bytes) or not speech.content:
                raise ValueError("Empty audio response")
            st.session_state.answer_audio = speech.content
            st.session_state.pop("audio_error", None)
        except Exception as exc:
            st.session_state.audio_error = str(service_error("Answer audio", exc))
    if st.session_state.get("audio_error"):
        st.error(st.session_state.audio_error)
        st.caption("Use Make answer audio to retry. The selected answer is preserved.")
    if st.session_state.get("answer_audio"):
        st.audio(st.session_state.answer_audio, format="audio/mp3")
        st.download_button("Download answer audio", st.session_state.answer_audio, "interview-answer.mp3", "audio/mpeg")

render_session_controls()
