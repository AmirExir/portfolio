"""Explicit session saving, evidence review, and practice controls."""

from datetime import datetime, timezone
import json

import streamlit as st
from streamlit_mic_recorder import mic_recorder

from interview_evidence import RESULT_STATUSES, REVIEW_STATUSES, coverage_map, review_for_story
from interview_retrieval import clear_candidate_pool, topic_score
from interview_services import (
    InterviewServiceError, parse_feedback, practice_messages, practice_timing,
    request_text, transcribe,
)
from interview_sessions import load_session, session_bytes


def render_session_controls() -> None:
    """Download explicitly selected session content or stage a validated import."""
    with st.sidebar.expander("Save or restore a job session"):
        st.text_input("Session name", key="session_name", max_chars=100)
        include_history = st.checkbox("Include conversation and practice history", value=False)
        st.caption(
            "Downloads contain job details and your review notes; history is optional. "
            "History exports include the latest 100 chat messages and 30 practice attempts. "
            "Save the file privately. Audio and API credentials are excluded. "
            "No session file is automatically saved on the server."
        )
        try:
            payload = session_bytes(st.session_state, include_history)
            st.download_button("Download job session", payload, "interview-session.json", "application/json")
        except ValueError as exc:
            st.error(str(exc))
        upload = st.file_uploader("Import a saved job session", type=["json"])
        if st.button("Restore session", disabled=upload is None):
            try:
                st.session_state.pending_session_import = load_session(upload.getvalue())
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.rerun()


def render_evidence(library: list[dict], job_description: str) -> None:
    """Expose exact saved sources, user-authored review metadata, and coverage gaps."""
    with st.expander("Story evidence and job coverage"):
        st.caption(
            "Saved stories are self-reported claims. Source locations show where text is stored, "
            "not independent verification. Phrase matches suggest relevant evidence; they do not "
            "establish that you meet a job requirement."
        )
        if not library:
            st.info("No structured STAR stories are available.")
            return
        by_id = {item["id"]: item for item in library}
        selected_id = st.selectbox(
            "Review a saved story", list(by_id), key="evidence_story",
            format_func=lambda identifier: by_id[identifier]["title"][:130],
        )
        item = by_id[selected_id]
        st.caption(f"Canonical story ID: {selected_id}")
        st.code(item["candidate"]["text"], language="text")
        with st.expander("Source locations and saved variants"):
            for source, variant in zip(item["sources"], item["variants"]):
                st.write(f"{source['file']} array item {source['index']} (zero-based); source ID: {source['source_id'] or 'not recorded'}")
                st.caption(f"Text SHA-256: {source['text_sha256']}")
                st.code(variant, language="text")
        review = review_for_story(item, st.session_state.story_reviews.get(selected_id))
        if review["stale"]:
            st.warning("This review is not bound to the current source text. Review the saved variants again; prior outcome classifications do not apply.")
        source_revision = "-".join(sorted(source["text_sha256"] for source in item["sources"]))
        session_revision = st.session_state.get("review_form_revision", 0)
        with st.form(f"review-{selected_id}-{source_revision}-{session_revision}"):
            status = st.selectbox("My review status", REVIEW_STATUSES, index=REVIEW_STATUSES.index(review["review_status"]))
            result_status = st.selectbox("How I can support the result", RESULT_STATUSES, index=RESULT_STATUSES.index(review["result_status"]))
            note = st.text_area("My evidence or correction notes", review["note"], max_chars=2000)
            if st.form_submit_button("Save review in this session"):
                st.session_state.story_reviews[selected_id] = {
                    "review_status": status, "result_status": result_status, "note": note,
                    "reviewed_at": datetime.now(timezone.utc).isoformat(),
                    "source_hashes": sorted({source["text_sha256"] for source in item["sources"]}),
                }
                if status == "Needs correction":
                    clear_candidate_pool(st.session_state)
                    st.session_state.pop("pending_answer", None)
                st.success("Your review note is saved in this session. Download the session to keep it.")
        if review["reviewed_at"]:
            st.caption(f"User-supplied review date: {review['reviewed_at']}")
        st.caption("Review notes do not rewrite source stories or certify their claims. Correct the source before using a disputed claim.")
        if job_description:
            rows = coverage_map(job_description, library)
            table = [{
                "Posting text": row["posting_text"], "Coverage": row["status"],
                "Related story IDs": ", ".join(match["story_id"] for match in row["matches"][:3]),
                "Matched phrases": ", ".join(sorted({term for match in row["matches"][:3] for term in match["matched_phrases"]})),
            } for row in rows]
            st.dataframe(table, hide_index=True, use_container_width=True)
            st.download_button("Download evidence map", json.dumps(rows, indent=2), "job-evidence-map.json", "application/json")
        else:
            st.info("Apply a job description to see its lines mapped to saved evidence and unmatched areas.")


def render_practice(client: object, library: list[dict], job_focus: str, job_context: str) -> None:
    """Let the user attempt an answer before receiving grounded coaching."""
    st.subheader("Practice your answer")
    st.caption(
        "Aim for 60–90 seconds. Choose one saved story, answer in your own words, then request feedback. "
        "Typed answers and recordings are sent to the configured AI service only when you request transcription or feedback."
    )
    available = [item for item in library
                 if st.session_state.story_reviews.get(item["id"], {}).get("review_status") != "Needs correction"
                 and (job_focus == "All roles" or topic_score(item["candidate"]["text"], job_focus))]
    if not available:
        st.info("No saved story matches this focus. Choose All roles to review the full library.")
        return
    by_id = {item["id"]: item for item in available}
    old_source = st.session_state.get("practice_source")
    if old_source not in by_id:
        st.session_state.pop("practice_source", None)
    identifier = st.selectbox(
        "Story to practice", list(by_id), key="practice_source",
        format_func=lambda value: by_id[value]["title"][:130],
    )
    item = by_id[identifier]
    source_revision = (identifier, tuple(sorted({source["text_sha256"] for source in item["sources"]})))
    if st.session_state.get("practice_loaded_source") != source_revision:
        st.session_state.practice_loaded_source = source_revision
        st.session_state.practice_question = item["candidate"]["question"]
        st.session_state.practice_answer = ""
        for key in ("practice_feedback", "practice_feedback_signature", "practice_audio", "practice_recording", "practice_transcript", "practice_error"):
            st.session_state.pop(key, None)
    st.text_input("Practice question", key="practice_question", max_chars=2000)
    with st.expander("Reference story (optional)"):
        st.code(item["candidate"]["text"], language="text")
        st.caption(f"Evidence ID: {identifier}; source claims have not been independently verified.")
    recording = mic_recorder(
        start_prompt="Record my practice answer", stop_prompt="Stop practice recording",
        just_once=True, key="practice_mic", format="wav",
    )
    if recording:
        st.session_state.practice_recording = recording["bytes"]
        st.session_state.pop("practice_feedback", None)
    if st.session_state.get("practice_recording"):
        st.audio(st.session_state.practice_recording, format="audio/wav")
        if st.button("Transcribe my practice answer", disabled=client is None):
            try:
                transcript = transcribe(client, st.session_state.practice_recording)
                if len(transcript) > 6000:
                    raise InterviewServiceError("Transcript exceeds 6,000 characters. Record a shorter attempt.")
                st.session_state.practice_answer = transcript
                st.session_state.practice_transcript = transcript
                st.session_state.practice_audio = st.session_state.practice_recording
                st.session_state.pop("practice_feedback", None)
                st.session_state.pop("practice_error", None)
            except InterviewServiceError as exc:
                st.session_state.practice_error = str(exc)
    attempt = st.text_area("My practice answer", key="practice_answer", height=200, max_chars=6000)
    measured_audio = st.session_state.get("practice_audio") if attempt == st.session_state.get("practice_transcript") else None
    timing = practice_timing(attempt, measured_audio)
    label = "Recorded duration" if timing["measured"] else "Estimated delivery at 130 words/minute"
    st.caption(f"{timing['words']} words · {label}: {timing['seconds']} seconds · Target: 60–90 seconds")
    signature = (source_revision, st.session_state.practice_question, attempt, job_context)
    if st.session_state.get("practice_feedback_signature") != signature:
        st.session_state.pop("practice_feedback", None)
    if st.button("Get practice feedback", disabled=client is None or not attempt.strip() or not st.session_state.practice_question.strip()):
        try:
            feedback = parse_feedback(request_text(client, practice_messages(
                st.session_state.practice_question, attempt, item["candidate"]["text"], job_context,
            ), structured=True))
            st.session_state.practice_feedback = feedback
            st.session_state.practice_feedback_signature = signature
            st.session_state.pop("practice_error", None)
            st.session_state.practice_history.append({
                "question": st.session_state.practice_question, "answer": attempt,
                "story_id": identifier, "feedback": json.dumps(feedback, ensure_ascii=False),
            })
            st.session_state.practice_history = st.session_state.practice_history[-30:]
        except InterviewServiceError as exc:
            st.session_state.practice_error = str(exc)
    if st.session_state.get("practice_error"):
        st.error(st.session_state.practice_error)
    feedback = st.session_state.get("practice_feedback")
    if feedback:
        st.caption("AI coaching feedback; this is not independent fact verification or a hiring prediction.")
        for field, value in feedback.items():
            st.markdown(f"**{field.replace('_', ' ').title()}:** {value}")
        st.download_button("Download practice feedback", json.dumps(feedback, indent=2), "practice-feedback.json", "application/json")
    if st.session_state.practice_history:
        with st.expander("Previous practice attempts"):
            st.caption("Saved attempts and AI feedback are practice history, not additional source evidence.")
            for number, prior in enumerate(reversed(st.session_state.practice_history), 1):
                st.markdown(f"**Attempt {len(st.session_state.practice_history) - number + 1}:** {prior['question']}")
                st.caption(f"Saved evidence ID: {prior['story_id']}")
                st.text(prior["answer"])
                st.code(prior["feedback"], language="json")
