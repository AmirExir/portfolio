"""Explicit session saving, evidence review, and practice controls."""

from datetime import datetime, timezone
import hashlib
import json
from typing import Any

import streamlit as st
from streamlit_mic_recorder import mic_recorder

from interview_evidence import (
    RESULT_STATUSES, REVIEW_STATUSES, coverage_map, planning_quick_view, review_for_story, story_id,
)
from interview_retrieval import clear_candidate_pool, topic_score
from interview_services import (
    InterviewServiceError, parse_feedback, practice_messages, practice_timing,
    request_text, transcribe,
)
from interview_sessions import load_session, session_bytes


def _discard_question_recording() -> None:
    """Clear a voice draft while retaining its token to prevent replay on rerun."""
    for key in ("question_transcript_draft", "question_transcript_editor", "question_recording", "transcription_error",
                "transcribe_question", "question_draft_error"):
        st.session_state.pop(key, None)


def _save_question_draft() -> None:
    """Retain edits even when changing modes removes the transcript widget."""
    st.session_state.question_transcript_draft = st.session_state.question_transcript_editor


def _submit_question_recording() -> None:
    """Queue only the user's reviewed text, then clear the recording and draft."""
    _save_question_draft()
    question = st.session_state.question_transcript_draft.strip()
    if len(question) > 2000:
        st.session_state.question_draft_error = "Please shorten the question to 2,000 characters or fewer. Your draft is preserved."
        return
    st.session_state.submitted_voice_question = question
    _discard_question_recording()


def render_question_recording(client: Any) -> str | None:
    """Transcribe each recording once and submit only an explicitly reviewed draft."""
    st.caption("Record in English. Review and edit the transcript before using it as your question.")
    audio = mic_recorder(
        start_prompt=" Start Recording (Tap Once)", stop_prompt=" Stop Recording",
        use_container_width=True, just_once=True, key="question_mic", format="wav",
    )
    if audio:
        recording = audio.get("bytes")
        if not isinstance(recording, bytes) or not recording or len(recording) > 15_000_000:
            _discard_question_recording()
            st.session_state.transcription_error = "Record a shorter question; audio must be nonempty and under 15 MB."
        else:
            token = (audio.get("id"), hashlib.sha256(recording).hexdigest())
            if token != st.session_state.get("question_recording_token"):
                _discard_question_recording()
                st.session_state.question_recording_token = token
                st.session_state.question_recording = recording
                st.session_state.transcribe_question = True
    if st.session_state.pop("transcribe_question", False):
        try:
            if client is None:
                raise InterviewServiceError("Transcription requires server API credentials.")
            st.session_state.question_transcript_draft = transcribe(client, st.session_state.question_recording)
            st.session_state.pop("transcription_error", None)
        except InterviewServiceError as exc:
            st.session_state.transcription_error = str(exc)
    if st.session_state.get("transcription_error"):
        st.error(st.session_state.transcription_error)
        if st.session_state.get("question_recording") and st.button("Retry question transcription"):
            st.session_state.transcribe_question = True
            st.rerun()
    if "question_transcript_draft" in st.session_state:
        if st.session_state.get("question_draft_error"):
            st.info(st.session_state.question_draft_error)
        if "question_transcript_editor" not in st.session_state:
            st.session_state.question_transcript_editor = st.session_state.question_transcript_draft
        st.text_area("Review your question", key="question_transcript_editor", height=120,
                     on_change=_save_question_draft)
        st.button("Use this question", on_click=_submit_question_recording)
    if any(key in st.session_state for key in ("question_recording", "question_transcript_draft", "transcription_error")):
        st.button("Discard recording", on_click=_discard_question_recording)
    return st.session_state.pop("submitted_voice_question", None)


def render_planning_quick_view(chunks: list[dict]) -> str | None:
    """Show the planning keyword map and return an explicitly chosen shortcut."""
    rows = planning_quick_view(chunks)
    if not rows:
        return None
    flagged_ids = {
        chunk.get("source_id") for chunk in chunks
        if chunk.get("response_type") == "story"
        and st.session_state.story_reviews.get(story_id(chunk["text"]), {}).get("review_status")
        == "Needs correction"
    }
    with st.expander("Planning interview cheat sheet"):
        st.caption(
            "One keyword, one assigned story or approach. Use the recall cue to rehearse, "
            "then open the full answer and its follow-up preparation. "
            "Proposed approaches and illustrative scenarios use ‘I would.’"
        )
        st.dataframe(
            [{
                "Shortcut": row["question_key"],
                "Story / approach": ("Needs correction · " if row["source_id"] in flagged_ids else "") + row["title"],
                "Recall cue": "Review this story before rehearsing." if row["source_id"] in flagged_ids else row["cue"],
                "Basis": row["basis"],
            } for row in rows],
            hide_index=True, use_container_width=True, height=440,
        )
        by_key = {row["question_key"]: row for row in rows}
        if st.session_state.get("planning_quick_view_question") not in by_key:
            st.session_state.pop("planning_quick_view_question", None)
        selected_key = st.selectbox(
            "Choose a keyword", list(by_key), key="planning_quick_view_question",
            format_func=lambda key: f"{key} · {by_key[key]['title']}",
        )
        selected = by_key[selected_key]
        st.markdown(f"**{selected['title']}** · {selected['basis']}")
        if selected["source_id"] in flagged_ids:
            st.warning(
                "You marked this story as needing correction. "
                "Review it in Story evidence and job coverage before rehearsing."
            )
        else:
            st.write(selected["cue"])
        if st.button("Show mapped answer"):
            return selected_key
    return None


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
