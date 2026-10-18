"""Explicit, validated JSON exports for private interview sessions."""

import json
import re
from collections.abc import MutableMapping
from datetime import datetime
from typing import Any

from interview_evidence import RESULT_STATUSES, REVIEW_STATUSES
from interview_retrieval import JOB_FOCUSES, MAX_JOB_DESCRIPTION_CHARS, clear_candidate_pool

MAX_SESSION_BYTES = 1_000_000
MAX_MESSAGES = 100


def _text(value: Any, field: str, limit: int) -> str:
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"{field} must be text of at most {limit:,} characters.")
    return value


def validate_session(data: Any) -> dict[str, Any]:
    """Allow only session data; imports cannot introduce prompts, files, or stories."""
    fields = {"version", "name", "job_focus", "job_background", "job_description", "messages", "reviews", "practice_history"}
    if not isinstance(data, dict) or set(data) - fields or type(data.get("version")) is not int or data["version"] != 1:
        raise ValueError("Unsupported session format. Import a version 1 InterviewBot session.")
    focus = data.get("job_focus", "All roles")
    if not isinstance(focus, str) or focus not in JOB_FOCUSES:
        raise ValueError("The saved job focus is not supported.")
    result = {
        "version": 1,
        "name": _text(data.get("name", "Interview session"), "Session name", 100),
        "job_focus": focus,
        "job_background": _text(data.get("job_background", ""), "Job details", 1500),
        "job_description": _text(data.get("job_description", ""), "Job description", MAX_JOB_DESCRIPTION_CHARS),
    }
    messages = data.get("messages", [])
    if not isinstance(messages, list) or len(messages) > MAX_MESSAGES:
        raise ValueError("A session can contain at most 100 chat messages.")
    result["messages"] = []
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"} or message["role"] not in ("user", "assistant"):
            raise ValueError("Chat history may contain only user and assistant text messages.")
        result["messages"].append({"role": message["role"], "content": _text(message["content"], "Message", 16000)})
    reviews = data.get("reviews", {})
    if not isinstance(reviews, dict) or len(reviews) > 200:
        raise ValueError("Invalid story review collection.")
    result["reviews"] = {}
    for identifier, review in reviews.items():
        _text(identifier, "Story ID", 100)
        review_fields = {"review_status", "result_status", "note", "reviewed_at"}
        if not isinstance(review, dict) or not review_fields.issubset(review) or set(review) - review_fields - {"source_hashes"}:
            raise ValueError("Invalid story review metadata.")
        if review["review_status"] not in REVIEW_STATUSES or review["result_status"] not in RESULT_STATUSES:
            raise ValueError("Unrecognized review or result status.")
        reviewed_at = _text(review["reviewed_at"], "Review date", 60)
        if reviewed_at:
            try:
                parsed_date = datetime.fromisoformat(reviewed_at)
                if parsed_date.tzinfo is None:
                    raise ValueError("Timezone required")
            except ValueError as exc:
                raise ValueError("Review dates must use ISO 8601 with a timezone.") from exc
        hashes = review.get("source_hashes", [])
        if not isinstance(hashes, list) or len(hashes) > 200 or any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes
        ):
            raise ValueError("Review source hashes must be SHA-256 strings.")
        result["reviews"][identifier] = {
            "review_status": review["review_status"], "result_status": review["result_status"],
            "note": _text(review["note"], "Review note", 2000),
            "reviewed_at": reviewed_at, "source_hashes": sorted(set(hashes)),
        }
    attempts = data.get("practice_history", [])
    if not isinstance(attempts, list) or len(attempts) > 30:
        raise ValueError("A session can contain at most 30 practice attempts.")
    result["practice_history"] = []
    for attempt in attempts:
        limits = {"question": 2000, "answer": 6000, "feedback": 12000, "story_id": 100}
        if not isinstance(attempt, dict) or set(attempt) != set(limits):
            raise ValueError("Invalid practice attempt.")
        result["practice_history"].append({key: _text(attempt[key], key, limit) for key, limit in limits.items()})
    return result


def session_bytes(state: MutableMapping[str, Any], include_history: bool = False) -> bytes:
    """Export approved fields only; omit conversations and attempts by default."""
    data = {"version": 1, "name": state.get("session_name", "Interview session")}
    for field in ("job_focus", "job_background", "job_description"):
        data[field] = state.get(field, "All roles" if field == "job_focus" else "")
    data["reviews"] = state.get("story_reviews", {})
    data["messages"] = state.get("messages", [])[-MAX_MESSAGES:] if include_history else []
    data["practice_history"] = state.get("practice_history", [])[-30:] if include_history else []
    payload = json.dumps(validate_session(data), ensure_ascii=False, indent=2).encode("utf-8")
    if len(payload) > MAX_SESSION_BYTES:
        raise ValueError("Session export exceeds 1 MB. Export without conversation history.")
    return payload


def load_session(payload: bytes) -> dict[str, Any]:
    """Parse bounded UTF-8 JSON without executing imported content."""
    if len(payload) > MAX_SESSION_BYTES:
        raise ValueError("Session file exceeds the 1 MB limit.")
    try:
        data = json.loads(payload.decode("utf-8"))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("Session file must contain valid UTF-8 JSON.") from exc
    return validate_session(data)


def restore_session(state: MutableMapping[str, Any], data: dict) -> None:
    """Apply a validated session and discard stale executable/UI state."""
    validated = validate_session(data)
    for field in ("job_focus", "job_background", "job_description", "messages", "practice_history"):
        state[field] = validated[field]
    state["session_name"] = validated["name"]
    state["story_reviews"] = validated["reviews"]
    # New widget identities prevent a previous session's unsaved review choices
    # from overriding imported metadata on the next form submission.
    state["review_form_revision"] = state.get("review_form_revision", 0) + 1
    clear_candidate_pool(state)
    for key in ("active_answer", "pending_answer", "pending_search", "answer_audio", "answer_error", "search_error", "audio_error", "practice_feedback", "practice_feedback_signature", "practice_answer", "practice_source", "practice_question", "practice_loaded_source", "practice_recording", "practice_audio", "practice_transcript", "practice_error", "question_recording", "transcription_error"):
        state.pop(key, None)
    state["job_description_draft"] = validated["job_description"]
