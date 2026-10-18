"""Bounded interview API requests and evidence-grounded prompts."""

import io
import json
import logging
import os
import re
import wave
from typing import Any

from interview_response_utils import assess_chat_completion
from interview_retrieval import JOB_CONTEXT_INSTRUCTIONS

LOGGER = logging.getLogger(__name__)
FEEDBACK_FIELDS = ("situation", "task", "action", "result", "ownership", "relevance", "evidence_check", "follow_up")


class InterviewServiceError(RuntimeError):
    """A safe user-facing failure that does not expose request data or credentials."""


def service_error(operation: str, error: Exception) -> InterviewServiceError:
    """Translate external failures without printing private exception payloads."""
    LOGGER.warning("Interview %s failed (%s)", operation, type(error).__name__)
    status = getattr(error, "status_code", None)
    if status in {401, 403}:
        detail = "Check the server's API credentials and access, then retry."
    elif status == 429:
        detail = "The service is busy or its quota is exhausted. Wait or check the account quota, then retry."
    else:
        detail = "Check the network or service availability, then retry. Your text is preserved."
    return InterviewServiceError(f"{operation} failed. {detail}")


def request_text(client: Any, messages: list[dict], *, structured: bool = False) -> str:
    """Request text, allowing one larger attempt for blank or truncated output."""
    for limit in (2048, 3072):
        try:
            options = {"response_format": {"type": "json_object"}} if structured else {}
            response = client.chat.completions.create(
                model=os.getenv("INTERVIEW_CHAT_MODEL", "gpt-6-astra"),
                messages=messages, max_completion_tokens=limit, reasoning_effort="low", **options,
            )
        except Exception as exc:
            raise service_error("Answer request", exc) from exc
        assessment = assess_chat_completion(response)
        if assessment.usable:
            return assessment.text
        if not assessment.retryable:
            break
    raise InterviewServiceError(f"No usable answer was returned because {assessment.diagnostic}. Retry the request.")


def answer_messages(question: str, candidate: dict, job_context: str) -> list[dict]:
    """Keep the selected saved source separate from job requirements and history."""
    if candidate["response_type"] == "story":
        format_instruction = "Use four concise paragraphs labeled Situation, Task, Action, Result."
    else:
        format_instruction = "Use two to four concise paragraphs without forcing STAR labels."
    return [
        {"role": "system", "content": (
            "Answer as Amir Exir in first person. " + format_instruction +
            " Use only the single saved source in selected_context. Never borrow facts from another "
            "story or prior answer. Preserve qualifiers such as potential, estimated, and approximately. "
            "If the selected source does not establish an answer, say what evidence is missing. " + JOB_CONTEXT_INSTRUCTIONS
        )},
        {"role": "user", "content": (
            f"Question: {question}\n\n<job_background>\n{job_context}\n</job_background>\n\n"
            f"<selected_context>\n{candidate['text']}\n</selected_context>"
        )},
    ]


def practice_messages(question: str, answer: str, source: str, job_context: str) -> list[dict]:
    """Ask for specific feedback, not a replacement story or invented achievements."""
    return [
        {"role": "system", "content": (
            "You are an interview practice coach. Return only JSON with these string fields: "
            + ", ".join(FEEDBACK_FIELDS) + ". Each field must be concise (at most 1200 characters). "
            "Assess each STAR component, personal ownership and relevance to the question. "
            "Quote short phrases from the attempted answer when helpful. Use selected_context as the "
            "only saved evidence; it is self-reported and not independently verified. In evidence_check "
            "identify claims unsupported by that source, particularly amounts and actual versus potential "
            "results. Do not treat a new assertion in the attempt as verified. Give practical corrections "
            "without inventing facts. follow_up must contain exactly one probing interview question. "
            "Do not give readiness percentages or a replacement answer. " + JOB_CONTEXT_INSTRUCTIONS
        )},
        {"role": "user", "content": json.dumps({
            "interview_question": question, "attempted_answer": answer,
            "selected_context": source, "job_background": job_context,
        }, ensure_ascii=False)},
    ]


def parse_feedback(text: str) -> dict[str, str]:
    """Reject malformed coaching output rather than presenting partial feedback."""
    try:
        value = json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise InterviewServiceError("Feedback was not valid JSON. Retry feedback; your answer is preserved.") from exc
    if not isinstance(value, dict) or set(value) != set(FEEDBACK_FIELDS):
        raise InterviewServiceError("Feedback omitted required sections. Retry feedback; your answer is preserved.")
    if any(not isinstance(item, str) or not item.strip() or len(item) > 1200 for item in value.values()):
        raise InterviewServiceError("Feedback contained invalid sections. Retry feedback; your answer is preserved.")
    if value["follow_up"].count("?") != 1:
        raise InterviewServiceError("Feedback must contain one follow-up question. Retry feedback.")
    return value


def transcribe(client: Any, audio: bytes) -> str:
    """Transcribe bounded audio in memory without storing it on the server."""
    if not isinstance(audio, bytes) or not audio or len(audio) > 15_000_000:
        raise InterviewServiceError("Record a shorter answer; audio must be nonempty and under 15 MB.")
    recording = io.BytesIO(audio)
    recording.name = "speech.wav"
    try:
        result = client.audio.transcriptions.create(model="whisper-1", file=recording)
    except Exception as exc:
        raise service_error("Transcription", exc) from exc
    text = getattr(result, "text", None)
    if not isinstance(text, str):
        raise InterviewServiceError("Transcription returned invalid text. Retry transcription or type your answer.")
    text = text.strip()
    if not text:
        raise InterviewServiceError("No speech was transcribed. Try recording again or type your answer.")
    return text


def practice_timing(text: str, audio: bytes | None = None) -> dict[str, Any]:
    """Measure valid WAV duration, otherwise estimate delivery at 130 words/minute."""
    words = len(re.findall(r"\b[\w'-]+\b", text))
    seconds = words / 130 * 60
    measured = False
    if audio:
        try:
            with wave.open(io.BytesIO(audio), "rb") as recording:
                seconds = recording.getnframes() / recording.getframerate()
                measured = True
        except (wave.Error, EOFError, ZeroDivisionError):
            pass
    return {"words": words, "seconds": round(seconds), "measured": measured}
