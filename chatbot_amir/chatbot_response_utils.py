from dataclasses import dataclass
from typing import Any


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _response_refusal(response: Any) -> str:
    for item in _field(response, "output", []) or []:
        for content in _field(item, "content", []) or []:
            if _field(content, "type", "") == "refusal":
                refusal = _field(content, "refusal", "")
                if refusal:
                    return str(refusal).strip()
    return ""


@dataclass(frozen=True)
class ResponseAssessment:
    text: str
    status: str
    incomplete_reason: str = ""
    refusal: str = ""
    error_message: str = ""

    @property
    def usable(self) -> bool:
        return bool(self.text) and self.status in {"", "completed"}

    @property
    def retryable(self) -> bool:
        return (
            not self.usable
            and not self.refusal
            and not self.error_message
            and self.status not in {"cancelled", "failed"}
        )

    @property
    def diagnostic(self) -> str:
        if self.refusal:
            return "the model declined the request"
        if self.error_message:
            return "the model reported an error"
        if self.incomplete_reason:
            return f"the response was incomplete ({self.incomplete_reason})"
        if self.status and self.status != "completed":
            return f"the response status was {self.status}"
        return "the response contained no answer text"


def assess_response(response: Any) -> ResponseAssessment:
    text = str(_field(response, "output_text", "") or "").strip()
    status = str(_field(response, "status", "") or "").strip().lower()
    incomplete_details = _field(response, "incomplete_details", None)
    incomplete_reason = str(
        _field(incomplete_details, "reason", "") or ""
    ).strip()
    error = _field(response, "error", None)
    error_message = str(_field(error, "message", "") or "").strip()
    return ResponseAssessment(
        text=text,
        status=status,
        incomplete_reason=incomplete_reason,
        refusal=_response_refusal(response),
        error_message=error_message,
    )


def compact_messages(messages: list[dict], max_recent_messages: int = 6) -> list[dict]:
    """Keep the resume system prompt and only the most recent conversation turns."""
    if max_recent_messages < 1:
        raise ValueError("max_recent_messages must be at least 1")

    system_message = next(
        (message for message in messages if message.get("role") == "system"),
        None,
    )
    recent = [
        message
        for message in messages
        if message.get("role") in {"user", "assistant"}
        and str(message.get("content", "")).strip()
    ][-max_recent_messages:]
    if recent and recent[0].get("role") == "assistant":
        recent = recent[1:]
    return ([system_message] if system_message else []) + recent
