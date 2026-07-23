"""Pure helpers for validating and compacting Responses API results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _response_refusal(response: Any) -> str:
    for output_item in _field(response, "output", ()) or ():
        for content_item in _field(output_item, "content", ()) or ():
            if str(_field(content_item, "type", "")).casefold() != "refusal":
                continue
            refusal = _field(content_item, "refusal", "")
            if refusal:
                return str(refusal).strip()
    return ""


@dataclass(frozen=True)
class ResponseAssessment:
    """Normalized visible-answer state from an OpenAI Responses object."""

    text: str
    status: str
    incomplete_reason: str
    error_message: str
    refusal: str

    @property
    def usable(self) -> bool:
        return bool(self.text) and self.status in {"", "completed"}

    @property
    def retryable(self) -> bool:
        return (
            not self.usable
            and not self.refusal
            and not self.error_message
            and self.status not in {"failed", "cancelled"}
        )

    @property
    def diagnostic(self) -> str:
        if self.refusal:
            return "the model declined to provide an answer"
        if self.error_message:
            return self.error_message
        if self.status == "incomplete":
            detail = f" ({self.incomplete_reason})" if self.incomplete_reason else ""
            return f"the model response was incomplete{detail}"
        if self.status and self.status != "completed":
            return f"the model response ended with status {self.status}"
        return "the model returned no visible answer text"


def assess_response(response: Any) -> ResponseAssessment:
    """Inspect status and visible text before citation validation."""

    status = str(_field(response, "status", "") or "").strip().casefold()
    text = str(_field(response, "output_text", "") or "").strip()
    incomplete_details = _field(response, "incomplete_details")
    incomplete_reason = str(
        _field(incomplete_details, "reason", "") or ""
    ).strip()
    error = _field(response, "error")
    error_message = str(
        _field(error, "message", "") or ""
    ).strip()
    return ResponseAssessment(
        text=text,
        status=status,
        incomplete_reason=incomplete_reason,
        error_message=error_message,
        refusal=_response_refusal(response),
    )


def compact_chat_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    max_messages: int = 6,
    max_characters_per_message: int = 6_000,
) -> list[dict[str, str]]:
    """Retain recent conversational intent without replaying source footers."""

    if max_messages < 1 or max_characters_per_message < 1:
        return []
    compacted: list[dict[str, str]] = []
    for message in messages[-max_messages:]:
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        content = content.split("\n\n**Retrieved sources**", 1)[0].rstrip()
        if len(content) > max_characters_per_message:
            content = content[:max_characters_per_message].rstrip() + "…"
        compacted.append({"role": role, "content": content})
    return compacted
