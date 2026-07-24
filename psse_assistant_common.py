"""Pure runtime safeguards shared by the PSS/E assistants.

The helpers in this module deliberately have no Streamlit, OpenAI, NumPy, or
filesystem dependencies so their failure behavior can be tested without
starting either app or making an API request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def extract_response_text(response: Any) -> str:
    """Return visible text from current and older Responses SDK objects."""

    direct = _field(response, "output_text", "")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    texts: list[str] = []
    for output_item in _field(response, "output", ()) or ():
        if str(_field(output_item, "type", "")).casefold() != "message":
            continue
        for content_item in _field(output_item, "content", ()) or ():
            text = _field(content_item, "text", "")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    return "\n".join(texts).strip()


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
    """Inspect status and visible text before treating generation as successful."""

    status = str(_field(response, "status", "") or "").strip().casefold()
    incomplete_details = _field(response, "incomplete_details")
    incomplete_reason = str(
        _field(incomplete_details, "reason", "") or ""
    ).strip()
    error = _field(response, "error")
    error_message = str(_field(error, "message", "") or "").strip()
    return ResponseAssessment(
        text=extract_response_text(response),
        status=status,
        incomplete_reason=incomplete_reason,
        error_message=error_message,
        refusal=_response_refusal(response),
    )


@dataclass(frozen=True)
class GenerationResult:
    """Result of a bounded visible-answer request."""

    assessment: ResponseAssessment
    retried: bool = False

    @property
    def usable(self) -> bool:
        return self.assessment.usable

    @property
    def text(self) -> str:
        return self.assessment.text

    @property
    def diagnostic(self) -> str:
        return self.assessment.diagnostic


def _exception_assessment(exc: Exception) -> ResponseAssessment:
    # Do not surface request bodies or provider responses in the UI.
    label = type(exc).__name__ or "API error"
    return ResponseAssessment(
        text="",
        status="failed",
        incomplete_reason="",
        error_message=f"the model request failed ({label})",
        refusal="",
    )


def request_visible_answer(
    create_response: Callable[..., Any],
    request: Mapping[str, Any],
    *,
    retry_request: Mapping[str, Any] | None = None,
) -> GenerationResult:
    """Make one request and at most one anomaly retry.

    The OpenAI client already handles transport-level retries. This helper only
    retries a completed API call that returned no usable visible answer, which
    prevents an unbounded agent loop and avoids retrying refusals or hard errors.
    """

    try:
        first = assess_response(create_response(**dict(request)))
    except Exception as exc:
        return GenerationResult(_exception_assessment(exc))

    if first.usable or not first.retryable or retry_request is None:
        return GenerationResult(first)

    try:
        second = assess_response(create_response(**dict(retry_request)))
    except Exception as exc:
        return GenerationResult(_exception_assessment(exc), retried=True)
    return GenerationResult(second, retried=True)


def compact_chat_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    max_messages: int = 6,
    max_characters_per_message: int = 6_000,
) -> list[dict[str, str]]:
    """Keep recent conversation without replaying unbounded prior answers."""

    if max_messages < 1 or max_characters_per_message < 1:
        return []
    compacted: list[dict[str, str]] = []
    for message in messages[-max_messages:]:
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        if len(content) > max_characters_per_message:
            content = content[:max_characters_per_message].rstrip() + "…"
        compacted.append({"role": role, "content": content})
    return compacted


def validate_saved_index(
    chunks: Sequence[Mapping[str, Any]],
    embeddings: Any,
    *,
    expected_dimension: int | None = None,
) -> None:
    """Reject missing, mismatched, or malformed saved RAG artifacts."""

    if not chunks:
        raise ValueError("the saved chunk file is empty")
    if any(not str(chunk.get("text") or "").strip() for chunk in chunks):
        raise ValueError("the saved chunk file contains an empty text record")

    shape = tuple(getattr(embeddings, "shape", ()) or ())
    if len(shape) != 2:
        raise ValueError("the saved embedding array must be two-dimensional")
    if shape[0] != len(chunks):
        raise ValueError(
            "saved chunk/embedding row mismatch: "
            f"{len(chunks)} chunks versus {shape[0]} embedding rows"
        )
    if shape[1] < 1:
        raise ValueError("the saved embedding array has no vector dimensions")
    if expected_dimension is not None and shape[1] != expected_dimension:
        raise ValueError(
            f"saved embedding dimension is {shape[1]}, expected {expected_dimension}"
        )


_TASK_PREFIX = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s*")


def parse_planner_tasks(planner_text: str, *, max_tasks: int = 12) -> list[str]:
    """Parse a bounded task list and reject planner-error text."""

    text = str(planner_text or "").strip()
    if not text or text.casefold().startswith("[planner error]") or max_tasks < 1:
        return []

    tasks: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        task = _TASK_PREFIX.sub("", raw_line).strip()
        if not task or task.startswith(("#", "```")) or task.endswith(":"):
            continue
        normalized = task.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        tasks.append(task)
        if len(tasks) >= max_tasks:
            break
    return tasks
