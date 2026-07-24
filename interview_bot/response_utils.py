from dataclasses import dataclass
from typing import Any


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


@dataclass(frozen=True)
class ChatCompletionAssessment:
    text: str
    finish_reason: str
    refusal: str = ""

    @property
    def usable(self) -> bool:
        return (
            bool(self.text)
            and not self.refusal
            and self.finish_reason in {"", "stop"}
        )

    @property
    def retryable(self) -> bool:
        return (
            not self.usable
            and not self.refusal
            and self.finish_reason not in {"content_filter"}
        )

    @property
    def diagnostic(self) -> str:
        if self.refusal:
            return "the model declined the request"
        if self.finish_reason == "length":
            return "the answer reached its output limit"
        if self.finish_reason == "content_filter":
            return "the answer was stopped by the content filter"
        if self.finish_reason and self.finish_reason != "stop":
            return f"the response ended with {self.finish_reason}"
        return "the response contained no answer text"


def assess_chat_completion(response: Any) -> ChatCompletionAssessment:
    choices = _field(response, "choices", []) or []
    if not choices:
        return ChatCompletionAssessment(text="", finish_reason="")

    choice = choices[0]
    message = _field(choice, "message", None)
    return ChatCompletionAssessment(
        text=str(_field(message, "content", "") or "").strip(),
        finish_reason=str(_field(choice, "finish_reason", "") or "").strip().lower(),
        refusal=str(_field(message, "refusal", "") or "").strip(),
    )
