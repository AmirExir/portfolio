"""Pure Chat Completions response validation for the DWG/SSWG app."""

from __future__ import annotations

from typing import Any, Mapping

from ERCOTAPI.rag_ingestion.response_handling import (
    ResponseAssessment,
    assess_response,
)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def assess_chat_completion(response: Any) -> ResponseAssessment:
    """Normalize a Chat Completions result into the shared response guard."""

    choices = _field(response, "choices", ()) or ()
    if not choices:
        return assess_response(
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "no_choices"},
            }
        )

    choice = choices[0]
    message = _field(choice, "message")
    content = str(_field(message, "content", "") or "").strip()
    refusal = str(_field(message, "refusal", "") or "").strip()
    finish_reason = str(_field(choice, "finish_reason", "") or "").casefold()

    normalized: dict[str, Any] = {
        "status": "completed",
        "output_text": content,
    }
    if refusal:
        normalized["output"] = [
            {
                "content": [
                    {
                        "type": "refusal",
                        "refusal": refusal,
                    }
                ]
            }
        ]
    elif finish_reason == "length":
        normalized["status"] = "incomplete"
        normalized["incomplete_details"] = {"reason": "max_output_tokens"}
    elif finish_reason == "content_filter":
        normalized["status"] = "failed"
        normalized["error"] = {
            "message": "the model response was blocked by the content filter"
        }

    return assess_response(normalized)
