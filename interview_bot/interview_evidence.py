"""Trace saved interview stories without treating their claims as verified facts."""

import hashlib
from typing import Any

from interview_coverage import coverage_map
from interview_retrieval import (
    build_candidate,
    content_identity,
    extract_tag_value,
)

REVIEW_STATUSES = ("Not reviewed", "Reviewed by me", "Needs correction")
RESULT_STATUSES = ("Not confirmed", "Reported actual outcome", "Estimate", "Potential benefit", "Mixed outcomes")


def story_id(text: str) -> str:
    """Return a stable ID for the same normalized story situation across aliases."""
    identity = content_identity(text, "story")
    return "story-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def build_story_library(chunks: list[dict]) -> list[dict[str, Any]]:
    """Group identical situations while retaining exact corpus locations and variants.

    Source locations establish where a claim is saved, not independent proof of it.
    Different situations are never automatically merged based on similarity.
    """
    records: dict[str, dict] = {}
    for index, chunk in enumerate(chunks):
        candidate = build_candidate(0, index, chunk)
        if candidate["response_type"] != "story":
            continue
        identifier = story_id(candidate["text"])
        record = records.setdefault(identifier, {
            "id": identifier,
            "title": extract_tag_value(candidate["text"], "Situation"),
            "candidate": candidate,
            "sources": [],
            "variants": [],
            "review_status": "Not reviewed",
            "result_status": "Not confirmed",
            "reviewed_at": None,
        })
        record["sources"].append({
            "file": "chunks_cleaned.json",
            "index": index,
            "source_id": chunk.get("source_id", ""),
            "text_sha256": hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest(),
        })
        record["variants"].append(candidate["text"])
    return list(records.values())


def review_for_story(record: dict, saved_review: dict | None = None) -> dict[str, Any]:
    """Invalidate review conclusions when source variants change or are unbound.

    A prior correction flag remains conservative until the user resolves it.
    Imported notes and dates remain user-supplied metadata, never verification.
    """
    review = {
        "review_status": "Not reviewed", "result_status": "Not confirmed",
        "note": "", "reviewed_at": "", "source_hashes": [], **(saved_review or {}),
    }
    current_hashes = sorted({source["text_sha256"] for source in record["sources"]})
    review["stale"] = bool(saved_review) and sorted(review["source_hashes"]) != current_hashes
    if review["stale"]:
        if review["review_status"] != "Needs correction":
            review["review_status"] = "Not reviewed"
        review["result_status"] = "Not confirmed"
    return review


def answer_markdown(answer: dict) -> str:
    """Export the generated answer with its question and traceable story ID."""
    return (
        f"# Interview practice answer\n\nQuestion: {answer['question']}\n\n"
        f"{answer['text']}\n\nSaved evidence ID: {answer.get('story_id', 'Direct answer')}\n"
        "\nGenerated wording; verify claims against the saved evidence before use.\n"
    )
