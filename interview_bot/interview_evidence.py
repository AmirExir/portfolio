"""Trace saved interview stories without treating their claims as verified facts."""

import hashlib
import re
from typing import Any

from interview_retrieval import (
    JOB_FOCUSES,
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


def build_story_library(
    chunks: list[dict], *, source_locations: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
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
            "file": (source_locations or {}).get(index, {}).get("file", "chunks_cleaned.json"),
            "index": (source_locations or {}).get(index, {}).get("index", index),
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


def coverage_map(description: str, library: list[dict]) -> list[dict[str, Any]]:
    """Map actual posting lines to story evidence with visible matched phrases.

    This is a conservative phrase match, not a judgment that a qualification is met.
    Every posting line is retained; unmatched requirements remain visible.
    """
    vocabulary = {term.lower() for profile in JOB_FOCUSES.values() for term in profile["keywords"]}
    vocabulary -= {"project", "projects"}

    def normalize(value: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[-\u2010-\u2015]", " ", value.lower())).strip()

    def includes(value: str, term: str) -> bool:
        return bool(re.search(rf"(?<!\w){re.escape(normalize(term))}(?!\w)", normalize(value)))

    evidence = {
        item["id"]: " ".join(
            extract_tag_value(item["candidate"]["text"], field)
            for field in ("Situation", "Task", "Action", "Result")
        ) for item in library
    }
    rows = []
    lines = dict.fromkeys(line.strip(" \t•-") for line in re.split(r"\n+|(?<=[.!?])\s+", description))
    for line in lines:
        if not line:
            continue
        terms = {term for term in vocabulary if includes(line, term)}
        matches = []
        for item in library:
            shared = sorted(term for term in terms if includes(evidence[item["id"]], term))
            if shared:
                matches.append({"story_id": item["id"], "matched_phrases": shared})
        matches.sort(key=lambda match: len(match["matched_phrases"]), reverse=True)
        rows.append({
            "posting_text": line,
            "status": "Related saved evidence; review required" if matches else "No phrase match; review manually",
            "matches": matches,
        })
    return rows


def answer_markdown(answer: dict) -> str:
    """Export saved or generated wording with its question and evidence ID."""
    wording = "Saved answer wording" if answer.get("candidate", {}).get("assigned_match") else "Generated wording"
    return (
        f"# Interview practice answer\n\nQuestion: {answer['question']}\n\n"
        f"{answer['text']}\n\nSaved evidence ID: {answer.get('story_id', 'Direct answer')}\n"
        f"\n{wording}; verify claims against the saved evidence before use.\n"
    )
