"""Local answer screening, explicitly separate from semantic support verification."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


_CITATION = re.compile(r"\[(E\d+)\]")
_WORDS = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*")
_STOP = frozenset("a an and are as at be been by can for from has have in is it its of on or that the their this to was were will with ercot".split())
_QUANTITIES = re.compile(
    r"(?<!\w)([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*"
    r"(MW|Mvar|MVA|kV|%|p\.?u\.?|days?|hours?|Hz|seconds?|minutes?)(?!\w)",
    re.IGNORECASE,
)
ABSTENTION_TEXT = "The supplied ERCOT documents do not establish that information."


def claim_lines(answer: str) -> list[str]:
    """Return candidate factual lines, not linguistically verified atomic claims."""

    return [
        line.strip() for line in str(answer or "").splitlines()
        if line.strip()
        and not line.lstrip().startswith(("#", "Retrieved sources", "Sources:"))
        and len(_CITATION.sub("", line).split()) >= 4
    ]


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _terms(text: str) -> set[str]:
    return {word.casefold() for word in _WORDS.findall(text)} - _STOP


def _quantities(text: str) -> set[str]:
    return {
        f"{number.replace(',', '')} {unit.casefold()}"
        for number, unit in _QUANTITIES.findall(text)
    }


def screen_answer_support(
    answer: str, chunks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Flag observable support gaps without presenting overlap as entailment.

    Matching terms, numbers, or quotations cannot verify a paraphrase, its
    applicability, or its interpretation. Flags are review signals, not a
    determination that a derived engineering statement is false.
    """

    evidence = {str(chunk.get("evidence_id")): chunk for chunk in chunks if chunk.get("evidence_id")}
    checks: list[dict[str, Any]] = []
    for line in claim_lines(answer):
        clean = _CITATION.sub("", line).strip()
        clean = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", clean)
        if _normalized(clean) == _normalized(ABSTENTION_TEXT):
            continue
        identifiers = list(dict.fromkeys(_CITATION.findall(line)))
        cited = [evidence[item] for item in identifiers if item in evidence]
        source_text = "\n".join(str(item.get("text") or "") for item in cited)
        issues: list[str] = []
        if not identifiers:
            issues.append("missing_citation")
        if any(identifier not in evidence for identifier in identifiers):
            issues.append("unknown_evidence_id")
        if cited and any(not str(item.get("text") or "").strip() for item in cited):
            issues.append("cited_text_unavailable")
        if any(item.get("extraction_report", {}).get("review_required") for item in cited):
            issues.append("source_extraction_requires_review")
        terms = _terms(clean)
        overlap = len(terms & _terms(source_text)) / len(terms) if terms else 0.0
        if source_text.strip() and terms and overlap < 0.15:
            issues.append("low_lexical_overlap")
        missing_quantities = sorted(_quantities(clean) - _quantities(source_text))
        if missing_quantities:
            issues.append("quantity_not_found_in_cited_text")
        quotations = [first or second for first, second in re.findall(r'"([^"\n]{8,})"|“([^”\n]{8,})”', clean)]
        unmatched_quotes = [quote for quote in quotations if _normalized(quote) not in _normalized(source_text)]
        if unmatched_quotes:
            issues.append("quotation_not_found_in_cited_text")
        checks.append({
            "claim_line": line,
            "evidence_ids": identifiers,
            "lexical_overlap_ratio": round(overlap, 3),
            "missing_quantities": missing_quantities,
            "unmatched_quotations": unmatched_quotes,
            "review_reasons": issues,
            "support_status": "review_required" if issues else "semantic_support_unverified",
        })
    flagged = sum(bool(item["review_reasons"]) for item in checks)
    return {
        "support_status": "review_required" if flagged else "semantic_support_unverified",
        "semantic_support_verified": False,
        "support_method": "literal_text_screening_only",
        "support_review_required": bool(flagged),
        "flagged_claim_lines": flagged,
        "claim_checks": checks,
        "abstention_detected": ABSTENTION_TEXT.casefold() in answer.casefold(),
    }


def support_notice(audit: Mapping[str, Any]) -> str:
    """Return an honest user-facing status suitable for every answer surface."""

    if audit.get("support_review_required"):
        return (
            f"Source-support review required for {audit.get('flagged_claim_lines', 0)} answer line(s). "
            "Check the cited passages, quantities, quotations, and extraction warnings. "
            "These checks do not verify semantic support or engineering applicability."
        )
    return (
        "Citation and text checks do not verify that each source supports the answer. "
        "Semantic support and engineering applicability remain unverified."
    )
