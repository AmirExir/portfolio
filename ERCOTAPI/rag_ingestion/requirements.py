"""Deterministic ERCOT requirement intent, authority, and lifecycle analysis."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any, Mapping, Sequence

from .change_tracking import logical_document_key, parse_numbered_sections


REVISION_PREFIXES = (
    "NPRR", "PGRR", "NOGRR", "OBDRR", "RRGRR", "VCMRR", "COPMGRR",
    "LPGRR", "RMGRR", "SMOGRR", "CMGRR", "SCR",
)
GOVERNING_KINDS = frozenset(
    {
        "PROTOCOL",
        "PROTOCOLS",
        "NODAL PROTOCOL",
        "ERCOT NODAL PROTOCOLS",
        "PLANNING GUIDE",
        "OPERATING GUIDE",
        "NODAL OPERATING GUIDE",
        "OTHER BINDING DOCUMENT",
        "OTHER BINDING DOCUMENTS",
        "FEE SCHEDULE",
    }
)
PROCEDURE_KINDS = frozenset(
    {
        "DWG",
        "SSWG",
        "RESOURCE INTEGRATION",
        "RESOURCE INTEGRATION HANDBOOK",
        "PROCEDURE MANUAL",
        "COMMERCIAL OPERATIONS MARKET GUIDE",
        "LOAD PROFILING GUIDE",
        "RETAIL MARKET GUIDE",
        "SETTLEMENT METERING OPERATING GUIDE",
        "COMPETITIVE METERING GUIDE",
        "VERIFIABLE COST MANUAL",
        "RESOURCE REGISTRATION GLOSSARY",
        "MARKET GUIDE",
        "STABILITY CRITERIA",
        "STEADY STATE CRITERIA",
    }
)
CURRENT_SOURCE_CATEGORIES = frozenset(
    {
        "planning_guide_uploads",
        "nodal_protocol_uploads",
        "dwg_sswg_uploads",
        "market_document_uploads",
    }
)
_MONTH_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2},\s+20\d{2}\b",
    re.IGNORECASE,
)
_DATED_SPLIT_FILENAME_RE = re.compile(
    r"(?:^|[-_])(?P<month>0[1-9]|1[0-2])(?P<day>0[1-9]|[12]\d|3[01])"
    r"(?P<year>\d{2})(?=\.[A-Za-z0-9]+$)",
)
_DATED_MONTH_FILENAME_RE = re.compile(
    r"(?:^|[-_ ])(?P<month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)[-_ ](?P<day>\d{1,2})[-_ ,]+"
    r"(?P<year>20\d{2})(?=[-_ .]|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QuestionAnalysis:
    intent: str
    as_of: str
    asks_for_changes: bool
    asks_for_history: bool
    asks_for_status: bool
    asks_for_current_requirement: bool
    requested_documents: tuple[str, ...]
    requested_sections: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_date(value: date | datetime | str | None) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value:
        try:
            return date.fromisoformat(str(value)[:10])
        except ValueError:
            pass
    return date.today()


def analyze_question(
    question: str,
    *,
    as_of: date | datetime | str | None = None,
) -> QuestionAnalysis:
    normalized = " ".join(question.lower().split())
    explicit_dates = re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", normalized)
    month_dates = re.findall(
        r"\b(?:january|february|march|april|may|june|july|august|september|"
        r"october|november|december)\s+\d{1,2},\s+20\d{2}\b",
        normalized,
        re.IGNORECASE,
    )
    parsed_month_date = None
    if month_dates:
        try:
            parsed_month_date = datetime.strptime(month_dates[-1], "%B %d, %Y").date()
        except ValueError:
            parsed_month_date = None
    selected_date = (
        _as_date(explicit_dates[-1])
        if explicit_dates
        else parsed_month_date or _as_date(as_of)
    )
    change = bool(
        re.search(
            r"\b(change|changed|changes|difference|different|compare|comparison|amend|amended|"
            r"redline|new language|prior version|previous version|supersed)\w*\b",
            normalized,
        )
    )
    history = change or bool(
        re.search(r"\b(historical|history|old|older|former|previous|prior|superseded)\b", normalized)
    )
    status = bool(
        re.search(
            r"\b(status|effective|approved|pending|withdrawn|rejected|adopted|implemented|governs?)\b",
            normalized,
        )
    )
    documents = tuple(
        dict.fromkeys(
            f"{prefix.upper()}{number}"
            for prefix, number in re.findall(
                r"\b(NPRR|PGRR|NOGRR|OBDRR|RRGRR|VCMRR|COPMGRR|LPGRR|RMGRR|SMOGRR|CMGRR|SCR)"
                r"\s*[-_ ]?\s*(\d{2,5})\b",
                question,
                re.IGNORECASE,
            )
        )
    )
    sections = tuple(
        dict.fromkeys(
            match.rstrip(".")
            for match in re.findall(
                r"(?:\bsection\s*|§\s*)(\d+(?:\.\d+){0,5}[A-Za-z]?)",
                question,
                re.IGNORECASE,
            )
        )
    )
    if change:
        intent = "change_comparison"
    elif status:
        intent = "status_or_effectiveness"
    elif sections:
        intent = "section_explanation"
    else:
        intent = "current_requirement"
    return QuestionAnalysis(
        intent=intent,
        as_of=selected_date.isoformat(),
        asks_for_changes=change,
        asks_for_history=history,
        asks_for_status=status,
        asks_for_current_requirement=not history,
        requested_documents=documents,
        requested_sections=sections,
    )


def is_notice(chunk: Mapping[str, Any]) -> bool:
    kind = str(chunk.get("source_kind") or "").upper()
    path = str(chunk.get("source_path") or chunk.get("source") or "").lower()
    url = str(chunk.get("original_url") or "").lower()
    return (
        kind in {"MARKET NOTICE", "MARKET NOTICES", "PUBLIC NOTICE", "PUBLIC NOTICES"}
        or "/market-notices/" in path
        or "/public-notices/" in path
        or "/services/comm/mkt_notices/" in url
        or "/marketnotice/" in url
    )


def is_revision_request(chunk: Mapping[str, Any]) -> bool:
    kind = str(chunk.get("source_kind") or "").upper()
    number = str(chunk.get("document_number") or "").upper()
    return (
        kind.startswith(REVISION_PREFIXES)
        or number.startswith(REVISION_PREFIXES)
        or kind.endswith("REVISION REQUEST")
    )


def authority_class(chunk: Mapping[str, Any]) -> str:
    if chunk.get("is_generated"):
        return "generated_summary"
    if is_notice(chunk):
        return "operational_notice"
    if is_revision_request(chunk):
        return "revision_request"
    kind = str(chunk.get("source_kind") or "").upper()
    if kind in GOVERNING_KINDS:
        return "governing_document"
    if kind in PROCEDURE_KINDS:
        return "procedure_or_criteria"
    if kind in {"BOARD", "BOARD OF DIRECTORS", "TAC", "ROS", "RPG", "RIWG", "LLWG"}:
        return "governance_record"
    return "official_supporting_document"


def _metadata_date(value: Any) -> date | None:
    candidate = str(value or "").strip()
    if not candidate:
        return None
    for pattern in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(candidate[:10] if pattern == "%Y-%m-%d" else candidate, pattern).date()
        except ValueError:
            continue
    return None


def _is_controlled_current_copy(chunk: Mapping[str, Any]) -> bool:
    if str(chunk.get("source_category") or "") in CURRENT_SOURCE_CATEGORIES:
        return True
    paths = [
        str(chunk.get("source_path") or chunk.get("source") or ""),
        *(str(value) for value in (chunk.get("aliases") or []) if value),
    ]
    return any(
        marker in path.replace("\\", "/")
        for path in paths
        for marker in (
            "ERCOTAPI/sources/official/planning_guides/",
            "ERCOTAPI/sources/official/nodal_protocols/",
            "ERCOTAPI/sources/official/dwg_sswg/",
            "ERCOTAPI/sources/official/market/",
        )
    )


def governing_document_date(chunk: Mapping[str, Any]) -> date | None:
    """Resolve an edition/effective date for a controlled governing document.

    Recent metadata already contains ``effective_date``.  Older saved chunks
    can predate that enrichment, while their controlled split filename (for
    example ``09-071126.docx``) or cover heading still contains the same date.
    Recovering it at query time updates citations without changing vectors.
    """

    explicit = _metadata_date(chunk.get("effective_date"))
    if explicit is not None:
        return explicit
    if authority_class(chunk) != "governing_document":
        return None

    paths = [
        str(chunk.get("source_path") or chunk.get("source") or ""),
        str(chunk.get("original_url") or ""),
        str(chunk.get("final_url") or ""),
        *(str(value) for value in (chunk.get("aliases") or []) if value),
        *(str(value) for value in (chunk.get("url_aliases") or []) if value),
    ]
    for value in paths:
        filename = value.replace("\\", "/").rsplit("/", 1)[-1]
        match = _DATED_SPLIT_FILENAME_RE.search(filename)
        if match is None:
            continue
        try:
            return date(
                2000 + int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
            )
        except ValueError:
            continue

    for value in paths:
        filename = value.replace("\\", "/").rsplit("/", 1)[-1]
        match = _DATED_MONTH_FILENAME_RE.search(filename)
        if match is None:
            continue
        try:
            return datetime.strptime(
                f"{match.group('month')} {match.group('day')}, {match.group('year')}",
                "%B %d, %Y",
            ).date()
        except ValueError:
            try:
                return datetime.strptime(
                    f"{match.group('month')} {match.group('day')}, {match.group('year')}",
                    "%b %d, %Y",
                ).date()
            except ValueError:
                continue

    try:
        first_chunk = int(chunk.get("chunk_index") or 0) == 0
    except (TypeError, ValueError):
        first_chunk = False
    document_artifact = any(
        re.search(r"\.(?:pdf|docx?)\b", value, re.IGNORECASE)
        for value in paths
    )
    if first_chunk and (_is_controlled_current_copy(chunk) or document_artifact):
        match = _MONTH_DATE_RE.search(str(chunk.get("text") or "")[:2_000])
        if match is not None:
            return _metadata_date(match.group(0))
    return None


def lifecycle_metadata(
    chunk: Mapping[str, Any],
    *,
    as_of: date | datetime | str | None = None,
) -> dict[str, Any]:
    """Classify effectiveness without treating an xRR as governing text."""

    selected_date = _as_date(as_of)
    status = " ".join(str(chunk.get("document_status") or "").lower().split())
    authority = authority_class(chunk)
    explicit_effective_date = _metadata_date(chunk.get("effective_date"))
    resolved_effective_date = (
        governing_document_date(chunk)
        if authority == "governing_document"
        else explicit_effective_date
    )
    inferred_effective_date = bool(
        resolved_effective_date is not None and explicit_effective_date is None
    )
    future = bool(resolved_effective_date and resolved_effective_date > selected_date)
    negative_effective_status = bool(
        re.search(r"\bnot\s+(?:yet\s+)?effective\b", status)
    )

    if any(
        term in status
        for term in (
            "withdrawn",
            "rejected",
            "cancelled",
            "canceled",
            "superseded",
            "inactive",
            "expired",
            "obsolete",
        )
    ):
        state, label, basis = (
            "not_effective",
            "Not effective",
            f"Document status is {status or 'inactive'}.",
        )
    elif negative_effective_status:
        state, label, basis = (
            "not_effective",
            f"Not effective as of {selected_date.isoformat()}",
            f"Document status is {status}.",
        )
    elif future:
        state, label, basis = (
            "approved_not_effective",
            f"Not effective as of {selected_date.isoformat()}",
            (
                f"The controlled document edition is dated {resolved_effective_date.isoformat()}."
                if inferred_effective_date
                else f"The recorded effective date is {resolved_effective_date.isoformat()}."
            ),
        )
    elif any(term in status for term in ("draft", "pending", "proposed", "tabled", "ballot", "redline")):
        state, label, basis = (
            "proposed_or_pending",
            "Proposed or pending",
            f"Document status is {status}.",
        )
    elif authority == "revision_request":
        if "effective" in status or "implemented" in status:
            state, label, basis = (
                "implemented_change_record",
                "Implemented change record; governing text must be verified",
                "The xRR records a change, but the incorporated Protocol/Guide/OBD is the governing source.",
            )
        elif "approved" in status:
            state, label, basis = (
                "approved_effectiveness_unverified",
                "Approved; effectiveness not established",
                "Approval alone does not prove that the change is implemented in governing text.",
            )
        else:
            state, label, basis = (
                "effectiveness_unknown",
                "Effectiveness not established",
                "No reliable implementation or effective status is present in metadata.",
            )
    elif authority == "governing_document" and resolved_effective_date is not None and (
        _is_controlled_current_copy(chunk) or "effective" in status or "current" in status
    ) and selected_date <= date.today():
        state, label, basis = (
            "effective",
            f"Effective as of {selected_date.isoformat()}",
            (
                f"Controlled current-document edition dated {resolved_effective_date.isoformat()}."
                if inferred_effective_date
                else f"Recorded effective date {resolved_effective_date.isoformat()}."
            ),
        )
    elif authority == "governing_document" and resolved_effective_date is not None:
        state, label, basis = (
            "effective_edition_currentness_unverified",
            "Edition date identified; currentness not established",
            (
                f"The document edition is dated {resolved_effective_date.isoformat()}, but the "
                f"available metadata does not prove it governed as of {selected_date.isoformat()}."
            ),
        )
    elif authority == "governing_document" and (
        _is_controlled_current_copy(chunk) or "effective" in status or "current" in status
    ) and selected_date == date.today():
        state, label, basis = (
            "effective",
            f"Effective as of {selected_date.isoformat()}",
            "Document is from the controlled current-document bundle.",
        )
    elif authority == "procedure_or_criteria" and "approved" in status:
        state, label, basis = (
            "approved_procedure",
            "Approved procedure or criteria",
            f"Document status is {status}.",
        )
    else:
        state, label, basis = (
            "effectiveness_unknown",
            "Effectiveness not established",
            "The available metadata does not establish an effective date or controlled-current status.",
        )

    return {
        "effective_state": state,
        "effectiveness_label": label,
        "effectiveness_basis": basis,
        "resolved_effective_date": (
            resolved_effective_date.isoformat() if resolved_effective_date else None
        ),
        "effective_date_inferred": inferred_effective_date,
        "as_of": selected_date.isoformat(),
    }


def evidence_role(chunk: Mapping[str, Any], *, as_of: date | datetime | str | None = None) -> str:
    authority = authority_class(chunk)
    state = lifecycle_metadata(chunk, as_of=as_of)["effective_state"]
    if authority == "governing_document" and state == "effective":
        return "current_governing_requirement"
    if authority == "governing_document":
        return "candidate_governing_text"
    if authority == "revision_request":
        return "related_change_record"
    if authority == "procedure_or_criteria":
        return "procedure_or_engineering_criteria"
    if state == "not_effective":
        return "historical_or_inactive"
    return "supporting_evidence"


def annotate_evidence(
    chunk: Mapping[str, Any],
    *,
    as_of: date | datetime | str | None = None,
    requested_sections: Sequence[str] = (),
) -> dict[str, Any]:
    annotated = dict(chunk)
    annotated["authority_class"] = authority_class(chunk)
    lifecycle = lifecycle_metadata(chunk, as_of=as_of)
    annotated.update(lifecycle)
    if (
        not annotated.get("effective_date")
        and lifecycle.get("resolved_effective_date")
        and lifecycle.get("effective_state") in {"effective", "approved_not_effective"}
    ):
        annotated["effective_date"] = lifecycle["resolved_effective_date"]
    annotated["evidence_role"] = evidence_role(chunk, as_of=as_of)
    annotated["is_governing"] = annotated["evidence_role"] == "current_governing_requirement"
    annotated["logical_document_id"] = logical_document_key(chunk)
    if not annotated.get("section_number"):
        parsed = parse_numbered_sections(str(chunk.get("text") or ""))
        selected = next(
            (
                section
                for requested in requested_sections
                for section in parsed
                if section.number.lower() == requested.lower()
            ),
            parsed[0] if parsed else None,
        )
        if selected is not None:
            annotated["section_number"] = selected.number
            if selected.title:
                annotated["section_title"] = selected.title
            if selected.page_start is not None:
                annotated["page_start"] = selected.page_start
                annotated["page_end"] = selected.page_end or selected.page_start
    if not annotated.get("page_start"):
        pages = [
            int(value)
            for value in re.findall(r"^\s*\[Page\s+(\d+)\]\s*$", str(chunk.get("text") or ""), re.MULTILINE | re.IGNORECASE)
        ]
        if pages:
            annotated["page_start"] = pages[0]
            annotated["page_end"] = pages[-1]
    return annotated


def document_identity(chunk: Mapping[str, Any]) -> str:
    return str(
        chunk.get("document_id")
        or chunk.get("content_hash")
        or chunk.get("source_path")
        or chunk.get("source")
        or chunk.get("chunk_id")
        or "unknown"
    )


def diversify_evidence(
    question: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
    as_of: date | datetime | str | None = None,
) -> list[dict[str, Any]]:
    """Select relevant evidence across governing, procedure, and change lanes."""

    if top_k < 1 or not candidates:
        return []
    analysis = analyze_question(question, as_of=as_of)
    annotated = [
        annotate_evidence(
            chunk,
            as_of=analysis.as_of,
            requested_sections=analysis.requested_sections,
        )
        for chunk in candidates
        if not is_notice(chunk)
    ]
    if not annotated:
        return []
    top_score = max(float(chunk.get("retrieval_score", 0.0)) for chunk in annotated)
    # Diversity must never pull a weak document merely to fill a category.
    relevant = [
        chunk for chunk in annotated
        if float(chunk.get("retrieval_score", 0.0)) >= top_score - 0.22
        or str(chunk.get("document_number") or "").upper() in analysis.requested_documents
    ]
    lane_order = (
        ("current_governing_requirement", "candidate_governing_text"),
        ("procedure_or_engineering_criteria",),
        ("related_change_record",),
        ("supporting_evidence", "historical_or_inactive"),
    )
    if analysis.asks_for_changes:
        lane_order = (lane_order[0], lane_order[2], lane_order[1], lane_order[3])

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    per_document: dict[str, int] = {}

    def add(chunk: dict[str, Any]) -> bool:
        chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or id(chunk))
        document = document_identity(chunk)
        if chunk_id in selected_ids or per_document.get(document, 0) >= 3:
            return False
        selected.append(chunk)
        selected_ids.add(chunk_id)
        per_document[document] = per_document.get(document, 0) + 1
        return True

    # First pass: one best item from each relevant document in each evidence lane.
    for roles in lane_order:
        seen_documents: set[str] = set()
        lane = [chunk for chunk in relevant if chunk.get("evidence_role") in roles]
        lane.sort(key=lambda chunk: float(chunk.get("retrieval_score", 0.0)), reverse=True)
        for chunk in lane:
            document = document_identity(chunk)
            if document in seen_documents:
                continue
            seen_documents.add(document)
            add(chunk)
            if len(selected) >= top_k:
                break
        if len(selected) >= top_k:
            break

    # Second pass: add the strongest continuation chunks while retaining a cap.
    for chunk in relevant:
        if len(selected) >= top_k:
            break
        add(chunk)

    selected.sort(
        key=lambda chunk: (
            -float(chunk.get("retrieval_score", 0.0)),
            str(chunk.get("chunk_id") or ""),
        )
    )
    for number, chunk in enumerate(selected, start=1):
        chunk["evidence_id"] = f"E{number}"
    return selected


def evidence_summary(chunks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    roles: dict[str, list[str]] = {
        "governing": [],
        "related_requirements": [],
        "change_records": [],
        "historical_or_uncertain": [],
    }
    for chunk in chunks:
        identity = document_identity(chunk)
        role = str(chunk.get("evidence_role") or "")
        if role == "current_governing_requirement":
            target = "governing"
        elif role == "procedure_or_engineering_criteria":
            target = "related_requirements"
        elif role == "related_change_record":
            target = "change_records"
        else:
            target = "historical_or_uncertain"
        if identity not in roles[target]:
            roles[target].append(identity)
    return {key: value for key, value in roles.items()}


def answer_contract(analysis: QuestionAnalysis) -> str:
    change_instruction = (
        "Compare the prior and current language section by section; label added, modified, removed, "
        "and unchanged requirements."
        if analysis.asks_for_changes
        else "Mention related change records only after explaining the governing requirement."
    )
    return f"""Use the evidence IDs exactly as inline citations, for example [E1].
Treat only evidence labelled current_governing_requirement as established governing text.
Never present an xRR, ballot, committee packet, approval, or proposed redline as an effective
requirement unless incorporated governing text is also supplied. State any effectiveness uncertainty.
Answer as of {analysis.as_of}. {change_instruction}

Use these headings when evidence exists:
1. Short answer
2. What governs and effectiveness
3. Combined requirements and applicability
4. Engineering impact
5. Changes, proposals, and uncertainty

Every material factual statement must have an evidence citation. If the evidence does not establish
the answer, say exactly: "The supplied ERCOT documents do not establish that information."
""".strip()


def validate_answer_citations(
    answer: str,
    chunks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reject invented evidence IDs and report deterministic citation coverage."""

    valid = {
        str(chunk.get("evidence_id"))
        for chunk in chunks
        if chunk.get("evidence_id")
    }
    cited = set(re.findall(r"\[(E\d+)\]", str(answer or "")))
    invalid = sorted(cited - valid)
    claim_lines = [
        line.strip()
        for line in str(answer or "").splitlines()
        if line.strip()
        and not line.lstrip().startswith(("#", "Retrieved sources", "Sources:"))
        and len(re.sub(r"\[(?:E\d+)\]", "", line).split()) >= 4
    ]
    cited_lines = sum(bool(re.search(r"\[E\d+\]", line)) for line in claim_lines)
    coverage = cited_lines / len(claim_lines) if claim_lines else 1.0
    return {
        "valid_evidence_ids": sorted(valid),
        "cited_evidence_ids": sorted(cited),
        "invalid_evidence_ids": invalid,
        "claim_line_coverage": round(coverage, 3),
        "passed": bool(cited) and not invalid and coverage >= 0.8,
    }
