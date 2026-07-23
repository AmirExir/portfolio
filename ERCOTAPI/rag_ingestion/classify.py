"""Centralized source classification and collection routing rules."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config import Collection, SIDECAR_SUFFIX, SourceRoot


DOCUMENT_NUMBER_RE = re.compile(
    r"\b(NPRR|PGRR|NOGRR|OBDRR|SCR|RRGRR|VCMRR|COPMGRR|LPGRR|RMGRR|SMOGRR|CMGRR)"
    r"\s*[-_ ]?\s*(\d{2,5})\b",
    re.IGNORECASE,
)
REVERSE_DOCUMENT_NUMBER_RE = re.compile(
    r"\b(\d{2,5})(NPRR|PGRR|NOGRR|OBDRR|SCR|RRGRR|VCMRR|COPMGRR|LPGRR|"
    r"RMGRR|SMOGRR|CMGRR)\b",
    re.IGNORECASE,
)
ISO_DATE_RE = re.compile(r"\b(20\d{2})[-_/](0[1-9]|1[0-2])[-_/]([0-2]\d|3[01])\b")
MONTH_DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+([0-3]?\d),\s+(20\d{2})\b",
    re.IGNORECASE,
)


def sidecar_path(path: Path) -> Path:
    """Return the metadata sidecar path used by the official downloader."""

    return path.with_name(f"{path.name}{SIDECAR_SUFFIX}")


def load_sidecar(path: Path) -> dict[str, Any]:
    """Load a JSON sidecar, raising a useful error for malformed metadata."""

    candidate = sidecar_path(path)
    if not candidate.exists():
        return {}
    if candidate.is_symlink():
        raise ValueError(f"Refusing symbolic-link metadata sidecar: {candidate}")
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid metadata sidecar {candidate}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Metadata sidecar must contain a JSON object: {candidate}")
    return value


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return sorted({_clean(item) for item in value if _clean(item)})


def _provenance_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _document_number_parts(searchable: str) -> tuple[str, str] | None:
    match = DOCUMENT_NUMBER_RE.search(searchable)
    if match:
        return match.group(1).upper(), match.group(2)
    reverse = REVERSE_DOCUMENT_NUMBER_RE.search(searchable)
    if reverse:
        return reverse.group(2).upper(), reverse.group(1)
    return None


def _detect_kind(searchable: str, default: str) -> tuple[str, str | None]:
    number_parts = _document_number_parts(searchable)
    if number_parts:
        prefix, sequence = number_parts
        number = f"{prefix}{sequence}"
        names = {
            "NPRR": "NPRR",
            "PGRR": "PGRR",
            "NOGRR": "NOGRR",
            "OBDRR": "OBDRR",
            "SCR": "System Change Request",
            "RRGRR": "Resource Registration Glossary Revision Request",
            "VCMRR": "Verifiable Cost Manual Revision Request",
            "COPMGRR": "Commercial Operations Market Guide Revision Request",
            "LPGRR": "Load Profiling Guide Revision Request",
            "RMGRR": "Retail Market Guide Revision Request",
            "SMOGRR": "Settlement Metering Operating Guide Revision Request",
            "CMGRR": "Competitive Metering Guide Revision Request",
        }
        return names[prefix], number

    lowered = searchable.lower()
    rules = (
        (("nodal protocol", "protocols", "ercotnodals", "_nodal"), "Protocol"),
        (("planning guide", "planning_guid", "ercotaiassistant"), "Planning Guide"),
        (("operating guide", "operating_guid"), "Operating Guide"),
        (("resource integration", "interconnection handbook", "ercotrihandbook", "qsa"), "Resource Integration"),
        (("steady state working group", "sswg"), "SSWG"),
        (("dynamics working group", "dwg"), "DWG"),
        (("market notice", "market_notice"), "Market Notice"),
        (("fee schedule",), "Fee Schedule"),
        (("report",), "ERCOT Report"),
    )
    for needles, kind in rules:
        if any(needle in lowered for needle in needles):
            return kind, None
    return default, None


def _collections_for(kind: str, root: SourceRoot, searchable: str) -> list[str]:
    collections = set(root.default_collections)
    if root.is_generated:
        collections.update({Collection.NEWS.value, Collection.MARKET.value})
        collections.discard(Collection.GENERAL.value)
        return sorted(collections)
    if not root.is_generated:
        collections.add(Collection.GENERAL.value)

    kind_upper = kind.upper()
    if kind_upper in {"NPRR", "PROTOCOL", "PROTOCOLS", "NODAL PROTOCOL"}:
        collections.add(Collection.PROTOCOLS.value)
        collections.add(Collection.MARKET.value)
    elif kind_upper == "OBDRR":
        collections.add(Collection.PROTOCOLS.value)
        collections.add(Collection.OPERATIONS.value)
    elif kind_upper in {"PGRR", "PLANNING GUIDE", "RPG", "RTP"}:
        collections.add(Collection.PLANNING.value)
    elif kind_upper in {"NOGRR", "OPERATING GUIDE"}:
        collections.add(Collection.OPERATIONS.value)
    elif kind_upper in {"SCR", "SYSTEM CHANGE REQUEST"}:
        collections.add(Collection.OPERATIONS.value)
    elif kind_upper in {"RESOURCE INTEGRATION", "RIWG"}:
        collections.add(Collection.RESOURCE_INTEGRATION.value)
    elif kind_upper in {"DWG", "SSWG"}:
        collections.add(Collection.DWG_SSWG.value)
        if kind_upper == "SSWG":
            collections.add(Collection.PLANNING.value)
    elif kind_upper in {
        "COPMGRR",
        "LPGRR",
        "RMGRR",
        "SMOGRR",
        "CMGRR",
        "RESOURCE REGISTRATION GLOSSARY REVISION REQUEST",
        "VERIFIABLE COST MANUAL REVISION REQUEST",
        "COMMERCIAL OPERATIONS MARKET GUIDE REVISION REQUEST",
        "LOAD PROFILING GUIDE REVISION REQUEST",
        "RETAIL MARKET GUIDE REVISION REQUEST",
        "SETTLEMENT METERING OPERATING GUIDE REVISION REQUEST",
        "COMPETITIVE METERING GUIDE REVISION REQUEST",
        "MARKET NOTICE",
        "MARKET NOTICES",
        "PUBLIC NOTICE",
        "PUBLIC NOTICES",
        "ERCOT REPORT",
        "FEE SCHEDULE",
    }:
        collections.add(Collection.MARKET.value)

    # Committee/board pages can cross-post a numbered revision request while
    # their downloader sidecar legitimately keeps a generic source label such
    # as TAC or RPG. Route from the normalized revision identifier as well as
    # source_kind so those official documents reach their domain chatbot.
    number_parts = _document_number_parts(searchable)
    number_prefix = number_parts[0] if number_parts else ""
    if number_prefix == "NPRR":
        collections.update({Collection.PROTOCOLS.value, Collection.MARKET.value})
    elif number_prefix == "PGRR":
        collections.add(Collection.PLANNING.value)
    elif number_prefix == "NOGRR":
        collections.add(Collection.OPERATIONS.value)
    elif number_prefix == "OBDRR":
        collections.update({Collection.PROTOCOLS.value, Collection.OPERATIONS.value})
    elif number_prefix == "SCR":
        collections.add(Collection.OPERATIONS.value)
    elif number_prefix in {"RRGRR", "VCMRR", "COPMGRR", "LPGRR", "RMGRR", "SMOGRR", "CMGRR"}:
        collections.add(Collection.MARKET.value)

    lowered = searchable.lower()
    if "dwg" in lowered or "sswg" in lowered:
        collections.add(Collection.DWG_SSWG.value)
    if "interconnection" in lowered or "resource integration" in lowered:
        collections.add(Collection.RESOURCE_INTEGRATION.value)
    if "market notice" in lowered or "public notice" in lowered:
        collections.add(Collection.MARKET.value)
    return sorted(collections)


def _document_status(searchable: str, sidecar: Mapping[str, Any]) -> str | None:
    explicit = _clean(sidecar.get("document_status"))
    if explicit:
        return explicit
    lowered = searchable.lower()
    for status in ("approved", "effective", "withdrawn", "rejected", "redline", "clean", "draft", "pending"):
        if re.search(rf"\b{status}\b", lowered):
            return status.title()
    return None


def _effective_date(searchable: str, sidecar: Mapping[str, Any]) -> str | None:
    explicit = _clean(sidecar.get("effective_date"))
    if explicit:
        return explicit
    match = ISO_DATE_RE.search(searchable)
    return "-".join(match.groups()) if match else None


def _month_date_to_iso(value: str) -> str | None:
    try:
        return datetime.strptime(value, "%B %d, %Y").date().isoformat()
    except ValueError:
        return None


def enrich_metadata_from_text(
    metadata: Mapping[str, Any],
    text: str,
) -> dict[str, Any]:
    """Fill identifiable version metadata from safely extracted document text.

    Downloader sidecars and explicit metadata retain precedence. This primarily
    covers checked-in DOCX guide sections whose filenames carry only section
    numbers while the authoritative heading contains the title and date.
    """

    enriched = dict(metadata)
    preview = text[:20_000]
    searchable = _clean(preview)
    lines = [_clean(line) for line in preview.splitlines() if _clean(line)]

    filename = str(enriched.get("filename") or "")
    fallback_title = Path(filename).stem.replace("_", " ").replace("-", " ")
    if lines and _clean(enriched.get("title")) == _clean(fallback_title):
        first = lines[0]
        second = lines[1] if len(lines) > 1 else ""
        third = lines[2] if len(lines) > 2 else ""
        if first.lower() in {"ercot nodal protocols", "ercot planning guide"}:
            title_parts = [first]
            if second.lower().startswith("section"):
                title_parts.append(second)
                if third.lower().startswith(("attachment", "form")):
                    title_parts.append(third)
            enriched["title"] = " — ".join(title_parts)
        elif "dynamics working group" in first.lower():
            enriched["title"] = f"{first} — {second}" if second else first
        elif first.lower().startswith("ercot fee schedule"):
            enriched["title"] = first

    approved_match = re.search(
        rf"\b(?:ROS|Board)\s+Approved\s*:\s*({MONTH_DATE_RE.pattern})",
        searchable,
        re.IGNORECASE,
    )
    effective_match = re.search(
        rf"\bEffective\s+({MONTH_DATE_RE.pattern})",
        searchable,
        re.IGNORECASE,
    )
    date_matches = list(MONTH_DATE_RE.finditer(searchable))

    if not enriched.get("effective_date"):
        date_value: str | None = None
        if effective_match:
            date_value = effective_match.group(1)
        elif str(enriched.get("source_kind") or "").upper() in {
            "PROTOCOL",
            "PLANNING GUIDE",
        } and date_matches:
            date_value = date_matches[0].group(0)
        if date_value:
            enriched["effective_date"] = _month_date_to_iso(date_value)

    if approved_match:
        approved_date = _month_date_to_iso(approved_match.group(1))
        if approved_date and not enriched.get("published_date"):
            enriched["published_date"] = approved_date
        if not enriched.get("document_status"):
            enriched["document_status"] = "Approved"

    if not enriched.get("revision"):
        revision_match = re.search(r"\bRevision\s+(\d+[A-Za-z]?)\b", searchable, re.IGNORECASE)
        if revision_match:
            enriched["revision"] = revision_match.group(1)
    return enriched


def classify_document(
    path: Path,
    source_root: SourceRoot,
    repo_root: Path,
    *,
    sidecar: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify one document and return normalized retrieval metadata.

    Directory trust settings are authoritative: a sidecar cannot silently turn
    generated content into an official ERCOT source.
    """

    supplied = dict(sidecar or {})
    title = _clean(supplied.get("title")) or path.stem.replace("_", " ").replace("-", " ")
    source_label = _clean(supplied.get("source_label") or supplied.get("source"))
    supplied_number = _clean(supplied.get("document_number"))
    searchable = " ".join(
        (
            path.name,
            title,
            source_label,
            _clean(supplied.get("source_kind")),
            supplied_number,
            _clean(supplied.get("original_url") or supplied.get("url")),
            _clean(supplied.get("source_page_url")),
            " ".join(_string_list(supplied.get("source_page_urls"))),
        )
    )
    detected_kind, detected_number = _detect_kind(searchable, source_root.default_source_kind)
    source_kind = _clean(supplied.get("source_kind")) or detected_kind
    document_number = supplied_number or detected_number

    try:
        relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        relative = path.resolve().as_posix()

    metadata: dict[str, Any] = {
        "title": title,
        "filename": path.name,
        "source_path": relative,
        "source_category": source_root.name,
        "source_authority": source_root.source_authority,
        "source_kind": source_kind,
        "is_generated": source_root.is_generated,
        "collections": _collections_for(source_kind, source_root, searchable),
        "original_url": _clean(supplied.get("original_url") or supplied.get("url")) or None,
        "url_aliases": _string_list(supplied.get("url_aliases")),
        "source_page_urls": _string_list(supplied.get("source_page_urls")),
        "provenance": _provenance_list(supplied.get("provenance")),
        "downloaded_at": _clean(supplied.get("downloaded_at")) or None,
        "published_date": _clean(
            supplied.get("published_date") or supplied.get("published_hint")
        )
        or None,
        "effective_date": _effective_date(searchable, supplied),
        "document_number": document_number,
        "document_status": _document_status(searchable, supplied),
        "revision": _clean(supplied.get("revision") or supplied.get("version")) or None,
    }
    return metadata


def authority_rank(metadata: Mapping[str, Any]) -> int:
    """Return a stable trust rank used as a retrieval tie-breaker."""

    if metadata.get("is_generated"):
        return 0
    if str(metadata.get("source_authority", "")).upper() == "ERCOT":
        return 2
    return 1


def utc_now() -> str:
    """Return an RFC 3339 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
