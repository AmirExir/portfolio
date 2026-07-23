"""Build and load a compact public feed of newly archived ERCOT documents."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXCLUDED_SOURCES = {"MARKET NOTICES", "PUBLIC NOTICES"}
EXCLUDED_URL_FRAGMENTS = (
    "/services/comm/mkt_notices/",
    "/marketnotice/",
)
REVISION_REQUEST_NAMES = {
    "NPRR": "Nodal Protocol Revision Request",
    "PGRR": "Planning Guide Revision Request",
    "NOGRR": "Nodal Operating Guide Revision Request",
    "OBDRR": "Other Binding Document Revision Request",
    "RRGRR": "Resource Registration Glossary Revision Request",
    "VCMRR": "Verifiable Cost Manual Revision Request",
    "COPMGRR": "Commercial Operations Market Guide Revision Request",
    "LPGRR": "Load Profiling Guide Revision Request",
    "RMGRR": "Retail Market Guide Revision Request",
    "SMOGRR": "Settlement Metering Operating Guide Revision Request",
    "CMGRR": "Competitive Metering Guide Revision Request",
    "SCR": "System Change Request",
}
REVISION_REQUEST_FAMILIES = tuple(REVISION_REQUEST_NAMES)
_REVISION_FAMILY_PATTERN = "|".join(
    sorted(REVISION_REQUEST_FAMILIES, key=len, reverse=True)
)
_ISSUE_URL_RE = re.compile(
    rf"/mktrules/issues/(?P<family>{_REVISION_FAMILY_PATTERN})"
    r"[-_ ]?(?P<number>\d{1,6})\b",
    re.IGNORECASE,
)
_REVERSE_REVISION_RE = re.compile(
    rf"\b(?P<number>\d{{1,6}})(?P<family>{_REVISION_FAMILY_PATTERN})"
    r"(?=[-_\s.]|$)",
    re.IGNORECASE,
)
_FORWARD_REVISION_RE = re.compile(
    rf"\b(?P<family>{_REVISION_FAMILY_PATTERN})[-_ ]?(?P<number>\d{{1,6}})\b",
    re.IGNORECASE,
)
EXCLUDED_NAVIGATION_TITLES = {
    "hide emil information show emil information",
    "mis log in",
    "nodal protocol revision requests",
    "nprr submission process",
    "protocol interpretation request submission process",
    "protocol library - nodal",
    "workshops",
}

_ISSUE_PAGE_PATH_RE = re.compile(
    rf"/mktrules/issues/(?P<revision>(?:{_REVISION_FAMILY_PATTERN})\d{{1,6}})"
    r"(?:[/?#]|$)",
    re.IGNORECASE,
)
_SPACE_RE = re.compile(r"\s+")
_DATE_RE = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")


def _compact_text(value: Any) -> str:
    return _SPACE_RE.sub(" ", str(value or "")).strip()


def _decode_html(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


class _RevisionIssueHTMLParser(HTMLParser):
    """Extract the stable tables from an official ERCOT xRR issue page."""

    _TARGET_SECTIONS = {
        "tab-summary": "summary",
        "tab-action": "action",
        "tab-votingrecord": "voting",
        "tab-background": "background",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: dict[str, list[list[str]]] = {
            section: [] for section in self._TARGET_SECTIONS.values()
        }
        self._div_sections: list[str | None] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._hidden_depth = 0

    @property
    def _section(self) -> str | None:
        return self._div_sections[-1] if self._div_sections else None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style", "noscript", "svg"}:
            self._hidden_depth += 1
        if lowered == "div":
            values = dict(attrs)
            selected = self._TARGET_SECTIONS.get(
                str(values.get("id") or "").lower(),
                self._section,
            )
            self._div_sections.append(selected)
        if self._section and lowered == "tr":
            self._row = []
        elif self._section and self._row is not None and lowered in {"th", "td"}:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._hidden_depth == 0 and self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"th", "td"} and self._cell is not None and self._row is not None:
            self._row.append(_compact_text(" ".join(self._cell)))
            self._cell = None
        elif lowered == "tr" and self._row is not None:
            if self._section and any(self._row):
                self.rows[self._section].append(self._row)
            self._row = None
            self._cell = None
        if lowered == "div" and self._div_sections:
            self._div_sections.pop()
        if lowered in {"script", "style", "noscript", "svg"}:
            self._hidden_depth = max(0, self._hidden_depth - 1)


def _row_mapping(rows: Sequence[Sequence[str]]) -> dict[str, str]:
    values: dict[str, str] = {}
    for row in rows:
        if len(row) < 2:
            continue
        key = _compact_text(row[0]).rstrip(":").casefold()
        value = _compact_text(" ".join(row[1:]))
        if key and value:
            values[key] = value
    return values


def _date_rank(value: str) -> tuple[int, str]:
    candidate = _compact_text(value)
    for pattern in ("%m/%d/%Y", "%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
        try:
            return int(datetime.strptime(candidate, pattern).strftime("%Y%m%d")), candidate
        except ValueError:
            continue
    return 0, candidate


def _latest_action(rows: Sequence[Sequence[str]]) -> dict[str, str] | None:
    actions: list[dict[str, str]] = []
    for row in rows:
        if any(
            "no updates have been made to this issue" in _compact_text(value).casefold()
            for value in row
        ):
            actions.append({"date": "", "governing_body": "", "action": "No action posted"})
            continue
        if len(row) < 3 or not _DATE_RE.fullmatch(_compact_text(row[0])):
            continue
        action = {
            "date": _compact_text(row[0]),
            "governing_body": _compact_text(row[1]),
            "action": _compact_text(row[2]),
        }
        if len(row) > 3 and _compact_text(row[3]):
            action["next_step"] = _compact_text(row[3])
        actions.append(action)
    if not actions:
        return None
    return max(actions, key=lambda item: _date_rank(item.get("date", "")))


def revision_effectiveness_note(status: str) -> tuple[str, str]:
    """Return a conservative state and note without treating an xRR as law."""

    lowered = _compact_text(status).casefold()
    if "pending" in lowered:
        return (
            "pending_proposal",
            "Pending proposal; it is not current governing text.",
        )
    if "approved" in lowered:
        return (
            "approved_not_independently_governing",
            "Approved revision request; confirm its effective date and incorporation in the "
            "current controlling document before relying on it as governing text.",
        )
    if "withdraw" in lowered or "reject" in lowered:
        return (
            "not_adopted",
            "Withdrawn or rejected revision request; it is not governing text.",
        )
    return (
        "unverified_revision_record",
        "Revision-request record; verify approval, effective date, and incorporation in the "
        "current controlling document before relying on it as governing text.",
    )


def _build_issue_details(
    revision_id: str,
    *,
    issue_url: str,
    title: str,
    status: str,
    date_posted: str,
    sponsor: str,
    urgent: str,
    sections: str,
    description: str,
    reason: str,
    next_group: str = "",
    next_step: str = "",
    latest_action: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state, note = revision_effectiveness_note(status)
    details: dict[str, Any] = {
        "revision_id": revision_id,
        "revision_family": re.sub(r"\d+$", "", revision_id),
        "issue_title": _compact_text(title),
        "official_description": _compact_text(description),
        "status": _compact_text(status),
        "date_posted": _compact_text(date_posted),
        "sponsor": _compact_text(sponsor),
        "urgent": _compact_text(urgent),
        "affected_sections": _compact_text(sections),
        "reason": _compact_text(reason),
        "issue_url": _compact_text(issue_url),
        "effective_state": state,
        "effectiveness_note": note,
        "explanation_source": "official_ercot_issue_page",
    }
    if _compact_text(next_group):
        details["next_group"] = _compact_text(next_group)
    if _compact_text(next_step):
        details["next_step"] = _compact_text(next_step)
    if latest_action:
        cleaned_action = {
            key: _compact_text(value)
            for key, value in latest_action.items()
            if _compact_text(value)
        }
        if cleaned_action:
            details["latest_action"] = cleaned_action
    return details


def parse_revision_issue_html(
    content: bytes,
    *,
    revision_id: str,
    issue_url: str,
) -> dict[str, Any]:
    """Parse one cached official issue page without network or model calls."""

    parser = _RevisionIssueHTMLParser()
    parser.feed(_decode_html(content))
    parser.close()
    summary = _row_mapping(parser.rows["summary"])
    background = _row_mapping(parser.rows["background"])
    return _build_issue_details(
        revision_id,
        issue_url=issue_url,
        title=summary.get("title", ""),
        status=background.get("status", summary.get("status", "")),
        date_posted=background.get("date posted", ""),
        sponsor=background.get("sponsor", ""),
        urgent=background.get("urgent", ""),
        sections=background.get("sections", ""),
        description=background.get("description", ""),
        reason=background.get("reason", ""),
        next_group=summary.get("next group", ""),
        next_step=summary.get("next step", ""),
        latest_action=_latest_action(parser.rows["action"]),
    )


def _capture_issue_field(text: str, start: str, end: str) -> str:
    match = re.search(
        rf"\b{re.escape(start)}\s*:?\s*(.*?)\s+\b{re.escape(end)}\s*:?",
        text,
        re.IGNORECASE,
    )
    return _compact_text(match.group(1)) if match else ""


def parse_revision_issue_text(
    text: str,
    *,
    revision_id: str,
    issue_url: str,
) -> dict[str, Any]:
    """Parse joined saved-index text when cached HTML is unavailable."""

    compact = _compact_text(text)
    title_match = re.search(
        r"\bSummary\s+Title\s+(.*?)\s+Next Group\b",
        compact,
        re.IGNORECASE,
    )
    title = _compact_text(title_match.group(1)) if title_match else ""
    background_matches = list(
        re.finditer(r"\bBackground\s+Status\s*:", compact, re.IGNORECASE)
    )
    background = compact[background_matches[-1].start() :] if background_matches else compact
    summary_match = re.search(
        r"\bSummary\s+Title\s+.*?\s+Next Group\s+(.*?)\s+Next Step\s+"
        r"(.*?)\s+Status(?:\s|:)",
        compact,
        re.IGNORECASE,
    )
    next_group = _compact_text(summary_match.group(1)) if summary_match else ""
    next_step = _compact_text(summary_match.group(2)) if summary_match else ""

    action_rows: list[list[str]] = []
    action_match = re.search(
        r"\bAction\s+Date\s+Gov Body\s+Action Taken\s+Next Steps\s+"
        r"(.*?)\s+Voting Record\b",
        compact,
        re.IGNORECASE,
    )
    if action_match:
        action_text = action_match.group(1)
        action_pattern = re.compile(
            r"(?P<date>\d{2}/\d{2}/\d{4})\s+"
            r"(?P<body>PUCT|BOARD|TAC|PRS|ROS|WMS|RMS|COPS|OWG|PLWG|"
            r"DWG|SSWG|VPWG|SPWG|LLWG)\s+"
            r"(?P<action>Recommended for Approval|Deferred/Tabled|Approved|"
            r"Rejected|Withdrawn|No Action)"
            r"(?P<next>.*?)(?=\s+\d{2}/\d{2}/\d{4}\s+|$)",
            re.IGNORECASE,
        )
        for match in action_pattern.finditer(action_text):
            action_rows.append(
                [
                    match.group("date"),
                    match.group("body").upper(),
                    _compact_text(match.group("action")),
                    _compact_text(match.group("next")),
                ]
            )
        if (
            not action_rows
            and "no updates have been made to this issue" in action_text.casefold()
        ):
            action_rows.append(["No updates have been made to this issue."])
    return _build_issue_details(
        revision_id,
        issue_url=issue_url,
        title=title,
        status=_capture_issue_field(background, "Status", "Date Posted"),
        date_posted=_capture_issue_field(background, "Date Posted", "Sponsor"),
        sponsor=_capture_issue_field(background, "Sponsor", "Urgent"),
        urgent=_capture_issue_field(background, "Urgent", "Sections"),
        sections=_capture_issue_field(background, "Sections", "Description"),
        description=_capture_issue_field(background, "Description", "Reason"),
        reason=_capture_issue_field(background, "Reason", "Key Documents"),
        next_group=next_group,
        next_step=next_step,
        latest_action=_latest_action(action_rows),
    )


def revision_request_identity(
    document_number: str = "",
    title: str = "",
    url: str = "",
) -> tuple[str, str] | None:
    """Return ``(revision_id, family)`` without confusing attachment numbers."""

    issue_match = _ISSUE_URL_RE.search(url)
    if issue_match:
        family = issue_match.group("family").upper()
        return f"{family}{issue_match.group('number')}", family

    for candidate in (document_number, title, url):
        reverse_match = _REVERSE_REVISION_RE.search(candidate)
        if reverse_match:
            family = reverse_match.group("family").upper()
            return f"{family}{reverse_match.group('number')}", family
        forward_match = _FORWARD_REVISION_RE.search(candidate)
        if forward_match:
            family = forward_match.group("family").upper()
            return f"{family}{forward_match.group('number')}", family
    return None


def _read_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path.with_name(f"{path.name}.metadata.json")
    try:
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _official_archive_roots(paths: Sequence[Path]) -> list[Path]:
    roots: set[Path] = set()
    for path in paths:
        for parent in path.parents:
            if parent.name.casefold() == "official" and parent.parent.name.casefold() == "news":
                roots.add(parent)
                break
    return sorted(roots)


def _cached_issue_page_paths(
    archive_roots: Iterable[Path],
    revision_ids: set[str],
) -> list[Path]:
    candidates: list[Path] = []
    families = {
        re.sub(r"\d+$", "", revision_id).casefold()
        for revision_id in revision_ids
    }
    for root in archive_roots:
        for family in families:
            family_root = root / family
            if not family_root.is_dir():
                continue
            for metadata_path in family_root.rglob("*.metadata.json"):
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    continue
                identity = revision_request_identity(
                    str(metadata.get("document_number") or ""),
                    str(metadata.get("title") or ""),
                    str(metadata.get("original_url") or metadata.get("final_url") or ""),
                )
                if not identity or identity[0] not in revision_ids:
                    continue
                url = str(metadata.get("original_url") or metadata.get("final_url") or "")
                issue_match = _ISSUE_PAGE_PATH_RE.search(url)
                if not issue_match or issue_match.group("revision").upper() != identity[0]:
                    continue
                source = metadata_path.with_name(
                    metadata_path.name[: -len(".metadata.json")]
                )
                if source.suffix.casefold() in {".html", ".htm"} and source.is_file():
                    candidates.append(source)
    return candidates


def _issue_details_from_html_paths(
    paths: Iterable[Path],
    revision_ids: set[str],
) -> dict[str, dict[str, Any]]:
    candidates: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
    for path in paths:
        if path.suffix.casefold() not in {".html", ".htm"} or not path.is_file():
            continue
        metadata = _read_metadata(path)
        url = str(metadata.get("original_url") or metadata.get("final_url") or "")
        issue_match = _ISSUE_PAGE_PATH_RE.search(url)
        if not issue_match:
            continue
        revision_id = issue_match.group("revision").upper()
        if revision_id not in revision_ids:
            continue
        try:
            details = parse_revision_issue_html(
                path.read_bytes(),
                revision_id=revision_id,
                issue_url=url,
            )
        except (OSError, UnicodeError):
            continue
        if not details.get("official_description"):
            continue
        candidates.setdefault(revision_id, []).append(
            (
                str(metadata.get("downloaded_at") or ""),
                path.as_posix(),
                details,
            )
        )
    return {
        revision_id: max(values, key=lambda value: (value[0], value[1]))[2]
        for revision_id, values in candidates.items()
    }


def _default_packaged_chunks_path() -> Path | None:
    store = Path(__file__).with_name("deployment_rag_store")
    try:
        generation = (store / "CURRENT").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not generation or Path(generation).name != generation:
        return None
    root = store / "generations" / generation
    for name in ("chunks.json", "chunks.json.gz"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def _read_chunks(path: Path) -> list[dict[str, Any]]:
    try:
        if path.suffix.casefold() == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                value = json.load(handle)
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _issue_details_from_chunks(
    chunks_path: Path,
    revision_ids: set[str],
) -> dict[str, dict[str, Any]]:
    documents: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for chunk in _read_chunks(chunks_path):
        revision_id = str(chunk.get("document_number") or "").upper()
        if revision_id not in revision_ids:
            continue
        url = str(chunk.get("original_url") or "")
        issue_match = _ISSUE_PAGE_PATH_RE.search(url)
        if not issue_match or issue_match.group("revision").upper() != revision_id:
            continue
        document_key = str(
            chunk.get("document_id")
            or chunk.get("content_hash")
            or chunk.get("source_path")
            or ""
        )
        if document_key:
            documents.setdefault((revision_id, document_key), []).append(chunk)

    candidates: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
    for (revision_id, document_key), document_chunks in documents.items():
        ordered = sorted(
            document_chunks,
            key=lambda chunk: (
                int(chunk.get("chunk_index") or 0),
                str(chunk.get("chunk_id") or ""),
            ),
        )
        text = "\n".join(str(chunk.get("text") or "") for chunk in ordered)
        url = str(ordered[0].get("original_url") or "")
        details = parse_revision_issue_text(
            text,
            revision_id=revision_id,
            issue_url=url,
        )
        if not details.get("official_description"):
            continue
        candidates.setdefault(revision_id, []).append(
            (
                str(ordered[0].get("downloaded_at") or ""),
                document_key,
                details,
            )
        )
    return {
        revision_id: max(values, key=lambda value: (value[0], value[1]))[2]
        for revision_id, values in candidates.items()
    }


def load_revision_issue_details(
    paths: Iterable[Path],
    revision_ids: Iterable[str],
    *,
    archive_roots: Iterable[Path] = (),
    packaged_chunks_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load official per-issue explanations without network or AI requests."""

    requested = {
        str(revision_id).upper()
        for revision_id in revision_ids
        if str(revision_id).strip()
    }
    if not requested:
        return {}
    source_paths = list(dict.fromkeys(Path(path) for path in paths))
    roots = list(dict.fromkeys([*_official_archive_roots(source_paths), *archive_roots]))
    cached_paths = _cached_issue_page_paths(roots, requested)
    details = _issue_details_from_html_paths(
        [*source_paths, *cached_paths],
        requested,
    )
    missing = requested.difference(details)
    chunks_path = packaged_chunks_path
    if chunks_path is None and roots:
        chunks_path = _default_packaged_chunks_path()
    if missing and chunks_path and chunks_path.is_file():
        details.update(_issue_details_from_chunks(chunks_path, missing))
    return {
        revision_id: details[revision_id]
        for revision_id in sorted(details)
        if revision_id in requested
    }


def enrich_latest_updates(
    payload: Mapping[str, Any],
    revision_issues: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Attach authoritative issue descriptions to a prebuilt latest feed."""

    enriched = dict(payload)
    raw_items = payload.get("items", [])
    items = [dict(item) for item in raw_items if isinstance(item, Mapping)]
    issue_payload: dict[str, dict[str, Any]] = {}
    for revision_id, raw_details in revision_issues.items():
        details = dict(raw_details)
        if not details.get("official_description"):
            continue
        normalized_id = str(revision_id).upper()
        issue_payload[normalized_id] = details

    for item in items:
        revision_id = str(item.get("revision_id") or "").upper()
        details = issue_payload.get(revision_id)
        if not details:
            continue
        item["explanation"] = details["official_description"]
        item["issue_title"] = details.get("issue_title", "")
        item["affected_sections"] = details.get("affected_sections", "")
        item["effectiveness_note"] = details.get("effectiveness_note", "")
        issue_url = str(details.get("issue_url") or "")
        if issue_url and _ISSUE_PAGE_PATH_RE.search(str(item.get("url") or "")):
            # ERCOT report indexes sometimes expose an implementation date as
            # though it were the issue's publication date. The issue page's
            # own "Date Posted" field is the authoritative display date.
            item["published_date"] = details.get("date_posted", item.get("published_date", ""))

    enriched["items"] = items
    enriched["count"] = len(items)
    enriched["generated_at"] = datetime.now(timezone.utc).isoformat()
    enriched["revision_issues"] = {
        revision_id: issue_payload[revision_id]
        for revision_id in sorted(issue_payload)
    }
    return enriched


def _explanation(source: str, title: str, document_number: str, url: str = "") -> str:
    source_upper = source.upper()
    revision = revision_request_identity(document_number, title, url)
    if revision:
        revision_id, prefix = revision
        return (
            f"New {REVISION_REQUEST_NAMES[prefix]} material. It may change or clarify ERCOT "
            f"requirements; search the assistant for {revision_id} to review details."
        )
    if source_upper in {"DWG", "SSWG"}:
        return "New working-group material relevant to dynamic or steady-state modeling and study procedures."
    if source_upper in {"PLANNING GUIDE", "OPERATING GUIDE", "PROTOCOLS"}:
        return f"New {source.lower()} material that may update an ERCOT requirement or implementation detail."
    return "New ERCOT technical material now available for cited search and explanation in the assistant."


def build_latest_updates(
    paths: Iterable[Path],
    *,
    output_path: Path,
    minimum_year: int = 2026,
    archive_roots: Iterable[Path] = (),
    packaged_chunks_path: Path | None = None,
) -> dict[str, Any]:
    source_paths = list(dict.fromkeys(Path(path) for path in paths))
    unique: dict[str, dict[str, Any]] = {}
    for path in source_paths:
        if not path.is_file() or path.name.endswith(".metadata.json"):
            continue
        metadata_path = path.with_name(f"{path.name}.metadata.json")
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        source = str(metadata.get("source_label") or path.parent.parent.name).strip()
        if source.upper() in EXCLUDED_SOURCES:
            continue
        original_url = str(metadata.get("original_url") or metadata.get("final_url") or "")
        if any(fragment in original_url.lower() for fragment in EXCLUDED_URL_FRAGMENTS):
            continue
        published = str(metadata.get("published_date") or metadata.get("published_hint") or "")
        published_years = [int(value) for value in re.findall(r"\b(20\d{2})\b", published)]
        path_years = [int(value) for value in re.findall(r"\b(20\d{2})\b", path.as_posix())]
        years = published_years or path_years
        if not years or max(years) < minimum_year:
            continue
        content_hash = str(metadata.get("content_sha256") or "")
        if not content_hash:
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        title = str(metadata.get("title") or path.name).strip()
        if title.casefold() in EXCLUDED_NAVIGATION_TITLES:
            continue
        number = str(metadata.get("document_number") or "").strip()
        revision = revision_request_identity(number, title, original_url)
        record = unique.setdefault(
            content_hash,
            {
                "content_sha256": content_hash,
                "title": title,
                "document_number": number,
                "source": source,
                "published_date": published,
                "downloaded_at": str(metadata.get("downloaded_at") or ""),
                "status": str(metadata.get("document_status") or ""),
                "url": original_url,
                "sources": [],
                "revision_id": revision[0] if revision else "",
                "revision_family": revision[1] if revision else "",
            },
        )
        if source and source not in record["sources"]:
            record["sources"].append(source)

    items = list(unique.values())
    for item in items:
        item["sources"].sort()
        item["explanation"] = _explanation(
            item["source"], item["title"], item["document_number"], item["url"]
        )
    items.sort(
        key=lambda item: (item["downloaded_at"], item["published_date"], item["title"]),
        reverse=True,
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "minimum_year": minimum_year,
        "count": len(items),
        "items": items,
    }
    revision_ids = {
        str(item.get("revision_id") or "")
        for item in items
        if item.get("revision_id")
    }
    if revision_ids:
        issue_details = load_revision_issue_details(
            source_paths,
            revision_ids,
            archive_roots=archive_roots,
            packaged_chunks_path=packaged_chunks_path,
        )
        payload = enrich_latest_updates(payload, issue_details)
        payload["items"].sort(
            key=lambda item: (
                item.get("downloaded_at", ""),
                item.get("published_date", ""),
                item.get("title", ""),
            ),
            reverse=True,
        )
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def load_latest_updates(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"count": 0, "items": []}
    return value if isinstance(value, dict) else {"count": 0, "items": []}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-file", type=Path)
    parser.add_argument(
        "--enrich-existing",
        type=Path,
        help="Enrich an existing feed from cached official issue pages without rebuilding its items.",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path(__file__).with_name("NEWS") / "official",
    )
    parser.add_argument("--packaged-chunks", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
    )
    args = parser.parse_args()
    if bool(args.path_file) == bool(args.enrich_existing):
        parser.error("provide exactly one of --path-file or --enrich-existing")

    output = args.output or args.enrich_existing or Path(__file__).with_name(
        "latest_ercot_updates.json"
    )
    if args.enrich_existing:
        payload = load_latest_updates(args.enrich_existing)
        revision_ids = {
            str(item.get("revision_id") or "")
            for item in payload.get("items", [])
            if isinstance(item, Mapping) and item.get("revision_id")
        }
        details = load_revision_issue_details(
            [],
            revision_ids,
            archive_roots=[args.archive_root],
            packaged_chunks_path=args.packaged_chunks or _default_packaged_chunks_path(),
        )
        payload = enrich_latest_updates(payload, details)
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "output": str(output),
                    "count": payload["count"],
                    "revision_issues": len(payload.get("revision_issues", {})),
                }
            )
        )
        return

    paths = [
        Path(line.strip())
        for line in args.path_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    payload = build_latest_updates(
        paths,
        output_path=output,
        archive_roots=[args.archive_root],
        packaged_chunks_path=args.packaged_chunks,
    )
    print(json.dumps({"output": str(output), "count": payload["count"]}))


if __name__ == "__main__":
    main()
