"""Pure helpers for identifying and comparing ERCOT document sections.

The ingestion pipeline intentionally treats exact file bytes as immutable
content.  This module adds a separate logical view for change reporting:

* group different files that represent versions of the same document;
* parse numbered sections without treating numbered paragraphs as headings;
* compare section text while ignoring whitespace-only edits; and
* retain an independently usable citation for each side of a comparison.

There is no filesystem, network, embedding, or model dependency here.  Callers
can therefore use these helpers during ingestion, in a dashboard, or in tests
without triggering paid work.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal, Mapping
from urllib.parse import unquote, urlsplit, urlunsplit


ChangeStatus = Literal["added", "modified", "removed", "unchanged"]

_PAGE_MARKER_RE = re.compile(r"^\s*\[Page\s+(\d+)\]\s*$", re.IGNORECASE)
_HEADING_RE = re.compile(
    r"^\s*(?:(?P<section_word>SECTION)\s+)?"
    r"(?P<number>\d{1,2}(?:\.\d+){0,7}[A-Za-z]?)"
    r"(?:\s*[-:–—]\s*|\s+)"
    r"(?P<title>\S.*?)\s*$",
    re.IGNORECASE,
)
_CONCATENATED_HEADING_RE = re.compile(
    r"^\s*(?P<number>\d{1,2}(?:\.\d+){0,7})(?P<title>[A-Z]\S.*?)\s*$"
)
_NUMBER_ONLY_HEADING_RE = re.compile(
    r"^\s*SECTION\s+(?P<number>\d{1,2}(?:\.\d+){0,7}[A-Za-z]?)\s*$",
    re.IGNORECASE,
)
_TOC_DOTS_RE = re.compile(r"\.{2,}\s*\d+\s*$")
_TOC_TAB_PAGE_RE = re.compile(r"\t+\s*\d+\s*$")
_REVISION_REQUEST_RE = re.compile(
    r"\b(NPRR|PGRR|NOGRR|OBDRR|SCR|RRGRR|VCMRR|COPMGRR|LPGRR|RMGRR|SMOGRR|CMGRR)"
    r"\s*[-_ ]?\s*(\d{1,6})\b",
    re.IGNORECASE,
)
_REVERSE_REVISION_REQUEST_RE = re.compile(
    r"\b(\d{1,6})(NPRR|PGRR|NOGRR|OBDRR|SCR|RRGRR|VCMRR|COPMGRR|LPGRR|"
    r"RMGRR|SMOGRR|CMGRR)\b",
    re.IGNORECASE,
)
_SECTION_REFERENCE_RE = re.compile(
    r"\bsection\s+(\d{1,2}(?:\.\d+){0,7}[A-Za-z]?)\b",
    re.IGNORECASE,
)
_SPLIT_SECTION_FILENAME_RE = re.compile(
    r"^(?P<section>\d{1,2}[A-Za-z]?)(?:[-_.\s]|$)",
    re.IGNORECASE,
)
_DATE_TOKEN_RE = re.compile(
    r"""
    (?:
        \b20\d{2}[-_/]\d{1,2}[-_/]\d{1,2}\b
        |
        \b\d{1,2}[-_/]\d{1,2}[-_/](?:20)?\d{2}\b
        |
        \b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|
        Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|
        Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)
        \s+\d{1,2},?\s+20\d{2}\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
_VERSION_TOKEN_RE = re.compile(
    r"\b(?:revision|rev(?:ision)?\.?|version|ver\.?)\s*[-_:]?\s*[A-Za-z]?\d+[A-Za-z]?\b",
    re.IGNORECASE,
)
_LIFECYCLE_TOKEN_RE = re.compile(
    r"\b(?:approved|effective|pending|withdrawn|rejected|draft|redline|clean)\b",
    re.IGNORECASE,
)
_DOCUMENT_EXTENSION_RE = re.compile(
    r"\.(?:pdf|docx?|xlsx?|html?|txt|csv|pptx?|zip)\b",
    re.IGNORECASE,
)
_DOCUMENT_FAMILY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("nodal-protocols", re.compile(r"\bnodal protocols?\b")),
    ("planning-guide", re.compile(r"\bplanning guides?\b|\bguides planning\b")),
    ("nodal-operating-guide", re.compile(r"\bnodal operating guides?\b")),
    ("operating-guide", re.compile(r"\boperating guides?\b")),
    ("other-binding-document", re.compile(r"\bother binding documents?\b")),
    ("commercial-operations-market-guide", re.compile(r"\bcommercial operations market guides?\b")),
    ("load-profiling-guide", re.compile(r"\bload profiling guides?\b")),
    ("retail-market-guide", re.compile(r"\bretail market guides?\b")),
    ("settlement-metering-operating-guide", re.compile(r"\bsettlement metering operating guides?\b")),
    ("competitive-metering-guide", re.compile(r"\bcompetitive metering guides?\b")),
    ("verifiable-cost-manual", re.compile(r"\bverifiable cost manuals?\b")),
    ("resource-registration-glossary", re.compile(r"\bresource registration glossar(?:y|ies)\b")),
    ("resource-integration", re.compile(r"\bresource integration\b|\binterconnection handbook\b")),
    ("dwg", re.compile(r"\bdynamics working group\b|\bdwg\b")),
    ("sswg", re.compile(r"\bsteady state working group\b|\bsswg\b")),
)


@dataclass(frozen=True)
class NumberedSection:
    """One numbered section parsed from a document's extracted text."""

    number: str
    title: str
    text: str
    normalized_text: str
    order: int
    page_start: int | None = None
    page_end: int | None = None

    @property
    def locator(self) -> str:
        """Return a human-readable section/page locator."""

        value = f"Section {self.number}"
        if self.page_start is not None:
            if self.page_end is not None and self.page_end != self.page_start:
                value += f", pages {self.page_start}-{self.page_end}"
            else:
                value += f", page {self.page_start}"
        return value


@dataclass(frozen=True)
class SectionChange:
    """The comparison result for one logical section number."""

    section_number: str
    status: ChangeStatus
    old_section: NumberedSection | None
    new_section: NumberedSection | None
    old_citation: str | None
    new_citation: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "section_number": self.section_number,
            "status": self.status,
            "old_section": asdict(self.old_section) if self.old_section else None,
            "new_section": asdict(self.new_section) if self.new_section else None,
            "old_citation": self.old_citation,
            "new_citation": self.new_citation,
        }


@dataclass(frozen=True)
class DocumentChangeReport:
    """A complete section comparison between two document versions."""

    logical_key: str
    old_logical_key: str
    new_logical_key: str
    same_logical_document: bool
    changes: tuple[SectionChange, ...]

    def _with_status(self, status: ChangeStatus) -> tuple[SectionChange, ...]:
        return tuple(change for change in self.changes if change.status == status)

    @property
    def added(self) -> tuple[SectionChange, ...]:
        return self._with_status("added")

    @property
    def modified(self) -> tuple[SectionChange, ...]:
        return self._with_status("modified")

    @property
    def removed(self) -> tuple[SectionChange, ...]:
        return self._with_status("removed")

    @property
    def unchanged(self) -> tuple[SectionChange, ...]:
        return self._with_status("unchanged")

    @property
    def counts(self) -> dict[str, int]:
        return {
            status: len(self._with_status(status))
            for status in ("added", "modified", "removed", "unchanged")
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "logical_key": self.logical_key,
            "old_logical_key": self.old_logical_key,
            "new_logical_key": self.new_logical_key,
            "same_logical_document": self.same_logical_document,
            "counts": self.counts,
            "changes": [change.to_dict() for change in self.changes],
        }


def _clean(value: Any) -> str:
    return " ".join(str(value or "").replace("\u00a0", " ").split())


def _slug(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", _clean(value)).casefold()
    return "-".join(re.findall(r"[a-z0-9]+", normalized))


def _normalize_section_number(value: str) -> str:
    match = re.fullmatch(r"(\d{1,2}(?:\.\d+){0,7})([A-Za-z]?)", value.strip())
    if match is None:
        return value.strip().upper()
    numeric, suffix = match.groups()
    parts = [str(int(part)) for part in numeric.split(".")]
    return ".".join(parts) + suffix.upper()


def _normalize_comparison_text(title: str, body: str) -> str:
    """Normalize layout whitespace while preserving substantive characters."""

    value = unicodedata.normalize("NFKC", f"{title}\n{body}")
    value = value.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", value).strip()


def _looks_like_top_level_heading(number: str, title: str, explicit: bool) -> bool:
    numeric = re.match(r"\d+", number)
    if numeric is None or int(numeric.group(0)) < 1:
        return False
    if explicit or "." in number:
        return True
    if number[-1:].isalpha():
        return True
    letters = [character for character in title if character.isalpha()]
    if not letters:
        return False
    uppercase = sum(character.isupper() for character in letters)
    if uppercase / len(letters) >= 0.75:
        return True
    return (
        len(title) <= 160
        and title[0].isupper()
        and title[-1] not in ".;,?!"
    )


def _heading_from_line(line: str) -> tuple[str, str] | None:
    if _TOC_DOTS_RE.search(line) or _TOC_TAB_PAGE_RE.search(line):
        return None
    number_only = _NUMBER_ONLY_HEADING_RE.match(line)
    if number_only:
        return _normalize_section_number(number_only.group("number")), ""
    match = _HEADING_RE.match(line)
    concatenated = False
    if match is None:
        match = _CONCATENATED_HEADING_RE.match(line)
        concatenated = match is not None
    if match is None:
        return None
    number = _normalize_section_number(match.group("number"))
    title = _clean(match.group("title"))
    if (
        concatenated
        and len(title) > 2
        and title[0].isupper()
        and title[1].isupper()
        and not title.split(maxsplit=1)[0].isupper()
        and any(character.islower() for character in title[2:])
    ):
        # Extracted DOCX headings commonly omit the separator in values such
        # as ``8ASettlement Process``.  Preserve the alpha section suffix while
        # leaving an all-caps heading such as ``9LARGE LOAD`` as Section 9.
        number += title[0]
        title = title[1:]
    if not re.search(r"[A-Za-z]", title):
        return None
    explicit = False if concatenated else bool(match.group("section_word"))
    if not _looks_like_top_level_heading(number, title, explicit):
        return None
    return number, title


def parse_numbered_sections(text: str) -> tuple[NumberedSection, ...]:
    """Parse numbered headings and their direct body text.

    Numbered paragraphs such as ``(1)`` are deliberately not headings.  When a
    table of contents and a substantive section repeat the same number, the
    later substantive occurrence wins.
    """

    occurrences: list[NumberedSection] = []
    current_number: str | None = None
    current_title = ""
    current_lines: list[str] = []
    current_order = 0
    current_page: int | None = None
    section_page_start: int | None = None
    section_page_end: int | None = None

    def finish() -> None:
        nonlocal current_number, current_title, current_lines
        nonlocal section_page_start, section_page_end
        if current_number is None:
            return
        body = "\n".join(current_lines).strip()
        occurrences.append(
            NumberedSection(
                number=current_number,
                title=current_title,
                text=body,
                normalized_text=_normalize_comparison_text(current_title, body),
                order=current_order,
                page_start=section_page_start,
                page_end=section_page_end,
            )
        )
        current_number = None
        current_title = ""
        current_lines = []
        section_page_start = None
        section_page_end = None

    for line in str(text or "").splitlines():
        page_match = _PAGE_MARKER_RE.match(line)
        if page_match:
            current_page = int(page_match.group(1))
            continue

        heading = _heading_from_line(line)
        if heading is not None:
            finish()
            current_number, current_title = heading
            current_order = len(occurrences)
            section_page_start = current_page
            section_page_end = current_page
            continue

        if current_number is not None:
            current_lines.append(line)
            if line.strip() and current_page is not None:
                section_page_end = current_page
    finish()

    # A table of contents or cover heading usually precedes the substantive
    # occurrence. Selecting the last occurrence also prevents a front-matter
    # date from being treated as the body of the real top-level section.
    best: dict[str, NumberedSection] = {}
    for section in occurrences:
        best[section.number] = section
    return tuple(sorted(best.values(), key=lambda section: section.order))


def _normalized_url(value: Any) -> str:
    raw = _clean(value)
    if not raw:
        return ""
    try:
        split = urlsplit(raw)
    except ValueError:
        return ""
    if split.scheme.lower() not in {"http", "https"} or not split.netloc:
        return ""
    path = re.sub(
        r"(?i)(?:[-_/](?:20)?\d{2}[-_/]\d{1,2}[-_/]\d{1,2})(?=\.|/|$)",
        "",
        split.path,
    )
    return urlunsplit((split.scheme.lower(), split.netloc.lower(), path.rstrip("/"), "", ""))


def _normalized_family_title(value: Any) -> str:
    title = unicodedata.normalize("NFKC", _clean(value))
    title = _DOCUMENT_EXTENSION_RE.sub(" ", title)
    title = _DATE_TOKEN_RE.sub(" ", title)
    title = _VERSION_TOKEN_RE.sub(" ", title)
    title = _LIFECYCLE_TOKEN_RE.sub(" ", title)
    return _slug(title)


def _revision_request_key(searchable: str) -> str | None:
    forward = _REVISION_REQUEST_RE.search(searchable)
    if forward:
        return f"revision-request:{forward.group(1).lower()}{int(forward.group(2))}"
    reverse = _REVERSE_REVISION_REQUEST_RE.search(searchable)
    if reverse:
        return f"revision-request:{reverse.group(2).lower()}{int(reverse.group(1))}"
    return None


def _document_family_hint(values: Mapping[str, Any]) -> str | None:
    """Prefer the artifact URL over a potentially inherited crawler label."""

    candidates: list[Any] = [
        values.get("original_url"),
        values.get("final_url"),
        *(values.get("url_aliases") or []),
        values.get("source_page_url"),
        *(values.get("source_page_urls") or []),
        values.get("filename"),
        values.get("source_path"),
        values.get("title"),
    ]
    for value in candidates:
        normalized = re.sub(
            r"[^a-z0-9]+",
            " ",
            unicodedata.normalize("NFKC", unquote(_clean(value))).casefold(),
        ).strip()
        if not normalized:
            continue
        for family, pattern in _DOCUMENT_FAMILY_PATTERNS:
            if pattern.search(normalized):
                return family

    # A crawler page can inherit the source-page label even when its artifact
    # is an unrelated HTML link.  Trust that fallback only for actual office
    # documents; otherwise keep a title/URL-specific logical key.
    artifact_values = [
        values.get("original_url"),
        values.get("final_url"),
        values.get("filename"),
        values.get("source_path"),
    ]
    document_artifact = any(
        re.search(r"\.(?:pdf|docx?)\b", _clean(value), re.IGNORECASE)
        for value in artifact_values
    )
    if document_artifact:
        for value in (values.get("source_kind"), values.get("source_label")):
            normalized = re.sub(
                r"[^a-z0-9]+",
                " ",
                unicodedata.normalize("NFKC", unquote(_clean(value))).casefold(),
            ).strip()
            for family, pattern in _DOCUMENT_FAMILY_PATTERNS:
                if pattern.search(normalized):
                    return family
    return None


def logical_document_key(metadata: Mapping[str, Any] | None) -> str:
    """Return a version-stable best-effort key for one logical document.

    Explicit family IDs and numbered revision requests take precedence.  Split
    current-guide filenames are grouped by document kind and section number.
    The remaining fallbacks use a date/version-stripped title and finally the
    canonical URL or source path.
    """

    values = dict(metadata or {})
    explicit = _clean(
        values.get("logical_document_key")
        or values.get("document_family_id")
        or values.get("series_id")
    )
    if explicit:
        return f"explicit:{_slug(explicit)}"

    number_search_values: list[Any] = [
        values.get("document_number"),
        values.get("title"),
        values.get("filename"),
        values.get("source_path"),
        values.get("original_url"),
        values.get("final_url"),
        values.get("source_page_url"),
        *(values.get("source_page_urls") or []),
        *(values.get("url_aliases") or []),
    ]
    revision_request = _revision_request_key(
        " ".join(_clean(value) for value in number_search_values)
    )
    if revision_request:
        return revision_request

    document_number = _clean(values.get("document_number"))
    if document_number:
        return f"document-number:{_slug(document_number)}"

    family = _document_family_hint(values)
    kind = family or _slug(
        values.get("source_kind")
        or values.get("source_label")
        or values.get("source_category")
        or "ercot-document"
    )
    section_search = " ".join(
        _clean(values.get(field))
        for field in ("title", "filename", "source_path")
    )
    section_match = _SECTION_REFERENCE_RE.search(section_search)
    if section_match:
        section = _normalize_section_number(section_match.group(1)).lower()
        return f"{kind}:section-{section}"

    filename = _clean(values.get("filename"))
    if not filename:
        filename = _clean(values.get("source_path")).replace("\\", "/").rsplit("/", 1)[-1]
    split_section = _SPLIT_SECTION_FILENAME_RE.match(filename)
    kind_upper = _clean(values.get("source_kind") or values.get("source_label")).upper()
    if split_section and any(
        term in kind_upper for term in ("PROTOCOL", "PLANNING GUIDE", "OPERATING GUIDE")
    ):
        section = _normalize_section_number(split_section.group("section")).lower()
        return f"{kind}:section-{section}"

    if family:
        return f"{family}:full"

    title_key = _normalized_family_title(values.get("title") or filename)
    if title_key:
        return f"{kind}:title-{title_key}"

    for field in ("original_url", "final_url", "source_page_url"):
        normalized_url = _normalized_url(values.get(field))
        if normalized_url:
            return f"{kind}:url-{_slug(normalized_url)}"

    source_path = _slug(values.get("source_path") or values.get("source"))
    return f"{kind}:path-{source_path or 'unknown'}"


def format_section_citation(
    section: NumberedSection,
    metadata: Mapping[str, Any] | None = None,
    *,
    version_label: str | None = None,
) -> str:
    """Format a compact citation for one version-specific section."""

    values = dict(metadata or {})
    parts: list[str] = []
    if version_label:
        parts.append(_clean(version_label))
    title = _clean(
        values.get("title")
        or values.get("document_number")
        or values.get("filename")
        or values.get("source_kind")
        or "ERCOT document"
    )
    parts.append(title)
    parts.append(section.locator)
    for label, field in (
        ("Status", "document_status"),
        ("Effective", "effective_date"),
        ("Published", "published_date"),
        ("Revision", "revision"),
    ):
        value = _clean(values.get(field))
        if value:
            parts.append(f"{label} {value}")
    source = _clean(values.get("original_url") or values.get("source_path") or values.get("source"))
    if source:
        parts.append(source)
    return "[" + " | ".join(parts) + "]"


def _section_sort_key(number: str) -> tuple[tuple[int, str], ...]:
    values: list[tuple[int, str]] = []
    for part in number.split("."):
        match = re.fullmatch(r"(\d+)([A-Za-z]?)", part)
        if match:
            values.append((int(match.group(1)), match.group(2).upper()))
        else:
            values.append((10**9, part.upper()))
    return tuple(values)


def _indexed_sections(sections: Iterable[NumberedSection]) -> dict[str, NumberedSection]:
    return {section.number: section for section in sections}


def compare_document_versions(
    old_text: str,
    new_text: str,
    old_metadata: Mapping[str, Any] | None = None,
    new_metadata: Mapping[str, Any] | None = None,
) -> DocumentChangeReport:
    """Compare two versions and classify every numbered section.

    The comparison includes each heading's title and body, but removes layout
    whitespace and page markers.  Added sections have only a new citation;
    removed sections have only an old citation; modified and unchanged sections
    retain both.
    """

    old_values = dict(old_metadata or {})
    new_values = dict(new_metadata or {})
    old_key = logical_document_key(old_values)
    new_key = logical_document_key(new_values)
    same_document = old_key == new_key
    report_key = old_key if same_document else f"{old_key} -> {new_key}"

    old_sections = _indexed_sections(parse_numbered_sections(old_text))
    new_sections = _indexed_sections(parse_numbered_sections(new_text))
    changes: list[SectionChange] = []
    for number in sorted(set(old_sections) | set(new_sections), key=_section_sort_key):
        old_section = old_sections.get(number)
        new_section = new_sections.get(number)
        if old_section is None:
            status: ChangeStatus = "added"
        elif new_section is None:
            status = "removed"
        elif old_section.normalized_text == new_section.normalized_text:
            status = "unchanged"
        else:
            status = "modified"
        changes.append(
            SectionChange(
                section_number=number,
                status=status,
                old_section=old_section,
                new_section=new_section,
                old_citation=(
                    format_section_citation(old_section, old_values, version_label="OLD")
                    if old_section
                    else None
                ),
                new_citation=(
                    format_section_citation(new_section, new_values, version_label="NEW")
                    if new_section
                    else None
                ),
            )
        )
    return DocumentChangeReport(
        logical_key=report_key,
        old_logical_key=old_key,
        new_logical_key=new_key,
        same_logical_document=same_document,
        changes=tuple(changes),
    )


# A short alias is convenient for change-report callers while the longer name
# remains self-documenting at integration boundaries.
compare_versions = compare_document_versions
