"""Monitor ERCOT pages for newly posted documents and summarize them.

This script is designed to be called from n8n, cron, or manually:
1. Read label=url pairs from `ercot_links`
2. Fetch each ERCOT page
3. Discover likely document/detail links in the page body
4. Compare against local state to find new items
5. Atomically archive new official documents with provenance metadata
6. Extract best-effort text from the persisted bytes and emit a JSON payload
7. Optionally invoke incremental RAG ingestion for the durable downloads

Example:
    python ercot_link_monitor.py

Optional environment variables:
    ERCOT_LINKS_FILE=/path/to/ercot_links
    ERCOT_LINK_STATE_FILE=/path/to/.ercot_link_state.json
    ERCOT_LINK_MAX_ITEMS_PER_SOURCE=5
    ERCOT_LINK_MAX_KNOWN_RECHECKS_PER_SOURCE=10
    ERCOT_LINK_MAX_UNSEEN_ATTEMPTS_PER_SOURCE=20
    ERCOT_LINK_MAX_NESTED_ITEMS_PER_SOURCE=20
    ERCOT_LINK_REPORT_WINDOW=100
    ERCOT_LINK_MAX_STATE_ITEMS_PER_SOURCE=2000
    ERCOT_LINK_MAX_RESPONSE_BYTES=52428800
    ERCOT_LINK_MAX_SUMMARY_CHARS=1200
    ERCOT_LINK_MAX_OUTPUT_ITEMS=40
    ERCOT_LINK_MAX_TELEGRAM_CHARS=3900
    ERCOT_LINK_TELEGRAM_BOT_TOKEN=...
    ERCOT_LINK_TELEGRAM_CHAT_ID=@ERCOTNEWS
    ERCOT_LINK_SEND_TELEGRAM=true
    ERCOT_LINK_SEND_NO_UPDATES=false
    ERCOT_OFFICIAL_DOCUMENT_DIR=/path/to/ERCOTAPI/NEWS/official
    ERCOT_RAG_AUTO_INGEST=true
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import os
import re
import tempfile
import zipfile
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urljoin, urlparse
from xml.etree import ElementTree as ET

import requests

try:
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None


DEFAULT_LINKS_FILE = Path(__file__).with_name("ercot_links")
DEFAULT_STATE_FILE = Path(__file__).with_name(".ercot_link_state.json")
DEFAULT_OFFICIAL_DOCUMENT_DIR = Path(__file__).with_name("NEWS") / "official"
REQUEST_TIMEOUT = 20
MAX_REDIRECTS = 5
MAX_ITEMS_PER_SOURCE = int(os.getenv("ERCOT_LINK_MAX_ITEMS_PER_SOURCE", "5"))
MAX_KNOWN_RECHECKS_PER_SOURCE = int(
    os.getenv("ERCOT_LINK_MAX_KNOWN_RECHECKS_PER_SOURCE", "10")
)
MAX_UNSEEN_ATTEMPTS_PER_SOURCE = int(
    os.getenv(
        "ERCOT_LINK_MAX_UNSEEN_ATTEMPTS_PER_SOURCE",
        str(max(20, MAX_ITEMS_PER_SOURCE * 4)),
    )
)
MAX_NESTED_ITEMS_PER_SOURCE = int(
    os.getenv("ERCOT_LINK_MAX_NESTED_ITEMS_PER_SOURCE", "20")
)
REPORT_CANDIDATE_WINDOW = int(os.getenv("ERCOT_LINK_REPORT_WINDOW", "100"))
MAX_STATE_ITEMS_PER_SOURCE = int(os.getenv("ERCOT_LINK_MAX_STATE_ITEMS_PER_SOURCE", "2000"))
MAX_RESPONSE_BYTES = int(os.getenv("ERCOT_LINK_MAX_RESPONSE_BYTES", str(50 * 1024 * 1024)))
MAX_SUMMARY_CHARS = int(os.getenv("ERCOT_LINK_MAX_SUMMARY_CHARS", "1200"))
MAX_OUTPUT_ITEMS = int(os.getenv("ERCOT_LINK_MAX_OUTPUT_ITEMS", "40"))
MAX_TELEGRAM_CHARS = int(os.getenv("ERCOT_LINK_MAX_TELEGRAM_CHARS", "3900"))
SEND_TELEGRAM = os.getenv("ERCOT_LINK_SEND_TELEGRAM", "false").lower() in {"1", "true", "yes"}
SEND_NO_UPDATES = os.getenv("ERCOT_LINK_SEND_NO_UPDATES", "false").lower() in {"1", "true", "yes"}
DEFAULT_TELEGRAM_CHAT_ID = "@ERCOTNEWS"
USER_AGENT = "ERCOT-Link-Monitor/1.0 (+https://github.com/AmirExir/portfolio)"
STATE_HASH_SEPARATOR = "|sha256="
STATE_CURSOR_PREFIX = "__ercot_monitor_cursor__|"

CONTENT_TYPE_EXTENSIONS = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/msword": ".doc",
    "application/vnd.ms-excel": ".xls",
    "application/zip": ".zip",
    "application/json": ".json",
    "text/csv": ".csv",
    "text/html": ".html",
    "text/plain": ".txt",
}

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkg": "http://schemas.openxmlformats.org/package/2006/relationships",
}

TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".xml", ".html", ".htm"}
DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".zip",
    ".csv",
    ".txt",
    ".json",
}
INGESTIBLE_ARCHIVE_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".html",
    ".htm",
    ".docx",
    ".csv",
    ".xlsx",
}
SKIP_LINK_PREFIXES = (
    "mailto:",
    "tel:",
    "javascript:",
    "#",
)
SKIP_PATH_PARTS = (
    "/sitemap",
    "/privacy",
    "/glossary",
    "/calendar",
    "/contact",
    "/careers",
    "/about",
    "/market-participants",
)
MEETING_HINTS = (
    "meeting",
    "agenda",
    "minutes",
    "presentation",
    "workshop",
    "webcast",
    "packet",
    "briefing",
    "operational overview",
)


def first_env(names: Sequence[str], default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


TELEGRAM_BOT_TOKEN = first_env(
    (
        "ERCOT_LINK_TELEGRAM_BOT_TOKEN",
        "ERCOT_NEWS_TELEGRAM_BOT_TOKEN",
        "TELEGRAM_BOT_TOKEN",
    )
)
TELEGRAM_CHAT_ID = first_env(
    (
        "ERCOT_LINK_TELEGRAM_CHAT_ID",
        "ERCOT_NEWS_TELEGRAM_CHAT_ID",
        "TELEGRAM_CHAT_ID",
    ),
    DEFAULT_TELEGRAM_CHAT_ID,
)


@dataclass
class SourceLink:
    label: str
    url: str


@dataclass
class DiscoveredItem:
    source_label: str
    source_url: str
    title: str
    url: str
    item_type: str
    context: str = ""
    published_hint: str = ""
    effective_date: str = ""
    state_tag: str = ""

    @property
    def fingerprint(self) -> str:
        suffix = f"|{self.state_tag}" if self.state_tag else ""
        return f"{self.source_label}|{self.url}{suffix}"


@dataclass
class ArchivedItem:
    path: Path
    metadata_path: Path
    content: bytes
    content_hash: str
    content_type: str
    final_url: str
    archive_status: str


def load_links(path: Path) -> List[SourceLink]:
    links: List[SourceLink] = []
    if not path.exists():
        raise FileNotFoundError(f"Links file not found: {path}")

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        label, url = line.split("=", 1)
        label = " ".join(label.strip().split())
        url = url.strip()
        if label and url:
            links.append(SourceLink(label=label, url=url))
    return links


def load_state(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        # State has historically been ``{source_label: [fingerprint, ...]}``.
        # Keep that public shape while ignoring malformed values instead of
        # accidentally turning a string into a list of individual characters.
        return {
            str(label): [str(entry) for entry in entries if isinstance(entry, str)]
            for label, entries in data.items()
            if isinstance(entries, list)
        }
    except Exception:
        return {}


def save_state(path: Path, state: Dict[str, List[str]]) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(state, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Durably replace a file without exposing partially written content."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _close_response(response: requests.Response) -> None:
    close = getattr(response, "close", None)
    if callable(close):
        close()


def _canonical_ercot_request_url(url: str) -> str:
    """Validate an ERCOT URL and force encrypted transport before a request."""

    if not is_allowed_domain(url):
        raise ValueError(f"Refusing to request a non-ERCOT host: {url}")
    parsed = urlparse(url)
    return parsed._replace(scheme="https").geturl()


def _safe_get(
    session: requests.Session,
    url: str,
    *,
    stream: bool,
) -> requests.Response:
    """GET an ERCOT URL while validating every redirect before following it."""

    current_url = _canonical_ercot_request_url(url)
    redirect_statuses = {301, 302, 303, 307, 308}
    for redirect_count in range(MAX_REDIRECTS + 1):
        response = session.get(
            current_url,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=False,
            stream=stream,
        )
        try:
            response.raise_for_status()
            raw_response_url = str(getattr(response, "url", "") or current_url)
            if not is_allowed_domain(raw_response_url):
                raise ValueError(
                    f"ERCOT request redirected to a disallowed host: {raw_response_url}"
                )
            response_url = _canonical_ercot_request_url(raw_response_url)
            status_code = int(getattr(response, "status_code", 200) or 200)
            location = str(getattr(response, "headers", {}).get("Location", "")).strip()
            if status_code not in redirect_statuses:
                return response
            if not location:
                raise ValueError(f"ERCOT redirect omitted a Location header: {response_url}")
            raw_next_url = urljoin(response_url, location)
            if not is_allowed_domain(raw_next_url):
                raise ValueError(
                    f"ERCOT request redirected to a disallowed host: {raw_next_url}"
                )
            next_url = _canonical_ercot_request_url(raw_next_url)
            if redirect_count >= MAX_REDIRECTS:
                raise ValueError(f"ERCOT request exceeded {MAX_REDIRECTS} redirects")
        except Exception:
            _close_response(response)
            raise
        _close_response(response)
        current_url = next_url
    raise ValueError(f"ERCOT request exceeded {MAX_REDIRECTS} redirects")


def fetch(session: requests.Session, url: str) -> requests.Response:
    return _safe_get(session, url, stream=False)


def fetch_archivable_content(
    session: requests.Session,
    url: str,
    *,
    max_bytes: int = MAX_RESPONSE_BYTES,
) -> Tuple[requests.Response, bytes]:
    """Fetch one archive candidate without buffering an unbounded response.

    Real ``requests`` responses are streamed and checked incrementally. Simple
    fake sessions used by tests and downstream callers need only expose a
    ``content`` attribute; that fallback is still size-checked before archival.
    """

    limit = max(1, int(max_bytes))
    response = _safe_get(session, url, stream=True)

    raw_length = str(getattr(response, "headers", {}).get("Content-Length", "")).strip()
    try:
        announced_length = int(raw_length) if raw_length else None
    except ValueError:
        announced_length = None
    if announced_length is not None and announced_length > limit:
        _close_response(response)
        raise ValueError(
            f"Downloaded item exceeds ERCOT_LINK_MAX_RESPONSE_BYTES "
            f"({announced_length} > {limit})"
        )

    iterator = getattr(response, "iter_content", None)
    if callable(iterator):
        chunks: List[bytes] = []
        received = 0
        try:
            for chunk in iterator(chunk_size=min(1024 * 1024, limit + 1)):
                if not chunk:
                    continue
                block = bytes(chunk)
                received += len(block)
                if received > limit:
                    raise ValueError(
                        f"Downloaded item exceeds ERCOT_LINK_MAX_RESPONSE_BYTES "
                        f"({received} > {limit})"
                    )
                chunks.append(block)
            content = b"".join(chunks)
        finally:
            _close_response(response)
    else:
        try:
            content = bytes(response.content)
            if len(content) > limit:
                raise ValueError(
                    f"Downloaded item exceeds ERCOT_LINK_MAX_RESPONSE_BYTES "
                    f"({len(content)} > {limit})"
                )
        finally:
            _close_response(response)
    return response, content


def fetch_source_payload(
    session: requests.Session,
    url: str,
    *,
    max_bytes: int = MAX_RESPONSE_BYTES,
) -> Tuple[requests.Response, bytes, str]:
    """Fetch and close a bounded source page, retaining its verified bytes."""

    response, content = fetch_archivable_content(session, url, max_bytes=max_bytes)
    # Minimal fake responses often provide ``text`` but no corresponding byte
    # body. Preserve that compatibility while applying the same byte limit.
    fake_text = (
        getattr(response, "text", "")
        if not callable(getattr(response, "iter_content", None))
        else ""
    )
    if not content and fake_text:
        fallback = str(fake_text).encode("utf-8")
        if len(fallback) > max(1, int(max_bytes)):
            raise ValueError(
                f"Source page exceeds ERCOT_LINK_MAX_RESPONSE_BYTES "
                f"({len(fallback)} > {max(1, int(max_bytes))})"
            )
        return response, fallback, str(fake_text)

    content_type = str(getattr(response, "headers", {}).get("Content-Type", ""))
    charset_match = re.search(r"charset=([^;\s]+)", content_type, re.IGNORECASE)
    encoding = str(getattr(response, "encoding", "") or "").strip()
    if not encoding and charset_match:
        encoding = charset_match.group(1).strip('"\'')
    text = content.decode(encoding or "utf-8", errors="replace")
    return response, content, text


def fetch_source_text(
    session: requests.Session,
    url: str,
    *,
    max_bytes: int = MAX_RESPONSE_BYTES,
) -> str:
    """Compatibility wrapper returning only bounded source-page text."""

    return fetch_source_payload(session, url, max_bytes=max_bytes)[2]


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def path_extension(url: str) -> str:
    return Path(urlparse(url).path).suffix.lower()


def env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def official_document_dir() -> Path:
    configured = first_env(
        ("ERCOT_OFFICIAL_DOCUMENT_DIR", "ERCOT_RAG_OFFICIAL_DIR"),
        str(DEFAULT_OFFICIAL_DOCUMENT_DIR),
    )
    return Path(configured).expanduser().resolve()


def safe_path_component(value: str, fallback: str = "source") -> str:
    component = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return component[:80] or fallback


def _response_extension(item: DiscoveredItem, response: requests.Response) -> str:
    disposition = response.headers.get("Content-Disposition", "")
    filename_match = re.search(
        r"filename\*?=(?:UTF-8''|\"?)([^\";]+)",
        disposition,
        flags=re.IGNORECASE,
    )
    if filename_match:
        extension = Path(filename_match.group(1).strip()).suffix.lower()
        if re.fullmatch(r"\.[a-z0-9]{1,8}", extension):
            return extension

    content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    if content_type in CONTENT_TYPE_EXTENSIONS and content_type != "application/octet-stream":
        return CONTENT_TYPE_EXTENSIONS[content_type]
    known_extensions = DOCUMENT_EXTENSIONS | TEXT_EXTENSIONS | {".htm"}
    for url in (str(getattr(response, "url", "") or ""), item.url):
        extension = path_extension(url)
        if extension in known_extensions:
            return extension
    if item.item_type and item.item_type != "page":
        extension = f".{item.item_type.lower().lstrip('.')}"
        if re.fullmatch(r"\.[a-z0-9]{1,8}", extension):
            return extension
    return ".bin"


def _archive_year(item: DiscoveredItem, response: requests.Response) -> str:
    hints = " ".join((item.published_hint, response.headers.get("Last-Modified", "")))
    match = re.search(r"\b(20\d{2})\b", hints)
    return match.group(1) if match else str(datetime.now(timezone.utc).year)


def _repository_path(path: Path) -> str:
    repository = Path(__file__).resolve().parent.parent
    try:
        return path.resolve().relative_to(repository).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _document_identity(item: DiscoveredItem) -> Optional[str]:
    match = re.search(
        r"\b(NPRR|PGRR|NOGRR|OBDRR|SCR|RRGRR|VCMRR)\s*[-_ ]?\s*(\d{1,6})\b",
        f"{item.title} {item.context}",
        re.IGNORECASE,
    )
    return f"{match.group(1).upper()}{match.group(2)}" if match else None


def _provenance_observation(
    item: DiscoveredItem,
    final_url: str,
    content_type: str,
) -> Dict[str, Any]:
    """Return the order-independent provenance record for one discovered URL."""

    return {
        "source_label": item.source_label,
        "source_kind": item.source_label,
        "source_page_url": item.source_url,
        "original_url": item.url,
        "final_url": final_url,
        "title": item.title,
        "document_number": _document_identity(item),
        "document_status": item.state_tag.title() if item.state_tag else None,
        "published_hint": item.published_hint or None,
        "published_date": item.published_hint or None,
        "effective_date": item.effective_date or None,
        "content_type": content_type,
    }


def _observation_identity(observation: Dict[str, Any]) -> Tuple[str, str, str]:
    """Identify a source observation while allowing its mutable status to change."""

    return (
        str(observation.get("source_label") or ""),
        str(observation.get("source_page_url") or ""),
        str(observation.get("original_url") or ""),
    )


def _observation_sort_key(observation: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(
        str(observation.get(field) or "")
        for field in (
            "original_url",
            "source_page_url",
            "source_label",
            "final_url",
            "title",
            "document_status",
            "published_date",
        )
    )


def _legacy_provenance(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Upgrade a version-one sidecar without losing its original citation."""

    original_url = str(metadata.get("original_url") or "").strip()
    if not original_url:
        return []
    return [
        {
            "source_label": metadata.get("source_label"),
            "source_kind": metadata.get("source_kind"),
            "source_page_url": metadata.get("source_page_url"),
            "original_url": original_url,
            "final_url": metadata.get("final_url") or original_url,
            "title": metadata.get("title"),
            "document_number": metadata.get("document_number"),
            "document_status": metadata.get("document_status"),
            "published_hint": metadata.get("published_hint"),
            "published_date": metadata.get("published_date") or metadata.get("published_hint"),
            "effective_date": metadata.get("effective_date"),
            "content_type": metadata.get("content_type"),
        }
    ]


def _merge_provenance(
    existing_metadata: Dict[str, Any],
    observation: Dict[str, Any],
) -> List[Dict[str, Any]]:
    raw = existing_metadata.get("provenance")
    existing = (
        [dict(value) for value in raw if isinstance(value, dict)]
        if isinstance(raw, list)
        else _legacy_provenance(existing_metadata)
    )
    by_identity = {_observation_identity(value): value for value in existing}
    # A repeated URL replaces its observation so lifecycle metadata can move
    # from Pending to Approved without accumulating contradictory records.
    by_identity[_observation_identity(observation)] = observation
    return sorted(by_identity.values(), key=_observation_sort_key)


def _newest_provenance_date(
    provenance: Sequence[Dict[str, Any]],
    *fields: str,
) -> Optional[str]:
    candidates: List[Tuple[datetime, str]] = []
    for observation in provenance:
        for field in fields:
            value = str(observation.get(field) or "").strip()
            parsed = _parsed_date(value)
            if value and parsed is not None:
                candidates.append((parsed, value))
    return max(candidates)[1] if candidates else None


def _aggregate_document_status(provenance: Sequence[Dict[str, Any]]) -> Optional[str]:
    priority = {
        "effective": 6,
        "approved": 5,
        "clean": 4,
        "pending": 3,
        "withdrawn": 2,
        "rejected": 1,
        "listed": 0,
    }
    candidates: List[Tuple[int, datetime, int, str]] = []
    for observation in provenance:
        status = str(observation.get("document_status") or "").strip()
        if not status:
            continue
        date_value = str(
            observation.get("effective_date")
            or observation.get("published_date")
            or observation.get("published_hint")
            or ""
        )
        parsed = _parsed_date(date_value)
        candidates.append(
            (
                1 if parsed is not None else 0,
                parsed or datetime.min,
                priority.get(status.lower(), 0),
                status,
            )
        )
    # Lifecycle recency is authoritative. Status precedence only breaks ties
    # between observations on the same date (or wholly undated aliases).
    return max(candidates)[3] if candidates else None


def _archive_metadata(
    *,
    item: DiscoveredItem,
    final_url: str,
    content_hash: str,
    content_type: str,
    destination: Path,
    size: int,
    existing_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    provenance = _merge_provenance(
        existing_metadata,
        _provenance_observation(item, final_url, content_type),
    )
    canonical = min(provenance, key=_observation_sort_key)
    downloaded_at = str(existing_metadata.get("downloaded_at") or "") or datetime.now(
        timezone.utc
    ).isoformat(timespec="seconds").replace("+00:00", "Z")
    url_aliases = sorted(
        {
            str(observation.get(field) or "")
            for observation in provenance
            for field in ("original_url", "final_url")
            if observation.get(field)
        }
    )
    source_page_urls = sorted(
        {
            str(observation.get("source_page_url") or "")
            for observation in provenance
            if observation.get("source_page_url")
        }
    )
    published_date = _newest_provenance_date(
        provenance,
        "published_date",
        "published_hint",
    ) or canonical.get("published_date") or canonical.get("published_hint")
    effective_date = (
        _newest_provenance_date(provenance, "effective_date")
        or canonical.get("effective_date")
    )
    return {
        "schema_version": 2,
        "source_authority": "ERCOT",
        "is_generated": False,
        "source_label": canonical.get("source_label"),
        "source_kind": canonical.get("source_kind"),
        "source_page_url": canonical.get("source_page_url"),
        "source_page_urls": source_page_urls,
        "original_url": canonical.get("original_url"),
        "final_url": canonical.get("final_url"),
        "url_aliases": url_aliases,
        "provenance": provenance,
        "title": canonical.get("title"),
        "document_number": canonical.get("document_number"),
        "document_status": _aggregate_document_status(provenance),
        "published_hint": published_date,
        "published_date": published_date,
        "effective_date": effective_date,
        "downloaded_at": downloaded_at,
        "content_sha256": content_hash,
        "content_type": canonical.get("content_type") or content_type,
        "size": size,
        "filename": destination.name,
        "source_path": _repository_path(destination),
    }


def archive_item(
    session: requests.Session,
    item: DiscoveredItem,
    archive_root: Path,
) -> ArchivedItem:
    """Fetch an item once and atomically persist its bytes and provenance."""

    response, content = fetch_archivable_content(
        session,
        item.url,
        max_bytes=MAX_RESPONSE_BYTES,
    )
    return archive_content(response, content, item, archive_root)


def archive_content(
    response: requests.Response,
    content: bytes,
    item: DiscoveredItem,
    archive_root: Path,
) -> ArchivedItem:
    """Persist already-fetched, bounded ERCOT bytes into the official archive."""

    if not content:
        raise ValueError("Downloaded item was empty")

    content_hash = hashlib.sha256(content).hexdigest()
    extension = _response_extension(item, response)
    source_root = archive_root / safe_path_component(item.source_label)
    source_directory = source_root / _archive_year(item, response)
    existing_destinations = (
        sorted(
            (
                path
                for path in source_root.glob(f"*/{content_hash}.*")
                if path.is_file()
                and re.fullmatch(rf"{content_hash}\.[a-z0-9]{{1,8}}", path.name)
            ),
            key=lambda path: (path.suffix.lower() == ".bin", path.as_posix()),
        )
        if source_root.exists()
        else []
    )
    # A content hash has one archive object per source, even if different ERCOT
    # aliases suggest different filenames, MIME types, or publication years.
    destination = (
        existing_destinations[0]
        if existing_destinations
        else source_directory / f"{content_hash}{extension}"
    )
    promotion_source: Optional[Path] = None
    if (
        destination.suffix.lower() == ".bin"
        and extension in INGESTIBLE_ARCHIVE_EXTENSIONS
    ):
        promotion_source = destination
        destination = destination.with_suffix(extension)
    metadata_path = destination.with_name(f"{destination.name}.metadata.json")
    promotion_metadata_path = (
        promotion_source.with_name(f"{promotion_source.name}.metadata.json")
        if promotion_source is not None
        else None
    )
    destination_existed = destination.exists()
    persisted_source = destination if destination_existed else promotion_source
    persisted = persisted_source.read_bytes() if persisted_source is not None else b""
    persisted_matches = (
        persisted_source is not None
        and hashlib.sha256(persisted).hexdigest() == content_hash
    )
    destination_matches = (
        destination_existed
        and hashlib.sha256(destination.read_bytes()).hexdigest() == content_hash
    )

    # Hash-addressed archives are immutable. Avoid rewriting an unchanged
    # document (and its sidecar), because a fresh mtime/download timestamp
    # would otherwise look like a material ingestion change on every poll.
    if not destination_matches:
        atomic_write_bytes(destination, content)
        persisted = destination.read_bytes()
        if hashlib.sha256(persisted).hexdigest() != content_hash:
            raise OSError(f"Archived content verification failed: {destination}")

    content_type = response.headers.get("Content-Type", "")
    final_url = str(getattr(response, "url", "") or item.url)
    existing_metadata: Dict[str, Any] = {}
    existing_metadata_path = (
        metadata_path
        if metadata_path.exists()
        else promotion_metadata_path
        if promotion_metadata_path is not None and promotion_metadata_path.exists()
        else None
    )
    if existing_metadata_path is not None:
        try:
            loaded_metadata = json.loads(existing_metadata_path.read_text(encoding="utf-8"))
            if isinstance(loaded_metadata, dict):
                existing_metadata = loaded_metadata
        except (OSError, UnicodeError, json.JSONDecodeError):
            existing_metadata = {}
    metadata = _archive_metadata(
        item=item,
        final_url=final_url,
        content_hash=content_hash,
        content_type=content_type,
        destination=destination,
        size=len(persisted),
        existing_metadata=existing_metadata,
    )
    if destination_matches and existing_metadata == metadata:
        return ArchivedItem(
            path=destination,
            metadata_path=metadata_path,
            content=persisted,
            content_hash=content_hash,
            content_type=content_type,
            final_url=final_url,
            archive_status="already_archived",
        )
    atomic_write_bytes(
        metadata_path,
        (json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    if promotion_source is not None and promotion_source != destination:
        # The supported replacement and its sidecar are complete before the
        # ambiguous object is removed, so interruption never loses the bytes.
        if hashlib.sha256(destination.read_bytes()).hexdigest() != content_hash:
            raise OSError(f"Promoted archive verification failed: {destination}")
        if promotion_metadata_path is not None:
            promotion_metadata_path.unlink(missing_ok=True)
        promotion_source.unlink(missing_ok=True)
    return ArchivedItem(
        path=destination,
        metadata_path=metadata_path,
        content=persisted,
        content_hash=content_hash,
        content_type=content_type,
        final_url=final_url,
        archive_status="metadata_updated" if persisted_matches else "archived",
    )


def archive_cached_observation(
    cached: ArchivedItem,
    item: DiscoveredItem,
) -> ArchivedItem:
    """Merge another parent observation without refetching identical URL bytes."""

    existing_metadata = _sidecar_alias_metadata(cached.metadata_path)
    metadata = _archive_metadata(
        item=item,
        final_url=cached.final_url,
        content_hash=cached.content_hash,
        content_type=cached.content_type,
        destination=cached.path,
        size=len(cached.content),
        existing_metadata=existing_metadata,
    )
    archive_status = "already_archived"
    if existing_metadata != metadata:
        atomic_write_bytes(
            cached.metadata_path,
            (
                json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False)
                + "\n"
            ).encode("utf-8"),
        )
        archive_status = "metadata_updated"
    return ArchivedItem(
        path=cached.path,
        metadata_path=cached.metadata_path,
        content=cached.content,
        content_hash=cached.content_hash,
        content_type=cached.content_type,
        final_url=cached.final_url,
        archive_status=archive_status,
    )


def is_allowed_domain(url: str) -> bool:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower().rstrip(".")
    return parsed.scheme.lower() in {"http", "https"} and (
        hostname == "ercot.com" or hostname.endswith(".ercot.com")
    )


def is_public_notices_source(url: str) -> bool:
    parsed = urlparse(url)
    return is_allowed_domain(url) and parsed.path.rstrip("/").lower() == (
        "/services/comm/mkt_notices/notices"
    )


def _newest_date_hint(text: str) -> str:
    candidates = [
        (parsed, value)
        for value in _date_hints(text)
        if (parsed := _parsed_date(value)) is not None
    ]
    return max(candidates)[1] if candidates else ""


def public_notices_snapshot_item(source_url: str, html_text: str) -> DiscoveredItem:
    """Describe the intentionally archived inline PUBLIC NOTICES page."""

    return DiscoveredItem(
        source_label="PUBLIC NOTICES",
        source_url=source_url,
        title="ERCOT Public Notices — current notices page",
        url=source_url,
        item_type="html",
        context="Official ERCOT inline public notices source page.",
        published_hint=_newest_date_hint(html_text),
    )


def archive_public_notices_snapshot(
    response: requests.Response,
    content: bytes,
    item: DiscoveredItem,
    archive_root: Path,
) -> ArchivedItem:
    """Atomically replace the one rolling inline-notices snapshot.

    Prior months are archived through their normal notice links.  The live
    notices page is therefore a latest-only object at a stable path, which
    also lets incremental ingestion retain the prior per-path chunks if
    parsing or embedding the replacement fails.

    Content and metadata are staged and verified before either stable file is
    touched.  Existing files are then moved into a private transaction
    directory and restored together if replacement or verification fails.
    """

    if not content:
        raise ValueError("Downloaded item was empty")
    if not is_public_notices_source(item.url):
        raise ValueError("Latest-only snapshots are restricted to PUBLIC NOTICES")

    content_hash = hashlib.sha256(content).hexdigest()
    snapshot_directory = (
        archive_root / safe_path_component(item.source_label) / "current"
    )
    destination = snapshot_directory / "current.html"
    metadata_path = snapshot_directory / "current.html.metadata.json"
    final_url = str(getattr(response, "url", "") or item.url)
    content_type = response.headers.get("Content-Type", "")

    previous_content = destination.read_bytes() if destination.exists() else b""
    previous_hash = (
        hashlib.sha256(previous_content).hexdigest() if destination.exists() else ""
    )
    existing_metadata = (
        _sidecar_alias_metadata(metadata_path) if metadata_path.exists() else {}
    )
    # A changed rolling snapshot is a new observation time, while a metadata
    # repair for unchanged bytes preserves the original download timestamp.
    metadata_seed = existing_metadata if previous_hash == content_hash else {}
    metadata = _archive_metadata(
        item=item,
        final_url=final_url,
        content_hash=content_hash,
        content_type=content_type,
        destination=destination,
        size=len(content),
        existing_metadata=metadata_seed,
    )
    if (
        previous_hash == content_hash
        and previous_content == content
        and existing_metadata == metadata
    ):
        return ArchivedItem(
            path=destination,
            metadata_path=metadata_path,
            content=previous_content,
            content_hash=content_hash,
            content_type=content_type,
            final_url=final_url,
            archive_status="already_archived",
        )

    snapshot_directory.mkdir(parents=True, exist_ok=True)
    transaction_directory = Path(
        tempfile.mkdtemp(prefix=".current-transaction-", dir=str(snapshot_directory))
    )
    staged_content = transaction_directory / "content.stage"
    staged_metadata = transaction_directory / "metadata.stage"
    backup_content = transaction_directory / "content.backup"
    backup_metadata = transaction_directory / "metadata.backup"
    had_content = destination.exists()
    had_metadata = metadata_path.exists()
    replacement_complete = False
    try:
        atomic_write_bytes(staged_content, content)
        atomic_write_bytes(
            staged_metadata,
            (
                json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False)
                + "\n"
            ).encode("utf-8"),
        )
        if hashlib.sha256(staged_content.read_bytes()).hexdigest() != content_hash:
            raise OSError("Staged PUBLIC NOTICES content verification failed")
        staged_metadata_value = json.loads(staged_metadata.read_text(encoding="utf-8"))
        if staged_metadata_value != metadata:
            raise OSError("Staged PUBLIC NOTICES metadata verification failed")

        if had_content:
            os.replace(destination, backup_content)
        if had_metadata:
            os.replace(metadata_path, backup_metadata)
        os.replace(staged_content, destination)
        os.replace(staged_metadata, metadata_path)

        if hashlib.sha256(destination.read_bytes()).hexdigest() != content_hash:
            raise OSError("Archived PUBLIC NOTICES content verification failed")
        persisted_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if persisted_metadata != metadata:
            raise OSError("Archived PUBLIC NOTICES metadata verification failed")
        replacement_complete = True
    except Exception:
        # Roll the stable pair back as a unit.  Backups are never discarded if
        # restoration itself fails, so the last known-good bytes are retained.
        if backup_content.exists():
            destination.unlink(missing_ok=True)
            os.replace(backup_content, destination)
        elif not had_content:
            destination.unlink(missing_ok=True)
        if backup_metadata.exists():
            metadata_path.unlink(missing_ok=True)
            os.replace(backup_metadata, metadata_path)
        elif not had_metadata:
            metadata_path.unlink(missing_ok=True)
        raise
    finally:
        staged_content.unlink(missing_ok=True)
        staged_metadata.unlink(missing_ok=True)
        if replacement_complete:
            backup_content.unlink(missing_ok=True)
            backup_metadata.unlink(missing_ok=True)
        try:
            transaction_directory.rmdir()
        except OSError:
            # A failed rollback may deliberately leave a private backup rather
            # than deleting the only copy of the prior working snapshot.
            pass

    return ArchivedItem(
        path=destination,
        metadata_path=metadata_path,
        content=content,
        content_hash=content_hash,
        content_type=content_type,
        final_url=final_url,
        archive_status="metadata_updated" if previous_hash == content_hash else "archived",
    )


def is_interesting_link(source_url: str, candidate_url: str, text: str) -> bool:
    if not candidate_url:
        return False
    lowered = candidate_url.lower()
    if any(lowered.startswith(prefix) for prefix in SKIP_LINK_PREFIXES):
        return False
    if not is_allowed_domain(candidate_url):
        return False
    if candidate_url.rstrip("/") == source_url.rstrip("/"):
        return False

    parsed = urlparse(candidate_url)
    path = parsed.path.lower()
    text_l = text.lower()
    extension = path_extension(candidate_url)

    if any(part in path for part in SKIP_PATH_PARTS):
        return False
    if extension in DOCUMENT_EXTENSIONS:
        return True
    if "/committees/" in path and any(hint in text_l for hint in MEETING_HINTS):
        return True
    if "/committees/" in path and re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", text_l):
        return True
    if "/mktrules/issues/" in path and re.search(
        r"\b(?:nprr|pgrr|nogrr|obdrr|scr|rrgrr|vcmrr)\s*[-_ ]?\s*\d+\b",
        text_l,
    ):
        return True
    if re.fullmatch(
        r"/mktrules/issues/reports/(?:nprr|pgrr|nogrr|obdrr|scr)"
        r"(?:/(?:pending|approved|withdrawn|rejected))?/?",
        path,
    ):
        return True
    if any(
        section in path
        for section in (
            "/services/comm/mkt_notices/",
            "/mktrules/nprotocols/",
            "/mktrules/guides/planning/",
            "/mktrules/guides/noperating/",
        )
    ):
        return True
    if "data-product-details" in path or "download" in path:
        return True
    return False


def extract_anchor_candidates(source_label: str, source_url: str, html_text: str) -> List[DiscoveredItem]:
    anchor_pattern = re.compile(r"<a\b[^>]*href=(['\"])(.*?)\1[^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)
    seen: set[str] = set()
    results: List[DiscoveredItem] = []
    lowered_html = html_text.lower()

    for match in anchor_pattern.finditer(html_text):
        href = clean_text(match.group(2))
        anchor_html = match.group(3)
        text = clean_text(re.sub(r"<[^>]+>", " ", anchor_html))
        absolute_url = urljoin(source_url, href)

        if not text or not is_interesting_link(source_url, absolute_url, text):
            continue

        key = absolute_url
        if key in seen:
            continue
        seen.add(key)

        extension = path_extension(absolute_url)
        item_type = extension.lstrip(".") if extension else "page"
        row_start = lowered_html.rfind("<tr", 0, match.start())
        row_end = lowered_html.find("</tr>", match.end())
        if row_start >= 0 and row_end >= 0 and row_end + 5 - row_start <= 10_000:
            context_start, context_end = row_start, row_end + 5
            row_html = html_text[context_start:context_end]
        else:
            context_start = max(0, match.start() - 600)
            context_end = min(len(html_text), match.end() + 600)
            row_html = ""
        context = clean_text(re.sub(r"<[^>]+>", " ", html_text[context_start:context_end]))
        published_hint = extract_published_hint(context)
        effective_date = ""
        state_tag = ""
        if re.fullmatch(
            r"/mktrules/issues/reports/(?:nprr|pgrr|nogrr|obdrr|scr)/?",
            urlparse(source_url).path.lower(),
        ):
            state_tag, published_hint, effective_date = _report_row_lifecycle(
                row_html,
                context,
            )
        results.append(
            DiscoveredItem(
                source_label=source_label,
                source_url=source_url,
                title=text,
                url=absolute_url,
                item_type=item_type or "page",
                context=context,
                published_hint=published_hint,
                effective_date=effective_date,
                state_tag=state_tag,
            )
        )

    return rank_items(results)


def extract_context_window(html_text: str, anchor_text: str, radius: int = 220) -> str:
    raw_text = clean_text(re.sub(r"<[^>]+>", " ", html_text))
    idx = raw_text.lower().find(anchor_text.lower())
    if idx < 0:
        return ""
    start = max(0, idx - radius)
    end = min(len(raw_text), idx + len(anchor_text) + radius)
    return raw_text[start:end].strip()


def extract_published_hint(text: str) -> str:
    hints = _date_hints(text)
    return hints[0] if hints else ""


def _date_hints(text: str) -> List[str]:
    if not text:
        return []
    pattern = re.compile(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{1,2}, \d{4}\b"
        r"|\b\d{4}-\d{2}-\d{2}\b"
        r"|\b(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])/(?:20\d{2})\b",
        re.IGNORECASE,
    )
    results: List[str] = []
    for match in pattern.finditer(text):
        value = match.group(0)
        numeric = re.fullmatch(
            r"(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/(20\d{2})",
            value,
        )
        if numeric:
            month, day, year = numeric.groups()
            value = f"{year}-{int(month):02d}-{int(day):02d}"
        if value not in results:
            results.append(value)
    return results


def _parsed_date(value: str) -> Optional[datetime]:
    cleaned = str(value or "").strip().replace("Sept ", "Sep ")
    for format_string in (
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%b. %d, %Y",
    ):
        try:
            return datetime.strptime(cleaned, format_string)
        except ValueError:
            continue
    return None


def _report_row_lifecycle(row_html: str, context: str) -> Tuple[str, str, str]:
    """Prefer explicit report cells over lifecycle words buried in descriptions."""

    cell_matches = re.findall(
        r"<(?:td|th)\b[^>]*>(.*?)</(?:td|th)>",
        row_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cells = [clean_text(re.sub(r"<[^>]+>", " ", value)) for value in cell_matches]
    statuses = ("pending", "approved", "effective", "clean", "withdrawn", "rejected")
    status_candidates: List[Tuple[int, int, str]] = []
    published_candidates: List[Tuple[int, datetime, int, str]] = []
    effective_candidates: List[Tuple[datetime, int, str]] = []
    for index, cell in enumerate(cells):
        lowered = cell.lower()
        for status in statuses:
            if re.fullmatch(rf"(?:current\s+)?(?:status\s*[:\-]?\s*)?{status}", lowered):
                status_candidates.append((3, index, status))
            elif re.search(rf"\bstatus\s*[:\-]\s*{status}\b", lowered):
                status_candidates.append((2, index, status))
        hints = _date_hints(cell)
        for hint in hints:
            parsed = _parsed_date(hint)
            if parsed is None:
                continue
            if "effective" in lowered:
                effective_candidates.append((parsed, index, hint))
                continue
            exact_date = clean_text(cell).lower() == hint.lower()
            labelled = bool(re.search(r"\b(?:posted|submitted|updated|date)\b", lowered))
            score = 3 if exact_date else 2 if labelled else 1
            published_candidates.append((score, parsed, index, hint))

    if status_candidates:
        state_tag = max(status_candidates)[2]
    else:
        fallback_statuses = re.findall(
            r"\b(pending|approved|effective|clean|withdrawn|rejected)\b",
            context,
            flags=re.IGNORECASE,
        )
        state_tag = fallback_statuses[-1].lower() if fallback_statuses else "listed"
    published_hint = max(published_candidates)[3] if published_candidates else extract_published_hint(context)
    effective_date = max(effective_candidates)[2] if effective_candidates else ""
    return state_tag, published_hint, effective_date


def rank_items(items: Sequence[DiscoveredItem]) -> List[DiscoveredItem]:
    def score(item: DiscoveredItem) -> Tuple[int, int, str, str]:
        title_l = item.title.lower()
        priority = 0
        if any(token in title_l for token in ("agenda", "minutes", "presentation", "packet", "overview", "report")):
            priority += 3
        if item.item_type in {"pdf", "docx", "xlsx", "csv"}:
            priority += 2
        number_match = re.search(
            r"\b(?:nprr|pgrr|nogrr|obdrr|scr|rrgrr|vcmrr)\s*[-_ ]?\s*\d+\b",
            title_l,
        )
        if number_match:
            priority += 2
        if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", title_l):
            priority += 1
        numeric_match = re.search(r"(\d{1,6})", number_match.group(0)) if number_match else None
        sequence = int(numeric_match.group(1)) if numeric_match else -1
        return (priority, sequence, item.published_hint, item.title)

    # State comparison must see the full candidate set. Truncating here made
    # lower-ranked documents permanently invisible on every later run.
    return sorted(items, key=score, reverse=True)


def summarize_text(text: str, max_chars: int = MAX_SUMMARY_CHARS) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return "No readable text could be extracted."

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    if len(sentences) == 1:
        return cleaned[:max_chars]

    summary_parts: List[str] = []
    total = 0
    for sentence in sentences:
        if len(sentence) < 40:
            continue
        next_len = total + len(sentence) + (1 if summary_parts else 0)
        if next_len > max_chars:
            break
        summary_parts.append(sentence)
        total = next_len
        if len(summary_parts) >= 4:
            break

    return " ".join(summary_parts) if summary_parts else cleaned[:max_chars]


def extract_html_summary(content: bytes, fallback_title: str = "") -> str:
    try:
        text = content.decode("utf-8", errors="ignore")
    except Exception:
        return f"{fallback_title} is an HTML page, but text extraction failed."

    body = re.sub(r"(?is)<script.*?</script>", " ", text)
    body = re.sub(r"(?is)<style.*?</style>", " ", body)
    body = re.sub(r"<[^>]+>", " ", body)
    return summarize_text(body)


def extract_json_summary(content: bytes) -> str:
    try:
        payload = json.loads(content.decode("utf-8", errors="ignore"))
    except Exception:
        return "JSON file detected, but it could not be parsed."

    if isinstance(payload, dict):
        parts = []
        for key, value in list(payload.items())[:12]:
            parts.append(f"{key}: {value}")
        return summarize_text(". ".join(parts))
    if isinstance(payload, list):
        preview = ". ".join(str(item) for item in payload[:8])
        return summarize_text(preview)
    return summarize_text(str(payload))


def extract_csv_summary(content: bytes) -> str:
    try:
        sample = content.decode("utf-8", errors="ignore")
        rows = list(csv.reader(io.StringIO(sample)))
    except Exception:
        return "CSV file detected, but it could not be parsed."

    if not rows:
        return "CSV file is empty."

    header = rows[0]
    row_count = max(0, len(rows) - 1)
    preview_rows = rows[1:4]
    pieces = [f"CSV with {row_count} data rows.", f"Columns: {', '.join(header[:12])}."]
    for row in preview_rows:
        pieces.append(", ".join(row[:8]))
    return summarize_text(" ".join(pieces))


def _read_bounded_zip_member(
    archive: zipfile.ZipFile,
    member_name: str,
    *,
    max_bytes: Optional[int] = None,
) -> bytes:
    limit = max(1, int(MAX_RESPONSE_BYTES if max_bytes is None else max_bytes))
    info = archive.getinfo(member_name)
    if info.file_size > limit:
        raise ValueError(
            f"ZIP member exceeds ERCOT_LINK_MAX_RESPONSE_BYTES "
            f"({info.file_size} > {limit})"
        )
    with archive.open(info) as handle:
        content = handle.read(limit + 1)
    if len(content) > limit:
        raise ValueError(
            f"ZIP member exceeds ERCOT_LINK_MAX_RESPONSE_BYTES "
            f"({len(content)} > {limit})"
        )
    return content


def extract_docx_summary(content: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            xml = _read_bounded_zip_member(zf, "word/document.xml")
    except ValueError as exc:
        return f"DOCX file detected, but text extraction failed: {exc}"
    except Exception:
        return "DOCX file detected, but text extraction failed."

    try:
        root = ET.fromstring(xml)
        paragraphs = []
        for paragraph in root.findall(".//w:p", NS):
            texts = [node.text or "" for node in paragraph.findall(".//w:t", NS)]
            line = clean_text("".join(texts))
            if line:
                paragraphs.append(line)
            if len(paragraphs) >= 8:
                break
        return summarize_text(" ".join(paragraphs))
    except Exception:
        return "DOCX file detected, but text extraction failed."


def extract_xlsx_summary(content: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            workbook_xml = _read_bounded_zip_member(zf, "xl/workbook.xml")
            root = ET.fromstring(workbook_xml)
            names = [
                sheet.attrib.get("name", "")
                for sheet in root.findall(".//x:sheets/x:sheet", NS)
                if sheet.attrib.get("name")
            ]
    except ValueError as exc:
        return f"XLSX file detected, but it could not be inspected: {exc}"
    except Exception:
        return "XLSX file detected, but it could not be inspected."

    if not names:
        return "XLSX file detected."
    return f"Spreadsheet detected. Sheet names: {', '.join(names[:10])}."


def extract_pdf_summary(content: bytes) -> str:
    if PdfReader is None:
        return "PDF detected. Install `pypdf` to extract and summarize PDF text."

    try:
        reader = PdfReader(io.BytesIO(content))
        texts = []
        for page in reader.pages[:3]:
            page_text = clean_text(page.extract_text() or "")
            if page_text:
                texts.append(page_text)
        return summarize_text(" ".join(texts)) if texts else "PDF detected, but no text was extracted."
    except Exception:
        return "PDF detected, but text extraction failed."


def summarize_binary_content(url: str, content: bytes, content_type: str) -> str:
    extension = path_extension(url)
    content_type = (content_type or "").lower()

    if extension in {".html", ".htm"} or "text/html" in content_type:
        return extract_html_summary(content)
    if extension in {".json"} or "application/json" in content_type:
        return extract_json_summary(content)
    if extension in {".csv"} or "text/csv" in content_type:
        return extract_csv_summary(content)
    if extension in {".txt", ".md", ".xml"} or content_type.startswith("text/"):
        return summarize_text(content.decode("utf-8", errors="ignore"))
    if extension == ".docx":
        return extract_docx_summary(content)
    if extension == ".xlsx":
        return extract_xlsx_summary(content)
    if extension == ".pdf":
        return extract_pdf_summary(content)

    return f"New {extension.lstrip('.') or 'file'} detected, but no extractor is configured for this format."


def summarize_item(session: requests.Session, item: DiscoveredItem) -> str:
    try:
        response = fetch(session, item.url)
    except Exception as exc:
        return f"Unable to open linked item: {exc}"

    try:
        summary = summarize_binary_content(
            item.url,
            response.content,
            response.headers.get("Content-Type", ""),
        )
    finally:
            _close_response(response)
    if item.context:
        context_summary = summarize_text(item.context, max_chars=280)
        if context_summary and context_summary != "No readable text could be extracted.":
            return f"{summary}\nContext: {context_summary}"
    return summary


def _decode_state_entry(entry: str) -> Tuple[str, Optional[str]]:
    """Return the historical fingerprint and optional last-seen content hash."""

    fingerprint, separator, content_hash = entry.rpartition(STATE_HASH_SEPARATOR)
    if separator and re.fullmatch(r"[a-f0-9]{64}", content_hash):
        return fingerprint, content_hash
    return entry, None


def _is_cursor_entry(entry: str) -> bool:
    return str(entry).startswith(STATE_CURSOR_PREFIX)


def _cursor_entry_parts(entry: str) -> Optional[Tuple[str, str]]:
    if not _is_cursor_entry(entry):
        return None
    remainder = str(entry)[len(STATE_CURSOR_PREFIX) :]
    kind, separator, value = remainder.rpartition("|")
    if not separator or not kind or not re.fullmatch(r"[a-f0-9]{64}", value):
        return None
    return kind, value


def _cursor_key(item: DiscoveredItem) -> str:
    identity = f"{item.source_label}|{item.source_url}|{item.url}|{item.state_tag}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _state_cursor(entries: Sequence[str], kind: str) -> Optional[str]:
    prefix = f"{STATE_CURSOR_PREFIX}{kind}|"
    for entry in entries:
        if str(entry).startswith(prefix):
            value = str(entry)[len(prefix) :]
            return value if re.fullmatch(r"[a-f0-9]{64}", value) else None
    return None


def _set_state_cursor(entries: Sequence[str], kind: str, item: DiscoveredItem) -> List[str]:
    prefix = f"{STATE_CURSOR_PREFIX}{kind}|"
    retained = [str(entry) for entry in entries if not str(entry).startswith(prefix)]
    retained.append(f"{prefix}{_cursor_key(item)}")
    return retained


def _rotate_after_cursor(
    items: Sequence[DiscoveredItem],
    cursor: Optional[str],
) -> List[DiscoveredItem]:
    values = list(items)
    if not values or not cursor:
        return values
    for index, item in enumerate(values):
        if _cursor_key(item) == cursor:
            start = index + 1
            return [*values[start:], *values[:start]]
    return values


def _document_state_entries(entries: Sequence[str]) -> List[str]:
    return [str(entry) for entry in entries if not _is_cursor_entry(str(entry))]


def _bounded_cursor_entries(entries: Sequence[str], nested_limit: int) -> List[str]:
    latest: Dict[str, str] = {}
    order: List[str] = []
    for entry in entries:
        parts = _cursor_entry_parts(str(entry))
        if parts is None:
            continue
        kind, value = parts
        latest[kind] = f"{STATE_CURSOR_PREFIX}{kind}|{value}"
        if kind in order:
            order.remove(kind)
        order.append(kind)
    fixed_kinds = [
        kind
        for kind in ("known", "unseen", "report", "nested:groups")
        if kind in latest
    ]
    nested_kinds = [
        kind
        for kind in order
        if kind.startswith("nested:") and kind != "nested:groups"
    ]
    nested_kinds = (
        nested_kinds[-nested_limit:] if nested_limit > 0 else []
    )
    return [latest[kind] for kind in [*fixed_kinds, *nested_kinds]]


def _url_fingerprint(item: DiscoveredItem) -> str:
    return f"{item.source_label}|{item.url}"


def _fingerprint_belongs_to_url(fingerprint: str, url_fingerprint: str) -> bool:
    return fingerprint == url_fingerprint or fingerprint.startswith(f"{url_fingerprint}|")


def _state_fingerprints(entries: Sequence[str]) -> set[str]:
    return {
        _decode_state_entry(str(entry))[0]
        for entry in entries
        if not _is_cursor_entry(str(entry))
    }


def _last_seen_hash(
    entries: Sequence[str],
    item: DiscoveredItem,
) -> Optional[str]:
    url_fingerprint = _url_fingerprint(item)
    for entry in entries:
        if _is_cursor_entry(str(entry)):
            continue
        fingerprint, content_hash = _decode_state_entry(str(entry))
        if _fingerprint_belongs_to_url(fingerprint, url_fingerprint):
            return content_hash
    return None


def _url_was_seen(entries: Sequence[str], item: DiscoveredItem) -> bool:
    url_fingerprint = _url_fingerprint(item)
    return any(
        _fingerprint_belongs_to_url(_decode_state_entry(str(entry))[0], url_fingerprint)
        for entry in entries
        if not _is_cursor_entry(str(entry))
    )


def _promote_state_entry(
    entries: Sequence[str],
    item: DiscoveredItem,
    content_hash: str,
) -> List[str]:
    """Promote a successful URL to the front and replace its prior state/hash."""

    url_fingerprint = _url_fingerprint(item)
    retained = [
        str(entry)
        for entry in entries
        if not _fingerprint_belongs_to_url(
            _decode_state_entry(str(entry))[0],
            url_fingerprint,
        )
    ]
    return [f"{item.fingerprint}{STATE_HASH_SEPARATOR}{content_hash}", *retained]


def _order_state_for_candidates(
    entries: Sequence[str],
    candidates: Sequence[DiscoveredItem],
) -> List[str]:
    """Keep newest-ranked visible candidates before older/nested state entries."""

    remaining = _document_state_entries(entries)
    ordered: List[str] = []
    for item in candidates:
        url_fingerprint = _url_fingerprint(item)
        for index, entry in enumerate(remaining):
            fingerprint, _ = _decode_state_entry(entry)
            if _fingerprint_belongs_to_url(fingerprint, url_fingerprint):
                ordered.append(entry)
                remaining.pop(index)
                break
    ordered.extend(remaining)
    return ordered


def _archive_url_hashes(source_root: Path) -> Dict[str, str]:
    """Return the most recently archived hash for each observed ERCOT URL."""

    observations: Dict[str, Tuple[int, str]] = {}
    if not source_root.exists():
        return {}
    metadata_limit = min(max(1, MAX_RESPONSE_BYTES), 2 * 1024 * 1024)
    for metadata_path in source_root.glob("*/*.metadata.json"):
        try:
            if metadata_path.stat().st_size > metadata_limit:
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                continue
            content_hash = str(metadata.get("content_sha256") or "")
            if not re.fullmatch(r"[a-f0-9]{64}", content_hash):
                continue
            raw_aliases = metadata.get("url_aliases", [])
            urls = (
                {
                    str(value)
                    for value in raw_aliases
                    if isinstance(value, str) and value
                }
                if isinstance(raw_aliases, list)
                else set()
            )
            for field in ("original_url", "final_url"):
                value = metadata.get(field)
                if isinstance(value, str) and value:
                    urls.add(value)
            provenance = metadata.get("provenance", [])
            if isinstance(provenance, list):
                for observation in provenance:
                    if not isinstance(observation, dict):
                        continue
                    for field in ("original_url", "final_url"):
                        value = observation.get(field)
                        if isinstance(value, str) and value:
                            urls.add(value)
            modified = metadata_path.stat().st_mtime_ns
            for url in urls:
                prior = observations.get(url)
                if prior is None or modified >= prior[0]:
                    observations[url] = (modified, content_hash)
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
    return {url: value[1] for url, value in observations.items()}


def _sidecar_alias_metadata(metadata_path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}


def _append_or_coalesce_change(
    changes: List[Dict[str, Any]],
    emitted_by_hash: Dict[Tuple[str, str], int],
    event: Dict[str, Any],
    archived: ArchivedItem,
    item: DiscoveredItem,
) -> None:
    key = (safe_path_component(item.source_label), archived.content_hash)
    metadata = _sidecar_alias_metadata(archived.metadata_path)
    raw_aliases = metadata.get("url_aliases", [])
    aliases = sorted(
        (
            {
                str(value)
                for value in raw_aliases
                if isinstance(value, str) and value
            }
            if isinstance(raw_aliases, list)
            else set()
        )
        | {item.url, archived.final_url}
    )
    raw_source_pages = metadata.get("source_page_urls", [])
    source_pages = sorted(
        (
            {
                str(value)
                for value in raw_source_pages
                if isinstance(value, str) and value
            }
            if isinstance(raw_source_pages, list)
            else set()
        )
        | {item.source_url}
    )
    event.update(
        {
            "url_aliases": aliases,
            "source_page_urls": source_pages,
            "alias_count": len(aliases),
        }
    )
    existing_index = emitted_by_hash.get(key)
    if existing_index is None:
        emitted_by_hash[key] = len(changes)
        changes.append(event)
        return

    existing = changes[existing_index]
    existing["url_aliases"] = sorted(
        set(existing.get("url_aliases", [])) | set(aliases)
    )
    existing["source_page_urls"] = sorted(
        set(existing.get("source_page_urls", [])) | set(source_pages)
    )
    existing["alias_count"] = len(existing["url_aliases"])
    existing["content_changed_since_last_seen"] = bool(
        existing.get("content_changed_since_last_seen")
        or event.get("content_changed_since_last_seen")
    )
    if event.get("status") == "updated":
        existing["status"] = "updated"
    if existing.get("download_status") != "archived" and event.get("download_status") == "archived":
        existing["download_status"] = "archived"
    existing_path = Path(str(existing.get("downloaded_path") or ""))
    event_path = Path(str(event.get("downloaded_path") or ""))
    if event_path and (
        not existing_path.exists()
        or (
            existing_path.suffix.lower() == ".bin"
            and event_path.suffix.lower() in INGESTIBLE_ARCHIVE_EXTENSIONS
        )
    ):
        for field in (
            "downloaded_path",
            "repository_path",
            "metadata_path",
            "content_type",
            "final_url",
        ):
            existing[field] = event.get(field)
    canonical_url = str(metadata.get("original_url") or "")
    if canonical_url:
        existing["url"] = canonical_url
    canonical_title = str(metadata.get("title") or "")
    if canonical_title:
        existing["title"] = canonical_title


def _nested_cursor_kind(parent: DiscoveredItem) -> str:
    parent_identity = f"{parent.source_label}|{parent.url}"
    digest = hashlib.sha256(parent_identity.encode("utf-8")).hexdigest()[:24]
    return f"nested:{digest}"


def _schedule_nested_items(
    groups: Sequence[Tuple[DiscoveredItem, Sequence[DiscoveredItem]]],
    state_entries: Sequence[str],
    budget: int,
) -> List[Tuple[DiscoveredItem, str, DiscoveredItem]]:
    """Round-robin bounded nested work, rotating independently per parent."""

    group_by_parent = {_cursor_key(parent): (parent, candidates) for parent, candidates in groups}
    rotated_parents = _rotate_after_cursor(
        [parent for parent, _ in groups],
        _state_cursor(state_entries, "nested:groups"),
    )
    rotated_groups: List[Tuple[DiscoveredItem, str, List[DiscoveredItem]]] = []
    for rotated_parent in rotated_parents:
        parent, candidates = group_by_parent[_cursor_key(rotated_parent)]
        cursor_kind = _nested_cursor_kind(parent)
        unique: Dict[str, DiscoveredItem] = {}
        for candidate in candidates:
            unique[_cursor_key(candidate)] = candidate
        rotated = _rotate_after_cursor(
            list(unique.values()),
            _state_cursor(state_entries, cursor_kind),
        )
        if rotated:
            rotated_groups.append((parent, cursor_kind, rotated))

    scheduled: List[Tuple[DiscoveredItem, str, DiscoveredItem]] = []
    limit = max(0, budget)
    while rotated_groups and len(scheduled) < limit:
        remaining_groups: List[Tuple[DiscoveredItem, str, List[DiscoveredItem]]] = []
        for parent, cursor_kind, candidates in rotated_groups:
            if len(scheduled) >= limit:
                remaining_groups.append((parent, cursor_kind, candidates))
                continue
            scheduled.append((candidates.pop(0), cursor_kind, parent))
            if candidates:
                remaining_groups.append((parent, cursor_kind, candidates))
        rotated_groups = remaining_groups
    return scheduled


def scan_sources(
    links: Sequence[SourceLink],
    state: Dict[str, List[str]],
    archive_root: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    headers = {"User-Agent": USER_AGENT}
    session = requests.Session()
    session.headers.update(headers)
    destination_root = (archive_root or official_document_dir()).resolve()

    changes: List[Dict[str, Any]] = []
    emitted_by_hash: Dict[Tuple[str, str], int] = {}
    new_state: Dict[str, List[str]] = {
        str(key): [str(entry) for entry in value if isinstance(entry, str)]
        for key, value in state.items()
        if isinstance(value, list)
    }

    for source in links:
        recent_entries = [str(value) for value in new_state.get(source.label, [])]
        source_snapshot: Optional[DiscoveredItem] = None
        prefetched_content: Dict[
            Tuple[str, str], Tuple[requests.Response, bytes]
        ] = {}
        try:
            source_response, source_content, source_text = fetch_source_payload(
                session,
                source.url,
                max_bytes=MAX_RESPONSE_BYTES,
            )
            if is_public_notices_source(source.url):
                source_snapshot = public_notices_snapshot_item(source.url, source_text)
                prefetched_content[
                    (
                        safe_path_component(source_snapshot.source_label),
                        source_snapshot.url,
                    )
                ] = (source_response, source_content)
            all_candidates = extract_anchor_candidates(source.label, source.url, source_text)
            is_report_source = bool(
                re.fullmatch(
                r"/mktrules/issues/reports/(?:nprr|pgrr|nogrr|obdrr|scr)/?",
                urlparse(source.url).path.lower(),
                )
            )
            candidates = all_candidates
            if is_report_source and all_candidates:
                # Keep source-page work bounded while rotating through the full
                # report history. Always include a newest prefix so a newly
                # posted high-numbered revision is handled immediately.
                report_window = max(1, REPORT_CANDIDATE_WINDOW)
                rotated = _rotate_after_cursor(
                    all_candidates,
                    _state_cursor(recent_entries, "report"),
                )
                rotating_window = rotated[:report_window]
                newest_prefix = all_candidates[
                    : min(report_window, max(1, MAX_ITEMS_PER_SOURCE))
                ]
                selected = {
                    item.fingerprint: item for item in [*newest_prefix, *rotating_window]
                }
                candidates = list(selected.values())
                if rotating_window:
                    recent_entries = _set_state_cursor(
                        recent_entries,
                        "report",
                        rotating_window[-1],
                    )
        except Exception as exc:
            changes.append(
                {
                    "source": source.label,
                    "source_url": source.url,
                    "title": "Source fetch failed",
                    "url": source.url,
                    "summary": f"Unable to scan source page: {exc}",
                    "published_hint": "",
                    "item_type": "source",
                    "status": "error",
                }
            )
            continue

        archive_source_root = destination_root / safe_path_component(source.label)
        archive_hashes = _archive_url_hashes(archive_source_root)
        # Recheck already-known URLs so stable links with replaced bytes are
        # detected. Archive provenance is an exact secondary seen ledger when
        # bounded recent state evicts an older URL.
        known_rotation = _rotate_after_cursor(
            candidates,
            _state_cursor(recent_entries, "known"),
        )
        unseen_rotation = _rotate_after_cursor(
            candidates,
            _state_cursor(recent_entries, "unseen"),
        )
        known_candidates_all = [
            item
            for item in known_rotation
            if _url_was_seen(recent_entries, item) or item.url in archive_hashes
        ]
        unseen_candidates = [
            item
            for item in unseen_rotation
            if not _url_was_seen(recent_entries, item) and item.url not in archive_hashes
        ]
        known_candidates = known_candidates_all[
            : max(0, MAX_KNOWN_RECHECKS_PER_SOURCE)
        ]
        attempt_limit = max(
            max(0, MAX_ITEMS_PER_SOURCE),
            max(0, MAX_UNSEEN_ATTEMPTS_PER_SOURCE),
        )
        queued: List[Tuple[DiscoveredItem, int, bool, Optional[str]]] = [
            (item, 0, False, "known") for item in known_candidates
        ]
        if source_snapshot is not None:
            # The bytes were already fetched for discovery, so snapshotting the
            # one intentional inline-notice source does not consume link caps.
            queued.insert(0, (source_snapshot, -1, False, None))
        queued.extend(
            (item, 0, True, "unseen")
            for item in unseen_candidates[:attempt_limit]
        )
        processed: set[str] = set()
        successful_unseen = 0
        nested_groups: List[Tuple[DiscoveredItem, Sequence[DiscoveredItem]]] = []
        nested_scheduled = False
        archive_cache: Dict[Tuple[str, str], ArchivedItem] = {}

        while queued or (nested_groups and not nested_scheduled):
            if not queued:
                scheduled_nested = _schedule_nested_items(
                    nested_groups,
                    recent_entries,
                    MAX_NESTED_ITEMS_PER_SOURCE,
                )
                nested_scheduled = True
                if scheduled_nested:
                    recent_entries = _set_state_cursor(
                        recent_entries,
                        "nested:groups",
                        scheduled_nested[-1][2],
                    )
                queued.extend(
                    (candidate, 1, False, cursor_kind)
                    for candidate, cursor_kind, _ in scheduled_nested
                )
                if not queued:
                    break
            item, depth, counts_as_unseen, cursor_kind = queued.pop(0)
            observation_key = _cursor_key(item)
            if observation_key in processed:
                continue
            if counts_as_unseen and successful_unseen >= max(0, MAX_ITEMS_PER_SOURCE):
                continue
            processed.add(observation_key)
            if cursor_kind:
                # Persist the attempt (including failures), so a permanently
                # broken prefix cannot monopolize every scheduled run.
                recent_entries = _set_state_cursor(
                    recent_entries,
                    cursor_kind,
                    item,
                )
            known_fingerprints = _state_fingerprints(recent_entries)
            was_known = (
                item.fingerprint in known_fingerprints
                or item.url in archive_hashes
            )
            url_was_known = (
                _url_was_seen(recent_entries, item)
                or item.url in archive_hashes
            )
            previous_content_hash = (
                _last_seen_hash(recent_entries, item)
                or archive_hashes.get(item.url)
            )
            try:
                cache_key = (safe_path_component(item.source_label), item.url)
                cached = archive_cache.get(cache_key)
                if cached is None:
                    prefetched = prefetched_content.get(cache_key)
                    if prefetched is None:
                        archived = archive_item(session, item, destination_root)
                    elif is_public_notices_source(item.url) and depth == -1:
                        archived = archive_public_notices_snapshot(
                            prefetched[0],
                            prefetched[1],
                            item,
                            destination_root,
                        )
                    else:
                        archived = archive_content(
                            prefetched[0],
                            prefetched[1],
                            item,
                            destination_root,
                        )
                    archive_cache[cache_key] = archived
                else:
                    archived = archive_cached_observation(cached, item)
                    archive_cache[cache_key] = archived
            except Exception as exc:
                changes.append(
                    {
                        "source": item.source_label,
                        "source_url": item.source_url,
                        "title": item.title,
                        "url": item.url,
                        "summary": f"Unable to archive linked item: {exc}",
                        "published_hint": item.published_hint,
                        "item_type": item.item_type,
                        "status": "error",
                        "download_status": "failed",
                        "downloaded_path": None,
                        "metadata_path": None,
                        "content_sha256": None,
                    }
                )
                # A failed new download remains unseen so a later run retries
                # it. Known URLs are fetched on every run so content replaced
                # at a stable URL is detected by SHA-256 rather than URL alone.
                continue

            content_type = archived.content_type.lower()
            if depth == 0 and (
                "text/html" in content_type
                or archived.path.suffix.lower() in {".html", ".htm"}
            ):
                try:
                    nested = extract_anchor_candidates(
                        item.source_label,
                        archived.final_url,
                        archived.content.decode("utf-8", errors="ignore"),
                    )
                except Exception:
                    nested = []
                for candidate in nested:
                    if item.state_tag and not candidate.state_tag:
                        candidate.state_tag = item.state_tag
                    if not candidate.published_hint:
                        candidate.published_hint = item.published_hint
                    if not candidate.effective_date:
                        candidate.effective_date = item.effective_date
                nested_groups.append((item, nested))

            # State advances only after both official bytes and provenance are
            # durably written and the persisted bytes pass hash verification.
            # Encoding the last-seen hash in the existing list-shaped state
            # makes A -> B -> A at a stable URL observable even when A's
            # hash-addressed archive already exists.
            recent_entries = _promote_state_entry(
                recent_entries,
                item,
                archived.content_hash,
            )
            if counts_as_unseen:
                successful_unseen += 1
            archive_hashes[item.url] = archived.content_hash
            archive_hashes[archived.final_url] = archived.content_hash
            alias_metadata = _sidecar_alias_metadata(archived.metadata_path)
            raw_aliases = alias_metadata.get("url_aliases", [])
            for alias in raw_aliases if isinstance(raw_aliases, list) else []:
                if isinstance(alias, str) and alias:
                    archive_hashes[alias] = archived.content_hash
            content_changed_since_last_seen = bool(
                previous_content_hash
                and previous_content_hash != archived.content_hash
            )

            if (
                was_known
                and archived.archive_status == "already_archived"
                and not content_changed_since_last_seen
            ):
                continue

            try:
                item_summary = summarize_binary_content(
                    str(archived.path),
                    archived.content,
                    archived.content_type,
                )
            except Exception as exc:
                item_summary = f"Official document archived, but summary extraction failed: {exc}"
            if item.context:
                context_summary = summarize_text(item.context, max_chars=280)
                if context_summary and context_summary != "No readable text could be extracted.":
                    item_summary = f"{item_summary}\nContext: {context_summary}"

            _append_or_coalesce_change(
                changes,
                emitted_by_hash,
                {
                    "source": item.source_label,
                    "source_url": item.source_url,
                    "title": item.title,
                    "url": item.url,
                    "summary": item_summary,
                    "published_hint": item.published_hint,
                    "item_type": item.item_type,
                    "status": "updated" if url_was_known else "new",
                    "download_status": archived.archive_status,
                    "downloaded_path": str(archived.path),
                    "repository_path": _repository_path(archived.path),
                    "metadata_path": str(archived.metadata_path),
                    "content_sha256": archived.content_hash,
                    "downloaded_bytes": len(archived.content),
                    "content_type": archived.content_type,
                    "final_url": archived.final_url,
                    "content_changed_since_last_seen": content_changed_since_last_seen,
                },
                archived,
                item,
            )

        # Bound document records, while retaining tiny cursor markers outside
        # that budget so known/unseen/report windows rotate across runs.
        state_limit = max(1, MAX_STATE_ITEMS_PER_SOURCE)
        ordered_documents = _order_state_for_candidates(
            recent_entries,
            [*([source_snapshot] if source_snapshot is not None else []), *all_candidates],
        )[:state_limit]
        cursor_entries = _bounded_cursor_entries(
            recent_entries,
            nested_limit=max(state_limit, REPORT_CANDIDATE_WINDOW),
        )
        new_state[source.label] = [*ordered_documents, *cursor_entries]

    return changes, new_state


def _reported_changes(
    changes: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """Bound command output while keeping successful updates ahead of scan errors."""

    limit = max(0, MAX_OUTPUT_ITEMS)
    ordered = [
        *[item for item in changes if item.get("status") in {"new", "updated"}],
        *[item for item in changes if item.get("status") not in {"new", "updated"}],
    ]
    reported = ordered[:limit]
    return reported, max(0, len(ordered) - len(reported))


def format_telegram_message(
    changes: Sequence[Dict[str, Any]],
    *,
    omitted_count: int = 0,
) -> str:
    if not changes:
        message = "ERCOT monitor: no new items found."
        if omitted_count:
            message = f"ERCOT monitor: {omitted_count} additional items omitted from this digest."
        return message[: max(0, MAX_TELEGRAM_CHARS)]

    lines = ["<b>ERCOT Monitor</b>", ""]
    for item in changes:
        status = item.get("status", "new")
        if status == "error":
            lines.append(f"<b>{html.escape(str(item['source']))}</b>")
            lines.append(html.escape(str(item["summary"])))
            lines.append("")
            continue

        title = html.escape(str(item["title"]))
        source = html.escape(str(item["source"]))
        link = html.escape(str(item["url"]))
        published_hint = html.escape(str(item.get("published_hint", "")))
        summary = html.escape(str(item.get("summary", "")))

        lines.append(f"<b>{source}</b>")
        lines.append(f"<a href=\"{link}\">{title}</a>")
        if published_hint:
            lines.append(f"Date hint: {published_hint}")
        if summary:
            lines.append(summary[:3500])
        lines.append("")

    if omitted_count:
        lines.extend(
            [
                f"{omitted_count} additional item(s) were archived and processed but omitted from this digest.",
                "",
            ]
        )

    message = "\n".join(lines).strip()
    limit = max(0, MAX_TELEGRAM_CHARS)
    if len(message) <= limit:
        return message
    suffix = "\n\n… digest truncated; all detected documents were still archived and processed."
    if limit <= len(suffix):
        return suffix[:limit]
    return f"{message[: limit - len(suffix)].rstrip()}{suffix}"


def send_telegram_message(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("Missing ERCOT_LINK_TELEGRAM_BOT_TOKEN or ERCOT_NEWS_TELEGRAM_BOT_TOKEN")
    if not TELEGRAM_CHAT_ID:
        raise ValueError("Missing ERCOT_LINK_TELEGRAM_CHAT_ID or ERCOT_NEWS_TELEGRAM_CHAT_ID")

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    response = requests.post(
        url,
        json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": False,
        },
        timeout=REQUEST_TIMEOUT,
    )
    try:
        response.raise_for_status()
    finally:
        response.close()


def invoke_incremental_ingestion(
    downloaded_paths: Sequence[Path],
    archive_root: Path,
    *,
    enabled: bool,
) -> Dict[str, Any]:
    """Run incremental ingestion without allowing it to fail the monitor."""

    result: Dict[str, Any] = {
        "enabled": enabled,
        "attempted": False,
        "status": "disabled" if not enabled else "no_downloads",
        "summary": None,
        "error": None,
    }
    if not enabled or not downloaded_paths:
        return result

    captured_stdout = io.StringIO()
    try:
        try:
            from ERCOTAPI.rag_ingestion import (  # type: ignore
                IngestionPipeline,
                SourceRoot,
                default_config,
            )
        except ModuleNotFoundError as exc:
            if exc.name != "ERCOTAPI":
                raise
            # Supports direct execution from inside ERCOTAPI/ as well as
            # ``python -m ERCOTAPI.ercot_link_monitor`` from the repo root.
            from rag_ingestion import IngestionPipeline, SourceRoot, default_config  # type: ignore

        config = default_config()
        roots = []
        for source in config.source_roots:
            if source.name == "official_downloads":
                source = SourceRoot(
                    name=source.name,
                    path=archive_root,
                    source_authority=source.source_authority,
                    is_generated=source.is_generated,
                    default_source_kind=source.default_source_kind,
                    default_collections=source.default_collections,
                )
            roots.append(source)
        config = config.with_source_roots(roots)

        result["attempted"] = True
        with redirect_stdout(captured_stdout):
            summary = IngestionPipeline(config).update(paths=list(downloaded_paths))
        result.update(
            {
                "status": "completed",
                "summary": json.loads(json.dumps(summary, default=str)),
            }
        )
    except Exception as exc:
        result.update(
            {
                "attempted": True,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    captured = captured_stdout.getvalue().strip()
    if captured:
        result["captured_stdout"] = captured[-4000:]
    return result


def main() -> None:
    links_file = Path(os.getenv("ERCOT_LINKS_FILE", str(DEFAULT_LINKS_FILE)))
    state_file = Path(os.getenv("ERCOT_LINK_STATE_FILE", str(DEFAULT_STATE_FILE)))
    archive_root = official_document_dir()
    rag_auto_ingest = env_flag("ERCOT_RAG_AUTO_INGEST", True)

    links = load_links(links_file)
    state = load_state(state_file)
    changes, new_state = scan_sources(links, state, archive_root)
    save_state(state_file, new_state)

    downloaded_paths = [
        Path(value)
        for value in dict.fromkeys(
            str(item["downloaded_path"])
            for item in changes
            if item.get("status") in {"new", "updated"} and item.get("downloaded_path")
        )
    ]
    # Reconcile the complete durable archive on every successful monitor run.
    # Passing only today's downloads can otherwise leave an older parse error,
    # repaired sidecar, or externally removed file stale forever while new
    # downloads keep arriving continuously.
    ingestion_paths = [archive_root] if archive_root.exists() else downloaded_paths
    ingestion = invoke_incremental_ingestion(
        ingestion_paths,
        archive_root,
        enabled=rag_auto_ingest,
    )
    reported_changes, omitted_change_count = _reported_changes(changes)

    telegram_sent = False
    payload = {
        "checked_sources": len(links),
        "new_items": len(
            [item for item in changes if item.get("status") in {"new", "updated"}]
        ),
        "has_updates": any(
            item.get("status") in {"new", "updated"} for item in changes
        ),
        "changes": reported_changes,
        "reported_changes": len(reported_changes),
        "omitted_changes": omitted_change_count,
        "total_changes": len(changes),
        "telegram_text": format_telegram_message(
            reported_changes,
            omitted_count=omitted_change_count,
        ),
        "telegram_chat_id": TELEGRAM_CHAT_ID,
        "telegram_send_enabled": SEND_TELEGRAM,
        "telegram_send_no_updates": SEND_NO_UPDATES,
        "state_file": str(state_file),
        "links_file": str(links_file),
        "official_document_dir": str(archive_root),
        "downloaded_items": len(downloaded_paths),
        "downloaded_paths": [str(path) for path in downloaded_paths],
        "rag_auto_ingest": rag_auto_ingest,
        "ingestion": ingestion,
    }

    if SEND_TELEGRAM and (payload["has_updates"] or SEND_NO_UPDATES):
        send_telegram_message(payload["telegram_text"])
        telegram_sent = True

    payload["telegram_sent"] = telegram_sent

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
