"""Monitor ERCOT pages for newly posted documents and summarize them.

This script is designed to be called from n8n, cron, or manually:
1. Read label=url pairs from `ercot_links`
2. Fetch each ERCOT page
3. Discover likely document/detail links in the page body
4. Compare against local state to find new items
5. Extract best-effort text from new items and emit a JSON payload

Example:
    python ercot_link_monitor.py

Optional environment variables:
    ERCOT_LINKS_FILE=/path/to/ercot_links
    ERCOT_LINK_STATE_FILE=/path/to/.ercot_link_state.json
    ERCOT_LINK_MAX_ITEMS_PER_SOURCE=5
    ERCOT_LINK_MAX_SUMMARY_CHARS=1200
    ERCOT_LINK_TELEGRAM_BOT_TOKEN=...
    ERCOT_LINK_TELEGRAM_CHAT_ID=@ERCOTNEWS
    ERCOT_LINK_SEND_TELEGRAM=true
    ERCOT_LINK_SEND_NO_UPDATES=false
"""

from __future__ import annotations

import csv
import html
import io
import json
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urljoin, urlparse
from xml.etree import ElementTree as ET

import requests

try:
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None


DEFAULT_LINKS_FILE = Path(__file__).with_name("ercot_links")
DEFAULT_STATE_FILE = Path(__file__).with_name(".ercot_link_state.json")
REQUEST_TIMEOUT = 20
MAX_ITEMS_PER_SOURCE = int(os.getenv("ERCOT_LINK_MAX_ITEMS_PER_SOURCE", "5"))
MAX_SUMMARY_CHARS = int(os.getenv("ERCOT_LINK_MAX_SUMMARY_CHARS", "1200"))
SEND_TELEGRAM = os.getenv("ERCOT_LINK_SEND_TELEGRAM", "false").lower() in {"1", "true", "yes"}
SEND_NO_UPDATES = os.getenv("ERCOT_LINK_SEND_NO_UPDATES", "false").lower() in {"1", "true", "yes"}
DEFAULT_TELEGRAM_CHAT_ID = "@ERCOTNEWS"
USER_AGENT = "ERCOT-Link-Monitor/1.0 (+https://github.com/AmirExir/portfolio)"

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
    "/services",
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

    @property
    def fingerprint(self) -> str:
        return f"{self.source_label}|{self.url}"


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
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_state(path: Path, state: Dict[str, List[str]]) -> None:
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def fetch(session: requests.Session, url: str) -> requests.Response:
    response = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
    response.raise_for_status()
    return response


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def path_extension(url: str) -> str:
    return Path(urlparse(url).path).suffix.lower()


def is_allowed_domain(url: str) -> bool:
    netloc = urlparse(url).netloc.lower()
    return netloc.endswith("ercot.com") or netloc.endswith(".ercot.com")


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
    if "/mktrules/issues/" in path and re.search(r"\b(?:nogrr|pgrr)\d+\b", text_l):
        return True
    if "data-product-details" in path or "download" in path:
        return True
    return False


def extract_anchor_candidates(source_label: str, source_url: str, html_text: str) -> List[DiscoveredItem]:
    anchor_pattern = re.compile(r"<a\b[^>]*href=(['\"])(.*?)\1[^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)
    seen: set[str] = set()
    results: List[DiscoveredItem] = []

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
        context = extract_context_window(html_text, text)
        published_hint = extract_published_hint(context)
        results.append(
            DiscoveredItem(
                source_label=source_label,
                source_url=source_url,
                title=text,
                url=absolute_url,
                item_type=item_type or "page",
                context=context,
                published_hint=published_hint,
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
    if not text:
        return ""
    match = re.search(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{1,2}, \d{4}\b",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(0)
    match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
    return match.group(0) if match else ""


def rank_items(items: Sequence[DiscoveredItem]) -> List[DiscoveredItem]:
    def score(item: DiscoveredItem) -> Tuple[int, str, str]:
        title_l = item.title.lower()
        priority = 0
        if any(token in title_l for token in ("agenda", "minutes", "presentation", "packet", "overview", "report")):
            priority += 3
        if item.item_type in {"pdf", "docx", "xlsx", "csv"}:
            priority += 2
        if re.search(r"\b(?:nogrr|pgrr)\d+\b", title_l):
            priority += 2
        if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", title_l):
            priority += 1
        return (priority, item.published_hint, item.title)

    ranked = sorted(items, key=score, reverse=True)
    return ranked[:MAX_ITEMS_PER_SOURCE]


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


def extract_docx_summary(content: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            xml = zf.read("word/document.xml")
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
            workbook_xml = zf.read("xl/workbook.xml")
            root = ET.fromstring(workbook_xml)
            names = [
                sheet.attrib.get("name", "")
                for sheet in root.findall(".//x:sheets/x:sheet", NS)
                if sheet.attrib.get("name")
            ]
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

    summary = summarize_binary_content(item.url, response.content, response.headers.get("Content-Type", ""))
    if item.context:
        context_summary = summarize_text(item.context, max_chars=280)
        if context_summary and context_summary != "No readable text could be extracted.":
            return f"{summary}\nContext: {context_summary}"
    return summary


def scan_sources(links: Sequence[SourceLink], state: Dict[str, List[str]]) -> Tuple[List[Dict[str, str]], Dict[str, List[str]]]:
    headers = {"User-Agent": USER_AGENT}
    session = requests.Session()
    session.headers.update(headers)

    changes: List[Dict[str, str]] = []
    new_state: Dict[str, List[str]] = {key: list(value) for key, value in state.items()}

    for source in links:
        try:
            response = fetch(session, source.url)
            candidates = extract_anchor_candidates(source.label, source.url, response.text)
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

        known = set(new_state.get(source.label, []))
        new_items = [item for item in candidates if item.fingerprint not in known]
        if not new_items:
            continue

        for item in new_items:
            item_summary = summarize_item(session, item)
            changes.append(
                {
                    "source": item.source_label,
                    "source_url": item.source_url,
                    "title": item.title,
                    "url": item.url,
                    "summary": item_summary,
                    "published_hint": item.published_hint,
                    "item_type": item.item_type,
                    "status": "new",
                }
            )
            known.add(item.fingerprint)

        # Keep state bounded so it does not grow forever.
        new_state[source.label] = sorted(known, reverse=True)[:100]

    return changes, new_state


def format_telegram_message(changes: Sequence[Dict[str, str]]) -> str:
    if not changes:
        return "ERCOT monitor: no new items found."

    lines = ["<b>ERCOT Monitor</b>", ""]
    for item in changes:
        status = item.get("status", "new")
        if status == "error":
            lines.append(f"<b>{html.escape(item['source'])}</b>")
            lines.append(html.escape(item["summary"]))
            lines.append("")
            continue

        title = html.escape(item["title"])
        source = html.escape(item["source"])
        link = html.escape(item["url"])
        published_hint = html.escape(item.get("published_hint", ""))
        summary = html.escape(item.get("summary", ""))

        lines.append(f"<b>{source}</b>")
        lines.append(f"<a href=\"{link}\">{title}</a>")
        if published_hint:
            lines.append(f"Date hint: {published_hint}")
        if summary:
            lines.append(summary[:3500])
        lines.append("")

    return "\n".join(lines).strip()


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
    response.raise_for_status()


def main() -> None:
    links_file = Path(os.getenv("ERCOT_LINKS_FILE", str(DEFAULT_LINKS_FILE)))
    state_file = Path(os.getenv("ERCOT_LINK_STATE_FILE", str(DEFAULT_STATE_FILE)))

    links = load_links(links_file)
    state = load_state(state_file)
    changes, new_state = scan_sources(links, state)
    save_state(state_file, new_state)

    telegram_sent = False
    payload = {
        "checked_sources": len(links),
        "new_items": len([item for item in changes if item.get("status") == "new"]),
        "has_updates": any(item.get("status") == "new" for item in changes),
        "changes": changes,
        "telegram_text": format_telegram_message(changes),
        "telegram_chat_id": TELEGRAM_CHAT_ID,
        "telegram_send_enabled": SEND_TELEGRAM,
        "telegram_send_no_updates": SEND_NO_UPDATES,
        "state_file": str(state_file),
        "links_file": str(links_file),
    }

    if SEND_TELEGRAM and (payload["has_updates"] or SEND_NO_UPDATES):
        send_telegram_message(payload["telegram_text"])
        telegram_sent = True

    payload["telegram_sent"] = telegram_sent

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
