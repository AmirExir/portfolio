"""Read timestamped publisher headlines for the dashboard's on-demand news check."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urldefrag, urlsplit

from lxml import etree
import requests

from market_agent.breaking_news_sources import BREAKING_NEWS_FEEDS


HEADLINE_WINDOW = timedelta(hours=36)
_MAX_FEED_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class PublisherHeadline:
    """One linked article with the publisher's stated publication time in UTC."""

    publisher: str
    title: str
    url: str
    published_at: datetime


@dataclass(frozen=True)
class HeadlineCheck:
    """A point-in-time feed check, including sources that failed or had no recent items."""

    checked_at: datetime
    headlines: tuple[PublisherHeadline, ...]
    failures: tuple[str, ...]
    empty_sources: tuple[str, ...]


def _publication_time(raw: str) -> datetime | None:
    try:
        published = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        try:
            published = parsedate_to_datetime(raw)
        except (TypeError, ValueError, OverflowError):
            return None
    if published.tzinfo is None:
        return None
    try:
        return published.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None


def _article_url(raw: str) -> str:
    raw = raw.strip()
    if not raw or "\\" in raw or any(ord(char) <= 32 or ord(char) == 127 for char in raw):
        return ""
    try:
        parts = urlsplit(raw)
        if parts.scheme in {"http", "https"} and parts.hostname and parts.username is None:
            return urldefrag(raw).url
    except ValueError:
        pass
    return ""


def parse_publisher_feed(body: bytes, publisher: str, checked_at: datetime) -> list[PublisherHeadline]:
    """Extract dated HTTP(S) articles from an RSS or Atom feed within the news window."""
    root = etree.fromstring(body, parser=etree.XMLParser(resolve_entities=False, no_network=True))
    if root.getroottree().docinfo.doctype:
        raise ValueError("feed contains an unsupported document type declaration")
    kind = etree.QName(root).localname
    if kind == "rss" and root.find("{*}channel") is not None:
        entries = root.findall("{*}channel/{*}item")
    elif kind == "feed":
        entries = root.findall("{*}entry")
    else:
        raise ValueError("response is not a supported RSS or Atom feed")

    headlines = []
    for entry in entries:
        title = " ".join((entry.findtext("{*}title") or "").split())
        raw_date = entry.findtext("{*}pubDate") or entry.findtext("{*}published") or entry.findtext("{*}date") or ""
        published = _publication_time(raw_date.strip())
        if not title or published is None or not checked_at - HEADLINE_WINDOW <= published <= checked_at:
            continue
        for link in entry.findall("{*}link"):
            if link.get("rel", "alternate") != "alternate":
                continue
            url = _article_url(link.get("href") or link.text or "")
            if url:
                headlines.append(PublisherHeadline(publisher, title, url, published))
                break
    return headlines


def _read_feed(url: str) -> bytes:
    with requests.get(url, headers={"User-Agent": "Market-Agent-News/1.0"}, timeout=(3, 6), stream=True) as response:
        response.raise_for_status()
        body = bytearray()
        for chunk in response.iter_content(chunk_size=65536):
            if len(body) + len(chunk) > _MAX_FEED_BYTES:
                raise ValueError("feed exceeds the size limit")
            body.extend(chunk)
    return bytes(body)


def check_publisher_headlines(
    feeds: tuple[tuple[str, str], ...] = BREAKING_NEWS_FEEDS, *, now: datetime | None = None,
) -> HeadlineCheck:
    """Check the configured publishers concurrently and preserve partial results."""
    checked_at = now or datetime.now(timezone.utc)

    def fetch(source: tuple[str, str]) -> tuple[list[PublisherHeadline], str]:
        publisher, url = source
        try:
            items = parse_publisher_feed(_read_feed(url), publisher, checked_at)
            return items, ""
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "error"
            return [], f"{publisher} (HTTP {status})"
        except requests.RequestException:
            return [], f"{publisher} (connection failed)"
        except (etree.XMLSyntaxError, ValueError):
            return [], f"{publisher} (invalid or unsupported feed)"

    by_url: dict[str, PublisherHeadline] = {}
    failures, empty_sources = [], []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for (publisher, _), (items, error) in zip(feeds, executor.map(fetch, feeds)):
            if error:
                failures.append(error)
            elif not items:
                empty_sources.append(publisher)
            for item in items:
                if item.url not in by_url or item.published_at > by_url[item.url].published_at:
                    by_url[item.url] = item
    headlines = sorted(by_url.values(), key=lambda item: item.published_at, reverse=True)
    return HeadlineCheck(checked_at, tuple(headlines), tuple(failures), tuple(empty_sources))
