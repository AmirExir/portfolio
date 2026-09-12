"""Extract stable ERCOT page content without altering archived source bytes."""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urldefrag, urljoin, urlparse


@dataclass
class PageContent:
    """Readable content and links used to compare two observations of a page."""

    text: str
    links: tuple[str, ...]


@dataclass
class _ContentRegion:
    priority: int
    text: list[str] = field(default_factory=list)
    links: set[str] = field(default_factory=set)


class _ContentParser(HTMLParser):
    """Prefer ERCOT's main content containers over shared navigation chrome."""

    _VOID = {
        "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr",
    }
    _EXCLUDED_TAGS = {
        "script", "style", "noscript", "head", "header", "mainheader", "nav",
        "footer", "aside", "svg", "template",
    }
    _EXCLUDED_IDS = {"bcrumb", "relatedcontent"}
    _EXCLUDED_CLASSES = {"tabnav", "breadcrumb", "right-panel"}

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.regions = [_ContentRegion(0)]
        self.stack: list[tuple[str, bool, int | None]] = []

    @property
    def _excluded(self) -> bool:
        return bool(self.stack and self.stack[-1][1])

    def _active_regions(self) -> list[_ContentRegion]:
        return [self.regions[0], *[
            self.regions[index]
            for _, _, index in self.stack
            if index is not None
        ]]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = set(str(values.get("class") or "").lower().split())
        element_id = str(values.get("id") or "").lower()
        excluded = (
            self._excluded or tag in self._EXCLUDED_TAGS
            or element_id in self._EXCLUDED_IDS
            or bool(classes & self._EXCLUDED_CLASSES)
        )
        region_index = None
        if not excluded:
            priority = (
                3 if "maincontent" in classes or element_id == "maincontent"
                else 2 if tag in {"main", "article"} or values.get("role") == "main"
                else 1 if "content-margin" in classes
                else 0
            )
            if priority:
                region_index = len(self.regions)
                self.regions.append(_ContentRegion(priority))
        if tag not in self._VOID:
            self.stack.append((tag, excluded, region_index))
        if not excluded and tag == "a":
            href = str(values.get("href") or "").strip()
            if href and not href.startswith("#"):
                url = urldefrag(urljoin(self.base_url, href))[0]
                if urlparse(url).scheme in {"http", "https"}:
                    for region in self._active_regions():
                        region.links.add(url)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in self._VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data: str) -> None:
        if not self._excluded and data.strip():
            for region in self._active_regions():
                region.text.append(data)

    def result(self) -> PageContent:
        priority = max(region.priority for region in self.regions)
        selected = [region for region in self.regions if region.priority == priority]
        return PageContent(
            text=" ".join(" ".join(part for region in selected for part in region.text).split()),
            links=tuple(sorted({link for region in selected for link in region.links})),
        )


def extract_page_content(content: bytes | str, *, base_url: str = "") -> PageContent:
    """Keep main text and target URLs, ignoring scripts, styling and site chrome.

    Link targets participate in comparison because a replacement attachment can
    keep the same displayed title. Query parameters are retained for downloads.
    Pages without a known main container fall back to all non-chrome content.
    """

    if isinstance(content, bytes):
        for encoding in ("utf-8-sig", "utf-16", "cp1252"):
            try:
                text = content.decode(encoding)
                break
            except UnicodeError:
                continue
        else:
            text = content.decode("utf-8", errors="replace")
    else:
        text = content
    parser = _ContentParser(base_url)
    parser.feed(text)
    parser.close()
    return parser.result()
