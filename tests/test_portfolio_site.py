"""Offline regression checks for the portfolio's published local addresses."""

from __future__ import annotations

import os
import re
import tempfile
import unittest
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CSS_URL = re.compile(r'''url\(\s*(?:"([^"]*)"|'([^']*)'|([^\s)]+))\s*\)''')


class _PageLinks(HTMLParser):
    """Collect static IDs, file references, and styles from one HTML page."""

    def __init__(self, content: str) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.references: list[tuple[str, str, str]] = []
        self.styles: list[str] = []
        self._in_style = False
        self.feed(content)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.append(values["id"] or "")
        for attribute in ("href", "src", "poster"):
            if values.get(attribute):
                self.references.append((tag, attribute, values[attribute] or ""))
        if values.get("style"):
            self.styles.append(values["style"] or "")
        self._in_style = self._in_style or tag == "style"

    def handle_endtag(self, tag: str) -> None:
        if tag == "style":
            self._in_style = False

    def handle_data(self, data: str) -> None:
        if self._in_style:
            self.styles.append(data)


def _local_target(address: str, source: Path, root: Path) -> Path | None:
    """Resolve a local URL against its containing file, ignoring query strings."""
    url = urlsplit(address)
    if url.scheme or url.netloc:
        return None
    path = unquote(url.path)
    if not path:
        return source
    target = root / path.lstrip("/") if path.startswith("/") else source.parent / path
    return Path(os.path.abspath(target))


def _exact_file_exists(path: Path, root: Path) -> bool:
    """Check each filename's exact spelling even on case-insensitive macOS."""
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return False
    current = root
    for part in parts:
        if not current.is_dir() or part not in {entry.name for entry in current.iterdir()}:
            return False
        current /= part
    return current.is_file()


def _css_errors(content: str, source: Path, root: Path) -> list[str]:
    """Resolve CSS assets relative to the stylesheet rather than the page."""
    content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
    errors = []
    for match in CSS_URL.finditer(content):
        address = next(group for group in match.groups() if group is not None)
        target = _local_target(address, source, root)
        if target is not None and not _exact_file_exists(target, root):
            errors.append(f"{source.name}: missing CSS asset {address}")
    return errors


def _page_errors(source: Path, root: Path) -> list[str]:
    """Validate static local links, fragment targets, and linked CSS assets."""
    page = _PageLinks(source.read_text(encoding="utf-8"))
    errors = [f"Duplicate ID: {name}" for name, count in Counter(page.ids).items() if count > 1]
    checked_stylesheets: set[Path] = set()
    for tag, attribute, address in page.references:
        target = _local_target(address, source, root)
        if target is None:
            continue
        if not _exact_file_exists(target, root):
            errors.append(f"{tag}[{attribute}]: missing local file {address}")
            continue
        fragment = unquote(urlsplit(address).fragment)
        if fragment and attribute == "href" and target.suffix.lower() == ".html":
            ids = page.ids if target == source else _PageLinks(target.read_text(encoding="utf-8")).ids
            if fragment not in ids:
                errors.append(f"Missing fragment: {address}")
        if target.suffix.lower() == ".css" and target not in checked_stylesheets:
            checked_stylesheets.add(target)
            errors.extend(_css_errors(target.read_text(encoding="utf-8"), target, root))
    for style in page.styles:
        errors.extend(_css_errors(style, source, root))
    return errors


class PortfolioAddressTests(unittest.TestCase):
    """Keep portfolio downloads, scripts, images, and navigation resolvable."""

    def test_portfolio_local_addresses_resolve(self) -> None:
        errors = _page_errors(REPOSITORY_ROOT / "index.html", REPOSITORY_ROOT)
        self.assertEqual(errors, [], "\n".join(errors))


class AddressValidatorTests(unittest.TestCase):
    """Prove that the address checks catch regressions independently of the site."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.page = self.root / "index.html"

    def _write(self, name: str, content: str = "") -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def test_query_strings_spaces_root_paths_and_css_relative_assets(self) -> None:
        self._write("Research Paper.pdf")
        self._write("assets/images/Control Room.jpeg")
        self._write("assets/js/portfolio.js")
        self._write("assets/css/portfolio.css", 'body { background: url("../images/Control%20Room.jpeg?v=2"); }')
        self._write("atlas.html", '<section id="map"></section>')
        self._write("index.html", '''
            <link href="assets/css/portfolio.css?v=2" rel="stylesheet">
            <script src="assets/js/portfolio.js" defer></script>
            <a href="Research%20Paper.pdf?v=1#page=2">Paper</a>
            <a href="/Research Paper.pdf">Paper</a>
            <a href="#work">Work</a><section id="work"></section>
            <a href="atlas.html#map">Map</a><iframe src="atlas.html?region=ercot"></iframe>
            <a href="https://example.com/missing">External</a>
            <a href="mailto:contact@example.com">Email</a>
        ''')
        self.assertEqual(_page_errors(self.page, self.root), [])

    def test_missing_files_and_case_mismatches_are_reported(self) -> None:
        self._write("AmirExir_Final.pdf")
        self._write("index.html", '''
            <a href="Amir_Exir_Final.pdf">Missing paper</a>
            <a href="amirexir_final.pdf">Incorrect case</a>
            <script src="missing.js"></script><iframe src="missing.html"></iframe>
        ''')
        errors = _page_errors(self.page, self.root)
        self.assertEqual(len(errors), 4)
        self.assertTrue(all("missing local file" in error for error in errors))

    def test_duplicate_ids_and_missing_fragments_are_reported(self) -> None:
        self._write("index.html", '<div id="work"></div><div id="work"></div><a href="#Work">Work</a>')
        self.assertEqual(_page_errors(self.page, self.root), ["Duplicate ID: work", "Missing fragment: #Work"])

    def test_missing_css_assets_are_reported_and_data_urls_are_ignored(self) -> None:
        self._write("assets/site.css", '''
            /* background: url(unused.jpg); */
            .hero { background: url("missing photo.jpg?v=1"); }
            .icon { background: url(data:image/png;base64,AAAA); }
        ''')
        self._write("index.html", '<link rel="stylesheet" href="assets/site.css">')
        self.assertEqual(_page_errors(self.page, self.root), ["site.css: missing CSS asset missing photo.jpg?v=1"])


if __name__ == "__main__":
    unittest.main()
