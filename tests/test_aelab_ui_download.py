"""Regression checks for the published AELab UI overview download."""

from __future__ import annotations

import hashlib
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlsplit


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PDF_NAME = "AELab25_Modern_UI_Overview.pdf"
EXPECTED_HREF = PDF_NAME + "?v=25.1-20260822"
EXPECTED_SHA256 = "97024efa8b70f9d1c200f947b6b234e3974c12ed648ad60693038aab314dc6e9"


class _AELabDownloadParser(HTMLParser):
    """Collect AELab download anchors and their enclosing DIV classes."""

    def __init__(self) -> None:
        super().__init__()
        self._div_classes: List[Set[str]] = []
        self.links: List[Tuple[Dict[str, str], Set[str]]] = []

    def handle_starttag(
        self, tag: str, attrs: List[Tuple[str, Optional[str]]]
    ) -> None:
        attributes = {key: value or "" for key, value in attrs}
        if tag == "div":
            self._div_classes.append(set(attributes.get("class", "").split()))
            return
        if tag != "a":
            return
        href = attributes.get("href", "")
        if Path(urlsplit(href).path).name != PDF_NAME:
            return
        enclosing_classes: Set[str] = set()
        for classes in self._div_classes:
            enclosing_classes.update(classes)
        self.links.append((attributes, enclosing_classes))

    def handle_endtag(self, tag: str) -> None:
        if tag == "div" and self._div_classes:
            self._div_classes.pop()


def test_aelab_ui_overview_is_current_and_linked_from_hero_and_card() -> None:
    parser = _AELabDownloadParser()
    parser.feed((REPOSITORY_ROOT / "index.html").read_text(encoding="utf-8"))

    assert len(parser.links) == 2
    assert {attributes["href"] for attributes, _ in parser.links} == {EXPECTED_HREF}
    assert {
        attributes.get("download") for attributes, _ in parser.links
    } == {PDF_NAME}
    assert any("hero-actions" in classes for _, classes in parser.links)
    assert any("card-actions" in classes for _, classes in parser.links)

    pdf_path = REPOSITORY_ROOT / PDF_NAME
    payload = pdf_path.read_bytes()
    assert payload.startswith(b"%PDF-")
    assert payload.rstrip().endswith(b"%%EOF")
    assert len(payload) > 5_000_000
    assert hashlib.sha256(payload).hexdigest() == EXPECTED_SHA256
