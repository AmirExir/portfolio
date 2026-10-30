"""Regression checks for the published AELab UI overview download."""

from __future__ import annotations

import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Dict, List, Optional, Set, Tuple
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

from scripts.build_aelab_overview import BUILDER, CATALOG, source_path, validate_source

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PDF_NAME = "AELab25_Modern_UI_Overview.pdf"
EXPECTED_HREF = PDF_NAME + "?v=25.1-overview-20261030"
EXPECTED_SHA256 = "682ff8423045dd63fc8b62882a46cb0e949896a796689884e3b3f8ddc767881c"


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


def test_overview_contains_the_complete_reference_atlas_and_source_record() -> None:
    content = json.loads(
        (REPOSITORY_ROOT / "docs/aelab-overview.json").read_text(encoding="utf-8")
    )
    assert content["reviewed_on"] == "2026-10-30"
    assert content["schema_version"] == 1
    assert re.fullmatch(r"[0-9a-f]{40}", content["source"]["revision"])
    files = content["source"]["files"]
    assert {BUILDER, CATALOG}.issubset(files)
    screenshots = {
        Path(path).name for path in files
        if path.startswith("Docs/AELab25_Screenshots/") and path.endswith(".png")
    }
    assert {int(name.split("_", 1)[0]) for name in screenshots} == set(range(1, 56))
    for path, digest in files.items():
        assert not Path(path).is_absolute() and ".." not in Path(path).parts
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
    for slide in content["slides"]:
        assert slide["sources"] and set(slide["sources"]).issubset(files)
    payload = (REPOSITORY_ROOT / PDF_NAME).read_bytes()
    assert len(re.findall(rb"/Type /Page\b", payload)) == 1 + len(content["slides"]) + 55


def test_overview_builder_rejects_changed_inputs_and_revision() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        revision = "a" * 40
        files = {}
        for name in (BUILDER, CATALOG):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"# Synthetic documentation input\n")
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        content = {
            "schema_version": 1,
            "reviewed_on": "2026-10-30",
            "source": {"revision": revision, "files": files},
        }
        with patch("scripts.build_aelab_overview.subprocess.run") as git:
            git.return_value = subprocess.CompletedProcess([], 0, stdout=revision + "\n")
            validate_source(root, content)
            (root / BUILDER).write_bytes(b"# Changed after review\n")
            with unittest.TestCase().assertRaisesRegex(ValueError, "changed reviewed source"):
                validate_source(root, content)
            git.return_value = subprocess.CompletedProcess([], 0, stdout="b" * 40 + "\n")
            with unittest.TestCase().assertRaisesRegex(ValueError, "revision mismatch"):
                validate_source(root, content)


def test_overview_builder_rejects_source_paths_outside_the_checkout() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        for unsafe in ("../outside.py", str(root / "absolute.py")):
            with unittest.TestCase().assertRaisesRegex(ValueError, "source-relative"):
                source_path(root, unsafe)
        (root / "outside").symlink_to(root.parent, target_is_directory=True)
        with unittest.TestCase().assertRaisesRegex(ValueError, "escapes"):
            source_path(root, "outside/renderer.py")
