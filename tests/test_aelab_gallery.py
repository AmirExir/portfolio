"""Offline regressions for the curated AELab screenshots and their provenance."""

from __future__ import annotations

import hashlib
import json
import struct
import unittest
from html.parser import HTMLParser
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRESERVED_SLIDES = {
    2: "MAP.png",
    4: "VOLTAGE_VIOLATIONSRESULTS_excell.jpg",
    10: "AELabDynamicPlots.png",
    23: "TARACAVISUAL.png",
    25: "TARAINTERACTIVE2.jpg",
    26: "TARA_ALT_Comparison.png",
    27: "TARAINTERACTIVE3.jpg",
    28: "TARAINTERACTIVE4.jpg",
}
ARCHIVED_SCREENSHOTS = {
    "05_psse_idv_generator.png",
    "07_psse_apply_batch_changes.png",
}


class _AELabGalleryParser(HTMLParser):
    """Collect only the AELab carousel and its selected-work thumbnail."""

    def __init__(self, content: str) -> None:
        super().__init__()
        self.galleries: list[list[dict[str, str]]] = []
        self.thumbnails: list[dict[str, str]] = []
        self._div_depth = 0
        self._gallery_depth: int | None = None
        self._in_thumbnail = False
        self.feed(content)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key: value or "" for key, value in attrs}
        if tag == "div":
            self._div_depth += 1
            if (
                "data-carousel" in attributes
                and attributes.get("aria-label") == "AELab screenshots"
            ):
                self._gallery_depth = self._div_depth
                self.galleries.append([])
        elif tag == "a":
            self._in_thumbnail = (
                attributes.get("href") == "#engineering-suite"
                and "work-image" in attributes.get("class", "").split()
            )
        elif tag == "img":
            if self._gallery_depth is not None:
                self.galleries[-1].append(attributes)
            if self._in_thumbnail:
                self.thumbnails.append(attributes)

    def handle_endtag(self, tag: str) -> None:
        if tag == "div":
            if self._div_depth == self._gallery_depth:
                self._gallery_depth = None
            self._div_depth -= 1
        elif tag == "a":
            self._in_thumbnail = False


class AELabGalleryTests(unittest.TestCase):
    """Keep requested result slides while replacing the old application UI."""

    def setUp(self) -> None:
        self.page = _AELabGalleryParser(
            (REPOSITORY_ROOT / "index.html").read_text(encoding="utf-8")
        )
        self.assertEqual(len(self.page.galleries), 1, "Expected one AELab gallery")
        self.slides = self.page.galleries[0]

    def test_gallery_retains_requested_result_slides_at_original_positions(self) -> None:
        self.assertEqual(len(self.slides), 28)
        for position, expected_src in PRESERVED_SLIDES.items():
            with self.subTest(position=position):
                self.assertEqual(self.slides[position - 1]["src"], expected_src)

    def test_replacement_screenshots_are_twenty_distinct_full_resolution_pngs(self) -> None:
        replacements = [
            slide
            for position, slide in enumerate(self.slides, start=1)
            if position not in PRESERVED_SLIDES
        ]
        self.assertEqual(len(replacements), 20)
        self.assertEqual(len({slide["src"] for slide in replacements}), 20)
        for slide in replacements:
            src = slide["src"]
            with self.subTest(src=src):
                path = Path(src)
                self.assertEqual(path.parent.as_posix(), "assets/images/aelab")
                self.assertEqual(path.suffix, ".png")
                self.assertNotIn(path.name, ARCHIVED_SCREENSHOTS)
                payload = (REPOSITORY_ROOT / path).read_bytes()
                self.assertEqual(payload[:8], b"\x89PNG\r\n\x1a\n")
                self.assertEqual(payload[8:16], b"\x00\x00\x00\rIHDR")
                self.assertEqual(struct.unpack(">II", payload[16:24]), (1600, 900))

    def test_selected_work_thumbnail_matches_first_gallery_slide(self) -> None:
        self.assertEqual(len(self.page.thumbnails), 1)
        self.assertEqual(self.page.thumbnails[0]["src"], self.slides[0]["src"])
        self.assertEqual(self.page.thumbnails[0].get("width"), "1600")
        self.assertEqual(self.page.thumbnails[0].get("height"), "900")

    def test_manifest_matches_slide_order_and_original_asset_bytes(self) -> None:
        manifest = json.loads(
            (REPOSITORY_ROOT / "docs/aelab-gallery.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["source"]["repository"], "psse_codes")
        self.assertEqual(manifest["source"]["catalog"], "Docs/AELab25_Screenshots")
        self.assertRegex(manifest["source"]["revision"], r"^[0-9a-f]{40}$")
        self.assertTrue(manifest["source"]["version"].strip())
        self.assertEqual(
            manifest["source"]["capture_script"], "Docs/capture_aelab25_screenshots.py"
        )
        self.assertEqual(manifest["source"]["captured_on"], "2026-09-23")
        entries = manifest["slides"]
        self.assertEqual([entry["position"] for entry in entries], list(range(1, 29)))
        self.assertEqual(len(entries), len(self.slides))
        for slide, entry in zip(self.slides, entries):
            position = entry["position"]
            with self.subTest(position=position):
                self.assertEqual(slide["src"], entry["src"])
                self.assertEqual(slide["alt"], entry["alt"])
                path = Path(entry["src"])
                self.assertFalse(path.is_absolute())
                self.assertNotIn("..", path.parts)
                payload = (REPOSITORY_ROOT / path).read_bytes()
                self.assertEqual(len(payload), entry["bytes"])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), entry["sha256"])
                if position in PRESERVED_SLIDES:
                    self.assertEqual(entry["origin"], "retained")
                    self.assertEqual(entry["source_file"], PRESERVED_SLIDES[position])
                else:
                    self.assertEqual(entry["origin"], "app-capture")
                    self.assertEqual(entry["source_file"], path.name)
                    self.assertEqual(
                        entry["catalog_entry"],
                        f"{manifest['source']['catalog']}/{path.name}",
                    )
                    self.assertNotIn(path.name, ARCHIVED_SCREENSHOTS)
                    self.assertEqual((entry["width"], entry["height"]), (1600, 900))
                    self.assertEqual(slide.get("width"), str(entry["width"]))
                    self.assertEqual(slide.get("height"), str(entry["height"]))
                    self.assertEqual(
                        struct.unpack(">II", payload[16:24]),
                        (entry["width"], entry["height"]),
                    )

    def test_gallery_images_remain_described_and_lazy_loaded(self) -> None:
        for position, slide in enumerate(self.slides, start=1):
            with self.subTest(position=position):
                self.assertTrue(slide.get("alt", "").strip())
                self.assertEqual(slide.get("loading"), "lazy")
                self.assertEqual(slide.get("decoding"), "async")


if __name__ == "__main__":
    unittest.main()
