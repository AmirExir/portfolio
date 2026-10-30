#!/usr/bin/env python3
"""Build the portfolio overview using a fingerprinted AELab portable renderer.

Only documentation code and archived screenshots are imported. This script does
not launch the application, capture its window, or run a licensed simulator.
"""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUILDER = "Docs/build_aelab25_presentations.py"
CATALOG = "Docs/capture_aelab25_screenshots.py"
SCREENSHOTS = "Docs/AELab25_Screenshots"


def source_path(root: Path, relative: str) -> Path:
    """Resolve a source-relative file, rejecting traversal and escaping links."""
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Expected a source-relative path: {relative!r}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Source path escapes the repository: {relative}")
    return resolved


def validate_source(root: Path, content: dict[str, Any]) -> None:
    """Check the reviewed revision and every fingerprint before source import."""
    if not isinstance(content, dict):
        raise ValueError("Overview content must be a JSON object.")
    if content.get("schema_version") != 1:
        raise ValueError("Expected overview content schema_version 1.")
    date.fromisoformat(content["reviewed_on"])
    source = content["source"]
    revision, files = source["revision"], source["files"]
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("source.revision must be a full 40-character Git revision.")
    actual = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    if actual != revision:
        raise ValueError(f"Source revision mismatch: expected {revision}, found {actual}. "
                         "Use the reviewed checkout or explicitly review and update the manifest.")
    if not isinstance(files, dict) or not {BUILDER, CATALOG}.issubset(files):
        raise ValueError("source.files must fingerprint the presentation builder and capture catalog.")
    for relative, digest in files.items():
        path = source_path(root, relative)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid SHA-256 fingerprint for {relative}.")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Missing or changed reviewed source: {relative}. "
                             "Restore it or review the change before updating its fingerprint.")


def load_renderer(root: Path) -> ModuleType:
    """Import the already-verified builder without invoking its command line."""
    name = "_portfolio_aelab_renderer"
    spec = importlib.util.spec_from_file_location(name, root / BUILDER)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the reviewed AELab presentation builder.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # Compile the fingerprinted bytes directly: do not read/write source caches.
    exec(compile((root / BUILDER).read_bytes(), str(root / BUILDER), "exec"), module.__dict__)
    return module


def build_slides(renderer: ModuleType, root: Path, content: dict[str, Any]) -> list[tuple[Any, list[str]]]:
    """Combine reviewed feature pages with all explicitly archived catalog views."""
    previous_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        screens = renderer.load_capture_catalog(root / CATALOG)
    finally:
        sys.dont_write_bytecode = previous_bytecode
    errors = renderer.catalog_validation_errors(screens, root / SCREENSHOTS)
    if errors or len(screens) != 55:
        raise ValueError("Expected the reviewed 55-screen archive. " + "; ".join(errors))
    fingerprints = content["source"]["files"]
    for screen in screens:
        relative = f"{SCREENSHOTS}/{screen.filename}"
        if relative not in fingerprints:
            raise ValueError(f"Screenshot fingerprint missing: {relative}")
    cover = content["cover"]
    pages = [(renderer.DeckSlide(kind="cover", **cover,
              image_path=renderer.resolve_screenshot_path(screens[0], root / SCREENSHOTS)), [])]
    for entry in content["slides"]:
        bullets, sources = entry["bullets"], entry["sources"]
        if not isinstance(bullets, list) or len(bullets) not in (3, 6):
            raise ValueError(f"Slide {entry['title']!r} needs three or six bullets.")
        if not isinstance(sources, list) or not sources or any(path not in fingerprints for path in sources):
            raise ValueError(f"Slide {entry['title']!r} has missing or unfingerprinted sources.")
        pages.append((renderer.DeckSlide(kind="groups", title=entry["title"],
                      breadcrumb=entry["breadcrumb"], description=entry["description"],
                      bullets=tuple(bullets)), sources))
    overrides = content.get("screenshot_overrides", {})
    unknown = set(overrides) - {screen.key for screen in screens}
    if unknown:
        raise ValueError(f"Unknown screenshot override keys: {sorted(unknown)}")
    revision = content["source"]["revision"][:12]
    for screen in screens:
        override = overrides.get(screen.key, {})
        if set(override) - {"title", "description"}:
            raise ValueError(f"Unsupported screenshot override fields: {screen.key}")
        description = override.get("description", screen.description)
        description += f" Archived UI; current controls may differ. Source review: {revision}."
        pages.append((renderer.DeckSlide(kind="screen",
                      title="Reference: " + override.get("title", screen.title),
                      breadcrumb="ARCHIVED UI | " + screen.breadcrumb,
                      description=description, group=screen.group,
                      image_path=renderer.resolve_screenshot_path(screen, root / SCREENSHOTS)), []))
    return pages


def validate_layout(renderer: ModuleType, slide: Any) -> None:
    """Reject text that would be truncated, overlap another region, or escape a box."""
    Image, ImageDraw, ImageFont, _ = renderer._pillow_modules()
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    def fits(text: str, size: int, width: int, lines: int, bold: bool = False) -> None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty or non-text field in {slide.title!r}.")
        font = renderer._font(ImageFont, size, bold=bold)
        wrapped = renderer._wrapped_lines(draw, text, font, width)
        if len(wrapped) > lines or any(renderer._text_width(draw, line, font) > width for line in wrapped):
            raise ValueError(f"Text exceeds the layout in {slide.title!r}: {text!r}. Shorten this field.")

    if slide.kind == "cover":
        fits(slide.title, 58, 650, 3, True)
        fits(slide.breadcrumb, 24, 630, 3, True)
        fits(slide.description, 26, 610, 6)
        fits(renderer.LONG_DISCLAIMER, 19, 610, 6)
    elif slide.kind == "screen":
        fits(slide.title, 42, 1240, 1, True)
        fits(slide.breadcrumb, 18, 1760, 2, True)
        fits(slide.description, 23, 360, 15)
    else:
        fits(slide.title, 48, 1650, 1, True)
        fits(slide.breadcrumb, 21, 1600, 2, True)
        fits(slide.description, 26, 1660, 4)
        for bullet in slide.bullets:
            fits(bullet, 22, 746 if len(slide.bullets) == 6 else 1596, 3, True)


def render_pdf(renderer: ModuleType, pages: list[tuple[Any, list[str]]], content: dict[str, Any],
               output: Path, preview_dir: Path | None) -> None:
    """Render temporary pages and atomically replace the complete verified PDF."""
    _, ImageDraw, ImageFont, _ = renderer._pillow_modules()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aelab-overview-", dir=output.parent) as temporary:
        staging = Path(temporary)
        pngs = []
        for index, (slide, sources) in enumerate(pages):
            renderer.DISCLAIMER = ("ARCHIVED UI REFERENCE | CAPTURE PREDATES THIS REVIEW | NO NEW SOLVER VALIDATION"
                                  if slide.kind == "screen" else
                                  "SOURCE-REVIEWED FEATURE OVERVIEW | NO LICENSED SOLVER OR ENGINEERING VALIDATION")
            canvas = renderer.render_slide_image(slide, index, len(pages))
            if slide.kind != "screen":
                draw = ImageDraw.Draw(canvas)
                font = renderer._font(ImageFont, 17)
                citations = "Sources: " + "; ".join(sources[:2]) if sources else "Cover image: archived UI reference"
                if len(sources) > 2:
                    citations += "; full references in docs/aelab-overview.json"
                if renderer._text_width(draw, citations, font) > 1776:
                    raise ValueError(f"Source citation exceeds slide width: {slide.title}")
                draw.text((94, 930), citations, font=font, fill=(132, 151, 174))
                draw.text((94, 958), f"Reviewed {content['reviewed_on']} | Source {content['source']['revision']}",
                          font=font, fill=(132, 151, 174))
            png = staging / f"slide-{index + 1:03d}.png"
            canvas.save(png)
            canvas.close()
            pngs.append(png)
        pdf = staging / output.name
        renderer.build_portable_pdf(pngs, pdf, title=content["cover"]["title"])
        renderer._verify_export(pdf, ".pdf", expected_slide_count=len(pages))
        if preview_dir is not None:
            preview_dir.mkdir(parents=True, exist_ok=True)
            for png in pngs:
                shutil.copyfile(png, preview_dir / png.name)
        os.replace(pdf, output)


def main() -> int:
    """Validate reviewed inputs, then optionally build the portfolio PDF."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--content", type=Path, default=ROOT / "docs/aelab-overview.json")
    parser.add_argument("--output", type=Path, default=ROOT / "AELab25_Modern_UI_Overview.pdf")
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    try:
        root = args.source_root.resolve()
        if args.output.suffix.lower() != ".pdf":
            raise ValueError("--output must name a .pdf file.")
        for target in (args.output, args.preview_dir):
            if target is not None and target.resolve().is_relative_to(root):
                raise ValueError("Outputs must be outside the read-only AELab source repository.")
        content = json.loads(args.content.read_text(encoding="utf-8"))
        validate_source(root, content)
        renderer = load_renderer(root)
        renderer.LONG_DISCLAIMER = ("Feature descriptions reflect reviewed source code. The cover image and reference "
                                    "gallery are archived captures. No licensed simulator run or engineering validation "
                                    "was performed for this update.")
        pages = build_slides(renderer, root, content)
        for slide, _ in pages:
            validate_layout(renderer, slide)
        if not args.check_inputs:
            render_pdf(renderer, pages, content, args.output, args.preview_dir)
        print(f"{'Validated' if args.check_inputs else 'Built'} {len(pages)} pages; source {content['source']['revision']}.")
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Overview build failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
