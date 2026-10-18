"""Inspect extraction coverage offline without publishing or embedding documents."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .loaders import DocumentLoadError, load_document_with_report


def audit_paths(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Report each explicit file independently; failed extraction is never hidden."""

    results = []
    for path in paths:
        record: dict[str, Any] = {"path": str(path)}
        try:
            loaded = load_document_with_report(path)
            record.update({
                "status": "review_required" if loaded.extraction_report["review_required"] else "text_extracted",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "characters": len(loaded.text),
                "extraction_report": loaded.extraction_report,
            })
        except (DocumentLoadError, OSError) as exc:
            record.update({"status": "error", "error": str(exc)})
        results.append(record)
    return results


def main(argv: Sequence[str] | None = None) -> None:
    """Print extraction observations; never change the saved retrieval generation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    results = audit_paths(args.paths)
    print(json.dumps({"files": results, "index_modified": False}, indent=2))
    if any(item["status"] == "error" for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
