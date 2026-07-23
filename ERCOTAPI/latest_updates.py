"""Build and load a compact public feed of newly archived ERCOT documents."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EXCLUDED_SOURCES = {"MARKET NOTICES", "PUBLIC NOTICES"}
EXCLUDED_URL_FRAGMENTS = (
    "/services/comm/mkt_notices/",
    "/marketnotice/",
)
REVISION_REQUEST_NAMES = {
    "NPRR": "Nodal Protocol Revision Request",
    "PGRR": "Planning Guide Revision Request",
    "NOGRR": "Nodal Operating Guide Revision Request",
    "OBDRR": "Other Binding Document Revision Request",
    "RRGRR": "Resource Registration Glossary Revision Request",
    "VCMRR": "Verifiable Cost Manual Revision Request",
    "COPMGRR": "Commercial Operations Market Guide Revision Request",
    "LPGRR": "Load Profiling Guide Revision Request",
    "RMGRR": "Retail Market Guide Revision Request",
    "SMOGRR": "Settlement Metering Operating Guide Revision Request",
    "CMGRR": "Competitive Metering Guide Revision Request",
    "SCR": "System Change Request",
}
REVISION_REQUEST_FAMILIES = tuple(REVISION_REQUEST_NAMES)
_REVISION_FAMILY_PATTERN = "|".join(
    sorted(REVISION_REQUEST_FAMILIES, key=len, reverse=True)
)
_ISSUE_URL_RE = re.compile(
    rf"/mktrules/issues/(?P<family>{_REVISION_FAMILY_PATTERN})"
    r"[-_ ]?(?P<number>\d{1,6})\b",
    re.IGNORECASE,
)
_REVERSE_REVISION_RE = re.compile(
    rf"\b(?P<number>\d{{1,6}})(?P<family>{_REVISION_FAMILY_PATTERN})"
    r"(?=[-_\s.]|$)",
    re.IGNORECASE,
)
_FORWARD_REVISION_RE = re.compile(
    rf"\b(?P<family>{_REVISION_FAMILY_PATTERN})[-_ ]?(?P<number>\d{{1,6}})\b",
    re.IGNORECASE,
)
EXCLUDED_NAVIGATION_TITLES = {
    "hide emil information show emil information",
    "mis log in",
    "nodal protocol revision requests",
    "nprr submission process",
    "protocol interpretation request submission process",
    "protocol library - nodal",
    "workshops",
}


def revision_request_identity(
    document_number: str = "",
    title: str = "",
    url: str = "",
) -> tuple[str, str] | None:
    """Return ``(revision_id, family)`` without confusing attachment numbers."""

    issue_match = _ISSUE_URL_RE.search(url)
    if issue_match:
        family = issue_match.group("family").upper()
        return f"{family}{issue_match.group('number')}", family

    for candidate in (document_number, title, url):
        reverse_match = _REVERSE_REVISION_RE.search(candidate)
        if reverse_match:
            family = reverse_match.group("family").upper()
            return f"{family}{reverse_match.group('number')}", family
        forward_match = _FORWARD_REVISION_RE.search(candidate)
        if forward_match:
            family = forward_match.group("family").upper()
            return f"{family}{forward_match.group('number')}", family
    return None


def _explanation(source: str, title: str, document_number: str, url: str = "") -> str:
    source_upper = source.upper()
    revision = revision_request_identity(document_number, title, url)
    if revision:
        revision_id, prefix = revision
        return (
            f"New {REVISION_REQUEST_NAMES[prefix]} material. It may change or clarify ERCOT "
            f"requirements; search the assistant for {revision_id} to review details."
        )
    if source_upper in {"DWG", "SSWG"}:
        return "New working-group material relevant to dynamic or steady-state modeling and study procedures."
    if source_upper in {"PLANNING GUIDE", "OPERATING GUIDE", "PROTOCOLS"}:
        return f"New {source.lower()} material that may update an ERCOT requirement or implementation detail."
    return "New ERCOT technical material now available for cited search and explanation in the assistant."


def build_latest_updates(
    paths: Iterable[Path],
    *,
    output_path: Path,
    minimum_year: int = 2026,
) -> dict[str, Any]:
    unique: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file() or path.name.endswith(".metadata.json"):
            continue
        metadata_path = path.with_name(f"{path.name}.metadata.json")
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        source = str(metadata.get("source_label") or path.parent.parent.name).strip()
        if source.upper() in EXCLUDED_SOURCES:
            continue
        original_url = str(metadata.get("original_url") or metadata.get("final_url") or "")
        if any(fragment in original_url.lower() for fragment in EXCLUDED_URL_FRAGMENTS):
            continue
        published = str(metadata.get("published_date") or metadata.get("published_hint") or "")
        published_years = [int(value) for value in re.findall(r"\b(20\d{2})\b", published)]
        path_years = [int(value) for value in re.findall(r"\b(20\d{2})\b", path.as_posix())]
        years = published_years or path_years
        if not years or max(years) < minimum_year:
            continue
        content_hash = str(metadata.get("content_sha256") or "")
        if not content_hash:
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        title = str(metadata.get("title") or path.name).strip()
        if title.casefold() in EXCLUDED_NAVIGATION_TITLES:
            continue
        number = str(metadata.get("document_number") or "").strip()
        revision = revision_request_identity(number, title, original_url)
        record = unique.setdefault(
            content_hash,
            {
                "content_sha256": content_hash,
                "title": title,
                "document_number": number,
                "source": source,
                "published_date": published,
                "downloaded_at": str(metadata.get("downloaded_at") or ""),
                "status": str(metadata.get("document_status") or ""),
                "url": original_url,
                "sources": [],
                "revision_id": revision[0] if revision else "",
                "revision_family": revision[1] if revision else "",
            },
        )
        if source and source not in record["sources"]:
            record["sources"].append(source)

    items = list(unique.values())
    for item in items:
        item["sources"].sort()
        item["explanation"] = _explanation(
            item["source"], item["title"], item["document_number"], item["url"]
        )
    items.sort(
        key=lambda item: (item["downloaded_at"], item["published_date"], item["title"]),
        reverse=True,
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "minimum_year": minimum_year,
        "count": len(items),
        "items": items,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def load_latest_updates(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"count": 0, "items": []}
    return value if isinstance(value, dict) else {"count": 0, "items": []}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-file", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("latest_ercot_updates.json"),
    )
    args = parser.parse_args()
    paths = [
        Path(line.strip())
        for line in args.path_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    payload = build_latest_updates(paths, output_path=args.output)
    print(json.dumps({"output": str(args.output), "count": payload["count"]}))


if __name__ == "__main__":
    main()
