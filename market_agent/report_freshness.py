"""Deterministic freshness selection for generated market reports."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Iterable, Mapping, TypeVar


_TIMESTAMP = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})[T ]"
    r"(?P<hour>\d{2})[:-](?P<minute>\d{2})"
    r"(?:(?:[:-])(?P<second>\d{2}))?"
    r"(?:[.:-]\d+)?(?:Z|\s+UTC)?",
    re.IGNORECASE,
)
_MINIMUM_TIMESTAMP = datetime.min.replace(tzinfo=timezone.utc)
_Payload = TypeVar("_Payload", bound=Mapping[str, object])


def parse_report_timestamp(value: object) -> datetime | None:
    """Parse the timestamp formats emitted in report bodies and filenames."""

    match = _TIMESTAMP.search(str(value or ""))
    if match is None:
        return None
    try:
        return datetime.fromisoformat(
            f"{match.group('date')}T{match.group('hour')}:"
            f"{match.group('minute')}:{match.group('second') or '00'}+00:00"
        )
    except ValueError:
        return None


def report_text_timestamp(report_text: str) -> datetime | None:
    """Return the explicit generated timestamp from a text report."""

    for line in report_text.splitlines():
        if line.strip().lower().startswith("generated:"):
            return parse_report_timestamp(line)
    return None


def _newest_known_timestamp(*values: datetime | None) -> datetime:
    return max(
        (value for value in values if value is not None),
        default=_MINIMUM_TIMESTAMP,
    )


def newest_timestamped_path(
    candidates: Iterable[str | Path],
) -> str | Path | None:
    """Return the path carrying the newest explicit report timestamp."""

    scored = [
        (parse_report_timestamp(Path(str(path)).name), str(path), path)
        for path in candidates
    ]
    known = [item for item in scored if item[0] is not None]
    if not known:
        return None
    return max(known, key=lambda item: item[:2])[2]


def report_path_is_newer(
    path: str | Path | None,
    current_timestamp: datetime | None,
) -> bool:
    """Return whether a timestamped path is newer than the current report."""

    timestamp = parse_report_timestamp(Path(str(path)).name) if path else None
    return timestamp is not None and (
        current_timestamp is None or timestamp > current_timestamp
    )


def newest_text_report_candidate(
    candidates: Iterable[tuple[str | Path, str]],
) -> tuple[datetime, str] | None:
    """Return the timestamp and content of the freshest text candidate."""

    scored: list[tuple[datetime, str, str]] = []
    for path, report_text in candidates:
        if not report_text.strip():
            continue
        path_text = str(path)
        timestamp = _newest_known_timestamp(
            report_text_timestamp(report_text),
            parse_report_timestamp(Path(path_text).name),
        )
        scored.append((timestamp, path_text, report_text))
    if not scored:
        return None
    timestamp, _, report_text = max(scored, key=lambda item: item[:2])
    return timestamp, report_text


def newest_text_report(
    candidates: Iterable[tuple[str | Path, str]],
) -> str | None:
    """Select text by generated timestamp, never filesystem modification time."""

    selected = newest_text_report_candidate(candidates)
    return selected[1] if selected is not None else None


def newest_json_payload_candidate(
    candidates: Iterable[tuple[str | Path, _Payload]],
) -> tuple[datetime, _Payload] | None:
    """Return the timestamp and value of the freshest JSON candidate."""

    scored: list[tuple[datetime, str, _Payload]] = []
    for path, payload in candidates:
        path_text = str(path)
        timestamp = _newest_known_timestamp(
            parse_report_timestamp(payload.get("generated_at")),
            parse_report_timestamp(Path(path_text).name),
        )
        scored.append((timestamp, path_text, payload))
    if not scored:
        return None
    timestamp, _, payload = max(scored, key=lambda item: item[:2])
    return timestamp, payload


def newest_json_payload(
    candidates: Iterable[tuple[str | Path, _Payload]],
) -> _Payload | None:
    """Select a JSON payload by its generated timestamp or timestamped filename."""

    selected = newest_json_payload_candidate(candidates)
    return selected[1] if selected is not None else None
