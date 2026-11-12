"""Readable report dates without changing the underlying market-session labels."""

from __future__ import annotations

import datetime as dt
import os
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def format_session_date(value: object) -> str:
    """Format an ISO market-session date without applying a timezone shift."""
    try:
        session = dt.date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return "unavailable"
    return f"{session:%b} {session.day}, {session.year}"


def format_report_timestamp(value: dt.datetime) -> str:
    """Display an aware timestamp in the configured report timezone."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Report timestamp must include a timezone")
    timezone_name = os.environ.get("MARKET_REPORT_TIMEZONE", "America/Chicago")
    try:
        timezone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"Invalid MARKET_REPORT_TIMEZONE: {timezone_name}") from exc
    local = value.astimezone(timezone)
    hour = local.hour % 12 or 12
    offset = local.strftime("%z")
    readable_offset = f"UTC{offset[:3]}:{offset[3:]}"
    return (
        f"{local:%b} {local.day}, {local.year} at "
        f"{hour}:{local.minute:02d} {local:%p %Z} ({readable_offset})"
    )
