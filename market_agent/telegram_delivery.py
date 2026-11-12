"""Preserve complete plain-text reports within Telegram's message limit."""

from __future__ import annotations


def split_telegram_text(text: str, limit: int = 3900) -> list[str]:
    """Split at line boundaries when possible, counting UTF-16 code units."""
    if limit < 2:
        raise ValueError("Telegram chunk limit must be at least two code units")
    chunks: list[str] = []
    remaining = text
    while remaining:
        units = 0
        end = 0
        for character in remaining:
            size = 2 if ord(character) > 0xFFFF else 1
            if units + size > limit:
                break
            units += size
            end += 1
        if end < len(remaining):
            newline = remaining.rfind("\n", 0, end)
            if newline >= 0:
                end = newline + 1
        chunks.append(remaining[:end])
        remaining = remaining[end:]
    return chunks
