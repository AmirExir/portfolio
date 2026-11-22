"""Select and describe completed-session price observations for market reports."""

from __future__ import annotations

import math

if __package__:
    from .agent import price_watch as rules
    from .report_dates import format_session_date
else:
    from agent import price_watch as rules
    from report_dates import format_session_date


def is_usable_observation(item: object) -> bool:
    """Check fields needed to render a saved price observation safely."""
    if not isinstance(item, dict) or not isinstance(item.get("symbol"), str) or not item["symbol"].strip():
        return False
    if format_session_date(item.get("as_of_session")) == "unavailable":
        return False
    for field in ("shock", "breakdown", "entry_watch", "entry_triggered"):
        if not isinstance(item.get(field), bool):
            return False
    for field in ("close", "prior_low", "prior_high", "change_pct", "shock_threshold_pct"):
        number = item.get(field)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            return False
    return (
        item["close"] > 0 and 0 < item["prior_low"] <= item["prior_high"]
        and item["shock_threshold_pct"] > 0
    )


def is_price_only_report(payload: dict) -> bool:
    """Accept empty forecasts only for a complete run with attributable, fresh observations."""
    if (
        payload.get("run_complete") is not True or payload.get("rows") != []
        or type(payload.get("ranked_count")) is not int or payload["ranked_count"] != 0
    ):
        return False
    watch = payload.get("price_watch")
    symbols = payload.get("symbols")
    if not isinstance(watch, dict) or not isinstance(symbols, list):
        return False
    if not all(isinstance(symbol, str) and symbol.strip() for symbol in symbols):
        return False
    observations = watch.get("observations")
    if not isinstance(observations, list) or not observations:
        return False
    universe = set(symbols)
    counts = (
        (payload.get("universe_count"), len(universe)),
        (watch.get("universe_count"), len(universe)),
        (watch.get("observed_count"), len(observations)),
    )
    if len(universe) != len(symbols) or any(type(count) is not int or count != expected for count, expected in counts):
        return False
    for item in observations:
        if (
            not is_usable_observation(item) or item["symbol"] not in universe
            or type(item.get("lag_sessions")) is not int or item["lag_sessions"] != 0
        ):
            return False
    sessions = sorted({item["as_of_session"] for item in observations})
    unique = {item["symbol"] for item in observations}
    return (
        len(unique) == len(observations)
        and watch.get("as_of_sessions") == sessions
        and payload.get("as_of_sessions") == sessions
    )


def has_scheduled_report_data(payload: object) -> bool:
    """Select completed price-only runs and preserve ranked reports from older formats."""
    if not isinstance(payload, dict) or payload.get("run_complete", True) is not True:
        return False
    rows = payload.get("rows")
    return isinstance(rows, list) and (bool(rows) or is_price_only_report(payload))


def _next_levels(item: dict) -> tuple[float, float] | None:
    low, high = item.get("next_low"), item.get("next_high")
    if all(
        isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)
        for value in (low, high)
    ) and 0 < low <= item["close"] <= high:
        return low, high
    return None


def _entry_priority(item: dict) -> tuple[bool, float, str]:
    levels = _next_levels(item)
    high = item["prior_high"] if item["entry_triggered"] or levels is None else levels[1]
    return not item["entry_triggered"], abs(item["close"] / high - 1), item["symbol"]


def build_price_watch(observations: list[dict], *, universe_count: int, limit: int) -> dict:
    """Rank observed downside and entry conditions separately. Zero limit shows all.

    Callers provide at most one freshness-checked observation per configured
    symbol. Dual downside triggers rank first, then declines relative to each
    symbol's own shock threshold. Breakouts rank ahead of pending entries,
    with prices closest to the breakout level first within each group.
    """
    if universe_count < 0 or limit < 0:
        raise ValueError("Universe count and price-watch limit must be nonnegative")
    observed = [item for item in observations if is_usable_observation(item)]
    by_symbol = {item["symbol"]: item for item in observed}
    if len(by_symbol) != len(observed) or len(observed) > universe_count:
        raise ValueError("Price watch requires one observation per configured symbol")
    downside = sorted(
        (item for item in observed if item["shock"] or item["breakdown"]),
        key=lambda item: (
            not (item["shock"] and item["breakdown"]),
            item["change_pct"] / item["shock_threshold_pct"],
            item["symbol"],
        ),
    )
    entries = sorted(
        (item for item in observed if item["entry_watch"] and not (item["shock"] or item["breakdown"])),
        key=_entry_priority,
    )
    return {
        "rules": {
            "version": "closing-price-watch-v1",
            "range_sessions": rules.RANGE_SESSIONS,
            "momentum_sessions": rules.MOMENTUM_SESSIONS,
            "shock_floor_pct": rules.MIN_SHOCK_PCT,
            "shock_volatility_multiplier": rules.SHOCK_VOL_MULTIPLIER,
            "entry_proximity_pct": rules.ENTRY_PROXIMITY_PCT,
        },
        "universe_count": universe_count,
        "observed_count": len(observed),
        "as_of_sessions": sorted({item["as_of_session"] for item in observed}),
        "observations": observed,
        "downside_count": len(downside),
        "entry_count": len(entries),
        "downside_watch": downside[:limit or None],
        "entry_watch": entries[:limit or None],
        "benchmark": by_symbol.get("SPY"),
    }


def _formatted_levels(item: dict) -> list[str]:
    values = [item[field] for field in ("close", "prior_low", "prior_high")]
    levels = _next_levels(item)
    if levels is not None:
        values.extend(levels)
    # Keep four significant digits for sub-dollar instruments. Add precision
    # when two distinct closing levels would otherwise display identically.
    precision = max(2, 3 - math.floor(math.log10(min(values)))) if min(values) < 1 else 2
    unique = set(values)
    while len({round(value, precision) for value in unique}) != len(unique):
        precision += 1
    return [f"{value:,.{precision}f}" for value in values]


def format_watch_prices(item: dict) -> tuple[str, str, str]:
    """Format close, prior low and prior high, retaining distinct price levels."""
    close, low, high = _formatted_levels(item)[:3]
    return close, low, high


def format_next_watch_prices(item: dict) -> tuple[str, str] | None:
    """Format the range for the next close when the saved observation supplies it."""
    levels = _formatted_levels(item)
    return (levels[3], levels[4]) if len(levels) == 5 else None


def _downside_line(item: dict) -> str:
    close, low, high = format_watch_prices(item)
    line = (
        f"{item['symbol']} ({format_session_date(item['as_of_session'])}) "
        f"closed at {close} ({item['change_pct']:+.2f}%)."
    )
    if item["shock"]:
        line += f" Loss reached the {item['shock_threshold_pct']:.2f}% alert threshold."
    if item["breakdown"]:
        return f"{line} Already below {low}. Recovery needs a close back above that level."
    levels = format_next_watch_prices(item)
    if levels is None:
        return f"{line} Prior 20-session closing range {low}–{high}. Next-close levels unavailable."
    return f"{line} Watch for a close below {levels[0]}. Recheck for a breakout on a close above {levels[1]}."


def _entry_line(item: dict) -> str:
    close, low, high = format_watch_prices(item)
    line = f"{item['symbol']} ({format_session_date(item['as_of_session'])}) closed at {close}."
    if item["entry_triggered"]:
        return (
            f"{line} Already above the prior 20-session closing high of {high}. "
            f"Watch the next close for follow-through. A close back below {high} invalidates the breakout."
        )
    levels = format_next_watch_prices(item)
    if levels is None:
        return f"{line} Prior 20-session closing range {low}–{high}. Next-close levels unavailable."
    return (
        f"{line} Watch for an entry on a close above {levels[1]}, the latest 20-session "
        f"closing high. A close below {levels[0]} invalidates the setup."
    )


def format_price_watch(plan: dict) -> list[str]:
    """Render a structured watch plan as plain-text lines, including observed coverage."""
    observed = plan["observed_count"]
    universe = plan["universe_count"]
    symbols = "symbol" if universe == 1 else "symbols"
    lines = ["COMPLETED-SESSION PRICE WATCH"]
    if not observed:
        return [*lines, f"Price watch unavailable. Usable closing history for 0 of {universe} {symbols}."]
    sessions = plan["as_of_sessions"]
    dates = format_session_date(sessions[0])
    if len(sessions) > 1:
        dates += f" to {format_session_date(sessions[-1])}"
    coverage = f"Usable closing history for {observed} of {universe} {symbols} ({dates})."
    if observed < universe:
        coverage += f" Price conditions unavailable for {universe - observed}."
    lines.append(coverage)
    benchmark = plan["benchmark"]
    if benchmark:
        close, low, high = format_watch_prices(benchmark)
        move = (
            f"SPY benchmark ({format_session_date(benchmark['as_of_session'])}) "
            f"{benchmark['change_pct']:+.2f}% in one session, closing at {close}."
        )
        if benchmark["breakdown"]:
            move += (
                f" Already below its prior 20-session closing low of {low}. "
                "Watch for a close back above it."
            )
        else:
            if benchmark["shock"]:
                move += f" Loss reached its {benchmark['shock_threshold_pct']:.2f}% alert threshold."
            levels = format_next_watch_prices(benchmark)
            if levels is None:
                move += f" Prior 20-session closing range {low}–{high}. Next-close levels unavailable."
            else:
                move += f" Watch for a close below its latest 20-session closing low of {levels[0]}."
                if not (benchmark["shock"] or benchmark["entry_triggered"]):
                    move += f" Its closing high over that window is {levels[1]}."
        lines.append(move)
    for heading, items, total, render in (
        ("Exit / avoid watch", plan["downside_watch"], plan["downside_count"], _downside_line),
        ("Entry watch", plan["entry_watch"], plan["entry_count"], _entry_line),
    ):
        lines.append("")
        if total == 0:
            lines.append(f"{heading} — no triggers in this scan.")
            continue
        count = f"{len(items)} of {total} shown" if len(items) < total else str(total)
        lines.append(f"{heading} ({count})")
        if heading == "Exit / avoid watch":
            lines.append(
                "If held, review an exit on a break of the 20-session closing low. "
                "For a new entry, watch the recovery or breakout level shown."
            )
        lines.extend(render(item) for item in items)
    if universe <= 5:
        neutral = [
            item for item in plan["observations"]
            if item["symbol"] != "SPY" and not (item["shock"] or item["breakdown"] or item["entry_watch"])
        ]
        if neutral:
            lines.extend(["", "Next closing levels for symbols without a trigger"])
            for item in neutral:
                close, low, high = format_watch_prices(item)
                line = f"{item['symbol']} ({format_session_date(item['as_of_session'])}) closed at {close}."
                levels = format_next_watch_prices(item)
                if levels is None:
                    line += f" Prior 20-session closing range {low}–{high}. Next-close levels unavailable."
                else:
                    line += f" Review downside below {levels[0]} or recheck for a breakout above {levels[1]} on a close."
                lines.append(line)
    return lines
