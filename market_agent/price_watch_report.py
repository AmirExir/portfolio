"""Select and describe completed-session price observations for market reports."""

from __future__ import annotations

import math
from collections.abc import Sequence

if __package__:
    from .agent import price_watch as rules
    from .agent.technicals import (
        DOUBLE_BOTTOM_LABEL, DOUBLE_TOP_LABEL, PATTERN_SESSIONS, RSI_OVERBOUGHT, RSI_OVERSOLD, RSI_PERIOD,
    )
    from .report_dates import format_session_date
else:
    from agent import price_watch as rules
    from agent.technicals import (
        DOUBLE_BOTTOM_LABEL, DOUBLE_TOP_LABEL, PATTERN_SESSIONS, RSI_OVERBOUGHT, RSI_OVERSOLD, RSI_PERIOD,
    )
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


def _format_prices(values: list[float]) -> list[str]:
    # Keep four significant digits for sub-dollar instruments. Add precision
    # when two distinct closing levels would otherwise display identically.
    precision = max(2, 3 - math.floor(math.log10(min(values)))) if min(values) < 1 else 2
    unique = set(values)
    while len({round(value, precision) for value in unique}) != len(unique):
        precision += 1
    return [f"{value:,.{precision}f}" for value in values]


def _formatted_levels(item: dict) -> list[str]:
    values = [item[field] for field in ("close", "prior_low", "prior_high")]
    return _format_prices([*values, *(_next_levels(item) or ())])


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


def _finite_number(value: object) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _format_chart_pattern(name: str, close: float, necklines: dict) -> str:
    level = necklines.get(name)
    if not _finite_number(level) or level <= 0:
        return name
    if name == DOUBLE_TOP_LABEL and close < level:
        direction = "below"
    elif name == DOUBLE_BOTTOM_LABEL and close > level:
        direction = "above"
    else:
        return name
    return f"{name} (closed {direction} neckline {_format_prices([close, level])[1]})"


def _technical_row(item: dict) -> list[str]:
    """Describe one completed close with its own date, indicators and confirmed swing levels."""
    data = item["technical"]
    trend = data.get("trend")
    windows = data.get("trend_windows") or []
    if trend in ("Uptrend", "Downtrend", "Mixed"):
        trend_text = "Mixed trend" if trend == "Mixed" else trend
        if isinstance(windows, (list, tuple)) and windows:
            trend_text += f" (SMA {'/'.join(map(str, windows))})"
    else:
        trend_text = "Trend unavailable"
    rsi = data.get("rsi")
    state = data.get("rsi_state")
    momentum = "RSI unavailable"
    if _finite_number(rsi) and 0 <= rsi <= 100 and state in ("Oversold", "Overbought", "Neutral"):
        displayed = f"{rsi:.1f}"
        if state == "Oversold" and float(displayed) == RSI_OVERSOLD:
            displayed = f"just below {RSI_OVERSOLD:g}"
        elif state == "Overbought" and float(displayed) == RSI_OVERBOUGHT:
            displayed = f"just above {RSI_OVERBOUGHT:g}"
        momentum = f"RSI({RSI_PERIOD}) {displayed} ({state.lower()})"
    first_line = f"{item['symbol']} ({format_session_date(item['as_of_session'])}) | {trend_text} | {momentum}"
    patterns = data.get("patterns")
    close = item["close"]
    if isinstance(patterns, list) and patterns and all(isinstance(name, str) for name in patterns):
        necklines = data.get("shape_necklines")
        if isinstance(necklines, dict):
            patterns = [_format_chart_pattern(name, close, necklines) for name in patterns]
        first_line += " | " + ", ".join(patterns)
    support, resistance = data.get("support"), data.get("resistance")
    levels = {}
    if _finite_number(support) and 0 < support < close:
        levels["Support"] = support
    if _finite_number(resistance) and resistance > close:
        levels["Resistance"] = resistance
    formatted = dict(zip(levels, _format_prices([close, *levels.values()])[1:]))
    return [
        first_line,
        f"  Support {formatted.get('Support', 'unconfirmed below')} | "
        f"Resistance {formatted.get('Resistance', 'unconfirmed above')}",
    ]


def _technical_lines(plan: dict, forecast_symbols: Sequence[str]) -> list[str]:
    observations = [
        item for item in plan.get("observations", [])
        if isinstance(item.get("technical"), dict) and item["technical"]
    ]
    if not observations:
        return []
    by_symbol = {item["symbol"]: item for item in observations}
    selected = []
    if plan.get("benchmark"):
        selected.append(plan["benchmark"]["symbol"])
    selected.extend(item["symbol"] for item in [*plan["downside_watch"], *plan["entry_watch"]])
    selected.extend(
        item["symbol"] for item in observations
        if plan["universe_count"] <= 5 or item.get("report_symbol", item["symbol"]) in forecast_symbols
    )
    shapes = [
        item["symbol"] for item in observations
        if isinstance(item["technical"].get("patterns"), list)
        and (DOUBLE_TOP_LABEL in item["technical"]["patterns"] or DOUBLE_BOTTOM_LABEL in item["technical"]["patterns"])
    ]
    selected.extend(shapes)
    measured = [
        item for item in observations if _finite_number(item["technical"].get("rsi"))
        and 0 <= item["technical"]["rsi"] <= 100
    ]
    measured.sort(key=lambda item: (item["technical"]["rsi"], item["symbol"]))
    oversold = [item for item in measured if item["technical"].get("rsi_state") == "Oversold"]
    overbought = [item for item in measured if item["technical"].get("rsi_state") == "Overbought"]
    if oversold:
        selected.append(oversold[0]["symbol"])
    if overbought:
        selected.append(overbought[-1]["symbol"])
    selected = list(dict.fromkeys(symbol for symbol in selected if symbol in by_symbol))
    if not selected:
        pattern_matches = [item for item in observations if item["technical"].get("patterns")]
        selected = [item["symbol"] for item in (pattern_matches or observations)[:3]]
    lines = ["", "TECHNICAL READINGS"]
    if not shapes:
        lines.append("No new confirmed double top or double bottom in the available closes.")
    if measured:
        noun = "symbol" if len(measured) == 1 else "symbols"
        lines.append(
            f"RSI({RSI_PERIOD}) across {len(measured)} {noun} | {len(oversold)} oversold "
            f"(below {RSI_OVERSOLD:g}) | {len(overbought)} overbought (above {RSI_OVERBOUGHT:g})"
        )
    lines.append(
        f"Support and resistance use the nearest confirmed swing closes from up to {PATTERN_SESSIONS} sessions."
    )
    for symbol in selected:
        lines.extend(_technical_row(by_symbol[symbol]))
    return lines


def format_price_watch(plan: dict, *, forecast_symbols: Sequence[str] = ()) -> list[str]:
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
    lines.extend(_technical_lines(plan, forecast_symbols))
    return lines
