"""Poll publisher headlines and run the existing quick market analysis for new events."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

from market_agent.publisher_headlines import HeadlineCheck, PublisherHeadline, check_publisher_headlines
from market_agent.breaking_news_sources import BREAKING_NEWS_FEEDS
from market_agent.report_freshness import parse_report_timestamp

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "reports" / "breaking_analysis"
MAX_AGE = timedelta(hours=2)
COOLDOWN = timedelta(minutes=30)
RETRY_DELAY = timedelta(minutes=15)
MAX_ATTEMPTS = 2
TRUSTED_PUBLISHERS = {name for name, _ in BREAKING_NEWS_FEEDS}

# Explicit headline aliases avoid expensive entity inference or treating ordinary
# words such as "C" or "V" as an exchange ticker.
COMPANIES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "AMD": "Advanced Micro Devices",
    "INTC": "Intel", "AVGO": "Broadcom", "MU": "Micron", "WDC": "Western Digital",
    "STX": "Seagate", "SNDK": "SanDisk", "GOOGL": "Alphabet|Google", "META": "Meta Platforms|Meta",
    "AMZN": "Amazon", "TSLA": "Tesla", "RIOT": "Riot Platforms", "GEV": "GE Vernova",
    "SPCX": "SpaceX|Space Exploration Technologies", "CRWV": "CoreWeave", "IREN": "IREN Limited",
    "NBIS": "Nebius", "ORCL": "Oracle", "NFLX": "Netflix", "JPM": "JPMorgan|JP Morgan",
    "BAC": "Bank of America", "WFC": "Wells Fargo", "C": "Citigroup", "GS": "Goldman Sachs",
    "MS": "Morgan Stanley", "V": "Visa", "UNH": "UnitedHealth", "LLY": "Eli Lilly",
    "JNJ": "Johnson & Johnson", "WMT": "Walmart", "PG": "Procter & Gamble", "KO": "Coca-Cola",
    "HD": "Home Depot", "PEP": "PepsiCo", "DELL": "Dell", "DAL": "Delta Air Lines",
    "UAL": "United Airlines", "AAL": "American Airlines", "LUV": "Southwest Airlines",
    "XOM": "ExxonMobil|Exxon", "CVX": "Chevron", "COP": "ConocoPhillips",
    "OXY": "Occidental", "SLB": "Schlumberger", "EOG": "EOG Resources",
}
_NAME_PATTERNS = {symbol: re.compile(r"\b(?:" + names + r")\b", re.I) for symbol, names in COMPANIES.items()}
_SPECULATION = re.compile(
    r"\b(?:might|could|should|would|must|rumou?rs?|reportedly|in talks|consider(?:s|ing)?|"
    r"expected to|calls? (?:on|for)|urges?|seeks?|predicts?|denies|denied|not to|will not|"
    r"has not|hasn't|did not|didn't|unlikely|hopes?|bets? on|last (?:year|month|week))\b|"
    r"\bmay (?:soon |also )?(?:be|cut|raise|hike|lower|appoint|resign|return|rehire|"
    r"reinstate|name|buy|acquire|file|merge|beat|miss|sell)\b", re.I,
)
_ROLE = r"(?:CEO|chief executive(?: officer)?)"
_AUTHORITY = r"\b(?:federal reserve|fed|fomc)\s+(?:(?:has|unexpectedly|unanimously|just)\s+|votes? to |voted to )*"
_ACTIONS = (
    ("ceo_appointment", re.compile(
        r"\b(?:reinstat(?:es?|ed)|rehir(?:es?|ed)|reappoint(?:s|ed)?|appoint(?:s|ed)?|names?|named)"
        rf"\b.{{0,85}}?\b{_ROLE}\b|\b(?:returns?|back) as (?:the )?{_ROLE}\b|"
        rf"\b{_ROLE}\b.{{0,40}}?\b(?:reinstated|rehired|reappointed)\b", re.I)),
    ("ceo_departure", re.compile(
        rf"\b{_ROLE}\b.{{0,55}}?\b(?:resigns?|resigned|steps? down|stepped down|is ousted|is fired|departs?)\b|"
        rf"\b(?:resigns?|resigned|steps? down|stepped down) as (?:the )?{_ROLE}\b|"
        rf"\b(?:ousts?|fires?)\b.{{0,50}}?\b{_ROLE}\b", re.I)),
    ("bankruptcy", re.compile(r"\b(?:files? for (?:bankruptcy|chapter 11)|filed for (?:bankruptcy|chapter 11)|"
                              r"declares? bankruptcy|enters? bankruptcy)\b", re.I)),
    ("acquisition", re.compile(r"\b(?:acquires?|acquired by|agrees? to (?:buy|acquire)|to buy .+ for|"
                               r"agrees? to merge|approve[sd]? .{0,50}merger|announces? .{0,40}merger)\b", re.I)),
    ("outlook_cut", re.compile(r"\b(?:cuts?|cutting|lowers?|slashes?|withdraws?) .{0,35}\b(?:outlook|guidance|forecast)\b", re.I)),
    ("outlook_raise", re.compile(r"\b(?:raises?|lifts?|hikes?) .{0,35}\b(?:outlook|guidance|forecast)\b", re.I)),
    ("earnings_beat", re.compile(r"\b(?:earnings beat|beats? (?:earnings |profit )?(?:estimates|expectations))\b", re.I)),
    ("earnings_miss", re.compile(r"\b(?:earnings miss|miss(?:es)? (?:earnings |profit )?(?:estimates|expectations))\b", re.I)),
)
_FED_CUT = re.compile(_AUTHORITY + r"(?:cuts?|lowers?|lowered)\b.{0,28}\brates?\b", re.I)
_FED_HIKE = re.compile(_AUTHORITY + r"(?:raises?|raised|hikes?|hiked)\b.{0,28}\brates?\b", re.I)


def classify_headline(item: PublisherHeadline, now: datetime) -> dict[str, Any] | None:
    """Identify a recent reported action at a covered company or the Federal Reserve."""
    if item.publisher not in TRUSTED_PUBLISHERS or not timedelta(0) <= now - item.published_at <= MAX_AGE:
        return None
    title = item.title
    for clause in re.findall(r"[^.!?;—]+[.!?;—]?", title):
        if clause.endswith("?") or _SPECULATION.search(clause):
            continue
        symbols = sorted(symbol for symbol, pattern in _NAME_PATTERNS.items() if pattern.search(clause) or (
            len(symbol) >= 3 and re.search(r"\b" + symbol + r"\b", clause)
        ))
        if _FED_CUT.search(clause):
            category, symbols = "fed_rate_cut", []
            break
        if _FED_HIKE.search(clause):
            category, symbols = "fed_rate_hike", []
            break
        if symbols:
            category = next((name for name, pattern in _ACTIONS if pattern.search(clause)), None)
            if category:
                break
    else:
        return None
    return {"category": category, "symbols": symbols, "title": title, "source": item.publisher,
            "url": item.url, "published_at": item.published_at.astimezone(timezone.utc).isoformat()}


def _same_event(first: dict, second: dict) -> bool:
    # A publisher may reuse its live coverage URL for a new board decision.
    if first["category"] != second["category"] or first["symbols"] != second["symbols"]:
        return False
    gap = abs(datetime.fromisoformat(first["published_at"]) - datetime.fromisoformat(second["published_at"]))
    return first["url"] == second["url"] or gap <= timedelta(hours=12)


def _read_state(directory: Path) -> dict:
    state_path = directory / "state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"version": 1, "latest": None, "pending": None, "queue": [], "receipts": []}
    if not isinstance(state, dict) or state.get("version") != 1 or not {"latest", "pending", "queue", "receipts"} <= state.keys():
        raise ValueError("Unrecognized breaking-news analysis state")
    return state


def _save_state(directory: Path, state: dict) -> None:
    with NamedTemporaryFile(mode="w", dir=directory, delete=False, encoding="utf-8") as file:
        path = Path(file.name)
        try:
            json.dump(state, file, ensure_ascii=False, allow_nan=False)
            file.flush()
            os.fsync(file.fileno())
            os.replace(path, directory / "state.json")
        finally:
            path.unlink(missing_ok=True)


def load_latest_analysis(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict | None:
    """Read the most recent completed event analysis, including its source evidence."""
    return _read_state(output_dir)["latest"]


def run_quick_forecast(symbols: tuple[str, ...], directory: Path, lock_fd: int) -> None:
    """Run the established report generator in an isolated directory with daily freshness enforced."""
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).with_name("daily_ml_forecast_report.py")),
               "--run-profile", "quick", "--no-rl-policy", "--symbols", ",".join(symbols),
               "--max-data-lag-sessions", "0", "--runtime-budget-minutes", "15", "--output-dir", str(directory)]
    # The poll lock also stays with the child if its parent is interrupted.
    # The event runs share their own price cache, independent of scheduled publications.
    env = dict(os.environ, MARKET_AGENT_DATA_CACHE_DIR=str(directory.parents[2] / "ohlcv_cache"))
    with (directory / "forecast_output.txt").open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, timeout=16 * 60, pass_fds=(lock_fd,), env=env, stdout=log, stderr=subprocess.STDOUT)


def _validated_result(directory: Path, root: Path, pending: dict) -> dict:
    from market_agent.price_watch_report import has_scheduled_report_data, is_usable_observation

    report_path = directory / "ml_forecast_rankings_latest.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("Quick analysis produced a malformed report")
    sessions = report.get("as_of_sessions")
    rows = report.get("rows")
    if (report.get("run_complete") is not True or not has_scheduled_report_data(report)
            or report.get("symbols") != pending["symbols"] or report.get("run_profile") != "quick"
            or not isinstance(sessions, list) or not sessions
            or not all(isinstance(value, str) and date.fromisoformat(value).isoformat() == value for value in sessions)
            or type(report.get("ranked_count")) is not int or report["ranked_count"] != len(rows)):
        raise ValueError("Quick analysis produced no completed report with attributable price sessions")
    covered = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("As Of Session") not in sessions:
            continue
        price = row.get("Last Price")
        symbol = row.get("Symbol")
        if isinstance(symbol, str) and type(price) in (int, float) and math.isfinite(price) and price > 0:
            covered.add(symbol)
    watch = report.get("price_watch") or {}
    if isinstance(watch, dict) and isinstance(watch.get("observations"), list):
        for obs in watch["observations"]:
            if is_usable_observation(obs) and obs.get("lag_sessions") == 0 and obs["as_of_session"] in sessions:
                covered.add(obs["symbol"])
    missing = set(pending["symbols"]) - covered
    if missing:
        raise ValueError("Quick analysis is missing dated price evidence for " + ", ".join(sorted(missing)))
    timestamp = parse_report_timestamp(report.get("generated_at"))
    summary = report.get("telegram_recommendations") or report.get("report_text")
    errors = report.get("errors", [])
    if timestamp is None or not isinstance(summary, str) or not summary.strip():
        raise ValueError("Quick report is missing its generation time or market analysis")
    if not isinstance(errors, list) or not all(isinstance(error, str) for error in errors):
        raise ValueError("Quick report contains invalid error metadata")
    return {"run_id": pending["run_id"], "generated_at": timestamp.isoformat(), "events": pending["events"],
            "symbols": pending["symbols"], "as_of_sessions": sessions, "ranked_count": report["ranked_count"],
            "errors": errors, "report_path": str(report_path.relative_to(root)), "summary": summary}


def poll_breaking_analysis(
    output_dir: Path = DEFAULT_OUTPUT_DIR, *, now: datetime | None = None,
    check_feeds: Callable[[], HeadlineCheck] = check_publisher_headlines,
    run_forecast: Callable[[tuple[str, ...], Path, int], None] = run_quick_forecast,
) -> dict:
    """Collect recent events and run one bounded, crash-recoverable analysis batch."""
    import fcntl

    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("A timezone-aware poll time is required")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "poll.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "busy"}
        state = _read_state(output_dir)
        pending = state["pending"]
        if pending and pending["attempts"]:
            previous = output_dir / "runs" / pending["run_id"] / str(pending["attempts"])
            if (previous / "ml_forecast_rankings_latest.json").exists():
                try:
                    result = _validated_result(previous, output_dir, pending)
                except ValueError:
                    pass  # An invalid artifact must be retried in a fresh attempt directory.
                else:
                    return _complete(output_dir, state, result)
        check = check_feeds()
        state["receipts"] = [e for e in state["receipts"] if now - datetime.fromisoformat(e["published_at"]) < timedelta(days=7)]
        state["queue"] = [e for e in state["queue"] if now - datetime.fromisoformat(e["published_at"]) <= MAX_AGE]
        known = state["receipts"] + state["queue"] + (pending["events"] if pending else [])
        for article in sorted(check.headlines, key=lambda item: item.published_at):
            event = classify_headline(article, now)
            if event and not any(_same_event(event, previous) for previous in known):
                state["queue"].append(event)
                known.append(event)
        if not pending and state["queue"] and (
            not state["latest"] or now - datetime.fromisoformat(state["latest"]["generated_at"]) >= COOLDOWN
        ):
            events, state["queue"] = state["queue"][:3], state["queue"][3:]
            symbols = list(dict.fromkeys([s for event in events for s in event["symbols"]] + ["SPY", "QQQ"]))
            state["pending"] = pending = {"run_id": uuid4().hex, "events": events, "symbols": symbols,
                                          "attempts": 0, "last_attempt": None}
        _save_state(output_dir, state)
        if not pending:
            if len(check.failures) == len(TRUSTED_PUBLISHERS):
                raise RuntimeError("No publisher feeds were available for the breaking-news poll")
            return {"status": "cooldown" if state["queue"] else "no_event", "feed_failures": check.failures}
        if pending["last_attempt"] and now - datetime.fromisoformat(pending["last_attempt"]) < RETRY_DELAY:
            return {"status": "retry_wait", "last_error": pending.get("last_error"), "feed_failures": check.failures}
        if pending["attempts"] >= MAX_ATTEMPTS or all(
            now - datetime.fromisoformat(event["published_at"]) > MAX_AGE for event in pending["events"]
        ):
            state["receipts"].extend(pending["events"])
            state["last_failed"] = pending
            state["pending"] = None
            _save_state(output_dir, state)
            raise RuntimeError("Breaking-news analysis expired or exhausted its retries")
        pending["attempts"] += 1
        pending["last_attempt"] = now.isoformat()
        _save_state(output_dir, state)
        directory = output_dir / "runs" / pending["run_id"] / str(pending["attempts"])
        try:
            run_forecast(tuple(pending["symbols"]), directory, lock.fileno())
            result = _validated_result(directory, output_dir, pending)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            pending["last_error"] = str(exc)
            _save_state(output_dir, state)
            raise
        return {**_complete(output_dir, state, result), "feed_failures": check.failures}


def _complete(root: Path, state: dict, result: dict) -> dict:
    state["receipts"].extend(result["events"])
    state.update(latest=result, pending=None)
    _save_state(root, state)
    return {"status": "completed", "run_id": result["run_id"], "symbols": result["symbols"],
            "as_of_sessions": result["as_of_sessions"], "report_path": result["report_path"]}


def main() -> int:
    """Run one market-news poll for the timer or an operator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(poll_breaking_analysis(args.output_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
