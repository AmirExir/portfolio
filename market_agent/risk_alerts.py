"""Completed-close risk warnings for the scheduled market report."""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

RANGE = 20
WINDOW = RANGE + 2  # 20 prior returns, plus the current close
STATE_VERSION = 1


def observe_closes(
    symbol: str,
    closes: pd.Series,
    source: dict,
    lag: int,
    revisions: dict | None = None,
) -> dict:
    """Calculate all levels from the same ordered, completed daily observations."""
    prices = closes.sort_index().tail(WINDOW)
    session = prices.index[-1].date()
    result = {
        "symbol": symbol,
        "as_of_session": session.isoformat(),
        "close": float(prices.iloc[-1]),
        "lag_sessions": lag,
        "data_source": source,
    }
    if len(prices) < WINDOW or not all(math.isfinite(x) and x > 0 for x in prices):
        return dict(result, issue="Insufficient valid completed closes")
    if symbol.endswith("-USD"):
        days = prices.index.normalize()
        if len(days.unique()) != WINDOW or (days[-1] - days[0]).days != WINDOW - 1:
            return dict(result, issue="Missing or duplicate UTC daily closes")
        result["window_closes"] = {
            x.date().isoformat(): float(v) for x, v in prices.items()
        }
        result["revisions"] = {
            day: values
            for day, values in (revisions or {}).items()
            if day in result["window_closes"]
        }
    if lag:
        return dict(result, issue=f"Latest completed close is {lag} session(s) stale")
    prior = prices.iloc[-RANGE - 1 : -1]
    close = result["close"]
    change = (close / float(prices.iloc[-2]) - 1) * 100
    threshold = max(2.0, 200 * float(prices.iloc[:-1].pct_change().dropna().std()))
    low, high = float(prior.min()), float(prior.max())
    recent = prices.iloc[-RANGE:]
    positive_momentum = close > prices.iloc[-6]
    return dict(
        result,
        prior_low=low,
        prior_high=high,
        next_low=float(recent.min()),
        next_high=float(recent.max()),
        change_pct=change,
        shock_threshold_pct=threshold,
        shock=change <= -threshold,
        breakdown=close < low,
        entry_triggered=close > high,
        entry_watch=bool(positive_momentum and change > 0 and close >= high * 0.98),
    )


def _flags(row: dict, obs: dict) -> list[str]:
    flags = []
    if row.get("Smart Policy") == "Sell / Avoid":
        flags.append("policy")
    flags.extend(key for key in ("breakdown", "shock") if obs.get(key))
    return flags


def _add_close(evidence: dict, session: str, close: float) -> None:
    if not math.isfinite(close) or close <= 0:
        return
    known = evidence.setdefault(session, [close])
    # Ignore only numerical noise in a float32 quote carried through CSV/JSON.
    if len(known) == 1 and not math.isclose(known[0], close, rel_tol=1e-7, abs_tol=0):
        known.append(close)


def disputed_candidates(observation: dict, item: dict) -> dict[str, float]:
    """Find fresh daily inputs with conflicting prior evidence and no matching hourly check."""
    evidence = item.get("closes", {})
    confirmed = item.get("confirmed", {})
    candidates = {}
    for day, close in observation.get("window_closes", {}).items():
        values = observation.get("revisions", {}).get(day, []) + evidence.get(day, [])
        disputed = any(
            not math.isclose(close, v, rel_tol=1e-7, abs_tol=0) for v in values
        )
        if disputed and not math.isclose(
            close, confirmed.get(day, 0), rel_tol=1e-7, abs_tol=0
        ):
            candidates[day] = close
    return candidates


def _remember(
    state: dict, row: dict, obs: dict, symbol: str, crypto: bool
) -> tuple[str, list]:
    item = state.setdefault(symbol, {})
    session = row["As Of Session"]
    last = item.get("session")
    if last and session < last:
        return "Older observation", []
    if crypto:
        evidence = item.setdefault("closes", {})
        earliest = (
            date.fromisoformat(session) - timedelta(days=WINDOW - 1)
        ).isoformat()
        for day, value in obs.get("window_closes", {}).items():
            _add_close(evidence, day, value)
        for day, values in obs.get("revisions", {}).items():
            for value in values:
                _add_close(evidence, day, value)
        _add_close(evidence, session, row["Last Price"])
        item["closes"] = {d: v for d, v in evidence.items() if d >= earliest}
        confirmed = item.setdefault("confirmed", {})
        for day, value in obs.get("window_closes", {}).items():
            if not math.isclose(value, confirmed.get(day, 0), rel_tol=1e-7, abs_tol=0):
                confirmed.pop(day, None)
        confirmed.update(obs.get("corroborated_closes", {}))
        item["confirmed"] = {d: v for d, v in confirmed.items() if d >= earliest}
    conflicts = [
        (d, values)
        for d, values in sorted(item.get("closes", {}).items())
        if len(values) > 1 and d not in item.get("confirmed", {})
    ]
    if conflicts or obs.get("issue"):
        return "Unverified", conflicts
    flags = _flags(row, obs)
    previous = set(item.get("flags", []))
    seen = set(item.get("seen", [])) if session == last else set()
    if not last:
        status = "First observed"
    elif not previous and set(flags) - seen:
        status = "New"
    elif set(flags) - previous - seen or (
        session != last and any(f != "policy" for f in flags)
    ):
        status = "Escalated" if previous else "New"
    else:
        status = "Ongoing"
    item.update(session=session, flags=flags, seen=sorted(seen.union(flags)))
    return status, conflicts


def _symbol_map(symbols: list[str], names: dict[str, str]) -> dict[str, str]:
    return {names.get(s, s): s for s in symbols}


def _observations(watch: dict) -> dict[str, dict]:
    return {v["symbol"]: v for v in watch.get("observations", [])}


def publication_history(
    directory: Path, horizon: int, cutoff: str, names: dict[str, str]
) -> dict:
    """Load committed state, or seed it from earlier timestamped reports on upgrade."""
    path = directory / state_filename(horizon)
    if path.exists():
        payload = json.loads(path.read_text())
        if payload.get("version") != STATE_VERSION:
            raise ValueError(f"Unknown market risk history version in {path.name}")
        return payload["symbols"]
    state = {}
    earliest = (
        date.fromisoformat(cutoff[:10]) - timedelta(days=WINDOW + 3)
    ).isoformat()
    prefix = "ml_forecast_rankings_"
    for report_path in sorted(directory.glob(prefix + "20*.json")):
        stamp = report_path.stem[len(prefix) :]
        if not earliest <= stamp < cutoff:
            continue
        report = json.loads(report_path.read_text())
        if (
            report.get("horizon_days") != horizon
            or report.get("run_complete", True) is not True
        ):
            continue
        symbols = _symbol_map(report.get("symbols", []), names)
        observations = _observations(report.get("price_watch", {}))
        for row in report.get("rows", []):
            symbol = symbols.get(row.get("Symbol"), row.get("Symbol"))
            if symbol and row.get("As Of Session") and row.get("Last Price"):
                _remember(
                    state,
                    row,
                    observations.get(symbol, {}),
                    symbol,
                    symbol.endswith("-USD"),
                )
    return state


def state_filename(horizon: int) -> str:
    """Key warning transitions by forecast horizon across subset runs."""
    return f"market_risk_history_v{STATE_VERSION}_{horizon}.json"


def evaluate_watch(
    rows: list[dict],
    observations: dict,
    state: dict,
    symbols: list[str],
    names: dict[str, str],
) -> dict:
    """Classify newly observed or worsening risks, updating staged publication state."""
    symbols_by_name = _symbol_map(symbols, names)
    warnings, entries, issues, checked = [], [], [], []
    for row in rows:
        symbol = symbols_by_name.get(row.get("Symbol"), row.get("Symbol"))
        if not symbol or not row.get("As Of Session") or not row.get("Last Price"):
            continue
        obs = dict(observations.get(symbol, {}))
        if obs and (
            obs["as_of_session"] != row["As Of Session"]
            or not math.isclose(
                obs["close"], row["Last Price"], rel_tol=1e-7, abs_tol=0
            )
        ):
            obs["issue"] = (
                "Forecast and price-watch inputs disagree on the completed close"
            )
        status, conflicts = _remember(state, row, obs, symbol, symbol.endswith("-USD"))
        if status == "Older observation":
            obs["issue"] = "A later session has already been published"
        if conflicts:
            day, values = conflicts[-1]
            obs["issue"] = (
                f"Conflicting completed closes for {day} ({values[0]:.8g} and {values[1]:.8g})"
            )
        if obs.get("issue"):
            issues.append({"symbol": symbol, "reason": obs["issue"]})
            obs.update(
                symbol=symbol,
                as_of_session=row["As Of Session"],
                close=row["Last Price"],
            )
            checked.append(
                {
                    k: v
                    for k, v in obs.items()
                    if k
                    not in (
                        "shock",
                        "breakdown",
                        "entry_watch",
                        "entry_triggered",
                        "change_pct",
                        "shock_threshold_pct",
                        "prior_low",
                        "prior_high",
                        "next_high",
                        "next_low",
                    )
                }
            )
            continue
        if obs:
            checked.append(obs)
        flags = _flags(row, obs)
        if flags:
            warnings.append(
                {
                    "symbol": symbol,
                    "status": status,
                    "flags": flags,
                    "observation": obs,
                    "close": row["Last Price"],
                    "session": row["As Of Session"],
                    "trend": row.get("Policy Trend Score"),
                    "momentum": row.get("Policy Momentum Score"),
                    "forecast": row.get("Forecast Return %"),
                    "model": row.get("Model Call"),
                    "reliability": row.get("Reliability"),
                }
            )
        if obs.get("entry_watch") or obs.get("entry_triggered"):
            entries.append(obs)
    downside = [obs for obs in checked if obs.get("shock") or obs.get("breakdown")]
    return {
        "observations": checked,
        "warnings": warnings,
        "data_issues": issues,
        "downside_count": len(downside),
        "downside_watch": downside,
        "entry_watch": entries,
        "entry_count": len(entries),
        "universe_count": len(symbols),
        "observed_count": len(checked),
        "benchmark": next((x for x in checked if x.get("symbol") == "SPY"), None),
        "as_of_sessions": sorted(
            {x["as_of_session"] for x in checked if x.get("as_of_session")}
        ),
        "rules": {
            "version": "closing-price-watch-v2",
            "range_sessions": RANGE,
            "momentum_sessions": 5,
            "shock_floor_pct": 2,
            "shock_volatility_multiplier": 2,
            "entry_proximity_pct": 2,
        },
    }


def _price(value: float) -> str:
    return f"{value:.2f}" if value >= 1 else f"{value:.8g}"


def format_watch(watch: dict) -> str:
    """Render all new and escalated warnings independently of forecast qualification."""
    lines = []
    benchmark = watch["benchmark"]
    if benchmark and not benchmark.get("issue"):
        lines.append(
            f"SPY benchmark ({benchmark['as_of_session']}): {_price(benchmark['close'])}, "
            f"{benchmark['change_pct']:+.2f}% in one session."
        )
    for status in ("New", "Escalated", "First observed", "Ongoing"):
        alerts = [a for a in watch["warnings"] if a["status"] == status]
        if alerts:
            lines.append(f"{status} risk warnings")
        alerts.sort(key=lambda a: (a["observation"].get("change_pct", 0), a["symbol"]))
        displayed = alerts[:5] if status == "Ongoing" else alerts
        for a in displayed:
            flags, obs = a["flags"], a["observation"]
            reasons = []
            if "policy" in flags:
                drivers = []
                if (a["trend"] or 0) < 0:
                    drivers.append("weak trend")
                if (a["momentum"] or 0) < 0:
                    drivers.append("negative momentum")
                reasons.append(
                    "Sell / Avoid"
                    + (" (" + ", ".join(drivers) + ")" if drivers else " policy")
                )
            if "breakdown" in flags:
                reasons.append(
                    f"below prior 20-session closing low {_price(obs['prior_low'])}"
                )
            if "shock" in flags:
                reasons.append(
                    f"{obs['change_pct']:+.2f}% daily move (alert {obs['shock_threshold_pct']:.2f}%)"
                )
            forecast = a["model"] or "unavailable"
            if a["forecast"] is not None and math.isfinite(a["forecast"]):
                forecast += f" {a['forecast']:+.2f}%"
            lines.append(
                f"{a['symbol']} at {_price(a['close'])} ({a['session']}) — "
                + ", ".join(reasons)
                + f". Forecast: {forecast}, {a['reliability'] or 'unknown'} reliability."
            )
        if len(displayed) < len(alerts):
            lines.append(
                "Also ongoing: "
                + ", ".join(a["symbol"] for a in alerts[len(displayed) :])
            )
        if alerts:
            lines.append("")
    for obs in watch["entry_watch"]:
        label = "Closed above" if obs["entry_triggered"] else "Watch for a close above"
        level = obs["prior_high"] if obs["entry_triggered"] else obs["next_high"]
        lines.append(
            f"{obs['symbol']} at {_price(obs['close'])} — {label} {_price(level)}. "
            f"A close below {_price(obs['next_low'])} invalidates the setup."
        )
    for issue in watch["data_issues"]:
        lines.append(f"{issue['symbol']} — {issue['reason']}. Price alert withheld.")
    return "\n".join(lines).strip()
