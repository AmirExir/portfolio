"""Point-in-time earnings interpretation for market-policy context.

The module is provider neutral: callers pass a small structured payload and
receive a bounded :class:`EarningsSignal`.  It intentionally does not place
orders or decide position sizes.  The signal is suitable for a forecasting or
shadow-policy feature only after the caller verifies ``policy_eligible``.

Canonical payload example::

    {
        "symbol": "XYZ",
        "reported_at": "2026-07-29T16:05:00-04:00",
        "fiscal_period": "Q2 2026",
        "eps": {"actual": 1.10, "estimate": 1.00},
        "revenue": {"actual": 5_100_000_000, "estimate": 5_000_000_000},
        "guidance": {"direction": "raised"},
        "source": "provider-name",
    }

For a daily policy, a pre-market release is assigned to that day's decision
session.  A release at or after the regular-session open is assigned to the
next trading session.  Supplying an authoritative ``trading_sessions`` list is
recommended. Production callers can use :func:`us_equity_trading_sessions` to
combine observed sessions with deterministic US-equity holiday rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
import json
import logging
import math
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


LOGGER = logging.getLogger(__name__)

DEFAULT_MARKET_TIMEZONE = "America/New_York"
DEFAULT_MARKET_OPEN = time(9, 30)
_DIRECTION_ALIASES = {
    "raise": "raised",
    "raised": "raised",
    "raising": "raised",
    "higher": "raised",
    "up": "raised",
    "positive": "raised",
    "lower": "lowered",
    "lowered": "lowered",
    "lowering": "lowered",
    "reduced": "lowered",
    "down": "lowered",
    "negative": "lowered",
    "reaffirm": "reaffirmed",
    "reaffirmed": "reaffirmed",
    "maintain": "reaffirmed",
    "maintained": "reaffirmed",
    "unchanged": "reaffirmed",
    "withdraw": "withdrawn",
    "withdrawn": "withdrawn",
    "suspended": "withdrawn",
}
_GUIDANCE_DIRECTION_SCORE = {
    "raised": 1.0,
    "lowered": -1.0,
    "reaffirmed": 0.0,
    "withdrawn": -0.75,
}
_FATAL_BLOCKERS = {
    "event_not_yet_published",
    "event_not_effective_for_decision_session",
    "stale_event",
    "no_comparable_metrics",
    "invalid_payload",
    "unverified_report_timestamp",
}


class EarningsDataError(ValueError):
    """Raised when a canonical earnings payload cannot be parsed safely."""


@dataclass(frozen=True)
class ReportedMetric:
    """Reported and consensus values for one earnings metric."""

    actual: float | None = None
    estimate: float | None = None


@dataclass(frozen=True)
class Guidance:
    """Structured management guidance.

    ``actual`` is the newly guided value or midpoint, while ``estimate`` is the
    comparable consensus value.  A qualitative ``direction`` may be supplied
    instead of, or in addition to, those numeric fields.
    """

    direction: str | None = None
    actual: float | None = None
    estimate: float | None = None


@dataclass(frozen=True)
class EarningsEvent:
    """Normalized point-in-time earnings event."""

    symbol: str
    reported_at: datetime
    eps: ReportedMetric = field(default_factory=ReportedMetric)
    revenue: ReportedMetric = field(default_factory=ReportedMetric)
    guidance: Guidance = field(default_factory=Guidance)
    fiscal_period: str = ""
    source: str = ""
    timestamp_quality: str = "exact"
    parse_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class EarningsInterpretationConfig:
    """Thresholds and market-session assumptions for interpretation."""

    market_timezone: str = DEFAULT_MARKET_TIMEZONE
    market_open: time = DEFAULT_MARKET_OPEN
    max_event_age_sessions: int = 2
    eps_in_line_tolerance_pct: float = 0.5
    revenue_in_line_tolerance_pct: float = 0.5
    guidance_in_line_tolerance_pct: float = 1.0
    surprise_score_scale_pct: float = 10.0

    def __post_init__(self) -> None:
        try:
            ZoneInfo(self.market_timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(
                f"unknown market_timezone: {self.market_timezone!r}"
            ) from exc
        if (
            isinstance(self.max_event_age_sessions, bool)
            or not isinstance(self.max_event_age_sessions, int)
        ):
            raise ValueError("max_event_age_sessions must be an integer")
        if self.max_event_age_sessions < 0:
            raise ValueError("max_event_age_sessions cannot be negative")
        if not isinstance(self.market_open, time):
            raise ValueError("market_open must be a datetime.time")
        for name in (
            "eps_in_line_tolerance_pct",
            "revenue_in_line_tolerance_pct",
            "guidance_in_line_tolerance_pct",
            "surprise_score_scale_pct",
        ):
            value = getattr(self, name)
            if not isinstance(value, Real) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class EarningsSignal:
    """Bounded earnings context safe for a policy decision.

    ``event_flag`` is one only when the event was known, effective for the
    requested decision session, fresh, and supported by at least one comparable
    metric or recognized guidance item.  ``event_score`` and ``confidence`` are
    always bounded to ``[-1, 1]`` and ``[0, 1]`` respectively.
    """

    symbol: str
    event_flag: int
    event_score: float
    confidence: float
    summary: str
    outcome: str
    reported_at: datetime | None
    effective_session: date | None
    decision_session: date | None
    eps_surprise_pct: float | None = None
    revenue_surprise_pct: float | None = None
    guidance_surprise_pct: float | None = None
    age_sessions: int | None = None
    is_stale: bool = False
    blockers: tuple[str, ...] = ()
    data_quality_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.event_flag not in (0, 1):
            raise ValueError("event_flag must be 0 or 1")
        if not -1.0 <= self.event_score <= 1.0:
            raise ValueError("event_score must be between -1 and 1")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    @property
    def policy_eligible(self) -> bool:
        """Whether this event may be exposed to a decision policy."""
        return self.event_flag == 1 and not any(
            blocker in _FATAL_BLOCKERS for blocker in self.blockers
        )

    def policy_context(self) -> dict[str, float | bool]:
        """Return only bounded numeric fields intended for model state."""
        eligible = self.policy_eligible
        flag = 1.0 if eligible else 0.0
        score = self.event_score if eligible else 0.0
        confidence = self.confidence if eligible else 0.0
        return {
            "event_flag": flag,
            "event_score": score,
            "event_confidence": confidence,
            "earnings_event_flag": flag,
            "earnings_event_score": score,
            "earnings_confidence": confidence,
            "earnings_policy_eligible": eligible,
        }


@dataclass(frozen=True)
class EarningsFetchResult:
    """Result from an optional external-data adapter."""

    available: bool
    payload: Mapping[str, Any] | None = None
    error_code: str | None = None


def load_earnings_payload_file(
    symbol: str,
    directory: str | Path | None,
) -> EarningsFetchResult:
    """Load one canonical verified-provider ``SYMBOL.json`` payload."""
    normalized_symbol = str(symbol).strip().upper()
    raw_directory = str(directory or "").strip()
    if not normalized_symbol:
        return EarningsFetchResult(False, error_code="invalid_symbol")
    if not raw_directory:
        return EarningsFetchResult(
            False,
            error_code="external_earnings_payload_not_configured",
        )
    safe_symbol = "".join(
        character
        if character.isalnum() or character in {".", "-"}
        else "_"
        for character in normalized_symbol
    )
    payload_path = (
        Path(raw_directory).expanduser() / f"{safe_symbol}.json"
    )
    if not payload_path.exists():
        return EarningsFetchResult(
            False,
            error_code="external_earnings_payload_missing",
        )
    try:
        raw_payload = json.loads(
            payload_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return EarningsFetchResult(
            False,
            error_code="external_earnings_payload_invalid",
        )
    if (
        isinstance(raw_payload, Mapping)
        and isinstance(raw_payload.get("payload"), Mapping)
    ):
        raw_payload = raw_payload["payload"]
    if not isinstance(raw_payload, Mapping):
        return EarningsFetchResult(
            False,
            error_code="external_earnings_payload_invalid",
        )
    payload = dict(raw_payload)
    payload_symbol = str(payload.get("symbol", "")).strip().upper()
    if payload_symbol and payload_symbol != normalized_symbol:
        return EarningsFetchResult(
            False,
            error_code="external_earnings_symbol_mismatch",
        )
    payload["symbol"] = normalized_symbol
    return EarningsFetchResult(True, payload=payload)


def parse_earnings_payload(
    payload: Mapping[str, Any],
    *,
    default_timezone: str | None = None,
) -> EarningsEvent:
    """Normalize a canonical or common flat earnings payload.

    Naive timestamps are rejected unless ``default_timezone`` (or a payload
    ``timezone`` field) is provided.  Optional invalid numeric fields are
    omitted and recorded in ``parse_flags`` rather than converted to zero.
    """
    if not isinstance(payload, Mapping):
        raise EarningsDataError("earnings payload must be a mapping")

    symbol = str(payload.get("symbol", "")).strip().upper()
    if not symbol:
        raise EarningsDataError("symbol is required")

    timezone_name = str(
        payload.get("timezone") or default_timezone or ""
    ).strip()
    reported_at = _parse_datetime(
        payload.get("reported_at"),
        field_name="reported_at",
        default_timezone=timezone_name or None,
    )

    flags: list[str] = []
    eps = _parse_metric(
        payload.get("eps"),
        flat_actual=payload.get("reported_eps"),
        flat_estimate=payload.get("estimated_eps"),
        name="eps",
        flags=flags,
    )
    revenue = _parse_metric(
        payload.get("revenue"),
        flat_actual=payload.get("reported_revenue"),
        flat_estimate=payload.get("estimated_revenue"),
        name="revenue",
        flags=flags,
    )
    guidance = _parse_guidance(payload.get("guidance"), flags)

    timestamp_quality = str(
        payload.get("timestamp_quality", "exact")
    ).strip().lower()
    if timestamp_quality not in {"exact", "provider_reported", "scheduled"}:
        flags.append("timestamp_quality_unknown")
        timestamp_quality = "unknown"
    if timestamp_quality in {"scheduled", "unknown"}:
        flags.append("imprecise_report_timestamp")

    return EarningsEvent(
        symbol=symbol,
        reported_at=reported_at,
        eps=eps,
        revenue=revenue,
        guidance=guidance,
        fiscal_period=str(payload.get("fiscal_period", "")).strip(),
        source=str(payload.get("source", "")).strip(),
        timestamp_quality=timestamp_quality,
        parse_flags=_ordered_unique(flags),
    )


def effective_decision_session(
    reported_at: datetime,
    *,
    config: EarningsInterpretationConfig | None = None,
    trading_sessions: Iterable[date | datetime | str] | None = None,
) -> date:
    """Resolve the first daily decision session that may use an event.

    A release strictly before the market open belongs to the same session when
    that date is a trading session.  All releases at or after the open are
    assigned to the next session so a daily close-to-close backtest cannot use
    an intra-session result before it was available.
    """
    settings = config or EarningsInterpretationConfig()
    if reported_at.tzinfo is None:
        raise ValueError("reported_at must be timezone-aware")

    local = reported_at.astimezone(ZoneInfo(settings.market_timezone))
    sessions = _normalized_sessions(trading_sessions)
    report_date = local.date()

    if local.time() < settings.market_open and _is_session(
        report_date, sessions
    ):
        return report_date
    return _next_session_after(report_date, sessions)


def us_equity_trading_sessions(
    *,
    as_of: datetime,
    observed_sessions: Iterable[date | datetime | str] | None = None,
    lookback_days: int = 400,
    future_session_count: int = 12,
) -> tuple[date, ...]:
    """Extend observed sessions with deterministic US-equity holiday rules.

    Observed price-history dates take precedence historically, capturing
    one-off closures present in the supplied data. Future dates use regular
    NYSE holiday rules. An exchange-calendar feed remains preferable when one
    is available.
    """
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    if (
        isinstance(lookback_days, bool)
        or int(lookback_days) < 1
        or isinstance(future_session_count, bool)
        or int(future_session_count) < 1
    ):
        raise ValueError(
            "lookback_days and future_session_count must be positive integers"
        )

    anchor = as_of.astimezone(ZoneInfo(DEFAULT_MARKET_TIMEZONE)).date()
    observed = set(_normalized_sessions(observed_sessions) or ())
    cursor = (
        max(observed) + timedelta(days=1)
        if observed
        else anchor - timedelta(days=int(lookback_days))
    )
    future_found = sum(1 for session in observed if session > anchor)
    while cursor <= anchor or future_found < int(future_session_count):
        if _is_us_equity_rule_session(cursor):
            observed.add(cursor)
            if cursor > anchor:
                future_found += 1
        cursor += timedelta(days=1)
    return tuple(sorted(observed))


def interpret_earnings_event(
    event: EarningsEvent,
    *,
    as_of: datetime,
    decision_session: date | datetime | str | None = None,
    config: EarningsInterpretationConfig | None = None,
    trading_sessions: Iterable[date | datetime | str] | None = None,
) -> EarningsSignal:
    """Interpret one normalized event without using future-effective data."""
    settings = config or EarningsInterpretationConfig()
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    if event.reported_at.tzinfo is None:
        raise ValueError("event.reported_at must be timezone-aware")

    sessions = _normalized_sessions(trading_sessions)
    effective = effective_decision_session(
        event.reported_at,
        config=settings,
        trading_sessions=sessions,
    )
    chosen_session = (
        _parse_date(decision_session, "decision_session")
        if decision_session is not None
        else _decision_session_for_as_of(as_of, settings, sessions)
    )
    common = {
        "symbol": event.symbol,
        "reported_at": event.reported_at,
        "effective_session": effective,
        "decision_session": chosen_session,
    }

    # This guard precedes surprise calculation so future actuals cannot leak
    # through score, summary, or diagnostic fields.
    if as_of < event.reported_at:
        return _blocked_signal(
            summary=f"{event.symbol} earnings were not published as of this decision.",
            blocker="event_not_yet_published",
            **common,
        )
    if chosen_session < effective:
        return _blocked_signal(
            summary=(
                f"{event.symbol} earnings were not effective for the "
                f"{chosen_session.isoformat()} decision session."
            ),
            blocker="event_not_effective_for_decision_session",
            **common,
        )

    age_sessions = _session_distance(effective, chosen_session, sessions)
    if age_sessions > settings.max_event_age_sessions:
        return EarningsSignal(
            event_flag=0,
            event_score=0.0,
            confidence=0.0,
            summary=(
                f"{event.symbol} earnings are stale for the "
                f"{chosen_session.isoformat()} decision session."
            ),
            outcome="stale",
            age_sessions=age_sessions,
            is_stale=True,
            blockers=("stale_event",),
            data_quality_flags=event.parse_flags,
            **common,
        )

    quality_flags = [
        *event.parse_flags,
        *_quality_flag_for_calendar(sessions),
    ]
    eps_surprise = _metric_surprise(
        event.eps, "eps", quality_flags
    )
    revenue_surprise = _metric_surprise(
        event.revenue, "revenue", quality_flags
    )
    guidance_surprise = _metric_surprise(
        ReportedMetric(
            actual=event.guidance.actual,
            estimate=event.guidance.estimate,
        ),
        "guidance",
        quality_flags,
        flag_absent=False,
    )
    direction_score = _GUIDANCE_DIRECTION_SCORE.get(event.guidance.direction)
    if (
        event.guidance.direction is not None
        and direction_score is None
    ):
        quality_flags.append("guidance_direction_unrecognized")

    components: list[tuple[str, float, float, float]] = []
    if eps_surprise is not None:
        components.append(
            (
                "eps",
                _surprise_score(
                    eps_surprise, settings.surprise_score_scale_pct
                ),
                0.45,
                _score_tolerance(
                    settings.eps_in_line_tolerance_pct,
                    settings.surprise_score_scale_pct,
                ),
            )
        )
    if revenue_surprise is not None:
        components.append(
            (
                "revenue",
                _surprise_score(
                    revenue_surprise, settings.surprise_score_scale_pct
                ),
                0.35,
                _score_tolerance(
                    settings.revenue_in_line_tolerance_pct,
                    settings.surprise_score_scale_pct,
                ),
            )
        )

    quantitative_guidance_score: float | None = None
    if guidance_surprise is not None:
        quantitative_guidance_score = _surprise_score(
            guidance_surprise, settings.surprise_score_scale_pct
        )
    guidance_score = _combine_guidance_scores(
        direction_score,
        quantitative_guidance_score,
        quality_flags,
    )
    if guidance_score is not None:
        components.append(
            (
                "guidance",
                guidance_score,
                0.20,
                _score_tolerance(
                    settings.guidance_in_line_tolerance_pct,
                    settings.surprise_score_scale_pct,
                ),
            )
        )

    if not components:
        return EarningsSignal(
            event_flag=0,
            event_score=0.0,
            confidence=0.0,
            summary=f"{event.symbol} earnings lack comparable reported data.",
            outcome="unavailable",
            age_sessions=age_sessions,
            blockers=("no_comparable_metrics",),
            data_quality_flags=_ordered_unique(quality_flags),
            **common,
        )

    total_weight = sum(weight for _, _, weight, _ in components)
    raw_score = sum(score * weight for _, score, weight, _ in components)
    raw_score /= total_weight
    freshness = max(
        0.50,
        1.0 - 0.20 * float(age_sessions),
    )
    event_score = _clip(raw_score * freshness, -1.0, 1.0)

    labels = [
        _component_label(score, tolerance)
        for _, score, _, tolerance in components
    ]
    outcome = _overall_outcome(labels)
    consistency = 0.75 if outcome == "mixed" else 1.0
    quality_factor = _quality_factor(quality_flags)
    confidence = _clip(
        total_weight * freshness * consistency * quality_factor,
        0.0,
        1.0,
    )

    blockers = (
        ("unverified_report_timestamp",)
        if event.timestamp_quality in {"scheduled", "unknown"}
        else ()
    )
    return EarningsSignal(
        event_flag=1,
        event_score=event_score,
        confidence=confidence,
        summary=_build_summary(
            event,
            outcome,
            eps_surprise,
            revenue_surprise,
            guidance_surprise,
        ),
        outcome=outcome,
        eps_surprise_pct=eps_surprise,
        revenue_surprise_pct=revenue_surprise,
        guidance_surprise_pct=guidance_surprise,
        age_sessions=age_sessions,
        blockers=blockers,
        data_quality_flags=_ordered_unique(quality_flags),
        **common,
    )


def interpret_earnings_payload(
    payload: Mapping[str, Any],
    *,
    as_of: datetime,
    decision_session: date | datetime | str | None = None,
    config: EarningsInterpretationConfig | None = None,
    trading_sessions: Iterable[date | datetime | str] | None = None,
    default_timezone: str | None = None,
) -> EarningsSignal:
    """Parse and interpret a payload, returning a blocked signal on bad data."""
    try:
        event = parse_earnings_payload(
            payload,
            default_timezone=default_timezone,
        )
    except (EarningsDataError, TypeError, ValueError):
        symbol = (
            str(payload.get("symbol", "")).strip().upper()
            if isinstance(payload, Mapping)
            else ""
        )
        return EarningsSignal(
            symbol=symbol or "UNKNOWN",
            event_flag=0,
            event_score=0.0,
            confidence=0.0,
            summary="Earnings event is unavailable because the payload is invalid.",
            outcome="unavailable",
            reported_at=None,
            effective_session=None,
            decision_session=(
                _parse_date(decision_session, "decision_session")
                if decision_session is not None
                else None
            ),
            blockers=("invalid_payload",),
            data_quality_flags=("payload_parse_failed",),
        )
    return interpret_earnings_event(
        event,
        as_of=as_of,
        decision_session=decision_session,
        config=config,
        trading_sessions=trading_sessions,
    )


def fetch_yfinance_earnings_payload(
    symbol: str,
    *,
    as_of: datetime,
    ticker_factory: Callable[[str], Any] | None = None,
    limit: int = 12,
) -> EarningsFetchResult:
    """Fetch the latest already-reported EPS row from yfinance.

    This adapter is deliberately thin and optional.  It returns unavailable on
    provider/import/schema failures and never substitutes estimates for actual
    results.  yfinance's earnings-date timestamp is marked ``scheduled`` so the
    interpreter reduces confidence and exposes the timestamp-quality flag.
    Tests should inject ``ticker_factory`` and must not use the network.
    """
    normalized_symbol = str(symbol).strip().upper()
    if not normalized_symbol:
        return EarningsFetchResult(False, error_code="invalid_symbol")
    if as_of.tzinfo is None:
        return EarningsFetchResult(False, error_code="naive_as_of")
    if not isinstance(limit, int) or limit <= 0:
        return EarningsFetchResult(False, error_code="invalid_limit")

    factory = ticker_factory
    if factory is None:
        try:
            import yfinance as yf
        except ImportError:
            return EarningsFetchResult(
                False, error_code="provider_dependency_unavailable"
            )
        factory = yf.Ticker

    try:
        earnings_dates = factory(normalized_symbol).get_earnings_dates(
            limit=limit
        )
        if earnings_dates is None or getattr(earnings_dates, "empty", True):
            return EarningsFetchResult(False, error_code="no_earnings_data")

        candidates: list[tuple[datetime, Any]] = []
        for row_index, row in earnings_dates.iterrows():
            reported_at = _coerce_provider_datetime(row_index)
            if reported_at is None or reported_at > as_of:
                continue
            actual = _row_value(row, ("Reported EPS", "reported_eps"))
            if _coerce_number(actual) is None:
                continue
            candidates.append((reported_at, row))
        if not candidates:
            return EarningsFetchResult(
                False, error_code="no_reported_earnings_as_of"
            )

        reported_at, row = max(candidates, key=lambda item: item[0])
        payload = {
            "symbol": normalized_symbol,
            "reported_at": reported_at.isoformat(),
            "eps": {
                "actual": _row_value(
                    row, ("Reported EPS", "reported_eps")
                ),
                "estimate": _row_value(
                    row, ("EPS Estimate", "estimated_eps")
                ),
            },
            "source": "yfinance",
            "timestamp_quality": "scheduled",
        }
        return EarningsFetchResult(True, payload=payload)
    except Exception as exc:  # Provider failures are contained and observable.
        LOGGER.warning(
            "yfinance earnings fetch failed for %s (%s): %s",
            normalized_symbol,
            type(exc).__name__,
            exc,
        )
        return EarningsFetchResult(
            False,
            error_code=f"provider_error:{type(exc).__name__}",
        )


def _blocked_signal(
    *,
    symbol: str,
    summary: str,
    blocker: str,
    reported_at: datetime,
    effective_session: date,
    decision_session: date,
) -> EarningsSignal:
    return EarningsSignal(
        symbol=symbol,
        event_flag=0,
        event_score=0.0,
        confidence=0.0,
        summary=summary,
        outcome="unavailable",
        reported_at=reported_at,
        effective_session=effective_session,
        decision_session=decision_session,
        blockers=(blocker,),
    )


def _parse_metric(
    raw: Any,
    *,
    flat_actual: Any,
    flat_estimate: Any,
    name: str,
    flags: list[str],
) -> ReportedMetric:
    if raw is None:
        actual_raw = flat_actual
        estimate_raw = flat_estimate
    elif isinstance(raw, Mapping):
        actual_raw = _first_present(raw, ("actual", "reported", "value"))
        estimate_raw = _first_present(
            raw, ("estimate", "consensus", "expected")
        )
    else:
        flags.append(f"{name}_data_invalid")
        actual_raw = flat_actual
        estimate_raw = flat_estimate

    actual = _optional_number(actual_raw, f"{name}_actual", flags)
    estimate = _optional_number(estimate_raw, f"{name}_estimate", flags)
    return ReportedMetric(actual=actual, estimate=estimate)


def _parse_guidance(raw: Any, flags: list[str]) -> Guidance:
    if raw is None:
        return Guidance()
    if isinstance(raw, str):
        direction = _DIRECTION_ALIASES.get(raw.strip().lower())
        if direction is None:
            flags.append("guidance_direction_unrecognized")
        return Guidance(direction=direction)
    if not isinstance(raw, Mapping):
        flags.append("guidance_data_invalid")
        return Guidance()

    direction_raw = _first_present(raw, ("direction", "status", "change"))
    direction: str | None = None
    if direction_raw is not None and str(direction_raw).strip():
        direction = _DIRECTION_ALIASES.get(
            str(direction_raw).strip().lower()
        )
        if direction is None:
            flags.append("guidance_direction_unrecognized")
    actual = _optional_number(
        _first_present(raw, ("actual", "value", "midpoint")),
        "guidance_actual",
        flags,
    )
    estimate = _optional_number(
        _first_present(raw, ("estimate", "consensus", "expected")),
        "guidance_estimate",
        flags,
    )
    return Guidance(
        direction=direction,
        actual=actual,
        estimate=estimate,
    )


def _first_present(mapping: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _optional_number(
    value: Any,
    field_name: str,
    flags: list[str],
) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    converted = _coerce_number(value)
    if converted is None:
        flags.append(f"{field_name}_invalid")
    return converted


def _coerce_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _parse_datetime(
    value: Any,
    *,
    field_name: str,
    default_timezone: str | None,
) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        normalized = value.strip()
        if "T" not in normalized and " " not in normalized:
            raise EarningsDataError(
                f"{field_name} must include an event time"
            )
        if normalized.endswith("Z"):
            normalized = f"{normalized[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise EarningsDataError(
                f"{field_name} must be an ISO timestamp"
            ) from exc
    else:
        raise EarningsDataError(f"{field_name} is required")

    if parsed.tzinfo is None:
        if not default_timezone:
            raise EarningsDataError(
                f"{field_name} must include a timezone"
            )
        try:
            parsed = parsed.replace(tzinfo=ZoneInfo(default_timezone))
        except ZoneInfoNotFoundError as exc:
            raise EarningsDataError(
                f"unknown timezone: {default_timezone!r}"
            ) from exc
    return parsed


def _parse_date(value: date | datetime | str, field_name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an ISO date") from exc
    raise ValueError(f"{field_name} must be a date")


def _normalized_sessions(
    values: Iterable[date | datetime | str] | None,
) -> tuple[date, ...] | None:
    if values is None:
        return None
    sessions = tuple(
        sorted({_parse_date(value, "trading_session") for value in values})
    )
    if not sessions:
        raise ValueError("trading_sessions cannot be empty")
    return sessions


def _is_session(value: date, sessions: tuple[date, ...] | None) -> bool:
    if sessions is not None:
        return value in sessions
    return value.weekday() < 5


def _next_session_after(
    value: date,
    sessions: tuple[date, ...] | None,
) -> date:
    if sessions is not None:
        for session in sessions:
            if session > value:
                return session
        raise ValueError(
            "trading_sessions do not include a session after the event"
        )
    candidate = value + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def _decision_session_for_as_of(
    as_of: datetime,
    config: EarningsInterpretationConfig,
    sessions: tuple[date, ...] | None,
) -> date:
    local = as_of.astimezone(ZoneInfo(config.market_timezone))
    if local.time() < config.market_open and _is_session(
        local.date(), sessions
    ):
        return local.date()
    return _next_session_after(local.date(), sessions)


def _session_distance(
    start: date,
    end: date,
    sessions: tuple[date, ...] | None,
) -> int:
    if end < start:
        raise ValueError("decision_session cannot precede effective_session")
    if sessions is not None:
        eligible = [
            session for session in sessions if start <= session <= end
        ]
        if not eligible or eligible[0] != start or eligible[-1] != end:
            raise ValueError(
                "trading_sessions must include effective and decision sessions"
            )
        return len(eligible) - 1

    count = 0
    cursor = start
    while cursor < end:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            count += 1
    return count


def _is_us_equity_rule_session(value: date) -> bool:
    if value.weekday() >= 5:
        return False
    holidays: set[date] = set()
    for year in (value.year - 1, value.year, value.year + 1):
        holidays.update(_us_equity_holidays(year))
    return value not in holidays


def _us_equity_holidays(year: int) -> set[date]:
    holidays = {
        _observed_fixed_holiday(date(year, 1, 1)),
        _nth_weekday(year, 2, weekday=0, occurrence=3),
        _gregorian_easter(year) - timedelta(days=2),
        _last_weekday(year, 5, weekday=0),
        _observed_fixed_holiday(date(year, 7, 4)),
        _nth_weekday(year, 9, weekday=0, occurrence=1),
        _nth_weekday(year, 11, weekday=3, occurrence=4),
        _observed_fixed_holiday(date(year, 12, 25)),
    }
    if year >= 1998:
        holidays.add(_nth_weekday(year, 1, weekday=0, occurrence=3))
    if year >= 2022:
        holidays.add(_observed_fixed_holiday(date(year, 6, 19)))
    return holidays


def _observed_fixed_holiday(value: date) -> date:
    if value.weekday() == 5:
        return value - timedelta(days=1)
    if value.weekday() == 6:
        return value + timedelta(days=1)
    return value


def _nth_weekday(
    year: int,
    month: int,
    *,
    weekday: int,
    occurrence: int,
) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (occurrence - 1))


def _last_weekday(year: int, month: int, *, weekday: int) -> date:
    if month == 12:
        cursor = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        cursor = date(year, month + 1, 1) - timedelta(days=1)
    return cursor - timedelta(days=(cursor.weekday() - weekday) % 7)


def _gregorian_easter(year: int) -> date:
    """Meeus/Jones/Butcher Gregorian Easter calculation."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = (h + ell - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def _metric_surprise(
    metric: ReportedMetric,
    name: str,
    flags: list[str],
    *,
    flag_absent: bool = True,
) -> float | None:
    if metric.actual is None:
        if metric.estimate is not None or flag_absent:
            flags.append(f"{name}_actual_missing")
        return None
    if metric.estimate is None:
        flags.append(f"{name}_estimate_missing")
        return None
    if metric.estimate == 0.0:
        flags.append(f"{name}_estimate_zero")
        return None
    return (metric.actual - metric.estimate) / abs(metric.estimate) * 100.0


def _surprise_score(surprise_pct: float, scale_pct: float) -> float:
    return _clip(math.tanh(surprise_pct / scale_pct), -1.0, 1.0)


def _score_tolerance(tolerance_pct: float, scale_pct: float) -> float:
    return abs(_surprise_score(tolerance_pct, scale_pct))


def _combine_guidance_scores(
    direction_score: float | None,
    quantitative_score: float | None,
    flags: list[str],
) -> float | None:
    if direction_score is None:
        return quantitative_score
    if quantitative_score is None:
        return direction_score
    if (
        abs(direction_score) > 1e-12
        and abs(quantitative_score) > 1e-12
        and math.copysign(1.0, direction_score)
        != math.copysign(1.0, quantitative_score)
    ):
        flags.append("guidance_signal_conflict")
    return (direction_score + quantitative_score) / 2.0


def _component_label(score: float, tolerance: float) -> str:
    if score > tolerance:
        return "positive"
    if score < -tolerance:
        return "negative"
    return "in_line"


def _overall_outcome(labels: Iterable[str]) -> str:
    label_set = set(labels)
    if "positive" in label_set and "negative" in label_set:
        return "mixed"
    if "positive" in label_set:
        return "beat"
    if "negative" in label_set:
        return "miss"
    return "in_line"


def _quality_factor(flags: Iterable[str]) -> float:
    factor = 1.0
    for flag in set(flags):
        if flag == "imprecise_report_timestamp":
            factor *= 0.80
        elif flag == "weekday_calendar_approximation":
            factor *= 0.95
        elif flag in {
            "guidance_signal_conflict",
            "guidance_direction_unrecognized",
            "timestamp_quality_unknown",
        }:
            factor *= 0.85
        elif flag.endswith("_invalid"):
            factor *= 0.80
    return max(0.25, factor)


def _build_summary(
    event: EarningsEvent,
    outcome: str,
    eps_surprise: float | None,
    revenue_surprise: float | None,
    guidance_surprise: float | None,
) -> str:
    period = f" {event.fiscal_period}" if event.fiscal_period else ""
    details: list[str] = []
    if eps_surprise is not None:
        details.append(f"EPS {_format_surprise(eps_surprise)}")
    if revenue_surprise is not None:
        details.append(f"revenue {_format_surprise(revenue_surprise)}")
    if event.guidance.direction is not None:
        details.append(f"guidance {event.guidance.direction}")
    elif guidance_surprise is not None:
        details.append(f"guidance {_format_surprise(guidance_surprise)}")
    joined = "; ".join(details)
    return (
        f"{event.symbol}{period} earnings: {joined}. "
        f"Overall result: {outcome.replace('_', ' ')}."
    )


def _format_surprise(value: float) -> str:
    if value > 0.0:
        return f"beat by {value:.1f}%"
    if value < 0.0:
        return f"missed by {abs(value):.1f}%"
    return "was in line"


def _quality_flag_for_calendar(
    sessions: tuple[date, ...] | None,
) -> tuple[str, ...]:
    return () if sessions is not None else ("weekday_calendar_approximation",)


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


def _coerce_provider_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif hasattr(value, "to_pydatetime"):
        parsed = value.to_pydatetime()
    else:
        try:
            parsed = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(DEFAULT_MARKET_TIMEZONE))
    return parsed


def _row_value(row: Any, names: Iterable[str]) -> Any:
    for name in names:
        try:
            if name in row:
                return row[name]
        except TypeError:
            continue
    return None
