"""Append-only, point-in-time records for market-agent evaluation.

The ledger deliberately stores predictions and outcomes as separate immutable
events.  A prediction is written before its target matures; its outcome may be
appended only on or after the recorded target session.  This prevents a later
result from silently changing the forecast that was actually available at
decision time.

Records are stored in a hash-chained JSON Lines file.  The chain is not a
substitute for access controls, but it makes accidental edits, truncation in the
middle of the file, and reordered records detectable during reads.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
import hashlib
import json
import math
from numbers import Integral
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, TextIO
from uuid import uuid4
from zoneinfo import ZoneInfo

from .earnings import us_equity_trading_sessions


UTC = timezone.utc
SCHEMA_VERSION = 2
DEFAULT_EXCHANGE_TIMEZONE = ZoneInfo("America/New_York")
DEFAULT_SESSION_CLOSE_HOUR = 16
DEFAULT_SESSION_OPEN = time(9, 30)


class LedgerError(RuntimeError):
    """Base class for prediction-ledger errors."""


class LedgerIntegrityError(LedgerError):
    """Raised when the on-disk append-only hash chain is invalid."""


class DuplicateLedgerRecordError(LedgerError):
    """Raised when a prediction or outcome identifier already exists."""


class UnknownPredictionError(LedgerError):
    """Raised when an outcome references a prediction not in the ledger."""


class ImmatureOutcomeError(LedgerError):
    """Raised when an outcome is recorded before its target session."""


def _default_target_maturity_utc(target_session: date) -> datetime:
    """Return regular US equity close for a target session."""
    local_close = datetime(
        target_session.year,
        target_session.month,
        target_session.day,
        DEFAULT_SESSION_CLOSE_HOUR,
        0,
        tzinfo=DEFAULT_EXCHANGE_TIMEZONE,
    )
    return local_close.astimezone(UTC)


def session_close_utc(session: date | str) -> datetime:
    """Return the regular US-equity close timestamp for a session."""
    return _default_target_maturity_utc(
        _as_date(session, "session")
    )


def _next_session_open_utc(as_of_session: date) -> datetime:
    calendar = us_equity_trading_sessions(
        as_of=datetime(
            as_of_session.year,
            as_of_session.month,
            as_of_session.day,
            12,
            tzinfo=DEFAULT_EXCHANGE_TIMEZONE,
        ),
        observed_sessions=(as_of_session,),
        future_session_count=2,
    )
    next_session = next(
        session for session in calendar if session > as_of_session
    )
    return datetime(
        next_session.year,
        next_session.month,
        next_session.day,
        DEFAULT_SESSION_OPEN.hour,
        DEFAULT_SESSION_OPEN.minute,
        tzinfo=DEFAULT_EXCHANGE_TIMEZONE,
    ).astimezone(UTC)


def _expected_target_session(
    as_of_session: date,
    horizon_sessions: int,
) -> date:
    calendar = us_equity_trading_sessions(
        as_of=datetime(
            as_of_session.year,
            as_of_session.month,
            as_of_session.day,
            12,
            tzinfo=DEFAULT_EXCHANGE_TIMEZONE,
        ),
        observed_sessions=None,
        future_session_count=horizon_sessions + 1,
    )
    if as_of_session not in calendar:
        raise ValueError(
            "as_of_session must be a regular US-equity trading session."
        )
    future = [
        session for session in calendar if session > as_of_session
    ]
    return future[horizon_sessions - 1]


def _as_date(value: date | str, field_name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO date.") from exc


def _as_utc_datetime(value: datetime | str, field_name: str) -> datetime:
    if isinstance(value, str):
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(
                f"{field_name} must be an ISO-8601 timestamp."
            ) from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise TypeError(f"{field_name} must be a datetime or ISO timestamp.")

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone.")
    return parsed.astimezone(UTC)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _finite_number(
    value: float | int | None,
    field_name: str,
) -> float | None:
    if value is None:
        return None
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} must be finite.")
    return converted


def _nonempty(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized


def _json_safe_mapping(
    value: Mapping[str, Any] | None,
    field_name: str,
) -> dict[str, Any]:
    converted = dict(value or {})
    try:
        json.dumps(converted, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain JSON-safe values.") from exc
    return converted


@dataclass(frozen=True)
class PredictionRecord:
    """A forecast and policy decision exactly as known at decision time.

    Returns and ``target_weight`` are decimal fractions of portfolio equity
    (``0.05`` means five percent). ``target_weight`` is long-only.
    """

    prediction_id: str
    created_at_utc: datetime
    data_cutoff_utc: datetime
    as_of_session: date
    target_session: date
    symbol: str
    horizon_sessions: int
    model_name: str
    model_version: str
    forecast_return: float
    target_weight: float
    return_start_session: date | None = None
    target_maturity_utc: datetime | None = None
    benchmark_symbol: str = "SPY"
    policy_version: str | None = None
    probability_positive: float | None = None
    lower_bound_return: float | None = None
    upper_bound_return: float | None = None
    feature_set_version: str | None = None
    feature_hash: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "prediction_id", _nonempty(self.prediction_id, "prediction_id")
        )
        object.__setattr__(
            self,
            "created_at_utc",
            _as_utc_datetime(self.created_at_utc, "created_at_utc"),
        )
        object.__setattr__(
            self,
            "data_cutoff_utc",
            _as_utc_datetime(self.data_cutoff_utc, "data_cutoff_utc"),
        )
        object.__setattr__(
            self, "as_of_session", _as_date(self.as_of_session, "as_of_session")
        )
        object.__setattr__(
            self,
            "target_session",
            _as_date(self.target_session, "target_session"),
        )
        return_start_session = (
            _as_date(
                self.return_start_session,
                "return_start_session",
            )
            if self.return_start_session is not None
            else self.as_of_session
        )
        object.__setattr__(
            self,
            "return_start_session",
            return_start_session,
        )
        maturity = (
            _as_utc_datetime(self.target_maturity_utc, "target_maturity_utc")
            if self.target_maturity_utc is not None
            else _default_target_maturity_utc(self.target_session)
        )
        object.__setattr__(self, "target_maturity_utc", maturity)
        object.__setattr__(self, "symbol", _nonempty(self.symbol, "symbol").upper())
        object.__setattr__(
            self, "model_name", _nonempty(self.model_name, "model_name")
        )
        object.__setattr__(
            self, "model_version", _nonempty(self.model_version, "model_version")
        )
        object.__setattr__(
            self,
            "benchmark_symbol",
            _nonempty(self.benchmark_symbol, "benchmark_symbol").upper(),
        )
        for field_name in (
            "policy_version",
            "feature_set_version",
            "feature_hash",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _nonempty(value, field_name),
                )
        object.__setattr__(
            self,
            "forecast_return",
            _finite_number(self.forecast_return, "forecast_return"),
        )
        object.__setattr__(
            self,
            "target_weight",
            _finite_number(self.target_weight, "target_weight"),
        )
        object.__setattr__(
            self,
            "probability_positive",
            _finite_number(self.probability_positive, "probability_positive"),
        )
        object.__setattr__(
            self,
            "lower_bound_return",
            _finite_number(self.lower_bound_return, "lower_bound_return"),
        )
        object.__setattr__(
            self,
            "upper_bound_return",
            _finite_number(self.upper_bound_return, "upper_bound_return"),
        )
        object.__setattr__(
            self, "metadata", _json_safe_mapping(self.metadata, "metadata")
        )

        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported prediction schema version {self.schema_version}."
            )
        if isinstance(self.horizon_sessions, bool) or not isinstance(
            self.horizon_sessions, Integral
        ):
            raise ValueError("horizon_sessions must be an integer.")
        if self.horizon_sessions <= 0:
            raise ValueError("horizon_sessions must be positive.")
        object.__setattr__(
            self,
            "horizon_sessions",
            int(self.horizon_sessions),
        )
        if self.target_session <= self.as_of_session:
            raise ValueError("target_session must be after as_of_session.")
        if self.return_start_session < self.as_of_session:
            raise ValueError(
                "return_start_session cannot precede as_of_session."
            )
        if self.return_start_session >= self.target_session:
            raise ValueError(
                "return_start_session must precede target_session."
            )
        expected_target = _expected_target_session(
            self.return_start_session,
            self.horizon_sessions,
        )
        if self.target_session != expected_target:
            raise ValueError(
                "target_session must be exactly horizon_sessions US-equity "
                "sessions after return_start_session; "
                f"expected {expected_target}."
            )
        if self.data_cutoff_utc > self.created_at_utc:
            raise ValueError("data_cutoff_utc cannot be after created_at_utc.")
        publication_deadline = _next_session_open_utc(
            self.as_of_session
        )
        if self.created_at_utc > publication_deadline:
            raise ValueError(
                "created_at_utc is after the next-session open; historical "
                "predictions cannot be backfilled after returns begin."
            )
        if self.data_cutoff_utc > publication_deadline:
            raise ValueError(
                "data_cutoff_utc is after the point-in-time decision deadline."
            )
        if self.created_at_utc >= self.target_maturity_utc:
            raise ValueError(
                "created_at_utc must be before target_maturity_utc; "
                "matured predictions cannot be backfilled."
            )
        if not 0.0 <= self.target_weight <= 1.0:
            raise ValueError("target_weight must be between 0 and 1.")
        if (
            self.probability_positive is not None
            and not 0.0 <= self.probability_positive <= 1.0
        ):
            raise ValueError("probability_positive must be between 0 and 1.")
        if (
            self.lower_bound_return is not None
            and self.upper_bound_return is not None
            and self.lower_bound_return > self.upper_bound_return
        ):
            raise ValueError(
                "lower_bound_return cannot exceed upper_bound_return."
            )

    @classmethod
    def create(
        cls,
        *,
        created_at_utc: datetime,
        data_cutoff_utc: datetime,
        as_of_session: date,
        target_session: date,
        symbol: str,
        horizon_sessions: int,
        model_name: str,
        model_version: str,
        forecast_return: float,
        target_weight: float,
        **kwargs: Any,
    ) -> "PredictionRecord":
        """Create a prediction with a unique immutable identifier."""

        return cls(
            prediction_id=f"pred-{uuid4()}",
            created_at_utc=created_at_utc,
            data_cutoff_utc=data_cutoff_utc,
            as_of_session=as_of_session,
            target_session=target_session,
            symbol=symbol,
            horizon_sessions=horizon_sessions,
            model_name=model_name,
            model_version=model_version,
            forecast_return=forecast_return,
            target_weight=target_weight,
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable representation."""

        return {
            "record_type": "prediction",
            "schema_version": self.schema_version,
            "prediction_id": self.prediction_id,
            "created_at_utc": _iso_utc(self.created_at_utc),
            "data_cutoff_utc": _iso_utc(self.data_cutoff_utc),
            "as_of_session": self.as_of_session.isoformat(),
            "return_start_session": (
                self.return_start_session.isoformat()
            ),
            "target_session": self.target_session.isoformat(),
            "target_maturity_utc": _iso_utc(self.target_maturity_utc),
            "symbol": self.symbol,
            "horizon_sessions": self.horizon_sessions,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "policy_version": self.policy_version,
            "forecast_return": self.forecast_return,
            "target_weight": self.target_weight,
            "benchmark_symbol": self.benchmark_symbol,
            "probability_positive": self.probability_positive,
            "lower_bound_return": self.lower_bound_return,
            "upper_bound_return": self.upper_bound_return,
            "feature_set_version": self.feature_set_version,
            "feature_hash": self.feature_hash,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PredictionRecord":
        """Parse a prediction from its ledger representation."""

        if value.get("record_type") != "prediction":
            raise ValueError("Expected a prediction record.")
        fields = dict(value)
        fields.pop("record_type", None)
        return cls(**fields)


@dataclass(frozen=True)
class OutcomeRecord:
    """A matured realized result linked to one immutable prediction."""

    outcome_id: str
    prediction_id: str
    recorded_at_utc: datetime
    target_session: date
    realized_return: float
    benchmark_return: float
    target_maturity_utc: datetime | None = None
    transaction_cost_return: float = 0.0
    entry_price: float | None = None
    exit_price: float | None = None
    max_adverse_excursion: float | None = None
    max_favorable_excursion: float | None = None
    stop_breached: bool | None = None
    data_source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "outcome_id", _nonempty(self.outcome_id, "outcome_id")
        )
        object.__setattr__(
            self, "prediction_id", _nonempty(self.prediction_id, "prediction_id")
        )
        object.__setattr__(
            self,
            "recorded_at_utc",
            _as_utc_datetime(self.recorded_at_utc, "recorded_at_utc"),
        )
        object.__setattr__(
            self,
            "target_session",
            _as_date(self.target_session, "target_session"),
        )
        maturity = (
            _as_utc_datetime(self.target_maturity_utc, "target_maturity_utc")
            if self.target_maturity_utc is not None
            else _default_target_maturity_utc(self.target_session)
        )
        object.__setattr__(self, "target_maturity_utc", maturity)
        for field_name in (
            "realized_return",
            "benchmark_return",
            "transaction_cost_return",
            "entry_price",
            "exit_price",
            "max_adverse_excursion",
            "max_favorable_excursion",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_number(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self, "metadata", _json_safe_mapping(self.metadata, "metadata")
        )
        if self.data_source is not None:
            object.__setattr__(
                self,
                "data_source",
                _nonempty(self.data_source, "data_source"),
            )

        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported outcome schema version {self.schema_version}."
            )
        if self.recorded_at_utc < self.target_maturity_utc:
            raise ImmatureOutcomeError(
                "recorded_at_utc cannot be before target-session maturity."
            )
        if self.transaction_cost_return < 0.0:
            raise ValueError("transaction_cost_return cannot be negative.")
        if self.stop_breached is not None and not isinstance(
            self.stop_breached, bool
        ):
            raise ValueError("stop_breached must be a boolean when provided.")
        if self.entry_price is not None and self.entry_price <= 0.0:
            raise ValueError("entry_price must be positive.")
        if self.exit_price is not None and self.exit_price <= 0.0:
            raise ValueError("exit_price must be positive.")
        if (
            self.max_adverse_excursion is not None
            and self.max_adverse_excursion > 0.0
        ):
            raise ValueError("max_adverse_excursion must be zero or negative.")
        if (
            self.max_favorable_excursion is not None
            and self.max_favorable_excursion < 0.0
        ):
            raise ValueError("max_favorable_excursion must be zero or positive.")

    @classmethod
    def create(
        cls,
        *,
        prediction_id: str,
        recorded_at_utc: datetime,
        target_session: date,
        realized_return: float,
        benchmark_return: float,
        **kwargs: Any,
    ) -> "OutcomeRecord":
        """Create an outcome with a unique immutable identifier."""

        return cls(
            outcome_id=f"outcome-{uuid4()}",
            prediction_id=prediction_id,
            recorded_at_utc=recorded_at_utc,
            target_session=target_session,
            realized_return=realized_return,
            benchmark_return=benchmark_return,
            **kwargs,
        )

    @property
    def net_return(self) -> float:
        """Realized asset return after the recorded execution cost."""

        return self.realized_return - self.transaction_cost_return

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable representation."""

        return {
            "record_type": "outcome",
            "schema_version": self.schema_version,
            "outcome_id": self.outcome_id,
            "prediction_id": self.prediction_id,
            "recorded_at_utc": _iso_utc(self.recorded_at_utc),
            "target_session": self.target_session.isoformat(),
            "target_maturity_utc": _iso_utc(self.target_maturity_utc),
            "realized_return": self.realized_return,
            "benchmark_return": self.benchmark_return,
            "transaction_cost_return": self.transaction_cost_return,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "max_adverse_excursion": self.max_adverse_excursion,
            "max_favorable_excursion": self.max_favorable_excursion,
            "stop_breached": self.stop_breached,
            "data_source": self.data_source,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OutcomeRecord":
        """Parse an outcome from its ledger representation."""

        if value.get("record_type") != "outcome":
            raise ValueError("Expected an outcome record.")
        fields = dict(value)
        fields.pop("record_type", None)
        return cls(**fields)


LedgerRecord = PredictionRecord | OutcomeRecord


@dataclass(frozen=True)
class LedgerEntry:
    """One verified event from the append-only file."""

    sequence: int
    previous_hash: str | None
    record_hash: str
    record: LedgerRecord


@dataclass(frozen=True)
class CompletedPrediction:
    """A point-in-time prediction joined to its matured outcome."""

    prediction: PredictionRecord
    outcome: OutcomeRecord


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _entry_hash(body: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()


@contextmanager
def _locked(handle: TextIO, *, exclusive: bool) -> Iterator[None]:
    """Use an advisory file lock where the platform provides ``fcntl``."""

    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows fallback
        yield
        return

    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    fcntl.flock(handle.fileno(), operation)
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class PredictionLedger:
    """Hash-chained JSONL prediction/outcome ledger.

    The class intentionally exposes append and read operations only.  Corrections
    should be represented as a new versioned prediction, never by editing an
    existing event.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        if self.path.exists() and self.path.is_dir():
            raise ValueError("Ledger path must be a file, not a directory.")

    def append_prediction(self, record: PredictionRecord) -> LedgerEntry:
        """Append one immutable prediction, rejecting duplicate identifiers."""

        if not isinstance(record, PredictionRecord):
            raise TypeError("record must be a PredictionRecord.")
        return self._append(record)

    def append_outcome(self, record: OutcomeRecord) -> LedgerEntry:
        """Append a matured outcome after validating its linked prediction."""

        if not isinstance(record, OutcomeRecord):
            raise TypeError("record must be an OutcomeRecord.")
        return self._append(record)

    def read_entries(self) -> tuple[LedgerEntry, ...]:
        """Read all events and verify the complete hash chain."""

        if not self.path.exists():
            return ()
        with self.path.open("r", encoding="utf-8") as handle:
            with _locked(handle, exclusive=False):
                return self._read_entries_from_handle(handle)

    def predictions(
        self,
        *,
        horizon_sessions: int | None = None,
    ) -> tuple[PredictionRecord, ...]:
        """Return predictions, optionally restricted to one forecast horizon."""

        records = (
            entry.record
            for entry in self.read_entries()
            if isinstance(entry.record, PredictionRecord)
        )
        return tuple(
            record
            for record in records
            if horizon_sessions is None
            or record.horizon_sessions == horizon_sessions
        )

    def outcomes(self) -> tuple[OutcomeRecord, ...]:
        """Return all outcome events."""

        return tuple(
            entry.record
            for entry in self.read_entries()
            if isinstance(entry.record, OutcomeRecord)
        )

    def matured_predictions(
        self,
        as_of_session: date | str,
        *,
        horizon_sessions: int | None = None,
    ) -> tuple[PredictionRecord, ...]:
        """Return predictions whose recorded target session has passed."""

        cutoff = _as_date(as_of_session, "as_of_session")
        return tuple(
            prediction
            for prediction in self.predictions(horizon_sessions=horizon_sessions)
            if prediction.target_session <= cutoff
        )

    def pending_outcomes(
        self,
        as_of_session: date | str,
        *,
        horizon_sessions: int | None = None,
    ) -> tuple[PredictionRecord, ...]:
        """Return matured predictions that still lack an outcome event."""

        outcome_ids = {outcome.prediction_id for outcome in self.outcomes()}
        return tuple(
            prediction
            for prediction in self.matured_predictions(
                as_of_session,
                horizon_sessions=horizon_sessions,
            )
            if prediction.prediction_id not in outcome_ids
        )

    def completed_predictions(
        self,
        as_of_session: date | str,
        *,
        horizon_sessions: int,
    ) -> tuple[CompletedPrediction, ...]:
        """Join matured predictions to outcomes for exactly one horizon."""

        cutoff = _as_date(as_of_session, "as_of_session")
        outcomes = {
            outcome.prediction_id: outcome for outcome in self.outcomes()
        }
        completed: list[CompletedPrediction] = []
        for prediction in self.predictions(horizon_sessions=horizon_sessions):
            if prediction.target_session > cutoff:
                continue
            outcome = outcomes.get(prediction.prediction_id)
            if outcome is not None:
                completed.append(
                    CompletedPrediction(prediction=prediction, outcome=outcome)
                )
        return tuple(completed)

    def _append(self, record: LedgerRecord) -> LedgerEntry:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as handle:
            with _locked(handle, exclusive=True):
                entries = self._read_entries_from_handle(handle)
                predictions = {
                    entry.record.prediction_id: entry.record
                    for entry in entries
                    if isinstance(entry.record, PredictionRecord)
                }
                outcomes_by_prediction = {
                    entry.record.prediction_id: entry.record
                    for entry in entries
                    if isinstance(entry.record, OutcomeRecord)
                }
                record_ids = {
                    record_id
                    for entry in entries
                    for record_id in (
                        getattr(entry.record, "prediction_id", None),
                        getattr(entry.record, "outcome_id", None),
                    )
                    if record_id is not None
                }

                own_id = (
                    record.prediction_id
                    if isinstance(record, PredictionRecord)
                    else record.outcome_id
                )
                if own_id in record_ids:
                    raise DuplicateLedgerRecordError(
                        f"Ledger record {own_id!r} already exists."
                    )

                if isinstance(record, PredictionRecord):
                    if record.prediction_id in predictions:
                        raise DuplicateLedgerRecordError(
                            f"Prediction {record.prediction_id!r} already exists."
                        )
                else:
                    prediction = predictions.get(record.prediction_id)
                    if prediction is None:
                        raise UnknownPredictionError(
                            f"Prediction {record.prediction_id!r} is not in the ledger."
                        )
                    if record.prediction_id in outcomes_by_prediction:
                        raise DuplicateLedgerRecordError(
                            "An outcome already exists for prediction "
                            f"{record.prediction_id!r}."
                        )
                    if record.target_session != prediction.target_session:
                        raise ValueError(
                            "Outcome target_session does not match its prediction."
                        )
                    if record.target_maturity_utc != prediction.target_maturity_utc:
                        raise ValueError(
                            "Outcome target_maturity_utc does not match its prediction."
                        )
                    if record.recorded_at_utc < prediction.target_maturity_utc:
                        raise ImmatureOutcomeError(
                            "Outcome cannot be appended before target maturity."
                        )

                sequence = len(entries) + 1
                previous_hash = entries[-1].record_hash if entries else None
                body = {
                    "sequence": sequence,
                    "previous_hash": previous_hash,
                    "record": record.to_dict(),
                }
                record_hash = _entry_hash(body)
                envelope = {**body, "record_hash": record_hash}
                handle.seek(0, os.SEEK_END)
                handle.write(_canonical_json(envelope) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                return LedgerEntry(
                    sequence=sequence,
                    previous_hash=previous_hash,
                    record_hash=record_hash,
                    record=record,
                )

    @staticmethod
    def _read_entries_from_handle(handle: TextIO) -> tuple[LedgerEntry, ...]:
        handle.seek(0)
        entries: list[LedgerEntry] = []
        expected_previous_hash: str | None = None
        expected_sequence = 1

        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                raise LedgerIntegrityError(
                    f"Blank ledger line at line {line_number}."
                )
            try:
                envelope = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerIntegrityError(
                    f"Invalid JSON at ledger line {line_number}."
                ) from exc
            if not isinstance(envelope, dict):
                raise LedgerIntegrityError(
                    f"Ledger line {line_number} is not a JSON object."
                )

            sequence = envelope.get("sequence")
            previous_hash = envelope.get("previous_hash")
            record_hash = envelope.get("record_hash")
            record_value = envelope.get("record")
            if sequence != expected_sequence:
                raise LedgerIntegrityError(
                    f"Unexpected sequence at ledger line {line_number}."
                )
            if previous_hash != expected_previous_hash:
                raise LedgerIntegrityError(
                    f"Broken hash chain at ledger line {line_number}."
                )
            if not isinstance(record_hash, str) or not isinstance(
                record_value, dict
            ):
                raise LedgerIntegrityError(
                    f"Incomplete envelope at ledger line {line_number}."
                )

            body = {
                "sequence": sequence,
                "previous_hash": previous_hash,
                "record": record_value,
            }
            if _entry_hash(body) != record_hash:
                raise LedgerIntegrityError(
                    f"Record hash mismatch at ledger line {line_number}."
                )

            record_type = record_value.get("record_type")
            try:
                if record_type == "prediction":
                    record: LedgerRecord = PredictionRecord.from_dict(record_value)
                elif record_type == "outcome":
                    record = OutcomeRecord.from_dict(record_value)
                else:
                    raise LedgerIntegrityError(
                        f"Unknown record type at ledger line {line_number}."
                    )
            except (TypeError, ValueError, LedgerError) as exc:
                if isinstance(exc, LedgerIntegrityError):
                    raise
                raise LedgerIntegrityError(
                    f"Invalid record at ledger line {line_number}: {exc}"
                ) from exc

            entries.append(
                LedgerEntry(
                    sequence=sequence,
                    previous_hash=previous_hash,
                    record_hash=record_hash,
                    record=record,
                )
            )
            expected_sequence += 1
            expected_previous_hash = record_hash

        return tuple(entries)
