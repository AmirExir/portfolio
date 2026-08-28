"""Leakage-resistant evaluation utilities for market forecasting policies.

This module keeps three concerns separate:

* purged, embargoed walk-forward folds for point-in-time model evaluation;
* forecast calibration/error metrics computed only from matured outcomes; and
* daily execution metrics and explicit promotion gates for a shadow policy.

All returns and weights are decimal fractions.  For example, ``0.02`` is a two
percent return and ``0.25`` is a 25 percent target weight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import hashlib
import json
import math
from numbers import Integral
import pickle
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .ledger import (
    CompletedPrediction,
    LedgerEntry,
    LedgerError,
    OutcomeRecord,
    PredictionLedger,
    PredictionRecord,
    SUPPORTED_SESSION_CALENDARS,
    US_EQUITY_SESSION_CALENDAR,
    prediction_strict_close_t_eligible,
)


UTC = timezone.utc
REQUIRED_NON_RL_BASELINES = frozenset({"ridge", "fixed-ensemble"})


class EvaluationError(RuntimeError):
    """Base class for evaluation failures."""


class DataLeakageError(EvaluationError):
    """Raised when a training label overlaps its test fold."""


class MixedHorizonError(EvaluationError):
    """Raised when records from different forecast horizons are combined."""


class MixedForecastCohortError(EvaluationError):
    """Raised when incomparable forecast provenance cohorts are pooled."""


def _as_date(value: date | datetime | str, field_name: str) -> date:
    if pd.isna(value):
        raise ValueError(f"{field_name} must not be missing.")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO date.") from exc


def _finite(value: Any, field_name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} must be finite.")
    return converted


def _positive_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field_name} must be an integer.")
    converted = int(value)
    if converted <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return converted


def _optional_exact_filter(
    value: str | None,
    field_name: str,
    *,
    case: str | None = None,
) -> str | None:
    """Normalize one optional exact-match provenance filter."""

    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty when provided.")
    if case == "lower":
        return normalized.lower()
    if case == "upper":
        return normalized.upper()
    return normalized


def _optional_session_calendar_filter(
    value: str | None,
    field_name: str,
) -> str | None:
    """Normalize and validate an optional supported session calendar."""

    normalized = _optional_exact_filter(value, field_name, case="lower")
    if normalized is not None and normalized not in SUPPORTED_SESSION_CALENDARS:
        supported = ", ".join(sorted(SUPPORTED_SESSION_CALENDARS))
        raise ValueError(f"{field_name} must be one of: {supported}.")
    return normalized


@dataclass(frozen=True)
class WalkForwardConfig:
    """Configuration for expanding or rolling walk-forward folds.

    ``purge_sessions`` is the gap between the last feature session admitted to
    training and the first test session.  ``embargo_sessions`` is the gap after a
    test block before the next test block may begin.  Both default to the target
    horizon and may not be shorter than it.
    """

    horizon_sessions: int
    minimum_training_sessions: int
    test_sessions: int
    purge_sessions: int | None = None
    embargo_sessions: int | None = None
    maximum_training_sessions: int | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "horizon_sessions",
            "minimum_training_sessions",
            "test_sessions",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive_integer(getattr(self, field_name), field_name),
            )

        purge = (
            self.horizon_sessions
            if self.purge_sessions is None
            else self.purge_sessions
        )
        embargo = (
            self.horizon_sessions
            if self.embargo_sessions is None
            else self.embargo_sessions
        )
        purge = _positive_integer(purge, "purge_sessions")
        embargo = _positive_integer(embargo, "embargo_sessions")
        if purge < self.horizon_sessions:
            raise ValueError(
                "purge_sessions must be at least horizon_sessions."
            )
        if embargo < self.horizon_sessions:
            raise ValueError(
                "embargo_sessions must be at least horizon_sessions."
            )
        if (
            self.maximum_training_sessions is not None
            and _positive_integer(
                self.maximum_training_sessions,
                "maximum_training_sessions",
            )
            < self.minimum_training_sessions
        ):
            raise ValueError(
                "maximum_training_sessions cannot be smaller than "
                "minimum_training_sessions."
            )
        object.__setattr__(self, "purge_sessions", int(purge))
        object.__setattr__(self, "embargo_sessions", int(embargo))
        if self.maximum_training_sessions is not None:
            object.__setattr__(
                self,
                "maximum_training_sessions",
                int(self.maximum_training_sessions),
            )


@dataclass(frozen=True)
class WalkForwardFold:
    """One chronological fold with an explicitly frozen policy test block."""

    fold_id: int
    horizon_sessions: int
    train_sessions: tuple[date, ...]
    purged_sessions: tuple[date, ...]
    test_sessions: tuple[date, ...]
    embargoed_sessions: tuple[date, ...]

    @property
    def train_start_session(self) -> date:
        return self.train_sessions[0]

    @property
    def train_end_session(self) -> date:
        return self.train_sessions[-1]

    @property
    def test_start_session(self) -> date:
        return self.test_sessions[0]

    @property
    def test_end_session(self) -> date:
        return self.test_sessions[-1]


def build_purged_walk_forward_folds(
    sessions: Sequence[date | datetime | str],
    config: WalkForwardConfig,
) -> tuple[WalkForwardFold, ...]:
    """Build ordered folds without mixing forecast horizons.

    The first test block starts only after the minimum training window and the
    full pre-test purge.  Later test blocks start after the preceding test block
    and embargo; the policy is fit once for each returned fold.
    """

    normalized = sorted(
        {_as_date(session, "session") for session in sessions}
    )
    first_test_start = (
        config.minimum_training_sessions + int(config.purge_sessions)
    )
    folds: list[WalkForwardFold] = []
    test_start = first_test_start

    while test_start + config.test_sessions <= len(normalized):
        train_end = test_start - int(config.purge_sessions)
        train_start = 0
        if config.maximum_training_sessions is not None:
            train_start = max(0, train_end - config.maximum_training_sessions)

        train = tuple(normalized[train_start:train_end])
        if len(train) < config.minimum_training_sessions:
            test_start += config.test_sessions + int(config.embargo_sessions)
            continue

        purge = tuple(normalized[train_end:test_start])
        test_end = test_start + config.test_sessions
        test = tuple(normalized[test_start:test_end])
        embargo_end = min(
            len(normalized),
            test_end + int(config.embargo_sessions),
        )
        embargo = tuple(normalized[test_end:embargo_end])
        folds.append(
            WalkForwardFold(
                fold_id=len(folds),
                horizon_sessions=config.horizon_sessions,
                train_sessions=train,
                purged_sessions=purge,
                test_sessions=test,
                embargoed_sessions=embargo,
            )
        )
        test_start = test_end + int(config.embargo_sessions)

    return tuple(folds)


@dataclass(frozen=True)
class FittedPolicy:
    """A versioned policy fitted once and held fixed for one test fold."""

    model: Any
    version: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.version).strip():
            raise ValueError("FittedPolicy.version must not be empty.")


@dataclass
class WalkForwardRun:
    """Outputs and trace metadata from a completed walk-forward run."""

    config: WalkForwardConfig
    folds: tuple[WalkForwardFold, ...]
    predictions: pd.DataFrame
    skipped_immature_fold_ids: tuple[int, ...]
    policy_versions: Mapping[int, str]
    policy_metadata: Mapping[int, Mapping[str, Any]]


FitPolicy = Callable[[pd.DataFrame, WalkForwardFold], FittedPolicy]
PredictPolicy = Callable[
    [Any, pd.DataFrame, WalkForwardFold],
    pd.DataFrame,
]


def run_frozen_walk_forward(
    frame: pd.DataFrame,
    config: WalkForwardConfig,
    *,
    feature_columns: Sequence[str],
    fit_policy: FitPolicy,
    predict_policy: PredictPolicy,
    session_column: str = "as_of_session",
    target_session_column: str = "target_session",
    horizon_column: str = "horizon_sessions",
    symbol_column: str = "symbol",
    outcome_columns: Sequence[str] = (
        "realized_return",
        "benchmark_return",
    ),
    evaluation_as_of_session: date | datetime | str | None = None,
) -> WalkForwardRun:
    """Fit once per fold and predict its entire matured test block once.

    Only the identity columns and explicit ``feature_columns`` are passed to the
    predictor.  Target/outcome columns are joined back after prediction.  A fold
    is omitted until every target in that fold has matured, avoiding partial-fold
    hindsight.  ``fit_policy`` receives only purged training rows whose labels
    mature before the test begins.

    The callback contract is intentionally strict: ``fit_policy`` must return a
    versioned :class:`FittedPolicy`, and ``predict_policy`` must perform inference
    without updating that object.
    """

    required = {
        session_column,
        target_session_column,
        horizon_column,
        symbol_column,
        *feature_columns,
        *outcome_columns,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing walk-forward columns: {missing}")

    leakage_features = set(feature_columns).intersection(
        {target_session_column, horizon_column, *outcome_columns}
    )
    if leakage_features:
        raise DataLeakageError(
            "Feature columns include target metadata or outcomes: "
            f"{sorted(leakage_features)}"
        )

    data = frame.copy()
    data[session_column] = data[session_column].map(
        lambda value: _as_date(value, session_column)
    )
    data[target_session_column] = data[target_session_column].map(
        lambda value: _as_date(value, target_session_column)
    )
    if (data[target_session_column] <= data[session_column]).any():
        raise ValueError(
            "Every target_session must be after its as_of_session."
        )

    numeric_horizons = pd.to_numeric(data[horizon_column], errors="raise")
    if not np.all(np.equal(numeric_horizons, np.floor(numeric_horizons))):
        raise ValueError("horizon_sessions values must be integers.")
    horizons = {int(value) for value in numeric_horizons}
    if horizons != {config.horizon_sessions}:
        raise MixedHorizonError(
            "Walk-forward data must contain exactly the configured horizon "
            f"{config.horizon_sessions}; found {sorted(horizons)}."
        )
    data[horizon_column] = numeric_horizons.astype(int)

    duplicate_keys = data.duplicated([session_column, symbol_column], keep=False)
    if duplicate_keys.any():
        raise ValueError(
            "Walk-forward data contains duplicate symbol/session rows."
        )
    data = data.sort_values(
        [session_column, symbol_column],
        kind="stable",
    ).reset_index(drop=True)

    cutoff = (
        _as_date(evaluation_as_of_session, "evaluation_as_of_session")
        if evaluation_as_of_session is not None
        else datetime.now(UTC).date()
    )
    all_folds = build_purged_walk_forward_folds(
        data[session_column].tolist(),
        config,
    )
    output_frames: list[pd.DataFrame] = []
    completed_folds: list[WalkForwardFold] = []
    skipped_folds: list[int] = []
    policy_versions: dict[int, str] = {}
    policy_metadata: dict[int, Mapping[str, Any]] = {}

    for fold in all_folds:
        test_mask = data[session_column].isin(fold.test_sessions)
        test_rows = data.loc[test_mask].copy()
        if test_rows.empty:
            continue
        if (test_rows[target_session_column] > cutoff).any():
            skipped_folds.append(fold.fold_id)
            continue

        train_mask = data[session_column].isin(fold.train_sessions)
        train_rows = data.loc[train_mask].copy()
        if train_rows.empty:
            raise EvaluationError(
                f"Fold {fold.fold_id} has no training observations."
            )
        if (
            train_rows[target_session_column] >= fold.test_start_session
        ).any():
            raise DataLeakageError(
                f"Fold {fold.fold_id} contains training labels that mature "
                "on or after its test start."
            )

        fitted = fit_policy(train_rows.copy(), fold)
        if not isinstance(fitted, FittedPolicy):
            raise TypeError("fit_policy must return a FittedPolicy.")

        identity_columns = [
            session_column,
            target_session_column,
            horizon_column,
            symbol_column,
        ]
        predictor_columns = [
            session_column,
            symbol_column,
            *feature_columns,
        ]
        predictor_frame = test_rows[predictor_columns].copy()
        model_state_before = _model_state_fingerprint(fitted.model)
        predicted = predict_policy(
            fitted.model,
            predictor_frame.copy(),
            fold,
        )
        model_state_after = _model_state_fingerprint(fitted.model)
        if model_state_after != model_state_before:
            raise DataLeakageError(
                f"Fold {fold.fold_id} predictor mutated the fitted policy."
            )
        if not isinstance(predicted, pd.DataFrame):
            raise TypeError("predict_policy must return a pandas DataFrame.")
        if len(predicted) != len(test_rows):
            raise ValueError(
                f"Fold {fold.fold_id} returned {len(predicted)} predictions "
                f"for {len(test_rows)} test rows."
            )
        prediction_keys = [session_column, symbol_column]
        missing_prediction_keys = sorted(set(prediction_keys) - set(predicted.columns))
        if missing_prediction_keys:
            raise ValueError(
                "Prediction output must include identity keys: "
                f"{missing_prediction_keys}"
            )
        predicted = predicted.copy()
        predicted[session_column] = predicted[session_column].map(
            lambda value: _as_date(value, session_column)
        )
        predicted[symbol_column] = predicted[symbol_column].astype(str).str.upper()
        if predicted.duplicated(prediction_keys, keep=False).any():
            raise ValueError(
                f"Fold {fold.fold_id} returned duplicate prediction identity keys."
            )
        expected_keys = test_rows[prediction_keys].copy()
        expected_keys[symbol_column] = expected_keys[symbol_column].astype(str).str.upper()
        key_check = expected_keys.merge(
            predicted[prediction_keys],
            on=prediction_keys,
            how="outer",
            indicator=True,
        )
        if not key_check["_merge"].eq("both").all():
            raise ValueError(
                f"Fold {fold.fold_id} prediction identity keys do not match test rows."
            )

        reserved_outputs = (
            set(predicted.columns) - set(prediction_keys)
        ).intersection(
            {*data.columns, "fold_id", "policy_version"}
        )
        if reserved_outputs:
            raise ValueError(
                "Prediction outputs collide with input columns: "
                f"{sorted(reserved_outputs)}"
            )

        fold_output = test_rows[
            [*identity_columns, *outcome_columns]
        ].copy()
        fold_output[symbol_column] = fold_output[symbol_column].astype(str).str.upper()
        prediction_values = predicted.drop(
            columns=[
                column
                for column in identity_columns
                if column in predicted.columns and column not in prediction_keys
            ],
            errors="ignore",
        )
        fold_output = fold_output.merge(
            prediction_values,
            on=prediction_keys,
            how="left",
            validate="one_to_one",
            sort=False,
        )
        fold_output["fold_id"] = fold.fold_id
        fold_output["policy_version"] = fitted.version
        output_frames.append(fold_output)
        completed_folds.append(fold)
        policy_versions[fold.fold_id] = fitted.version
        policy_metadata[fold.fold_id] = dict(fitted.metadata)

    if output_frames:
        predictions = pd.concat(output_frames, ignore_index=True)
    else:
        predictions = pd.DataFrame()

    return WalkForwardRun(
        config=config,
        folds=tuple(completed_folds),
        predictions=predictions,
        skipped_immature_fold_ids=tuple(skipped_folds),
        policy_versions=policy_versions,
        policy_metadata=policy_metadata,
    )


def _model_state_fingerprint(model: Any) -> str:
    """Fingerprint model state before and after inference.

    Pickle covers ordinary estimators and containers. A deterministic fallback
    based on the object's public state keeps the frozen-policy contract useful
    for lightweight custom test models.
    """
    try:
        payload = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        try:
            payload = repr(vars(model)).encode("utf-8")
        except Exception as exc:
            raise TypeError(
                "Fitted policy must expose fingerprintable immutable state."
            ) from exc
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ForecastObservation:
    """One matured forecast and its target-horizon realized return."""

    prediction_id: str
    as_of_session: date
    target_session: date
    symbol: str
    horizon_sessions: int
    predicted_return: float
    realized_return: float
    benchmark_return: float
    probability_positive: float | None = None
    lower_bound_return: float | None = None
    upper_bound_return: float | None = None
    session_calendar: str = US_EQUITY_SESSION_CALENDAR
    benchmark_symbol: str = "SPY"
    model_version: str = ""
    feature_set_version: str = ""
    postprocessor_version: str = ""
    strict_close_t_eligible: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "as_of_session", _as_date(self.as_of_session, "as_of_session")
        )
        object.__setattr__(
            self,
            "target_session",
            _as_date(self.target_session, "target_session"),
        )
        for field_name in (
            "predicted_return",
            "realized_return",
            "benchmark_return",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite(getattr(self, field_name), field_name),
            )
        for field_name in (
            "probability_positive",
            "lower_bound_return",
            "upper_bound_return",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self, field_name, _finite(value, field_name)
                )
        object.__setattr__(
            self,
            "horizon_sessions",
            _positive_integer(self.horizon_sessions, "horizon_sessions"),
        )
        if self.target_session <= self.as_of_session:
            raise ValueError("target_session must be after as_of_session.")
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
        session_calendar = str(self.session_calendar).strip().lower()
        if not session_calendar:
            raise ValueError("session_calendar must not be empty.")
        if session_calendar not in SUPPORTED_SESSION_CALENDARS:
            supported = ", ".join(sorted(SUPPORTED_SESSION_CALENDARS))
            raise ValueError(
                f"session_calendar must be one of: {supported}."
            )
        object.__setattr__(self, "session_calendar", session_calendar)
        benchmark_symbol = str(self.benchmark_symbol).strip().upper()
        if not benchmark_symbol:
            raise ValueError("benchmark_symbol must not be empty.")
        object.__setattr__(self, "benchmark_symbol", benchmark_symbol)
        object.__setattr__(
            self,
            "model_version",
            str(self.model_version).strip(),
        )
        object.__setattr__(
            self,
            "feature_set_version",
            str(self.feature_set_version).strip(),
        )
        object.__setattr__(
            self,
            "postprocessor_version",
            str(self.postprocessor_version).strip(),
        )
        if not isinstance(self.strict_close_t_eligible, bool):
            raise ValueError("strict_close_t_eligible must be a boolean.")


@dataclass(frozen=True)
class ForecastMetrics:
    """Error, direction, interval, and probability-calibration metrics."""

    horizon_sessions: int
    sample_count: int
    mae: float
    rmse: float
    mean_error: float
    direction_accuracy: float
    rank_information_coefficient: float | None
    probability_count: int
    brier_score: float | None
    expected_calibration_error: float | None
    interval_count: int
    interval_coverage: float | None


def forecast_observations_from_ledger(
    ledger: PredictionLedger,
    *,
    as_of_session: date | datetime | str,
    horizon_sessions: int,
    candidate_model_name: str,
    candidate_policy_version: str,
    session_calendar: str | None = None,
    benchmark_symbol: str | None = None,
    model_version: str | None = None,
    feature_set_version: str | None = None,
    postprocessor_version: str | None = None,
    strict_close_t_eligible: bool | None = True,
) -> tuple[ForecastObservation, ...]:
    """Return one candidate's matured observations from a verified ledger.

    Model name and policy version are exact filters. Optional provenance
    filters are exact too. Any omitted provenance dimension must be homogeneous
    in the resulting cohort or evaluation fails closed. Delayed close-t records
    remain queryable by passing ``strict_close_t_eligible=False``, but are
    excluded from promotion-oriented queries by default.
    """

    entries = ledger.read_entries()
    return _forecast_observations_from_entries(
        entries,
        as_of_session=_as_date(as_of_session, "as_of_session"),
        horizon_sessions=horizon_sessions,
        candidate_model_name=candidate_model_name,
        candidate_policy_version=candidate_policy_version,
        session_calendar=session_calendar,
        benchmark_symbol=benchmark_symbol,
        model_version=model_version,
        feature_set_version=feature_set_version,
        postprocessor_version=postprocessor_version,
        strict_close_t_eligible=strict_close_t_eligible,
    )


def _forecast_observations_from_entries(
    entries: Sequence[LedgerEntry],
    *,
    as_of_session: date,
    horizon_sessions: int,
    candidate_model_name: str,
    candidate_policy_version: str,
    session_calendar: str | None = None,
    benchmark_symbol: str | None = None,
    model_version: str | None = None,
    feature_set_version: str | None = None,
    postprocessor_version: str | None = None,
    strict_close_t_eligible: bool | None = True,
) -> tuple[ForecastObservation, ...]:
    horizon = _positive_integer(horizon_sessions, "horizon_sessions")
    model_name = str(candidate_model_name).strip()
    policy_version = str(candidate_policy_version).strip()
    if not model_name:
        raise ValueError("candidate_model_name must not be empty.")
    if not policy_version:
        raise ValueError("candidate_policy_version must not be empty.")
    calendar_filter = _optional_session_calendar_filter(
        session_calendar,
        "session_calendar",
    )
    benchmark_filter = _optional_exact_filter(
        benchmark_symbol,
        "benchmark_symbol",
        case="upper",
    )
    model_version_filter = _optional_exact_filter(
        model_version,
        "model_version",
    )
    feature_set_filter = _optional_exact_filter(
        feature_set_version,
        "feature_set_version",
    )
    postprocessor_filter = _optional_exact_filter(
        postprocessor_version,
        "postprocessor_version",
    )
    if (
        strict_close_t_eligible is not None
        and not isinstance(strict_close_t_eligible, bool)
    ):
        raise ValueError(
            "strict_close_t_eligible must be a boolean or None."
        )

    predictions: dict[str, PredictionRecord] = {}
    outcomes: dict[str, OutcomeRecord] = {}
    for entry in entries:
        if isinstance(entry.record, PredictionRecord):
            predictions[entry.record.prediction_id] = entry.record
        elif isinstance(entry.record, OutcomeRecord):
            outcomes[entry.record.prediction_id] = entry.record

    completed: list[CompletedPrediction] = []
    for prediction in predictions.values():
        if prediction.horizon_sessions != horizon:
            continue
        if prediction.model_name != model_name:
            continue
        if prediction.policy_version != policy_version:
            continue
        if prediction.target_session > as_of_session:
            continue
        prediction_postprocessor = str(
            prediction.metadata.get("postprocessor_version", "")
        ).strip()
        prediction_strict_timing = (
            prediction_strict_close_t_eligible(prediction)
        )
        if (
            calendar_filter is not None
            and prediction.session_calendar != calendar_filter
        ):
            continue
        if (
            benchmark_filter is not None
            and prediction.benchmark_symbol != benchmark_filter
        ):
            continue
        if (
            model_version_filter is not None
            and prediction.model_version != model_version_filter
        ):
            continue
        if (
            feature_set_filter is not None
            and (prediction.feature_set_version or "")
            != feature_set_filter
        ):
            continue
        if (
            postprocessor_filter is not None
            and prediction_postprocessor != postprocessor_filter
        ):
            continue
        if (
            strict_close_t_eligible is not None
            and prediction_strict_timing != strict_close_t_eligible
        ):
            continue
        outcome = outcomes.get(prediction.prediction_id)
        if outcome is None:
            continue
        completed.append(
            CompletedPrediction(prediction=prediction, outcome=outcome)
        )

    unfiltered_dimensions = {
        "session_calendar": (
            calendar_filter,
            {item.prediction.session_calendar for item in completed},
        ),
        "benchmark_symbol": (
            benchmark_filter,
            {item.prediction.benchmark_symbol for item in completed},
        ),
        "model_version": (
            model_version_filter,
            {item.prediction.model_version for item in completed},
        ),
        "feature_set_version": (
            feature_set_filter,
            {
                item.prediction.feature_set_version or ""
                for item in completed
            },
        ),
        "postprocessor_version": (
            postprocessor_filter,
            {
                str(
                    item.prediction.metadata.get(
                        "postprocessor_version",
                        "",
                    )
                ).strip()
                for item in completed
            },
        ),
        "strict_close_t_eligible": (
            strict_close_t_eligible,
            {
                prediction_strict_close_t_eligible(item.prediction)
                for item in completed
            },
        ),
    }
    for dimension, (exact_filter, values) in unfiltered_dimensions.items():
        if exact_filter is None and len(values) > 1:
            raise ValueError(
                f"Mixed {dimension} cohort; provide an exact {dimension} filter."
            )

    completed.sort(
        key=lambda item: (
            item.prediction.as_of_session,
            item.prediction.symbol,
            item.prediction.prediction_id,
        )
    )
    return tuple(_observation_from_completed(item) for item in completed)


def _observation_from_completed(
    item: CompletedPrediction,
) -> ForecastObservation:
    prediction = item.prediction
    outcome = item.outcome
    return ForecastObservation(
        prediction_id=prediction.prediction_id,
        as_of_session=prediction.as_of_session,
        target_session=prediction.target_session,
        symbol=prediction.symbol,
        horizon_sessions=prediction.horizon_sessions,
        predicted_return=prediction.forecast_return,
        realized_return=outcome.realized_return,
        benchmark_return=outcome.benchmark_return,
        probability_positive=prediction.probability_positive,
        lower_bound_return=prediction.lower_bound_return,
        upper_bound_return=prediction.upper_bound_return,
        session_calendar=prediction.session_calendar,
        benchmark_symbol=prediction.benchmark_symbol,
        model_version=prediction.model_version,
        feature_set_version=prediction.feature_set_version or "",
        postprocessor_version=str(
            prediction.metadata.get("postprocessor_version", "")
        ).strip(),
        strict_close_t_eligible=prediction_strict_close_t_eligible(
            prediction
        ),
    )


def evaluate_forecasts(
    observations: Sequence[ForecastObservation],
    *,
    horizon_sessions: int,
    calibration_bins: int = 10,
) -> ForecastMetrics:
    """Evaluate only matured predictions from one explicitly selected horizon."""

    horizon_sessions = _positive_integer(
        horizon_sessions,
        "horizon_sessions",
    )
    calibration_bins = _positive_integer(calibration_bins, "calibration_bins")
    if not observations:
        raise ValueError("At least one matured forecast is required.")
    horizons = {item.horizon_sessions for item in observations}
    if horizons != {horizon_sessions}:
        raise MixedHorizonError(
            "Forecast metrics require one horizon; found "
            f"{sorted(horizons)} instead of {horizon_sessions}."
        )
    provenance_dimensions = {
        "session_calendar": {item.session_calendar for item in observations},
        "benchmark_symbol": {item.benchmark_symbol for item in observations},
        "model_version": {item.model_version for item in observations},
        "feature_set_version": {
            item.feature_set_version for item in observations
        },
        "postprocessor_version": {
            item.postprocessor_version for item in observations
        },
        "strict_close_t_eligible": {
            item.strict_close_t_eligible for item in observations
        },
    }
    for dimension, values in provenance_dimensions.items():
        if len(values) > 1:
            raise MixedForecastCohortError(
                f"Forecast metrics require one {dimension} cohort; "
                f"found {len(values)}."
            )

    predicted = np.asarray(
        [item.predicted_return for item in observations],
        dtype=float,
    )
    realized = np.asarray(
        [item.realized_return for item in observations],
        dtype=float,
    )
    errors = predicted - realized
    direction_accuracy = float(
        np.mean((predicted > 0.0) == (realized > 0.0))
    )

    rank_ic: float | None
    if (
        len(observations) < 2
        or np.allclose(predicted, predicted[0])
        or np.allclose(realized, realized[0])
    ):
        rank_ic = None
    else:
        correlation = pd.Series(predicted).corr(
            pd.Series(realized),
            method="spearman",
        )
        rank_ic = (
            float(correlation) if correlation is not None and np.isfinite(correlation)
            else None
        )

    probability_pairs = [
        (item.probability_positive, float(item.realized_return > 0.0))
        for item in observations
        if item.probability_positive is not None
    ]
    brier_score: float | None = None
    calibration_error: float | None = None
    if probability_pairs:
        probabilities = np.asarray(
            [pair[0] for pair in probability_pairs],
            dtype=float,
        )
        outcomes = np.asarray(
            [pair[1] for pair in probability_pairs],
            dtype=float,
        )
        brier_score = float(np.mean((probabilities - outcomes) ** 2))
        calibration_error = _expected_calibration_error(
            probabilities,
            outcomes,
            calibration_bins,
        )

    interval_pairs = [
        (
            item.lower_bound_return,
            item.upper_bound_return,
            item.realized_return,
        )
        for item in observations
        if item.lower_bound_return is not None
        and item.upper_bound_return is not None
    ]
    interval_coverage = (
        float(
            np.mean(
                [
                    lower <= realized_value <= upper
                    for lower, upper, realized_value in interval_pairs
                ]
            )
        )
        if interval_pairs
        else None
    )

    return ForecastMetrics(
        horizon_sessions=horizon_sessions,
        sample_count=len(observations),
        mae=float(np.mean(np.abs(errors))),
        rmse=float(np.sqrt(np.mean(errors**2))),
        mean_error=float(np.mean(errors)),
        direction_accuracy=direction_accuracy,
        rank_information_coefficient=rank_ic,
        probability_count=len(probability_pairs),
        brier_score=brier_score,
        expected_calibration_error=calibration_error,
        interval_count=len(interval_pairs),
        interval_coverage=interval_coverage,
    )


def _expected_calibration_error(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    bins: int,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.minimum(
        np.searchsorted(edges, probabilities, side="right") - 1,
        bins - 1,
    )
    weighted_error = 0.0
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        count = int(np.sum(mask))
        if count == 0:
            continue
        weighted_error += (
            count
            / len(probabilities)
            * abs(float(np.mean(probabilities[mask]) - np.mean(outcomes[mask])))
        )
    return float(weighted_error)


@dataclass(frozen=True)
class PolicyPeriod:
    """One portfolio evaluation period after target changes and costs."""

    session: date
    gross_return: float
    benchmark_component: float
    gross_excess_return: float
    transaction_cost: float
    net_return: float
    net_excess_return: float
    gross_exposure: float
    turnover: float


@dataclass(frozen=True)
class PolicyPerformance:
    """Portfolio-level performance from daily, non-overlapping returns."""

    periods: tuple[PolicyPeriod, ...]
    session_count: int
    cumulative_net_return: float
    cumulative_net_excess_return: float
    annualized_net_return: float
    annualized_volatility: float
    sharpe: float | None
    excess_sharpe: float | None
    max_drawdown: float
    cvar: float
    total_turnover: float
    total_transaction_cost: float
    average_gross_exposure: float
    transaction_cost_bps: float
    periods_per_year: int = 252
    cvar_confidence: float = 0.95

    def __post_init__(self) -> None:
        """Reject aggregates that were not derived from the period path."""
        periods = tuple(self.periods)
        object.__setattr__(self, "periods", periods)
        if not periods:
            raise ValueError("PolicyPerformance requires at least one period.")
        if self.session_count != len(periods):
            raise ValueError("session_count must equal the number of periods.")
        sessions = tuple(period.session for period in periods)
        if sessions != tuple(sorted(sessions)) or len(sessions) != len(set(sessions)):
            raise ValueError("Policy performance sessions must be unique and ordered.")
        periods_per_year = _positive_integer(
            self.periods_per_year,
            "periods_per_year",
        )
        object.__setattr__(self, "periods_per_year", periods_per_year)
        if not 0.0 < float(self.cvar_confidence) < 1.0:
            raise ValueError("cvar_confidence must be between 0 and 1.")
        cost_bps = float(self.transaction_cost_bps)
        if not np.isfinite(cost_bps) or cost_bps < 0.0:
            raise ValueError("transaction_cost_bps must be finite and nonnegative.")

        for period in periods:
            values = (
                period.gross_return,
                period.benchmark_component,
                period.gross_excess_return,
                period.transaction_cost,
                period.net_return,
                period.net_excess_return,
                period.gross_exposure,
                period.turnover,
            )
            if not all(np.isfinite(float(value)) for value in values):
                raise ValueError("Policy period fields must be finite.")
            if period.gross_exposure < 0.0 or period.turnover < 0.0:
                raise ValueError("Exposure and turnover cannot be negative.")
            if period.transaction_cost < 0.0:
                raise ValueError("Transaction cost cannot be negative.")
            expected_cost = period.turnover * cost_bps / 10_000.0
            if not np.isclose(
                period.transaction_cost,
                expected_cost,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "Period transaction cost does not match turnover and cost bps."
                )
            expected_excess = (
                period.gross_return - period.benchmark_component
            )
            if not np.isclose(
                period.gross_excess_return,
                expected_excess,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("Period gross excess return is inconsistent.")
            if not np.isclose(
                period.net_return,
                period.gross_return - period.transaction_cost,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("Period net return is inconsistent.")
            if not np.isclose(
                period.net_excess_return,
                period.gross_excess_return - period.transaction_cost,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("Period net excess return is inconsistent.")

        net_returns = np.asarray(
            [period.net_return for period in periods],
            dtype=float,
        )
        excess_returns = np.asarray(
            [period.net_excess_return for period in periods],
            dtype=float,
        )
        if np.any(net_returns <= -1.0):
            raise ValueError("A period net return cannot be <= -100%.")
        expected = {
            "cumulative_net_return": float(
                np.prod(1.0 + net_returns) - 1.0
            ),
            "cumulative_net_excess_return": float(
                np.prod(1.0 + excess_returns) - 1.0
            ),
            "annualized_volatility": _annualized_volatility(
                net_returns,
                periods_per_year,
            ),
            "max_drawdown": _maximum_drawdown(net_returns),
            "cvar": _conditional_value_at_risk(
                net_returns,
                float(self.cvar_confidence),
            ),
            "total_turnover": float(
                sum(period.turnover for period in periods)
            ),
            "total_transaction_cost": float(
                sum(period.transaction_cost for period in periods)
            ),
            "average_gross_exposure": float(
                np.mean([period.gross_exposure for period in periods])
            ),
        }
        expected["annualized_net_return"] = float(
            (1.0 + expected["cumulative_net_return"])
            ** (periods_per_year / len(periods))
            - 1.0
        )
        optional_expected = {
            "sharpe": _annualized_sharpe(
                net_returns,
                periods_per_year,
            ),
            "excess_sharpe": _annualized_sharpe(
                excess_returns,
                periods_per_year,
            ),
        }
        for field_name, expected_value in expected.items():
            if not np.isclose(
                float(getattr(self, field_name)),
                expected_value,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError(
                    f"{field_name} does not match the immutable period path."
                )
        for field_name, expected_value in optional_expected.items():
            actual_value = getattr(self, field_name)
            if expected_value is None:
                if actual_value is not None:
                    raise ValueError(
                        f"{field_name} must be None for this period path."
                    )
            elif actual_value is None or not np.isclose(
                float(actual_value),
                expected_value,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError(
                    f"{field_name} does not match the immutable period path."
                )


def evaluate_policy_returns(
    frame: pd.DataFrame,
    *,
    transaction_cost_bps: float,
    session_column: str = "session",
    symbol_column: str = "symbol",
    target_weight_column: str = "target_weight",
    asset_return_column: str = "asset_return",
    benchmark_return_column: str = "benchmark_return",
    periods_per_year: int = 252,
    cvar_confidence: float = 0.95,
    maximum_gross_exposure: float = 1.0,
    liquidate_at_end: bool = False,
) -> PolicyPerformance:
    """Evaluate long-only target weights with costs only when weights change.

    Each input return must cover one non-overlapping execution period (normally
    one trading session).  Horizon-label returns must not be supplied here
    because overlapping 30-session labels would overstate the sample size.
    The supplied target weight must have been executable before the associated
    return period; close-generated decisions therefore need a one-session shift.
    Missing symbols on a later included session are interpreted as zero target
    weight and therefore incur exit turnover.
    """

    required = {
        session_column,
        symbol_column,
        target_weight_column,
        asset_return_column,
        benchmark_return_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing policy return columns: {missing}")
    if frame.empty:
        raise ValueError("At least one policy return observation is required.")
    if transaction_cost_bps < 0.0:
        raise ValueError("transaction_cost_bps cannot be negative.")
    periods_per_year = _positive_integer(periods_per_year, "periods_per_year")
    if not 0.0 < cvar_confidence < 1.0:
        raise ValueError("cvar_confidence must be between 0 and 1.")
    if maximum_gross_exposure <= 0.0:
        raise ValueError("maximum_gross_exposure must be positive.")

    data = frame[
        [
            session_column,
            symbol_column,
            target_weight_column,
            asset_return_column,
            benchmark_return_column,
        ]
    ].copy()
    data[session_column] = data[session_column].map(
        lambda value: _as_date(value, session_column)
    )
    data[symbol_column] = data[symbol_column].astype(str).str.upper().str.strip()
    if (data[symbol_column] == "").any():
        raise ValueError("Policy symbols must not be empty.")
    if data.duplicated([session_column, symbol_column]).any():
        raise ValueError("Policy data contains duplicate symbol/session rows.")

    for column in (
        target_weight_column,
        asset_return_column,
        benchmark_return_column,
    ):
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain only finite values.")
    if (
        (data[target_weight_column] < 0.0)
        | (data[target_weight_column] > 1.0)
    ).any():
        raise ValueError("target weights must be between 0 and 1.")

    data = data.sort_values(
        [session_column, symbol_column],
        kind="stable",
    )
    cost_rate = transaction_cost_bps / 10_000.0
    previous_weights: dict[str, float] = {}
    periods: list[PolicyPeriod] = []

    for session, group in data.groupby(session_column, sort=True):
        current_weights = {
            str(row[symbol_column]): float(row[target_weight_column])
            for _, row in group.iterrows()
        }
        gross_exposure = float(sum(abs(weight) for weight in current_weights.values()))
        if gross_exposure > maximum_gross_exposure + 1e-12:
            raise ValueError(
                f"Gross exposure {gross_exposure:.6f} exceeds "
                f"{maximum_gross_exposure:.6f} on {session}."
            )

        all_symbols = set(previous_weights).union(current_weights)
        turnover = float(
            sum(
                abs(
                    current_weights.get(symbol, 0.0)
                    - previous_weights.get(symbol, 0.0)
                )
                for symbol in all_symbols
            )
        )
        gross_return = float(
            np.sum(
                group[target_weight_column].to_numpy(dtype=float)
                * group[asset_return_column].to_numpy(dtype=float)
            )
        )
        benchmark_component = float(
            np.sum(
                group[target_weight_column].to_numpy(dtype=float)
                * group[benchmark_return_column].to_numpy(dtype=float)
            )
        )
        gross_excess = gross_return - benchmark_component
        transaction_cost = turnover * cost_rate
        periods.append(
            PolicyPeriod(
                session=session,
                gross_return=gross_return,
                benchmark_component=benchmark_component,
                gross_excess_return=gross_excess,
                transaction_cost=transaction_cost,
                net_return=gross_return - transaction_cost,
                net_excess_return=gross_excess - transaction_cost,
                gross_exposure=gross_exposure,
                turnover=turnover,
            )
        )
        previous_weights = current_weights

    if liquidate_at_end and periods:
        liquidation_turnover = float(
            sum(abs(value) for value in previous_weights.values())
        )
        liquidation_cost = liquidation_turnover * cost_rate
        final = periods[-1]
        periods[-1] = PolicyPeriod(
            session=final.session,
            gross_return=final.gross_return,
            benchmark_component=final.benchmark_component,
            gross_excess_return=final.gross_excess_return,
            transaction_cost=final.transaction_cost + liquidation_cost,
            net_return=final.net_return - liquidation_cost,
            net_excess_return=final.net_excess_return - liquidation_cost,
            gross_exposure=final.gross_exposure,
            turnover=final.turnover + liquidation_turnover,
        )

    return policy_performance_from_periods(
        periods,
        transaction_cost_bps=transaction_cost_bps,
        periods_per_year=periods_per_year,
        cvar_confidence=cvar_confidence,
    )


def policy_performance_from_periods(
    periods: Sequence[PolicyPeriod],
    *,
    transaction_cost_bps: float,
    periods_per_year: int = 252,
    cvar_confidence: float = 0.95,
) -> PolicyPerformance:
    """Recompute performance for an exact immutable period-path slice."""
    period_path = tuple(periods)
    if not period_path:
        raise ValueError("At least one policy period is required.")
    periods_per_year = _positive_integer(
        periods_per_year,
        "periods_per_year",
    )
    net_returns = np.asarray(
        [period.net_return for period in period_path],
        dtype=float,
    )
    excess_returns = np.asarray(
        [period.net_excess_return for period in period_path],
        dtype=float,
    )
    if np.any(net_returns <= -1.0):
        raise ValueError(
            "A period net return cannot be less than or equal to -100%."
        )
    cumulative_net = float(np.prod(1.0 + net_returns) - 1.0)
    cumulative_excess = float(np.prod(1.0 + excess_returns) - 1.0)
    return PolicyPerformance(
        periods=period_path,
        session_count=len(period_path),
        cumulative_net_return=cumulative_net,
        cumulative_net_excess_return=cumulative_excess,
        annualized_net_return=float(
            (1.0 + cumulative_net)
            ** (periods_per_year / len(period_path))
            - 1.0
        ),
        annualized_volatility=_annualized_volatility(
            net_returns,
            periods_per_year,
        ),
        sharpe=_annualized_sharpe(net_returns, periods_per_year),
        excess_sharpe=_annualized_sharpe(
            excess_returns,
            periods_per_year,
        ),
        max_drawdown=_maximum_drawdown(net_returns),
        cvar=_conditional_value_at_risk(
            net_returns,
            cvar_confidence,
        ),
        total_turnover=float(
            sum(period.turnover for period in period_path)
        ),
        total_transaction_cost=float(
            sum(period.transaction_cost for period in period_path)
        ),
        average_gross_exposure=float(
            np.mean(
                [period.gross_exposure for period in period_path]
            )
        ),
        transaction_cost_bps=float(transaction_cost_bps),
        periods_per_year=int(periods_per_year),
        cvar_confidence=float(cvar_confidence),
    )


def _annualized_volatility(
    returns: np.ndarray,
    periods_per_year: int,
) -> float:
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns, ddof=1) * math.sqrt(periods_per_year))


def _annualized_sharpe(
    returns: np.ndarray,
    periods_per_year: int,
) -> float | None:
    if len(returns) < 2:
        return None
    standard_deviation = float(np.std(returns, ddof=1))
    if standard_deviation <= 1e-15:
        return None
    return float(
        np.mean(returns) / standard_deviation * math.sqrt(periods_per_year)
    )


def _maximum_drawdown(returns: np.ndarray) -> float:
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], equity)))
    drawdowns = 1.0 - np.concatenate(([1.0], equity)) / peaks
    return float(np.max(drawdowns))


def _conditional_value_at_risk(
    returns: np.ndarray,
    confidence: float,
) -> float:
    tail_probability = 1.0 - confidence
    quantile = float(np.quantile(returns, tail_probability))
    tail = returns[returns <= quantile]
    return float(np.mean(tail)) if len(tail) else quantile


@dataclass(frozen=True)
class FoldPerformance:
    """Candidate and champion performance for the same frozen test fold."""

    fold_id: int
    candidate: PolicyPerformance
    baseline: PolicyPerformance
    horizon_sessions: int
    candidate_policy_version: str
    baseline_policy_version: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "horizon_sessions",
            _positive_integer(self.horizon_sessions, "horizon_sessions"),
        )
        if not str(self.candidate_policy_version).strip():
            raise ValueError("candidate_policy_version must not be empty.")
        if not str(self.baseline_policy_version).strip():
            raise ValueError("baseline_policy_version must not be empty.")
        if _performance_sessions(self.candidate) != _performance_sessions(self.baseline):
            raise ValueError(
                "Candidate and baseline fold performance must use identical sessions."
            )


@dataclass(frozen=True)
class PromotionGateConfig:
    """Evidence thresholds required before a shadow policy may be promoted."""

    minimum_shadow_sessions: int = 60
    minimum_sharpe_improvement: float = 0.25
    minimum_positive_fold_fraction: float = 0.70
    maximum_expected_calibration_error: float = 0.10
    maximum_brier_score: float = 0.25
    minimum_probability_samples: int = 60
    drawdown_tolerance: float = 0.0
    cvar_tolerance: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_shadow_sessions",
            _positive_integer(
                self.minimum_shadow_sessions,
                "minimum_shadow_sessions",
            ),
        )
        if self.minimum_sharpe_improvement < 0.0:
            raise ValueError("minimum_sharpe_improvement cannot be negative.")
        if not 0.0 <= self.minimum_positive_fold_fraction <= 1.0:
            raise ValueError(
                "minimum_positive_fold_fraction must be between 0 and 1."
            )
        if not 0.0 <= self.maximum_expected_calibration_error <= 1.0:
            raise ValueError(
                "maximum_expected_calibration_error must be between 0 and 1."
            )
        if not 0.0 <= self.maximum_brier_score <= 1.0:
            raise ValueError("maximum_brier_score must be between 0 and 1.")
        object.__setattr__(
            self,
            "minimum_probability_samples",
            _positive_integer(
                self.minimum_probability_samples,
                "minimum_probability_samples",
            ),
        )
        if self.drawdown_tolerance < 0.0 or self.cvar_tolerance < 0.0:
            raise ValueError("Risk tolerances cannot be negative.")


@dataclass(frozen=True)
class PromotionEvidence:
    """Candidate/baseline evidence, including doubled-cost stress results."""

    shadow_sessions: int
    candidate: PolicyPerformance
    baseline: PolicyPerformance
    candidate_doubled_cost: PolicyPerformance
    baseline_doubled_cost: PolicyPerformance
    folds: tuple[FoldPerformance, ...]
    candidate_forecast_metrics: ForecastMetrics
    horizon_sessions: int
    evaluation_id: str
    candidate_policy_version: str
    baseline_policy_version: str
    baseline_name: str
    baseline_candidates: Mapping[str, PolicyPerformance]
    baseline_candidate_versions: Mapping[str, str]
    candidate_forecast_prediction_ids: tuple[str, ...]
    candidate_model_name: str
    forecast_as_of_session: date
    ledger_head_hash: str
    candidate_session_calendar: str | None = None
    candidate_benchmark_symbol: str | None = None
    candidate_model_version: str | None = None
    candidate_feature_set_version: str | None = None
    candidate_postprocessor_version: str | None = None
    candidate_strict_close_t_eligible: bool | None = True

    def __post_init__(self) -> None:
        if isinstance(self.shadow_sessions, bool) or not isinstance(
            self.shadow_sessions, Integral
        ):
            raise ValueError("shadow_sessions must be an integer.")
        if self.shadow_sessions < 0:
            raise ValueError("shadow_sessions cannot be negative.")
        object.__setattr__(
            self,
            "horizon_sessions",
            _positive_integer(self.horizon_sessions, "horizon_sessions"),
        )
        for field_name in (
            "evaluation_id",
            "candidate_model_name",
            "candidate_policy_version",
            "baseline_policy_version",
            "baseline_name",
            "ledger_head_hash",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must not be empty.")
        object.__setattr__(
            self,
            "candidate_model_name",
            str(self.candidate_model_name).strip(),
        )
        object.__setattr__(
            self,
            "candidate_session_calendar",
            _optional_session_calendar_filter(
                self.candidate_session_calendar,
                "candidate_session_calendar",
            ),
        )
        object.__setattr__(
            self,
            "candidate_benchmark_symbol",
            _optional_exact_filter(
                self.candidate_benchmark_symbol,
                "candidate_benchmark_symbol",
                case="upper",
            ),
        )
        object.__setattr__(
            self,
            "candidate_model_version",
            _optional_exact_filter(
                self.candidate_model_version,
                "candidate_model_version",
            ),
        )
        object.__setattr__(
            self,
            "candidate_feature_set_version",
            _optional_exact_filter(
                self.candidate_feature_set_version,
                "candidate_feature_set_version",
            ),
        )
        object.__setattr__(
            self,
            "candidate_postprocessor_version",
            _optional_exact_filter(
                self.candidate_postprocessor_version,
                "candidate_postprocessor_version",
            ),
        )
        if (
            self.candidate_strict_close_t_eligible is not None
            and not isinstance(
                self.candidate_strict_close_t_eligible,
                bool,
            )
        ):
            raise ValueError(
                "candidate_strict_close_t_eligible must be a boolean or None."
            )
        object.__setattr__(
            self,
            "forecast_as_of_session",
            _as_date(
                self.forecast_as_of_session,
                "forecast_as_of_session",
            ),
        )
        ledger_head_hash = str(self.ledger_head_hash).strip().lower()
        if len(ledger_head_hash) != 64 or any(
            character not in "0123456789abcdef"
            for character in ledger_head_hash
        ):
            raise ValueError("ledger_head_hash must be a SHA-256 hex digest.")
        object.__setattr__(self, "ledger_head_hash", ledger_head_hash)
        prediction_ids = tuple(
            str(prediction_id).strip()
            for prediction_id in self.candidate_forecast_prediction_ids
        )
        if (
            not all(prediction_ids)
            or len(prediction_ids) != len(set(prediction_ids))
        ):
            raise ValueError(
                "candidate_forecast_prediction_ids must be nonempty and unique."
            )
        if len(prediction_ids) != self.candidate_forecast_metrics.sample_count:
            raise ValueError(
                "candidate_forecast_prediction_ids must match the forecast sample count."
            )
        object.__setattr__(
            self,
            "candidate_forecast_prediction_ids",
            prediction_ids,
        )
        if self.shadow_sessions != self.candidate.session_count:
            raise ValueError(
                "shadow_sessions must equal the candidate evaluation session count."
            )
        full_sessions = _performance_sessions(self.candidate)
        for label, performance in (
            ("baseline", self.baseline),
            ("candidate_doubled_cost", self.candidate_doubled_cost),
            ("baseline_doubled_cost", self.baseline_doubled_cost),
        ):
            if _performance_sessions(performance) != full_sessions:
                raise ValueError(
                    f"{label} must use the same sessions as the candidate."
                )
        if not _is_doubled_cost_replay(
            self.candidate,
            self.candidate_doubled_cost,
        ):
            raise ValueError(
                "candidate_doubled_cost must replay the candidate path at exactly 2x costs."
            )
        if not _is_doubled_cost_replay(
            self.baseline,
            self.baseline_doubled_cost,
        ):
            raise ValueError(
                "baseline_doubled_cost must replay the baseline path at exactly 2x costs."
            )
        if self.candidate_forecast_metrics.horizon_sessions != self.horizon_sessions:
            raise MixedHorizonError(
                "Candidate forecast metrics do not match promotion horizon."
            )
        baselines = dict(self.baseline_candidates)
        baseline_versions = {
            str(name): str(version).strip()
            for name, version in dict(
                self.baseline_candidate_versions
            ).items()
        }
        if set(baseline_versions) != set(baselines):
            raise ValueError(
                "baseline_candidate_versions must exactly match baseline_candidates."
            )
        if not all(baseline_versions.values()):
            raise ValueError("Baseline candidate versions must not be empty.")
        if self.baseline_name not in baselines:
            raise ValueError(
                "baseline_candidates must include baseline_name."
            )
        if baselines[self.baseline_name] != self.baseline:
            raise ValueError(
                "baseline_candidates entry does not match baseline performance."
            )
        if baseline_versions[self.baseline_name] != self.baseline_policy_version:
            raise ValueError(
                "Selected baseline candidate version does not match baseline_policy_version."
            )
        normalized_baselines = {
            _normalized_baseline_name(name) for name in baselines
        }
        missing_baselines = sorted(
            REQUIRED_NON_RL_BASELINES - normalized_baselines
        )
        if missing_baselines:
            raise ValueError(
                "baseline_candidates must include the registered non-RL baselines: "
                + ", ".join(missing_baselines)
            )
        ordinary_cost_bps = float(self.candidate.transaction_cost_bps)
        for name, performance in baselines.items():
            if _performance_sessions(performance) != full_sessions:
                raise ValueError(
                    f"Baseline candidate {name!r} must use the candidate sessions."
                )
            if not np.isclose(
                performance.transaction_cost_bps,
                ordinary_cost_bps,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"Baseline candidate {name!r} must use the candidate cost assumption."
                )
        strongest_name = max(
            baselines,
            key=lambda name: _baseline_score(baselines[name]),
        )
        if strongest_name != self.baseline_name:
            raise ValueError(
                "baseline must be the strongest registered non-RL baseline."
            )
        if _looks_like_rl(self.baseline_name) or _looks_like_rl(
            self.baseline_policy_version
        ):
            raise ValueError("Promotion baseline must be non-RL.")
        rl_named_baselines = sorted(
            name
            for name in baselines
            if _looks_like_rl(name)
            or _looks_like_rl(baseline_versions.get(name, ""))
        )
        if rl_named_baselines:
            raise ValueError(
                "baseline_candidates must contain only non-RL baselines: "
                + ", ".join(rl_named_baselines)
            )
        fold_ids = [fold.fold_id for fold in self.folds]
        if len(fold_ids) != len(set(fold_ids)):
            raise ValueError("Promotion fold identifiers must be unique.")
        fold_sessions: list[date] = []
        candidate_periods = {
            period.session: period for period in self.candidate.periods
        }
        baseline_periods = {
            period.session: period for period in self.baseline.periods
        }
        for fold in self.folds:
            if fold.horizon_sessions != self.horizon_sessions:
                raise MixedHorizonError(
                    "Promotion fold horizon does not match evidence horizon."
                )
            if fold.candidate_policy_version != self.candidate_policy_version:
                raise ValueError("Candidate fold policy version mismatch.")
            if fold.baseline_policy_version != self.baseline_policy_version:
                raise ValueError("Baseline fold policy version mismatch.")
            if not np.isclose(
                fold.candidate.transaction_cost_bps,
                ordinary_cost_bps,
                rtol=0.0,
                atol=1e-12,
            ) or not np.isclose(
                fold.baseline.transaction_cost_bps,
                ordinary_cost_bps,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "Promotion folds must use the ordinary candidate cost assumption."
                )
            sessions = _performance_sessions(fold.candidate)
            expected_candidate = tuple(
                candidate_periods.get(session) for session in sessions
            )
            expected_baseline = tuple(
                baseline_periods.get(session) for session in sessions
            )
            if (
                None in expected_candidate
                or fold.candidate.periods != expected_candidate
            ):
                raise ValueError(
                    "Candidate fold path must exactly match its full evaluation slice."
                )
            if (
                None in expected_baseline
                or fold.baseline.periods != expected_baseline
            ):
                raise ValueError(
                    "Baseline fold path must exactly match its full evaluation slice."
                )
            fold_sessions.extend(sessions)
        if len(fold_sessions) != len(set(fold_sessions)):
            raise ValueError("Promotion fold sessions must not overlap.")
        if tuple(sorted(fold_sessions)) != tuple(sorted(full_sessions)):
            raise ValueError(
                "Promotion folds must exactly partition the full evaluation sessions."
            )
        expected_evaluation_id = promotion_evaluation_id(
            horizon_sessions=self.horizon_sessions,
            candidate_policy_version=self.candidate_policy_version,
            baseline_policy_version=self.baseline_policy_version,
            baseline_name=self.baseline_name,
            candidate=self.candidate,
            baseline=self.baseline,
            candidate_doubled_cost=self.candidate_doubled_cost,
            baseline_doubled_cost=self.baseline_doubled_cost,
            folds=self.folds,
            baseline_candidates=baselines,
            baseline_candidate_versions=baseline_versions,
            candidate_forecast_metrics=self.candidate_forecast_metrics,
            candidate_forecast_prediction_ids=prediction_ids,
            candidate_model_name=self.candidate_model_name,
            forecast_as_of_session=self.forecast_as_of_session,
            ledger_head_hash=ledger_head_hash,
            candidate_session_calendar=self.candidate_session_calendar,
            candidate_benchmark_symbol=self.candidate_benchmark_symbol,
            candidate_model_version=self.candidate_model_version,
            candidate_feature_set_version=(
                self.candidate_feature_set_version
            ),
            candidate_postprocessor_version=(
                self.candidate_postprocessor_version
            ),
            candidate_strict_close_t_eligible=(
                self.candidate_strict_close_t_eligible
            ),
        )
        if self.evaluation_id != expected_evaluation_id:
            raise ValueError(
                "evaluation_id does not match the immutable promotion evidence."
            )


def build_ledger_backed_promotion_evidence(
    ledger: PredictionLedger,
    *,
    forecast_as_of_session: date | datetime | str,
    horizon_sessions: int,
    candidate_model_name: str,
    candidate_policy_version: str,
    candidate: PolicyPerformance,
    baseline: PolicyPerformance,
    candidate_doubled_cost: PolicyPerformance,
    baseline_doubled_cost: PolicyPerformance,
    folds: Sequence[FoldPerformance],
    baseline_policy_version: str,
    baseline_name: str,
    baseline_candidates: Mapping[str, PolicyPerformance],
    baseline_candidate_versions: Mapping[str, str],
    candidate_session_calendar: str | None = None,
    candidate_benchmark_symbol: str | None = None,
    candidate_model_version: str | None = None,
    candidate_feature_set_version: str | None = None,
    candidate_postprocessor_version: str | None = None,
    candidate_strict_close_t_eligible: bool | None = True,
) -> PromotionEvidence:
    """Build promotion evidence from one verified ledger snapshot.

    Forecast metrics, prediction identifiers, and the snapshot head hash are
    derived internally.  Every candidate portfolio session must have at least
    one matching matured forecast in the selected model/policy/horizon slice.
    """

    cutoff = _as_date(
        forecast_as_of_session,
        "forecast_as_of_session",
    )
    entries = ledger.read_entries()
    if not entries:
        raise ValueError("Cannot build promotion evidence from an empty ledger.")
    observations = _forecast_observations_from_entries(
        entries,
        as_of_session=cutoff,
        horizon_sessions=horizon_sessions,
        candidate_model_name=candidate_model_name,
        candidate_policy_version=candidate_policy_version,
        session_calendar=candidate_session_calendar,
        benchmark_symbol=candidate_benchmark_symbol,
        model_version=candidate_model_version,
        feature_set_version=candidate_feature_set_version,
        postprocessor_version=candidate_postprocessor_version,
        strict_close_t_eligible=candidate_strict_close_t_eligible,
    )
    if not observations:
        raise ValueError(
            "No matured ledger forecasts match the candidate model, "
            "policy version, and horizon."
        )
    _require_matching_forecast_sessions(observations, candidate)

    forecast_metrics = evaluate_forecasts(
        observations,
        horizon_sessions=horizon_sessions,
    )
    prediction_ids = tuple(
        observation.prediction_id for observation in observations
    )
    ledger_head_hash = entries[-1].record_hash
    folds_tuple = tuple(folds)
    evaluation_id = promotion_evaluation_id(
        horizon_sessions=horizon_sessions,
        candidate_policy_version=candidate_policy_version,
        baseline_policy_version=baseline_policy_version,
        baseline_name=baseline_name,
        candidate=candidate,
        baseline=baseline,
        candidate_doubled_cost=candidate_doubled_cost,
        baseline_doubled_cost=baseline_doubled_cost,
        folds=folds_tuple,
        baseline_candidates=baseline_candidates,
        baseline_candidate_versions=baseline_candidate_versions,
        candidate_forecast_metrics=forecast_metrics,
        candidate_forecast_prediction_ids=prediction_ids,
        candidate_model_name=candidate_model_name,
        forecast_as_of_session=cutoff,
        ledger_head_hash=ledger_head_hash,
        candidate_session_calendar=candidate_session_calendar,
        candidate_benchmark_symbol=candidate_benchmark_symbol,
        candidate_model_version=candidate_model_version,
        candidate_feature_set_version=candidate_feature_set_version,
        candidate_postprocessor_version=candidate_postprocessor_version,
        candidate_strict_close_t_eligible=(
            candidate_strict_close_t_eligible
        ),
    )
    return PromotionEvidence(
        shadow_sessions=candidate.session_count,
        candidate=candidate,
        baseline=baseline,
        candidate_doubled_cost=candidate_doubled_cost,
        baseline_doubled_cost=baseline_doubled_cost,
        folds=folds_tuple,
        candidate_forecast_metrics=forecast_metrics,
        horizon_sessions=horizon_sessions,
        evaluation_id=evaluation_id,
        candidate_model_name=candidate_model_name,
        candidate_policy_version=candidate_policy_version,
        baseline_policy_version=baseline_policy_version,
        baseline_name=baseline_name,
        baseline_candidates=baseline_candidates,
        baseline_candidate_versions=baseline_candidate_versions,
        candidate_forecast_prediction_ids=prediction_ids,
        forecast_as_of_session=cutoff,
        ledger_head_hash=ledger_head_hash,
        candidate_session_calendar=candidate_session_calendar,
        candidate_benchmark_symbol=candidate_benchmark_symbol,
        candidate_model_version=candidate_model_version,
        candidate_feature_set_version=candidate_feature_set_version,
        candidate_postprocessor_version=candidate_postprocessor_version,
        candidate_strict_close_t_eligible=(
            candidate_strict_close_t_eligible
        ),
    )


@dataclass(frozen=True)
class GateCheck:
    """One auditable promotion criterion."""

    name: str
    passed: bool
    observed: float | int | None
    required: str


@dataclass(frozen=True)
class PromotionDecision:
    """Aggregate result of all policy promotion gates."""

    promoted: bool
    checks: tuple[GateCheck, ...]

    @property
    def failed_checks(self) -> tuple[GateCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)


def evaluate_promotion_gates(
    evidence: PromotionEvidence,
    config: PromotionGateConfig | None = None,
    *,
    ledger: PredictionLedger | None = None,
) -> PromotionDecision:
    """Apply conservative, all-or-nothing shadow-policy promotion gates.

    A matching :class:`PredictionLedger` is mandatory for promotion.  Omitting
    it, passing a different ledger, or passing a ledger that cannot reproduce
    the candidate prediction IDs and metrics fails closed.
    """

    thresholds = config or PromotionGateConfig()
    sharpe_improvement = _difference_or_none(
        evidence.candidate.sharpe,
        evidence.baseline.sharpe,
    )
    positive_fold_fraction = (
        float(
            np.mean(
                [
                    fold.candidate.cumulative_net_return
                    > fold.baseline.cumulative_net_return
                    for fold in evidence.folds
                ]
            )
        )
        if evidence.folds
        else 0.0
    )
    metrics = evidence.candidate_forecast_metrics

    provenance_matches = evidence.evaluation_id == promotion_evaluation_id(
        horizon_sessions=evidence.horizon_sessions,
        candidate_policy_version=evidence.candidate_policy_version,
        baseline_policy_version=evidence.baseline_policy_version,
        baseline_name=evidence.baseline_name,
        candidate=evidence.candidate,
        baseline=evidence.baseline,
        candidate_doubled_cost=evidence.candidate_doubled_cost,
        baseline_doubled_cost=evidence.baseline_doubled_cost,
        folds=evidence.folds,
        baseline_candidates=evidence.baseline_candidates,
        baseline_candidate_versions=evidence.baseline_candidate_versions,
        candidate_forecast_metrics=evidence.candidate_forecast_metrics,
        candidate_forecast_prediction_ids=(
            evidence.candidate_forecast_prediction_ids
        ),
        candidate_model_name=evidence.candidate_model_name,
        forecast_as_of_session=evidence.forecast_as_of_session,
        ledger_head_hash=evidence.ledger_head_hash,
        candidate_session_calendar=evidence.candidate_session_calendar,
        candidate_benchmark_symbol=evidence.candidate_benchmark_symbol,
        candidate_model_version=evidence.candidate_model_version,
        candidate_feature_set_version=(
            evidence.candidate_feature_set_version
        ),
        candidate_postprocessor_version=(
            evidence.candidate_postprocessor_version
        ),
        candidate_strict_close_t_eligible=(
            evidence.candidate_strict_close_t_eligible
        ),
    )
    ledger_provenance_matches = _ledger_provenance_matches(
        evidence,
        ledger,
    )
    doubled_cost_matches = _is_doubled_cost_replay(
        evidence.candidate,
        evidence.candidate_doubled_cost,
    ) and _is_doubled_cost_replay(
        evidence.baseline,
        evidence.baseline_doubled_cost,
    )
    strongest_baseline = max(
        evidence.baseline_candidates,
        key=lambda name: _baseline_score(
            evidence.baseline_candidates[name]
        ),
    )

    checks = (
        GateCheck(
            name="ledger_backed_forecast_provenance",
            passed=ledger_provenance_matches,
            observed=len(evidence.candidate_forecast_prediction_ids),
            required=(
                "verified ledger snapshot, exact candidate IDs, and "
                "recomputed forecast metrics"
            ),
        ),
        GateCheck(
            name="matched_evaluation_provenance",
            passed=provenance_matches,
            observed=len(_performance_sessions(evidence.candidate)),
            required=(
                f"evaluation={evidence.evaluation_id}; horizon="
                f"{evidence.horizon_sessions}; matched sessions and versions"
            ),
        ),
        GateCheck(
            name="strongest_non_rl_baseline",
            passed=(
                strongest_baseline == evidence.baseline_name
                and not _looks_like_rl(evidence.baseline_name)
            ),
            observed=len(evidence.baseline_candidates),
            required=f"champion={evidence.baseline_name}",
        ),
        GateCheck(
            name="doubled_cost_assumption",
            passed=doubled_cost_matches,
            observed=min(
                _cost_multiple(
                    evidence.candidate_doubled_cost.transaction_cost_bps,
                    evidence.candidate.transaction_cost_bps,
                ),
                _cost_multiple(
                    evidence.baseline_doubled_cost.transaction_cost_bps,
                    evidence.baseline.transaction_cost_bps,
                ),
            ),
            required=">=2x candidate and baseline costs",
        ),
        GateCheck(
            name="shadow_sessions",
            passed=evidence.shadow_sessions
            >= thresholds.minimum_shadow_sessions,
            observed=evidence.shadow_sessions,
            required=f">={thresholds.minimum_shadow_sessions}",
        ),
        GateCheck(
            name="net_sharpe_improvement",
            passed=sharpe_improvement is not None
            and sharpe_improvement
            >= thresholds.minimum_sharpe_improvement,
            observed=sharpe_improvement,
            required=f">={thresholds.minimum_sharpe_improvement:.4f}",
        ),
        GateCheck(
            name="doubled_cost_net_return",
            passed=evidence.candidate_doubled_cost.cumulative_net_return
            > evidence.baseline_doubled_cost.cumulative_net_return,
            observed=(
                evidence.candidate_doubled_cost.cumulative_net_return
                - evidence.baseline_doubled_cost.cumulative_net_return
            ),
            required=">0 versus baseline",
        ),
        GateCheck(
            name="maximum_drawdown",
            passed=evidence.candidate.max_drawdown
            <= evidence.baseline.max_drawdown + thresholds.drawdown_tolerance,
            observed=evidence.candidate.max_drawdown
            - evidence.baseline.max_drawdown,
            required=f"<={thresholds.drawdown_tolerance:.4f} worse",
        ),
        GateCheck(
            name="cvar",
            passed=evidence.candidate.cvar
            >= evidence.baseline.cvar - thresholds.cvar_tolerance,
            observed=evidence.candidate.cvar - evidence.baseline.cvar,
            required=f">={-thresholds.cvar_tolerance:.4f} versus baseline",
        ),
        GateCheck(
            name="positive_fold_fraction",
            passed=positive_fold_fraction
            >= thresholds.minimum_positive_fold_fraction,
            observed=positive_fold_fraction,
            required=f">={thresholds.minimum_positive_fold_fraction:.4f}",
        ),
        GateCheck(
            name="calibration_sample_count",
            passed=metrics.probability_count
            >= thresholds.minimum_probability_samples,
            observed=metrics.probability_count,
            required=f">={thresholds.minimum_probability_samples}",
        ),
        GateCheck(
            name="expected_calibration_error",
            passed=metrics.expected_calibration_error is not None
            and metrics.expected_calibration_error
            <= thresholds.maximum_expected_calibration_error,
            observed=metrics.expected_calibration_error,
            required=f"<={thresholds.maximum_expected_calibration_error:.4f}",
        ),
        GateCheck(
            name="brier_score",
            passed=metrics.brier_score is not None
            and metrics.brier_score <= thresholds.maximum_brier_score,
            observed=metrics.brier_score,
            required=f"<={thresholds.maximum_brier_score:.4f}",
        ),
    )
    return PromotionDecision(
        promoted=all(check.passed for check in checks),
        checks=checks,
    )


def _difference_or_none(
    candidate: float | None,
    baseline: float | None,
) -> float | None:
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def _cost_multiple(stressed_bps: float, ordinary_bps: float) -> float:
    if ordinary_bps <= 0.0:
        return 0.0
    return stressed_bps / ordinary_bps


def _performance_sessions(performance: PolicyPerformance) -> tuple[date, ...]:
    return tuple(period.session for period in performance.periods)


def _require_matching_forecast_sessions(
    observations: Sequence[ForecastObservation],
    candidate: PolicyPerformance,
) -> None:
    forecast_sessions = tuple(
        sorted({observation.as_of_session for observation in observations})
    )
    candidate_sessions = _performance_sessions(candidate)
    if forecast_sessions != candidate_sessions:
        raise ValueError(
            "Matured candidate forecast sessions must exactly match the "
            "candidate policy evaluation sessions."
        )


def _ledger_entries_through_head(
    ledger: PredictionLedger,
    ledger_head_hash: str,
) -> tuple[LedgerEntry, ...]:
    entries = ledger.read_entries()
    for index, entry in enumerate(entries):
        if entry.record_hash == ledger_head_hash:
            return entries[: index + 1]
    raise ValueError(
        "ledger_head_hash is not present in the verified ledger chain."
    )


def _ledger_provenance_matches(
    evidence: PromotionEvidence,
    ledger: PredictionLedger | None,
) -> bool:
    if ledger is None:
        return False
    try:
        entries = _ledger_entries_through_head(
            ledger,
            evidence.ledger_head_hash,
        )
        observations = _forecast_observations_from_entries(
            entries,
            as_of_session=evidence.forecast_as_of_session,
            horizon_sessions=evidence.horizon_sessions,
            candidate_model_name=evidence.candidate_model_name,
            candidate_policy_version=evidence.candidate_policy_version,
            session_calendar=evidence.candidate_session_calendar,
            benchmark_symbol=evidence.candidate_benchmark_symbol,
            model_version=evidence.candidate_model_version,
            feature_set_version=evidence.candidate_feature_set_version,
            postprocessor_version=(
                evidence.candidate_postprocessor_version
            ),
            strict_close_t_eligible=(
                evidence.candidate_strict_close_t_eligible
            ),
        )
        _require_matching_forecast_sessions(
            observations,
            evidence.candidate,
        )
        prediction_ids = tuple(
            observation.prediction_id for observation in observations
        )
        if prediction_ids != evidence.candidate_forecast_prediction_ids:
            return False
        derived_metrics = evaluate_forecasts(
            observations,
            horizon_sessions=evidence.horizon_sessions,
        )
        return derived_metrics == evidence.candidate_forecast_metrics
    except (
        EvaluationError,
        LedgerError,
        OSError,
        TypeError,
        ValueError,
    ):
        return False


def _baseline_score(performance: PolicyPerformance) -> tuple[float, float]:
    sharpe = performance.sharpe
    return (
        float(sharpe) if sharpe is not None and np.isfinite(sharpe) else -np.inf,
        float(performance.cumulative_net_return),
    )


def promotion_evaluation_id(
    *,
    horizon_sessions: int,
    candidate_policy_version: str,
    baseline_policy_version: str,
    baseline_name: str,
    candidate: PolicyPerformance,
    baseline: PolicyPerformance,
    candidate_doubled_cost: PolicyPerformance,
    baseline_doubled_cost: PolicyPerformance,
    folds: Sequence[FoldPerformance],
    baseline_candidates: Mapping[str, PolicyPerformance],
    baseline_candidate_versions: Mapping[str, str],
    candidate_forecast_metrics: ForecastMetrics,
    candidate_forecast_prediction_ids: Sequence[str],
    candidate_model_name: str,
    forecast_as_of_session: date | datetime | str,
    ledger_head_hash: str,
    candidate_session_calendar: str | None = None,
    candidate_benchmark_symbol: str | None = None,
    candidate_model_version: str | None = None,
    candidate_feature_set_version: str | None = None,
    candidate_postprocessor_version: str | None = None,
    candidate_strict_close_t_eligible: bool | None = True,
) -> str:
    """Hash the complete immutable evidence path into an evaluation identity."""
    if (
        candidate_strict_close_t_eligible is not None
        and not isinstance(candidate_strict_close_t_eligible, bool)
    ):
        raise ValueError(
            "candidate_strict_close_t_eligible must be a boolean or None."
        )
    payload = {
        "horizon_sessions": int(horizon_sessions),
        "candidate_policy_version": str(candidate_policy_version),
        "baseline_policy_version": str(baseline_policy_version),
        "baseline_name": str(baseline_name),
        "candidate": _performance_manifest(candidate),
        "baseline": _performance_manifest(baseline),
        "candidate_doubled_cost": _performance_manifest(
            candidate_doubled_cost
        ),
        "baseline_doubled_cost": _performance_manifest(
            baseline_doubled_cost
        ),
        "folds": [
            {
                "fold_id": int(fold.fold_id),
                "horizon_sessions": int(fold.horizon_sessions),
                "candidate_policy_version": fold.candidate_policy_version,
                "baseline_policy_version": fold.baseline_policy_version,
                "candidate": _performance_manifest(fold.candidate),
                "baseline": _performance_manifest(fold.baseline),
            }
            for fold in folds
        ],
        "baseline_candidates": {
            str(name): _performance_manifest(performance)
            for name, performance in sorted(
                baseline_candidates.items(),
                key=lambda item: str(item[0]),
            )
        },
        "baseline_candidate_versions": {
            str(name): str(version)
            for name, version in sorted(
                baseline_candidate_versions.items(),
                key=lambda item: str(item[0]),
            )
        },
        "candidate_forecast_metrics": _forecast_metrics_manifest(
            candidate_forecast_metrics
        ),
        "candidate_forecast_prediction_ids": [
            str(prediction_id)
            for prediction_id in candidate_forecast_prediction_ids
        ],
        "candidate_model_name": str(candidate_model_name),
        "forecast_as_of_session": _as_date(
            forecast_as_of_session,
            "forecast_as_of_session",
        ).isoformat(),
        "ledger_head_hash": str(ledger_head_hash),
    }
    optional_provenance = {
        "candidate_session_calendar": _optional_session_calendar_filter(
            candidate_session_calendar,
            "candidate_session_calendar",
        ),
        "candidate_benchmark_symbol": _optional_exact_filter(
            candidate_benchmark_symbol,
            "candidate_benchmark_symbol",
            case="upper",
        ),
        "candidate_model_version": _optional_exact_filter(
            candidate_model_version,
            "candidate_model_version",
        ),
        "candidate_feature_set_version": _optional_exact_filter(
            candidate_feature_set_version,
            "candidate_feature_set_version",
        ),
        "candidate_postprocessor_version": _optional_exact_filter(
            candidate_postprocessor_version,
            "candidate_postprocessor_version",
        ),
    }
    payload.update(
        {
            field_name: value
            for field_name, value in optional_provenance.items()
            if value is not None
        }
    )
    if candidate_strict_close_t_eligible is not True:
        payload["candidate_strict_close_t_eligible"] = (
            candidate_strict_close_t_eligible
        )
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "evaluation-" + hashlib.sha256(raw).hexdigest()[:32]


def _forecast_metrics_manifest(
    metrics: ForecastMetrics,
) -> dict[str, object]:
    return {
        field_name: getattr(metrics, field_name)
        for field_name in ForecastMetrics.__dataclass_fields__
    }


def _performance_manifest(
    performance: PolicyPerformance,
) -> dict[str, object]:
    return {
        "transaction_cost_bps": float(performance.transaction_cost_bps),
        "periods_per_year": int(performance.periods_per_year),
        "cvar_confidence": float(performance.cvar_confidence),
        "periods": [
            {
                "session": period.session.isoformat(),
                "gross_return": float(period.gross_return),
                "benchmark_component": float(period.benchmark_component),
                "gross_excess_return": float(period.gross_excess_return),
                "transaction_cost": float(period.transaction_cost),
                "net_return": float(period.net_return),
                "net_excess_return": float(period.net_excess_return),
                "gross_exposure": float(period.gross_exposure),
                "turnover": float(period.turnover),
            }
            for period in performance.periods
        ],
    }


def _looks_like_rl(value: object) -> bool:
    normalized = str(value).strip().lower().replace("_", "-")
    return (
        normalized == "rl"
        or normalized.startswith("rl-")
        or "reinforcement" in normalized
    )


def _normalized_baseline_name(value: object) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    aliases = {
        "ensemble": "fixed-ensemble",
        "fixed-ensemble-baseline": "fixed-ensemble",
        "ridge-regression": "ridge",
    }
    return aliases.get(normalized, normalized)


def _is_doubled_cost_replay(
    ordinary: PolicyPerformance,
    stressed: PolicyPerformance,
) -> bool:
    if ordinary.transaction_cost_bps <= 0.0:
        return False
    if not np.isclose(
        stressed.transaction_cost_bps,
        2.0 * ordinary.transaction_cost_bps,
        rtol=0.0,
        atol=1e-12,
    ):
        return False
    if len(ordinary.periods) != len(stressed.periods):
        return False
    for base_period, stress_period in zip(ordinary.periods, stressed.periods):
        if base_period.session != stress_period.session:
            return False
        for field_name in (
            "gross_return",
            "benchmark_component",
            "gross_excess_return",
            "gross_exposure",
            "turnover",
        ):
            if not np.isclose(
                getattr(base_period, field_name),
                getattr(stress_period, field_name),
                rtol=0.0,
                atol=1e-12,
            ):
                return False
        if not np.isclose(
            stress_period.transaction_cost,
            2.0 * base_period.transaction_cost,
            rtol=0.0,
            atol=1e-12,
        ):
            return False
    return True
