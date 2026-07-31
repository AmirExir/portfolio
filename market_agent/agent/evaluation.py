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
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .ledger import CompletedPrediction, PredictionLedger


UTC = timezone.utc


class EvaluationError(RuntimeError):
    """Base class for evaluation failures."""


class DataLeakageError(EvaluationError):
    """Raised when a training label overlaps its test fold."""


class MixedHorizonError(EvaluationError):
    """Raised when records from different forecast horizons are combined."""


def _as_date(value: date | datetime | str, field_name: str) -> date:
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
        if self.horizon_sessions <= 0:
            raise ValueError("horizon_sessions must be positive.")
        if self.minimum_training_sessions <= 0:
            raise ValueError("minimum_training_sessions must be positive.")
        if self.test_sessions <= 0:
            raise ValueError("test_sessions must be positive.")

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
            and self.maximum_training_sessions < self.minimum_training_sessions
        ):
            raise ValueError(
                "maximum_training_sessions cannot be smaller than "
                "minimum_training_sessions."
            )
        object.__setattr__(self, "purge_sessions", int(purge))
        object.__setattr__(self, "embargo_sessions", int(embargo))


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
        predicted = predict_policy(
            fitted.model,
            predictor_frame.copy(),
            fold,
        )
        if not isinstance(predicted, pd.DataFrame):
            raise TypeError("predict_policy must return a pandas DataFrame.")
        if len(predicted) != len(test_rows):
            raise ValueError(
                f"Fold {fold.fold_id} returned {len(predicted)} predictions "
                f"for {len(test_rows)} test rows."
            )
        reserved_outputs = set(predicted.columns).intersection(
            {*data.columns, "fold_id", "policy_version"}
        )
        if reserved_outputs:
            raise ValueError(
                "Prediction outputs collide with input columns: "
                f"{sorted(reserved_outputs)}"
            )

        fold_output = test_rows[
            [*identity_columns, *outcome_columns]
        ].reset_index(drop=True)
        fold_output = pd.concat(
            [fold_output, predicted.reset_index(drop=True)],
            axis=1,
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
        if self.horizon_sessions <= 0:
            raise ValueError("horizon_sessions must be positive.")
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
) -> tuple[ForecastObservation, ...]:
    """Create evaluation observations from completed, matured ledger events."""

    completed = ledger.completed_predictions(
        _as_date(as_of_session, "as_of_session"),
        horizon_sessions=horizon_sessions,
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
    )


def evaluate_forecasts(
    observations: Sequence[ForecastObservation],
    *,
    horizon_sessions: int,
    calibration_bins: int = 10,
) -> ForecastMetrics:
    """Evaluate only matured predictions from one explicitly selected horizon."""

    if calibration_bins <= 0:
        raise ValueError("calibration_bins must be positive.")
    if not observations:
        raise ValueError("At least one matured forecast is required.")
    horizons = {item.horizon_sessions for item in observations}
    if horizons != {horizon_sessions}:
        raise MixedHorizonError(
            "Forecast metrics require one horizon; found "
            f"{sorted(horizons)} instead of {horizon_sessions}."
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
        np.mean(np.sign(predicted) == np.sign(realized))
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
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
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

    net_returns = np.asarray([period.net_return for period in periods])
    excess_returns = np.asarray(
        [period.net_excess_return for period in periods]
    )
    if np.any(net_returns <= -1.0):
        raise ValueError("A period net return cannot be less than or equal to -100%.")

    cumulative_net = float(np.prod(1.0 + net_returns) - 1.0)
    cumulative_excess = float(np.prod(1.0 + excess_returns) - 1.0)
    annualized_net = float(
        (1.0 + cumulative_net) ** (periods_per_year / len(periods)) - 1.0
    )
    annualized_volatility = _annualized_volatility(
        net_returns,
        periods_per_year,
    )
    sharpe = _annualized_sharpe(net_returns, periods_per_year)
    excess_sharpe = _annualized_sharpe(excess_returns, periods_per_year)
    max_drawdown = _maximum_drawdown(net_returns)
    cvar = _conditional_value_at_risk(net_returns, cvar_confidence)

    return PolicyPerformance(
        periods=tuple(periods),
        session_count=len(periods),
        cumulative_net_return=cumulative_net,
        cumulative_net_excess_return=cumulative_excess,
        annualized_net_return=annualized_net,
        annualized_volatility=annualized_volatility,
        sharpe=sharpe,
        excess_sharpe=excess_sharpe,
        max_drawdown=max_drawdown,
        cvar=cvar,
        total_turnover=float(sum(period.turnover for period in periods)),
        total_transaction_cost=float(
            sum(period.transaction_cost for period in periods)
        ),
        average_gross_exposure=float(
            np.mean([period.gross_exposure for period in periods])
        ),
        transaction_cost_bps=float(transaction_cost_bps),
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
        if self.minimum_shadow_sessions <= 0:
            raise ValueError("minimum_shadow_sessions must be positive.")
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
        if self.minimum_probability_samples <= 0:
            raise ValueError("minimum_probability_samples must be positive.")
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
) -> PromotionDecision:
    """Apply conservative, all-or-nothing shadow-policy promotion gates."""

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

    checks = (
        GateCheck(
            name="doubled_cost_assumption",
            passed=(
                evidence.candidate.transaction_cost_bps > 0.0
                and evidence.baseline.transaction_cost_bps > 0.0
                and evidence.candidate_doubled_cost.transaction_cost_bps
                >= 2.0 * evidence.candidate.transaction_cost_bps
                and evidence.baseline_doubled_cost.transaction_cost_bps
                >= 2.0 * evidence.baseline.transaction_cost_bps
            ),
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
