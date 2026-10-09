"""Descriptive baseline and coverage checks for matured forecast cohorts.

These diagnostics never select, promote, or trade a model. Fixed always-up,
always-not-up, zero-return, and 50% probability predictions are comparators;
none is fitted using the outcomes being measured. Returns are decimal fractions
on input, while fields ending in ``_pct`` express percentage points or percent.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Sequence

from .evaluation import ForecastObservation, evaluate_forecasts


def forecast_performance_diagnostics(
    observations: Sequence[ForecastObservation],
    *,
    horizon_sessions: int,
) -> dict[str, Any]:
    """Compare one matured, homogeneous cohort with fixed naive predictions.

    Supply observations in immutable ledger order. Repeated forecasts for the
    same symbol, origin, and target retain their first publication, rather than
    allowing later revisions or repeated runs to overweight a return window.
    The existing evaluator validates horizons and all provenance dimensions.

    The non-overlapping subset is selected independently for each asset using
    earliest-finish interval scheduling. Adjacent windows may share an endpoint
    close but no return increment. This reduces serial overlap; assets can
    remain correlated, so its count is not an independent sample size or a
    statistical confidence estimate. These descriptive figures do not change
    any promotion gates, and do not represent executable portfolio returns.
    """

    # Validate every input before deduplication so a mixed-provenance repeat
    # cannot be hidden by its matching asset and forecast dates.
    evaluate_forecasts(observations, horizon_sessions=horizon_sessions)
    first_by_window: dict[tuple[str, date, date], ForecastObservation] = {}
    for observation in observations:
        key = (
            observation.symbol,
            observation.as_of_session,
            observation.target_session,
        )
        first_by_window.setdefault(key, observation)
    unique = tuple(first_by_window.values())

    last_target_by_symbol: dict[str, date] = {}
    non_overlapping: list[ForecastObservation] = []
    for observation in sorted(
        unique,
        key=lambda item: (
            item.target_session,
            item.as_of_session,
            item.symbol,
            item.prediction_id,
        ),
    ):
        last_target = last_target_by_symbol.get(observation.symbol)
        if last_target is None or observation.as_of_session >= last_target:
            non_overlapping.append(observation)
            last_target_by_symbol[observation.symbol] = observation.target_session

    return {
        "raw_observation_count": len(observations),
        "duplicate_forecast_count": len(observations) - len(unique),
        "deduplication_rule": "first_in_ledger_per_symbol_origin_target",
        **_subset_diagnostics(unique, horizon_sessions=horizon_sessions),
        "non_overlapping": _subset_diagnostics(
            non_overlapping,
            horizon_sessions=horizon_sessions,
        ),
        "limitations": (
            "Descriptive matured forecast errors, not trading returns. "
            "Non-overlap is enforced per asset; cross-asset correlation remains. "
            "Fixed direction comparators are reported separately; choosing the "
            "better comparator after observing outcomes is retrospective. "
            "The 50% probability comparator is not a training-base-rate model."
        ),
    }


def _subset_diagnostics(
    observations: Sequence[ForecastObservation],
    *,
    horizon_sessions: int,
) -> dict[str, Any]:
    evaluated = evaluate_forecasts(
        observations,
        horizon_sessions=horizon_sessions,
    )
    count = len(observations)
    no_change_mae = sum(abs(item.realized_return) for item in observations) / count
    up_fraction = sum(item.realized_return > 0.0 for item in observations) / count
    has_probabilities = evaluated.probability_count > 0
    return {
        "sample_count": count,
        "symbol_count": len({item.symbol for item in observations}),
        "origin_session_count": len({item.as_of_session for item in observations}),
        "first_origin_session": min(item.as_of_session for item in observations).isoformat(),
        "last_origin_session": max(item.as_of_session for item in observations).isoformat(),
        "first_target_session": min(item.target_session for item in observations).isoformat(),
        "last_target_session": max(item.target_session for item in observations).isoformat(),
        "mae_pct": evaluated.mae * 100.0,
        "no_change_mae_pct": no_change_mae * 100.0,
        "mae_skill_score": (
            1.0 - evaluated.mae / no_change_mae if no_change_mae > 0.0 else None
        ),
        "direction_accuracy_pct": evaluated.direction_accuracy * 100.0,
        "always_up_accuracy_pct": up_fraction * 100.0,
        "always_not_up_accuracy_pct": (1.0 - up_fraction) * 100.0,
        "probability_count": evaluated.probability_count,
        "brier_score": evaluated.brier_score,
        "fair_coin_brier_score": 0.25 if has_probabilities else None,
        "brier_skill_score_vs_fair_coin": (
            1.0 - evaluated.brier_score / 0.25
            if evaluated.brier_score is not None
            else None
        ),
    }
