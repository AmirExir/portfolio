"""Mature immutable prediction-ledger outcomes from realized close prices."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd

from .ledger import (
    OutcomeRecord,
    PredictionLedger,
)


UTC = timezone.utc
PriceLoader = Callable[[str], pd.DataFrame]


@dataclass(frozen=True)
class OutcomeMaturityResult:
    appended: int
    duplicates: int
    pending_not_mature: int
    skipped: tuple[str, ...]


def append_matured_outcomes(
    ledger: PredictionLedger,
    *,
    price_loader: PriceLoader,
    as_of_utc: datetime | None = None,
) -> OutcomeMaturityResult:
    """Append outcomes only after the prediction's exact maturity timestamp."""
    now = (as_of_utc or datetime.now(UTC)).astimezone(UTC)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("as_of_utc must be timezone-aware")
    predictions = ledger.pending_outcomes(now.date())
    cache: dict[str, pd.Series] = {}
    outcome_candidates: list[OutcomeRecord] = []
    immature = 0
    skipped: list[str] = []

    def close_for(symbol: str) -> pd.Series:
        if symbol not in cache:
            frame = price_loader(symbol)
            cache[symbol] = _clean_close(frame)
        return cache[symbol]

    for prediction in predictions:
        if now < prediction.target_maturity_utc:
            immature += 1
            continue
        try:
            asset = close_for(prediction.symbol)
            benchmark = close_for(prediction.benchmark_symbol)
            entry_price = _session_close(
                asset,
                prediction.return_start_session,
            )
            exit_price = _session_close(asset, prediction.target_session)
            benchmark_entry = _session_close(
                benchmark,
                prediction.return_start_session,
            )
            benchmark_exit = _session_close(
                benchmark,
                prediction.target_session,
            )
            window = _session_window(
                asset,
                prediction.return_start_session,
                prediction.target_session,
            )
            relative = window / entry_price - 1.0
            outcome = OutcomeRecord(
                outcome_id=f"outcome-{prediction.prediction_id}",
                prediction_id=prediction.prediction_id,
                recorded_at_utc=now,
                target_session=prediction.target_session,
                target_maturity_utc=prediction.target_maturity_utc,
                realized_return=float(exit_price / entry_price - 1.0),
                benchmark_return=float(
                    benchmark_exit / benchmark_entry - 1.0
                ),
                transaction_cost_return=0.0,
                entry_price=float(entry_price),
                exit_price=float(exit_price),
                max_adverse_excursion=float(min(relative.min(), 0.0)),
                max_favorable_excursion=float(max(relative.max(), 0.0)),
                data_source="daily-close-price-loader",
                metadata={
                    "outcome_kind": "raw_close_to_close",
                    "session_calendar": prediction.session_calendar,
                    "benchmark_symbol": prediction.benchmark_symbol,
                    "decision_session": (
                        prediction.as_of_session.isoformat()
                    ),
                    "return_start_session": (
                        prediction.return_start_session.isoformat()
                    ),
                    "transaction_costs_applied_later": True,
                },
            )
            outcome_candidates.append(outcome)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            skipped.append(f"{prediction.prediction_id}: {exc}")

    batch_result = ledger.append_outcomes(outcome_candidates)
    skipped.extend(
        f"{failure.prediction_id}: {failure.error}"
        for failure in batch_result.failures
    )
    return OutcomeMaturityResult(
        appended=batch_result.appended_count,
        duplicates=batch_result.duplicate_count,
        pending_not_mature=immature,
        skipped=tuple(skipped),
    )


def _clean_close(frame: pd.DataFrame) -> pd.Series:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "close" not in frame:
        raise ValueError("price loader returned no close history")
    close = frame["close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    close = close[close > 0.0].sort_index()
    close = close[~close.index.duplicated(keep="last")]
    if close.empty:
        raise ValueError("price loader returned no valid close history")
    return close


def _session_close(series: pd.Series, session) -> float:
    mask = pd.Index(series.index).map(lambda value: pd.Timestamp(value).date()) == session
    values = series.loc[mask]
    if values.empty:
        raise ValueError(f"missing close for session {session}")
    value = float(values.iloc[-1])
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"invalid close for session {session}")
    return value


def _session_window(
    series: pd.Series,
    start_session,
    end_session,
) -> pd.Series:
    dates = pd.Index(series.index).map(lambda value: pd.Timestamp(value).date())
    window = series.loc[(dates >= start_session) & (dates <= end_session)]
    if window.empty:
        raise ValueError("no prices in prediction outcome window")
    return window
