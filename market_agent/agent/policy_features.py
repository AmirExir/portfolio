"""Point-in-time market state for the long-only shadow policy.

The feature frame is built from information observable at each closing bell.
Forward returns are added only as labels. A close-generated decision has a
one-session execution lag, so its reward begins after the following close; the
latest, not-yet-matured row is returned separately for a frozen decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


SECTOR_ETF_BY_SYMBOL: dict[str, str] = {
    **{
        symbol: "XLK"
        for symbol in (
            "AAPL",
            "MSFT",
            "NVDA",
            "AMD",
            "INTC",
            "AVGO",
            "MU",
            "WDC",
            "STX",
            "SNDK",
            "ORCL",
            "DELL",
        )
    },
    **{symbol: "XLY" for symbol in ("AMZN", "TSLA", "HD")},
    **{symbol: "XLF" for symbol in ("JPM", "BAC", "WFC", "C", "GS", "MS", "V")},
    **{symbol: "XLV" for symbol in ("UNH", "LLY", "JNJ")},
    **{symbol: "XLE" for symbol in ("XOM", "CVX", "COP", "OXY", "SLB", "EOG")},
}

SECTOR_CONTEXT_TICKERS: tuple[str, ...] = ("XLK", "XLF", "XLE", "XLV", "XLY")
MARKET_CONTEXT_COLUMNS: tuple[str, ...] = (
    "decision_horizon",
    "asset_momentum",
    "asset_volatility",
    "spy_trend",
    "qqq_trend",
    "vix_level",
    "tlt_trend",
    "gap",
    "volume_shock",
    "event_flag",
    "event_score",
    "event_confidence",
    "sector_relative_strength",
    "sector_breadth",
)
FORECAST_CONTEXT_COLUMNS: tuple[str, ...] = (
    "forecast_return",
    "forecast_probability_up",
    "forecast_lower_bound",
    "forecast_model_agreement",
    "forecast_uncertainty",
)
REQUIRED_CONTEXT_COLUMNS: tuple[str, ...] = (
    *MARKET_CONTEXT_COLUMNS,
    *FORECAST_CONTEXT_COLUMNS,
)


@dataclass(frozen=True)
class ShadowPolicyData:
    """Matured training labels and the newest decision-time observation."""

    training_frame: pd.DataFrame
    latest_context: dict[str, float]
    as_of: object
    horizon_days: int
    benchmark_source: str
    sector_proxy: str | None
    correlation_source: str
    forecast_context_source: str = "unavailable"
    forecast_context_version: str = ""
    forecast_context_samples: int = 0
    execution_lag_sessions: int = 1


def build_shadow_policy_data(
    df: pd.DataFrame,
    *,
    context_df: pd.DataFrame | None = None,
    horizon_days: int = 1,
    symbol: str | None = None,
    event_flags: pd.Series | Mapping[object, object] | None = None,
    latest_event_context: Mapping[str, object] | None = None,
    portfolio_returns: pd.Series | None = None,
    forecast_context_df: pd.DataFrame | None = None,
    latest_forecast_context: Mapping[str, object] | None = None,
    forecast_context_source: str = "unavailable",
    forecast_context_version: str = "",
) -> ShadowPolicyData:
    """Build an as-of feature set and matured one-session execution labels.

    Every policy learns from non-overlapping daily transitions. Features at
    close t produce a target executable at close t+1, and the paired reward is
    the close t+1 to close t+2 return. ``decision_horizon`` is a state component
    and each horizon is trained, cached, and evaluated as a separate policy.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("OHLCV data must be a non-empty DataFrame")
    horizon = max(int(horizon_days), 1)
    close = _numeric_series(df, "close", required=True)
    close = close[close > 0.0].sort_index()
    close = close[~close.index.duplicated(keep="last")]
    if len(close) < horizon + 45:
        raise ValueError("not enough price history to build shadow-policy state")

    aligned = df.reindex(close.index)
    returns = close.pct_change()
    features = pd.DataFrame(index=close.index)
    features["decision_horizon"] = float(horizon)
    features["asset_momentum"] = close.pct_change(20)
    features["asset_volatility"] = returns.rolling(20, min_periods=10).std(ddof=0) * np.sqrt(252.0)

    market = _normalize_context(context_df, close.index)
    spy = _context_series(market, "SPY")
    qqq = _context_series(market, "QQQ")
    tlt = _context_series(market, "TLT")
    vix = _context_series(market, "^VIX")

    if spy is not None:
        features["spy_trend"] = spy.pct_change(20)
        benchmark_close = spy
        benchmark_source = "SPY"
    else:
        features["spy_trend"] = features["asset_momentum"]
        benchmark_close = None
        benchmark_source = "cash"

    if qqq is not None:
        features["qqq_trend"] = qqq.pct_change(20)
    else:
        features["qqq_trend"] = features["spy_trend"]

    if vix is not None:
        features["vix_level"] = vix
    else:
        # Neutral fallback is explicit and constant; it does not use later data.
        features["vix_level"] = 20.0

    features["tlt_trend"] = tlt.pct_change(20) if tlt is not None else 0.0

    open_price = _numeric_series(aligned, "open")
    if open_price is not None:
        features["gap"] = open_price / close.shift(1) - 1.0
    else:
        features["gap"] = 0.0

    volume = _numeric_series(aligned, "volume")
    if volume is not None:
        normal_volume = volume.shift(1).rolling(20, min_periods=10).median()
        features["volume_shock"] = volume / normal_volume.replace(0.0, np.nan) - 1.0
    else:
        features["volume_shock"] = 0.0

    features["event_flag"] = _event_series(event_flags, close.index)
    features["event_score"] = 0.0
    features["event_confidence"] = 0.0

    normalized_symbol = str(symbol or "").strip().upper()
    sector_proxy = SECTOR_ETF_BY_SYMBOL.get(normalized_symbol)
    sector_close = _context_series(market, sector_proxy) if sector_proxy else None
    sector_trend = sector_close.pct_change(20) if sector_close is not None else features["spy_trend"]
    features["sector_relative_strength"] = features["asset_momentum"] - sector_trend

    breadth_components = []
    for ticker in SECTOR_CONTEXT_TICKERS:
        series = _context_series(market, ticker)
        if series is not None:
            breadth_components.append((series.pct_change(20) > 0.0).astype(float))
    if breadth_components:
        features["sector_breadth"] = pd.concat(breadth_components, axis=1).mean(axis=1)
    else:
        features["sector_breadth"] = (features["spy_trend"] > 0.0).astype(float)

    correlation_source = "unavailable"
    if portfolio_returns is not None:
        portfolio = pd.to_numeric(portfolio_returns, errors="coerce").reindex(close.index).ffill()
        features["portfolio_correlation"] = returns.rolling(60, min_periods=20).corr(portfolio)
        features["portfolio_risk"] = portfolio.rolling(20, min_periods=10).std(ddof=0).clip(lower=0.0)
        correlation_source = "supplied_portfolio"
    else:
        features["portfolio_correlation"] = np.nan
        features["portfolio_risk"] = (
            features["asset_volatility"] / np.sqrt(252.0)
        ).clip(lower=0.0)

    # Historical forecast context must already be point-in-time/OOS. Exact
    # timestamp alignment is deliberate: a missing prediction stays missing
    # instead of inheriting a later or stale forecast through filling.
    forecast_context = _normalize_forecast_context(
        forecast_context_df,
        close.index,
    )
    for column in FORECAST_CONTEXT_COLUMNS:
        features[column] = forecast_context[column]

    # A decision uses finalized close/volume at t and therefore cannot earn the
    # t->t+1 close return. Shift the execution and reward window one full
    # session: target at t becomes active at close t+1 for t+1->t+2.
    forward_asset = close.shift(-2) / close.shift(-1)
    features["forward_asset_return"] = forward_asset - 1.0
    if benchmark_close is not None:
        forward_benchmark = (
            benchmark_close.shift(-2) / benchmark_close.shift(-1)
        )
        features["forward_benchmark_return"] = forward_benchmark - 1.0
    else:
        features["forward_benchmark_return"] = 0.0

    required_decision = list(MARKET_CONTEXT_COLUMNS) + ["portfolio_risk"]
    decision_rows = features.dropna(subset=required_decision)
    if decision_rows.empty:
        raise ValueError("no complete decision-time shadow-policy observations")
    latest_row = decision_rows.iloc[-1]
    latest_context = {
        column: float(latest_row[column])
        for column in MARKET_CONTEXT_COLUMNS
    }
    latest_context.update(
        _validated_forecast_context(
            latest_forecast_context,
            argument_name="latest_forecast_context",
        )
    )
    latest_correlation = latest_row.get("portfolio_correlation")
    latest_context["portfolio_correlation"] = (
        float(latest_correlation) if pd.notna(latest_correlation) else np.nan
    )
    latest_context["portfolio_risk"] = float(latest_row["portfolio_risk"])
    if latest_event_context:
        latest_event_flag = bool(
            latest_event_context.get(
                "earnings_event_flag",
                latest_event_context.get("event_flag", False),
            )
        )
        latest_event_score = float(
            np.clip(
                float(
                    latest_event_context.get(
                        "earnings_event_score",
                        latest_event_context.get("event_score", 0.0),
                    )
                    or 0.0
                ),
                -1.0,
                1.0,
            )
        )
        latest_event_confidence = float(
            np.clip(
                float(
                    latest_event_context.get(
                        "earnings_confidence",
                        latest_event_context.get("event_confidence", 0.0),
                    )
                    or 0.0
                ),
                0.0,
                1.0,
            )
        )
        latest_context.update(
            {
                "event_flag": float(latest_event_flag),
                "event_score": latest_event_score if latest_event_flag else 0.0,
                "event_confidence": (
                    latest_event_confidence if latest_event_flag else 0.0
                ),
            }
        )

    matured_columns = (
        list(REQUIRED_CONTEXT_COLUMNS)
        + [
            "portfolio_correlation",
            "portfolio_risk",
            "forward_asset_return",
            "forward_benchmark_return",
        ]
    )
    matured = features[matured_columns].dropna(
        subset=[
            *REQUIRED_CONTEXT_COLUMNS,
            "portfolio_risk",
            "forward_asset_return",
            "forward_benchmark_return",
        ]
    )
    if matured.empty:
        raise ValueError("no matured shadow-policy outcomes")

    return ShadowPolicyData(
        training_frame=matured,
        latest_context=latest_context,
        as_of=decision_rows.index[-1],
        horizon_days=horizon,
        benchmark_source=benchmark_source,
        sector_proxy=sector_proxy,
        correlation_source=correlation_source,
        forecast_context_source=str(forecast_context_source or "unavailable"),
        forecast_context_version=str(forecast_context_version or ""),
        forecast_context_samples=int(
            forecast_context[list(FORECAST_CONTEXT_COLUMNS)]
            .notna()
            .all(axis=1)
            .sum()
        ),
        execution_lag_sessions=1,
    )


def _numeric_series(
    frame: pd.DataFrame,
    column: str,
    *,
    required: bool = False,
) -> pd.Series | None:
    if column not in frame.columns:
        if required:
            raise ValueError(f"OHLCV data must include {column!r}")
        return None
    series = frame[column]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    numeric = pd.to_numeric(series, errors="coerce")
    if required:
        return numeric.dropna()
    return numeric


def _normalize_context(context_df: pd.DataFrame | None, index: pd.Index) -> pd.DataFrame:
    if context_df is None or context_df.empty:
        return pd.DataFrame(index=index)
    context = context_df.copy()
    context.columns = [
        str(column).removeprefix("context_").upper()
        for column in context.columns
    ]
    context = context.apply(pd.to_numeric, errors="coerce")
    return context.reindex(index).ffill()


def _context_series(context: pd.DataFrame, ticker: str | None) -> pd.Series | None:
    if not ticker or ticker.upper() not in context.columns:
        return None
    series = pd.to_numeric(context[ticker.upper()], errors="coerce").ffill()
    if series.notna().sum() < 10:
        return None
    return series


def _normalize_forecast_context(
    forecast_context_df: pd.DataFrame | None,
    index: pd.Index,
) -> pd.DataFrame:
    """Align point-in-time forecasts without carrying values across dates."""
    if forecast_context_df is None:
        return pd.DataFrame(
            np.nan,
            index=index,
            columns=FORECAST_CONTEXT_COLUMNS,
            dtype=float,
        )
    if not isinstance(forecast_context_df, pd.DataFrame):
        raise TypeError("forecast_context_df must be a pandas DataFrame")
    if not forecast_context_df.index.is_unique:
        raise ValueError("forecast_context_df index must be unique")

    missing = sorted(
        set(FORECAST_CONTEXT_COLUMNS) - set(forecast_context_df.columns)
    )
    if missing:
        raise ValueError(
            "forecast_context_df is missing required columns: "
            f"{missing}"
        )

    numeric = forecast_context_df.loc[:, FORECAST_CONTEXT_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    supplied = numeric.notna()
    partial_rows = supplied.any(axis=1) & ~supplied.all(axis=1)
    if partial_rows.any():
        bad_indices = list(numeric.index[partial_rows][:5])
        raise ValueError(
            "forecast_context_df rows must supply every forecast field; "
            f"partial rows at {bad_indices}"
        )
    complete = numeric.loc[supplied.all(axis=1)]
    if not complete.empty:
        _validate_forecast_context_frame(complete)
    return numeric.reindex(index)


def _validated_forecast_context(
    context: Mapping[str, object] | None,
    *,
    argument_name: str,
) -> dict[str, float]:
    if context is None:
        raise ValueError(
            f"{argument_name} is required for the contextual shadow policy"
        )
    missing = sorted(set(FORECAST_CONTEXT_COLUMNS) - set(context))
    if missing:
        raise ValueError(f"{argument_name} is missing required fields: {missing}")

    values: dict[str, float] = {}
    for column in FORECAST_CONTEXT_COLUMNS:
        try:
            value = float(context[column])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{argument_name}.{column} must be numeric"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(f"{argument_name}.{column} must be finite")
        values[column] = value
    _validate_forecast_context_values(values, argument_name=argument_name)
    return values


def _validate_forecast_context_frame(frame: pd.DataFrame) -> None:
    values = {
        column: pd.to_numeric(frame[column], errors="coerce")
        for column in FORECAST_CONTEXT_COLUMNS
    }
    for column, series in values.items():
        if (~np.isfinite(series.to_numpy(dtype=float))).any():
            raise ValueError(f"forecast_context_df.{column} must be finite")
    for row_index, row in frame.iterrows():
        _validate_forecast_context_values(
            {column: float(row[column]) for column in FORECAST_CONTEXT_COLUMNS},
            argument_name=f"forecast_context_df[{row_index!r}]",
        )


def _validate_forecast_context_values(
    values: Mapping[str, float],
    *,
    argument_name: str,
) -> None:
    probability = float(values["forecast_probability_up"])
    if probability < 0.0 or probability > 1.0:
        raise ValueError(
            f"{argument_name}.forecast_probability_up must be between 0 and 1"
        )
    agreement = float(values["forecast_model_agreement"])
    if agreement < 0.0 or agreement > 1.0:
        raise ValueError(
            f"{argument_name}.forecast_model_agreement must be between 0 and 1"
        )
    uncertainty = float(values["forecast_uncertainty"])
    if uncertainty < 0.0:
        raise ValueError(
            f"{argument_name}.forecast_uncertainty must be nonnegative"
        )
    if (
        float(values["forecast_lower_bound"])
        > float(values["forecast_return"]) + 1e-12
    ):
        raise ValueError(
            f"{argument_name}.forecast_lower_bound cannot exceed forecast_return"
        )


def _event_series(
    event_flags: pd.Series | Mapping[object, object] | None,
    index: pd.Index,
) -> pd.Series:
    if event_flags is None:
        return pd.Series(0.0, index=index)
    if isinstance(event_flags, pd.Series):
        raw = event_flags
    else:
        raw = pd.Series(dict(event_flags))
    aligned = pd.to_numeric(raw, errors="coerce").reindex(index).fillna(0.0)
    return (aligned > 0.0).astype(float)
