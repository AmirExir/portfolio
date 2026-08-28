#!/usr/bin/env python3
import argparse
import contextlib
import datetime as dt
import errno
import hashlib
import io
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.data import get_ohlcv
from agent.earnings import (
    DEFAULT_MARKET_TIMEZONE,
    fetch_yfinance_earnings_payload,
    interpret_earnings_payload,
    load_earnings_payload_file,
    us_equity_trading_sessions,
)
from agent.forecast import compare_forecast_models
from agent.evaluation import ForecastObservation, evaluate_forecasts
from agent.ledger import (
    OutcomeRecord,
    PredictionLedger,
    PredictionRecord,
    US_EQUITY_SESSION_CALENDAR,
    UTC_DAILY_24_7_SESSION_CALENDAR,
    UTC_DAILY_STRICT_CLOSE_T_MAX_LAG_SECONDS,
    prediction_strict_close_t_eligible,
    session_close_utc,
)
from agent.outcomes import append_matured_outcomes
from agent.policy import SmartPolicyConfig, smart_policy_report
from agent.portfolio import PortfolioConstraints, allocate_target_weights
from agent.strategy import sma_crossover
from forecast_cache import (
    MODEL_RESULT_CACHE_VERSION,
    build_cache_key,
    cache_payload_fresh,
    forecast_result_from_dict,
    forecast_cache_path,
    frame_fingerprint,
    model_result_cache_path,
    select_model_name,
    snapshot_from_model_results,
    snapshots_to_ranking_frame,
)
from patterns import recognize_patterns


DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMD", "INTC", "GOOGL", "AMZN", "META", "TSLA", "RIOT", "AVGO", "GEV",
    "MU", "WDC", "STX", "SNDK", "SPCX",
    "ORCL", "NFLX", "JPM", "BAC", "WFC", "C", "GS", "MS", "V",
    "UNH", "LLY", "JNJ", "WMT", "PG", "KO", "HD", "PEP", "DELL",
    "DAL", "UAL", "AAL", "LUV", "XOM", "CVX", "COP", "OXY", "SLB", "EOG",
    "SPY", "VOO", "QQQ", "IWM", "DIA", "GLD", "SLV", "USO", "TLT",
    "XLK", "XLF", "XLE", "XLV", "XLY",
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "BNB-USD", "AVAX-USD",
    "ORCA-USD", "PNUT-USD", "DOGE-USD", "SHIB-USD", "FLOKI-USD", "PEPE-USD", "ONDO-USD",
    "ZEC-USD", "COMP5692-USD", "HYPE32196-USD", "MNT27075-USD", "UNI7083-USD", "ENA-USD", "DOT-USD",
]
REPORT_SYMBOLS = {
    "SPCX": "SPCX",
    "BTC-USD": "BTC",
    "ETH-USD": "ETH",
    "SOL-USD": "SOL",
    "XRP-USD": "XRP",
    "ADA-USD": "ADA",
    "BNB-USD": "BNB",
    "AVAX-USD": "AVAX",
    "ORCA-USD": "ORCA",
    "PNUT-USD": "PNUT",
    "DOGE-USD": "DOGE",
    "SHIB-USD": "SHIB",
    "FLOKI-USD": "FLOKI",
    "PEPE-USD": "PEPE",
    "ONDO-USD": "ONDO",
    "ZEC-USD": "ZEC",
    "COMP5692-USD": "COMP",
    "HYPE32196-USD": "HYPE",
    "MNT27075-USD": "MNT",
    "UNI7083-USD": "UNI",
    "ENA-USD": "ENA",
    "DOT-USD": "DOT",
}
MARKET_CONTEXT_TICKERS = [
    "SPY", "VOO", "QQQ", "IWM", "DIA", "^VIX",
    "XLK", "XLF", "XLE", "XLV", "XLY",
    "TLT", "GLD", "SLV", "USO", "AVGO",
]
SEQUENCE_MODEL_NAMES = {"LSTM", "Transformer"}
DEFAULT_ADAPTIVE_EXPLORATION_QUOTA = 2
DEFAULT_ADAPTIVE_MIN_NONOVERLAPPING_SAMPLES = 2
EXTERNAL_RUNTIME_CEILING_MINUTES = 270.0
EXTERNAL_RUNTIME_MARGIN_MINUTES = 30.0
PUBLICATION_FINALIZATION_RESERVE_MINUTES = 30.0
OVERNIGHT_PUBLICATION_DEADLINE_MINUTES = (
    EXTERNAL_RUNTIME_CEILING_MINUTES - EXTERNAL_RUNTIME_MARGIN_MINUTES
)
OVERNIGHT_RANKING_CUTOFF_MINUTES = (
    OVERNIGHT_PUBLICATION_DEADLINE_MINUTES
    - PUBLICATION_FINALIZATION_RESERVE_MINUTES
)
PUBLICATION_LOCK_FILENAME = ".market_agent_publication.lock"
DEFAULT_PUBLICATION_LOCK_TIMEOUT_SECONDS = 120.0
PUBLICATION_LOCK_POLL_SECONDS = 0.05
MARKET_UNIVERSE = "Stocks + crypto + commodities"
MODEL_BUY_CALLS = {"Strong Buy", "Buy"}
MODEL_SELL_CALLS = {"Strong Sell", "Sell"}
POLICY_BUY_CALLS = {"Strong Buy", "Buy"}
POLICY_SELL_TERMS = ("Sell", "Avoid")
RELIABILITY_POLICY_CONFIG = SmartPolicyConfig()
NON_EARNINGS_SECTORS = {
    "Broad Market ETF",
    "Sector ETF",
    "Macro ETF",
}

SYMBOL_SECTORS = {
    **{symbol: "Technology" for symbol in ("AAPL", "MSFT", "NVDA", "AMD", "INTC", "AVGO", "MU", "WDC", "STX", "SNDK", "ORCL", "DELL")},
    **{symbol: "Crypto Infrastructure" for symbol in ("RIOT",)},
    **{symbol: "Communication Services" for symbol in ("GOOGL", "META", "NFLX")},
    **{symbol: "Consumer Discretionary" for symbol in ("AMZN", "TSLA", "HD")},
    **{symbol: "Financials" for symbol in ("JPM", "BAC", "WFC", "C", "GS", "MS", "V")},
    **{symbol: "Health Care" for symbol in ("UNH", "LLY", "JNJ")},
    **{symbol: "Consumer Staples" for symbol in ("WMT", "PG", "KO", "PEP")},
    **{symbol: "Industrials" for symbol in ("GEV", "DAL", "UAL", "AAL", "LUV", "SPCX")},
    **{symbol: "Energy" for symbol in ("XOM", "CVX", "COP", "OXY", "SLB", "EOG")},
    **{symbol: "Broad Market ETF" for symbol in ("SPY", "VOO", "QQQ", "IWM", "DIA")},
    **{symbol: "Sector ETF" for symbol in ("XLK", "XLF", "XLE", "XLV", "XLY")},
    **{symbol: "Macro ETF" for symbol in ("GLD", "SLV", "USO", "TLT")},
    **{REPORT_SYMBOLS.get(symbol, symbol): "Crypto" for symbol in DEFAULT_SYMBOLS if symbol.endswith("-USD")},
}

SYMBOL_CLUSTERS = {
    **{symbol: "AI Mega Cap" for symbol in ("AAPL", "MSFT", "NVDA", "AMD", "AVGO", "GOOGL", "AMZN", "META", "ORCL", "DELL", "XLK", "QQQ")},
    **{symbol: "Legacy Semiconductors" for symbol in ("INTC",)},
    **{symbol: "Memory and Storage" for symbol in ("MU", "WDC", "STX", "SNDK")},
    **{symbol: "Banks" for symbol in ("JPM", "BAC", "WFC", "C", "GS", "MS")},
    **{symbol: "Payments" for symbol in ("V",)},
    **{symbol: "High Beta Growth" for symbol in ("TSLA", "RIOT")},
    **{symbol: "Streaming Media" for symbol in ("NFLX",)},
    **{symbol: "Health Care" for symbol in ("UNH", "LLY", "JNJ", "XLV")},
    **{symbol: "Defensive Consumer" for symbol in ("WMT", "PG", "KO", "PEP")},
    **{symbol: "Consumer Cyclicals" for symbol in ("HD", "XLY")},
    **{symbol: "Airlines" for symbol in ("DAL", "UAL", "AAL", "LUV")},
    **{symbol: "Industrial Growth" for symbol in ("GEV", "SPCX")},
    **{symbol: "Energy Complex" for symbol in ("XOM", "CVX", "COP", "OXY", "SLB", "EOG", "XLE", "USO")},
    **{symbol: "Broad Index" for symbol in ("SPY", "VOO", "IWM", "DIA")},
    **{symbol: "Financial Sector ETF" for symbol in ("XLF",)},
    **{symbol: "Precious Metals" for symbol in ("GLD", "SLV")},
    **{symbol: "Rates" for symbol in ("TLT",)},
    **{REPORT_SYMBOLS.get(symbol, symbol): "Crypto" for symbol in DEFAULT_SYMBOLS if symbol.endswith("-USD")},
}


def parse_symbols(symbols_text: str) -> list[str]:
    symbols = []
    for raw_symbol in symbols_text.replace("\n", ",").split(","):
        symbol = raw_symbol.strip().upper()
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols


def parse_horizons(horizons_text: str, main_horizon: int | None = None) -> list[int]:
    horizons = []
    for raw_horizon in str(horizons_text or "").replace(";", ",").split(","):
        raw_horizon = raw_horizon.strip()
        if not raw_horizon:
            continue
        try:
            horizon = max(1, int(raw_horizon))
        except ValueError:
            continue
        if horizon != int(main_horizon or 0) and horizon not in horizons:
            horizons.append(horizon)
    return horizons


def report_symbol(symbol: str) -> str:
    return REPORT_SYMBOLS.get(str(symbol), str(symbol))


def session_calendar_for_symbol(symbol: str) -> str:
    """Return the daily-session convention for a configured market symbol."""

    if str(symbol).strip().upper().endswith("-USD"):
        return UTC_DAILY_24_7_SESSION_CALENDAR
    return US_EQUITY_SESSION_CALENDAR


@contextlib.contextmanager
def _quiet_yfinance_output():
    sink = io.StringIO()
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield
    finally:
        logging.disable(previous_disable_level)


def _yf_download(tickers, **kwargs) -> pd.DataFrame:
    kwargs.setdefault("progress", False)
    kwargs.setdefault("threads", False)
    try:
        with _quiet_yfinance_output():
            return yf.download(tickers, **kwargs)
    except Exception:
        return pd.DataFrame()


def load_market_context(
    history_days: int,
    max_lag_sessions: int = 1,
    *,
    now: dt.datetime | None = None,
) -> pd.DataFrame:
    """Load completed, sufficiently fresh daily market-context bars."""

    reference_now = now or dt.datetime.now(dt.timezone.utc)
    if reference_now.tzinfo is None or reference_now.utcoffset() is None:
        reference_now = reference_now.replace(tzinfo=dt.timezone.utc)
    else:
        reference_now = reference_now.astimezone(dt.timezone.utc)
    requested_start = pd.Timestamp(reference_now.date()) - pd.Timedelta(
        days=max(int(history_days), 1)
    )
    start = requested_start.strftime("%Y-%m-%d")
    raw = _yf_download(MARKET_CONTEXT_TICKERS, start=start, interval="1d")
    if raw.empty:
        raise ValueError("No market-context data returned.")

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = close.rename(columns={ticker: f"context_{ticker}" for ticker in close.columns})
    context = close.dropna(how="all").ffill()
    context.index = pd.to_datetime(context.index, errors="coerce")
    if getattr(context.index, "tz", None) is not None:
        context.index = context.index.tz_convert(None)
    context = context.loc[~context.index.isna()].sort_index()
    context = context.loc[context.index >= requested_start].copy()
    completed = require_fresh_daily_market_data(
        context,
        "SPY",
        max_lag_sessions,
        now=now,
    )
    completed.attrs["history_coverage"] = {
        "requested_calendar_days": int(history_days),
        "requested_start": requested_start.date().isoformat(),
        "available_start": (
            completed.index.min().date().isoformat()
            if not completed.empty
            else None
        ),
        "available_rows": int(len(completed)),
    }
    return completed


def signal_quality(confidence_pct: float) -> str:
    edge_pct = max(float(confidence_pct) - 50.0, 0.0)
    if edge_pct >= 15.0:
        return "High Edge"
    if edge_pct >= 8.0:
        return "Moderate Edge"
    if edge_pct >= 3.0:
        return "Weak Edge"
    return "No Edge"


def model_call(forecast_change_pct: float, expected_error_pct: float, confidence_pct: float) -> str:
    edge_pct = max(float(confidence_pct) - 50.0, 0.0)
    if edge_pct < 3.0 or abs(forecast_change_pct) < expected_error_pct * 0.5:
        return "Neutral / No Edge"
    if forecast_change_pct >= expected_error_pct and edge_pct >= 8.0:
        return "Strong Buy"
    if forecast_change_pct > 0:
        return "Buy"
    if forecast_change_pct <= -expected_error_pct and edge_pct >= 8.0:
        return "Strong Sell"
    if forecast_change_pct < 0:
        return "Sell"
    return "Neutral / No Edge"


def best_model_name(model_results: dict, preferred: str = "Ensemble") -> str:
    if preferred in model_results and not model_results[preferred].forecast.empty:
        return preferred

    candidates = [
        (name, result)
        for name, result in model_results.items()
        if result.forecast is not None and not result.forecast.empty
    ]
    if not candidates:
        return ""
    return min(candidates, key=lambda item: item[1].metrics.get("holdout_mae_pct", np.inf))[0]


def result_metric(model_results: dict, model_name: str, metric_name: str, default=np.nan):
    result = model_results.get(model_name)
    if result is None:
        return default
    return result.metrics.get(metric_name, default)


def clean_close(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return pd.to_numeric(close, errors="coerce").dropna()


def completed_daily_market_data(
    df: pd.DataFrame,
    symbol: str,
    *,
    now: dt.datetime | None = None,
) -> tuple[pd.DataFrame, dt.date, dt.date, int]:
    """Drop an incomplete daily bar and measure completed-session data lag.

    US equities use the latest regular session whose close should already be
    available, with a 30-minute settlement buffer. Crypto uses the latest
    completed UTC date. The returned lag is expressed in exchange sessions for
    equities and completed UTC days for crypto.
    """

    if df.empty:
        raise ValueError(f"No OHLCV data returned for {symbol}.")
    current_time = now or dt.datetime.now(dt.timezone.utc)
    if current_time.tzinfo is None or current_time.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    current_time = current_time.astimezone(dt.timezone.utc)
    observed_sessions = tuple(
        pd.Timestamp(value).date() for value in df.index
    )

    if str(symbol).upper().endswith("-USD"):
        expected_session = current_time.date() - dt.timedelta(days=1)
        completed_sessions = tuple(
            session
            for session in observed_sessions
            if session <= expected_session
        )
        if not completed_sessions:
            raise ValueError(
                f"No completed daily OHLCV bars returned for {symbol}."
            )
        latest_session = max(completed_sessions)
        lag_sessions = max(
            (expected_session - latest_session).days,
            0,
        )
    else:
        market_time = current_time.astimezone(
            ZoneInfo(DEFAULT_MARKET_TIMEZONE)
        )
        close_ready_time = dt.time(16, 30)
        cutoff_date = market_time.date()
        if market_time.time() < close_ready_time:
            cutoff_date -= dt.timedelta(days=1)
        completed_observed_sessions = tuple(
            session
            for session in observed_sessions
            if session <= cutoff_date
        )
        trading_sessions = us_equity_trading_sessions(
            as_of=current_time,
            observed_sessions=completed_observed_sessions,
            future_session_count=1,
        )
        completed_sessions = tuple(
            session
            for session in trading_sessions
            if session <= cutoff_date
        )
        if not completed_sessions:
            raise ValueError(
                f"No completed daily OHLCV bars returned for {symbol}."
            )
        expected_session = max(completed_sessions)
        available_sessions = tuple(
            session
            for session in completed_observed_sessions
            if session <= expected_session
        )
        if not available_sessions:
            raise ValueError(
                f"No completed daily OHLCV bars returned for {symbol}."
            )
        latest_session = max(available_sessions)
        lag_sessions = sum(
            latest_session < session <= expected_session
            for session in trading_sessions
        )

    completed = df[
        pd.Index(
            [pd.Timestamp(value).date() for value in df.index]
        )
        <= expected_session
    ].copy()
    return completed, latest_session, expected_session, lag_sessions


def require_fresh_daily_market_data(
    df: pd.DataFrame,
    symbol: str,
    max_lag_sessions: int,
    *,
    now: dt.datetime | None = None,
) -> pd.DataFrame:
    """Return completed daily bars or reject a stale cache explicitly."""

    if isinstance(max_lag_sessions, bool) or int(max_lag_sessions) < 0:
        raise ValueError("max_lag_sessions must be a non-negative integer")
    completed, latest, expected, lag = completed_daily_market_data(
        df,
        symbol,
        now=now,
    )
    if lag > int(max_lag_sessions):
        unit = "completed UTC days" if str(symbol).endswith("-USD") else "trading sessions"
        raise ValueError(
            f"stale OHLCV data: latest {latest.isoformat()}, expected "
            f"{expected.isoformat()}, lag {lag} {unit} exceeds allowed "
            f"{int(max_lag_sessions)}"
        )
    return completed


def row_value(row: dict, *keys, default=None):
    for key in keys:
        if key in row:
            return row[key]
    return default


def finite_row_float(row: dict, *keys: str, default: float = np.nan) -> float:
    """Return a finite numeric row value or a fail-closed default."""

    try:
        value = float(row_value(row, *keys, default=default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def as_of_sessions_from_rows(rows: list[dict]) -> list[str]:
    """Return the sorted completed market sessions represented by report rows."""

    return sorted(
        {
            str(row_value(row, "As Of Session", default="") or "").strip()
            for row in rows
            if str(row_value(row, "As Of Session", default="") or "").strip()
        }
    )


def ranking_score(row: dict) -> float:
    try:
        score = float(row_value(row, "Policy Score", "Score", default=0.0))
        return score if np.isfinite(score) else 0.0
    except Exception:
        return 0.0


def forecast_return_pct(row: dict) -> float:
    try:
        value = float(row_value(row, "forecast_return_pct", "Forecast Return %", default=0.0) or 0.0)
        return value if np.isfinite(value) else 0.0
    except Exception:
        return 0.0


def min_signal_return_pct_from_args(args: argparse.Namespace) -> float:
    try:
        return max(0.0, float(getattr(args, "min_signal_return_pct", 2.0) or 0.0))
    except Exception:
        return 2.0


def max_signal_rows_from_args(args: argparse.Namespace) -> int:
    try:
        max_signal_rows = max(0, int(getattr(args, "max_signal_rows", 0) or 0))
        if max_signal_rows > 0:
            return max_signal_rows
        return max(0, int(getattr(args, "top_n", 0) or 0))
    except Exception:
        return 0


def signal_threshold_metadata(args: argparse.Namespace) -> dict:
    """Describe the forecast and calibration gates used in published signals."""

    return {
        "min_forecast_return_pct": min_signal_return_pct_from_args(args),
        "max_rows_per_side": max_signal_rows_from_args(args),
        "directional_call_required": True,
        "rl_selected_model_allowed": False,
        "require_oos_validation": True,
        "oos_validation_required_value": True,
        "oos_validation_must_be_literal_boolean": True,
        "minimum_validation_samples": (
            RELIABILITY_POLICY_CONFIG.min_validation_samples
        ),
        "minimum_nonoverlapping_validation_samples": (
            RELIABILITY_POLICY_CONFIG.min_nonoverlapping_validation_samples
        ),
        "minimum_direction_hit_rate_pct": (
            RELIABILITY_POLICY_CONFIG.min_direction_accuracy_pct
        ),
        "minimum_direction_skill_pct_exclusive": (
            RELIABILITY_POLICY_CONFIG.min_direction_skill_pct
        ),
        "maximum_calibration_error_pct": (
            RELIABILITY_POLICY_CONFIG.max_calibration_error_pct
        ),
        "maximum_brier_score": RELIABILITY_POLICY_CONFIG.max_brier_score,
        "minimum_mae_skill_score": (
            RELIABILITY_POLICY_CONFIG.min_mae_skill_score
        ),
        "minimum_brier_skill_score": (
            RELIABILITY_POLICY_CONFIG.min_brier_skill_score
        ),
    }


def cap_signal_rows(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    max_rows = max_signal_rows_from_args(args)
    return rows[:max_rows] if max_rows > 0 else rows


def model_call_text(row: dict) -> str:
    return str(row_value(row, "model_call", "Model Call", default="")).strip()


def smart_policy_text(row: dict) -> str:
    return str(row_value(row, "Smart Policy", default="")).strip()


def has_buy_call(row: dict) -> bool:
    return model_call_text(row) in MODEL_BUY_CALLS or smart_policy_text(row) in POLICY_BUY_CALLS


def has_sell_call(row: dict) -> bool:
    smart_policy = smart_policy_text(row)
    return model_call_text(row) in MODEL_SELL_CALLS or any(term in smart_policy for term in POLICY_SELL_TERMS)


def has_model_buy_call(row: dict) -> bool:
    return model_call_text(row) in MODEL_BUY_CALLS


def has_model_sell_call(row: dict) -> bool:
    return model_call_text(row) in MODEL_SELL_CALLS


def is_policy_buy(row: dict) -> bool:
    if row_value(row, "Smart Policy", default=None):
        try:
            target_pct = float(row_value(row, "Policy Target %", default=0.0) or 0.0)
        except Exception:
            target_pct = 0.0
        return target_pct > 0.0 and ranking_score(row) > 0.0
    return float(row_value(row, "Forecast Return %", default=0.0) or 0.0) > 0.0


def is_policy_sell(row: dict) -> bool:
    if row_value(row, "Smart Policy", default=None):
        return ranking_score(row) < 0.0 or "Sell" in str(row_value(row, "Smart Policy", default=""))
    return float(row_value(row, "Forecast Return %", default=0.0) or 0.0) < 0.0


def is_threshold_buy(row: dict, args: argparse.Namespace) -> bool:
    """Return whether a forecast qualifies for research-report visibility.

    Portfolio authorization is deliberately evaluated separately.  Missing
    broker weights or covariance must zero executable allocation, but it must
    not erase an otherwise valid, non-RL out-of-sample forecast from the
    research report.
    """
    return not model_signal_qualification_failures(row, args, side="buy")


def is_threshold_sell(row: dict, args: argparse.Namespace) -> bool:
    return not model_signal_qualification_failures(row, args, side="sell")


def model_signal_qualification_failures(
    row: dict,
    args: argparse.Namespace,
    *,
    side: str,
) -> tuple[str, ...]:
    """Return exact model-signal gate codes for one requested side."""

    if side not in {"buy", "sell"}:
        raise ValueError("side must be 'buy' or 'sell'")
    failures: list[str] = []
    if str(
        row_value(row, "Selected Model", "selected_model", default="")
    ).strip() == "RL Policy":
        failures.append("selected_model_rl_policy")
    threshold = min_signal_return_pct_from_args(args)
    forecast_return = forecast_return_pct(row)
    if side == "buy":
        if forecast_return < threshold:
            failures.append("forecast_return_below_buy_threshold")
        if not has_model_buy_call(row):
            failures.append("model_call_not_buy")
    else:
        if forecast_return > -threshold:
            failures.append("forecast_return_above_sell_threshold")
        if not has_model_sell_call(row):
            failures.append("model_call_not_sell")
    failures.extend(reliability_gate_failures(row))
    return tuple(dict.fromkeys(failures))


def is_unqualified_model_buy(row: dict, args: argparse.Namespace) -> bool:
    """Return a visible raw buy call that failed qualification gates."""

    return (
        str(
            row_value(row, "Selected Model", "selected_model", default="")
        ).strip()
        != "RL Policy"
        and forecast_return_pct(row) >= min_signal_return_pct_from_args(args)
        and has_model_buy_call(row)
        and bool(model_signal_qualification_failures(row, args, side="buy"))
    )


def is_unqualified_model_sell(row: dict, args: argparse.Namespace) -> bool:
    """Return a visible raw sell call that failed qualification gates."""

    return (
        str(
            row_value(row, "Selected Model", "selected_model", default="")
        ).strip()
        != "RL Policy"
        and forecast_return_pct(row) <= -min_signal_return_pct_from_args(args)
        and has_model_sell_call(row)
        and bool(model_signal_qualification_failures(row, args, side="sell"))
    )


def is_policy_watch_buy(row: dict, args: argparse.Namespace) -> bool:
    if reliability_grade(row) == "Low":
        return False
    if is_threshold_buy(row, args):
        return False
    if smart_policy_text(row) not in POLICY_BUY_CALLS:
        return False
    try:
        target_pct = float(row_value(row, "Policy Target %", default=0.0) or 0.0)
    except Exception:
        target_pct = 0.0
    return target_pct > 0.0 and ranking_score(row) > 0.0 and forecast_return_pct(row) >= min_signal_return_pct_from_args(args)


def is_policy_watch_sell(row: dict, args: argparse.Namespace) -> bool:
    if reliability_grade(row) == "Low":
        return False
    if is_threshold_sell(row, args):
        return False
    policy = smart_policy_text(row)
    if not any(term in policy for term in POLICY_SELL_TERMS):
        return False
    return ranking_score(row) < 0.0 and forecast_return_pct(row) <= -min_signal_return_pct_from_args(args)


def reliability_gate_failures(row: dict) -> tuple[str, ...]:
    """Return stable gate codes shared by grading and report explanations."""

    failures: list[str] = []
    if str(
        row_value(row, "selected_model", "Selected Model", default="")
    ).strip() == "RL Policy":
        failures.append("selected_model_rl_policy")
    if bool(
        row_value(row, "forecast_outlier", "Forecast Outlier", default=False)
    ):
        failures.append("forecast_outlier")
    if row_value(
        row,
        "validation_is_oos",
        "Validation Is OOS",
        default=None,
    ) is not True:
        failures.append("validation_is_oos_not_literal_true")
    forecast_return = abs(forecast_return_pct(row))
    expected_error = abs(
        finite_row_float(
            row,
            "expected_error_pct",
            "Expected Error %",
        )
    )
    edge = abs(
        finite_row_float(
            row,
            "model_edge_pct",
            "Model Edge %",
        )
    )
    hit_rate = finite_row_float(row, "Direction Hit Rate %")
    direction_skill_pct = finite_row_float(row, "Direction Skill %")
    validation_mae = abs(finite_row_float(row, "Validation MAE %"))
    validation_samples = finite_row_float(row, "Validation Samples")
    nonoverlapping_samples = finite_row_float(
        row,
        "Nonoverlapping Validation Samples",
    )
    calibration_error = finite_row_float(row, "Calibration Error %")
    brier_score = finite_row_float(row, "Brier Score")
    mae_skill_score = finite_row_float(row, "MAE Skill Score")
    brier_skill_score = finite_row_float(row, "Brier Skill Score")
    error_ratio = forecast_return / expected_error if expected_error > 0 else np.inf

    gates = (
        (np.isfinite(expected_error), "expected_error_not_finite"),
        (np.isfinite(edge), "model_edge_not_finite"),
        (np.isfinite(validation_mae), "validation_mae_not_finite"),
        (
            np.isfinite(validation_samples)
            and validation_samples
            >= RELIABILITY_POLICY_CONFIG.min_validation_samples,
            "validation_samples_below_minimum",
        ),
        (
            np.isfinite(nonoverlapping_samples)
            and nonoverlapping_samples
            >= RELIABILITY_POLICY_CONFIG.min_nonoverlapping_validation_samples,
            "nonoverlapping_validation_samples_below_minimum",
        ),
        (
            np.isfinite(hit_rate)
            and hit_rate
            >= RELIABILITY_POLICY_CONFIG.min_direction_accuracy_pct,
            "direction_hit_rate_below_minimum",
        ),
        (
            np.isfinite(direction_skill_pct)
            and direction_skill_pct
            > RELIABILITY_POLICY_CONFIG.min_direction_skill_pct,
            "direction_skill_not_above_minimum",
        ),
        (
            np.isfinite(calibration_error)
            and calibration_error
            <= RELIABILITY_POLICY_CONFIG.max_calibration_error_pct,
            "calibration_error_above_maximum",
        ),
        (
            np.isfinite(brier_score)
            and brier_score <= RELIABILITY_POLICY_CONFIG.max_brier_score,
            "brier_score_above_maximum",
        ),
        (
            np.isfinite(mae_skill_score)
            and mae_skill_score
            > RELIABILITY_POLICY_CONFIG.min_mae_skill_score,
            "mae_skill_not_above_minimum",
        ),
        (
            np.isfinite(brier_skill_score)
            and brier_skill_score
            > RELIABILITY_POLICY_CONFIG.min_brier_skill_score,
            "brier_skill_not_above_minimum",
        ),
    )
    failures.extend(code for passed, code in gates if not passed)
    if (
        np.isfinite(validation_mae)
        and validation_mae > 0
        and forecast_return < validation_mae * 0.5
    ):
        failures.append("forecast_magnitude_below_half_validation_mae")
    if np.isfinite(edge) and edge < 3.0:
        failures.append("model_edge_below_speculative_minimum")
    return tuple(failures)


def reliability_grade(row: dict) -> str:
    """Grade only forecasts with usable out-of-sample calibration evidence."""

    if reliability_gate_failures(row):
        return "Low"
    forecast_return = abs(forecast_return_pct(row))
    expected_error = abs(
        finite_row_float(row, "expected_error_pct", "Expected Error %")
    )
    edge = abs(finite_row_float(row, "model_edge_pct", "Model Edge %"))
    hit_rate = finite_row_float(row, "Direction Hit Rate %")
    error_ratio = forecast_return / expected_error if expected_error > 0 else np.inf
    if np.isfinite(hit_rate) and hit_rate >= 56.0 and edge >= 15.0 and error_ratio >= 0.75:
        return "High"
    if (not np.isfinite(hit_rate) or hit_rate >= 52.0) and edge >= 8.0 and error_ratio >= 0.45:
        return "Moderate"
    if edge >= 3.0:
        return "Speculative"
    return "Low"


def signal_tier(row: dict, args: argparse.Namespace) -> str:
    if is_threshold_buy(row, args):
        return "Model-Qualified Buy"
    if is_threshold_sell(row, args):
        return "Model-Qualified Sell/Avoid"
    if is_policy_watch_buy(row, args):
        return "Policy Watch Buy"
    if is_policy_watch_sell(row, args):
        return "Policy Watch Sell/Avoid"
    if is_unqualified_model_buy(row, args):
        return "Unqualified Candidate Buy"
    if is_unqualified_model_sell(row, args):
        return "Unqualified Candidate Sell/Avoid"
    return "Neutral / Monitor"


def enrich_rows_with_signal_metadata(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    enriched = []
    for row in rows:
        item = dict(row)
        item["Reliability"] = reliability_grade(item)
        if item["Reliability"] == "Low":
            prior_target = float(row_value(item, "Policy Target %", default=0.0) or 0.0)
            item["Policy Target %"] = 0.0
            if smart_policy_text(item) in POLICY_BUY_CALLS:
                item["Smart Policy"] = "Hold / Watch"
            if prior_target > 0.0:
                prior_reason = str(row_value(item, "Policy Reason", default="") or "").strip()
                item["Policy Reason"] = (
                    "allocation blocked: Low reliability"
                    + (f"; {prior_reason}" if prior_reason else "")
                )
        item["Signal Tier"] = signal_tier(item, args)
        item["Qualified Model Signal"] = item["Signal Tier"].startswith(
            "Model-Qualified"
        )
        item["Policy Watchlist"] = item["Signal Tier"].startswith("Policy Watch")
        item["Unqualified Candidate"] = item["Signal Tier"].startswith(
            "Unqualified Candidate"
        )
        if has_model_buy_call(item):
            qualification_failures = model_signal_qualification_failures(
                item,
                args,
                side="buy",
            )
        elif has_model_sell_call(item):
            qualification_failures = model_signal_qualification_failures(
                item,
                args,
                side="sell",
            )
        else:
            qualification_failures = ()
        item["Qualification Gate Failures"] = list(
            qualification_failures
        )
        item["Risk Overlay"] = smart_policy_text(item) or "Unavailable"
        enriched.append(item)
    return enriched


def apply_portfolio_constraints(
    rows: list[dict],
    close_history: dict[str, pd.Series],
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    """Normalize independent symbol targets into one constrained portfolio."""
    proposed = {
        str(row_value(row, "Symbol", default="")).upper(): max(
            float(row_value(row, "Policy Target %", default=0.0) or 0.0) / 100.0,
            0.0,
        )
        for row in rows
        if str(row_value(row, "Symbol", default="")).strip()
    }
    constraints = PortfolioConstraints(
        cash_reserve=float(getattr(args, "portfolio_cash_reserve_pct", 15.0)) / 100.0,
        max_name_weight=float(getattr(args, "portfolio_max_name_pct", 5.0)) / 100.0,
        max_sector_weight=float(getattr(args, "portfolio_max_sector_pct", 20.0)) / 100.0,
        max_cluster_weight=float(getattr(args, "portfolio_max_cluster_pct", 15.0)) / 100.0,
        max_annual_volatility=float(getattr(args, "portfolio_max_volatility_pct", 15.0)) / 100.0,
        max_turnover=float(getattr(args, "portfolio_max_turnover_pct", 20.0)) / 100.0,
        drawdown_circuit_breaker=float(getattr(args, "portfolio_drawdown_breaker_pct", 10.0)) / 100.0,
    )
    active_proposed = {symbol: weight for symbol, weight in proposed.items() if weight > 0.0}
    current_weights, current_weight_source = current_portfolio_weights(args)
    portfolio_state_verified = current_weight_source == "explicit_cli"
    selected_symbols = sorted(set(active_proposed) | set(current_weights))
    covariance = _annual_covariance(close_history, selected_symbols)
    covariance_verified = covariance is not None or not active_proposed
    current_drawdown = -abs(float(getattr(args, "portfolio_drawdown_pct", 0.0))) / 100.0
    sectors = {symbol: SYMBOL_SECTORS[symbol] for symbol in selected_symbols if symbol in SYMBOL_SECTORS}
    clusters = {symbol: SYMBOL_CLUSTERS[symbol] for symbol in selected_symbols if symbol in SYMBOL_CLUSTERS}
    missing_sector_symbols = sorted(
        set(selected_symbols) - set(sectors)
    )
    missing_cluster_symbols = sorted(
        set(selected_symbols) - set(clusters)
    )
    classification_verified = not (
        missing_sector_symbols or missing_cluster_symbols
    )

    try:
        allocation = allocate_target_weights(
            active_proposed,
            sectors=sectors,
            correlation_clusters=clusters,
            current_weights=current_weights,
            annual_covariance=covariance,
            current_drawdown=current_drawdown,
            constraints=constraints,
        )
    except ValueError as exc:
        logging.warning("Portfolio covariance unavailable; applying non-volatility constraints: %s", exc)
        allocation = allocate_target_weights(
            active_proposed,
            sectors=sectors,
            correlation_clusters=clusters,
            current_weights=current_weights,
            annual_covariance=None,
            current_drawdown=current_drawdown,
            constraints=constraints,
        )

    allocation_eligible = bool(
        portfolio_state_verified
        and classification_verified
        and (
            covariance_verified
            or allocation.circuit_breaker_triggered
        )
    )
    authorization_blockers: list[str] = []
    if not portfolio_state_verified:
        authorization_blockers.append(
            "current broker/executed weights were not verified"
        )
    if not covariance_verified and not allocation.circuit_breaker_triggered:
        authorization_blockers.append(
            "portfolio covariance was unavailable"
        )
    if not classification_verified:
        authorization_blockers.append(
            "sector/correlation classification was unavailable for "
            + ", ".join(
                sorted(
                    set(missing_sector_symbols)
                    | set(missing_cluster_symbols)
                )
            )
        )
    binding_constraints = list(allocation.binding_constraints)
    if not portfolio_state_verified:
        binding_constraints.append("unverified_portfolio_state")
    if not covariance_verified and not allocation.circuit_breaker_triggered:
        binding_constraints.append("covariance_unavailable")
    if not classification_verified:
        binding_constraints.append("classification_unavailable")
    binding_text = ", ".join(dict.fromkeys(binding_constraints))
    authorized_targets = (
        dict(allocation.target_weights)
        if allocation_eligible
        else {symbol: 0.0 for symbol in selected_symbols}
    )
    authorized_gross = (
        allocation.gross_exposure if allocation_eligible else 0.0
    )
    authorized_cash = 1.0 - authorized_gross
    authorized_turnover = allocation.turnover if allocation_eligible else 0.0
    authorized_volatility = (
        allocation.annualized_volatility if allocation_eligible else None
    )
    normalized_rows = []
    for row in rows:
        item = dict(row)
        symbol = str(row_value(item, "Symbol", default="")).upper()
        proposed_pct = max(float(row_value(item, "Policy Target %", default=0.0) or 0.0), 0.0)
        final_pct = float(authorized_targets.get(symbol, 0.0) * 100.0)
        item["Pre-Portfolio Target %"] = proposed_pct
        item["Policy Target %"] = final_pct
        item["Portfolio Gross %"] = authorized_gross * 100.0
        item["Portfolio Cash %"] = authorized_cash * 100.0
        item["Portfolio Binding Constraints"] = binding_text
        item["Portfolio State Verified"] = portfolio_state_verified
        item["Portfolio Covariance Verified"] = covariance_verified
        item["Portfolio Classification Verified"] = (
            classification_verified
        )
        item["Policy Allocation Eligible"] = allocation_eligible
        item["Portfolio Allocation Blocked"] = bool(proposed_pct > 0.0 and final_pct <= 0.0)
        if final_pct < proposed_pct - 1e-9:
            reason = str(row_value(item, "Policy Reason", default="") or "").strip()
            suffix = (
                "allocation blocked: " + "; ".join(authorization_blockers)
                if not allocation_eligible
                else f"portfolio constrained to {final_pct:.2f}%"
            )
            item["Policy Reason"] = f"{reason}; {suffix}" if reason else suffix
        if proposed_pct > 0.0 and final_pct <= 0.0 and smart_policy_text(item) in POLICY_BUY_CALLS:
            item["Smart Policy"] = "Hold / Watch"
        item["Risk Overlay"] = smart_policy_text(item) or "Unavailable"
        item["Signal Tier"] = signal_tier(item, args)
        item["Qualified Model Signal"] = item["Signal Tier"].startswith(
            "Model-Qualified"
        )
        item["Policy Watchlist"] = item["Signal Tier"].startswith(
            "Policy Watch"
        )
        item["Unqualified Candidate"] = item["Signal Tier"].startswith(
            "Unqualified Candidate"
        )
        normalized_rows.append(item)

    diagnostics = {
        "gross_exposure_pct": authorized_gross * 100.0,
        "cash_weight_pct": authorized_cash * 100.0,
        "turnover_pct": authorized_turnover * 100.0,
        "annualized_volatility_pct": (
            authorized_volatility * 100.0
            if authorized_volatility is not None
            else None
        ),
        "sector_exposures_pct": {
            key: value * 100.0 for key, value in allocation.sector_exposures.items()
        },
        "cluster_exposures_pct": {
            key: value * 100.0 for key, value in allocation.cluster_exposures.items()
        },
        "binding_constraints": list(dict.fromkeys(binding_constraints)),
        "warnings": list(allocation.warnings) + authorization_blockers,
        "circuit_breaker_triggered": allocation.circuit_breaker_triggered,
        "target_weights": authorized_targets,
        "indicative_target_weights": allocation.target_weights,
        "current_weight_source": current_weight_source,
        "portfolio_state_verified": portfolio_state_verified,
        "covariance_verified": covariance_verified,
        "classification_verified": classification_verified,
        "missing_sector_symbols": missing_sector_symbols,
        "missing_cluster_symbols": missing_cluster_symbols,
        "allocation_eligible": allocation_eligible,
    }
    return normalized_rows, diagnostics


def current_portfolio_weights(
    args: argparse.Namespace,
) -> tuple[dict[str, float], str]:
    explicit = str(
        getattr(args, "portfolio_current_weights_json", "") or ""
    ).strip()
    if explicit:
        try:
            payload = json.loads(explicit)
            weights = {
                str(symbol).strip().upper(): float(weight)
                for symbol, weight in dict(payload).items()
                if float(weight) >= 0.0
            }
            return weights, "explicit_cli"
        except (TypeError, ValueError, json.JSONDecodeError):
            logging.warning("Ignoring invalid --portfolio-current-weights-json")

    state_path = Path(args.output_dir) / "policy_portfolio_state.json"
    payload = load_json_payload(state_path)
    weights_payload = payload.get("target_weights") or {}
    try:
        weights = {
            str(symbol).strip().upper(): float(weight)
            for symbol, weight in dict(weights_payload).items()
            if float(weight) >= 0.0
        }
    except (TypeError, ValueError):
        weights = {}
    return weights, "previous_recommended_targets" if weights else "assumed_cash_first_run"


def _annual_covariance(
    close_history: dict[str, pd.Series],
    symbols: list[str],
) -> pd.DataFrame | None:
    available = {
        symbol: pd.to_numeric(close_history[symbol], errors="coerce")
        for symbol in symbols
        if symbol in close_history and len(close_history[symbol]) > 1
    }
    if not available or set(available) != set(symbols):
        return None
    prices = pd.concat(available, axis=1).sort_index()
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(how="any").tail(252)
    if len(returns) < 30:
        return None
    covariance = returns.cov() * 252.0
    covariance = (covariance + covariance.T) / 2.0
    covariance += np.eye(len(covariance)) * 1e-12
    return covariance


def build_smart_policy_for_snapshot(
    df: pd.DataFrame,
    model_results: dict,
    selected_model: str,
    args: argparse.Namespace,
    earnings_context: dict | None = None,
) -> dict:
    selected_result = model_results.get(selected_model)
    forecast_metrics = dict(
        getattr(selected_result, "metrics", {})
        if selected_result is not None
        else {}
    )
    forecast_metrics.update(earnings_context or {})
    try:
        signal = sma_crossover(df, args.pattern_short_window, args.pattern_long_window)
    except Exception:
        signal = None
    try:
        return smart_policy_report(
            df=df,
            risk_fraction=float(
                getattr(args, "policy_risk_fraction", 0.05) or 0.05
            ),
            signal=signal,
            forecast_metrics=forecast_metrics,
            model_results=model_results,
        )
    except Exception as exc:
        return {
            "policy_call": "Policy Unavailable",
            "policy_score": 0.0,
            "policy_target_pct": 0.0,
            "policy_reason": str(exc),
        }


def enrich_snapshot_with_smart_policy(
    snapshot: dict,
    df: pd.DataFrame,
    args: argparse.Namespace,
    earnings_context: dict | None = None,
) -> dict:
    model_results = model_results_from_snapshot(snapshot)
    selected_model = select_model_name(model_results, preferred=primary_model_from_args(args))
    if not selected_model:
        return snapshot
    enriched = dict(snapshot)
    enriched["smart_policy"] = build_smart_policy_for_snapshot(
        df,
        model_results,
        selected_model,
        args,
        earnings_context=earnings_context,
    )
    return enriched


def compact_short_horizon_reports(short_horizon_reports: list[dict], args: argparse.Namespace) -> list[dict]:
    compact_reports = []
    for short_report in short_horizon_reports or []:
        rows = short_report.get("rows") or []
        sorted_rows = sorted(rows, key=ranking_score, reverse=True)
        buys = cap_signal_rows([row for row in sorted_rows if is_threshold_buy(row, args)], args)
        sells = sorted(
            [row for row in rows if is_threshold_sell(row, args)],
            key=ranking_score,
        )
        sells = cap_signal_rows(sells, args)
        compact_reports.append(
            {
                "horizon_days": short_report.get("horizon_days"),
                "sequence_model": short_report.get("sequence_model"),
                "top_buys": buys,
                "top_sells": sells,
                "signal_buys": buys,
                "signal_sells": sells,
                "errors": (short_report.get("errors") or [])[:10],
                "timings": short_report.get("timings") or {},
            }
        )
    return compact_reports


def format_duration(seconds: float) -> str:
    seconds = max(float(seconds or 0.0), 0.0)
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {remainder:.0f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m {remainder:.0f}s"


def load_json_payload(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def save_json_payload(path: Path, payload: dict) -> None:
    write_text_atomic(path, json_dumps_strict(payload, indent=2))


def write_text_atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp_path.write_text(text)
    tmp_path.replace(path)


def model_payload_metric(model_payload: dict, metric_name: str, default=np.inf):
    try:
        value = (model_payload.get("metrics") or {}).get(metric_name, default)
        value = float(value)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def parse_report_generated_at(value: object) -> dt.datetime | None:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    iso_normalized = (
        normalized[:-1] + "+00:00"
        if normalized.endswith("Z")
        else normalized
    )
    try:
        parsed = dt.datetime.fromisoformat(iso_normalized)
    except ValueError:
        parsed = None
        for timestamp_format in (
            "%Y-%m-%dT%H-%M-%SZ",
            "%Y-%m-%dT%H-%M-%S-%fZ",
        ):
            try:
                parsed = dt.datetime.strptime(
                    normalized,
                    timestamp_format,
                ).replace(tzinfo=dt.timezone.utc)
                break
            except ValueError:
                continue
        if parsed is None:
            return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def adaptive_model_evidence_is_valid(
    model_payload: dict,
    *,
    min_nonoverlapping_samples: int,
) -> bool:
    """Require leakage-safe, baseline-positive evidence for adaptive compute."""

    metrics = model_payload.get("metrics") or {}
    if metrics.get("validation_is_oos") is not True:
        return False
    mae = model_payload_metric(model_payload, "holdout_mae_pct")
    mae_skill = model_payload_metric(
        model_payload,
        "mae_skill_score",
        default=float("nan"),
    )
    brier_skill = model_payload_metric(
        model_payload,
        "brier_skill_score",
        default=float("nan"),
    )
    direction_skill = model_payload_metric(
        model_payload,
        "direction_skill_pct",
        default=float("nan"),
    )
    try:
        nonoverlapping_samples = int(
            metrics.get("holdout_nonoverlapping_samples", 0) or 0
        )
    except (TypeError, ValueError):
        return False
    return bool(
        np.isfinite(mae)
        and mae >= 0.0
        and np.isfinite(mae_skill)
        and mae_skill > 0.0
        and np.isfinite(brier_skill)
        and brier_skill > 0.0
        and np.isfinite(direction_skill)
        and direction_skill > 0.0
        and nonoverlapping_samples >= min_nonoverlapping_samples
    )


def session_distance_for_symbol(
    first: dt.date,
    second: dt.date,
    symbol: str,
) -> int:
    """Return completed daily sessions between two as-of labels."""

    start, end = sorted((first, second))
    if str(symbol).upper().endswith("-USD"):
        return max((end - start).days, 0)
    sessions = us_equity_trading_sessions(
        as_of=dt.datetime(
            end.year,
            end.month,
            end.day,
            18,
            tzinfo=dt.timezone.utc,
        ),
        lookback_days=max((end - start).days + 7, 1),
        future_session_count=1,
    )
    return sum(1 for session in sessions if start < session <= end)


def deterministic_adaptive_exploration_models(
    args: argparse.Namespace,
    already_selected: dict[str, str],
) -> dict[str, str]:
    """Rotate a small research-only sequence sample across the universe."""

    try:
        quota = max(
            0,
            int(getattr(args, "adaptive_sequence_exploration_quota", 0) or 0),
        )
    except (TypeError, ValueError):
        return {}
    if quota <= 0 or primary_model_from_args(args) != "Best Validation":
        return {}

    candidates = [
        symbol.upper()
        for symbol in parse_symbols(str(getattr(args, "symbols", "") or ""))
        if symbol.upper() not in already_selected
    ]
    if not candidates:
        return {}
    seed = str(
        getattr(args, "adaptive_sequence_exploration_seed", "") or ""
    ).strip()
    if not seed:
        seed = dt.datetime.now(dt.timezone.utc).date().isoformat()
    horizon = max(int(getattr(args, "horizon", 1) or 1), 1)
    digest = hashlib.sha256(f"{seed}:{horizon}".encode("utf-8")).digest()
    start_index = int.from_bytes(digest[:8], "big") % len(candidates)
    rotated = candidates[start_index:] + candidates[:start_index]
    exploration: dict[str, str] = {}
    for symbol in rotated[: min(quota, len(rotated))]:
        family_digest = hashlib.sha256(
            f"{seed}:{horizon}:{symbol}".encode("utf-8")
        ).digest()
        exploration[symbol] = (
            "lstm" if family_digest[0] % 2 == 0 else "transformer"
        )
    return exploration


def adaptive_sequence_models_from_reports(
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, str]:
    """Select one sequence family per symbol from unique prior evaluations.

    Report files can contain duplicate publications for the same completed
    session.  Counting files therefore overstates the evidence and scheduled
    reports without sequence results dilute the denominator.  This selector
    counts each symbol/as-of/horizon evaluation once, compares the best
    sequence candidate with conventional component models (not the ensemble
    that already contains those components), and enables only the sequence
    family that actually accumulated the most prior wins.

    This remains a research-compute selector, not a model-promotion decision;
    published signals still pass the independent OOS reliability gates.
    """
    override_symbols = parse_symbols(
        str(getattr(args, "adaptive_sequence_symbols", "") or "")
    )
    if override_symbols:
        return {symbol.upper(): "both" for symbol in override_symbols}

    try:
        report_limit = max(
            1,
            int(getattr(args, "adaptive_sequence_report_limit", 40) or 40),
        )
        min_wins = max(
            1,
            int(getattr(args, "adaptive_sequence_min_wins", 2) or 2),
        )
        min_share = max(
            0.0,
            float(getattr(args, "adaptive_sequence_min_share", 0.50) or 0.0),
        )
        min_nonoverlapping_samples = max(
            1,
            int(
                getattr(
                    args,
                    "adaptive_sequence_min_nonoverlap",
                    DEFAULT_ADAPTIVE_MIN_NONOVERLAPPING_SAMPLES,
                )
                or DEFAULT_ADAPTIVE_MIN_NONOVERLAPPING_SAMPLES
            ),
        )
    except (TypeError, ValueError):
        report_limit = 40
        min_wins = 2
        min_share = 0.50
        min_nonoverlapping_samples = (
            DEFAULT_ADAPTIVE_MIN_NONOVERLAPPING_SAMPLES
        )

    indexed_reports: list[tuple[dt.datetime, str, Path]] = []
    for report_path in output_dir.glob("ml_forecast_rankings_20*.json"):
        payload = load_json_payload(report_path)
        generated_at = parse_report_generated_at(payload.get("generated_at"))
        if generated_at is None:
            continue
        indexed_reports.append((generated_at, report_path.name, report_path))
    indexed_reports.sort(key=lambda item: (item[0], item[1]), reverse=True)

    sequence_wins: dict[str, int] = {}
    family_wins: dict[str, dict[str, int]] = {}
    valid_runs: dict[str, int] = {}
    seen_evaluations: set[tuple[str, str, int]] = set()
    accepted_sessions: dict[str, list[dt.date]] = {}
    reports_read = 0

    for _, _, report_path in indexed_reports:
        payload = load_json_payload(report_path)
        if payload.get("run_complete") is not True:
            continue
        snapshots = payload.get("snapshots") or []
        if not snapshots:
            continue
        report_horizon = int(payload.get("horizon_days", 0) or 0)
        requested_horizon = int(getattr(args, "horizon", report_horizon) or 0)
        if report_horizon and requested_horizon and report_horizon != requested_horizon:
            continue
        report_contributed = False
        for snapshot in snapshots:
            symbol = str(snapshot.get("symbol") or "").upper()
            as_of_session = str(snapshot.get("as_of_session") or "").strip()
            evaluation_key = (symbol, as_of_session, report_horizon)
            if not symbol or not as_of_session or evaluation_key in seen_evaluations:
                continue
            try:
                as_of_date = dt.date.fromisoformat(as_of_session)
            except ValueError:
                continue
            if any(
                session_distance_for_symbol(
                    as_of_date,
                    prior_as_of,
                    symbol,
                )
                < max(report_horizon, 1)
                for prior_as_of in accepted_sessions.get(symbol, [])
            ):
                continue
            model_payloads = snapshot.get("models") or {}
            sequence_candidates = [
                (model_payload_metric(model_payload, "holdout_mae_pct"), model_name)
                for model_name, model_payload in model_payloads.items()
                if model_name in SEQUENCE_MODEL_NAMES
                and adaptive_model_evidence_is_valid(
                    model_payload,
                    min_nonoverlapping_samples=min_nonoverlapping_samples,
                )
            ]
            sequence_candidates = [
                (mae, model_name)
                for mae, model_name in sequence_candidates
                if np.isfinite(mae)
            ]
            conventional_candidates = [
                (model_payload_metric(model_payload, "holdout_mae_pct"), model_name)
                for model_name, model_payload in model_payloads.items()
                if model_name
                in {
                    "Ridge",
                    "XGBoost",
                    "Neural Net",
                    "Random Forest",
                    "Gradient Boosting",
                }
                and adaptive_model_evidence_is_valid(
                    model_payload,
                    min_nonoverlapping_samples=min_nonoverlapping_samples,
                )
            ]
            conventional_candidates = [
                (mae, model_name)
                for mae, model_name in conventional_candidates
                if np.isfinite(mae)
            ]
            if not sequence_candidates or not conventional_candidates:
                continue
            seen_evaluations.add(evaluation_key)
            accepted_sessions.setdefault(symbol, []).append(as_of_date)
            report_contributed = True
            valid_runs[symbol] = valid_runs.get(symbol, 0) + 1
            sequence_mae, best_sequence = min(
                sequence_candidates,
                key=lambda item: item[0],
            )
            conventional_mae = min(
                conventional_candidates,
                key=lambda item: item[0],
            )[0]
            if sequence_mae < conventional_mae:
                sequence_wins[symbol] = sequence_wins.get(symbol, 0) + 1
                symbol_family_wins = family_wins.setdefault(symbol, {})
                symbol_family_wins[best_sequence] = (
                    symbol_family_wins.get(best_sequence, 0) + 1
                )
        if report_contributed:
            reports_read += 1
        if reports_read >= report_limit:
            break

    selected: dict[str, str] = {}
    for symbol, wins in sequence_wins.items():
        runs = valid_runs.get(symbol, 0)
        if runs and wins >= min_wins and wins / float(runs) >= min_share:
            winning_family = max(
                family_wins.get(symbol, {}).items(),
                key=lambda item: (item[1], item[0]),
            )[0]
            selected[symbol] = winning_family.lower().replace(" ", "_")
    exploration = deterministic_adaptive_exploration_models(args, selected)
    selected.update(exploration)
    return dict(sorted(selected.items()))


def adaptive_sequence_symbols_from_reports(
    args: argparse.Namespace,
    output_dir: Path,
) -> list[str]:
    """Return symbols selected for adaptive sequence research compute."""

    return list(adaptive_sequence_models_from_reports(args, output_dir))


def model_results_from_snapshot(snapshot: dict) -> dict:
    return {
        model_name: forecast_result_from_dict(model_payload)
        for model_name, model_payload in (snapshot.get("models") or {}).items()
    }


def sequence_model_from_args(args: argparse.Namespace) -> str:
    primary = str(getattr(args, "primary_model", "") or "").strip()
    if primary == "LSTM":
        return "lstm"
    if primary == "Transformer":
        return "transformer"
    return str(getattr(args, "sequence_model", "off") or "off").strip().lower()


def short_sequence_model_from_args(args: argparse.Namespace) -> str:
    short_sequence_model = str(getattr(args, "short_sequence_model", "same") or "same").strip().lower()
    if short_sequence_model in {"off", "lstm", "transformer", "both", "adaptive"}:
        return short_sequence_model
    return sequence_model_from_args(args)


def request_text_from_args(args: argparse.Namespace) -> str:
    return str(getattr(args, "request_text", "") or os.getenv("MARKET_AGENT_REQUEST_TEXT", "") or "")


def no_rl_policy_from_args(args: argparse.Namespace) -> bool:
    request_text = request_text_from_args(args).lower().replace("-", " ")
    negative_phrases = [
        "without rl",
        "with no rl",
        "no rl",
        "not rl",
        "skip rl",
        "exclude rl",
        "without reinforcement",
        "no reinforcement",
        "skip reinforcement",
        "exclude reinforcement",
    ]
    return bool(getattr(args, "no_rl_policy", False) or any(phrase in request_text for phrase in negative_phrases))


def primary_model_from_args(args: argparse.Namespace) -> str:
    primary = str(getattr(args, "primary_model", "") or "Best Validation").strip()
    if primary == "RL Policy":
        return "Best Validation"
    return primary


def primary_model_display_from_args(args: argparse.Namespace) -> str:
    """Return an honest user-facing label for the registered champion rule."""

    primary = primary_model_from_args(args)
    if primary == "Best Validation":
        return "Registered Ensemble Champion"
    return primary


def include_rl_policy_from_args(args: argparse.Namespace) -> bool:
    if no_rl_policy_from_args(args):
        return False
    return bool(getattr(args, "include_rl_policy", True))


def earnings_context_for_symbol(
    symbol: str,
    args: argparse.Namespace,
    *,
    observed_sessions: tuple[dt.date, ...] = (),
) -> dict:
    normalized_symbol = report_symbol(str(symbol).upper())
    if (
        getattr(args, "no_earnings_context", False)
        or str(symbol).endswith("-USD")
        or SYMBOL_SECTORS.get(normalized_symbol) in NON_EARNINGS_SECTORS
    ):
        return {}
    as_of = dt.datetime.now(dt.timezone.utc)
    external_payload, external_error = _load_external_earnings_payload(
        symbol,
        args,
    )
    if external_error:
        return {
            "earnings_available": False,
            "earnings_event_flag": False,
            "earnings_event_score": 0.0,
            "earnings_confidence": 0.0,
            "earnings_policy_eligible": False,
            "earnings_error_code": external_error,
            "earnings_calendar_source": (
                "observed_sessions_plus_us_equity_rules"
            ),
        }
    if external_payload is None:
        fetched = fetch_yfinance_earnings_payload(symbol, as_of=as_of)
        payload = fetched.payload if fetched.available else None
        fetch_error = fetched.error_code
    else:
        payload = external_payload
        fetch_error = None
    if not payload:
        return {
            "earnings_available": False,
            "earnings_event_flag": False,
            "earnings_event_score": 0.0,
            "earnings_confidence": 0.0,
            "earnings_policy_eligible": False,
            "earnings_error_code": (
                fetch_error or "earnings_unavailable"
            ),
            "earnings_calendar_source": (
                "observed_sessions_plus_us_equity_rules"
            ),
        }
    sessions = us_equity_trading_sessions(
        as_of=as_of,
        observed_sessions=observed_sessions,
    )
    signal = interpret_earnings_payload(
        payload,
        as_of=as_of,
        trading_sessions=sessions,
    )
    return {
        "earnings_available": True,
        "earnings_event_flag": bool(signal.event_flag),
        "earnings_event_score": float(signal.event_score),
        "earnings_confidence": float(signal.confidence),
        "earnings_summary": signal.summary,
        "earnings_outcome": signal.outcome,
        "earnings_effective_session": (
            signal.effective_session.isoformat()
            if signal.effective_session
            else ""
        ),
        "earnings_reported_at": (
            signal.reported_at.isoformat()
            if signal.reported_at
            else ""
        ),
        "earnings_policy_eligible": bool(signal.policy_eligible),
        "earnings_age_sessions": signal.age_sessions,
        "earnings_is_stale": bool(signal.is_stale),
        "earnings_blockers": list(signal.blockers),
        "earnings_data_quality_flags": list(signal.data_quality_flags),
        "earnings_calendar_source": (
            "observed_sessions_plus_us_equity_rules"
        ),
        "earnings_error_code": "",
    }


def _load_external_earnings_payload(
    symbol: str,
    args: argparse.Namespace,
) -> tuple[dict | None, str | None]:
    """Load a verified provider payload when the workflow supplied one."""
    result = load_earnings_payload_file(
        symbol,
        getattr(args, "earnings_payload_dir", ""),
    )
    if result.available and result.payload:
        return dict(result.payload), None
    if result.error_code in {
        "external_earnings_payload_not_configured",
        "external_earnings_payload_missing",
    }:
        return None, None
    return None, result.error_code or "external_earnings_payload_invalid"


def cli_flag_present(raw_args: list[str] | None, flag_name: str) -> bool:
    return any(arg == flag_name or arg.startswith(f"{flag_name}=") for arg in raw_args or [])


def apply_run_profile(args: argparse.Namespace, raw_args: list[str] | None = None) -> argparse.Namespace:
    profile = str(getattr(args, "run_profile", "quality") or "quality").strip().lower()
    if profile == "custom":
        return args

    if profile not in {"quick", "scheduled", "overnight", "quality", "research"}:
        profile = "quality"
        args.run_profile = profile

    if not cli_flag_present(raw_args, "--primary-model"):
        args.primary_model = "Best Validation"

    if profile == "quick":
        if not cli_flag_present(raw_args, "--sequence-model"):
            args.sequence_model = "off"
        if not cli_flag_present(raw_args, "--short-horizons"):
            args.short_horizons = ""
        if not cli_flag_present(raw_args, "--short-sequence-model"):
            args.short_sequence_model = "off"
        args.no_optimize = True
        return args

    if profile == "scheduled":
        if not cli_flag_present(raw_args, "--sequence-model"):
            args.sequence_model = "off"
        if not cli_flag_present(raw_args, "--short-horizons"):
            args.short_horizons = ""
        if not cli_flag_present(raw_args, "--short-sequence-model"):
            args.short_sequence_model = "off"
        args.no_optimize = True
        return args

    if profile == "overnight":
        if not cli_flag_present(raw_args, "--history-days"):
            args.history_days = max(int(args.history_days), 1825)
        if not cli_flag_present(raw_args, "--sequence-model"):
            args.sequence_model = "adaptive"
        if not cli_flag_present(raw_args, "--short-horizons"):
            args.short_horizons = ""
        if not cli_flag_present(raw_args, "--short-sequence-model"):
            args.short_sequence_model = "off"
        if not cli_flag_present(raw_args, "--runtime-budget-minutes"):
            args.runtime_budget_minutes = OVERNIGHT_PUBLICATION_DEADLINE_MINUTES
        if not cli_flag_present(
            raw_args,
            "--adaptive-sequence-exploration-quota",
        ):
            args.adaptive_sequence_exploration_quota = (
                DEFAULT_ADAPTIVE_EXPLORATION_QUOTA
            )
        if not (
            cli_flag_present(raw_args, "--include-rl-policy")
            or cli_flag_present(raw_args, "--no-include-rl-policy")
            or cli_flag_present(raw_args, "--no-rl-policy")
        ):
            args.include_rl_policy = False
        return args

    if profile == "quality":
        if not cli_flag_present(raw_args, "--sequence-model"):
            args.sequence_model = "adaptive"
        if not cli_flag_present(raw_args, "--short-horizons"):
            args.short_horizons = "1"
        if not cli_flag_present(raw_args, "--short-sequence-model"):
            args.short_sequence_model = sequence_model_from_args(args) if cli_flag_present(raw_args, "--sequence-model") else "adaptive"
        return args

    if profile == "research":
        if not cli_flag_present(raw_args, "--sequence-model"):
            args.sequence_model = "both"
        if not cli_flag_present(raw_args, "--short-horizons"):
            args.short_horizons = "1"
        if not cli_flag_present(raw_args, "--short-sequence-model"):
            args.short_sequence_model = sequence_model_from_args(args) if cli_flag_present(raw_args, "--sequence-model") else "both"
        return args

    return args


def runtime_deadline_from_args(
    args: argparse.Namespace,
    started_at: float,
) -> float | None:
    try:
        minutes = float(getattr(args, "runtime_budget_minutes", 0.0) or 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime_budget_minutes must be finite and nonnegative") from exc
    if not np.isfinite(minutes) or minutes < 0.0:
        raise ValueError("runtime_budget_minutes must be finite and nonnegative")
    return started_at + minutes * 60.0 if minutes > 0.0 else None


def runtime_deadline_reached(
    deadline_monotonic: float | None,
    *,
    now_monotonic: float | None = None,
) -> bool:
    if deadline_monotonic is None:
        return False
    current = time.perf_counter() if now_monotonic is None else now_monotonic
    return current >= deadline_monotonic


def ranking_deadline_from_publication_deadline(
    publication_deadline_monotonic: float | None,
    workflow_started_at: float,
) -> float | None:
    """Reserve finalization time without consuming tiny smoke budgets."""

    if publication_deadline_monotonic is None:
        return None
    total_budget_seconds = max(
        publication_deadline_monotonic - workflow_started_at,
        0.0,
    )
    reserve_seconds = min(
        PUBLICATION_FINALIZATION_RESERVE_MINUTES * 60.0,
        total_budget_seconds * 0.125,
    )
    return publication_deadline_monotonic - reserve_seconds


def append_runtime_incomplete_error(
    errors: list[str],
    phase: str,
) -> None:
    message = (
        f"runtime publication deadline reached {phase}; partial output is "
        "diagnostic-only and must not be published"
    )
    if message not in errors:
        errors.append(message)


def run_rankings(
    args: argparse.Namespace,
    *,
    deadline_monotonic: float | None = None,
    ranking_deadline_monotonic: float | None = None,
) -> tuple[list[dict], list[str], list[dict], dict]:
    symbols = parse_symbols(args.symbols)
    requested_sequence_model = sequence_model_from_args(args)
    include_rl_policy = include_rl_policy_from_args(args)
    output_dir = Path(args.output_dir)
    started_at = time.perf_counter()
    if deadline_monotonic is None:
        deadline_monotonic = runtime_deadline_from_args(args, started_at)
    if ranking_deadline_monotonic is None:
        ranking_deadline_monotonic = (
            ranking_deadline_from_publication_deadline(
                deadline_monotonic,
                started_at,
            )
        )
    elif deadline_monotonic is not None:
        ranking_deadline_monotonic = min(
            ranking_deadline_monotonic,
            deadline_monotonic,
        )
    adaptive_sequence_models = (
        adaptive_sequence_models_from_reports(args, output_dir)
        if requested_sequence_model == "adaptive"
        else {}
    )
    context_started_at = time.perf_counter()
    context_df = (
        pd.DataFrame()
        if args.no_market_context
        else load_market_context(
            args.history_days,
            args.max_data_lag_sessions,
        )
    )
    context_fingerprint = None if args.no_market_context else frame_fingerprint(context_df)
    context_seconds = time.perf_counter() - context_started_at
    snapshots = []
    patterns_by_symbol = {}
    portfolio_closes: dict[str, pd.Series] = {}
    errors = []
    symbol_timings = []

    runtime_budget_exceeded = False

    for symbol_index, symbol in enumerate(symbols):
        if runtime_deadline_reached(ranking_deadline_monotonic):
            remaining = symbols[symbol_index:]
            runtime_budget_exceeded = True
            append_runtime_incomplete_error(
                errors,
                "at the finalization cutoff before starting "
                f"{len(remaining)} remaining symbols",
            )
            break
        symbol_started_at = time.perf_counter()
        symbol_status = "ok"
        symbol_sequence_model = (
            adaptive_sequence_models.get(symbol.upper(), "off")
            if requested_sequence_model == "adaptive"
            else requested_sequence_model
        )
        try:
            df = get_ohlcv(symbol, args.history_days)
            history_coverage = dict(
                df.attrs.get("history_coverage") or {}
            )
            df = require_fresh_daily_market_data(
                df,
                symbol,
                args.max_data_lag_sessions,
            )
            observed_sessions = tuple(
                pd.Timestamp(value).date() for value in df.index
            )
            earnings_context = earnings_context_for_symbol(
                symbol,
                args,
                observed_sessions=observed_sessions,
            )
            portfolio_closes[report_symbol(symbol)] = clean_close(df)
            data_fingerprint = frame_fingerprint(df)
            symbol_context_fingerprint = context_fingerprint
            if earnings_context:
                event_hash = hashlib.sha256(
                    json.dumps(
                        earnings_context,
                        sort_keys=True,
                        default=_json_default,
                    ).encode("utf-8")
                ).hexdigest()[:16]
                symbol_context_fingerprint = {
                    "market": context_fingerprint,
                    "earnings_event_hash": event_hash,
                    "earnings_effective_session": earnings_context.get(
                        "earnings_effective_session"
                    ),
                }
            model_cache_path = model_result_cache_path(
                output_dir,
                symbol,
                args.history_days,
                args.horizon,
                args.lookback,
                args.ridge_alpha,
                not args.no_optimize,
                not args.no_market_context,
                symbol_sequence_model,
                data_fingerprint,
                symbol_context_fingerprint,
                include_rl_policy,
            )

            if not args.force_retrain:
                cached_payload = load_json_payload(model_cache_path)
                cached_snapshot = cached_payload.get("snapshot") or {}
                if cached_snapshot and cache_payload_fresh(cached_payload, args.model_cache_max_age_days):
                    cached_snapshot = enrich_snapshot_with_smart_policy(
                        cached_snapshot,
                        df,
                        args,
                        earnings_context=earnings_context,
                    )
                    cached_snapshot["history_coverage"] = history_coverage
                    cached_as_of = cached_snapshot.get("as_of_session")
                    if cached_as_of:
                        cached_snapshot["data_cutoff_utc"] = (
                            session_close_utc(
                                str(cached_as_of),
                                session_calendar_for_symbol(symbol),
                            ).isoformat()
                        )
                    snapshots.append(cached_snapshot)
                    patterns_by_symbol[symbol] = cached_payload.get("pattern_info") or {
                        "Primary Pattern": "Unavailable",
                        "All Patterns": "",
                    }
                    cached_payload = dict(cached_payload)
                    cached_payload["snapshot"] = cached_snapshot
                    cached_payload["history_coverage"] = history_coverage
                    save_json_payload(model_cache_path, cached_payload)
                    symbol_status = "cached"
                    if runtime_deadline_reached(
                        ranking_deadline_monotonic
                    ):
                        runtime_budget_exceeded = True
                        append_runtime_incomplete_error(
                            errors,
                            "at the finalization cutoff after completing "
                            f"cached symbol {symbol}",
                        )
                        break
                    continue

            try:
                pattern_info = recognize_patterns(
                    df,
                    short_window=args.pattern_short_window,
                    long_window=args.pattern_long_window,
                )
            except Exception:
                pattern_info = {
                    "Primary Pattern": "Unavailable",
                    "All Patterns": "",
                }
            patterns_by_symbol[symbol] = pattern_info

            model_results = compare_forecast_models(
                df,
                horizon_days=args.horizon,
                lookback_window=args.lookback,
                ridge_alpha=args.ridge_alpha,
                optimize_model=not args.no_optimize,
                context_df=context_df,
                sequence_model=symbol_sequence_model,
                symbol=symbol,
                force_retrain=args.force_retrain,
                include_rl=include_rl_policy,
                earnings_context=earnings_context,
            )
            close = clean_close(df)
            primary_model = select_model_name(model_results, preferred=primary_model_from_args(args))
            if not primary_model:
                raise ValueError("No usable model forecast.")
            smart_policy = build_smart_policy_for_snapshot(
                df,
                model_results,
                primary_model,
                args,
                earnings_context=earnings_context,
            )
            snapshot = snapshot_from_model_results(
                symbol,
                float(close.iloc[-1]),
                model_results,
                smart_policy=smart_policy,
                as_of_session=close.index[-1],
                data_cutoff_utc=session_close_utc(
                    pd.Timestamp(close.index[-1]).date(),
                    session_calendar_for_symbol(symbol),
                ),
            )
            snapshot["history_coverage"] = history_coverage
            snapshots.append(snapshot)

            save_json_payload(
                model_cache_path,
                {
                    "model_result_cache_version": MODEL_RESULT_CACHE_VERSION,
                    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "symbol": symbol,
                    "history_days": args.history_days,
                    "horizon_days": args.horizon,
                    "lookback_window": args.lookback,
                    "ridge_alpha": args.ridge_alpha,
                    "optimize_model": not args.no_optimize,
                    "use_market_context": not args.no_market_context,
                    "sequence_model": symbol_sequence_model,
                    "requested_sequence_model": requested_sequence_model,
                    "include_rl_policy": include_rl_policy,
                    "data_fingerprint": data_fingerprint,
                    "context_fingerprint": symbol_context_fingerprint,
                    "earnings_context": earnings_context,
                    "history_coverage": history_coverage,
                    "pattern_info": pattern_info,
                    "snapshot": snapshot,
                },
            )
        except Exception as exc:
            symbol_status = "error"
            errors.append(f"{symbol}: {exc}")
        finally:
            symbol_timings.append(
                {
                    "symbol": symbol,
                    "seconds": round(time.perf_counter() - symbol_started_at, 3),
                    "status": symbol_status,
                    "sequence_model": symbol_sequence_model,
                }
            )
        if runtime_deadline_reached(ranking_deadline_monotonic):
            runtime_budget_exceeded = True
            append_runtime_incomplete_error(
                errors,
                f"at the finalization cutoff after completing {symbol}",
            )
            break

    rows_frame, row_errors = snapshots_to_ranking_frame(snapshots, primary_model_from_args(args))
    if not rows_frame.empty:
        rows_frame["Primary Pattern"] = rows_frame["Symbol"].map(
            lambda symbol: patterns_by_symbol.get(symbol, {}).get("Primary Pattern", "Unavailable")
        )
        rows_frame["All Patterns"] = rows_frame["Symbol"].map(
            lambda symbol: patterns_by_symbol.get(symbol, {}).get("All Patterns", "")
        )
        rows_frame["Symbol"] = rows_frame["Symbol"].map(report_symbol)
    rows = enrich_rows_with_signal_metadata(rows_frame.to_dict(orient="records"), args)
    rows, portfolio_diagnostics = apply_portfolio_constraints(rows, portfolio_closes, args)
    errors.extend(row_errors)
    if runtime_deadline_reached(ranking_deadline_monotonic):
        runtime_budget_exceeded = True
        append_runtime_incomplete_error(
            errors,
            "at the finalization cutoff during ranking finalization",
        )
    runtime_budget_minutes = float(
        getattr(args, "runtime_budget_minutes", 0.0) or 0.0
    )
    timings = {
        "total_seconds": round(time.perf_counter() - started_at, 3),
        "context_seconds": round(context_seconds, 3),
        "symbols_attempted": len(symbols),
        "symbols_ranked": len(rows),
        "symbols_cached": sum(1 for item in symbol_timings if item["status"] == "cached"),
        "symbols_failed": sum(1 for item in symbol_timings if item["status"] not in {"ok", "cached"}),
        "symbol_timings": symbol_timings,
        "requested_sequence_model": requested_sequence_model,
        "adaptive_sequence_symbols": sorted(adaptive_sequence_models),
        "adaptive_sequence_models": dict(adaptive_sequence_models),
        "run_profile": str(getattr(args, "run_profile", "custom") or "custom"),
        "runtime_budget_minutes": runtime_budget_minutes,
        "ranking_cutoff_reserve_minutes": (
            round(
                max(
                    (deadline_monotonic - ranking_deadline_monotonic) / 60.0,
                    0.0,
                ),
                3,
            )
            if deadline_monotonic is not None
            and ranking_deadline_monotonic is not None
            else 0.0
        ),
        "runtime_budget_exceeded": runtime_budget_exceeded,
        "run_complete": not runtime_budget_exceeded,
        "market_context_history_coverage": dict(
            context_df.attrs.get("history_coverage") or {}
        ),
        "portfolio": portfolio_diagnostics,
    }
    return rows, errors, snapshots, timings


def run_short_horizon_reports(
    args: argparse.Namespace,
    *,
    deadline_monotonic: float | None = None,
    ranking_deadline_monotonic: float | None = None,
) -> list[dict]:
    reports = []
    for horizon in parse_horizons(getattr(args, "short_horizons", ""), getattr(args, "horizon", None)):
        if runtime_deadline_reached(ranking_deadline_monotonic):
            reports.append(
                {
                    "horizon_days": horizon,
                    "sequence_model": short_sequence_model_from_args(args),
                    "rows": [],
                    "errors": [
                        "runtime finalization cutoff reached before short "
                        f"horizon {horizon}; partial output must not be published"
                    ],
                    "snapshots": [],
                    "timings": {
                        "runtime_budget_exceeded": True,
                        "run_complete": False,
                    },
                }
            )
            break
        horizon_args = argparse.Namespace(**vars(args))
        horizon_args.horizon = horizon
        horizon_args.sequence_model = short_sequence_model_from_args(args)
        rows, errors, snapshots, timings = run_rankings(
            horizon_args,
            deadline_monotonic=deadline_monotonic,
            ranking_deadline_monotonic=ranking_deadline_monotonic,
        )
        reports.append(
            {
                "horizon_days": horizon,
                "sequence_model": sequence_model_from_args(horizon_args),
                "rows": rows,
                "errors": errors,
                "snapshots": snapshots,
                "timings": timings,
            }
        )
        if timings.get("run_complete") is not True:
            break
    return reports


def short_horizon_reports_are_complete(
    args: argparse.Namespace,
    reports: list[dict],
) -> bool:
    requested = parse_horizons(
        getattr(args, "short_horizons", ""),
        getattr(args, "horizon", None),
    )
    if len(reports) != len(requested):
        return False
    return all(
        (report.get("timings") or {}).get("run_complete") is True
        for report in reports
    )


def format_row(row: dict) -> str:
    symbol = str(row_value(row, "symbol", "Symbol", default=""))
    forecast_return = float(row_value(row, "forecast_return_pct", "Forecast Return %", default=0.0))
    model_call_text = str(row_value(row, "model_call", "Model Call", default=""))
    selected_model = str(row_value(row, "selected_model", "Selected Model", default=""))
    probability_up = float(row_value(row, "Probability Up %", default=0.0))
    edge = float(row_value(row, "model_edge_pct", "Model Edge %", default=0.0))
    expected_error = float(row_value(row, "expected_error_pct", "Expected Error %", default=0.0))
    primary_pattern = row_value(row, "Primary Pattern", "primary_pattern", default="Unavailable")
    smart_policy = str(row_value(row, "Smart Policy", default="")).strip()
    signal_tier_text = str(row_value(row, "Signal Tier", default="")).strip()
    reliability = str(row_value(row, "Reliability", default="")).strip()
    target_session = str(
        row_value(row, "Target Session", default="")
        or ""
    ).strip()
    policy_score = ranking_score(row)
    try:
        policy_target = float(row_value(row, "Policy Target %", default=np.nan))
    except Exception:
        policy_target = np.nan
    try:
        indicative_target = float(
            row_value(row, "Pre-Portfolio Target %", default=np.nan)
        )
    except Exception:
        indicative_target = np.nan
    allocation_blocked = bool(
        row_value(row, "Portfolio Allocation Blocked", default=False)
    )

    parts = [
        (
            f"{symbol}: {forecast_return:+.2f}% through {target_session}"
            if target_session
            else f"{symbol}: {forecast_return:+.2f}%"
        ),
        model_call_text,
    ]
    if signal_tier_text:
        parts.append(signal_tier_text)
    if reliability:
        parts.append(f"reliability {reliability}")
    if smart_policy:
        policy_text = f"policy {smart_policy} ({policy_score:+.2f}"
        if allocation_blocked and np.isfinite(indicative_target):
            policy_text += (
                f", indicative {indicative_target:.1f}%, "
                f"executable {max(policy_target, 0.0):.1f}%"
            )
        elif np.isfinite(policy_target):
            policy_text += f", target {policy_target:.1f}%"
        policy_text += ")"
        parts.append(policy_text)
    parts.extend(
        [
            f"model {selected_model}",
            f"prob up {probability_up:.1f}%",
            f"edge {edge:.1f}%",
            f"err +/-{expected_error:.2f}%",
        ]
    )
    if primary_pattern and primary_pattern != "Unavailable":
        parts.append(f"pattern {primary_pattern}")
    if bool(row_value(row, "Earnings Event", default=False)):
        earnings_score = float(
            row_value(row, "Earnings Score", default=0.0) or 0.0
        )
        earnings_confidence = float(
            row_value(row, "Earnings Confidence %", default=0.0) or 0.0
        )
        earnings_summary = str(
            row_value(row, "Earnings Summary", default="") or ""
        ).strip()
        earnings_text = (
            f"earnings {earnings_score:+.2f} @ {earnings_confidence:.0f}%"
        )
        if earnings_summary:
            compact_summary = " ".join(earnings_summary.split())
            if len(compact_summary) > 140:
                compact_summary = compact_summary[:137].rstrip() + "..."
            earnings_text += f" ({compact_summary})"
        parts.append(earnings_text)
    return " | ".join(parts)


def format_candidate_row(
    row: dict,
    args: argparse.Namespace,
    *,
    side: str,
) -> str:
    """Format an unqualified raw call with its baseline-relative evidence."""

    nonoverlapping = finite_row_float(
        row,
        "Nonoverlapping Validation Samples",
    )
    mae_skill = finite_row_float(row, "MAE Skill Score")
    direction_skill = finite_row_float(row, "Direction Skill %")
    brier_skill = finite_row_float(row, "Brier Skill Score")

    def metric_text(value: float, *, percent: bool = False) -> str:
        if not np.isfinite(value):
            return "missing"
        return f"{value * 100.0:+.1f}%" if percent else f"{value:.1f}"

    return (
        format_row(row)
        + " | not qualified: failed gates ["
        + ", ".join(
            model_signal_qualification_failures(row, args, side=side)
        )
        + "] | non-overlap n "
        + metric_text(nonoverlapping)
        + ", MAE skill "
        + metric_text(mae_skill, percent=True)
        + ", direction skill "
        + metric_text(direction_skill)
        + " pp"
        + ", Brier skill "
        + metric_text(brier_skill, percent=True)
    )


def build_market_report(
    rows: list[dict],
    errors: list[str],
    args: argparse.Namespace,
    timings: dict | None = None,
    short_horizon_reports: list[dict] | None = None,
) -> dict:
    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sorted_rows = sorted(rows, key=ranking_score, reverse=True)
    policy_buys = [
        row
        for row in sorted_rows
        if is_policy_buy(row) and reliability_grade(row) != "Low"
    ]
    model_buys = cap_signal_rows([row for row in sorted_rows if is_threshold_buy(row, args)], args)
    model_sells = cap_signal_rows(
        sorted(
            [row for row in rows if is_threshold_sell(row, args)],
            key=ranking_score,
        ),
        args,
    )
    watch_buys = cap_signal_rows([row for row in sorted_rows if is_policy_watch_buy(row, args)], args)
    watch_sells = cap_signal_rows(
        sorted(
            [row for row in rows if is_policy_watch_sell(row, args)],
            key=ranking_score,
        ),
        args,
    )
    candidate_limit = max_signal_rows_from_args(args) or 5
    candidate_buys = [
        row for row in sorted_rows if is_unqualified_model_buy(row, args)
    ][:candidate_limit]
    candidate_sells = sorted(
        [row for row in rows if is_unqualified_model_sell(row, args)],
        key=forecast_return_pct,
    )[:candidate_limit]
    threshold_text = f"{min_signal_return_pct_from_args(args):.2f}%"
    max_rows = max_signal_rows_from_args(args)
    cap_text = f"; capped at {max_rows} per side" if max_rows > 0 else ""
    threshold_detail = (
        f"absolute forecast return >= {threshold_text}; model call must be "
        "Buy/Sell or Strong Buy/Strong Sell; selected model must be non-RL; "
        f"OOS validation requires >= "
        f"{RELIABILITY_POLICY_CONFIG.min_validation_samples} samples, >= "
        f"{RELIABILITY_POLICY_CONFIG.min_nonoverlapping_validation_samples} "
        "non-overlapping windows, positive MAE skill versus the zero-return "
        "baseline, positive direction and Brier skill versus training-only "
        "baselines, >= "
        f"{RELIABILITY_POLICY_CONFIG.min_direction_accuracy_pct:.0f}% direction "
        f"hit rate, <= "
        f"{RELIABILITY_POLICY_CONFIG.max_calibration_error_pct:.0f}% calibration "
        f"error, and Brier <= {RELIABILITY_POLICY_CONFIG.max_brier_score:.2f}"
        f"{cap_text}"
    )
    watchlist_detail = f"smart-policy overlay only; model call did not qualify for the primary signal section{cap_text}"
    as_of_sessions = as_of_sessions_from_rows(rows)
    if len(as_of_sessions) == 1:
        data_as_of_text = f"Data as of completed session: {as_of_sessions[0]}"
    elif as_of_sessions:
        data_as_of_text = (
            "Data as-of range across symbols: "
            f"{as_of_sessions[0]} through {as_of_sessions[-1]}"
        )
    else:
        data_as_of_text = "Data as of completed session: unavailable"

    top_buy_symbols = [str(row_value(row, "Symbol", default="")) for row in model_buys if row_value(row, "Symbol", default="")]
    top_sell_symbols = [str(row_value(row, "Symbol", default="")) for row in model_sells if row_value(row, "Symbol", default="")]
    reliability_counts = {}
    tier_counts = {}
    for row in rows:
        reliability_counts[str(row_value(row, "Reliability", default="Unknown"))] = reliability_counts.get(str(row_value(row, "Reliability", default="Unknown")), 0) + 1
        tier_counts[str(row_value(row, "Signal Tier", default="Unknown"))] = tier_counts.get(str(row_value(row, "Signal Tier", default="Unknown")), 0) + 1
    portfolio_state_verified = bool(rows) and all(
        row_value(row, "Portfolio State Verified", default=False) is True
        for row in rows
    )
    portfolio_covariance_verified = bool(rows) and all(
        row_value(
            row,
            "Portfolio Covariance Verified",
            default=False,
        )
        is True
        for row in rows
    )
    portfolio_classification_verified = bool(rows) and all(
        row_value(
            row,
            "Portfolio Classification Verified",
            default=False,
        )
        is True
        for row in rows
    )

    lines = [
        "Market Intelligence Forecast Report",
        f"Generated: {generated_at}",
        data_as_of_text,
        (
            f"Horizon: {args.horizon} asset sessions | Primary: "
            f"{primary_model_display_from_args(args)}"
        ),
        (
            "Asset-session meaning: US-exchange sessions for securities and "
            "UTC calendar-day sessions for crypto"
        ),
        (
            "Horizon meaning: point-to-point target return, not an immediate "
            "or monotonic path forecast; realized adverse/favorable excursion "
            "is tracked after maturity"
        ),
        f"Run profile: {getattr(args, 'run_profile', 'custom')}",
        f"Pattern windows: {args.pattern_short_window}/{args.pattern_long_window} asset sessions",
        f"Sequence model: {sequence_model_from_args(args)}",
        (
            "RL policy: off"
            if no_rl_policy_from_args(args)
            else "RL policy: shadow diagnostics only (excluded from selection, scoring, reliability, and orders)"
        ),
        (
            "Earnings context: off"
            if bool(getattr(args, "no_earnings_context", False))
            else "Earnings context: point-in-time interpretation included once in policy decisions"
        ),
        (
            "General news context: separate point-in-time digest only; not "
            "included in forecast training or scoring until historical "
            "walk-forward validation demonstrates incremental skill"
        ),
        (
            "Smart policy: on | "
            f"per-name cap {float(getattr(args, 'portfolio_max_name_pct', 5.0)):.1f}% | "
            f"cash reserve {float(getattr(args, 'portfolio_cash_reserve_pct', 15.0)):.1f}%"
        ),
        (
            "Portfolio allocation authorization: verified"
            if (
                portfolio_state_verified
                and portfolio_covariance_verified
                and portfolio_classification_verified
            )
            else (
                "Portfolio allocation authorization: blocked; research targets "
                "remain visible but executable targets are zero until current "
                "weights, covariance, and classifications are verified"
            )
        ),
        (
            f"Universe: {len(rows)} ranked / "
            f"{len(parse_symbols(args.symbols))} configured | {MARKET_UNIVERSE}"
        ),
        f"Primary signal rule: {threshold_detail}",
        (
            "Signal counts: "
            f"{len(model_buys)} model-qualified buys, "
            f"{len(model_sells)} model-qualified sells/avoids, "
            f"{len(watch_buys)} policy buy watchlist, "
            f"{len(watch_sells)} policy sell/avoid watchlist, "
            f"{len(candidate_buys)} unqualified buy candidates, "
            f"{len(candidate_sells)} unqualified sell/avoid candidates"
        ),
    ]
    if reliability_counts:
        ordered_reliability = ["High", "Moderate", "Speculative", "Low", "Unknown"]
        reliability_text = ", ".join(
            f"{name}: {reliability_counts[name]}"
            for name in ordered_reliability
            if name in reliability_counts
        )
        if reliability_text:
            lines.append(f"Reliability mix: {reliability_text}")
    if rows:
        portfolio_gross = float(row_value(rows[0], "Portfolio Gross %", default=0.0) or 0.0)
        portfolio_cash = float(row_value(rows[0], "Portfolio Cash %", default=100.0) or 100.0)
        binding = str(row_value(rows[0], "Portfolio Binding Constraints", default="") or "").strip()
        portfolio_line = f"Portfolio targets: gross {portfolio_gross:.1f}% | cash {portfolio_cash:.1f}%"
        if binding:
            portfolio_line += f" | binding {binding}"
        lines.append(portfolio_line)
    adaptive_symbols = (timings or {}).get("adaptive_sequence_symbols") or []
    if adaptive_symbols:
        lines.append("Adaptive sequence symbols: " + ", ".join(report_symbol(symbol) for symbol in adaptive_symbols))
    if timings:
        slowest = sorted(timings.get("symbol_timings", []), key=lambda item: item.get("seconds", 0.0), reverse=True)[:5]
        slowest_text = ", ".join(
            f"{report_symbol(item.get('symbol'))}: {format_duration(item.get('seconds', 0.0))}"
            for item in slowest
        )
        lines.extend(
            [
                (
                    "Calculation time: "
                    f"{format_duration(timings.get('total_seconds', 0.0))} total | "
                    f"context {format_duration(timings.get('context_seconds', 0.0))} | "
                    f"{timings.get('symbols_ranked', len(rows))}/{timings.get('symbols_attempted', len(rows))} ranked | "
                    f"{timings.get('symbols_cached', 0)} cached"
                ),
                f"Slowest symbols: {slowest_text}" if slowest_text else "Slowest symbols: unavailable",
            ]
        )
        if timings.get("runtime_budget_exceeded"):
            lines.append(
                "Run completeness: INCOMPLETE (runtime safety budget reached; "
                "diagnostic output must not be published)"
            )
    lines.append("")

    qualified_research_rows = [*model_buys, *model_sells]
    if policy_buys or qualified_research_rows:
        if policy_buys:
            best_row = policy_buys[0]
            best_heading = "Best Smart Policy Pick"
        else:
            best_row = max(
                qualified_research_rows,
                key=lambda row: abs(forecast_return_pct(row)),
            )
            best_heading = "Highest-Conviction Qualified Research Forecast"
        lines.extend(
            [
                best_heading,
                (
                    f"{row_value(best_row, 'Symbol', default='')}: {row_value(best_row, 'Smart Policy', 'Model Call', default='')} | "
                    f"Policy score: {float(row_value(best_row, 'Policy Score', 'Score', default=0.0)):+.2f} | "
                    f"Pattern: {row_value(best_row, 'Primary Pattern', default='Unavailable')} | "
                    f"Forecast Return: {float(row_value(best_row, 'Forecast Return %', default=0.0)):+.2f}% | "
                    f"Selected model: {row_value(best_row, 'Selected Model', default='')}"
                ),
                "",
            ]
        )

    if short_horizon_reports:
        lines.extend(["Short-Term Forecast Signals"])
        for short_report in short_horizon_reports:
            horizon = int(short_report.get("horizon_days", 0) or 0)
            short_sequence_model = short_report.get("sequence_model") or short_sequence_model_from_args(args)
            short_rows = short_report.get("rows") or []
            short_sorted = sorted(
                short_rows,
                key=ranking_score,
                reverse=True,
            )
            short_buys = cap_signal_rows([row for row in short_sorted if is_threshold_buy(row, args)], args)
            short_sells = cap_signal_rows(
                sorted(
                    [row for row in short_rows if is_threshold_sell(row, args)],
                    key=ranking_score,
                ),
                args,
            )
            short_watch_buys = cap_signal_rows([row for row in short_sorted if is_policy_watch_buy(row, args)], args)
            short_watch_sells = cap_signal_rows(
                sorted(
                    [row for row in short_rows if is_policy_watch_sell(row, args)],
                    key=ranking_score,
                ),
                args,
            )
            horizon_label = f"{horizon} asset session" + (
                "" if horizon == 1 else "s"
            )
            lines.append(f"{horizon_label} | sequence model: {short_sequence_model} | threshold: {threshold_detail}")
            if short_buys:
                lines.append("Model-qualified buys: " + " ; ".join(format_row(row) for row in short_buys))
            else:
                lines.append("Model-qualified buys: no signals met threshold.")
            if short_sells:
                lines.append("Model-qualified sells/avoids: " + " ; ".join(format_row(row) for row in short_sells))
            else:
                lines.append("Model-qualified sells/avoids: no signals met threshold.")
            if short_watch_buys:
                lines.append("Policy buy watchlist: " + " ; ".join(format_row(row) for row in short_watch_buys))
            if short_watch_sells:
                lines.append("Policy sell/avoid watchlist: " + " ; ".join(format_row(row) for row in short_watch_sells))
        lines.append("")

    lines.extend([
        "Model-Qualified Buy Forecasts",
        f"Threshold: {threshold_detail}",
        (
            "Research visibility is separate from execution authorization; "
            "an unverified portfolio always keeps executable allocation at zero."
        ),
    ])
    if model_buys:
        lines.extend(format_row(row) for row in model_buys)
    else:
        lines.append("No model-qualified buy signals met threshold.")

    lines.extend(["", "Model-Qualified Sell / Avoid Forecasts", f"Threshold: {threshold_detail}"])
    if model_sells:
        lines.extend(format_row(row) for row in model_sells)
    else:
        lines.append("No model-qualified sell/avoid signals met threshold.")

    lines.extend(["", "Smart Policy Watchlist", f"Rule: {watchlist_detail}"])
    if watch_buys:
        lines.append("Buy watchlist:")
        lines.extend(format_row(row) for row in watch_buys)
    else:
        lines.append("Buy watchlist: no policy-only buy setups met threshold.")
    if watch_sells:
        lines.append("Sell / avoid watchlist:")
        lines.extend(format_row(row) for row in watch_sells)
    else:
        lines.append("Sell / avoid watchlist: no policy-only sell/avoid setups met threshold.")

    lines.extend(
        [
            "",
            "Unqualified Model Candidates",
            (
                "Raw directional calls shown for research visibility only; "
                "they failed one or more baseline-skill, calibration, or "
                "independent-evidence gates and are not recommendations."
            ),
        ]
    )
    if candidate_buys:
        lines.append("Buy candidates:")
        lines.extend(
            format_candidate_row(row, args, side="buy")
            for row in candidate_buys
        )
    else:
        lines.append("Buy candidates: none.")
    if candidate_sells:
        lines.append("Sell / avoid candidates:")
        lines.extend(
            format_candidate_row(row, args, side="sell")
            for row in candidate_sells
        )
    else:
        lines.append("Sell / avoid candidates: none.")

    # Keep the report compact for LLMs; full details are available in JSON rows.

    if errors:
        lines.extend(["", "Skipped"])
        lines.extend(errors[:10])

    lines.extend(["", "Model output only. Not financial advice."])
    report_text = "\n".join(lines)

    return {
        "report_text": report_text,
        "website_recommendations": report_text,
        "telegram_recommendations": report_text,
        "top_buys": top_buy_symbols,
        "top_sells": top_sell_symbols,
        "policy_watch_buys": [str(row_value(row, "Symbol", default="")) for row in watch_buys if row_value(row, "Symbol", default="")],
        "policy_watch_sells": [str(row_value(row, "Symbol", default="")) for row in watch_sells if row_value(row, "Symbol", default="")],
        "unqualified_candidate_buys": [
            str(row_value(row, "Symbol", default=""))
            for row in candidate_buys
            if row_value(row, "Symbol", default="")
        ],
        "unqualified_candidate_sells": [
            str(row_value(row, "Symbol", default=""))
            for row in candidate_sells
            if row_value(row, "Symbol", default="")
        ],
        "signal_summary": {
            "model_confirmed_buys": len(model_buys),
            "model_confirmed_sells": len(model_sells),
            "model_qualified_buys": len(model_buys),
            "model_qualified_sells": len(model_sells),
            "policy_watch_buys": len(watch_buys),
            "policy_watch_sells": len(watch_sells),
            "unqualified_candidate_buys": len(candidate_buys),
            "unqualified_candidate_sells": len(candidate_sells),
            "reliability_counts": reliability_counts,
            "tier_counts": tier_counts,
        },
    }


def build_telegram_text(
    rows: list[dict],
    errors: list[str],
    args: argparse.Namespace,
    timings: dict | None = None,
    short_horizon_reports: list[dict] | None = None,
) -> str:
    return build_market_report(rows, errors, args, timings, short_horizon_reports)["report_text"]


def append_prediction_records(
    *,
    output_dir: Path,
    rows: list[dict],
    snapshots: list[dict],
    horizon_days: int,
    policy_version: str = "smart-policy-v2",
    created_at_utc: dt.datetime | None = None,
) -> dict:
    """Append the first point-in-time decision for each symbol/session/horizon."""
    ledger = PredictionLedger(output_dir / "prediction_ledger.jsonl")
    snapshots_by_symbol = {
        report_symbol(snapshot.get("symbol", "")): snapshot
        for snapshot in snapshots
    }
    candidate_records: list[PredictionRecord] = []
    skipped: list[str] = []

    def collect_ledger_candidate(
        label: str,
        record_factory: Callable[[], PredictionRecord],
    ) -> None:
        """Collect one validated record without aborting report publication."""

        try:
            candidate_records.append(record_factory())
        except (TypeError, ValueError, RuntimeError) as exc:
            skipped.append(f"{label}: {exc}")

    if (
        created_at_utc is not None
        and (
            created_at_utc.tzinfo is None
            or created_at_utc.utcoffset() is None
        )
    ):
        raise ValueError("created_at_utc must be timezone-aware")
    created_at = (
        created_at_utc.astimezone(dt.timezone.utc)
        if created_at_utc is not None
        else dt.datetime.now(dt.timezone.utc)
    )

    for row in rows:
        symbol = str(row_value(row, "Symbol", default="")).strip().upper()
        snapshot = snapshots_by_symbol.get(symbol)
        raw_symbol = str((snapshot or {}).get("symbol", "")).strip().upper()
        if not symbol or snapshot is None or not raw_symbol:
            continue
        session_calendar = session_calendar_for_symbol(raw_symbol)
        is_crypto = session_calendar == UTC_DAILY_24_7_SESSION_CALENDAR
        ledger_symbol = raw_symbol if is_crypto else symbol
        benchmark_symbol = "BTC-USD" if is_crypto else "SPY"
        as_of_value = snapshot.get("as_of_session") or row_value(
            row,
            "As Of Session",
            default=None,
        )
        target_value = row_value(row, "Target Session", default=None)
        if not as_of_value or not target_value:
            skipped.append(f"{symbol}: missing as-of or target session")
            continue
        try:
            as_of_session = dt.date.fromisoformat(str(as_of_value))
            target_session = dt.date.fromisoformat(str(target_value))
        except ValueError:
            skipped.append(f"{symbol}: invalid as-of or target session")
            continue
        target_maturity = session_close_utc(
            target_session,
            session_calendar,
        )

        selected_model = str(
            row_value(row, "Selected Model", default="")
        ).strip()
        model_payload = (snapshot.get("models") or {}).get(selected_model) or {}
        metrics = model_payload.get("metrics") or {}
        data_cutoff_value = snapshot.get("data_cutoff_utc")
        if not data_cutoff_value:
            skipped.append(f"{symbol}: missing snapshot data_cutoff_utc")
            continue
        try:
            data_cutoff = dt.datetime.fromisoformat(
                str(data_cutoff_value).replace("Z", "+00:00")
            )
            if data_cutoff.tzinfo is None or data_cutoff.utcoffset() is None:
                raise ValueError("timestamp must include a timezone")
            data_cutoff = data_cutoff.astimezone(dt.timezone.utc)
        except (TypeError, ValueError):
            skipped.append(f"{symbol}: malformed snapshot data_cutoff_utc")
            continue
        if data_cutoff > created_at:
            skipped.append(
                f"{symbol}: snapshot data_cutoff_utc is after created_at_utc"
            )
            continue
        if is_crypto:
            data_cutoff = session_close_utc(
                as_of_session,
                session_calendar,
            )
        publication_lag_seconds = max(
            (created_at - data_cutoff).total_seconds(),
            0.0,
        )
        strict_close_t_eligible = (
            not is_crypto
            or publication_lag_seconds
            <= UTC_DAILY_STRICT_CLOSE_T_MAX_LAG_SECONDS
        )
        publication_timing_class = (
            "prospective_equity"
            if not is_crypto
            else (
                "strict_close_t"
                if strict_close_t_eligible
                else "delayed_research_only"
            )
        )
        timing_metadata = {
            "publication_lag_seconds": publication_lag_seconds,
            "strict_close_t_eligible": strict_close_t_eligible,
            "publication_timing_class": publication_timing_class,
            "strict_close_t_max_lag_seconds": (
                UTC_DAILY_STRICT_CLOSE_T_MAX_LAG_SECONDS
                if is_crypto
                else None
            ),
        }

        identity = (
            f"{ledger_symbol}|{as_of_session.isoformat()}|{int(horizon_days)}|"
            f"{policy_version}"
        )
        prediction_id = "pred-" + hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()[:24]
        feature_payload = {
            "as_of_session": as_of_session.isoformat(),
            "selected_model": selected_model,
            "horizon_days": int(horizon_days),
            "metrics": metrics,
        }
        feature_hash = hashlib.sha256(
            json.dumps(
                feature_payload,
                sort_keys=True,
                default=_json_default,
            ).encode("utf-8")
        ).hexdigest()
        forecast_return = (
            float(row_value(row, "Forecast Return %", default=0.0) or 0.0)
            / 100.0
        )
        expected_error = (
            abs(float(row_value(row, "Expected Error %", default=0.0) or 0.0))
            / 100.0
        )
        try:
            record = PredictionRecord(
                prediction_id=prediction_id,
                created_at_utc=created_at,
                data_cutoff_utc=data_cutoff,
                as_of_session=as_of_session,
                target_session=target_session,
                target_maturity_utc=target_maturity,
                session_calendar=session_calendar,
                symbol=ledger_symbol,
                horizon_sessions=int(horizon_days),
                model_name=selected_model,
                model_version=(
                    f"{selected_model.lower().replace(' ', '-')}-"
                    f"cache-v{MODEL_RESULT_CACHE_VERSION}"
                ),
                policy_version=policy_version,
                forecast_return=forecast_return,
                target_weight=max(
                    float(
                        row_value(
                            row,
                            "Policy Target %",
                            default=0.0,
                        )
                        or 0.0
                    )
                    / 100.0,
                    0.0,
                ),
                benchmark_symbol=benchmark_symbol,
                probability_positive=float(
                    row_value(
                        row,
                        "Probability Up %",
                        default=50.0,
                    )
                    or 50.0
                )
                / 100.0,
                lower_bound_return=forecast_return - expected_error,
                upper_bound_return=forecast_return + expected_error,
                feature_set_version=(
                    "ohlcv-market-context-training-postprocessor-v3"
                ),
                feature_hash=feature_hash,
                metadata={
                    "asset_class": "crypto" if is_crypto else "security",
                    "display_symbol": symbol,
                    **timing_metadata,
                    "postprocessor_version": str(
                        metrics.get("postprocessor_version", "")
                    ).strip(),
                    "reliability": row_value(
                        row,
                        "Reliability",
                        default="",
                    ),
                    "signal_tier": row_value(
                        row,
                        "Signal Tier",
                        default="",
                    ),
                    "allocation_eligible": bool(
                        row_value(
                            row,
                            "Policy Allocation Eligible",
                            default=False,
                        )
                    ),
                    "allocation_blockers": row_value(
                        row,
                        "Policy Allocation Blockers",
                        default=[],
                    ),
                    "rl_mode": "shadow",
                },
            )
            candidate_records.append(record)
        except (TypeError, ValueError, RuntimeError) as exc:
            skipped.append(f"{symbol}: {exc}")

        # Persist every available non-RL component as a zero-allocation shadow
        # forecast.  The selected ensemble alone cannot tell us prospectively
        # whether Ridge, XGBoost, MLP, LSTM, or Transformer added real skill.
        # These immutable component records let later promotion use matured
        # point-in-time outcomes instead of repeatedly reusing report holdouts.
        component_policy_version = "component-shadow-v1"
        for component_name, component_payload in (
            snapshot.get("models") or {}
        ).items():
            if component_name == "RL Policy":
                continue
            component_metrics = component_payload.get("metrics") or {}
            component_forecast = component_payload.get("forecast") or []
            if not component_forecast and component_name != selected_model:
                continue
            try:
                component_target = (
                    dt.date.fromisoformat(
                        str(component_forecast[-1]["date"])[:10]
                    )
                    if component_forecast
                    else target_session
                )
                if component_target != target_session:
                    raise ValueError(
                        "component target session does not match selected horizon"
                    )
                component_return = float(
                    component_metrics.get(
                        "forecast_change_pct",
                        forecast_return * 100.0
                        if component_name == selected_model
                        else None,
                    )
                ) / 100.0
                component_probability = float(
                    component_metrics.get(
                        "probability_up_pct",
                        (
                            float(
                                row_value(
                                    row,
                                    "Probability Up %",
                                    default=50.0,
                                )
                                or 50.0
                            )
                            if component_name == selected_model
                            else 50.0
                        ),
                    )
                ) / 100.0
                component_error = abs(
                    float(
                        component_metrics.get(
                            "expected_error_pct",
                            expected_error * 100.0
                            if component_name == selected_model
                            else 0.0,
                        )
                    )
                ) / 100.0
                if not all(
                    np.isfinite(value)
                    for value in (
                        component_return,
                        component_probability,
                        component_error,
                    )
                ):
                    raise ValueError("component forecast metrics are not finite")
            except (KeyError, TypeError, ValueError) as exc:
                skipped.append(f"{symbol} {component_name} shadow: {exc}")
                continue

            component_identity = (
                f"{ledger_symbol}|{as_of_session.isoformat()}|"
                f"{int(horizon_days)}|{component_policy_version}|"
                f"{component_name}"
            )
            component_feature_hash = hashlib.sha256(
                json.dumps(
                    {
                        "as_of_session": as_of_session.isoformat(),
                        "horizon_days": int(horizon_days),
                        "model_name": component_name,
                        "metrics": component_metrics,
                    },
                    sort_keys=True,
                    default=_json_default,
                ).encode("utf-8")
            ).hexdigest()
            collect_ledger_candidate(
                f"{symbol} {component_name} shadow",
                lambda component_name=component_name,
                component_return=component_return,
                component_probability=component_probability,
                component_error=component_error,
                component_feature_hash=component_feature_hash: PredictionRecord(
                    prediction_id="pred-"
                    + hashlib.sha256(
                        component_identity.encode("utf-8")
                    ).hexdigest()[:24],
                    created_at_utc=created_at,
                    data_cutoff_utc=data_cutoff,
                    as_of_session=as_of_session,
                    target_session=target_session,
                    target_maturity_utc=target_maturity,
                    session_calendar=session_calendar,
                    symbol=ledger_symbol,
                    horizon_sessions=int(horizon_days),
                    model_name=component_name,
                    model_version=(
                        f"{component_name.lower().replace(' ', '-')}-"
                        f"cache-v{MODEL_RESULT_CACHE_VERSION}"
                    ),
                    policy_version=component_policy_version,
                    forecast_return=component_return,
                    target_weight=0.0,
                    benchmark_symbol=benchmark_symbol,
                    probability_positive=component_probability,
                    lower_bound_return=component_return - component_error,
                    upper_bound_return=component_return + component_error,
                    feature_set_version="ohlcv-market-context-v3",
                    feature_hash=component_feature_hash,
                    metadata={
                        "asset_class": (
                            "crypto" if is_crypto else "security"
                        ),
                        "display_symbol": symbol,
                        **timing_metadata,
                        "record_role": "component_shadow_forecast",
                        "live_eligible": component_metrics.get(
                            "live_eligible",
                            True,
                        ),
                        "selected_model_at_decision": selected_model,
                        "postprocessor_version": str(
                            component_metrics.get(
                                "postprocessor_version",
                                "",
                            )
                            or ""
                        ).strip(),
                    },
                ),
            )

        rl_payload = (snapshot.get("models") or {}).get("RL Policy") or {}
        rl_metrics = rl_payload.get("metrics") or {}
        if rl_payload and rl_metrics.get("shadow_mode"):
            rl_policy_version = str(
                rl_metrics.get("policy_version", "")
            ).strip()
            rl_model_version = str(
                rl_metrics.get("model_version", "")
            ).strip()
            rl_feature_set_version = str(
                rl_metrics.get("policy_feature_set_version", "")
            ).strip()
            try:
                rl_execution_start = dt.date.fromisoformat(
                    str(
                        rl_metrics[
                            "policy_execution_start_session"
                        ]
                    )
                )
                rl_execution_target = dt.date.fromisoformat(
                    str(
                        rl_metrics[
                            "policy_execution_target_session"
                        ]
                    )
                )
                rl_execution_horizon = int(
                    rl_metrics[
                        "policy_execution_horizon_sessions"
                    ]
                )
            except (KeyError, TypeError, ValueError):
                skipped.append(
                    f"{symbol} RL shadow: missing execution timing metadata"
                )
                continue
            if (
                not rl_policy_version
                or not rl_model_version
                or not rl_feature_set_version
                or rl_metrics.get("rl_live_allocation_enabled") is not False
            ):
                skipped.append(
                    f"{symbol} RL shadow: invalid policy identity or live-mode flag"
                )
                continue
            rl_identity = (
                f"{ledger_symbol}|{as_of_session.isoformat()}|"
                f"{rl_execution_horizon}|"
                f"{rl_policy_version}"
            )
            rl_feature_hash = hashlib.sha256(
                json.dumps(
                    {
                        "as_of_session": as_of_session.isoformat(),
                        "horizon_days": int(horizon_days),
                        "metrics": rl_metrics,
                    },
                    sort_keys=True,
                    default=_json_default,
                ).encode("utf-8")
            ).hexdigest()
            context_horizon = int(
                rl_metrics.get(
                    "forecast_context_horizon_sessions",
                    horizon_days,
                )
            )
            context_return = float(
                rl_metrics.get("forecast_context_return", 0.0)
                or 0.0
            )
            context_probability = float(
                rl_metrics.get(
                    "forecast_context_probability_up",
                    0.5,
                )
                or 0.5
            )
            context_lower_bound = float(
                rl_metrics.get(
                    "forecast_context_lower_bound",
                    context_return,
                )
                or 0.0
            )
            context_uncertainty = max(
                float(
                    rl_metrics.get(
                        "forecast_context_uncertainty",
                        0.0,
                    )
                    or 0.0
                ),
                0.0,
            )
            collect_ledger_candidate(
                f"{symbol} RL forecast context",
                lambda: PredictionRecord(
                    prediction_id="pred-"
                    + hashlib.sha256(
                        (
                            f"{ledger_symbol}|{as_of_session.isoformat()}|"
                            f"{context_horizon}|{rl_policy_version}|"
                            "forecast-context"
                        ).encode("utf-8")
                    ).hexdigest()[:24],
                    created_at_utc=created_at,
                    data_cutoff_utc=data_cutoff,
                    as_of_session=as_of_session,
                    target_session=target_session,
                    target_maturity_utc=target_maturity,
                    session_calendar=session_calendar,
                    symbol=ledger_symbol,
                    horizon_sessions=context_horizon,
                    model_name="RL Forecast Context",
                    model_version=str(
                        rl_metrics.get(
                            "forecast_context_version",
                            "fixed-non-rl-oos-v1",
                        )
                    ),
                    policy_version=rl_policy_version,
                    forecast_return=context_return,
                    target_weight=0.0,
                    benchmark_symbol=benchmark_symbol,
                    probability_positive=context_probability,
                    lower_bound_return=context_lower_bound,
                    upper_bound_return=(
                        context_return
                        + 1.645 * context_uncertainty
                    ),
                    feature_set_version=rl_feature_set_version,
                    feature_hash=rl_feature_hash,
                    metadata={
                        "asset_class": (
                            "crypto" if is_crypto else "security"
                        ),
                        "display_symbol": symbol,
                        **timing_metadata,
                        "postprocessor_version": str(
                            rl_metrics.get(
                                "forecast_context_postprocessor_version",
                                rl_metrics.get("postprocessor_version", ""),
                            )
                        ).strip(),
                        "shadow_mode": True,
                        "live_eligible": False,
                        "record_role": "calibrated_forecast_context",
                        "forecast_context_horizon_sessions": (
                            context_horizon
                        ),
                        "forecast_context_source": rl_metrics.get(
                            "forecast_context_source",
                            "",
                        ),
                        "forecast_context_model_agreement": (
                            rl_metrics.get(
                                "forecast_context_model_agreement"
                            )
                        ),
                        "forecast_context_actual_outcomes_used": False,
                    },
                ),
            )

            collect_ledger_candidate(
                f"{symbol} RL shadow",
                lambda: PredictionRecord(
                    prediction_id="pred-"
                    + hashlib.sha256(
                        rl_identity.encode("utf-8")
                    ).hexdigest()[:24],
                    created_at_utc=created_at,
                    data_cutoff_utc=data_cutoff,
                    as_of_session=as_of_session,
                    return_start_session=rl_execution_start,
                    target_session=rl_execution_target,
                    target_maturity_utc=session_close_utc(
                        rl_execution_target,
                        session_calendar,
                    ),
                    session_calendar=session_calendar,
                    symbol=ledger_symbol,
                    horizon_sessions=rl_execution_horizon,
                    model_name="RL Policy",
                    model_version=rl_model_version,
                    policy_version=rl_policy_version,
                    forecast_return=0.0,
                    target_weight=max(
                        float(
                            rl_metrics.get(
                                "rl_target_weight",
                                0.0,
                            )
                            or 0.0
                        ),
                        0.0,
                    ),
                    benchmark_symbol=benchmark_symbol,
                    probability_positive=None,
                    feature_set_version=rl_feature_set_version,
                    feature_hash=rl_feature_hash,
                    metadata={
                        "asset_class": (
                            "crypto" if is_crypto else "security"
                        ),
                        "display_symbol": symbol,
                        **timing_metadata,
                        "postprocessor_version": str(
                            rl_metrics.get("postprocessor_version", "")
                        ).strip(),
                        "shadow_mode": True,
                        "live_eligible": False,
                        "action": rl_metrics.get("rl_action", "hold"),
                        "action_fraction": rl_metrics.get(
                            "rl_action_fraction",
                            0.0,
                        ),
                        "state_visits": rl_metrics.get(
                            "rl_state_visits",
                            0,
                        ),
                        "abstained": rl_metrics.get(
                            "rl_abstained",
                            True,
                        ),
                        "abstention_reason": rl_metrics.get(
                            "rl_abstention_reason",
                            "",
                        ),
                        "validation_scheme": rl_metrics.get(
                            "validation_scheme",
                            "",
                        ),
                        "policy_execution_start_session": (
                            rl_execution_start.isoformat()
                        ),
                        "policy_execution_target_session": (
                            rl_execution_target.isoformat()
                        ),
                        "policy_execution_horizon_sessions": (
                            rl_execution_horizon
                        ),
                        "policy_decision_refresh_sessions": int(
                            rl_metrics.get(
                                "policy_decision_refresh_sessions",
                                1,
                            )
                        ),
                        "forecast_context_horizon_sessions": (
                            context_horizon
                        ),
                        "forecast_context_source": rl_metrics.get(
                            "forecast_context_source",
                            "",
                        ),
                        "forecast_context_version": rl_metrics.get(
                            "forecast_context_version",
                            "",
                        ),
                        "forecast_context_return": rl_metrics.get(
                            "forecast_context_return"
                        ),
                        "forecast_context_probability_up": (
                            rl_metrics.get(
                                "forecast_context_probability_up"
                            )
                        ),
                        "forecast_context_lower_bound": rl_metrics.get(
                            "forecast_context_lower_bound"
                        ),
                        "forecast_context_model_agreement": (
                            rl_metrics.get(
                                "forecast_context_model_agreement"
                            )
                        ),
                        "forecast_context_uncertainty": rl_metrics.get(
                            "forecast_context_uncertainty"
                        ),
                        "rl_position_state_source": rl_metrics.get(
                            "rl_position_state_source",
                            "",
                        ),
                        "rl_position_state_as_of": rl_metrics.get(
                            "rl_position_state_as_of",
                            "",
                        ),
                        "rl_position_state_auditable": bool(
                            rl_metrics.get(
                                "rl_position_state_auditable",
                                False,
                            )
                        ),
                        "rl_live_allocation_enabled": False,
                    },
                ),
            )

    try:
        batch_result = ledger.append_predictions(candidate_records)
        appended = batch_result.appended_count
        duplicates = batch_result.duplicate_count
    except (TypeError, ValueError, RuntimeError) as exc:
        appended = 0
        duplicates = 0
        skipped.append(f"prediction ledger batch: {exc}")
    return {
        "path": str(ledger.path),
        "horizon_days": int(horizon_days),
        "appended": appended,
        "duplicates": duplicates,
        "skipped": skipped,
    }


def prediction_ledger_summary(ledger: PredictionLedger) -> dict:
    """Summarize matured evidence without ever promoting a policy implicitly."""
    predictions = ledger.predictions()
    outcomes = {item.prediction_id: item for item in ledger.outcomes()}
    groups: dict[
        tuple[str, int, int, str, str, str, str, str, str, bool],
        list[ForecastObservation],
    ] = {}
    path_outcomes: dict[
        tuple[str, int, int, str, str, str, str, str, str, bool],
        list[OutcomeRecord],
    ] = {}
    for prediction in predictions:
        outcome = outcomes.get(prediction.prediction_id)
        if outcome is None:
            continue
        policy_version = prediction.policy_version or prediction.model_version
        context_horizon = int(
            prediction.metadata.get(
                "forecast_context_horizon_sessions",
                prediction.horizon_sessions,
            )
        )
        postprocessor_version = str(
            prediction.metadata.get("postprocessor_version", "")
        ).strip()
        strict_timing = prediction_strict_close_t_eligible(prediction)
        cohort_key = (
            policy_version,
            prediction.horizon_sessions,
            context_horizon,
            prediction.model_name,
            prediction.session_calendar,
            prediction.benchmark_symbol,
            prediction.model_version,
            prediction.feature_set_version or "",
            postprocessor_version,
            strict_timing,
        )
        groups.setdefault(cohort_key, []).append(
            ForecastObservation(
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
                postprocessor_version=postprocessor_version,
                strict_close_t_eligible=strict_timing,
            )
        )
        path_outcomes.setdefault(cohort_key, []).append(outcome)

    metrics = []
    shadow_sessions_by_policy: dict[
        tuple[str, int, int, str, str, str, str, str, bool],
        set[dt.date],
    ] = {}
    for prediction in predictions:
        if not (
            (prediction.policy_version or "").startswith("rl-shadow")
            and prediction.model_name == "RL Policy"
        ):
            continue
        shadow_sessions_by_policy.setdefault(
            (
                prediction.policy_version or "",
                int(
                    prediction.metadata.get(
                        "forecast_context_horizon_sessions",
                        prediction.horizon_sessions,
                    )
                ),
                prediction.horizon_sessions,
                prediction.session_calendar,
                prediction.benchmark_symbol,
                prediction.model_version,
                prediction.feature_set_version or "",
                str(
                    prediction.metadata.get(
                        "postprocessor_version",
                        "",
                    )
                ).strip(),
                prediction_strict_close_t_eligible(prediction),
            ),
            set(),
        )
    for (
        policy_version,
        horizon,
        context_horizon,
        model_name,
        session_calendar,
        benchmark_symbol,
        model_version,
        feature_set_version,
        postprocessor_version,
        strict_timing,
    ), observations in sorted(groups.items()):
        evaluated = evaluate_forecasts(
            observations,
            horizon_sessions=horizon,
        )
        if (
            policy_version.startswith("rl-shadow")
            and model_name == "RL Policy"
        ):
            shadow_sessions_by_policy.setdefault(
                (
                    policy_version,
                    context_horizon,
                    horizon,
                    session_calendar,
                    benchmark_symbol,
                    model_version,
                    feature_set_version,
                    postprocessor_version,
                    strict_timing,
                ),
                set(),
            ).update(
                item.as_of_session for item in observations
            )
        realized_paths = path_outcomes[
            (
                policy_version,
                horizon,
                context_horizon,
                model_name,
                session_calendar,
                benchmark_symbol,
                model_version,
                feature_set_version,
                postprocessor_version,
                strict_timing,
            )
        ]
        adverse_excursions = np.asarray(
            [
                float(item.max_adverse_excursion)
                for item in realized_paths
                if item.max_adverse_excursion is not None
            ],
            dtype=float,
        )
        favorable_excursions = np.asarray(
            [
                float(item.max_favorable_excursion)
                for item in realized_paths
                if item.max_favorable_excursion is not None
            ],
            dtype=float,
        )
        recovered_after_drawdown = [
            item.realized_return > 0.0
            for item in realized_paths
            if item.max_adverse_excursion is not None
            and item.max_adverse_excursion < 0.0
        ]
        metrics.append(
            {
                "policy_version": policy_version,
                "model_name": model_name,
                "model_version": model_version,
                "feature_set_version": feature_set_version,
                "postprocessor_version": postprocessor_version,
                "session_calendar": session_calendar,
                "benchmark_symbol": benchmark_symbol,
                "strict_close_t_eligible": strict_timing,
                "publication_timing_class": (
                    "prospective_equity"
                    if session_calendar == US_EQUITY_SESSION_CALENDAR
                    else (
                        "strict_close_t"
                        if strict_timing
                        else "delayed_research_only"
                    )
                ),
                "promotion_evidence_eligible": strict_timing,
                "horizon_days": horizon,
                "forecast_context_horizon_days": (
                    context_horizon
                    if (
                        policy_version.startswith("rl-shadow")
                        and model_name == "RL Policy"
                    )
                    else horizon
                ),
                "metric_kind": (
                    "daily_policy_execution_outcomes"
                    if model_name == "RL Policy"
                    else "forecast_calibration"
                ),
                "sample_count": evaluated.sample_count,
                "mae_pct": (
                    None
                    if model_name == "RL Policy"
                    else evaluated.mae * 100.0
                ),
                "direction_accuracy_pct": (
                    None
                    if model_name == "RL Policy"
                    else evaluated.direction_accuracy * 100.0
                ),
                "brier_score": (
                    None
                    if model_name == "RL Policy"
                    else evaluated.brier_score
                ),
                "expected_calibration_error": (
                    None
                    if model_name == "RL Policy"
                    else evaluated.expected_calibration_error
                ),
                "average_max_adverse_excursion_pct": float(
                    adverse_excursions.mean() * 100.0
                )
                if adverse_excursions.size
                else None,
                "worst_max_adverse_excursion_pct": float(
                    adverse_excursions.min() * 100.0
                )
                if adverse_excursions.size
                else None,
                "average_max_favorable_excursion_pct": float(
                    favorable_excursions.mean() * 100.0
                )
                if favorable_excursions.size
                else None,
                "positive_at_maturity_after_drawdown_pct": (
                    float(np.mean(recovered_after_drawdown) * 100.0)
                    if recovered_after_drawdown
                    else None
                ),
            }
        )
    promotion_horizons = [
        {
            "policy_version": policy_version,
            "forecast_context_horizon_days": int(
                context_horizon
            ),
            "execution_horizon_days": int(execution_horizon),
            "session_calendar": session_calendar,
            "benchmark_symbol": benchmark_symbol,
            "model_version": model_version,
            "feature_set_version": feature_set_version,
            "postprocessor_version": postprocessor_version,
            "strict_close_t_eligible": strict_timing,
            "publication_timing_class": (
                "prospective_equity"
                if session_calendar == US_EQUITY_SESSION_CALENDAR
                else (
                    "strict_close_t"
                    if strict_timing
                    else "delayed_research_only"
                )
            ),
            "shadow_sessions": len(sessions),
            "minimum_shadow_sessions": 60,
            "eligible_for_gate_evaluation": (
                strict_timing and len(sessions) >= 60
            ),
        }
        for (
            policy_version,
            context_horizon,
            execution_horizon,
            session_calendar,
            benchmark_symbol,
            model_version,
            feature_set_version,
            postprocessor_version,
            strict_timing,
        ), sessions in sorted(shadow_sessions_by_policy.items())
    ]
    return {
        "prediction_count": len(predictions),
        "outcome_count": len(outcomes),
        "metrics": metrics,
        "promotion": {
            "status": "shadow",
            "horizons": promotion_horizons,
            "automatic_promotion": False,
            "reason": (
                "Each horizon is promoted independently and requires matched "
                "purged folds, the strongest "
                "registered non-RL baseline, exact doubled-cost replay, "
                "drawdown/CVaR checks, calibrated probabilities, and a "
                "strict publication-timing cohort."
            ),
        },
    }


def _require_publication_time(
    deadline_monotonic: float | None,
    phase: str,
) -> None:
    if runtime_deadline_reached(deadline_monotonic):
        raise TimeoutError(
            "runtime publication deadline reached "
            f"{phase}; no report or ledger mutation was committed"
        )


@contextlib.contextmanager
def publication_output_lock(
    output_dir: Path,
    *,
    timeout_seconds: float = DEFAULT_PUBLICATION_LOCK_TIMEOUT_SECONDS,
    deadline_monotonic: float | None = None,
):
    """Serialize report/ledger publication with a bounded advisory lock.

    This uses the same ``fcntl.flock`` semantics as ``PredictionLedger``. The
    acquisition is non-blocking with bounded polling so a crashed publisher
    cannot turn lock contention into an unbounded workflow deadlock.
    """

    try:
        timeout = float(timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "publication lock timeout must be finite and nonnegative"
        ) from exc
    if not np.isfinite(timeout) or timeout < 0.0:
        raise ValueError(
            "publication lock timeout must be finite and nonnegative"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    lock_path = output_path / PUBLICATION_LOCK_FILENAME
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - matches ledger fallback
            logging.warning(
                "fcntl unavailable; publication lock is advisory-only no-op"
            )
            yield
            return

        lock_started_at = time.perf_counter()
        while True:
            try:
                fcntl.flock(
                    lock_handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if runtime_deadline_reached(deadline_monotonic):
                    raise TimeoutError(
                        "runtime publication deadline reached while waiting "
                        f"for {lock_path.name}"
                    ) from exc
                waited = time.perf_counter() - lock_started_at
                if waited >= timeout:
                    raise TimeoutError(
                        "timed out waiting for publication lock "
                        f"{lock_path} after {timeout:.3f} seconds"
                    ) from exc
                time.sleep(
                    min(
                        PUBLICATION_LOCK_POLL_SECONDS,
                        max(timeout - waited, 0.0),
                    )
                )
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    finally:
        lock_handle.close()


def _require_complete_publication_inputs(
    args: argparse.Namespace,
    timings: dict,
    short_horizon_reports: list[dict] | None,
) -> None:
    if timings.get("run_complete") is not True:
        raise RuntimeError(
            "publication inputs are missing literal run_complete=true"
        )
    if not short_horizon_reports_are_complete(
        args,
        short_horizon_reports or [],
    ):
        raise RuntimeError(
            "publication inputs do not contain every requested complete "
            "short-horizon run"
        )


def _write_outputs_to_staging(
    rows: list[dict],
    errors: list[str],
    telegram_text: str,
    args: argparse.Namespace,
    snapshots: list[dict],
    timings: dict,
    short_horizon_reports: list[dict] | None = None,
    *,
    deadline_monotonic: float | None = None,
) -> dict:
    _require_complete_publication_inputs(
        args,
        timings,
        short_horizon_reports,
    )
    _require_publication_time(
        deadline_monotonic,
        "before staging output or ledger changes",
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    symbols = parse_symbols(args.symbols)
    sequence_model = sequence_model_from_args(args)
    include_rl_policy = include_rl_policy_from_args(args)
    cache_path = forecast_cache_path(
        output_dir,
        symbols,
        args.history_days,
        args.horizon,
        args.lookback,
        args.ridge_alpha,
        not args.no_optimize,
        not args.no_market_context,
        sequence_model,
        include_rl_policy,
    )

    report = build_market_report(rows, errors, args, timings if args.show_timing else None, short_horizon_reports)
    ledger_path = output_dir / "prediction_ledger.jsonl"

    def deadline_checked_price_loader(symbol: str) -> pd.DataFrame:
        _require_publication_time(
            deadline_monotonic,
            f"before loading outcome prices for {symbol}",
        )
        frame = get_ohlcv(
            symbol,
            max(int(args.history_days), 365),
        )
        _require_publication_time(
            deadline_monotonic,
            f"after loading outcome prices for {symbol}",
        )
        return frame

    maturity_result = append_matured_outcomes(
        PredictionLedger(ledger_path),
        price_loader=deadline_checked_price_loader,
    )
    _require_publication_time(
        deadline_monotonic,
        "after outcome maturity evaluation",
    )
    ledger_runs = [
        append_prediction_records(
            output_dir=output_dir,
            rows=rows,
            snapshots=snapshots,
            horizon_days=args.horizon,
        )
    ]
    for short_report in short_horizon_reports or []:
        _require_publication_time(
            deadline_monotonic,
            "before staging a short-horizon ledger batch",
        )
        ledger_runs.append(
            append_prediction_records(
                output_dir=output_dir,
                rows=short_report.get("rows") or [],
                snapshots=short_report.get("snapshots") or [],
                horizon_days=int(short_report.get("horizon_days", 1) or 1),
            )
        )
    _require_publication_time(
        deadline_monotonic,
        "before staging the ledger summary",
    )
    ledger_summary = prediction_ledger_summary(PredictionLedger(ledger_path))
    workflow_started_at = getattr(
        args,
        "workflow_started_at_monotonic",
        None,
    )
    if isinstance(workflow_started_at, (int, float)) and np.isfinite(
        workflow_started_at
    ):
        timings["workflow_elapsed_seconds"] = round(
            max(time.perf_counter() - float(workflow_started_at), 0.0),
            3,
        )

    payload = {
        "generated_at": timestamp,
        "run_complete": timings.get("run_complete") is True,
        "horizon_days": args.horizon,
        "universe": MARKET_UNIVERSE,
        "universe_count": len(symbols),
        "ranked_count": len(rows),
        "as_of_sessions": as_of_sessions_from_rows(rows),
        "run_profile": str(getattr(args, "run_profile", "custom") or "custom"),
        "primary_model": primary_model_from_args(args),
        "requested_primary_model": args.primary_model,
        "sequence_model": sequence_model,
        "include_rl_policy": include_rl_policy,
        "rl_mode": "off" if no_rl_policy_from_args(args) else "shadow",
        "no_rl_policy": no_rl_policy_from_args(args),
        "short_sequence_model": short_sequence_model_from_args(args),
        "smart_policy": {
            "enabled": True,
            "risk_fraction": float(args.policy_risk_fraction),
            "description": "One selected non-RL forecast + trend + momentum + one fresh earnings event with uncertainty gates and portfolio constraints",
        },
        "earnings_context": {
            "enabled": not bool(
                getattr(args, "no_earnings_context", False)
            ),
            "mode": "point_in_time_fail_closed",
        },
        "portfolio": timings.get("portfolio", {}),
        "prediction_ledger": {
            "mode": "append_only",
            "outcomes": {
                "appended": maturity_result.appended,
                "duplicates": maturity_result.duplicates,
                "pending_not_mature": maturity_result.pending_not_mature,
                "skipped": list(maturity_result.skipped),
            },
            "runs": ledger_runs,
            "summary": ledger_summary,
        },
        "signal_threshold": signal_threshold_metadata(args),
        "model_selection": {
            "run_profile": str(getattr(args, "run_profile", "custom") or "custom"),
            "profile": "adaptive_sequence" if sequence_model == "adaptive" else "standard",
            "champion_rule": (
                "pre_registered_fixed_ensemble_order"
                if primary_model_from_args(args) == "Best Validation"
                else "explicit_model_request"
            ),
            "requested_sequence_model": sequence_model,
            "adaptive_sequence_symbols": timings.get("adaptive_sequence_symbols", []),
            "adaptive_sequence_models": timings.get("adaptive_sequence_models", {}),
            "adaptive_sequence_min_wins": int(getattr(args, "adaptive_sequence_min_wins", 2) or 2),
            "adaptive_sequence_min_share": float(getattr(args, "adaptive_sequence_min_share", 0.50) or 0.50),
            "adaptive_sequence_min_nonoverlap": int(
                getattr(
                    args,
                    "adaptive_sequence_min_nonoverlap",
                    DEFAULT_ADAPTIVE_MIN_NONOVERLAPPING_SAMPLES,
                )
                or DEFAULT_ADAPTIVE_MIN_NONOVERLAPPING_SAMPLES
            ),
            "adaptive_sequence_exploration_quota": int(
                getattr(args, "adaptive_sequence_exploration_quota", 0)
                or 0
            ),
            "market_context_history_coverage": timings.get(
                "market_context_history_coverage",
                {},
            ),
        },
        "symbols": symbols,
        "cache_key": build_cache_key(
            symbols,
            args.history_days,
            args.horizon,
            args.lookback,
            args.ridge_alpha,
            not args.no_optimize,
            not args.no_market_context,
            sequence_model,
            include_rl_policy,
        ),
        "rows": rows,
        "snapshots": snapshots,
        "short_horizon_reports": short_horizon_reports or [],
        "errors": errors,
        "timings": timings,
        "model_cache": {
            "force_retrain": bool(args.force_retrain),
            "max_age_days": float(args.model_cache_max_age_days),
            "include_rl_policy": include_rl_policy,
            "no_rl_policy": no_rl_policy_from_args(args),
        },
        "telegram_text": telegram_text,
        "report_text": report["report_text"],
        "website_recommendations": report["website_recommendations"],
        "telegram_recommendations": report["telegram_recommendations"],
        "top_buys": report["top_buys"],
        "top_sells": report["top_sells"],
        "policy_watch_buys": report["policy_watch_buys"],
        "policy_watch_sells": report["policy_watch_sells"],
        "unqualified_candidate_buys": report[
            "unqualified_candidate_buys"
        ],
        "unqualified_candidate_sells": report[
            "unqualified_candidate_sells"
        ],
        "signal_summary": report["signal_summary"],
    }

    optimization_summary_dir = output_dir / "optimization_summaries"
    optimization_summary_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"ml_forecast_rankings_{timestamp}.json"
    txt_path = optimization_summary_dir / f"ml_forecast_rankings_{timestamp}.txt"
    latest_json_path = output_dir / "ml_forecast_rankings_latest.json"
    latest_txt_path = output_dir / "ml_forecast_rankings_latest.txt"

    json_payload = json_dumps_strict(payload, indent=2)
    _require_publication_time(
        deadline_monotonic,
        "before staging serialized report artifacts",
    )
    write_text_atomic(json_path, json_payload)
    write_text_atomic(latest_json_path, json_payload)
    write_text_atomic(cache_path, json_payload)
    write_text_atomic(txt_path, telegram_text)
    write_text_atomic(latest_txt_path, telegram_text)
    write_text_atomic(
        output_dir / "policy_portfolio_state.json",
        json_dumps_strict(
            {
                "generated_at": timestamp,
                "state_kind": "recommended_targets_not_broker_execution",
                "target_weights": (
                    timings.get("portfolio", {}).get("target_weights", {})
                ),
                "current_weight_source": (
                    timings.get("portfolio", {}).get(
                        "current_weight_source",
                        "unknown",
                    )
                ),
            },
            indent=2,
        ),
    )

    _require_publication_time(
        deadline_monotonic,
        "after staging all report artifacts",
    )

    return {
        "json": str(json_path),
        "txt": str(txt_path),
        "latest_json": str(latest_json_path),
        "latest_txt": str(latest_txt_path),
        "cache": str(cache_path),
    }


def _copy_file_atomic(source: Path, destination: Path) -> None:
    """Copy one staged file into place without exposing a partial file."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        shutil.copyfile(source, temporary_path)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_outputs_with_lock_held(
    rows: list[dict],
    errors: list[str],
    telegram_text: str,
    args: argparse.Namespace,
    snapshots: list[dict],
    timings: dict,
    short_horizon_reports: list[dict] | None = None,
    *,
    deadline_monotonic: float | None = None,
) -> dict:
    """Stage a complete publication, then commit it as one rollback unit."""

    _require_complete_publication_inputs(
        args,
        timings,
        short_horizon_reports,
    )
    _require_publication_time(
        deadline_monotonic,
        "before starting the publication transaction",
    )
    destination_dir = Path(args.output_dir)
    with tempfile.TemporaryDirectory(
        prefix="market-agent-publication-",
    ) as temporary_directory:
        transaction_root = Path(temporary_directory)
        staging_dir = transaction_root / "staged"
        backup_dir = transaction_root / "backup"
        staging_dir.mkdir(parents=True)

        destination_ledger = destination_dir / "prediction_ledger.jsonl"
        if destination_ledger.exists():
            staged_ledger = staging_dir / "prediction_ledger.jsonl"
            staged_ledger.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(destination_ledger, staged_ledger)

        staging_args = argparse.Namespace(**vars(args))
        staging_args.output_dir = str(staging_dir)
        staged_paths = _write_outputs_to_staging(
            rows,
            errors,
            telegram_text,
            staging_args,
            snapshots,
            timings,
            short_horizon_reports,
            deadline_monotonic=deadline_monotonic,
        )
        _require_publication_time(
            deadline_monotonic,
            "before committing the publication transaction",
        )

        staged_files = [
            path
            for path in staging_dir.rglob("*")
            if path.is_file()
        ]
        staged_files.sort(
            key=lambda path: (
                "latest" in path.name,
                path.relative_to(staging_dir).as_posix(),
            )
        )
        existing_destinations: set[Path] = set()
        committed_destinations: list[Path] = []

        try:
            for staged_file in staged_files:
                relative_path = staged_file.relative_to(staging_dir)
                destination = destination_dir / relative_path
                if destination.exists():
                    existing_destinations.add(relative_path)
                    backup_path = backup_dir / relative_path
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(destination, backup_path)
                _copy_file_atomic(staged_file, destination)
                committed_destinations.append(relative_path)

            if runtime_deadline_reached(deadline_monotonic):
                raise TimeoutError(
                    "runtime publication deadline elapsed while committing "
                    "the publication transaction"
                )
        except BaseException:
            for relative_path in reversed(committed_destinations):
                destination = destination_dir / relative_path
                if relative_path in existing_destinations:
                    _copy_file_atomic(
                        backup_dir / relative_path,
                        destination,
                    )
                else:
                    destination.unlink(missing_ok=True)
            raise

        return {
            key: str(
                destination_dir
                / Path(staged_path).relative_to(staging_dir)
            )
            for key, staged_path in staged_paths.items()
        }


def write_outputs(
    rows: list[dict],
    errors: list[str],
    telegram_text: str,
    args: argparse.Namespace,
    snapshots: list[dict],
    timings: dict,
    short_horizon_reports: list[dict] | None = None,
    *,
    deadline_monotonic: float | None = None,
) -> dict:
    """Publish while holding one output-directory transaction lock."""

    _require_complete_publication_inputs(
        args,
        timings,
        short_horizon_reports,
    )
    _require_publication_time(
        deadline_monotonic,
        "before acquiring the publication lock",
    )
    lock_timeout = getattr(
        args,
        "publication_lock_timeout_seconds",
        DEFAULT_PUBLICATION_LOCK_TIMEOUT_SECONDS,
    )
    with publication_output_lock(
        Path(args.output_dir),
        timeout_seconds=lock_timeout,
        deadline_monotonic=deadline_monotonic,
    ):
        return _write_outputs_with_lock_held(
            rows,
            errors,
            telegram_text,
            args,
            snapshots,
            timings,
            short_horizon_reports,
            deadline_monotonic=deadline_monotonic,
        )


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if pd.isna(value):
        return None
    return str(value)


def sanitize_json_value(value):
    if isinstance(value, dict):
        return {str(key): sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [sanitize_json_value(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def json_dumps_strict(payload, **kwargs) -> str:
    return json.dumps(
        sanitize_json_value(payload),
        default=_json_default,
        allow_nan=False,
        **kwargs,
    )


def send_telegram(text: str) -> None:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        raise ValueError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to send Telegram messages.")

    response = requests.post(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=20,
    )
    response.raise_for_status()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate daily ML forecast rankings for Telegram/n8n.")
    parser.add_argument(
        "--run-profile",
        choices=[
            "quick",
            "scheduled",
            "overnight",
            "quality",
            "research",
            "custom",
        ],
        default=os.getenv("MARKET_AGENT_RUN_PROFILE", "quality"),
        help=(
            "quick = fast on-demand forecast; scheduled = bounded daily publication; "
            "overnight = one optimized 30-session adaptive pass with a "
            "210-minute ranking cutoff and 240-minute publication deadline; "
            "quality = optimized adaptive comparison; research = all deep models; "
            "custom = honor raw flags only."
        ),
    )
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument(
        "--history-days",
        type=int,
        default=913,
        help=(
            "Calendar-day OHLCV lookback. A shared cache may retain more "
            "history, but the model receives this deterministic window."
        ),
    )
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--short-horizons", default="", help="Optional comma-separated extra horizons, such as 1.")
    parser.add_argument(
        "--short-sequence-model",
        choices=["same", "off", "lstm", "transformer", "both", "adaptive"],
        default="same",
        help="Sequence model for --short-horizons. Use adaptive to train sequence models only for symbols where prior reports favored them.",
    )
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--pattern-short-window", type=int, default=20)
    parser.add_argument("--pattern-long-window", type=int, default=50)
    parser.add_argument(
        "--primary-model",
        choices=[
            "Ensemble",
            "Best Validation",
            "Ridge",
            "XGBoost",
            "Neural Net",
            "LSTM",
            "Transformer",
            "RL Policy",
        ],
        default="Best Validation",
        help=(
            "Forecast model preference. Best Validation is a compatibility "
            "alias for the pre-registered fixed ensemble champion; it does not "
            "reuse the current outer holdout to choose a winner. RL Policy is "
            "accepted for workflow "
            "compatibility but is always mapped to Best Validation; RL remains "
            "shadow-only."
        ),
    )
    parser.add_argument(
        "--sequence-model",
        choices=["off", "lstm", "transformer", "both", "adaptive"],
        default="off",
    )
    parser.add_argument(
        "--adaptive-sequence-symbols",
        default="",
        help="Optional comma-separated symbols to receive sequence models when --sequence-model adaptive is used.",
    )
    parser.add_argument(
        "--adaptive-sequence-report-limit",
        type=int,
        default=40,
        help="How many recent report JSON files to scan for adaptive sequence-model selection.",
    )
    parser.add_argument(
        "--adaptive-sequence-min-wins",
        type=int,
        default=2,
        help="Minimum prior validation wins required before adaptive mode trains sequence models for a symbol.",
    )
    parser.add_argument(
        "--adaptive-sequence-min-share",
        type=float,
        default=0.50,
        help="Minimum share of prior validation wins required before adaptive mode trains sequence models for a symbol.",
    )
    parser.add_argument(
        "--adaptive-sequence-min-nonoverlap",
        type=int,
        default=DEFAULT_ADAPTIVE_MIN_NONOVERLAPPING_SAMPLES,
        help=(
            "Minimum nonoverlapping outer-holdout samples required for both "
            "a sequence candidate and its conventional comparator."
        ),
    )
    parser.add_argument(
        "--adaptive-sequence-exploration-quota",
        type=int,
        default=0,
        help=(
            "Research-only sequence families to rotate across otherwise "
            "unselected symbols. Overnight defaults to two; use zero to disable."
        ),
    )
    parser.add_argument(
        "--adaptive-sequence-exploration-seed",
        default="",
        help=(
            "Optional deterministic exploration seed. The current UTC date is "
            "used when omitted."
        ),
    )
    parser.add_argument(
        "--include-rl-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate RL shadow diagnostics. RL is excluded from selections, scores, reliability, and orders.",
    )
    parser.add_argument("--no-rl-policy", action="store_true")
    parser.add_argument("--request-text", default="")
    parser.add_argument(
        "--min-signal-return-pct",
        type=float,
        default=2.0,
        help="Minimum absolute forecast return required for buy/sell signals in the text report.",
    )
    parser.add_argument(
        "--max-signal-rows",
        type=int,
        default=0,
        help="Optional cap per side for threshold buy/sell sections. Use 0 for no cap.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="Deprecated alias for --max-signal-rows when --max-signal-rows is unset.",
    )
    parser.add_argument("--policy-risk-fraction", type=float, default=0.05)
    parser.add_argument("--portfolio-cash-reserve-pct", type=float, default=15.0)
    parser.add_argument("--portfolio-max-name-pct", type=float, default=5.0)
    parser.add_argument("--portfolio-max-sector-pct", type=float, default=20.0)
    parser.add_argument("--portfolio-max-cluster-pct", type=float, default=15.0)
    parser.add_argument("--portfolio-max-volatility-pct", type=float, default=15.0)
    parser.add_argument("--portfolio-max-turnover-pct", type=float, default=20.0)
    parser.add_argument(
        "--portfolio-current-weights-json",
        default="",
        help=(
            "JSON object of verified current executed equity weights. Without it, "
            "the report shows research suggestions but authorizes zero allocation."
        ),
    )
    parser.add_argument(
        "--portfolio-drawdown-pct",
        type=float,
        default=0.0,
        help="Current portfolio drawdown magnitude in percent; reaching the breaker moves targets to cash.",
    )
    parser.add_argument("--portfolio-drawdown-breaker-pct", type=float, default=10.0)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "reports"))
    parser.add_argument("--no-market-context", action="store_true")
    parser.add_argument(
        "--no-earnings-context",
        action="store_true",
        help="Disable point-in-time earnings-result interpretation.",
    )
    parser.add_argument(
        "--earnings-payload-dir",
        default="",
        help=(
            "Optional directory containing SYMBOL.json payloads from a verified "
            "earnings provider. Exact/provider-reported timestamps, EPS, revenue, "
            "and guidance can enter policy only after normal eligibility checks."
        ),
    )
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument(
        "--runtime-budget-minutes",
        type=float,
        default=0.0,
        help=(
            "Absolute end-to-end publication deadline shared by the main horizon, "
            "optional short horizons, and finalization. A partial run is marked "
            "incomplete and rejected by the n8n publication guard. The overnight "
            "default is 240 minutes: ranking stops at 210 minutes to reserve "
            "30 minutes for finalization, leaving another 30 minutes before "
            "the external 4.5-hour ceiling. Smaller explicit budgets reserve "
            "12.5% (capped at 30 minutes). Use 0 to disable the guard."
        ),
    )
    parser.add_argument(
        "--publication-lock-timeout-seconds",
        type=float,
        default=DEFAULT_PUBLICATION_LOCK_TIMEOUT_SECONDS,
        help=(
            "Maximum wait for another publisher using the same output "
            "directory. Lock timeout fails closed without staging or commit."
        ),
    )
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--model-cache-max-age-days", type=float, default=7.0)
    parser.add_argument(
        "--max-data-lag-sessions",
        type=int,
        default=1,
        help=(
            "Reject cached daily OHLCV inputs that trail the latest completed "
            "market session by more than this many sessions."
        ),
    )
    parser.add_argument("--show-timing", action="store_true")
    parser.add_argument("--send-telegram", action="store_true")
    parser.add_argument("--json-only", action="store_true")
    return parser


def main() -> int:
    workflow_started_at = time.perf_counter()
    raw_args = sys.argv[1:]
    args = apply_run_profile(build_parser().parse_args(raw_args), raw_args)
    args.workflow_started_at_monotonic = workflow_started_at
    deadline_monotonic = runtime_deadline_from_args(args, workflow_started_at)
    ranking_deadline_monotonic = (
        ranking_deadline_from_publication_deadline(
            deadline_monotonic,
            workflow_started_at,
        )
    )
    rows, errors, snapshots, timings = run_rankings(
        args,
        deadline_monotonic=deadline_monotonic,
        ranking_deadline_monotonic=ranking_deadline_monotonic,
    )
    run_complete = timings.get("run_complete") is True
    short_horizon_reports = (
        run_short_horizon_reports(
            args,
            deadline_monotonic=deadline_monotonic,
            ranking_deadline_monotonic=ranking_deadline_monotonic,
        )
        if run_complete
        else []
    )
    short_runs_complete = short_horizon_reports_are_complete(
        args,
        short_horizon_reports,
    )
    if run_complete and not short_runs_complete:
        run_complete = False
        timings["runtime_budget_exceeded"] = any(
            bool(
                (short_report.get("timings") or {}).get(
                    "runtime_budget_exceeded",
                    False,
                )
            )
            for short_report in short_horizon_reports
        )
        timings["run_complete"] = False
        for short_report in short_horizon_reports:
            if (
                (short_report.get("timings") or {}).get("run_complete")
                is not True
            ):
                horizon = short_report.get("horizon_days", "unknown")
                errors.extend(
                    f"short horizon {horizon}: {message}"
                    for message in short_report.get("errors") or []
                )
    if run_complete and runtime_deadline_reached(deadline_monotonic):
        run_complete = False
        timings["runtime_budget_exceeded"] = True
        timings["run_complete"] = False
        append_runtime_incomplete_error(errors, "before publication finalization")
    timings["workflow_elapsed_seconds"] = round(
        time.perf_counter() - workflow_started_at,
        3,
    )
    timings["short_horizon_runs_complete"] = short_runs_complete
    report_timings = (
        timings if args.show_timing or not run_complete else None
    )
    telegram_text = build_telegram_text(
        rows,
        errors,
        args,
        report_timings,
        short_horizon_reports,
    )
    paths: dict[str, str] = {}
    if run_complete:
        try:
            paths = write_outputs(
                rows,
                errors,
                telegram_text,
                args,
                snapshots,
                timings,
                short_horizon_reports,
                deadline_monotonic=deadline_monotonic,
            )
        except TimeoutError as exc:
            run_complete = False
            timings["runtime_budget_exceeded"] = True
            timings["run_complete"] = False
            errors.append(str(exc))
            paths = {}

    if not run_complete:
        report_timings = timings
        telegram_text = build_telegram_text(
            rows,
            errors,
            args,
            report_timings,
            short_horizon_reports,
        )

    timings["workflow_elapsed_seconds"] = round(
        time.perf_counter() - workflow_started_at,
        3,
    )

    if (
        args.send_telegram
        and run_complete
        and not runtime_deadline_reached(deadline_monotonic)
    ):
        send_telegram(telegram_text)

    if args.json_only:
        report_summary = build_market_report(
            rows,
            errors,
            args,
            report_timings,
            short_horizon_reports,
        )
        configured_symbols = parse_symbols(args.symbols)
        output_payload = {
            "generated_at": dt.datetime.now(dt.timezone.utc).strftime(
                "%Y-%m-%dT%H-%M-%SZ"
            ),
            "run_complete": run_complete,
            "horizon_days": int(args.horizon),
            "universe": MARKET_UNIVERSE,
            "universe_count": len(configured_symbols),
            "ranked_count": len(rows),
            "as_of_sessions": as_of_sessions_from_rows(rows),
            "symbols": configured_symbols,
            "run_profile": str(
                getattr(args, "run_profile", "custom") or "custom"
            ),
            "primary_model": primary_model_from_args(args),
            "sequence_model": sequence_model_from_args(args),
            "short_sequence_model": short_sequence_model_from_args(args),
            "rl_mode": "off" if no_rl_policy_from_args(args) else "shadow",
            "paths": paths,
            "telegram_text": telegram_text,
            "rows": rows,
            "short_horizon_reports": compact_short_horizon_reports(short_horizon_reports, args),
            "top_buys": report_summary["top_buys"],
            "top_sells": report_summary["top_sells"],
            "policy_watch_buys": report_summary["policy_watch_buys"],
            "policy_watch_sells": report_summary["policy_watch_sells"],
            "unqualified_candidate_buys": report_summary[
                "unqualified_candidate_buys"
            ],
            "unqualified_candidate_sells": report_summary[
                "unqualified_candidate_sells"
            ],
            "signal_summary": report_summary["signal_summary"],
            "signal_threshold": signal_threshold_metadata(args),
            "model_selection": {
                "run_profile": str(getattr(args, "run_profile", "custom") or "custom"),
                "requested_sequence_model": sequence_model_from_args(args),
                "champion_rule": (
                    "pre_registered_fixed_ensemble_order"
                    if primary_model_from_args(args) == "Best Validation"
                    else "explicit_model_request"
                ),
                "adaptive_sequence_symbols": timings.get("adaptive_sequence_symbols", []),
                "adaptive_sequence_models": timings.get("adaptive_sequence_models", {}),
                "adaptive_sequence_min_wins": int(getattr(args, "adaptive_sequence_min_wins", 2) or 2),
                "adaptive_sequence_min_share": float(getattr(args, "adaptive_sequence_min_share", 0.50) or 0.50),
                "adaptive_sequence_min_nonoverlap": int(
                    getattr(
                        args,
                        "adaptive_sequence_min_nonoverlap",
                        DEFAULT_ADAPTIVE_MIN_NONOVERLAPPING_SAMPLES,
                    )
                    or DEFAULT_ADAPTIVE_MIN_NONOVERLAPPING_SAMPLES
                ),
                "adaptive_sequence_exploration_quota": int(
                    getattr(args, "adaptive_sequence_exploration_quota", 0)
                    or 0
                ),
                "market_context_history_coverage": timings.get(
                    "market_context_history_coverage",
                    {},
                ),
            },
            "smart_policy": {
                "enabled": True,
                "risk_fraction": float(args.policy_risk_fraction),
            },
            "errors": errors,
        }
        if args.show_timing:
            output_payload["timings"] = timings
        print(json_dumps_strict(output_payload))
    else:
        print(telegram_text)

    return 0 if run_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
