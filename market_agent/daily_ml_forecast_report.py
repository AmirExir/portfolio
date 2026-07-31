#!/usr/bin/env python3
import argparse
import contextlib
import datetime as dt
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.data import get_ohlcv
from agent.forecast import compare_forecast_models
from agent.policy import smart_policy_report
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
MODEL_BUY_CALLS = {"Strong Buy", "Buy"}
MODEL_SELL_CALLS = {"Strong Sell", "Sell"}
POLICY_BUY_CALLS = {"Strong Buy", "Buy"}
POLICY_SELL_TERMS = ("Sell", "Avoid")

SYMBOL_SECTORS = {
    **{symbol: "Technology" for symbol in ("AAPL", "MSFT", "NVDA", "AMD", "INTC", "AVGO", "MU", "WDC", "STX", "SNDK", "ORCL", "DELL")},
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
    **{symbol: "AI Mega Cap" for symbol in ("MSFT", "NVDA", "AMD", "AVGO", "GOOGL", "AMZN", "META", "ORCL", "DELL", "XLK", "QQQ")},
    **{symbol: "Memory and Storage" for symbol in ("MU", "WDC", "STX", "SNDK")},
    **{symbol: "Banks" for symbol in ("JPM", "BAC", "WFC", "C", "GS", "MS")},
    **{symbol: "Airlines" for symbol in ("DAL", "UAL", "AAL", "LUV")},
    **{symbol: "Energy Complex" for symbol in ("XOM", "CVX", "COP", "OXY", "SLB", "EOG", "XLE", "USO")},
    **{symbol: "Broad Index" for symbol in ("SPY", "VOO", "IWM", "DIA")},
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


def load_market_context(history_days: int) -> pd.DataFrame:
    start = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=history_days * 2)).strftime("%Y-%m-%d")
    raw = _yf_download(MARKET_CONTEXT_TICKERS, start=start, interval="1d")
    if raw.empty:
        return pd.DataFrame()

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = close.rename(columns={ticker: f"context_{ticker}" for ticker in close.columns})
    return close.dropna(how="all").ffill()


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


def row_value(row: dict, *keys, default=None):
    for key in keys:
        if key in row:
            return row[key]
    return default


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
    if str(row_value(row, "Reliability", default="")).strip() == "Low":
        return False
    return forecast_return_pct(row) >= min_signal_return_pct_from_args(args) and has_model_buy_call(row)


def is_threshold_sell(row: dict, args: argparse.Namespace) -> bool:
    if str(row_value(row, "Reliability", default="")).strip() == "Low":
        return False
    return forecast_return_pct(row) <= -min_signal_return_pct_from_args(args) and has_model_sell_call(row)


def is_policy_watch_buy(row: dict, args: argparse.Namespace) -> bool:
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
    if is_threshold_sell(row, args):
        return False
    policy = smart_policy_text(row)
    if not any(term in policy for term in POLICY_SELL_TERMS):
        return False
    return ranking_score(row) < 0.0 and forecast_return_pct(row) <= -min_signal_return_pct_from_args(args)


def reliability_grade(row: dict) -> str:
    if str(row_value(row, "selected_model", "Selected Model", default="")).strip() == "RL Policy":
        return "Low"
    if bool(row_value(row, "forecast_outlier", "Forecast Outlier", default=False)):
        return "Low"
    if row_value(row, "validation_is_oos", "Validation Is OOS", default=True) is False:
        return "Low"

    forecast_return = abs(forecast_return_pct(row))
    expected_error = abs(float(row_value(row, "expected_error_pct", "Expected Error %", default=0.0) or 0.0))
    edge = abs(float(row_value(row, "model_edge_pct", "Model Edge %", default=0.0) or 0.0))
    hit_rate = float(row_value(row, "Direction Hit Rate %", default=np.nan))
    validation_mae = abs(float(row_value(row, "Validation MAE %", default=np.nan)))
    error_ratio = forecast_return / expected_error if expected_error > 0 else np.inf

    if np.isfinite(validation_mae) and validation_mae > 0 and forecast_return < validation_mae * 0.5:
        return "Low"
    if np.isfinite(hit_rate) and hit_rate >= 56.0 and edge >= 15.0 and error_ratio >= 0.75:
        return "High"
    if (not np.isfinite(hit_rate) or hit_rate >= 52.0) and edge >= 8.0 and error_ratio >= 0.45:
        return "Moderate"
    if edge >= 3.0:
        return "Speculative"
    return "Low"


def signal_tier(row: dict, args: argparse.Namespace) -> str:
    if is_threshold_buy(row, args):
        return "Model-Confirmed Buy"
    if is_threshold_sell(row, args):
        return "Model-Confirmed Sell/Avoid"
    if is_policy_watch_buy(row, args):
        return "Policy Watch Buy"
    if is_policy_watch_sell(row, args):
        return "Policy Watch Sell/Avoid"
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
        item["Qualified Model Signal"] = item["Signal Tier"].startswith("Model-Confirmed")
        item["Policy Watchlist"] = item["Signal Tier"].startswith("Policy Watch")
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
    selected_symbols = list(active_proposed)
    covariance = _annual_covariance(close_history, selected_symbols)
    current_drawdown = -abs(float(getattr(args, "portfolio_drawdown_pct", 0.0))) / 100.0
    sectors = {symbol: SYMBOL_SECTORS[symbol] for symbol in selected_symbols if symbol in SYMBOL_SECTORS}
    clusters = {symbol: SYMBOL_CLUSTERS[symbol] for symbol in selected_symbols if symbol in SYMBOL_CLUSTERS}

    try:
        allocation = allocate_target_weights(
            active_proposed,
            sectors=sectors,
            correlation_clusters=clusters,
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
            annual_covariance=None,
            current_drawdown=current_drawdown,
            constraints=constraints,
        )

    binding_text = ", ".join(allocation.binding_constraints)
    normalized_rows = []
    for row in rows:
        item = dict(row)
        symbol = str(row_value(item, "Symbol", default="")).upper()
        proposed_pct = max(float(row_value(item, "Policy Target %", default=0.0) or 0.0), 0.0)
        final_pct = float(allocation.target_weights.get(symbol, 0.0) * 100.0)
        item["Pre-Portfolio Target %"] = proposed_pct
        item["Policy Target %"] = final_pct
        item["Portfolio Gross %"] = allocation.gross_exposure * 100.0
        item["Portfolio Cash %"] = allocation.cash_weight * 100.0
        item["Portfolio Binding Constraints"] = binding_text
        item["Portfolio Allocation Blocked"] = bool(proposed_pct > 0.0 and final_pct <= 0.0)
        if final_pct < proposed_pct - 1e-9:
            reason = str(row_value(item, "Policy Reason", default="") or "").strip()
            suffix = f"portfolio constrained to {final_pct:.2f}%"
            item["Policy Reason"] = f"{reason}; {suffix}" if reason else suffix
        if proposed_pct > 0.0 and final_pct <= 0.0 and smart_policy_text(item) in POLICY_BUY_CALLS:
            item["Smart Policy"] = "Hold / Watch"
        item["Risk Overlay"] = smart_policy_text(item) or "Unavailable"
        normalized_rows.append(item)

    diagnostics = {
        "gross_exposure_pct": allocation.gross_exposure * 100.0,
        "cash_weight_pct": allocation.cash_weight * 100.0,
        "turnover_pct": allocation.turnover * 100.0,
        "annualized_volatility_pct": (
            allocation.annualized_volatility * 100.0
            if allocation.annualized_volatility is not None
            else None
        ),
        "sector_exposures_pct": {
            key: value * 100.0 for key, value in allocation.sector_exposures.items()
        },
        "cluster_exposures_pct": {
            key: value * 100.0 for key, value in allocation.cluster_exposures.items()
        },
        "binding_constraints": list(allocation.binding_constraints),
        "warnings": list(allocation.warnings),
        "circuit_breaker_triggered": allocation.circuit_breaker_triggered,
    }
    return normalized_rows, diagnostics


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
) -> dict:
    selected_result = model_results.get(selected_model)
    forecast_metrics = getattr(selected_result, "metrics", {}) if selected_result is not None else {}
    try:
        signal = sma_crossover(df, args.pattern_short_window, args.pattern_long_window)
    except Exception:
        signal = None
    try:
        return smart_policy_report(
            df=df,
            risk_fraction=float(getattr(args, "policy_risk_fraction", 0.10) or 0.10),
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


def enrich_snapshot_with_smart_policy(snapshot: dict, df: pd.DataFrame, args: argparse.Namespace) -> dict:
    model_results = model_results_from_snapshot(snapshot)
    selected_model = select_model_name(model_results, preferred=primary_model_from_args(args))
    if not selected_model:
        return snapshot
    enriched = dict(snapshot)
    enriched["smart_policy"] = build_smart_policy_for_snapshot(df, model_results, selected_model, args)
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


def adaptive_sequence_symbols_from_reports(args: argparse.Namespace, output_dir: Path) -> list[str]:
    override_symbols = parse_symbols(str(getattr(args, "adaptive_sequence_symbols", "") or ""))
    if override_symbols:
        return override_symbols

    try:
        report_limit = max(1, int(getattr(args, "adaptive_sequence_report_limit", 40) or 40))
        min_wins = max(1, int(getattr(args, "adaptive_sequence_min_wins", 5) or 5))
        min_share = max(0.0, float(getattr(args, "adaptive_sequence_min_share", 0.20) or 0.0))
    except Exception:
        report_limit = 40
        min_wins = 5
        min_share = 0.20

    report_paths = sorted(
        output_dir.glob("ml_forecast_rankings_20*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    sequence_wins: dict[str, int] = {}
    valid_runs: dict[str, int] = {}
    reports_read = 0

    for report_path in report_paths:
        payload = load_json_payload(report_path)
        snapshots = payload.get("snapshots") or []
        if not snapshots:
            continue
        reports_read += 1
        for snapshot in snapshots:
            symbol = str(snapshot.get("symbol") or "").upper()
            model_payloads = snapshot.get("models") or {}
            candidates = [
                (model_payload_metric(model_payload, "holdout_mae_pct"), model_name)
                for model_name, model_payload in model_payloads.items()
            ]
            candidates = [
                (mae, model_name)
                for mae, model_name in candidates
                if np.isfinite(mae)
            ]
            if not symbol or not candidates:
                continue
            valid_runs[symbol] = valid_runs.get(symbol, 0) + 1
            best_model = min(candidates, key=lambda item: item[0])[1]
            if best_model in SEQUENCE_MODEL_NAMES:
                sequence_wins[symbol] = sequence_wins.get(symbol, 0) + 1
        if reports_read >= report_limit:
            break

    selected = []
    for symbol, wins in sequence_wins.items():
        runs = valid_runs.get(symbol, 0)
        if runs and wins >= min_wins and wins / float(runs) >= min_share:
            selected.append(symbol)
    return sorted(selected)


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


def include_rl_policy_from_args(args: argparse.Namespace) -> bool:
    if no_rl_policy_from_args(args):
        return False
    return True


def cli_flag_present(raw_args: list[str] | None, flag_name: str) -> bool:
    return any(arg == flag_name or arg.startswith(f"{flag_name}=") for arg in raw_args or [])


def apply_run_profile(args: argparse.Namespace, raw_args: list[str] | None = None) -> argparse.Namespace:
    profile = str(getattr(args, "run_profile", "quality") or "quality").strip().lower()
    if profile == "custom":
        return args

    if profile not in {"quick", "quality", "research"}:
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


def run_rankings(args: argparse.Namespace) -> tuple[list[dict], list[str], list[dict], dict]:
    symbols = parse_symbols(args.symbols)
    requested_sequence_model = sequence_model_from_args(args)
    include_rl_policy = include_rl_policy_from_args(args)
    output_dir = Path(args.output_dir)
    adaptive_sequence_symbols = (
        set(adaptive_sequence_symbols_from_reports(args, output_dir))
        if requested_sequence_model == "adaptive"
        else set()
    )
    started_at = time.perf_counter()
    context_started_at = time.perf_counter()
    context_df = pd.DataFrame() if args.no_market_context else load_market_context(args.history_days)
    context_fingerprint = None if args.no_market_context else frame_fingerprint(context_df)
    context_seconds = time.perf_counter() - context_started_at
    snapshots = []
    patterns_by_symbol = {}
    portfolio_closes: dict[str, pd.Series] = {}
    errors = []
    symbol_timings = []

    for symbol in symbols:
        symbol_started_at = time.perf_counter()
        symbol_status = "ok"
        symbol_sequence_model = (
            "both"
            if requested_sequence_model == "adaptive" and symbol.upper() in adaptive_sequence_symbols
            else ("off" if requested_sequence_model == "adaptive" else requested_sequence_model)
        )
        try:
            df = get_ohlcv(symbol, args.history_days)
            portfolio_closes[report_symbol(symbol)] = clean_close(df)
            data_fingerprint = frame_fingerprint(df)
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
                context_fingerprint,
                include_rl_policy,
            )

            if not args.force_retrain:
                cached_payload = load_json_payload(model_cache_path)
                cached_snapshot = cached_payload.get("snapshot") or {}
                if cached_snapshot and cache_payload_fresh(cached_payload, args.model_cache_max_age_days):
                    cached_snapshot = enrich_snapshot_with_smart_policy(cached_snapshot, df, args)
                    snapshots.append(cached_snapshot)
                    patterns_by_symbol[symbol] = cached_payload.get("pattern_info") or {
                        "Primary Pattern": "Unavailable",
                        "All Patterns": "",
                    }
                    symbol_status = "cached"
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
            )
            close = clean_close(df)
            primary_model = select_model_name(model_results, preferred=primary_model_from_args(args))
            if not primary_model:
                raise ValueError("No usable model forecast.")
            smart_policy = build_smart_policy_for_snapshot(df, model_results, primary_model, args)
            snapshot = snapshot_from_model_results(symbol, float(close.iloc[-1]), model_results, smart_policy=smart_policy)
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
                    "context_fingerprint": context_fingerprint,
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
    timings = {
        "total_seconds": round(time.perf_counter() - started_at, 3),
        "context_seconds": round(context_seconds, 3),
        "symbols_attempted": len(symbols),
        "symbols_ranked": len(rows),
        "symbols_cached": sum(1 for item in symbol_timings if item["status"] == "cached"),
        "symbols_failed": sum(1 for item in symbol_timings if item["status"] not in {"ok", "cached"}),
        "symbol_timings": symbol_timings,
        "requested_sequence_model": requested_sequence_model,
        "adaptive_sequence_symbols": sorted(adaptive_sequence_symbols),
        "run_profile": str(getattr(args, "run_profile", "custom") or "custom"),
        "portfolio": portfolio_diagnostics,
    }
    return rows, errors, snapshots, timings


def run_short_horizon_reports(args: argparse.Namespace) -> list[dict]:
    reports = []
    for horizon in parse_horizons(getattr(args, "short_horizons", ""), getattr(args, "horizon", None)):
        horizon_args = argparse.Namespace(**vars(args))
        horizon_args.horizon = horizon
        horizon_args.sequence_model = short_sequence_model_from_args(args)
        rows, errors, snapshots, timings = run_rankings(horizon_args)
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
    return reports


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
    policy_score = ranking_score(row)
    try:
        policy_target = float(row_value(row, "Policy Target %", default=np.nan))
    except Exception:
        policy_target = np.nan

    parts = [
        f"{symbol}: {forecast_return:+.2f}%",
        model_call_text,
    ]
    if signal_tier_text:
        parts.append(signal_tier_text)
    if reliability:
        parts.append(f"reliability {reliability}")
    if smart_policy:
        policy_text = f"policy {smart_policy} ({policy_score:+.2f}"
        if np.isfinite(policy_target):
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
    return " | ".join(parts)


def build_market_report(
    rows: list[dict],
    errors: list[str],
    args: argparse.Namespace,
    timings: dict | None = None,
    short_horizon_reports: list[dict] | None = None,
) -> dict:
    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sorted_rows = sorted(rows, key=ranking_score, reverse=True)
    policy_buys = [row for row in sorted_rows if is_policy_buy(row)]
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
    threshold_text = f"{min_signal_return_pct_from_args(args):.2f}%"
    max_rows = max_signal_rows_from_args(args)
    cap_text = f"; capped at {max_rows} per side" if max_rows > 0 else ""
    threshold_detail = f"absolute forecast return >= {threshold_text}; model call must be Buy/Sell or Strong Buy/Strong Sell{cap_text}"
    watchlist_detail = f"smart-policy overlay only; model call did not qualify for the primary signal section{cap_text}"

    top_buy_symbols = [str(row_value(row, "Symbol", default="")) for row in model_buys if row_value(row, "Symbol", default="")]
    top_sell_symbols = [str(row_value(row, "Symbol", default="")) for row in model_sells if row_value(row, "Symbol", default="")]
    reliability_counts = {}
    tier_counts = {}
    for row in rows:
        reliability_counts[str(row_value(row, "Reliability", default="Unknown"))] = reliability_counts.get(str(row_value(row, "Reliability", default="Unknown")), 0) + 1
        tier_counts[str(row_value(row, "Signal Tier", default="Unknown"))] = tier_counts.get(str(row_value(row, "Signal Tier", default="Unknown")), 0) + 1

    lines = [
        "Market Intelligence Forecast Report",
        f"Generated: {generated_at}",
        f"Horizon: {args.horizon} trading days | Primary: {primary_model_from_args(args)}",
        f"Run profile: {getattr(args, 'run_profile', 'custom')}",
        f"Pattern windows: {args.pattern_short_window}/{args.pattern_long_window} trading days",
        f"Sequence model: {sequence_model_from_args(args)}",
        (
            "RL policy: off"
            if no_rl_policy_from_args(args)
            else "RL policy: shadow diagnostics only (excluded from selection, scoring, reliability, and orders)"
        ),
        (
            "Smart policy: on | "
            f"per-name cap {float(getattr(args, 'portfolio_max_name_pct', 5.0)):.1f}% | "
            f"cash reserve {float(getattr(args, 'portfolio_cash_reserve_pct', 15.0)):.1f}%"
        ),
        f"Universe: {len(rows)} symbols | Stocks + crypto + commodities",
        f"Primary signal rule: {threshold_detail}",
        (
            "Signal counts: "
            f"{len(model_buys)} model-confirmed buys, "
            f"{len(model_sells)} model-confirmed sells/avoids, "
            f"{len(watch_buys)} policy buy watchlist, "
            f"{len(watch_sells)} policy sell/avoid watchlist"
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
    lines.append("")

    if sorted_rows:
        best_row = policy_buys[0] if policy_buys else sorted_rows[0]
        lines.extend(
            [
                "Best Smart Policy Pick",
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
            horizon_label = f"{horizon} trading day" + ("" if horizon == 1 else "s")
            lines.append(f"{horizon_label} | sequence model: {short_sequence_model} | threshold: {threshold_detail}")
            if short_buys:
                lines.append("Model-confirmed buys: " + " ; ".join(format_row(row) for row in short_buys))
            else:
                lines.append("Model-confirmed buys: no signals met threshold.")
            if short_sells:
                lines.append("Model-confirmed sells/avoids: " + " ; ".join(format_row(row) for row in short_sells))
            else:
                lines.append("Model-confirmed sells/avoids: no signals met threshold.")
            if short_watch_buys:
                lines.append("Policy buy watchlist: " + " ; ".join(format_row(row) for row in short_watch_buys))
            if short_watch_sells:
                lines.append("Policy sell/avoid watchlist: " + " ; ".join(format_row(row) for row in short_watch_sells))
        lines.append("")

    lines.extend([
        "Model-Confirmed Buy Signals",
        f"Threshold: {threshold_detail}",
    ])
    if model_buys:
        lines.extend(format_row(row) for row in model_buys)
    else:
        lines.append("No model-confirmed buy signals met threshold.")

    lines.extend(["", "Model-Confirmed Sell / Avoid Signals", f"Threshold: {threshold_detail}"])
    if model_sells:
        lines.extend(format_row(row) for row in model_sells)
    else:
        lines.append("No model-confirmed sell/avoid signals met threshold.")

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
        "signal_summary": {
            "model_confirmed_buys": len(model_buys),
            "model_confirmed_sells": len(model_sells),
            "policy_watch_buys": len(watch_buys),
            "policy_watch_sells": len(watch_sells),
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


def write_outputs(
    rows: list[dict],
    errors: list[str],
    telegram_text: str,
    args: argparse.Namespace,
    snapshots: list[dict],
    timings: dict,
    short_horizon_reports: list[dict] | None = None,
) -> dict:
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

    payload = {
        "generated_at": timestamp,
        "horizon_days": args.horizon,
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
            "description": "One selected non-RL forecast + trend + momentum with uncertainty gates and portfolio constraints",
        },
        "portfolio": timings.get("portfolio", {}),
        "signal_threshold": {
            "min_forecast_return_pct": min_signal_return_pct_from_args(args),
            "max_rows_per_side": max_signal_rows_from_args(args),
            "directional_call_required": True,
        },
        "model_selection": {
            "run_profile": str(getattr(args, "run_profile", "custom") or "custom"),
            "profile": "adaptive_sequence" if sequence_model == "adaptive" else "standard",
            "requested_sequence_model": sequence_model,
            "adaptive_sequence_symbols": timings.get("adaptive_sequence_symbols", []),
            "adaptive_sequence_min_wins": int(getattr(args, "adaptive_sequence_min_wins", 5) or 5),
            "adaptive_sequence_min_share": float(getattr(args, "adaptive_sequence_min_share", 0.20) or 0.20),
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
        "signal_summary": report["signal_summary"],
    }

    optimization_summary_dir = output_dir / "optimization_summaries"
    optimization_summary_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"ml_forecast_rankings_{timestamp}.json"
    txt_path = optimization_summary_dir / f"ml_forecast_rankings_{timestamp}.txt"
    latest_json_path = output_dir / "ml_forecast_rankings_latest.json"
    latest_txt_path = output_dir / "ml_forecast_rankings_latest.txt"

    json_payload = json_dumps_strict(payload, indent=2)
    write_text_atomic(json_path, json_payload)
    write_text_atomic(latest_json_path, json_payload)
    write_text_atomic(cache_path, json_payload)
    write_text_atomic(txt_path, telegram_text)
    write_text_atomic(latest_txt_path, telegram_text)

    return {
        "json": str(json_path),
        "txt": str(txt_path),
        "latest_json": str(latest_json_path),
        "latest_txt": str(latest_txt_path),
        "cache": str(cache_path),
    }


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
        choices=["quick", "quality", "research", "custom"],
        default=os.getenv("MARKET_AGENT_RUN_PROFILE", "quality"),
        help=(
            "quick = fast on-demand forecast; quality = scheduled prediction profile; "
            "research = all deep models; custom = honor raw flags only."
        ),
    )
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--history-days", type=int, default=913)
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
    parser.add_argument("--primary-model", choices=["Ensemble", "Best Validation", "Ridge", "XGBoost", "Neural Net", "LSTM", "Transformer", "RL Policy"], default="Best Validation")
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
        default=5,
        help="Minimum prior validation wins required before adaptive mode trains sequence models for a symbol.",
    )
    parser.add_argument(
        "--adaptive-sequence-min-share",
        type=float,
        default=0.20,
        help="Minimum share of prior validation wins required before adaptive mode trains sequence models for a symbol.",
    )
    parser.add_argument(
        "--include-rl-policy",
        action="store_true",
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
    parser.add_argument("--policy-risk-fraction", type=float, default=0.10)
    parser.add_argument("--portfolio-cash-reserve-pct", type=float, default=15.0)
    parser.add_argument("--portfolio-max-name-pct", type=float, default=5.0)
    parser.add_argument("--portfolio-max-sector-pct", type=float, default=20.0)
    parser.add_argument("--portfolio-max-cluster-pct", type=float, default=15.0)
    parser.add_argument("--portfolio-max-volatility-pct", type=float, default=15.0)
    parser.add_argument("--portfolio-max-turnover-pct", type=float, default=20.0)
    parser.add_argument(
        "--portfolio-drawdown-pct",
        type=float,
        default=0.0,
        help="Current portfolio drawdown magnitude in percent; reaching the breaker moves targets to cash.",
    )
    parser.add_argument("--portfolio-drawdown-breaker-pct", type=float, default=10.0)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "reports"))
    parser.add_argument("--no-market-context", action="store_true")
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--model-cache-max-age-days", type=float, default=7.0)
    parser.add_argument("--show-timing", action="store_true")
    parser.add_argument("--send-telegram", action="store_true")
    parser.add_argument("--json-only", action="store_true")
    return parser


def main() -> int:
    raw_args = sys.argv[1:]
    args = apply_run_profile(build_parser().parse_args(raw_args), raw_args)
    rows, errors, snapshots, timings = run_rankings(args)
    short_horizon_reports = run_short_horizon_reports(args)
    telegram_text = build_telegram_text(
        rows,
        errors,
        args,
        timings if args.show_timing else None,
        short_horizon_reports,
    )
    paths = write_outputs(rows, errors, telegram_text, args, snapshots, timings, short_horizon_reports)

    if args.send_telegram:
        send_telegram(telegram_text)

    if args.json_only:
        report_summary = build_market_report(rows, errors, args, timings if args.show_timing else None, short_horizon_reports)
        output_payload = {
            "paths": paths,
            "telegram_text": telegram_text,
            "rows": rows,
            "short_horizon_reports": compact_short_horizon_reports(short_horizon_reports, args),
            "top_buys": report_summary["top_buys"],
            "top_sells": report_summary["top_sells"],
            "policy_watch_buys": report_summary["policy_watch_buys"],
            "policy_watch_sells": report_summary["policy_watch_sells"],
            "signal_summary": report_summary["signal_summary"],
            "signal_threshold": {
                "min_forecast_return_pct": min_signal_return_pct_from_args(args),
                "max_rows_per_side": max_signal_rows_from_args(args),
                "directional_call_required": True,
            },
            "model_selection": {
                "run_profile": str(getattr(args, "run_profile", "custom") or "custom"),
                "requested_sequence_model": sequence_model_from_args(args),
                "adaptive_sequence_symbols": timings.get("adaptive_sequence_symbols", []),
                "adaptive_sequence_min_wins": int(getattr(args, "adaptive_sequence_min_wins", 5) or 5),
                "adaptive_sequence_min_share": float(getattr(args, "adaptive_sequence_min_share", 0.20) or 0.20),
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
