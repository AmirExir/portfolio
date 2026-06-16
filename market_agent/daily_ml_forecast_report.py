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
from agent.strategy import sma_crossover
from forecast_cache import (
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
    "AAPL", "MSFT", "NVDA", "AMD", "INTC", "GOOGL", "AMZN", "META", "TSLA", "RIOT", "AVGO", "GEV", "SPCX",
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


def compact_short_horizon_reports(short_horizon_reports: list[dict]) -> list[dict]:
    compact_reports = []
    for short_report in short_horizon_reports or []:
        rows = short_report.get("rows") or []
        sorted_rows = sorted(rows, key=ranking_score, reverse=True)
        buys = [row for row in sorted_rows if is_policy_buy(row)]
        sells = sorted(
            [row for row in rows if is_policy_sell(row)],
            key=ranking_score,
        )
        compact_reports.append(
            {
                "horizon_days": short_report.get("horizon_days"),
                "sequence_model": short_report.get("sequence_model"),
                "top_buys": buys[:10],
                "top_sells": sells[:10],
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps_strict(payload, indent=2))


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
    if short_sequence_model in {"off", "lstm", "transformer", "both"}:
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
    if no_rl_policy_from_args(args) and primary == "RL Policy":
        return "Best Validation"
    return primary


def include_rl_policy_from_args(args: argparse.Namespace) -> bool:
    if no_rl_policy_from_args(args):
        return False
    return True


def run_rankings(args: argparse.Namespace) -> tuple[list[dict], list[str], list[dict], dict]:
    symbols = parse_symbols(args.symbols)
    sequence_model = sequence_model_from_args(args)
    include_rl_policy = include_rl_policy_from_args(args)
    output_dir = Path(args.output_dir)
    started_at = time.perf_counter()
    context_started_at = time.perf_counter()
    context_df = pd.DataFrame() if args.no_market_context else load_market_context(args.history_days)
    context_fingerprint = None if args.no_market_context else frame_fingerprint(context_df)
    context_seconds = time.perf_counter() - context_started_at
    snapshots = []
    patterns_by_symbol = {}
    errors = []
    symbol_timings = []

    for symbol in symbols:
        symbol_started_at = time.perf_counter()
        symbol_status = "ok"
        try:
            df = get_ohlcv(symbol, args.history_days)
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
                sequence_model,
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
                sequence_model=sequence_model,
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
                    "model_result_cache_version": 2,
                    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "symbol": symbol,
                    "history_days": args.history_days,
                    "horizon_days": args.horizon,
                    "lookback_window": args.lookback,
                    "ridge_alpha": args.ridge_alpha,
                    "optimize_model": not args.no_optimize,
                    "use_market_context": not args.no_market_context,
                    "sequence_model": sequence_model,
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
    rows = rows_frame.to_dict(orient="records")
    errors.extend(row_errors)
    timings = {
        "total_seconds": round(time.perf_counter() - started_at, 3),
        "context_seconds": round(context_seconds, 3),
        "symbols_attempted": len(symbols),
        "symbols_ranked": len(rows),
        "symbols_cached": sum(1 for item in symbol_timings if item["status"] == "cached"),
        "symbols_failed": sum(1 for item in symbol_timings if item["status"] not in {"ok", "cached"}),
        "symbol_timings": symbol_timings,
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
    policy_score = ranking_score(row)
    try:
        policy_target = float(row_value(row, "Policy Target %", default=np.nan))
    except Exception:
        policy_target = np.nan

    parts = [
        f"{symbol}: {forecast_return:+.2f}%",
        model_call_text,
    ]
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
    sorted_buy = [row for row in sorted_rows if is_policy_buy(row)]
    sorted_sell = sorted(
        [row for row in rows if is_policy_sell(row)],
        key=ranking_score,
    )

    top_buy_symbols = [str(row_value(row, "Symbol", default="")) for row in sorted_buy[: args.top_n] if row_value(row, "Symbol", default="")]
    top_sell_symbols = [str(row_value(row, "Symbol", default="")) for row in sorted_sell[: args.top_n] if row_value(row, "Symbol", default="")]

    lines = [
        "ML Forecast Rankings",
        f"Generated: {generated_at}",
        f"Horizon: {args.horizon} trading days | Primary: {primary_model_from_args(args)}",
        f"Pattern windows: {args.pattern_short_window}/{args.pattern_long_window} trading days",
        f"Sequence model: {sequence_model_from_args(args)}",
        f"RL policy: {'off (disabled by request)' if no_rl_policy_from_args(args) else ('on' if include_rl_policy_from_args(args) else 'off')}",
        f"Smart policy: on | risk cap {float(getattr(args, 'policy_risk_fraction', 0.10) or 0.10) * 100.0:.1f}% target allocation",
        f"Universe: {len(rows)} symbols | Stocks + crypto + commodities",
    ]
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
        best_row = sorted_buy[0] if sorted_buy else sorted_rows[0]
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
            short_buys = [row for row in short_sorted if is_policy_buy(row)]
            short_sells = sorted(
                [row for row in short_rows if is_policy_sell(row)],
                key=ranking_score,
            )
            horizon_label = f"{horizon} trading day" + ("" if horizon == 1 else "s")
            lines.append(f"{horizon_label} | sequence model: {short_sequence_model}")
            if short_buys:
                lines.append("Buys: " + " ; ".join(format_row(row) for row in short_buys[:3]))
            else:
                lines.append("Buys: no positive forecast candidates.")
            if short_sells:
                lines.append("Sells: " + " ; ".join(format_row(row) for row in short_sells[:3]))
            else:
                lines.append("Sells: no negative forecast candidates.")
        lines.append("")

    lines.extend([
        "Smart Policy Buy Forecasts",
    ])
    if sorted_buy:
        lines.extend(format_row(row) for row in sorted_buy[: args.top_n])
    else:
        lines.append("No positive forecast candidates.")

    lines.extend(["", "Smart Policy Sell / Avoid Forecasts"])
    if sorted_sell:
        lines.extend(format_row(row) for row in sorted_sell[: args.top_n])
    else:
        lines.append("No negative forecast candidates.")

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
        "primary_model": primary_model_from_args(args),
        "requested_primary_model": args.primary_model,
        "sequence_model": sequence_model,
        "include_rl_policy": include_rl_policy,
        "no_rl_policy": no_rl_policy_from_args(args),
        "short_sequence_model": short_sequence_model_from_args(args),
        "smart_policy": {
            "enabled": True,
            "risk_fraction": float(args.policy_risk_fraction),
            "description": "Forecast edge + ensemble + RL policy + trend + momentum + volatility-targeted allocation",
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
    }

    optimization_summary_dir = output_dir / "optimization_summaries"
    optimization_summary_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"ml_forecast_rankings_{timestamp}.json"
    txt_path = optimization_summary_dir / f"ml_forecast_rankings_{timestamp}.txt"
    latest_json_path = output_dir / "ml_forecast_rankings_latest.json"
    latest_txt_path = output_dir / "ml_forecast_rankings_latest.txt"

    json_payload = json_dumps_strict(payload, indent=2)
    json_path.write_text(json_payload)
    latest_json_path.write_text(json_payload)
    cache_path.write_text(json_payload)
    txt_path.write_text(telegram_text)
    latest_txt_path.write_text(telegram_text)

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
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--history-days", type=int, default=913)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--short-horizons", default="", help="Optional comma-separated extra horizons, such as 1.")
    parser.add_argument(
        "--short-sequence-model",
        choices=["same", "off", "lstm", "transformer", "both"],
        default="same",
        help="Sequence model for --short-horizons. Use both to enable LSTM and Transformer on the 1-day report.",
    )
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--pattern-short-window", type=int, default=20)
    parser.add_argument("--pattern-long-window", type=int, default=50)
    parser.add_argument("--primary-model", choices=["Ensemble", "Best Validation", "Ridge", "XGBoost", "Neural Net", "LSTM", "Transformer", "RL Policy"], default="Best Validation")
    parser.add_argument("--sequence-model", choices=["off", "lstm", "transformer", "both"], default="off")
    parser.add_argument("--include-rl-policy", action="store_true", help="Include the warm-start RL policy. Enabled by default.")
    parser.add_argument("--no-rl-policy", action="store_true")
    parser.add_argument("--request-text", default="")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--policy-risk-fraction", type=float, default=0.10)
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
    args = build_parser().parse_args()
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
        output_payload = {
            "paths": paths,
            "telegram_text": telegram_text,
            "rows": rows,
            "short_horizon_reports": compact_short_horizon_reports(short_horizon_reports),
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
