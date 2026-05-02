#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.data import get_ohlcv
from agent.forecast import compare_forecast_models
from forecast_cache import (
    build_cache_key,
    forecast_cache_path,
    select_model_name,
    snapshot_from_model_results,
    snapshots_to_ranking_frame,
)


DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM",
    "XOM", "UNH", "SPY", "VOO", "AVGO", "GLD", "SLV", "USO",
    "ORCA-USD", "PNUT-USD", "DOGE-USD", "SHIB-USD", "FOLKI-USD", "PEPE-USD",
]
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


def load_market_context(history_days: int) -> pd.DataFrame:
    start = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=history_days * 2)).strftime("%Y-%m-%d")
    raw = yf.download(MARKET_CONTEXT_TICKERS, start=start, interval="1d", progress=False)
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


def run_rankings(args: argparse.Namespace) -> tuple[list[dict], list[str], list[dict]]:
    symbols = parse_symbols(args.symbols)
    context_df = pd.DataFrame() if args.no_market_context else load_market_context(args.history_days)
    snapshots = []
    errors = []

    for symbol in symbols:
        try:
            df = get_ohlcv(symbol, args.history_days)
            model_results = compare_forecast_models(
                df,
                horizon_days=args.horizon,
                lookback_window=args.lookback,
                ridge_alpha=args.ridge_alpha,
                optimize_model=not args.no_optimize,
                context_df=context_df,
            )
            close = clean_close(df)
            snapshots.append(snapshot_from_model_results(symbol, float(close.iloc[-1]), model_results))

            primary_model = select_model_name(model_results, preferred=args.primary_model)
            if not primary_model:
                raise ValueError("No usable model forecast.")

            result = model_results[primary_model]
            forecast_price = float(result.forecast["forecast_close"].iloc[-1])
            forecast_return = float(result.metrics.get("forecast_change_pct", 0.0))
            probability_up = float(result.metrics.get("probability_up_pct", 50.0))
            confidence = float(result.metrics.get("confidence_pct", 50.0))
            expected_error = float(result.metrics.get("expected_error_pct", 0.0))
            edge = max(confidence - 50.0, 0.0)
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")

    rows_frame, row_errors = snapshots_to_ranking_frame(snapshots, args.primary_model)
    rows = rows_frame.to_dict(orient="records")
    errors.extend(row_errors)
    return rows, errors, snapshots


def format_row(row: dict) -> str:
    ridge_return = float(row_value(row, "Ridge Return %", default=0.0))
    xgboost_return = float(row_value(row, "XGBoost Return %", default=0.0))
    ensemble_return = float(row_value(row, "Ensemble Return %", default=0.0))
    return (
        f"{row_value(row, 'symbol', 'Symbol')}: {float(row_value(row, 'forecast_return_pct', 'Forecast Return %', default=0.0)):+.2f}% "
        f"({row_value(row, 'model_call', 'Model Call')}, selected {row_value(row, 'selected_model', 'Selected Model')}, "
        f"prob up {float(row_value(row, 'Probability Up %', default=0.0)):.1f}%, edge {float(row_value(row, 'model_edge_pct', 'Model Edge %', default=0.0)):.1f}%, "
        f"err +/-{float(row_value(row, 'expected_error_pct', 'Expected Error %', default=0.0)):.2f}%, "
        f"Ridge {ridge_return:+.2f}%, XGBoost {xgboost_return:+.2f}%, Ensemble {ensemble_return:+.2f}%)"
    )


def build_market_report(rows: list[dict], errors: list[str], args: argparse.Namespace) -> dict:
    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sorted_rows = sorted(rows, key=lambda r: float(row_value(r, "Score", default=0.0)), reverse=True)
    sorted_buy = [row for row in sorted_rows if float(row_value(row, "Forecast Return %", default=0.0)) > 0]
    sorted_sell = [row for row in sorted_rows if float(row_value(row, "Forecast Return %", default=0.0)) < 0]

    top_buy_symbols = [str(row_value(row, "Symbol", default="")) for row in sorted_buy[: args.top_n] if row_value(row, "Symbol", default="")]
    top_sell_symbols = [str(row_value(row, "Symbol", default="")) for row in sorted_sell[: args.top_n] if row_value(row, "Symbol", default="")]

    lines = [
        "ML Forecast Rankings",
        f"Generated: {generated_at}",
        f"Horizon: {args.horizon} trading days | Primary: {args.primary_model}",
        f"Universe: {len(rows)} symbols | Stocks + crypto + commodities",
        "",
        "Strongest Buy Forecasts",
    ]
    if sorted_buy:
        lines.extend(format_row(row) for row in sorted_buy[: args.top_n])
    else:
        lines.append("No positive forecast candidates.")

    lines.extend(["", "Strongest Sell Forecasts"])
    if sorted_sell:
        lines.extend(format_row(row) for row in sorted_sell[: args.top_n])
    else:
        lines.append("No negative forecast candidates.")

    lines.extend(["", "Model Comparison Snapshot"])
    if sorted_rows:
        lines.extend(format_row(row) for row in sorted_rows)
    else:
        lines.append("No ranked rows were generated.")

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


def build_telegram_text(rows: list[dict], errors: list[str], args: argparse.Namespace) -> str:
    return build_market_report(rows, errors, args)["report_text"]


def write_outputs(rows: list[dict], errors: list[str], telegram_text: str, args: argparse.Namespace, snapshots: list[dict]) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    symbols = parse_symbols(args.symbols)
    cache_path = forecast_cache_path(
        output_dir,
        symbols,
        args.history_days,
        args.horizon,
        args.lookback,
        args.ridge_alpha,
        not args.no_optimize,
        not args.no_market_context,
    )

    report = build_market_report(rows, errors, args)

    payload = {
        "generated_at": timestamp,
        "horizon_days": args.horizon,
        "primary_model": args.primary_model,
        "symbols": symbols,
        "cache_key": build_cache_key(
            symbols,
            args.history_days,
            args.horizon,
            args.lookback,
            args.ridge_alpha,
            not args.no_optimize,
            not args.no_market_context,
        ),
        "rows": rows,
        "snapshots": snapshots,
        "errors": errors,
        "telegram_text": telegram_text,
        "report_text": report["report_text"],
        "website_recommendations": report["website_recommendations"],
        "telegram_recommendations": report["telegram_recommendations"],
        "top_buys": report["top_buys"],
        "top_sells": report["top_sells"],
    }

    json_path = output_dir / f"ml_forecast_rankings_{timestamp}.json"
    txt_path = output_dir / f"ml_forecast_rankings_{timestamp}.txt"
    latest_json_path = output_dir / "ml_forecast_rankings_latest.json"
    latest_txt_path = output_dir / "ml_forecast_rankings_latest.txt"

    json_payload = json.dumps(payload, indent=2, default=_json_default)
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
    parser.add_argument("--history-days", type=int, default=365)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--primary-model", choices=["Ensemble", "Best Validation", "Ridge", "XGBoost"], default="Ensemble")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "reports"))
    parser.add_argument("--no-market-context", action="store_true")
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--send-telegram", action="store_true")
    parser.add_argument("--json-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows, errors, snapshots = run_rankings(args)
    telegram_text = build_telegram_text(rows, errors, args)
    paths = write_outputs(rows, errors, telegram_text, args, snapshots)

    if args.send_telegram:
        send_telegram(telegram_text)

    if args.json_only:
        print(json.dumps({"paths": paths, "telegram_text": telegram_text, "rows": rows, "errors": errors}, default=_json_default))
    else:
        print(telegram_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
