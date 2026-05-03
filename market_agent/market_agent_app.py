import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import os
import sys
import requests
import base64
import json

import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go


# Add the parent directory to the path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import agent modules, with fallback handling
try:
    from agent.data import get_ohlcv
    from agent.forecast import backtest_forecasts, compare_forecast_models, forecast_close_prices
    from agent.strategy import sma_crossover
    from agent.backtest import simple_vector_backtest
    from agent.broker import get_account, submit_order, cancel_open_orders
    from agent.risk import target_position_qty
    from forecast_cache import (
        forecast_cache_path,
        forecast_result_from_dict,
        snapshots_to_ranking_frame,
        snapshot_from_model_results,
        select_model_name,
    )
    from patterns import scan_patterns
except ImportError as e:
    st.error(f" Failed to import agent modules: {e}")
    st.info("Please ensure the 'agent' folder exists in the same directory as this app.")
    st.stop()


def get_secret(name, default=None):
    """Read Streamlit secrets when configured, otherwise fall back to env/default."""
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)


def _reports_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "reports")


def _trade_log_path() -> str:
    return os.path.join(_reports_dir(), "paper_trade_log.jsonl")


def _trade_state_path() -> str:
    return os.path.join(_reports_dir(), "paper_trade_state.json")


def _read_trade_state() -> dict:
    path = _trade_state_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _write_trade_state(state: dict) -> None:
    os.makedirs(_reports_dir(), exist_ok=True)
    with open(_trade_state_path(), "w") as handle:
        json.dump(state, handle, indent=2, default=str)


def _append_trade_log(entry: dict) -> None:
    os.makedirs(_reports_dir(), exist_ok=True)
    with open(_trade_log_path(), "a") as handle:
        handle.write(json.dumps(entry, default=str) + "\n")


def load_trade_log(limit: int = 300) -> pd.DataFrame:
    path = _trade_log_path()
    if not os.path.exists(path):
        return pd.DataFrame()

    rows = []
    try:
        with open(path, "r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "timestamp_utc" in df.columns:
        df = df.sort_values("timestamp_utc", ascending=False)
    return df.head(limit)


def _alpaca_position_qty(symbol: str) -> int:
    key = get_secret("ALPACA_KEY")
    secret = get_secret("ALPACA_SECRET")
    endpoint = get_secret("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets").rstrip("/")
    if not key or not secret:
        return 0

    url = f"{endpoint}/v2/positions/{symbol}"
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            payload = response.json()
            return int(float(payload.get("qty", 0)))
        return 0
    except Exception:
        return 0


def _fetch_live_account_snapshot() -> dict:
    try:
        account = get_account()
        if isinstance(account, dict) and "equity" in account:
            return {
                "equity": float(account.get("equity", 0.0)),
                "cash": float(account.get("cash", 0.0)),
                "buying_power": float(account.get("buying_power", 0.0)),
            }
    except Exception:
        pass
    return {}


def _fetch_alpaca_portfolio_history(period: str = "1M", timeframe: str = "1D") -> pd.DataFrame:
    key = get_secret("ALPACA_KEY")
    secret = get_secret("ALPACA_SECRET")
    endpoint = get_secret("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets").rstrip("/")
    if not key or not secret:
        return pd.DataFrame()

    try:
        response = requests.get(
            f"{endpoint}/v2/account/portfolio/history",
            headers={
                "APCA-API-KEY-ID": key,
                "APCA-API-SECRET-KEY": secret,
            },
            params={"period": period, "timeframe": timeframe, "extended_hours": "false"},
            timeout=12,
        )
        response.raise_for_status()
        payload = response.json()
        timestamps = payload.get("timestamp", [])
        equity = payload.get("equity", [])
        if not timestamps or not equity:
            return pd.DataFrame()

        history_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(timestamps, unit="s", utc=True),
                "equity": pd.to_numeric(equity, errors="coerce"),
                "profit_loss": pd.to_numeric(payload.get("profit_loss", []), errors="coerce"),
            }
        ).dropna(subset=["timestamp", "equity"])
        return history_df.sort_values("timestamp")
    except Exception:
        return pd.DataFrame()


def _money_history_from_trade_log(trade_log_df: pd.DataFrame, current_equity: float, current_cash: float) -> pd.DataFrame:
    rows = []
    if not trade_log_df.empty and "timestamp_utc" in trade_log_df.columns:
        for _, row in trade_log_df.iterrows():
            ts = pd.to_datetime(row.get("timestamp_utc"), utc=True, errors="coerce")
            eq = pd.to_numeric(row.get("equity_after"), errors="coerce")
            cs = pd.to_numeric(row.get("cash_after"), errors="coerce")
            if pd.notna(ts) and (pd.notna(eq) or pd.notna(cs)):
                rows.append(
                    {
                        "timestamp": ts,
                        "equity": float(eq) if pd.notna(eq) else np.nan,
                        "cash": float(cs) if pd.notna(cs) else np.nan,
                    }
                )

    rows.append(
        {
            "timestamp": pd.Timestamp.now(tz="UTC"),
            "equity": float(current_equity),
            "cash": float(current_cash),
        }
    )
    money_df = pd.DataFrame(rows).sort_values("timestamp")
    return money_df.drop_duplicates(subset=["timestamp"], keep="last")


def _trade_summary(trade_log_df: pd.DataFrame, symbol: str, current_price: float) -> dict:
    if trade_log_df.empty:
        return {
            "buy_qty": 0,
            "sell_qty": 0,
            "buy_notional": 0.0,
            "sell_notional": 0.0,
            "avg_buy_price": np.nan,
            "avg_sell_price": np.nan,
            "position_qty": 0.0,
            "open_position_value": 0.0,
            "estimated_total_pnl": 0.0,
            "estimated_total_return_pct": 0.0,
        }

    symbol_trades = trade_log_df.copy()
    if "symbol" in symbol_trades.columns:
        symbol_trades = symbol_trades[symbol_trades["symbol"] == symbol]
    if symbol_trades.empty:
        return {
            "buy_qty": 0,
            "sell_qty": 0,
            "buy_notional": 0.0,
            "sell_notional": 0.0,
            "avg_buy_price": np.nan,
            "avg_sell_price": np.nan,
            "position_qty": 0.0,
            "open_position_value": 0.0,
            "estimated_total_pnl": 0.0,
            "estimated_total_return_pct": 0.0,
        }

    if "qty" not in symbol_trades.columns:
        symbol_trades["qty"] = 0
    if "price" not in symbol_trades.columns:
        symbol_trades["price"] = np.nan

    buy_mask = symbol_trades["action"].astype(str).str.upper().eq("BUY")
    sell_mask = symbol_trades["action"].astype(str).str.upper().eq("SELL")

    buy_qty = float(pd.to_numeric(symbol_trades.loc[buy_mask, "qty"], errors="coerce").fillna(0).sum())
    sell_qty = float(pd.to_numeric(symbol_trades.loc[sell_mask, "qty"], errors="coerce").fillna(0).sum())
    buy_notional = float((pd.to_numeric(symbol_trades.loc[buy_mask, "qty"], errors="coerce").fillna(0) * pd.to_numeric(symbol_trades.loc[buy_mask, "price"], errors="coerce").fillna(current_price)).sum())
    sell_notional = float((pd.to_numeric(symbol_trades.loc[sell_mask, "qty"], errors="coerce").fillna(0) * pd.to_numeric(symbol_trades.loc[sell_mask, "price"], errors="coerce").fillna(current_price)).sum())

    avg_buy_price = buy_notional / buy_qty if buy_qty else np.nan
    avg_sell_price = sell_notional / sell_qty if sell_qty else np.nan
    position_qty = buy_qty - sell_qty
    open_position_value = position_qty * float(current_price)
    estimated_total_pnl = sell_notional + open_position_value - buy_notional
    estimated_total_return_pct = (estimated_total_pnl / buy_notional * 100.0) if buy_notional else 0.0

    return {
        "buy_qty": buy_qty,
        "sell_qty": sell_qty,
        "buy_notional": buy_notional,
        "sell_notional": sell_notional,
        "avg_buy_price": avg_buy_price,
        "avg_sell_price": avg_sell_price,
        "position_qty": position_qty,
        "open_position_value": open_position_value,
        "estimated_total_pnl": estimated_total_pnl,
        "estimated_total_return_pct": estimated_total_return_pct,
    }


def maybe_execute_auto_trade(
    symbol: str,
    sig: pd.Series,
    close_prices: pd.Series,
    equity: float,
    risk_fraction: float,
    enabled: bool,
    demo_mode: bool,
) -> dict:
    if not enabled:
        return {"status": "disabled"}
    if len(sig.dropna()) < 2:
        return {"status": "skipped", "reason": "not enough signal history"}

    prev_sig = int(sig.iloc[-2])
    last_sig = int(sig.iloc[-1])
    if prev_sig == last_sig:
        return {"status": "skipped", "reason": "no signal flip"}

    signal_ts = pd.to_datetime(sig.index[-1]).strftime("%Y-%m-%d")
    action_key = f"{signal_ts}:{prev_sig}->{last_sig}"
    state = _read_trade_state()
    symbol_state = state.get(symbol, {})
    if symbol_state.get("last_action_key") == action_key:
        return {"status": "skipped", "reason": "already executed for this signal"}

    last_price = float(close_prices.iloc[-1])
    suggested_qty = max(target_position_qty(float(equity), last_price, risk_fraction), 1)

    if prev_sig == 0 and last_sig == 1:
        side = "buy"
        action = "BUY"
        qty = suggested_qty
    elif prev_sig == 1 and last_sig == 0:
        side = "sell"
        action = "SELL"
        qty = _alpaca_position_qty(symbol) if not demo_mode else suggested_qty
        qty = max(int(qty), 1)
    else:
        return {"status": "skipped", "reason": "unsupported signal transition"}

    if demo_mode:
        order_result = {"demo": True, "symbol": symbol, "side": side, "qty": qty}
    else:
        order_result = submit_order(symbol, qty, side)

    account_snapshot = {
        "equity": float(equity),
        "cash": None,
        "buying_power": None,
    }
    if not demo_mode:
        live_snapshot = _fetch_live_account_snapshot()
        if live_snapshot:
            account_snapshot = live_snapshot

    log_entry = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "source": "auto",
        "symbol": symbol,
        "action": action,
        "side": side,
        "qty": int(qty),
        "price": round(last_price, 4),
        "prev_signal": prev_sig,
        "last_signal": last_sig,
        "risk_fraction": float(risk_fraction),
        "demo_mode": bool(demo_mode),
        "equity_after": account_snapshot.get("equity"),
        "cash_after": account_snapshot.get("cash"),
        "buying_power_after": account_snapshot.get("buying_power"),
        "order_result": order_result,
    }
    _append_trade_log(log_entry)

    state[symbol] = {
        "last_action_key": action_key,
        "last_signal": last_sig,
        "last_timestamp": log_entry["timestamp_utc"],
    }
    _write_trade_state(state)

    return {"status": "executed", "entry": log_entry}


def maybe_execute_auto_trade_ml(
    symbol: str,
    forecast_change_pct: float,
    equity: float,
    risk_fraction: float,
    enabled: bool,
    demo_mode: bool,
) -> dict:
    """Execute an auto trade based on the ML forecast direction (buy if positive, sell if negative).

    Uses the same duplicate-protection state file but stores a separate ML action key.
    """
    if not enabled:
        return {"status": "disabled"}

    state = _read_trade_state()
    signal_ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    direction = 1 if float(forecast_change_pct) > 0 else 0
    action_key = f"ml:{signal_ts}:{direction}"
    symbol_state = state.get(symbol, {})
    if symbol_state.get("last_ml_action_key") == action_key:
        return {"status": "skipped", "reason": "already executed ML action for today"}

    last_price = None
    try:
        df = load_ohlcv(symbol, 5)
        if not df.empty:
            last_close = df["close"]
            if isinstance(last_close, pd.DataFrame):
                last_close = last_close.iloc[:, 0]
            last_price = float(last_close.iloc[-1])
    except Exception:
        last_price = None

    if last_price is None:
        return {"status": "skipped", "reason": "no price available"}

    suggested_qty = max(target_position_qty(float(equity), last_price, risk_fraction), 1)

    if direction == 1:
        action = "BUY"
        side = "buy"
        qty = int(suggested_qty)
    else:
        action = "SELL"
        side = "sell"
        qty = _alpaca_position_qty(symbol) if not demo_mode else int(suggested_qty)
        qty = max(int(qty), 1)

    if demo_mode:
        order_result = {"demo": True, "symbol": symbol, "side": side, "qty": qty}
    else:
        order_result = submit_order(symbol, qty, side)

    account_snapshot = {"equity": float(equity), "cash": None, "buying_power": None}
    if not demo_mode:
        live_snapshot = _fetch_live_account_snapshot()
        if live_snapshot:
            account_snapshot = live_snapshot

    log_entry = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "source": "auto_ml",
        "symbol": symbol,
        "action": action,
        "side": side,
        "qty": int(qty),
        "price": round(last_price, 4),
        "forecast_change_pct": float(forecast_change_pct),
        "risk_fraction": float(risk_fraction),
        "demo_mode": bool(demo_mode),
        "equity_after": account_snapshot.get("equity"),
        "cash_after": account_snapshot.get("cash"),
        "buying_power_after": account_snapshot.get("buying_power"),
        "order_result": order_result,
    }
    _append_trade_log(log_entry)

    state[symbol] = symbol_state if symbol_state else {}
    state[symbol]["last_ml_action_key"] = action_key
    state[symbol]["last_ml_signal"] = direction
    state[symbol]["last_ml_timestamp"] = log_entry["timestamp_utc"]
    _write_trade_state(state)

    return {"status": "executed", "entry": log_entry}


st.set_page_config(page_title="📈 Market Agent Dashboard", layout="wide")

st.title("🤖 Amir Exir Stock Market & Crypto AI Agent")

SYMBOL_OPTIONS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO",
    "ORCL", "NFLX", "JPM", "BAC", "WFC", "C", "GS", "MS", "V",
    "UNH", "LLY", "JNJ", "WMT", "PG", "KO", "HD", "PEP",
    "DAL", "UAL", "AAL", "LUV", "XOM", "CVX", "COP", "OXY", "SLB", "EOG",
    "SPY", "VOO", "QQQ", "IWM", "DIA", "GLD", "SLV", "USO", "TLT",
    "XLK", "XLF", "XLE", "XLV", "XLY",
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "BNB-USD", "AVAX-USD",
    "ORCA-USD", "PNUT-USD", "DOGE-USD", "SHIB-USD", "FOLKI-USD", "FLOKI-USD", "PEPE-USD",
]
DEFAULT_FORECAST_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO",
    "ORCL", "NFLX", "JPM", "BAC", "WFC", "C", "GS", "MS", "V",
    "UNH", "LLY", "JNJ", "WMT", "PG", "KO", "HD", "PEP",
    "DAL", "UAL", "AAL", "LUV", "XOM", "CVX", "COP", "OXY", "SLB", "EOG",
    "SPY", "VOO", "QQQ", "IWM", "DIA", "GLD", "SLV", "USO", "TLT",
    "XLK", "XLF", "XLE", "XLV", "XLY",
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "BNB-USD", "AVAX-USD",
    "ORCA-USD", "PNUT-USD", "DOGE-USD", "SHIB-USD", "FOLKI-USD", "FLOKI-USD", "PEPE-USD",
]
SYMBOL_LABELS = {
    "AVGO": "Broadcom (AVGO)",
    "ORCL": "Oracle (ORCL)",
    "NFLX": "Netflix (NFLX)",
    "WFC": "Wells Fargo (WFC)",
    "C": "Citigroup (C)",
    "GS": "Goldman Sachs (GS)",
    "MS": "Morgan Stanley (MS)",
    "DAL": "Delta Air Lines (DAL)",
    "UAL": "United Airlines (UAL)",
    "AAL": "American Airlines (AAL)",
    "LUV": "Southwest Airlines (LUV)",
    "COP": "ConocoPhillips (COP)",
    "OXY": "Occidental (OXY)",
    "SLB": "SLB (SLB)",
    "EOG": "EOG Resources (EOG)",
    "GLD": "Gold (GLD)",
    "SLV": "Silver (SLV)",
    "USO": "Oil (USO)",
    "BTC-USD": "Bitcoin (BTC)",
    "ETH-USD": "Ethereum (ETH)",
    "SOL-USD": "Solana (SOL)",
    "XRP-USD": "XRP (XRP)",
    "ADA-USD": "Cardano (ADA)",
    "BNB-USD": "BNB (BNB)",
    "AVAX-USD": "Avalanche (AVAX)",
    "ORCA-USD": "Orca (ORCA)",
    "PNUT-USD": "Peanut (PNUT)",
    "DOGE-USD": "Dogecoin (DOGE)",
    "SHIB-USD": "Shiba Inu (SHIB)",
    "FOLKI-USD": "Floki (FOLKI)",
    "FLOKI-USD": "Floki (FLOKI)",
    "PEPE-USD": "Pepe (PEPE)",
}
MARKET_CONTEXT_TICKERS = [
    "SPY", "VOO", "QQQ", "IWM", "DIA", "^VIX",
    "XLK", "XLF", "XLE", "XLV", "XLY",
    "TLT", "GLD", "SLV", "USO", "AVGO",
]


@st.cache_data(ttl=900, show_spinner=False)
def load_ohlcv(symbol: str, history_days: int) -> pd.DataFrame:
    return get_ohlcv(symbol, history_days)


def _load_market_context_data(history_days: int) -> pd.DataFrame:
    start = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=history_days * 2)).strftime("%Y-%m-%d")
    raw = yf.download(MARKET_CONTEXT_TICKERS, start=start, interval="1d", progress=False)
    if raw.empty:
        return pd.DataFrame()

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = close.rename(columns={ticker: f"context_{ticker}" for ticker in close.columns})
    return close.dropna(how="all").ffill()


@st.cache_data(ttl=900, show_spinner=False)
def load_market_context(history_days: int) -> pd.DataFrame:
    return _load_market_context_data(history_days)


@st.cache_data(ttl=900, show_spinner=False)
def load_stock_heatmap_data(tickers: tuple[str, ...], lookback_days: int) -> pd.DataFrame:
    period_days = lookback_days + 3
    hist = yf.download(list(tickers), period=f"{period_days}d", interval="1d", progress=False)["Close"]

    if hist.shape[0] <= lookback_days:
        base_sp = hist.iloc[0]
    else:
        base_sp = hist.iloc[-(lookback_days + 1)]

    last_sp = hist.iloc[-1]
    pct_change = ((last_sp - base_sp) / base_sp * 100).fillna(0)
    market_caps = {}
    for ticker in tickers:
        market_caps[ticker] = yf.Ticker(ticker).info.get("marketCap", 1)

    df = pd.DataFrame(
        {
            "Ticker": pct_change.index,
            "Percent Change": pct_change.values,
            "Market Cap": [market_caps[ticker] for ticker in pct_change.index],
        }
    )
    df["Label"] = df.apply(lambda row: f"{row['Ticker']}\n{row['Percent Change']:.2f}%", axis=1)
    return df


@st.cache_data(ttl=900, show_spinner=False)
def load_watchlist_heatmap_data(
    watchlist_tickers: tuple[str, ...],
    watchlist_labels: dict,
    lookback_days: int,
) -> pd.DataFrame:
    period_days = lookback_days + 3
    watch_hist = yf.download(list(watchlist_tickers), period=f"{period_days}d", interval="1d", progress=False)["Close"]

    if watch_hist.shape[0] <= lookback_days:
        watch_base = watch_hist.iloc[0]
    else:
        watch_base = watch_hist.iloc[-(lookback_days + 1)]

    watch_last = watch_hist.iloc[-1]
    watch_pct_change = ((watch_last - watch_base) / watch_base * 100).fillna(0)
    watch_df = pd.DataFrame(
        {
            "Ticker": watch_pct_change.index,
            "Label": [watchlist_labels.get(ticker, ticker) for ticker in watch_pct_change.index],
            "Percent Change": watch_pct_change.values,
            "Weight": 1,
        }
    )
    watch_df["Display"] = watch_df.apply(
        lambda row: f"{row['Label']}\n{row['Percent Change']:.2f}%",
        axis=1,
    )
    return watch_df


@st.cache_data(ttl=900, show_spinner=False)
def load_crypto_heatmap_data(crypto_tickers: tuple[str, ...], lookback_days: int) -> pd.DataFrame:
    period_days = lookback_days + 5
    crypto_hist = yf.download(list(crypto_tickers), period=f"{period_days}d", interval="1d", progress=False)["Close"]

    if crypto_hist.shape[0] <= lookback_days:
        base = crypto_hist.iloc[0]
    else:
        base = crypto_hist.iloc[-(lookback_days + 1)]

    last = crypto_hist.iloc[-1]
    crypto_pct_change = ((last - base) / base * 100.0).fillna(0)
    crypto_market_caps = {}
    for ticker in crypto_tickers:
        crypto_market_caps[ticker] = yf.Ticker(ticker).info.get("marketCap", 1)

    crypto_df = pd.DataFrame(
        {
            "Crypto": crypto_pct_change.index,
            "Percent Change": crypto_pct_change.values,
            "Market Cap": [crypto_market_caps[ticker] for ticker in crypto_pct_change.index],
        }
    )
    crypto_df["Symbol"] = crypto_df["Crypto"].apply(lambda value: value.split("-")[0])
    crypto_df["Label"] = crypto_df.apply(
        lambda row: f"{row['Symbol']}\n{row['Percent Change']:.2f}%",
        axis=1,
    )
    return crypto_df


def ticker_label(symbol: str) -> str:
    return SYMBOL_LABELS.get(symbol, symbol)


def is_timestamped_summary(filename: str) -> bool:
    if not filename.startswith("summary_") or not filename.endswith(".txt"):
        return False
    timestamp = filename.removeprefix("summary_").removesuffix(".txt")
    return len(timestamp) >= 11 and timestamp[:4].isdigit() and timestamp[4] == "-" and "T" in timestamp


def summary_timestamp_caption(filename: str) -> str | None:
    try:
        timestamp = filename.removeprefix("summary_").removesuffix(".txt")
        date_part, time_part = timestamp.split("T", 1)
        time_part = time_part.replace("Z", "").replace("-", ":")
        return f"📅 Last updated: {date_part} at {time_part} UTC"
    except Exception:
        return None


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


def format_ranking_table(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.reset_index(drop=True).copy()
    if "Rank" not in display_df.columns:
        display_df.insert(0, "Rank", np.arange(1, len(display_df) + 1))

    display_columns = [
        "Rank",
        "Symbol",
        "Model Call",
        "Primary Pattern",
        "Selected Model",
        "Last Price",
        "Forecast Price",
        "Forecast Return %",
        "Ridge Return %",
        "XGBoost Return %",
        "Neural Net Return %",
        "Ensemble Return %",
        "Probability Up %",
        "Directional Probability %",
        "Model Edge %",
        "Signal Quality",
        "Expected Error %",
        "Validation MAE %",
        "Direction Hit Rate %",
        "Score",
    ]
    display_columns = [column for column in display_columns if column in display_df.columns]
    return display_df[display_columns].style.format(
        {
            "Rank": "{:.0f}",
            "Last Price": "${:,.2f}",
            "Forecast Price": "${:,.2f}",
            "Forecast Return %": "{:+.2f}%",
            "Ridge Return %": "{:+.2f}%",
            "XGBoost Return %": "{:+.2f}%",
            "Neural Net Return %": "{:+.2f}%",
            "Ensemble Return %": "{:+.2f}%",
            "Probability Up %": "{:.1f}%",
            "Directional Probability %": "{:.1f}%",
            "Model Edge %": "{:.1f}%",
            "Expected Error %": "{:.2f}%",
            "Validation MAE %": "{:.2f}%",
            "Direction Hit Rate %": "{:.1f}%",
            "Score": "{:+.2f}",
        }
    ).hide(axis="index")


def model_results_table(model_results: dict) -> pd.DataFrame:
    rows = []
    for name, result in model_results.items():
        if result.forecast is None or result.forecast.empty:
            rows.append({"Model": name, "Status": result.metrics.get("error", "Unavailable")})
            continue

        confidence = result.metrics.get("confidence_pct", 50.0)
        rows.append(
            {
                "Model": name,
                "Status": "OK",
                "Forecast Price": float(result.forecast["forecast_close"].iloc[-1]),
                "Forecast Return %": result.metrics.get("forecast_change_pct", 0.0),
                "Probability Up %": result.metrics.get("probability_up_pct", 50.0),
                "Model Edge %": max(confidence - 50.0, 0.0),
                "Signal Quality": signal_quality(confidence),
                "Expected Error %": result.metrics.get("expected_error_pct", 0.0),
                "Validation MAE %": result.metrics.get("holdout_mae_pct", np.nan),
                "Validation Direction %": result.metrics.get("holdout_direction_accuracy", np.nan),
                "Score": result.metrics.get("forecast_score", np.nan),
            }
        )
    return pd.DataFrame(rows)


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

def best_ranked_symbol(ranking_table: pd.DataFrame, fallback: str = "AAPL") -> str:
    if ranking_table.empty or "Symbol" not in ranking_table.columns:
        return fallback

    positive_candidates = ranking_table[ranking_table.get("Forecast Return %", pd.Series(dtype=float)) > 0]
    if not positive_candidates.empty and "Score" in positive_candidates.columns:
        return positive_candidates.sort_values("Score", ascending=False).iloc[0]["Symbol"]

    if "Score" in ranking_table.columns:
        return ranking_table.sort_values("Score", ascending=False).iloc[0]["Symbol"]

    return ranking_table.iloc[0]["Symbol"]


def _clean_series(frame: pd.DataFrame, column: str) -> pd.Series:
    series = frame[column]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return pd.to_numeric(series, errors="coerce").dropna()


def _latest_value(series: pd.Series, default=np.nan) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return default
    return float(clean.iloc[-1])


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    changes = close.diff()
    gains = changes.clip(lower=0).rolling(window).mean()
    losses = (-changes.clip(upper=0)).rolling(window).mean()
    rs = gains / losses.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _recent_cross(short_ma: pd.Series, long_ma: pd.Series, direction: str, lookback: int = 5) -> bool:
    diff = (short_ma - long_ma).dropna()
    if len(diff) < 2:
        return False
    recent = diff.tail(max(2, lookback + 1))
    previous = recent.shift(1)
    if direction == "bullish":
        return bool(((previous <= 0) & (recent > 0)).any())
    return bool(((previous >= 0) & (recent < 0)).any())


@st.cache_data(ttl=900, show_spinner=False)
def scan_common_patterns(
    symbols: tuple[str, ...],
    history_days: int,
    short_window: int,
    long_window: int,
    cache_buster: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    return scan_patterns(
        symbols=symbols,
        history_days=history_days,
        get_ohlcv=get_ohlcv,
        short_window=short_window,
        long_window=long_window,
    )


def build_quick_price_chart(df: pd.DataFrame, short_window: int, long_window: int, symbol: str) -> go.Figure:
    close = df["close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()

    short_ma = close.rolling(short_window).mean()
    long_ma = close.rolling(long_window).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=close.index,
            y=close,
            mode="lines",
            name="Price",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="%{x}<br>Price: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=short_ma.index,
            y=short_ma,
            mode="lines",
            name=f"{short_window}-day MA",
            line=dict(color="#ff7f0e", width=1.8),
            hovertemplate="%{x}<br>Short MA: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=long_ma.index,
            y=long_ma,
            mode="lines",
            name=f"{long_window}-day MA",
            line=dict(color="#2ca02c", width=1.8),
            hovertemplate="%{x}<br>Long MA: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"⚡ Quick Price View for {ticker_label(symbol)}",
        xaxis_title="Date",
        yaxis_title="Price",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _forecast_reports_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "reports")


def _load_forecast_payload(cache_path: str) -> dict:
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _save_forecast_payload(cache_path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)


@st.cache_data(ttl=900, show_spinner=False)
def cached_model_results(
    symbol: str,
    history_days: int,
    forecast_horizon: int,
    forecast_lookback: int,
    forecast_alpha: float,
    optimize_forecast_model: bool,
    use_market_context: bool,
    ranking_symbols: tuple[str, ...] | None = None,
    cache_buster: int = 0,
) -> dict:
    ranking_symbols = ranking_symbols or tuple()
    ranking_cache_path = None
    if symbol in ranking_symbols:
        ranking_cache_path = forecast_cache_path(
            _forecast_reports_dir(),
            ranking_symbols,
            history_days,
            forecast_horizon,
            forecast_lookback,
            forecast_alpha,
            optimize_forecast_model,
            use_market_context,
        )

    cache_path = forecast_cache_path(
        _forecast_reports_dir(),
        (symbol,),
        history_days,
        forecast_horizon,
        forecast_lookback,
        forecast_alpha,
        optimize_forecast_model,
        use_market_context,
    )

    if cache_buster == 0:
        for candidate_cache_path in [ranking_cache_path, cache_path]:
            if not candidate_cache_path:
                continue
            cached_payload = _load_forecast_payload(str(candidate_cache_path))
            snapshots = cached_payload.get("snapshots") or []
            if snapshots:
                snapshot = next((item for item in snapshots if item.get("symbol") == symbol), None)
                if snapshot:
                    model_results = {
                        model_name: forecast_result_from_dict(model_payload)
                        for model_name, model_payload in (snapshot.get("models") or {}).items()
                    }
                    if model_results:
                        return model_results

    df = get_ohlcv(symbol, history_days)
    context_df = _load_market_context_data(history_days) if use_market_context else pd.DataFrame()
    model_results = compare_forecast_models(
        df,
        horizon_days=forecast_horizon,
        lookback_window=forecast_lookback,
        ridge_alpha=forecast_alpha,
        optimize_model=optimize_forecast_model,
        context_df=context_df,
    )

    close = df["close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ"),
        "symbols": [symbol],
        "history_days": history_days,
        "horizon_days": forecast_horizon,
        "lookback_window": forecast_lookback,
        "ridge_alpha": forecast_alpha,
        "optimize_model": optimize_forecast_model,
        "use_market_context": use_market_context,
        "primary_model": "Ensemble",
        "snapshots": [snapshot_from_model_results(symbol, float(close.iloc[-1]), model_results)],
        "rows": [],
        "errors": [],
        "telegram_text": "",
    }
    _save_forecast_payload(str(cache_path), payload)
    return model_results


@st.cache_data(ttl=900, show_spinner=False)
def cached_historical_forecasts(
    symbol: str,
    history_days: int,
    forecast_horizon: int,
    selected_window: int,
    selected_alpha: float,
    historical_test_points: int,
    primary_model: str,
    use_market_context: bool,
    cache_buster: int = 0,
):
    df = get_ohlcv(symbol, history_days)
    context_df = _load_market_context_data(history_days) if use_market_context else pd.DataFrame()
    return backtest_forecasts(
        df,
        horizon_days=forecast_horizon,
        lookback_window=int(selected_window),
        ridge_alpha=float(selected_alpha),
        max_points=historical_test_points,
        optimize_model=False,
        model_type=primary_model.lower(),
        context_df=context_df,
    )


@st.cache_data(ttl=900, show_spinner=False)
def cached_forecast_rankings(
    ranking_symbols: tuple[str, ...],
    history_days: int,
    forecast_horizon: int,
    forecast_lookback: int,
    forecast_alpha: float,
    optimize_forecast_model: bool,
    use_market_context: bool,
    primary_model_choice: str,
    cache_buster: int = 0,
) -> tuple[pd.DataFrame, list[str]]:
    ranking_cache_path = forecast_cache_path(
        _forecast_reports_dir(),
        ranking_symbols,
        history_days,
        forecast_horizon,
        forecast_lookback,
        forecast_alpha,
        optimize_forecast_model,
        use_market_context,
    )

    ranking_payload = _load_forecast_payload(str(ranking_cache_path)) if cache_buster == 0 else {}
    ranking_snapshots = ranking_payload.get("snapshots") or []
    ranking_errors = list(ranking_payload.get("errors", [])) if ranking_snapshots else []

    if not ranking_snapshots:
        ranking_context_df = _load_market_context_data(history_days) if use_market_context else pd.DataFrame()
        ranking_snapshots = []

        for ranking_symbol in ranking_symbols:
            try:
                ranking_df = get_ohlcv(ranking_symbol, history_days)
                ranking_results = compare_forecast_models(
                    ranking_df,
                    horizon_days=forecast_horizon,
                    lookback_window=forecast_lookback,
                    ridge_alpha=forecast_alpha,
                    optimize_model=optimize_forecast_model,
                    context_df=ranking_context_df,
                )
                ranking_close = ranking_df["close"]
                if isinstance(ranking_close, pd.DataFrame):
                    ranking_close = ranking_close.iloc[:, 0]
                ranking_close = pd.to_numeric(ranking_close, errors="coerce").dropna()
                ranking_snapshots.append(snapshot_from_model_results(ranking_symbol, float(ranking_close.iloc[-1]), ranking_results))
            except Exception as ranking_error:
                ranking_errors.append(f"{ranking_symbol}: {ranking_error}")

        ranking_payload = {
            "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ"),
            "horizon_days": forecast_horizon,
            "primary_model": primary_model_choice,
            "symbols": list(ranking_symbols),
            "cache_key": "",
            "rows": [],
            "snapshots": ranking_snapshots,
            "errors": ranking_errors,
            "telegram_text": "",
        }
        _save_forecast_payload(str(ranking_cache_path), ranking_payload)

    ranking_table, ranking_errors_from_snapshots = snapshots_to_ranking_frame(ranking_snapshots, primary_model_choice)
    if ranking_errors_from_snapshots:
        ranking_errors.extend(ranking_errors_from_snapshots)
    return ranking_table, ranking_errors


def report_generated_caption(report_text: str) -> str | None:
    for line in report_text.splitlines():
        if line.startswith("Generated:"):
            return f"📅 {line}"
    return None


@st.cache_data(ttl=300)
def fetch_latest_ml_report():
    local_path = os.path.join(
        os.path.dirname(__file__) if __file__ else ".",
        "reports",
        "ml_forecast_rankings_latest.txt",
    )
    if os.path.exists(local_path):
        try:
            with open(local_path, "r") as f:
                return f.read()
        except Exception:
            pass

    contents_url = (
        "https://api.github.com/repos/AmirExir/portfolio/contents/"
        "market_agent/reports/ml_forecast_rankings_latest.txt"
    )
    try:
        response = requests.get(
            contents_url,
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Streamlit-Market-Agent",
            },
            timeout=10,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        download_url = response.json().get("download_url")
        if not download_url:
            return None
        report_response = requests.get(download_url, timeout=10)
        report_response.raise_for_status()
        return report_response.text
    except Exception:
        return None


# --- Fetch the latest summary from GitHub ---
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_latest_summary():
    """Fetch the latest summary file from GitHub with better error handling"""
    contents_url = "https://api.github.com/repos/AmirExir/portfolio/contents/market_agent"
    
    try:
        # Add headers to avoid rate limiting
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Streamlit-Market-Agent"
        }
        response = requests.get(contents_url, headers=headers, timeout=10)
        
        # Check for rate limiting
        if response.status_code == 403 and 'rate limit' in response.text.lower():
            return None
        
        response.raise_for_status()
        files = response.json()
        return files
    except requests.exceptions.Timeout:
        return None
    except Exception as e:
        # Silently fail and let the fallback handle it
        return None

# Sidebar: Owner key, Alpaca creds, account summary, strategy and forecast settings
owner_key_input = st.sidebar.text_input("🔑 Enter Owner Key", type="password")
OWNER_KEY = get_secret("OWNER_KEY", "")

if owner_key_input == OWNER_KEY and OWNER_KEY != "":
    demo_mode = st.sidebar.checkbox("🎭 Demo Mode", value=False, help="Toggle between live and demo mode")
    if demo_mode:
        st.sidebar.info("Demo Mode active — trades will not be executed.")
    else:
        st.sidebar.success(" Live Mode active — connected to Alpaca paper trading.")
else:
    demo_mode = True
    st.sidebar.info("Demo Mode forced ON for public viewers — safe demo mode.")

if OWNER_KEY == "":
    st.sidebar.caption("Owner key not configured: live mode is disabled, auto-trades run in demo simulation only.")

# --- Load Alpaca credentials from Streamlit Secrets ---
ALPACA_KEY = get_secret("ALPACA_KEY")
ALPACA_SECRET = get_secret("ALPACA_SECRET")
ALPACA_ENDPOINT = get_secret("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets")

# --- Account Summary ---
if demo_mode or not ALPACA_KEY or not ALPACA_SECRET:
    equity, cash, buying_power = 100000.0, 100000.0, 200000.0
    if not ALPACA_KEY or not ALPACA_SECRET:
        st.sidebar.warning("Alpaca API keys not found — using demo values.")
else:
    try:
        acct = get_account()
        if isinstance(acct, dict) and "equity" in acct:
            equity = float(acct.get("equity", 0))
            cash = float(acct.get("cash", 0))
            buying_power = float(acct.get("buying_power", 0))
        else:
            st.sidebar.warning("Invalid response from Alpaca — using demo values.")
            equity, cash, buying_power = 100000.0, 100000.0, 200000.0
    except Exception as e:
        st.sidebar.error(f"Failed to fetch Alpaca account: {e}")
        equity, cash, buying_power = 100000.0, 100000.0, 200000.0

st.sidebar.header("💼 Account Summary (Paper Trading)")
st.sidebar.metric("Equity", f"${equity:,.2f}")
st.sidebar.metric("Cash", f"${cash:,.2f}")
st.sidebar.metric("Buying Power", f"${buying_power:,.2f}")
if st.sidebar.button("🔄 Refresh Account and Money Chart"):
    st.rerun()

# --- Add a Cancel Orders Button ---
if st.sidebar.button("🧹 Cancel Open Orders"):
    if not demo_mode:
        try:
            cancel_open_orders()
            st.sidebar.success("All open orders canceled!")
        except Exception as e:
            st.sidebar.error(f"Failed to cancel orders: {e}")
    else:
        st.sidebar.info("(Demo) Orders not canceled in demo mode")

# --- Strategy Settings ---
st.sidebar.header("⚙️ Strategy Settings")
short_window = st.sidebar.number_input("📏 Short-term MA window", min_value=1, max_value=100, value=20, step=1)
long_window = st.sidebar.number_input("📐 Long-term MA window", min_value=1, max_value=200, value=50, step=1)

st.sidebar.header("🤖 Auto Paper Trading")
auto_trade_enabled = st.sidebar.checkbox(
    "Enable automatic paper trades",
    value=False,
    help="Places BUY/SELL orders when SMA crossover flips.",
)
auto_trade_risk_fraction = st.sidebar.slider(
    "Auto-trade position size (% of equity)",
    min_value=0.01,
    max_value=0.50,
    value=0.10,
    step=0.01,
)
if auto_trade_enabled and demo_mode:
    st.sidebar.info("Auto trading enabled in demo mode: orders are simulated and still logged.")
auto_trade_trigger = st.sidebar.selectbox(
    "Auto-trade trigger",
    ["SMA Crossover", "ML Forecast", "Either"],
    index=0,
    help="Choose whether to trigger auto-trades from SMA crossover, ML forecast direction, or either",
)

st.sidebar.header("🔮 ML Forecast Settings")
history_options = {
    "~2 years": 365,
    "~5 years": 913,
    "~10 years": 1825,
    "~15 years": 2738,
}
history_label = st.sidebar.selectbox("Historical data lookback", list(history_options.keys()), index=1)
history_days = history_options[history_label]
forecast_horizon = st.sidebar.slider("Forecast horizon", min_value=5, max_value=60, value=30, step=5)
forecast_lookback = st.sidebar.slider("Lag window", min_value=5, max_value=60, value=20, step=5)
historical_test_points = st.sidebar.slider("Previous forecast test points", min_value=10, max_value=100, value=50, step=10)
optimize_forecast_model = st.sidebar.checkbox("Optimize ML model", value=True)
use_market_context = st.sidebar.checkbox("Use market context features", value=True)
primary_model_choice = st.sidebar.selectbox("Primary forecast model", ["Ensemble", "Best Validation", "Ridge", "XGBoost", "Neural Net"], index=1)
forecast_alpha = st.sidebar.number_input(
    "Ridge regularization",
    min_value=0.1,
    max_value=100.0,
    value=10.0,
    step=0.5,
)
run_forecast_rankings = st.sidebar.checkbox("Run forecast rankings", value=True)
selected_forecast_symbols = st.sidebar.multiselect(
    "Forecast ranking tickers",
    options=SYMBOL_OPTIONS,
    default=DEFAULT_FORECAST_SYMBOLS,
    format_func=ticker_label,
)

if "ranking_refresh_nonce" not in st.session_state:
    st.session_state["ranking_refresh_nonce"] = 0
if "symbol_refresh_nonce" not in st.session_state:
    st.session_state["symbol_refresh_nonce"] = 0

refresh_all = st.sidebar.button("🔄 Force recalculate all rankings", help="Ignore saved ranking caches and recompute every symbol.")
refresh_selected = st.sidebar.button("🔄 Force recalculate selected stock", help="Ignore the saved selected-stock cache and recompute only the active symbol.")

if refresh_all:
    st.session_state["ranking_refresh_nonce"] += 1
    st.session_state["symbol_refresh_nonce"] += 1
    st.rerun()

if refresh_selected:
    st.session_state["symbol_refresh_nonce"] += 1
    st.rerun()

ranking_table = pd.DataFrame()
ranking_errors: list[str] = []
best_stock_for_analysis = "AAPL"
pattern_summary = pd.DataFrame()
pattern_details = pd.DataFrame()
pattern_errors: list[str] = []

if run_forecast_rankings and selected_forecast_symbols:
    try:
        with st.spinner("Computing ML forecast rankings and optimization..."):
            ranking_table, ranking_errors = cached_forecast_rankings(
                tuple(selected_forecast_symbols),
                history_days,
                forecast_horizon,
                forecast_lookback,
                forecast_alpha,
                optimize_forecast_model,
                use_market_context,
                primary_model_choice,
                st.session_state["ranking_refresh_nonce"],
            )
        best_stock_for_analysis = best_ranked_symbol(ranking_table, fallback="AAPL")
    except Exception as ranking_error:
        st.warning(f"Could not compute rankings initially: {ranking_error}")

# Top-level tabs: Portfolio Balance first, then AI-Generated Market Summary, patterns, then Market Analysis
top_portfolio_tab, top_summary_tab, top_patterns_tab, top_analysis_tab = st.tabs([
    "💼 Portfolio Balance",
    "📰 AI-Generated Market Summary",
    "🧩 Most Common Patterns",
    "📈 Market Analysis",
])

with top_summary_tab:
    st.markdown("📰 AI-Generated Market Summary")
    if st.button("🔄 Refresh News", help="Fetch the latest news from GitHub"):
        st.cache_data.clear()
        st.rerun()

    try:
        files = fetch_latest_summary()

        # Fallback: If GitHub API fails, try reading from local directory (for Streamlit Cloud deployment)
        if files is None:
            st.info(" GitHub API unavailable. Using local files...")
            local_dir = os.path.dirname(__file__) if __file__ else "."

            try:
                local_files = [f for f in os.listdir(local_dir) if is_timestamped_summary(f)]

                if local_files:
                    # Sort and get latest
                    local_files_sorted = sorted(local_files, reverse=True)
                    latest_local_file = local_files_sorted[0]

                    with open(os.path.join(local_dir, latest_local_file), "r") as f:
                        summary_text = f.read()

                    caption = summary_timestamp_caption(latest_local_file)
                    if caption:
                        st.caption(caption)

                    st.info(summary_text.strip())
                else:
                    st.warning("No local summary files found.")
            except Exception as local_error:
                st.error(f"Failed to read local files: {local_error}")
        else:
            # Filter timestamped summary files across years.
            summary_files = [
                f for f in files
                if f.get("type") == "file"
                and is_timestamped_summary(f.get("name", ""))
            ]

            if not summary_files:
                st.info("No summary files found yet. The n8n workflow will create summary_*.txt files on first run.")
            else:
                # Sort by name descending to get the latest (ISO format sorts correctly)
                summary_files_sorted = sorted(summary_files, key=lambda x: x["name"], reverse=True)
                latest_file = summary_files_sorted[0]

                # Extract timestamp from filename (format: summary_2026-05-01T11-00-28-733Z.txt)
                filename = latest_file.get("name", "")
                caption = summary_timestamp_caption(filename)
                if caption:
                    st.caption(caption)

                download_url = latest_file.get("download_url")

                if download_url:
                    content_response = requests.get(download_url, timeout=10)
                    content_response.raise_for_status()
                    summary_decoded = content_response.text

                    # Try parsing as JSON if it's not plain text
                    try:
                        maybe_json = json.loads(summary_decoded)
                        if isinstance(maybe_json, dict):
                            summary_text = maybe_json.get("content") or maybe_json.get("message") or str(maybe_json)
                        elif isinstance(maybe_json, list):
                            summary_text = "\n".join([str(item) for item in maybe_json])
                        else:
                            summary_text = str(maybe_json)
                    except json.JSONDecodeError:
                        summary_text = summary_decoded

                    st.info(summary_text.strip())
                else:
                    st.warning(" Could not find download URL for the latest summary file.")
    except Exception as e:
        st.error(f"⚠️ Unable to load summary: {e}")
        st.info("The news summary will be available when GitHub API or local files are accessible.")

with top_patterns_tab:
    st.markdown("🧩 Most Common Patterns")
    if not selected_forecast_symbols:
        st.info("Select forecast ranking tickers to scan current patterns.")
    else:
        try:
            with st.spinner("Scanning current technical patterns..."):
                pattern_summary, pattern_details, pattern_errors = scan_common_patterns(
                    tuple(selected_forecast_symbols),
                    history_days,
                    short_window,
                    long_window,
                    st.session_state["ranking_refresh_nonce"],
                )

            best_pattern_row = pd.DataFrame()
            if not pattern_details.empty:
                best_pattern_row = pattern_details[pattern_details["Symbol"] == best_stock_for_analysis]
            best_rank_row = pd.DataFrame()
            if not ranking_table.empty and "Symbol" in ranking_table.columns:
                best_rank_row = ranking_table[ranking_table["Symbol"] == best_stock_for_analysis]
            if not best_pattern_row.empty:
                picked = best_pattern_row.iloc[0]
                picked_cols = st.columns(4)
                picked_cols[0].metric("ML Picked Stock", ticker_label(best_stock_for_analysis))
                picked_cols[1].metric("Recognized Pattern", str(picked["Primary Pattern"]))
                if not best_rank_row.empty:
                    picked_cols[2].metric(
                        "Forecast Return",
                        f"{float(best_rank_row.iloc[0]['Forecast Return %']):+.2f}%",
                        str(best_rank_row.iloc[0]["Model Call"]),
                    )
                else:
                    picked_cols[2].metric("20D Return", f"{float(picked['20D Return %']):+.2f}%")
                picked_cols[3].metric("RSI", f"{float(picked['RSI']):.1f}")
                st.caption(f"{best_stock_for_analysis} patterns: {picked['All Patterns']}")
            elif best_stock_for_analysis:
                st.caption(f"ML picked {ticker_label(best_stock_for_analysis)}, but no pattern row was available for that ticker.")

            if pattern_summary.empty:
                st.info("No pattern data was available for the selected tickers.")
            else:
                top_pattern = pattern_summary.iloc[0]
                metric_cols = st.columns(3)
                metric_cols[0].metric("Most Common Pattern", str(top_pattern["Pattern"]))
                metric_cols[1].metric("Tickers Matching", f"{int(top_pattern['Count'])}")
                metric_cols[2].metric("Avg 20D Return", f"{float(top_pattern['Avg 20D Return %']):+.2f}%")

                chart_df = pattern_summary.head(12)
                pattern_fig = px.bar(
                    chart_df,
                    x="Count",
                    y="Pattern",
                    orientation="h",
                    color="Avg 20D Return %",
                    color_continuous_scale="RdYlGn",
                    hover_data={
                        "Count": True,
                        "Avg 20D Return %": ":.2f",
                        "Avg 60D Return %": ":.2f",
                        "Symbols": True,
                    },
                    title="Most Common Patterns Across Selected Tickers",
                )
                pattern_fig.update_layout(
                    yaxis=dict(categoryorder="total ascending"),
                    margin=dict(l=10, r=10, t=45, b=10),
                )
                st.plotly_chart(pattern_fig, use_container_width=True)

                st.dataframe(
                    pattern_summary.style.format(
                        {
                            "Count": "{:.0f}",
                            "Avg 20D Return %": "{:+.2f}%",
                            "Avg 60D Return %": "{:+.2f}%",
                        }
                    ).hide(axis="index"),
                    use_container_width=True,
                )

            if not pattern_details.empty:
                with st.expander("Pattern details by ticker", expanded=True):
                    st.dataframe(
                        pattern_details.sort_values(["Primary Pattern", "20D Return %"], ascending=[True, False])
                        .style.format(
                            {
                                "Last Price": "${:,.4f}",
                                "5D Return %": "{:+.2f}%",
                                "20D Return %": "{:+.2f}%",
                                "60D Return %": "{:+.2f}%",
                                "RSI": "{:.1f}",
                                "20D Volatility %": "{:.1f}%",
                                "Volume Ratio": "{:.2f}x",
                            }
                        )
                        .hide(axis="index"),
                        use_container_width=True,
                    )

            if pattern_errors:
                with st.expander("Symbols skipped during pattern scan"):
                    st.write("\n".join(pattern_errors))
        except Exception as pattern_scan_error:
            st.warning(f"Pattern scan unavailable: {pattern_scan_error}")

if not ranking_table.empty and not pattern_details.empty:
    ranking_pattern_columns = pattern_details[["Symbol", "Primary Pattern"]].drop_duplicates("Symbol")
    ranking_table = ranking_table.drop(columns=["Primary Pattern"], errors="ignore").merge(
        ranking_pattern_columns,
        on="Symbol",
        how="left",
    )

with top_analysis_tab:
    try:
        latest_ml_report = fetch_latest_ml_report()
        if latest_ml_report:
            st.markdown("🧠 Scheduled ML Forecast Rankings")
            report_caption = report_generated_caption(latest_ml_report)
            if report_caption:
                st.caption(report_caption)
            st.info(latest_ml_report.strip())
        else:
            st.caption(
                "Scheduled ML forecast rankings will appear here after n8n writes "
                "market_agent/reports/ml_forecast_rankings_latest.txt."
            )
    except Exception as e:
        st.caption(f"Scheduled ML forecast report unavailable: {e}")

    st.markdown("---")

    #  Real-Time S&P 500 Heatmap (Market Cap Weighted + Labels)
    st.subheader("🧭 Real-Time S&P 500 Heatmap")

    # Timeframe selector
    sp_tf = st.selectbox(
        "📉 Stock change timeframe",
        ["1D", "7D", "1M", "3M", "1Y", "5Y"],
        index=0,
        key="stock_tf"
    )

    # Time-to-days map
    sp_days_map = {
        "1D": 1,
        "7D": 7,
        "1M": 30,
        "3M": 90,
        "1Y": 365,
        "5Y": 365 * 5,
    }

    lookback_days = sp_days_map[sp_tf]

    # S&P 500 sample tickers
    tickers = [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM",
        "UNH", "XOM", "V", "JNJ", "WMT", "PG", "KO", "HD", "BAC", "CVX",
        "LLY", "PEP", "AVGO"
    ]

    try:
        df = load_stock_heatmap_data(tuple(tickers), lookback_days)
        fig = px.treemap(
            df,
            path=["Ticker"],
            values="Market Cap",
            color="Percent Change",
            color_continuous_scale="RdYlGn",
            hover_data={"Market Cap": ":,.0f", "Percent Change": ":.2f"},
            title=f"📊 S&P 500 Change ({sp_tf}) – Sized by Market Cap"
        )

        fig.update_traces(text=df["Label"])
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error generating heatmap: {e}")
        st.info("Please try a different timeframe or check your network connection.")


    # ETF / commodity watchlist requested for broader market context
    st.subheader("🧾 ETF and Commodity Watchlist")

    watchlist_tickers = ["SPY", "VOO", "GLD", "SLV", "USO"]
    watchlist_labels = {
        "SPY": "SPY",
        "VOO": "VOO",
        "GLD": "Gold (GLD)",
        "SLV": "Silver (SLV)",
        "USO": "Oil (USO)",
    }

    try:
        watch_df = load_watchlist_heatmap_data(tuple(watchlist_tickers), watchlist_labels, lookback_days)
        watch_fig = px.treemap(
            watch_df,
            path=["Label"],
            values="Weight",
            color="Percent Change",
            color_continuous_scale="RdYlGn",
            hover_data={"Percent Change": ":.2f"},
            title=f"SPY, VOO, Gold, Silver, and Oil Change ({sp_tf})",
        )
        watch_fig.update_traces(text=watch_df["Display"])
        st.plotly_chart(watch_fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating watchlist heatmap: {e}")


    # --- Real-Time Crypto Heatmap (Market Cap Weighted + Labels)
    st.subheader("🪙 Real-Time Crypto Heatmap")

    # Timeframe selector for crypto
    crypto_tf = st.selectbox(
        "⏱️ Crypto change timeframe",
        ["24H", "7D", "1M", "3M", "1Y", "5Y"],
        index=0
    )

    # Map timeframe -> lookback in days
    tf_days_map = {
        "24H": 1,
        "7D": 7,
        "1M": 30,
        "3M": 90,
        "1Y": 365,
        "5Y": 365 * 5,
    }

    lookback_days = tf_days_map[crypto_tf]

    crypto_tickers = [
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD",
        "DOGE-USD", "AVAX-USD", "TON-USD", "DOT-USD"
    ]

    try:
        crypto_df = load_crypto_heatmap_data(tuple(crypto_tickers), lookback_days)
        crypto_fig = px.treemap(
            crypto_df,
            path=["Symbol"],
            values="Market Cap",
            color="Percent Change",
            color_continuous_scale="RdYlGn",
            hover_data={"Market Cap": ":,.0f", "Percent Change": ":.2f"},
            title=f"🪙 Crypto Change ({crypto_tf}) – Sized by Market Cap"
        )

        crypto_fig.update_traces(text=crypto_df["Label"])
        st.plotly_chart(crypto_fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error generating crypto heatmap: {e}")

    # --- Quick Price + SMA Crossover View ---
    quick_symbol = st.selectbox(
        "⚡ Quick chart symbol",
        options=SYMBOL_OPTIONS,
        index=SYMBOL_OPTIONS.index("AAPL"),
        format_func=ticker_label,
        key="quick_symbol_select",
    )

    st.subheader("⚡ Quick Price + SMA Crossover")
    try:
        quick_df = load_ohlcv(quick_symbol, history_days)
        st.plotly_chart(
            build_quick_price_chart(quick_df, short_window, long_window, quick_symbol),
            use_container_width=True,
        )
    except Exception as quick_error:
        quick_df = None
        st.warning(f"Quick price chart unavailable: {quick_error}")

    if run_forecast_rankings and not ranking_table.empty:
        st.markdown("---")
        st.subheader("🔎 ML Based Forecast Rankings (Early View)")

        buy_candidates = ranking_table[ranking_table["Forecast Return %"] > 0]
        sell_candidates = ranking_table[ranking_table["Forecast Return %"] < 0]
        strongest_buy = buy_candidates.sort_values("Score", ascending=False).head(10)
        strongest_sell = sell_candidates.sort_values("Score", ascending=True).head(10)

        buy_col, sell_col = st.columns(2)
        with buy_col:
            st.markdown("**Strongest Buy Forecasts**")
            if strongest_buy.empty:
                st.info("No positive forecast candidates.")
            else:
                st.dataframe(format_ranking_table(strongest_buy), use_container_width=True)
        with sell_col:
            st.markdown("**Strongest Sell Forecasts**")
            if strongest_sell.empty:
                st.info("No negative forecast candidates.")
            else:
                st.dataframe(format_ranking_table(strongest_sell), use_container_width=True)

        with st.expander("All forecast ranking results"):
            st.dataframe(format_ranking_table(ranking_table.sort_values("Score", ascending=False)), use_container_width=True)

        if ranking_errors:
            with st.expander("Symbols skipped during forecast ranking"):
                st.write("\n".join(ranking_errors))

        st.markdown("---")

    # --- Symbol input (defaults to best stock from rankings) ---
    symbol = st.selectbox(
        "🏷️ Symbol",
        options=SYMBOL_OPTIONS,
        index=SYMBOL_OPTIONS.index(best_stock_for_analysis) if best_stock_for_analysis in SYMBOL_OPTIONS else SYMBOL_OPTIONS.index("AAPL"),
        format_func=ticker_label,
        key="symbol_select",
    )

# Re-use the previously created top-level tabs so Portfolio Balance stays at the very top
analysis_tab = top_analysis_tab
money_tab = top_portfolio_tab

# --- Load data and backtest ---
try:
    df = load_ohlcv(symbol, history_days)
    
    sig = sma_crossover(df, short_window, long_window)
    bt = simple_vector_backtest(df, sig)
    actual_close = df["close"]
    if isinstance(actual_close, pd.DataFrame):
        actual_close = actual_close.iloc[:, 0]
    actual_close = pd.to_numeric(actual_close, errors="coerce").dropna()
    context_df = load_market_context(history_days) if use_market_context else pd.DataFrame()
    
    with analysis_tab:
        st.subheader("📈 Actual Value, ML Based Forecast, and Crossover Strategy")
        try:
            model_results = cached_model_results(
                symbol,
                history_days,
                forecast_horizon,
                forecast_lookback,
                forecast_alpha,
                optimize_forecast_model,
                use_market_context,
                tuple(selected_forecast_symbols),
                st.session_state["symbol_refresh_nonce"],
            )
            if primary_model_choice == "Best Validation":
                primary_model = best_model_name(model_results, preferred="")
            else:
                primary_model = best_model_name(model_results, preferred=primary_model_choice)
            if not primary_model:
                raise ValueError("No forecast model produced a usable forecast.")

            forecast_result = model_results[primary_model]
            forecast_df = forecast_result.forecast
            historical_result = None
            historical_forecasts = pd.DataFrame()
            forecast_change = forecast_result.metrics.get("forecast_change_pct", 0.0)
            probability_up = forecast_result.metrics.get("probability_up_pct", 0.0)
            probability_down = forecast_result.metrics.get("probability_down_pct", 0.0)
            confidence = forecast_result.metrics.get("confidence_pct", 0.0)
            model_edge = max(confidence - 50.0, 0.0)
            quality_label = signal_quality(confidence)
            expected_error = forecast_result.metrics.get("expected_error_pct", 0.0)
            selected_window = forecast_result.metrics.get("selected_lookback_window", forecast_lookback)
            selected_alpha = forecast_result.metrics.get("selected_ridge_alpha", forecast_alpha)
            shrink_factor = forecast_result.metrics.get("shrink_factor", 1.0)
            raw_forecast = forecast_result.metrics.get("raw_forecast_change_pct", forecast_change)
            try:
                historical_result = cached_historical_forecasts(
                    symbol,
                    history_days,
                    forecast_horizon,
                    int(selected_window),
                    float(selected_alpha),
                    historical_test_points,
                    primary_model,
                    use_market_context,
                    st.session_state["symbol_refresh_nonce"],
                )
                historical_forecasts = historical_result.forecasts
            except Exception as history_error:
                st.info(f"Previous ML forecast test unavailable: {history_error}")

            latest_actual_date = actual_close.index[-1]
            latest_actual_price = float(actual_close.iloc[-1])
            latest_actual_label = f"Actual: ${latest_actual_price:,.2f}"
            final_forecast_date = forecast_df.index[-1]
            final_forecast_price = float(forecast_df["forecast_close"].iloc[-1])
            final_forecast_label = f"{primary_model}: ${final_forecast_price:,.2f} ({forecast_change:+.2f}%)"
            forecast_plot_x = [latest_actual_date, *forecast_df.index.tolist()]
            forecast_plot_y = [latest_actual_price, *forecast_df["forecast_close"].tolist()]

            price_fig = go.Figure()
            price_fig.add_trace(
                go.Scatter(
                    x=actual_close.index,
                    y=actual_close,
                    mode="lines",
                    name="Actual Value",
                    line=dict(color="#1f77b4", width=2),
                    hovertemplate="%{x}<br>Actual value: $%{y:,.2f}<extra></extra>",
                )
            )
            price_fig.add_trace(
                go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df["upper_estimate"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            price_fig.add_trace(
                go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df["lower_estimate"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(44, 160, 44, 0.15)",
                    name="90% estimate range",
                    hoverinfo="skip",
                )
            )
            price_fig.add_trace(
                go.Scatter(
                    x=forecast_plot_x,
                    y=forecast_plot_y,
                    mode="lines",
                    name=f"Primary ML Forecast ({primary_model})",
                    line=dict(color="#2ca02c", width=2, dash="dash"),
                    hovertemplate="%{x}<br>Primary ML forecast: $%{y:,.2f}<extra></extra>",
                )
            )
            comparison_colors = {
                "Ridge": "#8c564b",
                "XGBoost": "#d62728",
                "Neural Net": "#17becf",
                "Ensemble": "#2ca02c",
            }
            for comparison_name, comparison_result in model_results.items():
                if comparison_name == primary_model or comparison_result.forecast is None or comparison_result.forecast.empty:
                    continue

                price_fig.add_trace(
                    go.Scatter(
                        x=[latest_actual_date, *comparison_result.forecast.index.tolist()],
                        y=[latest_actual_price, *comparison_result.forecast["forecast_close"].tolist()],
                        mode="lines",
                        name=f"{comparison_name} Forecast",
                        line=dict(
                            color=comparison_colors.get(comparison_name, "#7f7f7f"),
                            width=1.5,
                            dash="dot",
                        ),
                        hovertemplate=f"%{{x}}<br>{comparison_name}: $%{{y:,.2f}}<extra></extra>",
                    )
                )
            if not historical_forecasts.empty:
                error_line_x = []
                error_line_y = []
                for _, row in historical_forecasts.iterrows():
                    error_line_x.extend([row["forecast_date"], row["forecast_date"], None])
                    error_line_y.extend([row["actual_close"], row["forecast_close"], None])

                price_fig.add_trace(
                    go.Scatter(
                        x=error_line_x,
                        y=error_line_y,
                        mode="lines",
                        name="Previous Forecast Error",
                        line=dict(color="rgba(255, 127, 14, 0.25)", width=1),
                        hoverinfo="skip",
                    )
                )
                price_fig.add_trace(
                    go.Scatter(
                        x=historical_forecasts["forecast_date"],
                        y=historical_forecasts["forecast_close"],
                        mode="markers",
                        name="Previous ML Forecasts",
                        marker=dict(color="#ff7f0e", size=7, symbol="diamond"),
                        customdata=historical_forecasts[
                            ["actual_close", "error_pct", "confidence_pct", "expected_error_pct"]
                        ],
                        hovertemplate=(
                            "%{x}<br>"
                            "Previous ML forecast: $%{y:,.2f}<br>"
                            "Actual close: $%{customdata[0]:,.2f}<br>"
                            "Error: %{customdata[1]:+.2f}%<br>"
                            "Confidence then: %{customdata[2]:.1f}%<br>"
                            "Expected error then: %{customdata[3]:.2f}%"
                            "<extra></extra>"
                        ),
                    )
                )
            price_fig.add_trace(
                go.Scatter(
                    x=[latest_actual_date],
                    y=[latest_actual_price],
                    mode="markers+text",
                    name="Latest Actual Value",
                    marker=dict(color="#1f77b4", size=10, line=dict(color="white", width=2)),
                    text=[latest_actual_label],
                    textposition="bottom left",
                    textfont=dict(color="#0d47a1", size=13),
                    hovertemplate="%{x}<br>Latest actual value: $%{y:,.2f}<extra></extra>",
                )
            )
            price_fig.add_trace(
                go.Scatter(
                    x=[final_forecast_date],
                    y=[final_forecast_price],
                    mode="markers+text",
                    name=f"{forecast_horizon}-Day {primary_model} Forecast",
                    marker=dict(color="#2ca02c", size=10, line=dict(color="white", width=2)),
                    text=[final_forecast_label],
                    textposition="top right",
                    textfont=dict(color="#1b5e20", size=13),
                    hovertemplate="%{x}<br>ML based forecast value: $%{y:,.2f}<extra></extra>",
                )
            )
            price_fig.add_trace(
                go.Scatter(
                    x=bt.index,
                    y=bt["curve"],
                    mode="lines",
                    name="Using the Crossover Strategy",
                    line=dict(color="#9467bd", width=2),
                    yaxis="y2",
                    hovertemplate="%{x}<br>Crossover strategy equity: %{y:.4f}<extra></extra>",
                )
            )
            price_fig.update_layout(
                xaxis_title="Date",
                yaxis=dict(title="Actual / Forecast Price"),
                yaxis2=dict(
                    title="Crossover Strategy Equity Curve",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                ),
                hovermode="x unified",
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(price_fig, use_container_width=True)

            metric_cols = st.columns(3)
            metric_cols[0].metric(
                f"{forecast_horizon}-day forecast",
                f"${final_forecast_price:,.2f}",
                f"{forecast_change:.2f}%",
            )
            metric_cols[1].metric("Probability Up", f"{probability_up:.1f}%")
            metric_cols[2].metric("Expected Error", f"±{expected_error:.2f}%")

            quality_cols = st.columns(3)
            quality_cols[0].metric("Model Edge", f"{model_edge:.1f}%", quality_label)
            if historical_result is not None:
                quality_cols[1].metric(
                    "Previous Forecast MAE",
                    f"{historical_result.metrics['historical_mae_pct']:.2f}%",
                )
                quality_cols[2].metric(
                    "Previous Direction Hit Rate",
                    f"{historical_result.metrics['historical_direction_accuracy']:.1f}%",
                )
            elif "holdout_direction_accuracy" in forecast_result.metrics:
                quality_cols[1].metric(
                    "Holdout Direction",
                    f"{forecast_result.metrics['holdout_direction_accuracy']:.1f}%",
                )
                quality_cols[2].metric(
                    "Holdout MAE",
                    f"{forecast_result.metrics['holdout_mae_pct']:.2f}%",
                )
            else:
                quality_cols[1].metric("Training Samples", forecast_result.metrics.get("training_samples", 0))
                quality_cols[2].metric("Probability Down", f"{probability_down:.1f}%")

            st.caption(
                "Model: "
                f"{forecast_result.model_name}. "
                f"Primary model: {primary_model}. "
                f"Market context features: {'on' if use_market_context and not context_df.empty else 'off'}. "
                f"Selected lag window: {selected_window}; "
                f"ridge alpha: {selected_alpha:g}; "
                f"raw forecast: {raw_forecast:+.2f}%; "
                f"calibrated forecast: {forecast_change:+.2f}%; "
                f"shrink factor: {shrink_factor:.2f}."
            )
            if quality_label == "No Edge":
                st.info(
                    "The optimized model is effectively neutral here: directional probability is close to 50%, "
                    "so the current feature set does not justify a strong forecast."
                )

            with st.expander("Model comparison"):
                comparison_table = model_results_table(model_results)
                st.dataframe(
                    comparison_table.style.format(
                        {
                            "Forecast Price": "${:,.2f}",
                            "Forecast Return %": "{:+.2f}%",
                            "Probability Up %": "{:.1f}%",
                            "Model Edge %": "{:.1f}%",
                            "Expected Error %": "{:.2f}%",
                            "Validation MAE %": "{:.2f}%",
                            "Validation Direction %": "{:.1f}%",
                            "Score": "{:+.3f}",
                        },
                        na_rep="",
                    ),
                    use_container_width=True,
                )

            if historical_result is not None:
                with st.expander("Previous forecast accuracy"):
                    st.caption(
                        "Each row shows what the ML model would have forecasted using only data available at that past date."
                    )
                    display_history = historical_forecasts.rename(
                        columns={
                            "as_of_date": "Forecast Made On",
                            "forecast_date": "Forecast Target Date",
                            "as_of_price": "Price Then",
                            "forecast_close": "Forecast Close",
                            "actual_close": "Actual Close",
                            "predicted_change_pct": "Forecast Return %",
                            "actual_change_pct": "Actual Return %",
                            "error_pct": "Error %",
                            "abs_error_pct": "Absolute Error %",
                            "confidence_pct": "Confidence %",
                            "expected_error_pct": "Expected Error %",
                            "direction_correct": "Direction Correct",
                        }
                    )
                    st.dataframe(display_history)

            with st.expander("Forecast values"):
                display_forecast = forecast_df.rename(
                    columns={
                        "forecast_close": "Forecast Close",
                        "lower_estimate": "Lower Estimate",
                        "upper_estimate": "Upper Estimate",
                        "expected_daily_return_pct": "Expected Daily Return %",
                    }
                )
                st.dataframe(display_forecast)
        except Exception as forecast_error:
            st.line_chart(actual_close)
            st.warning(f"ML forecast unavailable: {forecast_error}")

        st.subheader("🧾 Paper Trading Controls")
        trade_log_df = load_trade_log(limit=500)
        trade_summary = _trade_summary(trade_log_df, symbol, float(actual_close.iloc[-1]))

        control_col1, control_col2 = st.columns(2)
        buy_qty = control_col1.number_input("Buy shares", min_value=1, max_value=100000, value=1, step=1)
        sell_qty = control_col2.number_input("Sell shares", min_value=1, max_value=100000, value=1, step=1)
        est_buy_dollars = float(buy_qty) * float(actual_close.iloc[-1])
        est_sell_dollars = float(sell_qty) * float(actual_close.iloc[-1])

        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button(f"🟢 Buy {buy_qty} shares of {symbol} (${est_buy_dollars:,.2f})"):
                if demo_mode:
                    st.info(f"(Demo) Pretending to buy {buy_qty} shares of {symbol}")
                    _append_trade_log(
                        {
                            "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "source": "manual",
                            "symbol": symbol,
                            "action": "BUY",
                            "side": "buy",
                            "qty": int(buy_qty),
                            "price": float(actual_close.iloc[-1]),
                            "notional": est_buy_dollars,
                            "equity_after": float(equity),
                            "cash_after": float(cash),
                            "buying_power_after": float(buying_power),
                            "demo_mode": True,
                            "order_result": {"demo": True},
                        }
                    )
                    st.rerun()
                else:
                    try:
                        result = submit_order(symbol, int(buy_qty), "buy")
                        st.success(f"Bought {buy_qty} shares of {symbol}")
                        st.json(result)
                        account_snapshot = _fetch_live_account_snapshot()
                        _append_trade_log(
                            {
                                "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                                "source": "manual",
                                "symbol": symbol,
                                "action": "BUY",
                                "side": "buy",
                                "qty": int(buy_qty),
                                "price": float(actual_close.iloc[-1]),
                                "notional": est_buy_dollars,
                                "equity_after": account_snapshot.get("equity"),
                                "cash_after": account_snapshot.get("cash"),
                                "buying_power_after": account_snapshot.get("buying_power"),
                                "demo_mode": False,
                                "order_result": result,
                            }
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to buy: {e}")
        with action_col2:
            if st.button(f"🔴 Sell {sell_qty} shares of {symbol} (${est_sell_dollars:,.2f})"):
                if demo_mode:
                    st.info(f"(Demo) Pretending to sell {sell_qty} shares of {symbol}")
                    _append_trade_log(
                        {
                            "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "source": "manual",
                            "symbol": symbol,
                            "action": "SELL",
                            "side": "sell",
                            "qty": int(sell_qty),
                            "price": float(actual_close.iloc[-1]),
                            "notional": est_sell_dollars,
                            "equity_after": float(equity),
                            "cash_after": float(cash),
                            "buying_power_after": float(buying_power),
                            "demo_mode": True,
                            "order_result": {"demo": True},
                        }
                    )
                    st.rerun()
                else:
                    try:
                        result = submit_order(symbol, int(sell_qty), "sell")
                        st.warning(f"Sold {sell_qty} shares of {symbol}")
                        st.json(result)
                        account_snapshot = _fetch_live_account_snapshot()
                        _append_trade_log(
                            {
                                "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                                "source": "manual",
                                "symbol": symbol,
                                "action": "SELL",
                                "side": "sell",
                                "qty": int(sell_qty),
                                "price": float(actual_close.iloc[-1]),
                                "notional": est_sell_dollars,
                                "equity_after": account_snapshot.get("equity"),
                                "cash_after": account_snapshot.get("cash"),
                                "buying_power_after": account_snapshot.get("buying_power"),
                                "demo_mode": False,
                                "order_result": result,
                            }
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to sell: {e}")

        st.markdown("---")
        signal_emoji = "BUY" if sig.iloc[-1] == 1 else "FLAT"
        st.write(f"**✨ Latest Signal:** {signal_emoji}")
        st.caption(f"Last updated {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC}")

        auto_trade_result = {"status": "disabled"}
        if auto_trade_enabled:
            if auto_trade_trigger == "SMA Crossover":
                auto_trade_result = maybe_execute_auto_trade(
                    symbol=symbol,
                    sig=sig,
                    close_prices=actual_close,
                    equity=equity,
                    risk_fraction=auto_trade_risk_fraction,
                    enabled=True,
                    demo_mode=demo_mode,
                )
            elif auto_trade_trigger == "ML Forecast":
                auto_trade_result = maybe_execute_auto_trade_ml(
                    symbol=symbol,
                    forecast_change_pct=forecast_change,
                    equity=equity,
                    risk_fraction=auto_trade_risk_fraction,
                    enabled=True,
                    demo_mode=demo_mode,
                )
            else:  # Either
                # Try SMA first, then ML; both functions have their own duplicate protection
                auto_trade_result = maybe_execute_auto_trade(
                    symbol=symbol,
                    sig=sig,
                    close_prices=actual_close,
                    equity=equity,
                    risk_fraction=auto_trade_risk_fraction,
                    enabled=True,
                    demo_mode=demo_mode,
                )
                if auto_trade_result.get("status") != "executed":
                    auto_trade_result = maybe_execute_auto_trade_ml(
                        symbol=symbol,
                        forecast_change_pct=forecast_change,
                        equity=equity,
                        risk_fraction=auto_trade_risk_fraction,
                        enabled=True,
                        demo_mode=demo_mode,
                    )
        if auto_trade_result.get("status") == "executed":
            entry = auto_trade_result["entry"]
            mode_label = "Demo" if entry.get("demo_mode") else "Live"
            st.success(
                f"{mode_label} auto-trade executed: {entry.get('action')} {entry.get('qty')} shares of {entry.get('symbol')} (${entry.get('qty') * entry.get('price', 0):,.2f})"
            )
            st.rerun()
        elif auto_trade_enabled and auto_trade_result.get("status") == "skipped":
            st.caption(f"Auto-trade check: {auto_trade_result.get('reason', 'skipped')}")

        st.subheader("📊 Trade Metrics")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Bought Shares", f"{trade_summary['buy_qty']:.0f}")
        metric_cols[1].metric("Sold Shares", f"{trade_summary['sell_qty']:.0f}")
        metric_cols[2].metric("Avg Buy Price", f"${trade_summary['avg_buy_price']:,.2f}" if pd.notna(trade_summary["avg_buy_price"]) else "n/a")
        metric_cols[3].metric("Avg Sell Price", f"${trade_summary['avg_sell_price']:,.2f}" if pd.notna(trade_summary["avg_sell_price"]) else "n/a")

        metric_cols_2 = st.columns(4)
        metric_cols_2[0].metric("Bought $", f"${trade_summary['buy_notional']:,.2f}")
        metric_cols_2[1].metric("Sold $", f"${trade_summary['sell_notional']:,.2f}")
        metric_cols_2[2].metric("Open Position Shares", f"{trade_summary['position_qty']:.0f}")
        metric_cols_2[3].metric("Open Position Value", f"${trade_summary['open_position_value']:,.2f}")

        metric_cols_3 = st.columns(2)
        metric_cols_3[0].metric("Estimated Total P/L", f"${trade_summary['estimated_total_pnl']:,.2f}")
        metric_cols_3[1].metric("Estimated Total Return", f"{trade_summary['estimated_total_return_pct']:.2f}%")
        st.caption(
            "Estimated total P/L is mark-to-market: sells plus the current value of any open shares, minus logged buys."
        )

    with money_tab:
        st.subheader("💹 Money Chart")
        trade_log_df = load_trade_log(limit=500)
        if not demo_mode:
            live_money_df = _fetch_alpaca_portfolio_history(period="1M", timeframe="1D")
        else:
            live_money_df = pd.DataFrame()

        if not live_money_df.empty:
            money_fig = go.Figure()
            money_fig.add_trace(
                go.Scatter(
                    x=live_money_df["timestamp"],
                    y=live_money_df["equity"],
                    mode="lines+markers",
                    name="Equity",
                    line=dict(color="#2ca02c", width=2),
                )
            )
            if "profit_loss" in live_money_df.columns and live_money_df["profit_loss"].notna().any():
                money_fig.add_trace(
                    go.Scatter(
                        x=live_money_df["timestamp"],
                        y=live_money_df["profit_loss"],
                        mode="lines",
                        name="Profit/Loss",
                        yaxis="y2",
                        line=dict(color="#ff7f0e", width=1.5, dash="dot"),
                    )
                )
                money_fig.update_layout(
                    yaxis2=dict(
                        title="Profit/Loss",
                        overlaying="y",
                        side="right",
                        showgrid=False,
                    )
                )
            money_fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Equity ($)",
                hovermode="x unified",
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(money_fig, use_container_width=True)
        else:
            fallback_money_df = _money_history_from_trade_log(trade_log_df, equity, cash)
            fallback_fig = go.Figure()
            fallback_fig.add_trace(
                go.Scatter(
                    x=fallback_money_df["timestamp"],
                    y=fallback_money_df["equity"],
                    mode="lines+markers",
                    name="Equity",
                    line=dict(color="#2ca02c", width=2),
                )
            )
            if "cash" in fallback_money_df.columns and fallback_money_df["cash"].notna().any():
                fallback_fig.add_trace(
                    go.Scatter(
                        x=fallback_money_df["timestamp"],
                        y=fallback_money_df["cash"],
                        mode="lines+markers",
                        name="Cash",
                        line=dict(color="#1f77b4", width=1.8),
                    )
                )
            fallback_fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Balance ($)",
                hovermode="x unified",
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fallback_fig, use_container_width=True)

        st.subheader("🧾 Recent Trade Log")
        if trade_log_df.empty:
            st.info("No trades logged yet.")
        else:
            display_cols = [
                col
                for col in [
                    "timestamp_utc",
                    "source",
                    "symbol",
                    "action",
                    "qty",
                    "price",
                    "notional",
                    "equity_after",
                    "cash_after",
                    "demo_mode",
                ]
                if col in trade_log_df.columns
            ]
            st.dataframe(trade_log_df.head(100)[display_cols], use_container_width=True)
    
except Exception as e:
    st.error(f" Error loading market data: {e}")
    st.info("Please check if the symbol is valid and try again.")

st.markdown("---")

# Debug info (only show in sidebar if needed)
if st.sidebar.checkbox("🐞 Show Debug Info", value=False):
    try:
        r = requests.get("https://paper-api.alpaca.markets/v2/orders", headers={
            "APCA-API-KEY-ID": ALPACA_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET,
        })
        st.sidebar.json(r.json())
    except Exception as e:
        st.sidebar.error(f"Failed to fetch orders: {e}")
