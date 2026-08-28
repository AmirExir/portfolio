import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import contextlib
import io
import logging
import os
import sys
import requests
import base64
import json
import glob
import hashlib
from zoneinfo import ZoneInfo

import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go


# Add the parent directory to the path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import agent modules, with fallback handling
try:
    from agent.data import get_ohlcv
    from agent.earnings import (
        fetch_yfinance_earnings_payload,
        interpret_earnings_payload,
        load_earnings_payload_file,
        us_equity_trading_sessions,
    )
    from agent.forecast import backtest_forecasts, compare_forecast_models, forecast_close_prices
    from agent.strategy import sma_crossover
    from agent.backtest import simple_vector_backtest
    from agent.broker import (
        BrokerError,
        BrokerPortfolioSnapshot,
        LiquidationError,
        cancel_open_orders,
        close_all_positions,
        get_account,
        get_latest_quote,
        get_portfolio_snapshot,
        get_positions,
        size_capped_buy_notional,
        submit_order,
    )
    from agent.policy import smart_policy_decision, smart_policy_report
    from agent.portfolio import (
        PortfolioConstraints,
        allocate_target_weights,
        constrain_incremental_target,
    )
    from agent.risk import target_position_qty
    from forecast_cache import (
        MODEL_RESULT_CACHE_VERSION,
        cache_payload_fresh,
        forecast_cache_path,
        forecast_result_from_dict,
        frame_fingerprint,
        model_result_cache_path,
        snapshots_to_ranking_frame,
        snapshot_from_model_results,
        select_model_name,
    )
    from patterns import scan_patterns
    from report_freshness import (
        newest_json_payload_candidate,
        newest_text_report_candidate,
        newest_timestamped_path,
        report_path_is_newer,
    )
except ImportError as e:
    st.error(f" Failed to import agent modules: {e}")
    st.info("Please ensure the 'agent' folder exists in the same directory as this app.")
    st.stop()


def get_secret(name, default=None):
    """Read Streamlit secrets when configured, otherwise fall back to env/default."""
    try:
        value = st.secrets.get(name, os.getenv(name, default))
    except Exception:
        value = os.getenv(name, default)
    if isinstance(value, str):
        return value.strip()
    return value


def get_first_secret(names, default=None):
    for name in names:
        value = get_secret(name)
        if value not in (None, ""):
            return value
    return default


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


def _alpaca_position_qty(symbol: str) -> float:
    """Return the exact broker quantity; never truncate fractional shares."""
    try:
        position = get_positions().get(str(symbol).strip().upper())
        return float(position.qty) if position is not None else 0.0
    except BrokerError:
        return 0.0


def _fetch_alpaca_positions(
    equity: float,
) -> tuple[dict[str, float], dict[str, float], str | None]:
    """Compatibility wrapper returning risk-reserved weights and exact qty."""
    if equity <= 0.0:
        return {}, {}, "Alpaca positions or equity are unavailable"
    try:
        snapshot = get_portfolio_snapshot(float(equity))
        quantities = {
            symbol: float(position.qty)
            for symbol, position in snapshot.positions.items()
        }
        return dict(snapshot.risk_weights), quantities, None
    except Exception as exc:
        return {}, {}, (
            f"could not verify broker positions ({type(exc).__name__})"
        )


def _fetch_alpaca_execution_snapshot(
    equity: float,
) -> tuple[BrokerPortfolioSnapshot | None, str | None]:
    """Fetch positions, entry prices, and pending-buy risk as one snapshot."""
    if equity <= 0.0:
        return None, "Alpaca positions or equity are unavailable"
    try:
        return get_portfolio_snapshot(float(equity)), None
    except Exception as exc:
        return None, (
            f"could not verify broker positions and open orders "
            f"({type(exc).__name__})"
        )


def _fetch_alpaca_position_weights(
    equity: float,
) -> tuple[dict[str, float], str | None]:
    weights, _quantities, error = _fetch_alpaca_positions(equity)
    return weights, error


def _current_portfolio_drawdown() -> float | None:
    history = _fetch_alpaca_portfolio_history(period="3M", timeframe="1D")
    if history.empty or "equity" not in history.columns:
        return None
    values = pd.to_numeric(history["equity"], errors="coerce").dropna()
    if values.empty or float(values.max()) <= 0.0:
        return None
    return min(float(values.iloc[-1] / values.cummax().iloc[-1] - 1.0), 0.0)


def _position_covariance(
    symbols: set[str],
    selected_symbol: str,
    selected_df: pd.DataFrame,
) -> pd.DataFrame | None:
    returns = {}
    for candidate in sorted(symbols):
        try:
            frame = (
                selected_df
                if candidate == selected_symbol
                else get_ohlcv(candidate, 365)
            )
            close = frame["close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            daily = (
                pd.to_numeric(close, errors="coerce")
                .dropna()
                .sort_index()
                .pct_change()
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if len(daily) < 60:
                return None
            returns[candidate] = daily.rename(candidate)
        except Exception:
            return None
    aligned = pd.concat(returns.values(), axis=1).dropna(how="any").tail(252)
    if len(aligned) < 60:
        return None
    return aligned.cov() * 252.0


def constrain_auto_order_target(
    *,
    symbol: str,
    desired_target: float,
    current_weights: dict[str, float],
    current_drawdown: float,
    price_history: pd.DataFrame,
    require_covariance: bool,
) -> tuple[float, dict, str | None]:
    """Return a broker-position-aware target for one risk-increasing order."""
    normalized = str(symbol).strip().upper()
    constraints = PortfolioConstraints(
        gross_limit=1.0,
        cash_reserve=0.15,
        max_name_weight=0.05,
        max_sector_weight=0.20,
        max_cluster_weight=0.15,
        max_annual_volatility=0.15,
        max_turnover=0.20,
        drawdown_circuit_breaker=0.10,
    )
    if normalized not in PORTFOLIO_SECTORS or normalized not in PORTFOLIO_CLUSTERS:
        return 0.0, {}, "symbol lacks a verified sector or correlation-cluster mapping"
    unknown_holdings = sorted(
        holding
        for holding, weight in current_weights.items()
        if weight > 0.0
        and (
            holding not in PORTFOLIO_SECTORS
            or holding not in PORTFOLIO_CLUSTERS
        )
    )
    if unknown_holdings:
        return 0.0, {}, (
            "cannot verify portfolio classifications for "
            + ", ".join(unknown_holdings)
        )
    if current_drawdown <= -constraints.drawdown_circuit_breaker:
        return 0.0, {"circuit_breaker_triggered": True}, "drawdown circuit breaker is active"

    current = {
        str(name).upper(): max(float(weight), 0.0)
        for name, weight in current_weights.items()
        if float(weight) > 0.0
    }
    symbols = set(current) | {normalized}
    covariance = _position_covariance(symbols, normalized, price_history)
    if covariance is None and require_covariance:
        return 0.0, {}, "portfolio covariance could not be verified"
    try:
        authorization = constrain_incremental_target(
            normalized,
            desired_target,
            current_weights=current,
            sectors={name: PORTFOLIO_SECTORS[name] for name in symbols},
            correlation_clusters={
                name: PORTFOLIO_CLUSTERS[name] for name in symbols
            },
            annual_covariance=covariance,
            current_drawdown=current_drawdown,
            constraints=constraints,
        )
    except ValueError as exc:
        return 0.0, {}, str(exc)

    diagnostics = {
        "current_target": authorization.current_target,
        "desired_target": authorization.desired_target,
        "allowed_target": authorization.allowed_target,
        "gross_before": authorization.gross_before,
        "portfolio_gross_target": authorization.gross_after,
        "portfolio_cash_target": float(
            max(1.0 - authorization.gross_after, 0.0)
        ),
        "annualized_volatility": authorization.annualized_volatility,
        "binding_constraints": list(
            authorization.allocation.binding_constraints
        ),
        "warnings": list(authorization.allocation.warnings),
    }
    return authorization.allowed_target, diagnostics, None


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
    return {
        "status": "skipped",
        "reason": (
            "standalone SMA auto trading is retired; only the calibrated "
            "Smart Policy may submit automatic orders"
        ),
    }

    # Retained below temporarily for historical log compatibility. This path
    # is intentionally unreachable so an old UI caller cannot bypass policy
    # uncertainty and portfolio gates.
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
    return {
        "status": "skipped",
        "reason": (
            "raw ML-direction auto trading is retired; only the calibrated "
            "Smart Policy may submit automatic orders"
        ),
    }

    # Retained below temporarily for historical log compatibility. This path
    # is intentionally unreachable so an old UI caller cannot bypass policy
    # uncertainty and portfolio gates.
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


def _execute_drawdown_liquidation(current_drawdown: float) -> dict:
    """Submit and log portfolio-wide broker closes for the circuit breaker."""
    try:
        liquidation = close_all_positions()
    except LiquidationError as exc:
        accepted_orders = tuple(
            order
            for order in exc.accepted_orders
            if order.get("broker_accepted") is True
        )
        if not accepted_orders:
            return {
                "status": "skipped",
                "reason": (
                    "drawdown circuit breaker could not obtain an accepted "
                    "close order for any broker position"
                ),
            }
        accepted_symbols = tuple(
            sorted({str(order["symbol"]) for order in accepted_orders})
        )
        partial_entry = {
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            ),
            "source": "auto_smart_policy_drawdown_circuit_breaker",
            "symbol": "PORTFOLIO",
            "action": "PARTIAL_LIQUIDATION",
            "side": "close",
            "qty": float(
                sum(
                    abs(exc.positions[symbol].qty)
                    for symbol in accepted_symbols
                )
            ),
            "price": None,
            "notional": float(
                sum(
                    abs(exc.positions[symbol].market_value)
                    for symbol in accepted_symbols
                )
            ),
            "demo_mode": False,
            "portfolio_drawdown": float(current_drawdown),
            "position_symbols": list(accepted_symbols),
            "failed_position_symbols": sorted(exc.failures),
            "order_result": list(accepted_orders),
            "cancellation_result": dict(exc.cancellations),
        }
        _append_trade_log(partial_entry)
        state = _read_trade_state()
        state["_portfolio_drawdown_circuit_breaker"] = {
            "status": "partial",
            "last_timestamp": partial_entry["timestamp_utc"],
            "drawdown": float(current_drawdown),
            "position_symbols": list(accepted_symbols),
            "failed_position_symbols": sorted(exc.failures),
            "accepted_order_ids": [
                order["id"] for order in accepted_orders
            ],
        }
        _write_trade_state(state)
        return {
            "status": "partial",
            "reason": (
                "drawdown circuit breaker accepted "
                f"{len(accepted_orders)} close order(s), but failed for "
                f"{', '.join(sorted(exc.failures))}"
            ),
            "entry": partial_entry,
        }
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": (
                "portfolio drawdown liquidation was not fully accepted "
                f"({type(exc).__name__})"
            ),
        }
    if (
        not liquidation.position_symbols
        or len(liquidation.accepted_orders)
        != len(liquidation.position_symbols)
    ):
        return {
            "status": "skipped",
            "reason": (
                "drawdown circuit breaker is active; no verified broker "
                "positions required liquidation"
            ),
        }
    if not all(
        order.get("broker_accepted") is True
        for order in liquidation.accepted_orders
    ):
        return {
            "status": "skipped",
            "reason": "one or more portfolio liquidation orders were not accepted",
        }
    liquidation_entry = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        ),
        "source": "auto_smart_policy_drawdown_circuit_breaker",
        "symbol": "PORTFOLIO",
        "action": "LIQUIDATE_ALL",
        "side": "close",
        "qty": float(
            sum(abs(position.qty) for position in liquidation.positions.values())
        ),
        "price": None,
        "notional": float(
            sum(
                abs(position.market_value)
                for position in liquidation.positions.values()
            )
        ),
        "demo_mode": False,
        "portfolio_drawdown": float(current_drawdown),
        "position_symbols": list(liquidation.position_symbols),
        "order_result": list(liquidation.accepted_orders),
        "cancellation_result": dict(liquidation.cancellations),
    }
    _append_trade_log(liquidation_entry)
    state = _read_trade_state()
    state["_portfolio_drawdown_circuit_breaker"] = {
        "status": "complete",
        "last_timestamp": liquidation_entry["timestamp_utc"],
        "drawdown": float(current_drawdown),
        "position_symbols": list(liquidation.position_symbols),
        "accepted_order_ids": [
            order["id"] for order in liquidation.accepted_orders
        ],
    }
    _write_trade_state(state)
    return {
        "status": "executed",
        "entry": liquidation_entry,
        "liquidation": liquidation,
    }


def maybe_execute_auto_trade_smart(
    symbol: str,
    df: pd.DataFrame,
    sig: pd.Series,
    equity: float,
    risk_fraction: float,
    enabled: bool,
    demo_mode: bool,
    trade_summary: dict | None = None,
    forecast_result=None,
    model_results: dict | None = None,
    earnings_context: dict | None = None,
) -> dict:
    if not enabled:
        return {"status": "disabled"}

    normalized_symbol = str(symbol).strip().upper()
    trade_summary = trade_summary or {}
    broker_snapshot: BrokerPortfolioSnapshot | None = None
    if demo_mode:
        current_qty = int(max(float(trade_summary.get("position_qty", 0.0) or 0.0), 0.0))
        latest_close = _latest_value(_clean_series(df, "close"))
        current_weights = {
            normalized_symbol: current_qty
            * latest_close
            / float(equity)
        }
        current_drawdown = 0.0
        position_error = None
        drawdown_error = None
        avg_entry_price = trade_summary.get("avg_buy_price")
    else:
        try:
            verified_account = get_account()
            verified_equity = float(verified_account.get("equity", 0.0))
            if not np.isfinite(verified_equity) or verified_equity <= 0.0:
                raise ValueError("broker equity is not positive")
            equity = verified_equity
        except Exception as exc:
            return {
                "status": "skipped",
                "reason": (
                    "broker account equity could not be verified "
                    f"({type(exc).__name__})"
                ),
            }
        current_drawdown_value = _current_portfolio_drawdown()
        if current_drawdown_value is None:
            current_drawdown = 0.0
            drawdown_error = "portfolio drawdown could not be verified"
        else:
            current_drawdown = current_drawdown_value
            drawdown_error = None
        if current_drawdown <= -0.10:
            # Do not depend on quote/open-buy valuation before a safety exit.
            return _execute_drawdown_liquidation(current_drawdown)
        broker_snapshot, position_error = _fetch_alpaca_execution_snapshot(
            float(equity)
        )
        if position_error or broker_snapshot is None:
            return {
                "status": "skipped",
                "reason": position_error or "broker snapshot is unavailable",
            }
        current_weights = dict(broker_snapshot.risk_weights)
        broker_position = broker_snapshot.positions.get(normalized_symbol)
        current_qty = (
            max(float(broker_position.qty), 0.0)
            if broker_position is not None
            else 0.0
        )
        avg_entry_price = (
            float(broker_position.avg_entry_price)
            if broker_position is not None
            else None
        )

    forecast_metrics = dict(
        getattr(forecast_result, "metrics", {})
        if forecast_result is not None
        else {}
    )
    forecast_metrics.update(earnings_policy_metrics(earnings_context))
    decision = smart_policy_decision(
        df=df,
        equity=equity,
        risk_fraction=risk_fraction,
        current_qty=current_qty,
        avg_entry_price=avg_entry_price,
        signal=sig,
        forecast_metrics=forecast_metrics,
        model_results=model_results,
    )

    if decision.action == "HOLD" or not decision.side or decision.qty <= 0:
        return {
            "status": "skipped",
            "reason": decision.reason,
            "decision": decision,
        }

    order_qty = float(decision.qty)
    order_notional: float | None = None
    execution_price = float(decision.last_price)
    portfolio_diagnostics: dict = {}
    portfolio_target_fraction = float(decision.target_position_fraction)
    if decision.side == "buy":
        if position_error or drawdown_error:
            return {
                "status": "skipped",
                "reason": position_error or drawdown_error,
                "decision": decision,
            }
        (
            portfolio_target_fraction,
            portfolio_diagnostics,
            portfolio_error,
        ) = constrain_auto_order_target(
            symbol=symbol,
            desired_target=decision.target_position_fraction,
            current_weights=current_weights,
            current_drawdown=current_drawdown,
            price_history=df,
            require_covariance=not demo_mode,
        )
        if portfolio_error:
            return {
                "status": "skipped",
                "reason": portfolio_error,
                "decision": decision,
            }
        if demo_mode:
            constrained_target_qty = int(
                (float(equity) * portfolio_target_fraction)
                // float(decision.last_price)
            )
            order_qty = float(
                min(
                    int(decision.qty),
                    max(constrained_target_qty - int(current_qty), 0),
                )
            )
        else:
            try:
                quote = get_latest_quote(normalized_symbol)
                sizing = size_capped_buy_notional(
                    equity=float(equity),
                    allowed_target_weight=portfolio_target_fraction,
                    reserved_symbol_weight=float(
                        current_weights.get(normalized_symbol, 0.0)
                    ),
                    quote=quote,
                    requested_qty=float(decision.qty),
                )
            except Exception as exc:
                return {
                    "status": "skipped",
                    "reason": (
                        "current broker ask could not be verified "
                        f"({type(exc).__name__})"
                    ),
                    "decision": decision,
                }
            order_notional = float(sizing.notional)
            order_qty = float(sizing.estimated_qty)
            execution_price = float(sizing.conservative_ask)
            portfolio_diagnostics.update(
                {
                    "verified_ask": float(quote.ask_price),
                    "quote_timestamp": quote.timestamp.isoformat(),
                    "conservative_ask": float(sizing.conservative_ask),
                    "available_target_notional": float(
                        sizing.available_target_notional
                    ),
                    "submitted_notional": order_notional,
                    "pending_buy_weight_reserved": float(
                        broker_snapshot.pending_buy_weights.get(
                            normalized_symbol,
                            0.0,
                        )
                        if broker_snapshot is not None
                        else 0.0
                    ),
                }
            )
        if order_qty <= 0.0 or (
            not demo_mode and (order_notional is None or order_notional < 1.0)
        ):
            return {
                "status": "skipped",
                "reason": "portfolio constraints allow no additional exposure",
                "decision": decision,
                "portfolio_diagnostics": portfolio_diagnostics,
            }
    elif not demo_mode:
        # A zero-target sale must include any fractional remainder recorded by
        # the broker instead of truncating to the policy's integer quantity.
        order_qty = (
            float(current_qty)
            if decision.target_position_fraction <= 0.0
            else min(float(decision.qty), float(current_qty))
        )
        if order_qty <= 0.0:
            return {
                "status": "skipped",
                "reason": "broker reports no position available to sell",
                "decision": decision,
            }

    signal_ts = pd.to_datetime(df.index[-1]).strftime("%Y-%m-%d") if not df.empty else dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    action_size = (
        f"${order_notional:.2f}"
        if order_notional is not None
        else f"{order_qty:.9f}"
    )
    action_key = f"smart:{signal_ts}:{decision.action}:{action_size}:{decision.target_qty}:{decision.score:.2f}"
    state = _read_trade_state()
    symbol_state = state.get(normalized_symbol, {})
    if symbol_state.get("last_smart_action_key") == action_key:
        return {"status": "skipped", "reason": "already executed smart policy action for this signal", "decision": decision}

    if demo_mode:
        order_qty = int(order_qty)
        order_result = {"demo": True, "symbol": normalized_symbol, "side": decision.side, "qty": order_qty}
    else:
        try:
            if order_notional is not None:
                order_result = submit_order(
                    normalized_symbol,
                    side=decision.side,
                    notional=order_notional,
                )
            else:
                order_result = submit_order(
                    normalized_symbol,
                    qty=order_qty,
                    side=decision.side,
                )
        except Exception as exc:
            return {
                "status": "skipped",
                "reason": (
                    "broker did not accept the order "
                    f"({type(exc).__name__})"
                ),
                "decision": decision,
                "portfolio_diagnostics": portfolio_diagnostics,
            }
        if order_result.get("broker_accepted") is not True:
            return {
                "status": "skipped",
                "reason": "broker response did not confirm an accepted order",
                "decision": decision,
                "portfolio_diagnostics": portfolio_diagnostics,
            }

    account_snapshot = {"equity": float(equity), "cash": None, "buying_power": None}
    if not demo_mode:
        live_snapshot = _fetch_live_account_snapshot()
        if live_snapshot:
            account_snapshot = live_snapshot

    log_entry = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "source": "auto_smart_policy",
        "symbol": normalized_symbol,
        "action": decision.action,
        "side": decision.side,
        "qty": float(order_qty),
        "price": round(float(execution_price), 4),
        "notional": (
            float(order_notional)
            if order_notional is not None
            else round(float(order_qty) * float(execution_price), 2)
        ),
        "risk_fraction": float(risk_fraction),
        "policy_score": float(decision.score),
        "policy_reason": decision.reason,
        "target_qty": int(decision.target_qty),
        "target_position_fraction": float(decision.target_position_fraction),
        "portfolio_target_position_fraction": float(portfolio_target_fraction),
        "demo_mode": bool(demo_mode),
        "equity_after": account_snapshot.get("equity"),
        "cash_after": account_snapshot.get("cash"),
        "buying_power_after": account_snapshot.get("buying_power"),
        "order_result": order_result,
        "policy_diagnostics": decision.diagnostics,
        "portfolio_diagnostics": portfolio_diagnostics,
    }
    _append_trade_log(log_entry)

    state[normalized_symbol] = symbol_state if symbol_state else {}
    state[normalized_symbol]["last_smart_action_key"] = action_key
    state[normalized_symbol]["last_smart_action"] = decision.action
    state[normalized_symbol]["last_smart_score"] = float(decision.score)
    state[normalized_symbol]["last_smart_timestamp"] = log_entry["timestamp_utc"]
    _write_trade_state(state)

    return {"status": "executed", "entry": log_entry, "decision": decision}


APP_BUILD = "2026-06-18 professional-market-ui-v1"
MARKET_BLUE = "#0b2f4f"
MARKET_CYAN = "#00a3c7"
MARKET_GREEN = "#16a34a"
MARKET_ORANGE = "#f59e0b"
MARKET_RED = "#dc2626"
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}
HEATMAP_MAX_SYMBOLS = 40


def selected_non_crypto_symbols(symbols) -> tuple[str, ...]:
    selected = []
    seen = set()
    for symbol in symbols or []:
        symbol_text = str(symbol).strip().upper()
        if not symbol_text or symbol_text.endswith("-USD") or symbol_text in seen:
            continue
        selected.append(symbol_text)
        seen.add(symbol_text)
        if len(selected) >= HEATMAP_MAX_SYMBOLS:
            break
    return tuple(selected)


def count_non_crypto_symbols(symbols) -> int:
    return len(
        {
            str(symbol).strip().upper()
            for symbol in symbols or []
            if str(symbol).strip() and not str(symbol).strip().upper().endswith("-USD")
        }
    )


def inject_market_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --market-blue: #0b2f4f;
            --market-cyan: #00a3c7;
            --market-green: #16a34a;
            --market-orange: #f59e0b;
            --market-red: #dc2626;
            --market-slate: #334155;
            --market-muted: #64748b;
            --market-border: #e2e8f0;
            --market-bg: #f6f8fb;
            --market-panel: #ffffff;
        }
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > .main {
            background: var(--market-bg);
            color: #0f172a;
        }
        [data-testid="stHeader"] {
            background: rgba(246,248,251,0.92);
            border-bottom: 1px solid rgba(226,232,240,0.75);
        }
        .block-container {
            max-width: 1480px;
            padding-top: 1.15rem;
            padding-bottom: 2.5rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #071f36 0%, #0b2f4f 54%, #123b63 100%);
            border-right: 1px solid rgba(255,255,255,0.10);
        }
        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] select {
            color: #0f172a !important;
        }
        .market-hero {
            padding: 1.45rem 1.55rem;
            border-radius: 24px;
            background:
                radial-gradient(circle at 85% 0%, rgba(0,163,199,0.26), transparent 30%),
                linear-gradient(135deg, #061a2e 0%, #0b2f4f 54%, #123b63 100%);
            color: #ffffff;
            border: 1px solid rgba(255,255,255,0.12);
            box-shadow: 0 24px 60px rgba(15, 23, 42, 0.20);
            margin-bottom: 1rem;
        }
        .market-hero h1 {
            margin: 0;
            color: #ffffff;
            font-size: 2.45rem;
            line-height: 1.05;
            letter-spacing: -0.045em;
        }
        .market-hero p {
            margin: 0.65rem 0 0 0;
            color: #dbeafe;
            max-width: 980px;
            font-size: 1.02rem;
        }
        .market-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }
        .market-badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.22);
            background: rgba(255,255,255,0.10);
            color: #f8fafc;
            padding: 0.32rem 0.72rem;
            font-weight: 750;
            font-size: 0.82rem;
        }
        .section-title {
            color: var(--market-blue);
            margin: 0.25rem 0 0.2rem 0;
            letter-spacing: -0.035em;
        }
        .section-subtitle {
            color: var(--market-muted);
            margin: 0 0 0.85rem 0;
            font-size: 0.95rem;
        }
        div[data-testid="stMetric"] {
            background: var(--market-panel);
            border: 1px solid var(--market-border);
            border-top: 4px solid var(--market-cyan);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
        }
        div[data-testid="stMetric"] label {
            color: var(--market-muted) !important;
            text-transform: uppercase;
            letter-spacing: 0.075em;
            font-size: 0.72rem !important;
            font-weight: 800;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.45rem;
            border-bottom: 1px solid var(--market-border);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.5rem 0.9rem;
        }
        .stTabs [data-baseweb="tab"] p {
            color: var(--market-slate) !important;
            font-weight: 750;
        }
        .stTabs [aria-selected="true"] {
            background: #e0f2fe;
        }
        .stTabs [aria-selected="true"] p {
            color: var(--market-blue) !important;
            font-weight: 850;
        }
        .stButton > button,
        .stLinkButton > a {
            background: #111827 !important;
            color: #ffffff !important;
            border: 1px solid #1f2937 !important;
            border-radius: 10px !important;
            font-weight: 750 !important;
            min-height: 2.5rem;
        }
        .stButton > button:hover,
        .stLinkButton > a:hover {
            background: var(--market-blue) !important;
            border-color: var(--market-cyan) !important;
        }
        h1, h2, h3 {
            color: #0f172a;
            letter-spacing: -0.035em;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--market-border);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_market_hero() -> None:
    st.markdown(
        f"""
        <div class="market-hero">
            <h1>Market Intelligence Agent</h1>
            <p>
                Professional multi-asset dashboard for scheduled ML forecasts, risk-aware policy signals,
                market heatmaps, paper-trading controls, and portfolio monitoring.
            </p>
            <div class="market-badges">
                <span class="market-badge">Model-confirmed signals</span>
                <span class="market-badge">Smart policy overlay</span>
                <span class="market-badge">Stocks + ETFs + crypto</span>
                <span class="market-badge">Paper trading controls</span>
                <span class="market-badge">Build {APP_BUILD}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"<h2 class='section-title'>{title}</h2>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='section-subtitle'>{subtitle}</div>", unsafe_allow_html=True)


def apply_market_chart_layout(
    fig: go.Figure,
    title: str | None = None,
    height: int | None = None,
    margin: dict | None = None,
) -> go.Figure:
    layout_kwargs = {
        "template": "plotly_white",
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "font": dict(color="#0f172a"),
        "hovermode": "closest",
        "margin": margin or dict(l=40, r=30, t=70 if title else 35, b=45),
    }
    if title:
        layout_kwargs["title"] = dict(text=title, x=0.01, xanchor="left", font=dict(size=20, color=MARKET_BLUE))
    if height:
        layout_kwargs["height"] = height
    fig.update_layout(**layout_kwargs)
    cartesian_types = {
        "bar",
        "box",
        "candlestick",
        "heatmap",
        "histogram",
        "ohlc",
        "scatter",
        "scattergl",
        "scatterpolar",
        "violin",
    }
    if any(getattr(trace, "type", "") in cartesian_types for trace in fig.data):
        fig.update_xaxes(showgrid=True, gridcolor="#eef2f7", linecolor="#cbd5e1", tickfont=dict(color="#0f172a"))
        fig.update_yaxes(
            showgrid=True,
            gridcolor="#eef2f7",
            zerolinecolor="#cbd5e1",
            linecolor="#cbd5e1",
            tickfont=dict(color="#0f172a"),
        )
    return fig


def render_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


st.set_page_config(
    page_title="Market Intelligence Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_market_css()
render_market_hero()

SYMBOL_OPTIONS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "RIOT", "AVGO", "STX", "SPCX",
    "ORCL", "NFLX", "JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MU", "WDC",
    "UNH", "LLY", "JNJ", "WMT", "PG", "KO", "HD", "PEP",
    "DAL", "UAL", "AAL", "LUV", "XOM", "CVX", "COP", "OXY", "SLB", "EOG",
    "SPY", "VOO", "QQQ", "IWM", "DIA", "GLD", "SLV", "USO", "TLT",
    "XLK", "XLF", "XLE", "XLV", "XLY", "SNDK",
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "BNB-USD", "AVAX-USD",
    "ORCA-USD", "PNUT-USD", "DOGE-USD", "SHIB-USD", "FLOKI-USD", "PEPE-USD",
    "ZEC-USD", "COMP5692-USD", "HYPE32196-USD", "MNT27075-USD", "UNI7083-USD", "ENA-USD", "DOT-USD",
]
DEFAULT_FORECAST_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "RIOT", "AVGO", "MU", "WDC", "STX", "SNDK", "SPCX",
    "ORCL", "NFLX", "JPM", "BAC", "WFC", "C", "GS", "MS", "V",
    "UNH", "LLY", "JNJ", "WMT", "PG", "KO", "HD", "PEP",
    "DAL", "UAL", "AAL", "LUV", "XOM", "CVX", "COP", "OXY", "SLB", "EOG",
    "SPY", "VOO", "QQQ", "IWM", "DIA", "GLD", "SLV", "USO", "TLT",
    "XLK", "XLF", "XLE", "XLV", "XLY",
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "BNB-USD", "AVAX-USD",
    "ORCA-USD", "PNUT-USD", "DOGE-USD", "SHIB-USD", "FLOKI-USD", "PEPE-USD",
    "ZEC-USD", "COMP5692-USD", "HYPE32196-USD", "MNT27075-USD", "UNI7083-USD", "ENA-USD", "DOT-USD",
]
PORTFOLIO_SECTORS = {
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
    **{symbol: "Crypto" for symbol in DEFAULT_FORECAST_SYMBOLS if symbol.endswith("-USD")},
}
PORTFOLIO_CLUSTERS = {
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
    **{symbol: "Crypto" for symbol in DEFAULT_FORECAST_SYMBOLS if symbol.endswith("-USD")},
}
SHORT_TERM_SIGNAL_HORIZONS = (1,)
SCHEDULED_ONE_DAY_FALLBACK_LIMIT = 10
SYMBOL_LABELS = {
    "MU": "Micron (MU)",
    "WDC": "Western Digital (WDC)",
    "STX": "Seagate (STX)",
    "SNDK": "SanDisk (SNDK)",
    "SPCX": "SPCX ETF (SPCX)",
    "AVGO": "Broadcom (AVGO)",
    "RIOT": "Riot Platforms (RIOT)",
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
    "FLOKI-USD": "Floki (FLOKI)",
    "PEPE-USD": "Pepe (PEPE)",
    "ZEC-USD": "Zcash (ZEC)",
    "COMP5692-USD": "Compound (COMP)",
    "HYPE32196-USD": "Hyperliquid (HYPE)",
    "MNT27075-USD": "Mantle (MNT)",
    "UNI7083-USD": "Uniswap (UNI)",
    "ENA-USD": "Ethena (ENA)",
    "DOT-USD": "Polkadot (DOT)",
}
MARKET_CONTEXT_TICKERS = [
    "SPY", "VOO", "QQQ", "IWM", "DIA", "^VIX",
    "XLK", "XLF", "XLE", "XLV", "XLY",
    "TLT", "GLD", "SLV", "USO", "AVGO",
]


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


def _ticker_tuple(tickers) -> tuple[str, ...]:
    if isinstance(tickers, str):
        tickers = [tickers]
    cleaned = []
    for ticker in tickers:
        ticker = str(ticker).strip().upper()
        if ticker and ticker not in cleaned:
            cleaned.append(ticker)
    return tuple(cleaned)


def _yf_download(tickers, **kwargs) -> pd.DataFrame:
    ticker_list = _ticker_tuple(tickers)
    if not ticker_list:
        return pd.DataFrame()

    target = ticker_list[0] if len(ticker_list) == 1 else list(ticker_list)
    kwargs.setdefault("progress", False)
    kwargs.setdefault("threads", False)
    try:
        with _quiet_yfinance_output():
            return yf.download(target, **kwargs)
    except Exception:
        return pd.DataFrame()


def _close_price_frame(raw: pd.DataFrame, tickers) -> pd.DataFrame:
    ticker_list = _ticker_tuple(tickers)
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw, pd.Series):
        close = raw.to_frame(name=ticker_list[0] if ticker_list else raw.name)
    elif isinstance(raw.columns, pd.MultiIndex):
        column_levels = raw.columns
        if "Close" in column_levels.get_level_values(0):
            close = raw["Close"]
        elif "Close" in column_levels.get_level_values(column_levels.nlevels - 1):
            close = raw.xs("Close", axis=1, level=column_levels.nlevels - 1)
        else:
            return pd.DataFrame()
    elif "Close" in raw.columns:
        close = raw[["Close"]].copy()
        if len(ticker_list) == 1:
            close.columns = [ticker_list[0]]
    else:
        close = raw.copy()

    if isinstance(close, pd.Series):
        close = close.to_frame(name=ticker_list[0] if ticker_list else close.name)

    close = close.copy()
    close.columns = [str(column[-1] if isinstance(column, tuple) else column) for column in close.columns]
    requested_columns = [ticker for ticker in ticker_list if ticker in close.columns]
    if requested_columns:
        close = close[requested_columns]

    close = close.apply(pd.to_numeric, errors="coerce")
    close = close.dropna(axis=1, how="all").dropna(how="all")
    return close.loc[:, ~close.columns.duplicated()]


def _download_close_prices(tickers, **kwargs) -> pd.DataFrame:
    ticker_list = _ticker_tuple(tickers)
    raw = _yf_download(ticker_list, **kwargs)
    return _close_price_frame(raw, ticker_list)


def _price_change_pct(close: pd.DataFrame, lookback_days: int) -> pd.Series:
    if close.empty:
        return pd.Series(dtype=float)

    close = close.sort_index().ffill().dropna(axis=1, how="all").dropna(how="all")
    if close.empty:
        return pd.Series(dtype=float)

    if close.shape[0] <= lookback_days:
        base = close.iloc[0]
    else:
        base = close.iloc[-(lookback_days + 1)]
    last = close.iloc[-1]
    pct_change = (last - base) / base * 100.0
    pct_change = pct_change.replace([np.inf, -np.inf], np.nan)
    return pct_change.dropna()


def _yf_market_cap(ticker: str) -> float:
    try:
        ticker_obj = yf.Ticker(ticker)
        with _quiet_yfinance_output():
            fast_info = getattr(ticker_obj, "fast_info", {}) or {}
            market_cap = fast_info.get("market_cap") if hasattr(fast_info, "get") else None
            if not market_cap:
                market_cap = (ticker_obj.info or {}).get("marketCap")
        market_cap = float(market_cap or 1.0)
        if np.isnan(market_cap) or market_cap <= 0:
            return 1.0
        return market_cap
    except Exception:
        return 1.0


@st.cache_data(ttl=900, show_spinner=False)
def load_ohlcv(symbol: str, history_days: int) -> pd.DataFrame:
    return get_ohlcv(symbol, history_days)


def _load_market_context_data(history_days: int) -> pd.DataFrame:
    start = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=history_days * 2)).strftime("%Y-%m-%d")
    raw = _yf_download(MARKET_CONTEXT_TICKERS, start=start, interval="1d")
    close = _close_price_frame(raw, MARKET_CONTEXT_TICKERS)
    if close.empty:
        return pd.DataFrame()

    close = close.rename(columns={ticker: f"context_{ticker}" for ticker in close.columns})
    return close.dropna(how="all").ffill()


@st.cache_data(ttl=300, show_spinner=False)
def load_latest_earnings_interpretation(
    symbol: str,
    observed_sessions: tuple[str, ...] = (),
) -> dict:
    """Fetch and interpret the latest already-published earnings result."""
    normalized = str(symbol).strip().upper()
    if not normalized or normalized.endswith("-USD"):
        return {"available": False, "error_code": "not_an_equity"}
    as_of = dt.datetime.now(dt.timezone.utc)
    external = load_earnings_payload_file(
        normalized,
        get_secret(
            "MARKET_AGENT_EARNINGS_PAYLOAD_DIR",
            os.getenv("MARKET_AGENT_EARNINGS_PAYLOAD_DIR", ""),
        ),
    )
    if external.available:
        fetched = external
    elif external.error_code in {
        "external_earnings_payload_not_configured",
        "external_earnings_payload_missing",
    }:
        fetched = fetch_yfinance_earnings_payload(
            normalized,
            as_of=as_of,
        )
    else:
        return {
            "available": False,
            "event_flag": 0,
            "event_score": 0.0,
            "confidence": 0.0,
            "policy_eligible": False,
            "error_code": (
                external.error_code
                or "external_earnings_payload_invalid"
            ),
        }
    if not fetched.available or not fetched.payload:
        return {
            "available": False,
            "event_flag": 0,
            "event_score": 0.0,
            "confidence": 0.0,
            "policy_eligible": False,
            "error_code": fetched.error_code or "earnings_unavailable",
        }
    sessions = us_equity_trading_sessions(
        as_of=as_of,
        observed_sessions=observed_sessions,
    )
    interpretation = interpret_earnings_payload(
        fetched.payload,
        as_of=as_of,
        trading_sessions=sessions,
    )
    local_now = as_of.astimezone(ZoneInfo("America/New_York"))
    session_dates = tuple(sorted(sessions))
    if (
        local_now.date() in session_dates
        and local_now.time() < dt.time(16, 0)
    ):
        current_session = local_now.date()
    else:
        current_session = next(
            session for session in session_dates
            if session > local_now.date()
        )
    policy_signal = interpret_earnings_payload(
        fetched.payload,
        as_of=as_of,
        decision_session=current_session,
        trading_sessions=sessions,
    )
    quality_flags = list(
        dict.fromkeys(
            [
                *interpretation.data_quality_flags,
                *policy_signal.data_quality_flags,
            ]
        )
    )
    return {
        "available": True,
        "symbol": interpretation.symbol,
        # These fields alone enter the live policy. A same-session event that
        # is next-session-effective remains visible below but is zeroed here.
        "event_flag": int(policy_signal.event_flag),
        "event_score": float(policy_signal.event_score),
        "confidence": float(policy_signal.confidence),
        "policy_eligible": bool(policy_signal.policy_eligible),
        "policy_decision_session": current_session.isoformat(),
        # Immediate human-readable interpretation is kept separate from the
        # execution-safe policy context.
        "interpretation_event_flag": int(interpretation.event_flag),
        "interpretation_event_score": float(interpretation.event_score),
        "interpretation_confidence": float(interpretation.confidence),
        "summary": interpretation.summary,
        "outcome": interpretation.outcome,
        "reported_at": (
            interpretation.reported_at.isoformat()
            if interpretation.reported_at
            else ""
        ),
        "effective_session": (
            interpretation.effective_session.isoformat()
            if interpretation.effective_session
            else ""
        ),
        "age_sessions": interpretation.age_sessions,
        "is_stale": bool(interpretation.is_stale),
        "blockers": list(policy_signal.blockers),
        "data_quality_flags": quality_flags,
        "calendar_source": "observed_sessions_plus_us_equity_rules",
        "eps_surprise_pct": interpretation.eps_surprise_pct,
        "revenue_surprise_pct": interpretation.revenue_surprise_pct,
        "guidance_surprise_pct": interpretation.guidance_surprise_pct,
    }


def earnings_policy_metrics(earnings_context: dict | None) -> dict:
    context = earnings_context or {}
    if not context.get("available"):
        return {
            "earnings_event_flag": False,
            "earnings_event_score": 0.0,
            "earnings_confidence": 0.0,
            "earnings_policy_eligible": False,
            "earnings_calendar_source": str(
                context.get("calendar_source", "") or ""
            ),
            "earnings_error_code": str(
                context.get("error_code", "earnings_unavailable") or ""
            ),
        }
    return {
        "earnings_event_flag": bool(context.get("event_flag", 0)),
        "earnings_event_score": float(context.get("event_score", 0.0) or 0.0),
        "earnings_confidence": float(context.get("confidence", 0.0) or 0.0),
        "earnings_summary": str(context.get("summary", "") or ""),
        "earnings_outcome": str(context.get("outcome", "") or ""),
        "earnings_effective_session": str(
            context.get("effective_session", "") or ""
        ),
        "earnings_reported_at": str(
            context.get("reported_at", "") or ""
        ),
        "earnings_is_stale": bool(context.get("is_stale", False)),
        "earnings_policy_eligible": bool(
            context.get("policy_eligible", False)
        ),
        "earnings_blockers": list(context.get("blockers", []) or []),
        "earnings_data_quality_flags": list(
            context.get("data_quality_flags", []) or []
        ),
        "earnings_calendar_source": str(
            context.get("calendar_source", "") or ""
        ),
        "earnings_error_code": str(context.get("error_code", "") or ""),
    }


@st.cache_data(ttl=900, show_spinner=False)
def load_market_context(history_days: int) -> pd.DataFrame:
    return _load_market_context_data(history_days)


@st.cache_data(ttl=900, show_spinner=False)
def load_stock_heatmap_data(tickers: tuple[str, ...], lookback_days: int) -> pd.DataFrame:
    period_days = lookback_days + 3
    hist = _download_close_prices(tickers, period=f"{period_days}d", interval="1d")
    pct_change = _price_change_pct(hist, lookback_days)
    if pct_change.empty:
        return pd.DataFrame(columns=["Ticker", "Percent Change", "Market Cap", "Label"])

    market_caps = {ticker: _yf_market_cap(ticker) for ticker in pct_change.index}

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
    watch_hist = _download_close_prices(watchlist_tickers, period=f"{period_days}d", interval="1d")
    watch_pct_change = _price_change_pct(watch_hist, lookback_days)
    if watch_pct_change.empty:
        return pd.DataFrame(columns=["Ticker", "Label", "Percent Change", "Weight", "Display"])

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
    crypto_hist = _download_close_prices(crypto_tickers, period=f"{period_days}d", interval="1d")
    crypto_pct_change = _price_change_pct(crypto_hist, lookback_days)
    if crypto_pct_change.empty:
        return pd.DataFrame(columns=["Crypto", "Percent Change", "Market Cap", "Symbol", "Label"])

    crypto_market_caps = {ticker: _yf_market_cap(ticker) for ticker in crypto_pct_change.index}

    crypto_df = pd.DataFrame(
        {
            "Crypto": crypto_pct_change.index,
            "Percent Change": crypto_pct_change.values,
            "Market Cap": [crypto_market_caps[ticker] for ticker in crypto_pct_change.index],
        }
    )
    crypto_df["Symbol"] = crypto_df["Crypto"].apply(ticker_short_label)
    crypto_df["Label"] = crypto_df.apply(
        lambda row: f"{row['Symbol']}\n{row['Percent Change']:.2f}%",
        axis=1,
    )
    return crypto_df


def ticker_label(symbol: str) -> str:
    return SYMBOL_LABELS.get(symbol, symbol)


def ticker_short_label(symbol: str) -> str:
    label = ticker_label(symbol)
    if label.endswith(")") and "(" in label:
        return label.rsplit("(", 1)[-1].rstrip(")")
    return str(symbol).replace("-USD", "")


NEWS_SUMMARY_DIR = "news_summaries"
NEWS_SUMMARY_PREFIXES = ("summary_", "news_summary_")


def _summary_timestamp(filename: str) -> str:
    for prefix in NEWS_SUMMARY_PREFIXES:
        if filename.startswith(prefix) and filename.endswith(".txt"):
            return filename.removeprefix(prefix).removesuffix(".txt")
    return ""


def is_timestamped_summary(filename: str) -> bool:
    timestamp = _summary_timestamp(filename)
    if not timestamp:
        return False
    return len(timestamp) >= 11 and timestamp[:4].isdigit() and timestamp[4] == "-" and "T" in timestamp


def summary_timestamp_caption(filename: str) -> str | None:
    try:
        timestamp = _summary_timestamp(filename)
        date_part, time_part = timestamp.split("T", 1)
        time_part = time_part.replace("Z", "").replace("-", ":")
        return f"📅 Last updated: {date_part} at {time_part} UTC"
    except Exception:
        return None


def summary_age_hours(filename: str, now: dt.datetime | None = None) -> float | None:
    timestamp = _summary_timestamp(filename)
    try:
        published_at = dt.datetime.strptime(timestamp, "%Y-%m-%dT%H-%M-%S-%fZ").replace(
            tzinfo=dt.timezone.utc
        )
    except (TypeError, ValueError):
        return None

    current_time = now or dt.datetime.now(dt.timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=dt.timezone.utc)
    return max((current_time.astimezone(dt.timezone.utc) - published_at).total_seconds() / 3600, 0.0)


def render_summary_timestamp(filename: str) -> None:
    caption = summary_timestamp_caption(filename)
    if caption:
        st.caption(caption)

    age_hours = summary_age_hours(filename)
    if age_hours is not None and age_hours > 36:
        age_days = age_hours / 24
        st.warning(
            f"This published summary is {age_days:.1f} days old. "
            "The automated research workflow may not have published a newer report yet."
        )


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
    if "Symbol" in display_df.columns:
        display_df["Symbol"] = display_df["Symbol"].map(ticker_label)

    display_columns = [
        "Rank",
        "Horizon",
        "Sequence Model",
        "Symbol",
        "Smart Policy",
        "Policy Score",
        "Policy Target %",
        "Unconstrained Policy Target %",
        "Portfolio Gross %",
        "Portfolio Cash %",
        "Portfolio Binding Constraints",
        "Signal Tier",
        "Reliability",
        "Model Call",
        "Primary Pattern",
        "Selected Model",
        "Last Price",
        "Forecast Price",
        "Forecast Return %",
        "Ridge Return %",
        "XGBoost Return %",
        "Neural Net Return %",
        "LSTM Return %",
        "Transformer Return %",
        "Ensemble Return %",
        "Probability Up %",
        "Directional Probability %",
        "Model Edge %",
        "Signal Quality",
        "Expected Error %",
        "Validation MAE %",
        "Direction Hit Rate %",
        "Policy Trend Score",
        "Policy Forecast Score",
        "Policy Momentum Score",
        "Policy Earnings Score",
        "Earnings Event",
        "Earnings Score",
        "Earnings Confidence %",
        "Earnings Summary",
        "Earnings Reported At",
        "Earnings Effective Session",
        "Earnings Policy Eligible",
        "Earnings Stale",
        "Earnings Blockers",
        "Earnings Data Quality",
        "Earnings Calendar Source",
        "Earnings Error",
        "RL Mode",
        "RL Policy Return %",
        "RL Shadow Action",
        "RL Shadow Target %",
        "RL Shadow Visits",
        "Policy Volatility %",
        "Score",
    ]
    display_columns = [column for column in display_columns if column in display_df.columns]
    return display_df[display_columns].style.format(
        {
            "Rank": "{:.0f}",
            "Last Price": "${:,.2f}",
            "Forecast Price": "${:,.2f}",
            "Forecast Return %": "{:+.2f}%",
            "Policy Score": "{:+.2f}",
            "Policy Target %": "{:.1f}%",
            "Unconstrained Policy Target %": "{:.1f}%",
            "Portfolio Gross %": "{:.1f}%",
            "Portfolio Cash %": "{:.1f}%",
            "Ridge Return %": "{:+.2f}%",
            "XGBoost Return %": "{:+.2f}%",
            "Neural Net Return %": "{:+.2f}%",
            "LSTM Return %": "{:+.2f}%",
            "Transformer Return %": "{:+.2f}%",
            "Ensemble Return %": "{:+.2f}%",
            "Probability Up %": "{:.1f}%",
            "Directional Probability %": "{:.1f}%",
            "Model Edge %": "{:.1f}%",
            "Expected Error %": "{:.2f}%",
            "Validation MAE %": "{:.2f}%",
            "Direction Hit Rate %": "{:.1f}%",
            "Policy Trend Score": "{:+.2f}",
            "Policy Forecast Score": "{:+.2f}",
            "Policy Momentum Score": "{:+.2f}",
            "Policy Earnings Score": "{:+.2f}",
            "Earnings Score": "{:+.2f}",
            "Earnings Confidence %": "{:.1f}%",
            "RL Policy Return %": "{:+.2f}%",
            "RL Shadow Target %": "{:.1f}%",
            "RL Shadow Visits": "{:.0f}",
            "Policy Volatility %": "{:.1f}%",
            "Score": "{:+.2f}",
        }
    ).hide(axis="index")


def scheduled_short_horizon_table(
    short_horizon_reports: list[dict],
    allowed_horizons: tuple[int, ...] = SHORT_TERM_SIGNAL_HORIZONS,
    max_rows_per_horizon: int = 10,
) -> pd.DataFrame:
    frames = []
    for short_report in short_horizon_reports or []:
        horizon = short_report.get("horizon_days")
        if allowed_horizons and int(horizon or 0) not in allowed_horizons:
            continue

        rows = short_report.get("rows") or []
        if not rows:
            rows = (short_report.get("top_buys") or []) + (short_report.get("top_sells") or [])
        if not rows:
            continue

        horizon_label = f"{int(horizon)}D" if horizon else "Short"
        horizon_df = pd.DataFrame(rows)
        if horizon_df.empty:
            continue
        horizon_df = sort_by_policy_score(horizon_df, ascending=False)
        horizon_df = horizon_df.head(max_rows_per_horizon).copy()
        horizon_df.insert(0, "Horizon", horizon_label)
        horizon_df.insert(1, "Sequence Model", short_report.get("sequence_model") or "")
        frames.append(horizon_df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def scheduled_one_day_fallback_symbols(payload: dict | None, limit: int = SCHEDULED_ONE_DAY_FALLBACK_LIMIT) -> tuple[str, ...]:
    payload = payload or {}
    rows = pd.DataFrame(payload.get("rows", []))
    symbols: list[str] = []

    if not rows.empty and "Symbol" in rows.columns:
        rows = sort_by_policy_score(rows, ascending=False)
        for symbol in rows["Symbol"].dropna().astype(str):
            if symbol and symbol not in symbols:
                symbols.append(symbol)
            if len(symbols) >= limit:
                return tuple(symbols)

    for symbol in payload.get("symbols", []) or []:
        symbol = str(symbol)
        if symbol and symbol not in symbols:
            symbols.append(symbol)
        if len(symbols) >= limit:
            break

    return tuple(symbols)


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
                "Mode": "Shadow only" if result.metrics.get("shadow_mode") else "Live eligible",
                "Status": "OK",
                "Forecast Price": float(result.forecast["forecast_close"].iloc[-1]),
                "Forecast Return %": result.metrics.get("forecast_change_pct", 0.0),
                "Probability Up %": result.metrics.get("probability_up_pct", 50.0),
                "Model Edge %": max(confidence - 50.0, 0.0),
                "Signal Quality": signal_quality(confidence),
                "Expected Error %": result.metrics.get("expected_error_pct", 0.0),
                "Validation MAE %": result.metrics.get("holdout_mae_pct", np.nan),
                "Validation Direction %": result.metrics.get("holdout_direction_accuracy", np.nan),
                "Validation Samples": result.metrics.get("holdout_samples", np.nan),
                "Purged OOS": result.metrics.get("validation_is_oos", False),
                "Score": result.metrics.get("forecast_score", np.nan),
            }
        )
    return pd.DataFrame(rows)


def policy_score_from_row(row: pd.Series | dict) -> float:
    try:
        if "Policy Score" in row and pd.notna(row["Policy Score"]):
            return float(row["Policy Score"])
        if "Score" in row and pd.notna(row["Score"]):
            return float(row["Score"])
    except Exception:
        pass
    return 0.0


def sort_by_policy_score(frame: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
    if frame.empty:
        return frame
    score_column = "Policy Score" if "Policy Score" in frame.columns else "Score"
    if score_column not in frame.columns:
        return frame
    sortable = frame.copy()
    sortable["_policy_sort"] = pd.to_numeric(sortable[score_column], errors="coerce").fillna(0.0)
    return sortable.sort_values("_policy_sort", ascending=ascending).drop(columns=["_policy_sort"])


def build_smart_policy_metadata(
    df: pd.DataFrame,
    model_results: dict,
    primary_model: str,
    risk_fraction: float = 0.05,
    earnings_context: dict | None = None,
) -> dict:
    selected_result = model_results.get(primary_model)
    forecast_metrics = dict(
        getattr(selected_result, "metrics", {})
        if selected_result is not None
        else {}
    )
    forecast_metrics.update(earnings_policy_metrics(earnings_context))
    try:
        signal = sma_crossover(df, 20, 50)
    except Exception:
        signal = None
    try:
        return smart_policy_report(
            df=df,
            risk_fraction=risk_fraction,
            signal=signal,
            forecast_metrics=forecast_metrics,
            model_results=model_results,
        )
    except Exception as policy_error:
        return {
            "policy_call": "Policy Unavailable",
            "policy_score": 0.0,
            "policy_target_pct": 0.0,
            "policy_reason": str(policy_error),
        }


def enrich_snapshot_with_smart_policy_for_app(
    snapshot: dict,
    df: pd.DataFrame,
    primary_model_choice: str,
    risk_fraction: float = 0.05,
    earnings_context: dict | None = None,
) -> dict:
    model_results = _model_results_from_snapshot(snapshot)
    primary_model = select_model_name(model_results, preferred=primary_model_choice)
    if not primary_model:
        return snapshot
    enriched = dict(snapshot)
    enriched["smart_policy"] = build_smart_policy_metadata(
        df,
        model_results,
        primary_model,
        risk_fraction,
        earnings_context=earnings_context,
    )
    return enriched


def short_term_signal_row(horizon_days: int, model_results: dict, primary_model_choice: str) -> dict:
    primary_model = best_model_name(
        model_results,
        preferred="" if primary_model_choice == "Best Validation" else primary_model_choice,
    )
    if not primary_model:
        raise ValueError("No usable short-term forecast.")

    result = model_results[primary_model]
    forecast = result.forecast
    if forecast is None or forecast.empty:
        raise ValueError("No usable short-term forecast.")

    metrics = result.metrics or {}
    confidence = float(metrics.get("confidence_pct", 50.0))
    expected_error = float(metrics.get("expected_error_pct", 0.0))
    forecast_return = float(metrics.get("forecast_change_pct", 0.0))
    probability_up = float(metrics.get("probability_up_pct", 50.0))
    target_date = forecast.index[-1]
    if isinstance(target_date, pd.Timestamp):
        target_date = target_date.strftime("%Y-%m-%d")

    return {
        "Horizon": f"{horizon_days} trading day" + ("" if int(horizon_days) == 1 else "s"),
        "Target Date": target_date,
        "Signal": model_call(forecast_return, expected_error, confidence),
        "Selected Model": primary_model,
        "Forecast Price": float(forecast["forecast_close"].iloc[-1]),
        "Forecast Return %": forecast_return,
        "Probability Up %": probability_up,
        "Model Edge %": max(confidence - 50.0, 0.0),
        "Signal Quality": signal_quality(confidence),
        "Expected Error %": expected_error,
        "Score": float(metrics.get("forecast_score", np.nan)),
    }


def best_model_name(model_results: dict, preferred: str = "Ensemble") -> str:
    """Choose a live-eligible forecast; shadow policies can never be primary."""
    return select_model_name(
        model_results,
        preferred=preferred or "Best Validation",
    )


def result_metric(model_results: dict, model_name: str, metric_name: str, default=np.nan):
    result = model_results.get(model_name)
    if result is None:
        return default
    return result.metrics.get(metric_name, default)

def best_ranked_symbol(ranking_table: pd.DataFrame, fallback: str = "AAPL") -> str:
    if ranking_table.empty or "Symbol" not in ranking_table.columns:
        return fallback

    if "Policy Score" in ranking_table.columns:
        default_targets = pd.Series(0.0, index=ranking_table.index)
        policy_candidates = ranking_table[
            pd.to_numeric(ranking_table.get("Policy Target %", default_targets), errors="coerce").fillna(0) > 0
        ].copy()
        if not policy_candidates.empty:
            policy_candidates["_policy_sort"] = pd.to_numeric(policy_candidates["Policy Score"], errors="coerce").fillna(0)
            return policy_candidates.sort_values("_policy_sort", ascending=False).iloc[0]["Symbol"]

    positive_candidates = ranking_table[pd.to_numeric(ranking_table.get("Forecast Return %", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0]
    if not positive_candidates.empty and "Score" in positive_candidates.columns:
        return sort_by_policy_score(positive_candidates, ascending=False).iloc[0]["Symbol"]

    if "Score" in ranking_table.columns:
        return sort_by_policy_score(ranking_table, ascending=False).iloc[0]["Symbol"]

    return ranking_table.iloc[0]["Symbol"]


def constrain_ranking_portfolio(
    ranking_table: pd.DataFrame,
    close_history: dict[str, pd.Series],
) -> tuple[pd.DataFrame, list[str]]:
    """Apply one portfolio-wide risk budget to independently scored symbols."""
    if ranking_table.empty or "Symbol" not in ranking_table.columns:
        return ranking_table, []

    constrained = ranking_table.copy()
    symbols = constrained["Symbol"].astype(str).str.strip().str.upper()
    targets = pd.to_numeric(
        constrained.get("Policy Target %", pd.Series(0.0, index=constrained.index)),
        errors="coerce",
    ).fillna(0.0)
    proposed = {
        symbol: max(float(target), 0.0) / 100.0
        for symbol, target in zip(symbols, targets)
        if symbol and float(target) > 0.0
    }
    constrained["Unconstrained Policy Target %"] = targets
    if not proposed:
        constrained["Portfolio Gross %"] = 0.0
        constrained["Portfolio Cash %"] = 100.0
        constrained["Portfolio Binding Constraints"] = ""
        return constrained, []

    covariance = None
    return_series = {}
    for symbol in proposed:
        close = close_history.get(symbol)
        if close is None:
            continue
        clean = pd.to_numeric(close, errors="coerce").dropna().sort_index()
        daily = clean.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if len(daily) >= 20:
            return_series[symbol] = daily.rename(symbol)
    if len(return_series) == len(proposed):
        aligned = pd.concat(return_series.values(), axis=1).dropna(how="any").tail(252)
        if len(aligned) >= 20:
            covariance = aligned.cov() * 252.0

    sectors = {symbol: PORTFOLIO_SECTORS[symbol] for symbol in proposed if symbol in PORTFOLIO_SECTORS}
    clusters = {symbol: PORTFOLIO_CLUSTERS[symbol] for symbol in proposed if symbol in PORTFOLIO_CLUSTERS}
    constraints = PortfolioConstraints(
        gross_limit=1.0,
        cash_reserve=0.15,
        max_name_weight=0.05,
        max_sector_weight=0.20,
        max_cluster_weight=0.15,
        max_annual_volatility=0.15,
        max_turnover=0.20,
        drawdown_circuit_breaker=0.10,
    )
    try:
        allocation = allocate_target_weights(
            proposed,
            sectors=sectors,
            correlation_clusters=clusters,
            annual_covariance=covariance,
            current_drawdown=0.0,
            constraints=constraints,
        )
    except ValueError as covariance_error:
        allocation = allocate_target_weights(
            proposed,
            sectors=sectors,
            correlation_clusters=clusters,
            annual_covariance=None,
            current_drawdown=0.0,
            constraints=constraints,
        )
        warnings = [f"Portfolio covariance was unavailable: {covariance_error}"]
    else:
        warnings = []

    final_targets = allocation.target_weights
    constrained["Policy Target %"] = [
        float(final_targets.get(symbol, 0.0) * 100.0)
        for symbol in symbols
    ]
    constrained["Portfolio Gross %"] = allocation.gross_exposure * 100.0
    constrained["Portfolio Cash %"] = allocation.cash_weight * 100.0
    constrained["Portfolio Binding Constraints"] = ", ".join(allocation.binding_constraints)
    warnings.extend(allocation.warnings)
    return constrained, list(dict.fromkeys(warnings))


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
    fig = apply_market_chart_layout(
        fig,
        title=f"Quick Price View: {ticker_label(symbol)}",
        height=460,
        margin=dict(l=55, r=25, t=72, b=45),
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Price")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
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


def _model_results_from_snapshot(snapshot: dict) -> dict:
    return {
        model_name: forecast_result_from_dict(model_payload)
        for model_name, model_payload in (snapshot.get("models") or {}).items()
    }


def _load_model_result_cache(cache_path: str, max_age_days: float = 7.0) -> dict:
    payload = _load_forecast_payload(cache_path)
    if not cache_payload_fresh(payload, max_age_days=max_age_days):
        return {}
    snapshot = payload.get("snapshot") or {}
    if not snapshot:
        return {}
    return payload


def _save_model_result_cache(cache_path: str, payload: dict) -> None:
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
    sequence_model_choice: str,
    include_rl_policy: bool,
    ranking_symbols: tuple[str, ...] | None = None,
    cache_buster: int = 0,
    earnings_context_json: str = "",
) -> dict:
    ranking_symbols = ranking_symbols or tuple()
    cache_path = forecast_cache_path(
        _forecast_reports_dir(),
        (symbol,),
        history_days,
        forecast_horizon,
        forecast_lookback,
        forecast_alpha,
        optimize_forecast_model,
        use_market_context,
        sequence_model_choice,
        include_rl_policy,
    )
    df = get_ohlcv(symbol, history_days)
    try:
        earnings_context = (
            json.loads(earnings_context_json)
            if earnings_context_json
            else {}
        )
    except json.JSONDecodeError:
        earnings_context = {}
    context_df = _load_market_context_data(history_days) if use_market_context else pd.DataFrame()
    data_fingerprint = frame_fingerprint(df)
    context_fingerprint = frame_fingerprint(context_df) if use_market_context else None
    if earnings_context:
        event_hash = hashlib.sha256(
            json.dumps(
                earnings_context,
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()[:16]
        context_fingerprint = {
            "market": context_fingerprint,
            "earnings_event_hash": event_hash,
            "earnings_effective_session": earnings_context.get(
                "effective_session"
            ),
        }
    model_cache_path = model_result_cache_path(
        _forecast_reports_dir(),
        symbol,
        history_days,
        forecast_horizon,
        forecast_lookback,
        forecast_alpha,
        optimize_forecast_model,
        use_market_context,
        sequence_model_choice,
        data_fingerprint,
        context_fingerprint,
        include_rl_policy,
    )

    if cache_buster == 0:
        cached_payload = _load_model_result_cache(str(model_cache_path))
        if cached_payload:
            model_results = _model_results_from_snapshot(cached_payload["snapshot"])
            if model_results:
                return model_results

    model_results = compare_forecast_models(
        df,
        horizon_days=forecast_horizon,
        lookback_window=forecast_lookback,
        ridge_alpha=forecast_alpha,
        optimize_model=optimize_forecast_model,
        context_df=context_df,
        sequence_model=sequence_model_choice,
        symbol=symbol,
        force_retrain=cache_buster != 0,
        include_rl=include_rl_policy,
        earnings_context=earnings_policy_metrics(earnings_context),
    )

    close = df["close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    primary_for_policy = select_model_name(model_results, preferred="Ensemble")
    smart_policy = (
        build_smart_policy_metadata(
            df,
            model_results,
            primary_for_policy,
            earnings_context=earnings_context,
        )
        if primary_for_policy
        else {}
    )
    snapshot = snapshot_from_model_results(
        symbol,
        float(close.iloc[-1]),
        model_results,
        smart_policy=smart_policy,
        as_of_session=close.index[-1],
        data_cutoff_utc=dt.datetime.now(dt.timezone.utc),
    )
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_result_cache_version": MODEL_RESULT_CACHE_VERSION,
        "symbols": [symbol],
        "history_days": history_days,
        "horizon_days": forecast_horizon,
        "lookback_window": forecast_lookback,
        "ridge_alpha": forecast_alpha,
        "optimize_model": optimize_forecast_model,
        "use_market_context": use_market_context,
        "sequence_model": sequence_model_choice,
        "include_rl_policy": include_rl_policy,
        "primary_model": "Ensemble",
        "data_fingerprint": data_fingerprint,
        "context_fingerprint": context_fingerprint,
        "snapshot": snapshot,
        "snapshots": [snapshot],
        "rows": [],
        "errors": [],
        "telegram_text": "",
    }
    _save_forecast_payload(str(cache_path), payload)
    _save_model_result_cache(str(model_cache_path), payload)
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
    sequence_model_choice: str,
    include_rl_policy: bool,
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
        sequence_model_choice,
        include_rl_policy,
    )

    ranking_context_df = _load_market_context_data(history_days) if use_market_context else pd.DataFrame()
    context_fingerprint = frame_fingerprint(ranking_context_df) if use_market_context else None
    ranking_snapshots = []
    ranking_errors: list[str] = []
    model_cache_hits = 0
    ranking_close_history: dict[str, pd.Series] = {}

    for ranking_symbol in ranking_symbols:
        try:
            ranking_df = get_ohlcv(ranking_symbol, history_days)
            ranking_close = ranking_df["close"]
            if isinstance(ranking_close, pd.DataFrame):
                ranking_close = ranking_close.iloc[:, 0]
            ranking_close = pd.to_numeric(ranking_close, errors="coerce").dropna()
            ranking_close_history[str(ranking_symbol).upper()] = ranking_close
            observed_sessions = tuple(
                sorted(
                    {
                        pd.Timestamp(value).date().isoformat()
                        for value in ranking_df.index
                    }
                )
            )
            earnings_context = load_latest_earnings_interpretation(
                ranking_symbol,
                observed_sessions,
            )
            data_fingerprint = frame_fingerprint(ranking_df)
            symbol_context_fingerprint = context_fingerprint
            if earnings_context:
                event_hash = hashlib.sha256(
                    json.dumps(
                        earnings_context,
                        sort_keys=True,
                        default=str,
                    ).encode("utf-8")
                ).hexdigest()[:16]
                symbol_context_fingerprint = {
                    "market": context_fingerprint,
                    "earnings_event_hash": event_hash,
                    "earnings_effective_session": earnings_context.get(
                        "effective_session"
                    ),
                }
            model_cache_path = model_result_cache_path(
                _forecast_reports_dir(),
                ranking_symbol,
                history_days,
                forecast_horizon,
                forecast_lookback,
                forecast_alpha,
                optimize_forecast_model,
                use_market_context,
                sequence_model_choice,
                data_fingerprint,
                symbol_context_fingerprint,
                include_rl_policy,
            )
            cached_payload = _load_model_result_cache(str(model_cache_path)) if cache_buster == 0 else {}
            if cached_payload:
                ranking_snapshots.append(
                    enrich_snapshot_with_smart_policy_for_app(
                        cached_payload["snapshot"],
                        ranking_df,
                        primary_model_choice,
                        earnings_context=earnings_context,
                    )
                )
                model_cache_hits += 1
                continue

            ranking_results = compare_forecast_models(
                ranking_df,
                horizon_days=forecast_horizon,
                lookback_window=forecast_lookback,
                ridge_alpha=forecast_alpha,
                optimize_model=optimize_forecast_model,
                context_df=ranking_context_df,
                sequence_model=sequence_model_choice,
                symbol=ranking_symbol,
                force_retrain=cache_buster != 0,
                include_rl=include_rl_policy,
                earnings_context=earnings_policy_metrics(
                    earnings_context
                ),
            )
            primary_for_policy = select_model_name(ranking_results, preferred=primary_model_choice)
            smart_policy = (
                build_smart_policy_metadata(
                    ranking_df,
                    ranking_results,
                    primary_for_policy,
                    earnings_context=earnings_context,
                )
                if primary_for_policy
                else {}
            )
            snapshot = snapshot_from_model_results(
                ranking_symbol,
                float(ranking_close.iloc[-1]),
                ranking_results,
                smart_policy=smart_policy,
                as_of_session=ranking_close.index[-1],
                data_cutoff_utc=dt.datetime.now(dt.timezone.utc),
            )
            ranking_snapshots.append(snapshot)
            _save_model_result_cache(
                str(model_cache_path),
                {
                    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "model_result_cache_version": MODEL_RESULT_CACHE_VERSION,
                    "symbol": ranking_symbol,
                    "history_days": history_days,
                    "horizon_days": forecast_horizon,
                    "lookback_window": forecast_lookback,
                    "ridge_alpha": forecast_alpha,
                    "optimize_model": optimize_forecast_model,
                    "use_market_context": use_market_context,
                    "sequence_model": sequence_model_choice,
                    "include_rl_policy": include_rl_policy,
                    "data_fingerprint": data_fingerprint,
                    "context_fingerprint": symbol_context_fingerprint,
                    "earnings_context": earnings_context,
                    "snapshot": snapshot,
                },
            )
        except Exception as ranking_error:
            ranking_errors.append(f"{ranking_symbol}: {ranking_error}")

    ranking_payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ"),
        "horizon_days": forecast_horizon,
        "primary_model": primary_model_choice,
        "sequence_model": sequence_model_choice,
        "include_rl_policy": include_rl_policy,
        "symbols": list(ranking_symbols),
        "cache_key": "",
        "model_cache_hits": model_cache_hits,
        "rows": [],
        "snapshots": ranking_snapshots,
        "errors": ranking_errors,
        "telegram_text": "",
    }
    _save_forecast_payload(str(ranking_cache_path), ranking_payload)

    ranking_table, ranking_errors_from_snapshots = snapshots_to_ranking_frame(ranking_snapshots, primary_model_choice)
    if ranking_errors_from_snapshots:
        ranking_errors.extend(ranking_errors_from_snapshots)
    ranking_table, portfolio_warnings = constrain_ranking_portfolio(
        ranking_table,
        ranking_close_history,
    )
    ranking_errors.extend(f"Portfolio: {warning}" for warning in portfolio_warnings)
    return ranking_table, ranking_errors


def report_generated_caption(report_text: str) -> str | None:
    for line in report_text.splitlines():
        if line.startswith("Generated:"):
            return f"📅 {line}"
    return None


def load_json_file_safely(path: str) -> dict | None:
    try:
        with open(path, "r") as handle:
            text = handle.read()
        if "<<<<<<<" in text or ">>>>>>>" in text or "=======" in text:
            return None
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def load_text_file_safely(path: str) -> str | None:
    try:
        with open(path, "r") as handle:
            text = handle.read()
        if "<<<<<<<" in text or ">>>>>>>" in text or "=======" in text:
            return None
        return text
    except Exception:
        return None


def newest_valid_local_ml_payload_candidate() -> tuple[dt.datetime, dict] | None:
    reports_dir = os.path.join(os.path.dirname(__file__) if __file__ else ".", "reports")
    latest_path = os.path.join(reports_dir, "ml_forecast_rankings_latest.json")
    newest_path = newest_timestamped_path(
        glob.glob(os.path.join(reports_dir, "ml_forecast_rankings_20*.json"))
    )
    candidate_paths = tuple(
        dict.fromkeys(path for path in (newest_path, latest_path) if path)
    )
    for candidate_path in candidate_paths:
        payload = load_json_file_safely(candidate_path)
        if payload and payload.get("rows"):
            selected = newest_json_payload_candidate([(candidate_path, payload)])
            if selected is not None:
                return selected[0], dict(selected[1])
    return None


def newest_valid_local_ml_payload() -> dict | None:
    candidate = newest_valid_local_ml_payload_candidate()
    return candidate[1] if candidate is not None else None


def newest_valid_local_ml_report_candidate() -> tuple[dt.datetime, str] | None:
    reports_dir = os.path.join(os.path.dirname(__file__) if __file__ else ".", "reports")
    latest_path = os.path.join(reports_dir, "ml_forecast_rankings_latest.txt")
    summary_dir = os.path.join(reports_dir, "optimization_summaries")
    newest_path = newest_timestamped_path(
        glob.glob(os.path.join(summary_dir, "ml_forecast_rankings_20*.txt"))
    )
    candidate_paths = tuple(
        dict.fromkeys(path for path in (newest_path, latest_path) if path)
    )
    for candidate_path in candidate_paths:
        report_text = load_text_file_safely(candidate_path)
        if report_text:
            selected = newest_text_report_candidate(
                [(candidate_path, report_text)]
            )
            if selected is not None:
                return selected
    return None


def newest_valid_local_ml_report() -> str | None:
    candidate = newest_valid_local_ml_report_candidate()
    return candidate[1] if candidate is not None else None


def fetch_github_directory(path: str, branch: str | None = "generated-output") -> list[dict]:
    contents_url = (
        "https://api.github.com/repos/AmirExir/portfolio/contents/"
        f"{path.strip('/')}"
    )
    if branch:
        contents_url = f"{contents_url}?ref={branch}"
    response = requests.get(
        contents_url,
        headers={
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Streamlit-Market-Agent",
        },
        timeout=10,
    )
    if response.status_code == 404:
        return []
    response.raise_for_status()
    files = response.json()
    return files if isinstance(files, list) else []


def fetch_text_from_download_url(download_url: str) -> str | None:
    try:
        response = requests.get(download_url, timeout=10)
        response.raise_for_status()
        text = response.text
        if "<<<<<<<" in text or ">>>>>>>" in text or "=======" in text:
            return None
        return text
    except Exception:
        return None


def fetch_json_from_download_url(download_url: str) -> dict | None:
    text = fetch_text_from_download_url(download_url)
    if not text:
        return None
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def newest_github_file(
    directory: str,
    prefix: str,
    suffix: str,
) -> dict | None:
    # Current publishers write validated reports to the default branch. Keep
    # the legacy branch as a fallback only so stale artifacts cannot win.
    for branch in (None, "generated-output"):
        try:
            files = fetch_github_directory(directory, branch=branch)
        except Exception:
            continue
        matches = [
            item for item in files
            if item.get("type") == "file"
            and item.get("name", "").startswith(prefix)
            and item.get("name", "").endswith(suffix)
        ]
        if matches:
            return sorted(matches, key=lambda item: item.get("name", ""), reverse=True)[0]
    return None


@st.cache_data(ttl=300)
def fetch_latest_ml_report():
    local_candidate = newest_valid_local_ml_report_candidate()
    newest_report = newest_github_file(
        "market_agent/reports/optimization_summaries",
        "ml_forecast_rankings_20",
        ".txt",
    )
    should_fetch_remote = newest_report and (
        local_candidate is None
        or report_path_is_newer(newest_report.get("name"), local_candidate[0])
    )
    if should_fetch_remote and newest_report.get("download_url"):
        text = fetch_text_from_download_url(newest_report["download_url"])
        if text:
            remote_candidate = newest_text_report_candidate(
                [(newest_report.get("name", ""), text)]
            )
            if remote_candidate and (
                local_candidate is None
                or remote_candidate[0] > local_candidate[0]
            ):
                return remote_candidate[1]

    if local_candidate is not None:
        return local_candidate[1]

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


@st.cache_data(ttl=300)
def fetch_latest_ml_payload() -> dict | None:
    local_candidate = newest_valid_local_ml_payload_candidate()
    newest_payload = newest_github_file(
        "market_agent/reports",
        "ml_forecast_rankings_20",
        ".json",
    )
    should_fetch_remote = newest_payload and (
        local_candidate is None
        or report_path_is_newer(newest_payload.get("name"), local_candidate[0])
    )
    if should_fetch_remote and newest_payload.get("download_url"):
        payload = fetch_json_from_download_url(newest_payload["download_url"])
        if payload and payload.get("rows"):
            remote_candidate = newest_json_payload_candidate(
                [(newest_payload.get("name", ""), payload)]
            )
            if remote_candidate and (
                local_candidate is None
                or remote_candidate[0] > local_candidate[0]
            ):
                return dict(remote_candidate[1])

    if local_candidate is not None:
        return local_candidate[1]

    contents_url = (
        "https://api.github.com/repos/AmirExir/portfolio/contents/"
        "market_agent/reports/ml_forecast_rankings_latest.json"
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
        return report_response.json()
    except Exception:
        return None


# --- Fetch the latest summary from GitHub ---
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_latest_summary():
    """Fetch the latest summary file from GitHub with better error handling"""
    contents_urls = [
        f"https://api.github.com/repos/AmirExir/portfolio/contents/market_agent/reports/{NEWS_SUMMARY_DIR}",
        "https://api.github.com/repos/AmirExir/portfolio/contents/market_agent/reports",
        "https://api.github.com/repos/AmirExir/portfolio/contents/market_agent",
    ]

    try:
        # Add headers to avoid rate limiting
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Streamlit-Market-Agent"
        }
        for contents_url in contents_urls:
            response = requests.get(contents_url, headers=headers, timeout=10)

            # Check for rate limiting
            if response.status_code == 403 and 'rate limit' in response.text.lower():
                return None
            if response.status_code == 404:
                continue

            response.raise_for_status()
            files = response.json()
            summary_files = [
                item for item in files
                if item.get("type") == "file" and is_timestamped_summary(item.get("name", ""))
            ]
            if summary_files:
                return summary_files
        return []
    except requests.exceptions.Timeout:
        return None
    except Exception as e:
        # Silently fail and let the fallback handle it
        return None

# Sidebar: Owner key, Alpaca creds, account summary, strategy and forecast settings
owner_key_input = st.sidebar.text_input("🔑 Enter Owner Key", type="password").strip()
OWNER_KEY = get_first_secret(["OWNER_KEY", "MARKET_AGENT_OWNER_KEY"], "")

if owner_key_input == OWNER_KEY and OWNER_KEY != "":
    demo_mode = st.sidebar.checkbox("🎭 Demo Mode", value=False, help="Toggle between live and demo mode")
    if demo_mode:
        st.sidebar.info("Demo Mode active — trades will not be executed.")
    else:
        st.sidebar.success(" Live Mode active — connected to Alpaca paper trading.")
else:
    demo_mode = True
    if OWNER_KEY:
        st.sidebar.info("Demo Mode forced ON until the owner key matches.")
    else:
        st.sidebar.info("Demo Mode forced ON for public viewers — safe demo mode.")

if OWNER_KEY == "":
    st.sidebar.caption("Owner key not configured in Streamlit secrets or environment: live mode is disabled, auto-trades run in demo simulation only.")

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
    help="Places BUY/SELL paper orders from the selected policy trigger.",
)
auto_trade_risk_fraction = st.sidebar.slider(
    "Auto-trade position size (% of equity)",
    min_value=0.03,
    max_value=0.05,
    value=0.05,
    step=0.01,
)
if auto_trade_enabled and demo_mode:
    st.sidebar.info("Auto trading enabled in demo mode: orders are simulated and still logged.")
auto_trade_trigger = "Smart Policy"
st.sidebar.caption(
    "Automatic orders use only the calibrated Smart Policy. Standalone SMA, "
    "raw ML direction, and RL-shadow outputs cannot submit orders."
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
forecast_horizon = st.sidebar.slider("Forecast horizon", min_value=1, max_value=60, value=30, step=1)
forecast_lookback = st.sidebar.slider("Lag window", min_value=5, max_value=60, value=20, step=5)
historical_test_points = st.sidebar.slider("Previous forecast test points", min_value=10, max_value=100, value=50, step=10)
optimize_forecast_model = st.sidebar.checkbox("Optimize ML model", value=True)
use_market_context = st.sidebar.checkbox("Use market context features", value=True)
primary_model_choice = st.sidebar.selectbox(
    "Primary forecast model",
    ["Ensemble", "Best Validation", "Ridge", "XGBoost", "Neural Net", "LSTM", "Transformer"],
    index=1,
)
sequence_model_label = st.sidebar.selectbox("Deep sequence model", ["Off", "LSTM", "Transformer", "Both"], index=0)
sequence_model_choice = {"Off": "off", "LSTM": "lstm", "Transformer": "transformer", "Both": "both"}[sequence_model_label]
if primary_model_choice == "LSTM":
    sequence_model_choice = "lstm"
elif primary_model_choice == "Transformer":
    sequence_model_choice = "transformer"
one_day_sequence_model_label = st.sidebar.selectbox(
    "1-day sequence model",
    ["Both", "LSTM", "Transformer", "Off", "Same as 30-day"],
    index=0,
    help="Controls the next-trading-day forecast only. Both enables LSTM and Transformer for the 1-day model.",
)
one_day_sequence_model_choice = {
    "Both": "both",
    "LSTM": "lstm",
    "Transformer": "transformer",
    "Off": "off",
    "Same as 30-day": sequence_model_choice,
}[one_day_sequence_model_label]
include_rl_policy = st.sidebar.checkbox(
    "Generate RL shadow diagnostics",
    value=True,
    help=(
        "Evaluates the policy for research only. RL is excluded from primary-model "
        "selection, ensembles, reliability grades, target weights, and orders."
    ),
)
forecast_alpha = st.sidebar.number_input(
    "Ridge regularization",
    min_value=0.1,
    max_value=100.0,
    value=10.0,
    step=0.5,
)
run_symbol_forecast = st.sidebar.checkbox(
    "Run selected-symbol ML forecast",
    value=False,
    key="run_symbol_forecast_v2",
    help="Adds the ML forecast overlay to the selected-symbol chart. Leave off for a fast public dashboard load.",
)
run_forecast_rankings = st.sidebar.checkbox(
    "Run forecast rankings",
    value=False,
    key="run_forecast_rankings_v2",
    help="Computes fresh ML rankings for the selected ticker universe. Leave off to show saved reports and quick charts immediately.",
)
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
if "forecast_work_paused" not in st.session_state:
    st.session_state["forecast_work_paused"] = False

refresh_all = st.sidebar.button("🔄 Force recalculate all rankings", help="Ignore saved ranking caches and recompute every symbol.")
refresh_selected = st.sidebar.button("🔄 Force recalculate selected stock", help="Ignore the saved selected-stock cache and recompute only the active symbol.")

if st.session_state["forecast_work_paused"]:
    st.sidebar.warning("Heavy ML forecast work is paused.")
    if st.sidebar.button("▶️ Resume ML forecasts"):
        st.session_state["forecast_work_paused"] = False
        st.rerun()
else:
    if st.sidebar.button("⏹️ Stop / pause ML forecasts", help="Prevents forecast rankings and selected-symbol ML forecasts from starting on rerun."):
        st.session_state["forecast_work_paused"] = True
        st.session_state["skip_heavy_once"] = True
        st.rerun()

if refresh_all:
    st.session_state["ranking_refresh_nonce"] += 1
    st.session_state["symbol_refresh_nonce"] += 1
    st.rerun()

if refresh_selected:
    st.session_state["symbol_refresh_nonce"] += 1
    st.rerun()

skip_heavy_once = bool(st.session_state.pop("skip_heavy_once", False))
forecast_work_paused = bool(st.session_state.get("forecast_work_paused", False))
if forecast_work_paused:
    run_forecast_rankings = False
    run_symbol_forecast = False

ranking_table = pd.DataFrame()
ranking_errors: list[str] = []
best_stock_for_analysis = "AAPL"
pattern_summary = pd.DataFrame()
pattern_details = pd.DataFrame()
pattern_errors: list[str] = []

# Top-level tabs: Portfolio Balance first, then AI-Generated Market Summary, patterns, then Market Analysis
top_portfolio_tab, top_summary_tab, top_patterns_tab, top_analysis_tab = st.tabs([
    "💼 Portfolio Balance",
    "📰 AI-Generated Market Summary",
    "🧩 Most Common Patterns",
    "📈 Market Analysis",
])

with top_summary_tab:
    render_section_header(
        "AI-Generated Market Summary",
        "Latest market-moving news summary produced by the automated research workflow.",
    )
    if st.button("🔄 Reload Published News", help="Reload the latest summary published to GitHub"):
        st.cache_data.clear()
        st.rerun()

    try:
        files = fetch_latest_summary()

        # Fallback: If GitHub API fails, try reading from local directory (for Streamlit Cloud deployment)
        if files is None:
            st.info(" GitHub API unavailable. Using local files...")
            base_local_dir = os.path.dirname(__file__) if __file__ else "."
            local_dirs = [
                os.path.join(base_local_dir, "reports", NEWS_SUMMARY_DIR),
                os.path.join(base_local_dir, "reports"),
                base_local_dir,
            ]

            try:
                local_candidates = []
                for local_dir in local_dirs:
                    if not os.path.isdir(local_dir):
                        continue
                    for filename in os.listdir(local_dir):
                        if is_timestamped_summary(filename):
                            local_candidates.append((filename, os.path.join(local_dir, filename)))

                if local_candidates:
                    # Sort and get latest
                    latest_local_file, latest_local_path = sorted(local_candidates, key=lambda item: _summary_timestamp(item[0]), reverse=True)[0]

                    with open(latest_local_path, "r") as f:
                        summary_text = f.read()

                    render_summary_timestamp(latest_local_file)

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
                st.info(f"No summary files found yet. The n8n workflow will create news_summary_*.txt files in reports/{NEWS_SUMMARY_DIR}/ on first run.")
            else:
                # Sort by name descending to get the latest (ISO format sorts correctly)
                summary_files_sorted = sorted(summary_files, key=lambda x: _summary_timestamp(x["name"]), reverse=True)
                latest_file = summary_files_sorted[0]

                # Extract timestamp from filename (format: summary_2026-05-01T11-00-28-733Z.txt)
                filename = latest_file.get("name", "")
                render_summary_timestamp(filename)

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
    render_section_header(
        "Most Common Patterns",
        "Technical pattern concentration across the selected forecast universe.",
    )
    if not selected_forecast_symbols:
        st.info("Select forecast ranking tickers to scan current patterns.")
    elif forecast_work_paused:
        st.info("Resume ML forecasts in the sidebar to calculate pattern summaries.")
    elif not run_forecast_rankings or skip_heavy_once:
        st.info("Enable Run forecast rankings to calculate pattern summaries.")
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
                pattern_fig.update_layout(yaxis=dict(categoryorder="total ascending"))
                pattern_fig = apply_market_chart_layout(
                    pattern_fig,
                    title="Most Common Patterns Across Selected Tickers",
                    height=470,
                    margin=dict(l=45, r=25, t=72, b=45),
                )
                render_chart(pattern_fig)

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
        latest_ml_payload = fetch_latest_ml_payload()
        latest_ml_report = fetch_latest_ml_report()
        if latest_ml_payload and latest_ml_payload.get("rows"):
            render_section_header(
                "Scheduled ML Forecast Rankings",
                "Latest n8n-generated model-qualified signals, smart-policy watchlists, and reliability metadata.",
            )
            generated_at = latest_ml_payload.get("generated_at")
            if generated_at:
                st.caption(f"📅 Generated: {generated_at}")
            signal_summary = latest_ml_payload.get("signal_summary") or {}
            if signal_summary:
                summary_cols = st.columns(4)
                with summary_cols[0]:
                    st.metric("Qualified Model Buys", int(signal_summary.get("model_qualified_buys", signal_summary.get("model_confirmed_buys", 0)) or 0))
                with summary_cols[1]:
                    st.metric("Qualified Model Sells/Avoids", int(signal_summary.get("model_qualified_sells", signal_summary.get("model_confirmed_sells", 0)) or 0))
                with summary_cols[2]:
                    st.metric("Policy Buy Watchlist", int(signal_summary.get("policy_watch_buys", 0) or 0))
                with summary_cols[3]:
                    st.metric("Policy Sell/Avoid Watchlist", int(signal_summary.get("policy_watch_sells", 0) or 0))

            rows_df = pd.DataFrame(latest_ml_payload.get("rows", []))
            if not rows_df.empty:
                rows_sorted = sort_by_policy_score(rows_df, ascending=False)
                st.caption(f"{int(latest_ml_payload.get('horizon_days', 30))} trading-day scheduled ranking")
                st.dataframe(format_ranking_table(rows_sorted.head(15)), use_container_width=True)

            short_horizon_df = scheduled_short_horizon_table(latest_ml_payload.get("short_horizon_reports", []))
            if not short_horizon_df.empty:
                st.markdown("**1-Day Scheduled Rankings**")
                st.caption("Separate scheduled model run for the next trading day.")
                st.dataframe(format_ranking_table(short_horizon_df), use_container_width=True)
            else:
                fallback_symbols = scheduled_one_day_fallback_symbols(latest_ml_payload)
                fallback_state = st.session_state.get("scheduled_one_day_fallback", {})
                fallback_cache_matches = (
                    fallback_state.get("source_generated_at") == generated_at
                    and fallback_state.get("sequence_model") == one_day_sequence_model_choice
                )
                fallback_rows = fallback_state.get("rows") if fallback_cache_matches else None
                fallback_errors = fallback_state.get("errors", []) if fallback_cache_matches else []

                if fallback_rows:
                    st.markdown("**1-Day Scheduled Rankings**")
                    st.caption(
                        "Manual next-trading-day model run for top symbols from "
                        "the latest scheduled report. "
                        f"Sequence model: {one_day_sequence_model_choice}."
                    )
                    fallback_df = pd.DataFrame(fallback_rows)
                    fallback_df = sort_by_policy_score(fallback_df, ascending=False)
                    if not fallback_df.empty and "Horizon" not in fallback_df.columns:
                        fallback_df.insert(0, "Horizon", "1D")
                    if not fallback_df.empty and "Sequence Model" not in fallback_df.columns:
                        fallback_df.insert(1, "Sequence Model", one_day_sequence_model_choice)
                    st.dataframe(format_ranking_table(fallback_df), use_container_width=True)
                else:
                    st.info(
                        "Routine scheduled runs publish only the bounded 30-day "
                        "report. Run the 1-day ranking here, or explicitly run "
                        "the quality/custom profile with --short-horizons 1."
                    )
                    if fallback_symbols:
                        st.caption(
                            f"Fast page fallback will rank the top {len(fallback_symbols)} symbols from the latest 30-day scheduled run: "
                            + ", ".join(fallback_symbols)
                            + f". Sequence model: {one_day_sequence_model_choice}."
                        )
                    if fallback_symbols and st.button("Run 1-Day Scheduled Ranking Now", key="run_scheduled_one_day_now"):
                        with st.spinner("Computing real 1-day ML ranking..."):
                            fallback_df, fallback_errors = cached_forecast_rankings(
                                fallback_symbols,
                                history_days,
                                1,
                                forecast_lookback,
                                forecast_alpha,
                                optimize_forecast_model,
                                use_market_context,
                                one_day_sequence_model_choice,
                                include_rl_policy,
                                primary_model_choice,
                                st.session_state["ranking_refresh_nonce"],
                            )
                        fallback_records = fallback_df.to_dict("records") if not fallback_df.empty else []
                        st.session_state["scheduled_one_day_fallback"] = {
                            "source_generated_at": generated_at,
                            "sequence_model": one_day_sequence_model_choice,
                            "rows": fallback_records,
                            "errors": fallback_errors,
                        }
                        st.markdown("**1-Day Scheduled Rankings**")
                        fallback_df = sort_by_policy_score(fallback_df, ascending=False)
                        if not fallback_df.empty and "Horizon" not in fallback_df.columns:
                            fallback_df.insert(0, "Horizon", "1D")
                        if not fallback_df.empty and "Sequence Model" not in fallback_df.columns:
                            fallback_df.insert(1, "Sequence Model", one_day_sequence_model_choice)
                        st.dataframe(format_ranking_table(fallback_df), use_container_width=True)

                if fallback_errors:
                    with st.expander("1-day scheduled ranking errors"):
                        st.write("\n".join(map(str, fallback_errors)))

            if latest_ml_report:
                with st.expander("Full scheduled report text"):
                    st.info(latest_ml_report.strip())
        elif latest_ml_report:
            render_section_header(
                "Scheduled ML Forecast Rankings",
                "Latest scheduled report text from the automation pipeline.",
            )
            report_caption = report_generated_caption(latest_ml_report)
            if report_caption:
                st.caption(report_caption)
            st.info(latest_ml_report.strip())
        else:
            st.caption(
                "Scheduled ML forecast rankings will appear here after n8n writes "
                "timestamped files under market_agent/reports/ and "
                "market_agent/reports/optimization_summaries/."
            )
    except Exception as e:
        st.caption(f"Scheduled ML forecast report unavailable: {e}")

    #  Selected non-crypto universe heatmap (market-cap weighted + labels)
    render_section_header(
        "Selected Universe Heatmap",
        "Selected non-crypto forecast symbols by timeframe, sized by available market capitalization.",
    )

    # Timeframe selector
    sp_tf = st.selectbox(
        "📉 Stock / ETF change timeframe",
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
    heatmap_symbols = selected_non_crypto_symbols(selected_forecast_symbols)
    if heatmap_symbols:
        non_crypto_count = count_non_crypto_symbols(selected_forecast_symbols)
        heatmap_note = f"Showing {len(heatmap_symbols)} selected non-crypto symbols."
        if non_crypto_count > len(heatmap_symbols):
            heatmap_note = (
                f"Showing first {len(heatmap_symbols)} of {non_crypto_count} selected non-crypto symbols "
                "for dashboard responsiveness."
            )
        if len(selected_forecast_symbols or []) > non_crypto_count:
            heatmap_note += " Crypto symbols are shown in the crypto heatmap below."
        st.caption(heatmap_note)

    try:
        if not heatmap_symbols:
            st.info("Select at least one non-crypto forecast symbol in the sidebar to render this heatmap.")
        else:
            df = load_stock_heatmap_data(heatmap_symbols, lookback_days)
            if df.empty:
                st.info("No selected-universe heatmap price data returned. Try refreshing or selecting a longer timeframe.")
            else:
                fig = px.treemap(
                    df,
                    path=["Ticker"],
                    values="Market Cap",
                    color="Percent Change",
                    color_continuous_scale="RdYlGn",
                    hover_data={"Market Cap": ":,.0f", "Percent Change": ":.2f"},
                    title=f"Selected Universe Change ({sp_tf}) - Sized by Market Cap"
                )

                fig.update_traces(text=df["Label"])
                fig = apply_market_chart_layout(fig, title=f"Selected Universe Change ({sp_tf})", height=540)
                render_chart(fig)

    except Exception as e:
        st.error(f"Error generating selected-universe heatmap: {e}")
        st.info("Please try a different timeframe or reduce the selected forecast universe.")


    # ETF / commodity watchlist requested for broader market context
    render_section_header(
        "ETF and Commodity Watchlist",
        "Broad market, rates, metals, and oil proxies for fast cross-asset context.",
    )

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
        if watch_df.empty:
            st.info("No ETF or commodity watchlist data returned.")
        else:
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
            watch_fig = apply_market_chart_layout(watch_fig, title=f"ETF and Commodity Change ({sp_tf})", height=430)
            render_chart(watch_fig)
    except Exception as e:
        st.error(f"Error generating watchlist heatmap: {e}")


    # --- Real-Time Crypto Heatmap (Market Cap Weighted + Labels)
    render_section_header(
        "Real-Time Crypto Heatmap",
        "Major crypto assets by selected timeframe, sized by estimated market capitalization.",
    )

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
        "DOGE-USD", "AVAX-USD", "TON-USD", "DOT-USD", "ZEC-USD",
        "COMP5692-USD", "HYPE32196-USD", "MNT27075-USD", "UNI7083-USD", "ENA-USD",
    ]

    try:
        crypto_df = load_crypto_heatmap_data(tuple(crypto_tickers), lookback_days)
        if crypto_df.empty:
            st.info("No crypto heatmap price data returned. Yahoo Finance may not have data for this moment.")
        else:
            crypto_fig = px.treemap(
                crypto_df,
                path=["Symbol"],
                values="Market Cap",
                color="Percent Change",
                color_continuous_scale="RdYlGn",
                hover_data={"Market Cap": ":,.0f", "Percent Change": ":.2f"},
                title=f"Crypto Change ({crypto_tf}) - Sized by Market Cap"
            )

            crypto_fig.update_traces(text=crypto_df["Label"])
            crypto_fig = apply_market_chart_layout(crypto_fig, title=f"Crypto Change ({crypto_tf})", height=540)
            render_chart(crypto_fig)

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

    render_section_header(
        "Quick Price + SMA Crossover",
        "Fast single-symbol trend view using the active moving-average settings.",
    )
    try:
        quick_df = load_ohlcv(quick_symbol, history_days)
        render_chart(build_quick_price_chart(quick_df, short_window, long_window, quick_symbol))
    except Exception as quick_error:
        quick_df = None
        st.warning(f"Quick price chart unavailable: {quick_error}")

    if run_forecast_rankings:
        render_section_header(
            "Live ML Forecast Rankings",
            "On-demand ranking run for the selected forecast universe using the sidebar model settings.",
        )

        if not selected_forecast_symbols:
            st.info("Select forecast ranking tickers in the sidebar to compute rankings.")
        elif skip_heavy_once:
            st.info("Heavy forecast work was skipped on this rerun. Re-enable or refresh rankings when ready.")
        elif forecast_work_paused:
            st.info("Resume ML forecasts in the sidebar to compute rankings.")
        else:
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
                        sequence_model_choice,
                        include_rl_policy,
                        primary_model_choice,
                        st.session_state["ranking_refresh_nonce"],
                    )
                best_stock_for_analysis = best_ranked_symbol(ranking_table, fallback="AAPL")
                if not ranking_table.empty and not pattern_details.empty:
                    ranking_pattern_columns = pattern_details[["Symbol", "Primary Pattern"]].drop_duplicates("Symbol")
                    ranking_table = ranking_table.drop(columns=["Primary Pattern"], errors="ignore").merge(
                        ranking_pattern_columns,
                        on="Symbol",
                        how="left",
                    )
            except Exception as ranking_error:
                st.warning(f"Could not compute forecast rankings: {ranking_error}")

        if not ranking_table.empty:
            if "Policy Score" in ranking_table.columns:
                policy_scores = pd.to_numeric(ranking_table["Policy Score"], errors="coerce").fillna(0.0)
                default_targets = pd.Series(0.0, index=ranking_table.index)
                policy_targets = pd.to_numeric(ranking_table.get("Policy Target %", default_targets), errors="coerce").fillna(0.0)
                buy_candidates = ranking_table[(policy_targets > 0.0) & (policy_scores > 0.0)]
                sell_candidates = ranking_table[policy_scores < 0.0]
            else:
                buy_candidates = ranking_table[ranking_table["Forecast Return %"] > 0]
                sell_candidates = ranking_table[ranking_table["Forecast Return %"] < 0]
            strongest_buy = sort_by_policy_score(buy_candidates, ascending=False).head(10)
            strongest_sell = sort_by_policy_score(sell_candidates, ascending=True).head(10)

            buy_col, sell_col = st.columns(2)
            with buy_col:
                st.markdown("**Smart Policy Buy Forecasts**")
                if strongest_buy.empty:
                    st.info("No smart-policy buy candidates.")
                else:
                    st.dataframe(format_ranking_table(strongest_buy), use_container_width=True)
            with sell_col:
                st.markdown("**Smart Policy Sell / Avoid Forecasts**")
                if strongest_sell.empty:
                    st.info("No smart-policy sell/avoid candidates.")
                else:
                    st.dataframe(format_ranking_table(strongest_sell), use_container_width=True)

            with st.expander("All forecast ranking results"):
                st.dataframe(format_ranking_table(sort_by_policy_score(ranking_table, ascending=False)), use_container_width=True)

            if ranking_errors:
                with st.expander("Symbols skipped during forecast ranking"):
                    st.write("\n".join(ranking_errors))

        st.divider()

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
    earnings_observed_sessions = tuple(
        sorted(
            {
                pd.Timestamp(value).date().isoformat()
                for value in df.index
            }
        )
    )
    earnings_context = load_latest_earnings_interpretation(
        symbol,
        earnings_observed_sessions,
    )
    
    with analysis_tab:
        render_section_header(
            "Actuals, ML Forecast, and Strategy Curve",
            "Selected-symbol price history, model forecast, actual close, and crossover backtest equity curve.",
        )
        if earnings_context.get("available"):
            score = float(
                earnings_context.get(
                    "interpretation_event_score",
                    earnings_context.get("event_score", 0.0),
                )
                or 0.0
            )
            confidence_value = float(
                earnings_context.get(
                    "interpretation_confidence",
                    earnings_context.get("confidence", 0.0),
                )
                or 0.0
            )
            st.markdown("**Latest Earnings Interpretation**")
            st.write(earnings_context.get("summary", ""))
            st.caption(
                f"Outcome: {earnings_context.get('outcome', 'unavailable')} · "
                f"event score {score:+.2f} · confidence {confidence_value:.0%} · "
                f"effective session {earnings_context.get('effective_session') or 'unknown'}. "
                f"Execution decision session "
                f"{earnings_context.get('policy_decision_session') or 'unknown'}. "
                "The interpretation is immediate; the execution policy uses it "
                "only once it is session-effective and otherwise zeros it."
            )
            if earnings_context.get("blockers"):
                st.warning(
                    "Earnings policy blockers: "
                    + ", ".join(earnings_context["blockers"])
                )
            if earnings_context.get("data_quality_flags"):
                st.caption(
                    "Earnings data-quality flags: "
                    + ", ".join(earnings_context["data_quality_flags"])
                )
        elif earnings_context.get("error_code") not in {
            "not_an_equity",
            "no_earnings_data",
            "no_reported_earnings_as_of",
        }:
            st.caption(
                "Latest earnings interpretation unavailable: "
                f"{earnings_context.get('error_code', 'unknown provider error')}."
            )
        forecast_change = 0.0
        ml_forecast_available = False
        forecast_result = None
        model_results = {}
        try:
            if forecast_work_paused:
                raise RuntimeError("ML forecast calculations are paused from the sidebar.")
            if not run_symbol_forecast:
                raise RuntimeError("Enable 'Run selected-symbol ML forecast' in the sidebar to calculate ML overlays.")
            model_results = cached_model_results(
                symbol,
                history_days,
                forecast_horizon,
                forecast_lookback,
                forecast_alpha,
                optimize_forecast_model,
                use_market_context,
                sequence_model_choice,
                include_rl_policy,
                tuple(selected_forecast_symbols),
                st.session_state["symbol_refresh_nonce"],
                earnings_context_json=json.dumps(
                    earnings_context,
                    sort_keys=True,
                    default=str,
                ),
            )
            if primary_model_choice == "Best Validation":
                primary_model = best_model_name(model_results, preferred="")
            else:
                primary_model = best_model_name(model_results, preferred=primary_model_choice)
            if not primary_model:
                raise ValueError("No forecast model produced a usable forecast.")

            forecast_result = model_results[primary_model]
            ml_forecast_available = True
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
                "LSTM": "#9467bd",
                "Transformer": "#e377c2",
                "RL Policy": "#bcbd22",
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
            price_fig = apply_market_chart_layout(
                price_fig,
                title=f"{ticker_label(symbol)} Forecast, Actuals, and Strategy Curve",
                height=560,
                margin=dict(l=55, r=65, t=78, b=48),
            )
            price_fig.update_layout(
                yaxis=dict(title="Actual / Forecast Price"),
                yaxis2=dict(
                    title="Crossover Strategy Equity Curve",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                ),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            price_fig.update_xaxes(title_text="Date")
            render_chart(price_fig)

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

            short_signal_rows = []
            short_signal_errors = []
            for short_horizon in SHORT_TERM_SIGNAL_HORIZONS:
                try:
                    if int(short_horizon) == int(forecast_horizon) and one_day_sequence_model_choice == sequence_model_choice:
                        short_model_results = model_results
                    else:
                        short_model_results = cached_model_results(
                            symbol,
                            history_days,
                            int(short_horizon),
                            forecast_lookback,
                            forecast_alpha,
                            optimize_forecast_model,
                            use_market_context,
                            one_day_sequence_model_choice,
                            include_rl_policy,
                            tuple(selected_forecast_symbols),
                            st.session_state["symbol_refresh_nonce"],
                            earnings_context_json=json.dumps(
                                earnings_context,
                                sort_keys=True,
                                default=str,
                            ),
                        )
                    short_signal_rows.append(
                        short_term_signal_row(int(short_horizon), short_model_results, primary_model_choice)
                    )
                except Exception as short_signal_error:
                    short_signal_errors.append(f"{short_horizon}D: {short_signal_error}")

            if short_signal_rows:
                st.markdown("**1-Day ML Signal**")
                st.caption(
                    "This is a separate direct forecast for the next trading day, not a slice of the 30-day forecast. "
                    f"Sequence model: {one_day_sequence_model_choice}."
                )
                short_signal_table = pd.DataFrame(short_signal_rows)
                st.dataframe(
                    short_signal_table.style.format(
                        {
                            "Forecast Price": "${:,.2f}",
                            "Forecast Return %": "{:+.2f}%",
                            "Probability Up %": "{:.1f}%",
                            "Model Edge %": "{:.1f}%",
                            "Expected Error %": "{:.2f}%",
                            "Score": "{:+.3f}",
                        },
                        na_rep="",
                    ).hide(axis="index"),
                    use_container_width=True,
                )
            if short_signal_errors:
                with st.expander("Short-term signal errors"):
                    st.write("\n".join(short_signal_errors))

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
            forecast_error_text = str(forecast_error)
            if forecast_error_text.startswith("Enable 'Run selected-symbol ML forecast'"):
                st.info(forecast_error_text)
            elif "paused from the sidebar" in forecast_error_text:
                st.info(forecast_error_text)
            else:
                st.warning(f"ML forecast unavailable: {forecast_error}")

        render_section_header(
            "Paper Trading Controls",
            "Manual and smart-policy paper-trading actions for the selected symbol.",
        )
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
                    st.session_state["skip_heavy_once"] = True
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
                        st.session_state["skip_heavy_once"] = True
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
                    st.session_state["skip_heavy_once"] = True
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
                        st.session_state["skip_heavy_once"] = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to sell: {e}")

        st.divider()
        signal_label = "BUY" if sig.iloc[-1] == 1 else "FLAT"
        st.metric("Latest Strategy Signal", signal_label)
        st.caption(f"Last updated {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC}")

        auto_trade_result = {"status": "disabled"}
        if auto_trade_enabled:
            if auto_trade_trigger == "Smart Policy":
                auto_trade_result = maybe_execute_auto_trade_smart(
                    symbol=symbol,
                    df=df,
                    sig=sig,
                    equity=equity,
                    risk_fraction=auto_trade_risk_fraction,
                    enabled=True,
                    demo_mode=demo_mode,
                    trade_summary=trade_summary,
                    forecast_result=forecast_result,
                    model_results=model_results,
                    earnings_context=earnings_context,
                )
        if auto_trade_result.get("status") == "executed":
            entry = auto_trade_result["entry"]
            mode_label = "Demo" if entry.get("demo_mode") else "Live"
            if entry.get("action") == "LIQUIDATE_ALL":
                st.success(
                    f"{mode_label} drawdown circuit breaker accepted close "
                    f"orders for {len(entry.get('position_symbols', []))} "
                    "broker positions."
                )
            else:
                st.success(
                    f"{mode_label} auto-trade accepted: "
                    f"{entry.get('action')} {float(entry.get('qty', 0.0)):.6f} "
                    f"estimated shares of {entry.get('symbol')} "
                    f"(${float(entry.get('notional', 0.0)):,.2f} notional)"
                )
            st.session_state["skip_heavy_once"] = True
            st.rerun()
        elif auto_trade_enabled and auto_trade_result.get("status") == "partial":
            st.error(auto_trade_result.get("reason", "Portfolio liquidation was partial."))
        elif auto_trade_enabled and auto_trade_result.get("status") == "skipped":
            st.caption(f"Auto-trade check: {auto_trade_result.get('reason', 'skipped')}")

        render_section_header(
            "Trade Metrics",
            "Paper-trading execution totals and mark-to-market performance estimates.",
        )
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
        render_section_header(
            "Portfolio Balance Chart",
            "Equity, cash, and profit/loss history from Alpaca or the local trade ledger.",
        )
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
            money_fig = apply_market_chart_layout(
                money_fig,
                title="Portfolio Equity History",
                height=460,
                margin=dict(l=55, r=65, t=72, b=45),
            )
            money_fig.update_layout(
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            money_fig.update_xaxes(title_text="Time")
            money_fig.update_yaxes(title_text="Equity ($)")
            render_chart(money_fig)
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
            fallback_fig = apply_market_chart_layout(
                fallback_fig,
                title="Portfolio Balance History",
                height=460,
                margin=dict(l=55, r=30, t=72, b=45),
            )
            fallback_fig.update_layout(
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            fallback_fig.update_xaxes(title_text="Time")
            fallback_fig.update_yaxes(title_text="Balance ($)")
            render_chart(fallback_fig)

        render_section_header(
            "Recent Trade Log",
            "Latest paper-trading actions, policy decisions, and account-state updates.",
        )
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

st.divider()

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
