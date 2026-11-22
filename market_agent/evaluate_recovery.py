"""Replay fixed recovery rules against next-open equity prices and explicit costs.

Run from the repository root with python -m market_agent.evaluate_recovery --help.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .agent.evaluation import (
    PolicyPerformance,
    PolicyPeriod,
    policy_performance_from_periods,
)
from .agent.policy import SmartPolicyConfig
from .recovery_strategy import recovery_path

SYMBOLS = [
    "NVDA",
    "AMD",
    "MU",
    "MSFT",
    "META",
    "INTC",
    "AVGO",
    "SPY",
    "QQQ",
    "XLK",
    "GLD",
    "SLV",
    "TLT",
    "USO",
    "OXY",
    "SLB",
]
MODES = ("invested", "half_invested", "trend", "immediate", "confirmed")


def market_history(
    symbols: list[str], start: str, end: str, directory: Path | None
) -> dict[str, pd.DataFrame]:
    """Read adjusted daily opens/closes; reject incomplete or misaligned panels."""
    if (
        not symbols
        or len(set(symbols)) != len(symbols)
        or any(s.endswith("-USD") for s in symbols)
    ):
        raise ValueError("Use unique US stock or ETF symbols for the next-open study")
    if directory is None:
        import yfinance as yf

        downloaded = yf.download(
            symbols,
            start=start,
            end=pd.Timestamp(end) + pd.Timedelta(days=1),
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=4,
        )
    frames = {}
    for symbol in symbols:
        if directory is None:
            frame = downloaded[symbol].rename(columns=str.lower)
        else:
            frame = pd.read_csv(
                directory / f"{symbol}.csv", index_col="date", parse_dates=True
            )
        frame = frame.loc[start:end, ["open", "close"]]
        if (
            frame.empty
            or not frame.index.is_unique
            or not frame.index.is_monotonic_increasing
            or not np.isfinite(frame).all().all()
            or (frame <= 0).any().any()
        ):
            raise ValueError(f"{symbol}: missing or invalid series")
        if frames and not frame.index.equals(next(iter(frames.values())).index):
            raise ValueError(f"{symbol}: session calendar differs from other symbols")
        frames[symbol] = frame
    return frames


def execution_targets(frames: dict[str, pd.DataFrame], mode: str) -> pd.DataFrame:
    """Assign each instrument one equal budget and delay close decisions to next open."""
    if mode in {"invested", "half_invested"}:
        fraction = 1.0 if mode == "invested" else 0.5
        positions = {s: pd.Series(fraction, index=f.index) for s, f in frames.items()}
    else:
        positions = {
            s: recovery_path(f["close"], entry=mode)["sleeve_fraction"]
            for s, f in frames.items()
        }
    return pd.DataFrame(positions).shift(1) / len(frames)


def simulate(
    targets: pd.DataFrame, opens: pd.DataFrame, *, start: str, end: str, cost_bps: float
) -> PolicyPerformance:
    """Start in cash, rebalance at the open, mark at the next open; charge drift-inclusive turnover."""
    prices = opens.loc[start:end]
    if len(prices) < 2 or not targets.columns.equals(prices.columns):
        raise ValueError("Need two aligned opening prices per asset")
    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 100:
        raise ValueError("Cost must be finite and from 0 to below 100 basis points")
    weights = targets.loc[prices.index[:-1]].to_numpy()
    if (
        not np.isfinite(weights).all()
        or (weights < 0).any()
        or (weights.sum(axis=1) > 1.0000001).any()
    ):
        raise ValueError("Missing or invalid execution weights")
    returns = prices.to_numpy()[1:] / prices.to_numpy()[:-1] - 1
    previous = np.zeros(prices.shape[1])
    rate = cost_bps / 10_000
    periods = []
    for session, desired, moves in zip(prices.index[1:], weights, returns):
        # Invest the available capital after fees; this also covers fully invested portfolios.
        capital = 1.0
        for _ in range(8):
            capital = 1 - rate * np.abs(capital * desired - previous).sum()
        positions = desired * capital
        turnover = float(np.abs(positions - previous).sum())
        fee = rate * turnover
        gross = float(positions @ moves)
        net = gross - fee
        exposure = float(positions.sum())
        benchmark = exposure * float(moves.mean())
        periods.append(
            PolicyPeriod(
                session=session.date(),
                gross_return=gross,
                benchmark_component=benchmark,
                gross_excess_return=gross - benchmark,
                transaction_cost=fee,
                net_return=net,
                net_excess_return=net - benchmark,
                gross_exposure=exposure,
                turnover=turnover,
            )
        )
        previous = positions * (1 + moves) / (1 + net)
    return policy_performance_from_periods(periods, transaction_cost_bps=cost_bps)


def main() -> None:
    """Print the comparable portfolio metrics for a specified research period."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument(
        "--prices-dir",
        type=Path,
        help="CSV files named SYMBOL.csv with date,open,close (adjusted)",
    )
    parser.add_argument("--history-start", default="2016-01-01")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-11-20")
    parser.add_argument(
        "--cost-bps", type=float, default=SmartPolicyConfig().trade_cost_bps
    )
    args = parser.parse_args()
    frames = market_history(args.symbols, args.history_start, args.end, args.prices_dir)
    opens = pd.DataFrame({s: f["open"] for s, f in frames.items()})
    if (opens.index < pd.Timestamp(args.start)).sum() < SmartPolicyConfig().slow_window:
        parser.error("Include at least 50 completed sessions before the test starts")
    records = []
    for mode in MODES:
        result = simulate(
            execution_targets(frames, mode),
            opens,
            start=args.start,
            end=args.end,
            cost_bps=args.cost_bps,
        )
        records.append(
            {
                "mode": mode,
                "sessions": result.session_count,
                "annual_return_pct": 100 * result.annualized_net_return,
                "max_drawdown_pct": 100 * result.max_drawdown,
                "Sharpe": result.sharpe,
                "mean_exposure_pct": 100 * result.average_gross_exposure,
            }
        )
    print(
        pd.DataFrame(records).to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )


if __name__ == "__main__":
    main()
