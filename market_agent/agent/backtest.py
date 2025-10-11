import pandas as pd
import numpy as np

def simple_vector_backtest(df: pd.DataFrame, signal: pd.Series, fee_bps: float = 1) -> pd.DataFrame:
    """
    Simple vectorized backtest for SMA crossover strategy.
    df: DataFrame with 'close' column
    signal: Series of 1 (buy/hold) or 0 (flat)
    fee_bps: trading fee in basis points (default 1 = 0.01%)
    """
    # Ensure alignment
    df = df.copy()
    signal = signal.reindex(df.index).fillna(0)

    # Daily returns
    df["ret"] = df["close"].pct_change().fillna(0)

    # Strategy returns (only earn when in position)
    df["strategy_ret"] = df["ret"] * signal.shift(1)

    # Apply trading fees when position changes
    df["trade"] = signal.diff().abs()
    df["strategy_ret"] -= df["trade"] * (fee_bps / 10000.0)

    # Compute cumulative returns
    df["curve"] = (1 + df["strategy_ret"]).cumprod()
    df["net"] = df["strategy_ret"]

    return df[["net", "curve"]]