# agent/backtest.py
import pandas as pd

def simple_vector_backtest(df: pd.DataFrame, signal: pd.Series, fee_bps=1) -> pd.DataFrame:
    ret = df["close"].pct_change().fillna(0.0)
    pos = signal.shift(1).fillna(0.0)  # enter at next bar
    gross = pos * ret
    # fees when position changes
    turns = (pos - pos.shift(1)).abs().fillna(0.0)
    fee = turns * (fee_bps/10000)
    net = gross - fee
    curve = (1 + net).cumprod()
    return pd.DataFrame({"ret": net, "curve": curve})