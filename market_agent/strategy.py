# agent/strategy.py
import pandas as pd
import numpy as np

def sma_crossover(df: pd.DataFrame, fast=20, slow=50) -> pd.Series:
    f = df["close"].rolling(fast).mean()
    s = df["close"].rolling(slow).mean()
    signal = np.where(f > s, 1, 0)  # long-or-flat
    # trade signal = entry when it flips
    return pd.Series(signal, index=df.index, name="signal")