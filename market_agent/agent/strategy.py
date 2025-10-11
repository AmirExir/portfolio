import pandas as pd
import numpy as np

def sma_crossover(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """
    Simple Moving Average crossover strategy.
    Returns a 1D pandas Series of signals (1 = buy/hold, 0 = flat).
    """
    f = df["close"].rolling(fast).mean()
    s = df["close"].rolling(slow).mean()

    # Boolean array for signal (1 if fast > slow else 0)
    signal = np.where(f > s, 1, 0).flatten()  # <- ensures 1D output
    return pd.Series(signal, index=df.index, name="signal")