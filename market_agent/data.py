# agent/data.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_ohlcv(symbol: str, lookback_days: int = 200, interval="1d") -> pd.DataFrame:
    start = (datetime.utcnow() - timedelta(days=lookback_days*2)).strftime("%Y-%m-%d")
    df = yf.download(symbol, start=start, interval=interval, progress=False)
    df = df.rename(columns=str.lower)[["open","high","low","close","volume"]].dropna()
    return df