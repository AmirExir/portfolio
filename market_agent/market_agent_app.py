import streamlit as st
from agent.data import get_ohlcv
from agent.strategy import sma_crossover
from agent.backtest import simple_vector_backtest
import datetime as dt

st.set_page_config(page_title="Market Agent", layout="wide")

st.title("📈 Stock Market & Crypto AI Agent")

symbol = st.text_input("Symbol", "AAPL")
df = get_ohlcv(symbol, 400)
sig = sma_crossover(df)
bt = simple_vector_backtest(df, sig)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Price Chart")
    st.line_chart(df["close"])
with col2:
    st.subheader("Strategy Equity Curve")
    st.line_chart(bt["curve"])

st.write("Latest Signal:", "🟢 BUY" if sig.iloc[-1] == 1 else "🔴 FLAT")
st.caption(f"Last updated {dt.datetime.utcnow():%Y-%m-%d %H:%M UTC}")