import streamlit as st
import pandas as pd
from agent.data import get_ohlcv
from agent.strategy import sma_crossover
from agent.backtest import simple_vector_backtest
from agent.broker import get_account, submit_order



st.set_page_config(page_title="Market Agent", layout="wide")

st.title("📈 Amir Exir Stock Market & Crypto AI Agent")

acct = get_account()
equity = float(acct.get("equity", 0))
cash = float(acct.get("cash", 0))
buying_power = float(acct.get("buying_power", 0))
st.sidebar.header("💰 Account Summary (Paper Trading)")
st.sidebar.metric("Equity", f"${equity:,.2f}")
st.sidebar.metric("Cash", f"${cash:,.2f}")
st.sidebar.metric("Buying Power", f"${buying_power:,.2f}")

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