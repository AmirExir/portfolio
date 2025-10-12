import streamlit as st
import pandas as pd
from agent.data import get_ohlcv
from agent.strategy import sma_crossover
from agent.backtest import simple_vector_backtest
from agent.broker import get_account, submit_order, cancel_open_orders  # make sure cancel_open_orders is imported
import datetime as dt
import os

st.set_page_config(page_title="Market Agent", layout="wide")

st.title("📈 Amir Exir Stock Market & Crypto AI Agent")


# Owner Key unlock system
owner_key_input = st.sidebar.text_input("Enter Owner Key", type="password")
OWNER_KEY = st.secrets.get("OWNER_KEY", "")

if owner_key_input == OWNER_KEY and OWNER_KEY != "":
    demo_mode = st.sidebar.checkbox("🧪 Demo Mode", value=False, help="Toggle between live and demo mode")
    if demo_mode:
        st.sidebar.info("🧪 Demo Mode active — trades will not be executed.")
    else:
        st.sidebar.success("✅ Live Mode active — connected to Alpaca paper trading.")
else:
    demo_mode = True
    st.sidebar.info("🧪 Demo Mode forced ON for public viewers — safe demo mode.")

# --- Load Alpaca credentials from Streamlit Secrets ---
ALPACA_KEY = st.secrets["ALPACA_KEY"]
ALPACA_SECRET = st.secrets["ALPACA_SECRET"]
ALPACA_ENDPOINT = st.secrets.get("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets")

# --- Account Summary ---
if demo_mode:
    equity = 100000
    cash = 100000
    buying_power = 200000
else:
    acct = get_account()
    equity = float(acct.get("equity", 0))
    cash = float(acct.get("cash", 0))
    buying_power = float(acct.get("buying_power", 0))
st.sidebar.header("💰 Account Summary (Paper Trading)")
st.sidebar.metric("Equity", f"${equity:,.2f}")
st.sidebar.metric("Cash", f"${cash:,.2f}")
st.sidebar.metric("Buying Power", f"${buying_power:,.2f}")

# --- Add a Cancel Orders Button ---
if st.sidebar.button("🧹 Cancel Open Orders"):
    cancel_open_orders()
    st.sidebar.success("All open orders canceled!")

# --- Strategy Settings ---
st.sidebar.header("📊 Strategy Settings")
short_window = st.sidebar.number_input("Short-term MA window", min_value=1, max_value=100, value=20, step=1)
long_window = st.sidebar.number_input("Long-term MA window", min_value=1, max_value=200, value=50, step=1)

# --- Symbol input ---
symbol = st.text_input("Symbol", "AAPL", key="symbol_input")

# --- Manual Trade Buttons ---
col1, col2 = st.columns(2)
with col1:
    if st.button(f"🟢 Buy {symbol}"):
        if demo_mode:
            st.info(f"(Demo) Pretending to buy 1 share of {symbol}")
        else:
            result = submit_order(symbol, 1, "buy")
            st.success(f"Bought 1 share of {symbol}")
            st.json(result)
with col2:
    if st.button(f"🔴 Sell {symbol}"):
        if demo_mode:
            st.info(f"(Demo) Pretending to sell 1 share of {symbol}")
        else:
            result = submit_order(symbol, 1, "sell")
            st.warning(f"Sold 1 share of {symbol}")
            st.json(result)




# --- Load data and backtest ---
df = get_ohlcv(symbol, 400)

sig = sma_crossover(df, short_window, long_window)
bt = simple_vector_backtest(df, sig)

st.subheader("📊 Price Chart")
st.line_chart(df["close"])

st.subheader("Strategy Equity Curve")
st.line_chart(bt["curve"])

# --- Latest Signal + Timestamp ---
st.write("Latest Signal:", "🟢 BUY" if sig.iloc[-1] == 1 else "🔴 FLAT")
st.caption(f"Last updated {dt.datetime.utcnow():%Y-%m-%d %H:%M UTC}")
st.markdown("---")
import requests

try:
    r = requests.get("https://paper-api.alpaca.markets/v2/orders", headers={
        "APCA-API-KEY-ID": os.getenv("ALPACA_KEY"),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET"),
    })
    print(r.json())
except Exception as e:
    st.error(f"Failed to fetch orders from Alpaca API: {e}")
