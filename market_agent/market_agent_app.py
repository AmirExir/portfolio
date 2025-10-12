import streamlit as st
import pandas as pd
from agent.data import get_ohlcv
from agent.strategy import sma_crossover
from agent.backtest import simple_vector_backtest
from agent.broker import get_account, submit_order, cancel_open_orders
import datetime as dt
import os
import sys
import requests

import base64
import json

sys.path.append(os.path.dirname(__file__))

st.set_page_config(page_title="Market Agent Dashboard", layout="wide")

st.title("📈 Amir Exir Stock Market & Crypto AI Agent")

# --- Fetch the latest summary from GitHub ---
st.markdown("### 🧠 Latest AI Summary")

url = "https://api.github.com/repos/AmirExir/portfolio/contents/market_agent/summary.txt"

try:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    # Decode Base64 content from GitHub API
    if "content" in data:
        # GitHub API returns content with newlines, so we need to remove them first
        content_base64 = data["content"].replace("\n", "")
        summary_text = base64.b64decode(content_base64).decode("utf-8")
    else:
        summary_text = "⚠️ No content found in the response"
        
except requests.exceptions.RequestException as e:
    summary_text = f"⚠️ Error fetching summary from GitHub: {e}"
except Exception as e:
    summary_text = f"⚠️ Error decoding summary: {e}"

# Display the summary in a nice format
st.markdown(summary_text)

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
ALPACA_KEY = st.secrets.get("ALPACA_KEY")
ALPACA_SECRET = st.secrets.get("ALPACA_SECRET")
ALPACA_ENDPOINT = st.secrets.get("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets")

# --- Account Summary ---
if demo_mode or not ALPACA_KEY or not ALPACA_SECRET:
    equity, cash, buying_power = 100000.0, 100000.0, 200000.0
    if not ALPACA_KEY or not ALPACA_SECRET:
        st.sidebar.warning("⚠️ Alpaca API keys not found — using demo values.")
else:
    try:
        acct = get_account()
        if isinstance(acct, dict) and "equity" in acct:
            equity = float(acct.get("equity", 0))
            cash = float(acct.get("cash", 0))
            buying_power = float(acct.get("buying_power", 0))
        else:
            st.sidebar.warning("⚠️ Invalid response from Alpaca — using demo values.")
            equity, cash, buying_power = 100000.0, 100000.0, 200000.0
    except Exception as e:
        st.sidebar.error(f"⚠️ Failed to fetch Alpaca account: {e}")
        equity, cash, buying_power = 100000.0, 100000.0, 200000.0

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

try:
    r = requests.get("https://paper-api.alpaca.markets/v2/orders", headers={
        "APCA-API-KEY-ID": os.getenv("ALPACA_KEY"),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET"),
    })
    print(r.json())
except Exception as e:
    st.error(f"Failed to fetch orders from Alpaca API: {e}")