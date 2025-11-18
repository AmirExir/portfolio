import streamlit as st
import pandas as pd
import datetime as dt
import os
import sys
import requests
import base64
import json

import yfinance as yf
import plotly.express as px


# Add the parent directory to the path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import agent modules, with fallback handling
try:
    from agent.data import get_ohlcv
    from agent.strategy import sma_crossover
    from agent.backtest import simple_vector_backtest
    from agent.broker import get_account, submit_order, cancel_open_orders
except ImportError as e:
    st.error(f" Failed to import agent modules: {e}")
    st.info("Please ensure the 'agent' folder exists in the same directory as this app.")
    st.stop()

st.set_page_config(page_title="Market Agent Dashboard", layout="wide")

st.title(" Amir Exir Stock Market & Crypto AI Agent")

# --- Fetch the latest summary from GitHub ---
st.markdown(" AI-Generated Market Summary")

try:
    contents_url = "https://api.github.com/repos/AmirExir/portfolio/contents/market_agent"
    response = requests.get(contents_url)
    response.raise_for_status()
    files = response.json()

    # Filter files starting with 'summary_' and ending with '.txt'
    summary_files = [f for f in files if f.get("type") == "file" and f.get("name", "").startswith("summary_") and f.get("name", "").endswith(".txt")]

    if not summary_files:
        st.info("No summary files found yet. The n8n workflow will create summary_*.txt files on first run.")
    else:
        # Sort by name descending to get the latest
        summary_files_sorted = sorted(summary_files, key=lambda x: x["name"], reverse=True)
        latest_file = summary_files_sorted[0]
        download_url = latest_file.get("download_url")

        if download_url:
            content_response = requests.get(download_url)
            content_response.raise_for_status()
            summary_decoded = content_response.text

            # Try parsing as JSON if it's not plain text
            try:
                maybe_json = json.loads(summary_decoded)
                if isinstance(maybe_json, dict):
                    summary_text = maybe_json.get("content") or maybe_json.get("message") or str(maybe_json)
                elif isinstance(maybe_json, list):
                    summary_text = "\n".join([str(item) for item in maybe_json])
                else:
                    summary_text = str(maybe_json)
            except json.JSONDecodeError:
                summary_text = summary_decoded

            st.info(summary_text.strip())
        else:
            st.warning(" Could not find download URL for the latest summary file.")
except requests.exceptions.RequestException as e:
    st.warning(f" Error fetching summary from GitHub: {e}")
    st.info("Make sure the summary files exist at: https://github.com/AmirExir/portfolio/tree/main/market_agent")
except Exception as e:
    st.warning(f" Error decoding summary: {e}")
    st.info("There might be an issue with the file encoding or format.")

st.markdown("---")

#  Real-Time S&P 500 Heatmap (Market Cap Weighted + Labels)
st.subheader(" Real-Time S&P 500 Heatmap")

tickers = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM",
    "UNH", "XOM", "V", "JNJ", "WMT", "PG", "KO", "HD", "BAC", "CVX",
    "LLY", "PEP"
]

try:
    # Fetch price data (2 days to compute pct change)
    hist = yf.download(tickers, period="2d")["Close"]
    pct_change = hist.pct_change().iloc[-1] * 100
    pct_change = pct_change.fillna(0)

    # Fetch market cap for weighting
    market_caps = {}
    for t in tickers:
        info = yf.Ticker(t).info
        market_caps[t] = info.get("marketCap", 1)  # fallback to 1 if missing

    df = pd.DataFrame({
        "Ticker": pct_change.index,
        "Percent Change": pct_change.values,
        "Market Cap": [market_caps[t] for t in pct_change.index]
    })

    # Text labels: ticker + %
    df["Label"] = df.apply(lambda row: f"{row['Ticker']}\n{row['Percent Change']:.2f}%", axis=1)

    fig = px.treemap(
        df,
        path=["Ticker"],
        values="Market Cap",
        color="Percent Change",
        color_continuous_scale="RdYlGn",
        hover_data={"Market Cap": ":,.0f", "Percent Change": ":.2f"},
        title="S&P 500 Percent Change"
    )

    # Show label inside block
    fig.update_traces(text=df["Label"])

    st.plotly_chart(fig, use_container_width=True)


except Exception as e:
    st.error(f"Error generating heatmap: {e}")


# --- Real-Time Crypto Heatmap (Market Cap Weighted + Labels)
crypto_hist = yf.download(crypto_tickers, period="3d", interval="1h")["Close"]
# Owner Key unlock system
owner_key_input = st.sidebar.text_input("Enter Owner Key", type="password")
OWNER_KEY = st.secrets.get("OWNER_KEY", "")

if owner_key_input == OWNER_KEY and OWNER_KEY != "":
    demo_mode = st.sidebar.checkbox("Demo Mode", value=False, help="Toggle between live and demo mode")
    if demo_mode:
        st.sidebar.info("Demo Mode active — trades will not be executed.")
    else:
        st.sidebar.success(" Live Mode active — connected to Alpaca paper trading.")
else:
    demo_mode = True
    st.sidebar.info("Demo Mode forced ON for public viewers — safe demo mode.")

# --- Load Alpaca credentials from Streamlit Secrets ---
ALPACA_KEY = st.secrets.get("ALPACA_KEY")
ALPACA_SECRET = st.secrets.get("ALPACA_SECRET")
ALPACA_ENDPOINT = st.secrets.get("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets")

# --- Account Summary ---
if demo_mode or not ALPACA_KEY or not ALPACA_SECRET:
    equity, cash, buying_power = 100000.0, 100000.0, 200000.0
    if not ALPACA_KEY or not ALPACA_SECRET:
        st.sidebar.warning("Alpaca API keys not found — using demo values.")
else:
    try:
        acct = get_account()
        if isinstance(acct, dict) and "equity" in acct:
            equity = float(acct.get("equity", 0))
            cash = float(acct.get("cash", 0))
            buying_power = float(acct.get("buying_power", 0))
        else:
            st.sidebar.warning("Invalid response from Alpaca — using demo values.")
            equity, cash, buying_power = 100000.0, 100000.0, 200000.0
    except Exception as e:
        st.sidebar.error(f"Failed to fetch Alpaca account: {e}")
        equity, cash, buying_power = 100000.0, 100000.0, 200000.0

st.sidebar.header("Account Summary (Paper Trading)")
st.sidebar.metric("Equity", f"${equity:,.2f}")
st.sidebar.metric("Cash", f"${cash:,.2f}")
st.sidebar.metric("Buying Power", f"${buying_power:,.2f}")

# --- Add a Cancel Orders Button ---
if st.sidebar.button("🧹 Cancel Open Orders"):
    if not demo_mode:
        try:
            cancel_open_orders()
            st.sidebar.success("All open orders canceled!")
        except Exception as e:
            st.sidebar.error(f"Failed to cancel orders: {e}")
    else:
        st.sidebar.info("(Demo) Orders not canceled in demo mode")

# --- Strategy Settings ---
st.sidebar.header("Strategy Settings")
short_window = st.sidebar.number_input("Short-term MA window", min_value=1, max_value=100, value=20, step=1)
long_window = st.sidebar.number_input("Long-term MA window", min_value=1, max_value=200, value=50, step=1)

# --- Symbol input ---
symbol = st.text_input("Symbol", "AAPL", key="symbol_input").upper()

# --- Manual Trade Buttons ---
col1, col2 = st.columns(2)
with col1:
    if st.button(f"Buy {symbol}"):
        if demo_mode:
            st.info(f"(Demo) Pretending to buy 1 share of {symbol}")
        else:
            try:
                result = submit_order(symbol, 1, "buy")
                st.success(f"Bought 1 share of {symbol}")
                st.json(result)
            except Exception as e:
                st.error(f"Failed to buy: {e}")
with col2:
    if st.button(f"Sell {symbol}"):
        if demo_mode:
            st.info(f"(Demo) Pretending to sell 1 share of {symbol}")
        else:
            try:
                result = submit_order(symbol, 1, "sell")
                st.warning(f"Sold 1 share of {symbol}")
                st.json(result)
            except Exception as e:
                st.error(f"Failed to sell: {e}")

# --- Load data and backtest ---
try:
    df = get_ohlcv(symbol, 400)
    
    sig = sma_crossover(df, short_window, long_window)
    bt = simple_vector_backtest(df, sig)
    
    st.subheader("Price Chart")
    st.line_chart(df["close"])
    
    st.subheader("Strategy Equity Curve")
    st.line_chart(bt["curve"])
    
    # --- Latest Signal + Timestamp ---
    signal_emoji = "BUY" if sig.iloc[-1] == 1 else "FLAT"
    st.write(f"**Latest Signal:** {signal_emoji}")
    st.caption(f"Last updated {dt.datetime.utcnow():%Y-%m-%d %H:%M UTC}")
    
except Exception as e:
    st.error(f" Error loading market data: {e}")
    st.info("Please check if the symbol is valid and try again.")

st.markdown("---")

# Debug info (only show in sidebar if needed)
if st.sidebar.checkbox("Show Debug Info", value=False):
    try:
        r = requests.get("https://paper-api.alpaca.markets/v2/orders", headers={
            "APCA-API-KEY-ID": ALPACA_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET,
        })
        st.sidebar.json(r.json())
    except Exception as e:
        st.sidebar.error(f"Failed to fetch orders: {e}")