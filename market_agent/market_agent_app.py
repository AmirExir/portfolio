import streamlit as st
import pandas as pd
from agent.data import get_ohlcv
from agent.strategy import sma_crossover
from agent.backtest import simple_vector_backtest
from agent.broker import get_account, submit_order
import datetime as dt
import plotly.graph_objects as go

st.set_page_config(page_title="Market Agent", layout="wide")

st.title("📈 Amir Exir Stock Market & Crypto AI Agent")

# --- Account Summary ---
acct = get_account()
equity = float(acct.get("equity", 0))
cash = float(acct.get("cash", 0))
buying_power = float(acct.get("buying_power", 0))
st.sidebar.header("💰 Account Summary (Paper Trading)")
st.sidebar.metric("Equity", f"${equity:,.2f}")
st.sidebar.metric("Cash", f"${cash:,.2f}")
st.sidebar.metric("Buying Power", f"${buying_power:,.2f}")

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
        result = submit_order(symbol, 1, "buy")
        st.success(f"Bought 1 share of {symbol}")
        st.json(result)
with col2:
    if st.button(f"🔴 Sell {symbol}"):
        result = submit_order(symbol, 1, "sell")
        st.warning(f"Sold 1 share of {symbol}")
        st.json(result)

# --- Load data and backtest ---
df = get_ohlcv(symbol, 400)

# --- Timeframe selection ---
timeframe = st.selectbox("📆 Timeframe", ["1Y", "6M", "1M", "1W", "1D", "1H"])

now = df.index.max()
if timeframe == "1Y":
    start_date = now - pd.Timedelta(days=365)
    df_filtered = df[df.index >= start_date]
elif timeframe == "6M":
    start_date = now - pd.Timedelta(days=180)
    df_filtered = df[df.index >= start_date]
elif timeframe == "1M":
    start_date = now - pd.Timedelta(days=30)
    df_filtered = df[df.index >= start_date]
elif timeframe == "1W":
    start_date = now - pd.Timedelta(days=7)
    df_filtered = df[df.index >= start_date]
elif timeframe == "1D":
    start_date = now - pd.Timedelta(hours=24)
    df_filtered = df[df.index >= start_date]
elif timeframe == "1H":
    # Resample to hourly if timestamps are available
    if pd.infer_freq(df.index) is None or "T" in pd.infer_freq(df.index) or "S" in pd.infer_freq(df.index):
        df_filtered = df.resample('H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    else:
        df_filtered = df.copy()
else:
    df_filtered = df.copy()

sig = sma_crossover(df_filtered, short_window, long_window)
bt = simple_vector_backtest(df_filtered, sig)

col1, col2 = st.columns(2)
# Compute moving averages for chart overlay
df_filtered["ma_short"] = df_filtered["close"].rolling(window=short_window).mean()
df_filtered["ma_long"] = df_filtered["close"].rolling(window=long_window).mean()
with col1:
    st.subheader("📊 Price and Moving Averages")
    # Prepare Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered["close"], mode="lines", name="Close",
                             line=dict(color='cyan', width=3)))
    fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered["ma_short"], mode="lines", name=f"MA {short_window}",
                             line=dict(color='white', width=1), opacity=0.6))
    fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered["ma_long"], mode="lines", name=f"MA {long_window}",
                             line=dict(color='white', width=1), opacity=0.6))

    # Find buy and sell points
    sig_shift = sig.shift(1).fillna(0)
    buy_points = (sig_shift == 0) & (sig == 1)
    sell_points = (sig_shift == 1) & (sig == 0)

    # Add buy markers
    fig.add_trace(go.Scatter(
        x=df_filtered.index[buy_points],
        y=df_filtered["close"][buy_points],
        mode="markers",
        marker=dict(symbol="triangle-up", color="green", size=12),
        name="Buy"
    ))
    # Add sell markers
    fig.add_trace(go.Scatter(
        x=df_filtered.index[sell_points],
        y=df_filtered["close"][sell_points],
        mode="markers",
        marker=dict(symbol="triangle-down", color="red", size=12),
        name="Sell"
    ))
    fig.update_layout(
        template="plotly_dark",
        legend=dict(x=1, y=1, xanchor='right', yanchor='top', bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=20, r=20, t=50, b=40),
        dragmode="drawline",
        newshape_line_color="yellow",
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        xaxis=dict(
            tickangle=-45,
            showgrid=True,
            gridcolor="#222222",
            zerolinecolor="#444444",
            showline=True,
            linecolor="#444444",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#222222",
            zerolinecolor="#444444",
            showline=True,
            linecolor="#444444",
        ),
        title=dict(text="Price Chart", x=0.5, xanchor='center', font=dict(color='white', size=16))
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("Use the toolbar above the chart to draw lines, shapes, or annotations directly on the chart.")
with col2:
    st.subheader("Strategy Equity Curve")
    st.line_chart(bt["curve"])

# --- Latest Signal + Timestamp ---
st.write("Latest Signal:", "🟢 BUY" if sig.iloc[-1] == 1 else "🔴 FLAT")
st.caption(f"Last updated {dt.datetime.utcnow():%Y-%m-%d %H:%M UTC}")
st.markdown("---")
import requests, os
r = requests.get("https://paper-api.alpaca.markets/v2/orders", headers={
    "APCA-API-KEY-ID": os.getenv("ALPACA_KEY"),
    "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET"),
})
print(r.json())