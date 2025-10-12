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
    
    # Debug: Check the data structure
    st.write("DataFrame columns:", df_filtered.columns.tolist())
    st.write("DataFrame shape:", df_filtered.shape)
    st.write(f"Data range: {df_filtered.index.min()} to {df_filtered.index.max()}")
    
    # Check if 'close' column exists, if not, try 'Close'
    if 'close' in df_filtered.columns:
        close_col = 'close'
    elif 'Close' in df_filtered.columns:
        close_col = 'Close'
    else:
        st.error(f"Cannot find 'close' or 'Close' column. Available columns: {df_filtered.columns.tolist()}")
        st.stop()
    
    st.write(f"Close price range: ${float(df_filtered[close_col].min()):.2f} to ${float(df_filtered[close_col].max()):.2f}")
    st.write(f"Number of data points: {len(df_filtered)}")
    
    # Prepare Plotly figure
    fig = go.Figure()
    
    # Convert data to numpy arrays first, then to lists
    x_data = df_filtered.index.values
    close_data = df_filtered[close_col].values
    ma_short_data = df_filtered["ma_short"].values
    ma_long_data = df_filtered["ma_long"].values
    
    # Add Close price trace
    fig.add_trace(go.Scatter(
        x=x_data, 
        y=close_data, 
        mode="lines", 
        name="Close",
        line=dict(color='cyan', width=2),
        visible=True
    ))
    
    # Add MA traces
    fig.add_trace(go.Scatter(
        x=x_data, 
        y=ma_short_data, 
        mode="lines", 
        name=f"MA {short_window}",
        line=dict(color='orange', width=2), 
        opacity=0.7,
        visible=True
    ))
    
    fig.add_trace(go.Scatter(
        x=x_data, 
        y=ma_long_data, 
        mode="lines", 
        name=f"MA {long_window}",
        line=dict(color='purple', width=2), 
        opacity=0.7,
        visible=True
    ))

    # Find buy and sell points
    sig_shift = sig.shift(1).fillna(0)
    buy_points = (sig_shift == 0) & (sig == 1)
    sell_points = (sig_shift == 1) & (sig == 0)

    # Add buy markers
    if buy_points.any():
        fig.add_trace(go.Scatter(
            x=df_filtered.index[buy_points].values,
            y=df_filtered[close_col][buy_points].values,
            mode="markers",
            marker=dict(symbol="triangle-up", color="lime", size=15, line=dict(color="darkgreen", width=2)),
            name="Buy",
            visible=True
        ))
    
    # Add sell markers
    if sell_points.any():
        fig.add_trace(go.Scatter(
            x=df_filtered.index[sell_points].values,
            y=df_filtered[close_col][sell_points].values,
            mode="markers",
            marker=dict(symbol="triangle-down", color="red", size=15, line=dict(color="darkred", width=2)),
            name="Sell",
            visible=True
        ))

    # Calculate y-axis range with padding
    y_min = float(df_filtered[close_col].min())
    y_max = float(df_filtered[close_col].max())
    y_padding = (y_max - y_min) * 0.1
    y_range = [y_min - y_padding, y_max + y_padding]

    # Determine x-axis tickformat based on timeframe
    if timeframe in ["1Y", "6M", "1M"]:
        x_tickformat = "%b %Y"
    elif timeframe in ["1W", "1D"]:
        x_tickformat = "%b %d"
    else:
        x_tickformat = "%H:%M"

    fig.update_layout(
        template="plotly_dark",
        height=600,
        legend=dict(
            x=0.01, 
            y=0.99, 
            xanchor='left', 
            yanchor='top', 
            bgcolor='rgba(0,0,0,0.7)',
            bordercolor='white',
            borderwidth=1
        ),
        margin=dict(l=60, r=20, t=80, b=60),
        dragmode="pan",
        plot_bgcolor="#0a0a0a",
        paper_bgcolor="#000000",
        xaxis=dict(
            title="Date",
            titlefont=dict(size=14, color='white'),
            tickformat=x_tickformat,
            tickangle=-45,
            showgrid=True,
            gridcolor="#2a2a2a",
            gridwidth=1,
            showline=True,
            linecolor="#555555",
            linewidth=2,
            mirror=True
        ),
        yaxis=dict(
            title="Price ($)",
            titlefont=dict(size=14, color='white'),
            showgrid=True,
            gridcolor="#2a2a2a",
            gridwidth=1,
            showline=True,
            linecolor="#555555",
            linewidth=2,
            mirror=True,
            range=y_range
        ),
        title=dict(
            text=f"📈 {symbol} Price Chart - {timeframe}", 
            x=0.5, 
            xanchor='center', 
            font=dict(color='white', size=20, family='Arial Black')
        ),
        hovermode='x unified',
        showlegend=True
    )
    
    # Add configuration for interactivity
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'{symbol}_chart',
            'height': 600,
            'width': 1200,
            'scale': 2
        }
    }
    
    st.plotly_chart(fig, use_container_width=True, config=config, key="price_chart")
    st.info("💡 Use the toolbar to zoom, pan, or reset the view.")
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