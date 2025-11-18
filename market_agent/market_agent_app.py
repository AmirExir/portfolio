st.subheader(" Real-Time S&P 500 Heatmap")

# Timeframe selector for stocks
sp_tf = st.selectbox(
    "Stock change timeframe",
    ["1D", "7D", "1M", "3M", "1Y", "5Y"],
    index=0
)

# Map timeframe -> lookback days
sp_days_map = {
    "1D": 1,
    "7D": 7,
    "1M": 30,
    "3M": 90,
    "1Y": 365,
    "5Y": 365 * 5,
}

sp_lookback = sp_days_map[sp_tf]

try:
    # Fetch enough history for selected timeframe
    sp_period_days = sp_lookback + 5
    hist = yf.download(tickers, period=f"{sp_period_days}d", interval="1d")["Close"]

    # Determine base price
    if hist.shape[0] <= sp_lookback:
        base_sp = hist.iloc[0]
    else:
        base_sp = hist.iloc[-(sp_lookback + 1)]

    last_sp = hist.iloc[-1]
    pct_change = (last_sp - base_sp) / base_sp * 100
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