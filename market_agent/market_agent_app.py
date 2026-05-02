import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import os
import sys
import requests
import base64
import json

import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go


# Add the parent directory to the path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import agent modules, with fallback handling
try:
    from agent.data import get_ohlcv
    from agent.forecast import backtest_forecasts, compare_forecast_models, forecast_close_prices
    from agent.strategy import sma_crossover
    from agent.backtest import simple_vector_backtest
    from agent.broker import get_account, submit_order, cancel_open_orders
except ImportError as e:
    st.error(f" Failed to import agent modules: {e}")
    st.info("Please ensure the 'agent' folder exists in the same directory as this app.")
    st.stop()


def get_secret(name, default=None):
    """Read Streamlit secrets when configured, otherwise fall back to env/default."""
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)


st.set_page_config(page_title="📈 Market Agent Dashboard", layout="wide")

st.title("🤖 Amir Exir Stock Market & Crypto AI Agent")

SYMBOL_OPTIONS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO",
    "JPM", "BAC", "V", "UNH", "LLY", "JNJ", "WMT", "PG", "KO", "HD",
    "XOM", "CVX", "PEP", "SPY", "VOO", "QQQ", "IWM", "DIA", "GLD",
    "SLV", "USO", "TLT", "XLK", "XLF", "XLE", "XLV", "XLY",
]
DEFAULT_FORECAST_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM",
    "XOM", "UNH", "SPY", "VOO", "AVGO", "GLD", "SLV", "USO",
]
SYMBOL_LABELS = {
    "AVGO": "Broadcom (AVGO)",
    "GLD": "Gold (GLD)",
    "SLV": "Silver (SLV)",
    "USO": "Oil (USO)",
}
MARKET_CONTEXT_TICKERS = [
    "SPY", "VOO", "QQQ", "IWM", "DIA", "^VIX",
    "XLK", "XLF", "XLE", "XLV", "XLY",
    "TLT", "GLD", "SLV", "USO", "AVGO",
]


@st.cache_data(ttl=900, show_spinner=False)
def load_ohlcv(symbol: str, history_days: int) -> pd.DataFrame:
    return get_ohlcv(symbol, history_days)


def _load_market_context_data(history_days: int) -> pd.DataFrame:
    start = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=history_days * 2)).strftime("%Y-%m-%d")
    raw = yf.download(MARKET_CONTEXT_TICKERS, start=start, interval="1d", progress=False)
    if raw.empty:
        return pd.DataFrame()

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = close.rename(columns={ticker: f"context_{ticker}" for ticker in close.columns})
    return close.dropna(how="all").ffill()


@st.cache_data(ttl=900, show_spinner=False)
def load_market_context(history_days: int) -> pd.DataFrame:
    return _load_market_context_data(history_days)


@st.cache_data(ttl=900, show_spinner=False)
def load_stock_heatmap_data(tickers: tuple[str, ...], lookback_days: int) -> pd.DataFrame:
    period_days = lookback_days + 3
    hist = yf.download(list(tickers), period=f"{period_days}d", interval="1d", progress=False)["Close"]

    if hist.shape[0] <= lookback_days:
        base_sp = hist.iloc[0]
    else:
        base_sp = hist.iloc[-(lookback_days + 1)]

    last_sp = hist.iloc[-1]
    pct_change = ((last_sp - base_sp) / base_sp * 100).fillna(0)
    market_caps = {}
    for ticker in tickers:
        market_caps[ticker] = yf.Ticker(ticker).info.get("marketCap", 1)

    df = pd.DataFrame(
        {
            "Ticker": pct_change.index,
            "Percent Change": pct_change.values,
            "Market Cap": [market_caps[ticker] for ticker in pct_change.index],
        }
    )
    df["Label"] = df.apply(lambda row: f"{row['Ticker']}\n{row['Percent Change']:.2f}%", axis=1)
    return df


@st.cache_data(ttl=900, show_spinner=False)
def load_watchlist_heatmap_data(
    watchlist_tickers: tuple[str, ...],
    watchlist_labels: dict,
    lookback_days: int,
) -> pd.DataFrame:
    period_days = lookback_days + 3
    watch_hist = yf.download(list(watchlist_tickers), period=f"{period_days}d", interval="1d", progress=False)["Close"]

    if watch_hist.shape[0] <= lookback_days:
        watch_base = watch_hist.iloc[0]
    else:
        watch_base = watch_hist.iloc[-(lookback_days + 1)]

    watch_last = watch_hist.iloc[-1]
    watch_pct_change = ((watch_last - watch_base) / watch_base * 100).fillna(0)
    watch_df = pd.DataFrame(
        {
            "Ticker": watch_pct_change.index,
            "Label": [watchlist_labels.get(ticker, ticker) for ticker in watch_pct_change.index],
            "Percent Change": watch_pct_change.values,
            "Weight": 1,
        }
    )
    watch_df["Display"] = watch_df.apply(
        lambda row: f"{row['Label']}\n{row['Percent Change']:.2f}%",
        axis=1,
    )
    return watch_df


@st.cache_data(ttl=900, show_spinner=False)
def load_crypto_heatmap_data(crypto_tickers: tuple[str, ...], lookback_days: int) -> pd.DataFrame:
    period_days = lookback_days + 5
    crypto_hist = yf.download(list(crypto_tickers), period=f"{period_days}d", interval="1d", progress=False)["Close"]

    if crypto_hist.shape[0] <= lookback_days:
        base = crypto_hist.iloc[0]
    else:
        base = crypto_hist.iloc[-(lookback_days + 1)]

    last = crypto_hist.iloc[-1]
    crypto_pct_change = ((last - base) / base * 100.0).fillna(0)
    crypto_market_caps = {}
    for ticker in crypto_tickers:
        crypto_market_caps[ticker] = yf.Ticker(ticker).info.get("marketCap", 1)

    crypto_df = pd.DataFrame(
        {
            "Crypto": crypto_pct_change.index,
            "Percent Change": crypto_pct_change.values,
            "Market Cap": [crypto_market_caps[ticker] for ticker in crypto_pct_change.index],
        }
    )
    crypto_df["Symbol"] = crypto_df["Crypto"].apply(lambda value: value.split("-")[0])
    crypto_df["Label"] = crypto_df.apply(
        lambda row: f"{row['Symbol']}\n{row['Percent Change']:.2f}%",
        axis=1,
    )
    return crypto_df


def ticker_label(symbol: str) -> str:
    return SYMBOL_LABELS.get(symbol, symbol)


def is_timestamped_summary(filename: str) -> bool:
    if not filename.startswith("summary_") or not filename.endswith(".txt"):
        return False
    timestamp = filename.removeprefix("summary_").removesuffix(".txt")
    return len(timestamp) >= 11 and timestamp[:4].isdigit() and timestamp[4] == "-" and "T" in timestamp


def summary_timestamp_caption(filename: str) -> str | None:
    try:
        timestamp = filename.removeprefix("summary_").removesuffix(".txt")
        date_part, time_part = timestamp.split("T", 1)
        time_part = time_part.replace("Z", "").replace("-", ":")
        return f"📅 Last updated: {date_part} at {time_part} UTC"
    except Exception:
        return None


def signal_quality(confidence_pct: float) -> str:
    edge_pct = max(float(confidence_pct) - 50.0, 0.0)
    if edge_pct >= 15.0:
        return "High Edge"
    if edge_pct >= 8.0:
        return "Moderate Edge"
    if edge_pct >= 3.0:
        return "Weak Edge"
    return "No Edge"


def model_call(forecast_change_pct: float, expected_error_pct: float, confidence_pct: float) -> str:
    edge_pct = max(float(confidence_pct) - 50.0, 0.0)
    if edge_pct < 3.0 or abs(forecast_change_pct) < expected_error_pct * 0.5:
        return "Neutral / No Edge"
    if forecast_change_pct >= expected_error_pct and edge_pct >= 8.0:
        return "Strong Buy"
    if forecast_change_pct > 0:
        return "Buy"
    if forecast_change_pct <= -expected_error_pct and edge_pct >= 8.0:
        return "Strong Sell"
    if forecast_change_pct < 0:
        return "Sell"
    return "Neutral / No Edge"


def format_ranking_table(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.reset_index(drop=True).copy()
    if "Rank" not in display_df.columns:
        display_df.insert(0, "Rank", np.arange(1, len(display_df) + 1))

    display_columns = [
        "Rank",
        "Symbol",
        "Model Call",
        "Selected Model",
        "Last Price",
        "Forecast Price",
        "Forecast Return %",
        "Ridge Return %",
        "XGBoost Return %",
        "Ensemble Return %",
        "Probability Up %",
        "Directional Probability %",
        "Model Edge %",
        "Signal Quality",
        "Expected Error %",
        "Score",
    ]
    return display_df[display_columns].style.format(
        {
            "Rank": "{:.0f}",
            "Last Price": "${:,.2f}",
            "Forecast Price": "${:,.2f}",
            "Forecast Return %": "{:+.2f}%",
            "Ridge Return %": "{:+.2f}%",
            "XGBoost Return %": "{:+.2f}%",
            "Ensemble Return %": "{:+.2f}%",
            "Probability Up %": "{:.1f}%",
            "Directional Probability %": "{:.1f}%",
            "Model Edge %": "{:.1f}%",
            "Expected Error %": "{:.2f}%",
            "Score": "{:+.2f}",
        }
    ).hide(axis="index")


def model_results_table(model_results: dict) -> pd.DataFrame:
    rows = []
    for name, result in model_results.items():
        if result.forecast is None or result.forecast.empty:
            rows.append({"Model": name, "Status": result.metrics.get("error", "Unavailable")})
            continue

        confidence = result.metrics.get("confidence_pct", 50.0)
        rows.append(
            {
                "Model": name,
                "Status": "OK",
                "Forecast Price": float(result.forecast["forecast_close"].iloc[-1]),
                "Forecast Return %": result.metrics.get("forecast_change_pct", 0.0),
                "Probability Up %": result.metrics.get("probability_up_pct", 50.0),
                "Model Edge %": max(confidence - 50.0, 0.0),
                "Signal Quality": signal_quality(confidence),
                "Expected Error %": result.metrics.get("expected_error_pct", 0.0),
                "Validation MAE %": result.metrics.get("holdout_mae_pct", np.nan),
                "Validation Direction %": result.metrics.get("holdout_direction_accuracy", np.nan),
                "Score": result.metrics.get("forecast_score", np.nan),
            }
        )
    return pd.DataFrame(rows)


def best_model_name(model_results: dict, preferred: str = "Ensemble") -> str:
    if preferred in model_results and not model_results[preferred].forecast.empty:
        return preferred

    candidates = [
        (name, result)
        for name, result in model_results.items()
        if result.forecast is not None and not result.forecast.empty
    ]
    if not candidates:
        return ""
    return min(candidates, key=lambda item: item[1].metrics.get("holdout_mae_pct", np.inf))[0]


def result_metric(model_results: dict, model_name: str, metric_name: str, default=np.nan):
    result = model_results.get(model_name)
    if result is None:
        return default
    return result.metrics.get(metric_name, default)


@st.cache_data(ttl=900, show_spinner=False)
def cached_model_results(
    symbol: str,
    history_days: int,
    forecast_horizon: int,
    forecast_lookback: int,
    forecast_alpha: float,
    optimize_forecast_model: bool,
    use_market_context: bool,
) -> dict:
    df = get_ohlcv(symbol, history_days)
    context_df = _load_market_context_data(history_days) if use_market_context else pd.DataFrame()
    return compare_forecast_models(
        df,
        horizon_days=forecast_horizon,
        lookback_window=forecast_lookback,
        ridge_alpha=forecast_alpha,
        optimize_model=optimize_forecast_model,
        context_df=context_df,
    )


@st.cache_data(ttl=900, show_spinner=False)
def cached_historical_forecasts(
    symbol: str,
    history_days: int,
    forecast_horizon: int,
    selected_window: int,
    selected_alpha: float,
    historical_test_points: int,
    primary_model: str,
    use_market_context: bool,
):
    df = get_ohlcv(symbol, history_days)
    context_df = _load_market_context_data(history_days) if use_market_context else pd.DataFrame()
    return backtest_forecasts(
        df,
        horizon_days=forecast_horizon,
        lookback_window=int(selected_window),
        ridge_alpha=float(selected_alpha),
        max_points=historical_test_points,
        optimize_model=False,
        model_type=primary_model.lower(),
        context_df=context_df,
    )


@st.cache_data(ttl=900, show_spinner=False)
def cached_forecast_rankings(
    ranking_symbols: tuple[str, ...],
    history_days: int,
    forecast_horizon: int,
    forecast_lookback: int,
    forecast_alpha: float,
    optimize_forecast_model: bool,
    use_market_context: bool,
    primary_model_choice: str,
) -> tuple[pd.DataFrame, list[str]]:
    ranking_context_df = _load_market_context_data(history_days) if use_market_context else pd.DataFrame()
    ranking_rows = []
    ranking_errors = []

    for ranking_symbol in ranking_symbols:
        try:
            ranking_df = get_ohlcv(ranking_symbol, history_days)
            ranking_results = compare_forecast_models(
                ranking_df,
                horizon_days=forecast_horizon,
                lookback_window=forecast_lookback,
                ridge_alpha=forecast_alpha,
                optimize_model=optimize_forecast_model,
                context_df=ranking_context_df,
            )
            if primary_model_choice == "Best Validation":
                ranking_primary = best_model_name(ranking_results, preferred="")
            else:
                ranking_primary = best_model_name(ranking_results, preferred=primary_model_choice)
            if not ranking_primary:
                raise ValueError("No usable forecast model result.")

            ranking_result = ranking_results[ranking_primary]
            ranking_close = ranking_df["close"]
            if isinstance(ranking_close, pd.DataFrame):
                ranking_close = ranking_close.iloc[:, 0]
            ranking_close = pd.to_numeric(ranking_close, errors="coerce").dropna()

            last_price = float(ranking_close.iloc[-1])
            forecast_price = float(ranking_result.forecast["forecast_close"].iloc[-1])
            forecast_return = ranking_result.metrics.get("forecast_change_pct", 0.0)
            probability_up = ranking_result.metrics.get("probability_up_pct", 0.0)
            confidence = ranking_result.metrics.get("confidence_pct", 0.0)
            edge = max(confidence - 50.0, 0.0)
            expected_error = ranking_result.metrics.get("expected_error_pct", 0.0)
            score = ranking_result.metrics.get("forecast_score", 0.0)

            ranking_rows.append(
                {
                    "Symbol": ranking_symbol,
                    "Model Call": model_call(forecast_return, expected_error, confidence),
                    "Selected Model": ranking_primary,
                    "Last Price": last_price,
                    "Forecast Price": forecast_price,
                    "Forecast Return %": forecast_return,
                    "Ridge Return %": result_metric(ranking_results, "Ridge", "forecast_change_pct"),
                    "XGBoost Return %": result_metric(ranking_results, "XGBoost", "forecast_change_pct"),
                    "Ensemble Return %": result_metric(ranking_results, "Ensemble", "forecast_change_pct"),
                    "Probability Up %": probability_up,
                    "Probability Down %": 100.0 - probability_up,
                    "Directional Probability %": confidence,
                    "Model Edge %": edge,
                    "Signal Quality": signal_quality(confidence),
                    "Expected Error %": expected_error,
                    "Score": score,
                }
            )
        except Exception as ranking_error:
            ranking_errors.append(f"{ranking_symbol}: {ranking_error}")

    return pd.DataFrame(ranking_rows), ranking_errors


def report_generated_caption(report_text: str) -> str | None:
    for line in report_text.splitlines():
        if line.startswith("Generated:"):
            return f"📅 {line}"
    return None


@st.cache_data(ttl=300)
def fetch_latest_ml_report():
    local_path = os.path.join(
        os.path.dirname(__file__) if __file__ else ".",
        "reports",
        "ml_forecast_rankings_latest.txt",
    )
    if os.path.exists(local_path):
        try:
            with open(local_path, "r") as f:
                return f.read()
        except Exception:
            pass

    contents_url = (
        "https://api.github.com/repos/AmirExir/portfolio/contents/"
        "market_agent/reports/ml_forecast_rankings_latest.txt"
    )
    try:
        response = requests.get(
            contents_url,
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Streamlit-Market-Agent",
            },
            timeout=10,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        download_url = response.json().get("download_url")
        if not download_url:
            return None
        report_response = requests.get(download_url, timeout=10)
        report_response.raise_for_status()
        return report_response.text
    except Exception:
        return None


# --- Fetch the latest summary from GitHub ---
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_latest_summary():
    """Fetch the latest summary file from GitHub with better error handling"""
    contents_url = "https://api.github.com/repos/AmirExir/portfolio/contents/market_agent"
    
    try:
        # Add headers to avoid rate limiting
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Streamlit-Market-Agent"
        }
        response = requests.get(contents_url, headers=headers, timeout=10)
        
        # Check for rate limiting
        if response.status_code == 403 and 'rate limit' in response.text.lower():
            return None
        
        response.raise_for_status()
        files = response.json()
        return files
    except requests.exceptions.Timeout:
        return None
    except Exception as e:
        # Silently fail and let the fallback handle it
        return None

col_summary, col_refresh = st.columns([4, 1])
with col_summary:
    st.markdown("📰 AI-Generated Market Summary")
with col_refresh:
    if st.button("🔄 Refresh News", help="Fetch the latest news from GitHub"):
        st.cache_data.clear()
        st.rerun()

try:
    files = fetch_latest_summary()
    
    # Fallback: If GitHub API fails, try reading from local directory (for Streamlit Cloud deployment)
    if files is None:
        st.info(" GitHub API unavailable. Using local files...")
        local_dir = os.path.dirname(__file__) if __file__ else "."
        
        try:
            local_files = [f for f in os.listdir(local_dir) if is_timestamped_summary(f)]
            
            if local_files:
                # Sort and get latest
                local_files_sorted = sorted(local_files, reverse=True)
                latest_local_file = local_files_sorted[0]
                
                with open(os.path.join(local_dir, latest_local_file), 'r') as f:
                    summary_text = f.read()
                
                caption = summary_timestamp_caption(latest_local_file)
                if caption:
                    st.caption(caption)
                
                st.info(summary_text.strip())
            else:
                st.warning("No local summary files found.")
        except Exception as local_error:
            st.error(f"Failed to read local files: {local_error}")
    else:
        # Filter timestamped summary files across years.
        summary_files = [
            f for f in files 
            if f.get("type") == "file" 
            and is_timestamped_summary(f.get("name", ""))
        ]

        if not summary_files:
            st.info("No summary files found yet. The n8n workflow will create summary_*.txt files on first run.")
        else:
            # Sort by name descending to get the latest (ISO format sorts correctly)
            summary_files_sorted = sorted(summary_files, key=lambda x: x["name"], reverse=True)
            latest_file = summary_files_sorted[0]
            
            # Extract timestamp from filename (format: summary_2026-05-01T11-00-28-733Z.txt)
            filename = latest_file.get("name", "")
            caption = summary_timestamp_caption(filename)
            if caption:
                st.caption(caption)
            
            download_url = latest_file.get("download_url")

            if download_url:
                content_response = requests.get(download_url, timeout=10)
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
except Exception as e:
    st.error(f"⚠️ Unable to load summary: {e}")
    st.info("The news summary will be available when GitHub API or local files are accessible.")

try:
    latest_ml_report = fetch_latest_ml_report()
    if latest_ml_report:
        st.markdown("🧠 Scheduled ML Forecast Rankings")
        report_caption = report_generated_caption(latest_ml_report)
        if report_caption:
            st.caption(report_caption)
        st.info(latest_ml_report.strip())
    else:
        st.caption(
            "Scheduled ML forecast rankings will appear here after n8n writes "
            "market_agent/reports/ml_forecast_rankings_latest.txt."
        )
except Exception as e:
    st.caption(f"Scheduled ML forecast report unavailable: {e}")

st.markdown("---")

#  Real-Time S&P 500 Heatmap (Market Cap Weighted + Labels)
st.subheader("🧭 Real-Time S&P 500 Heatmap")

# Timeframe selector
sp_tf = st.selectbox(
    "📉 Stock change timeframe",
    ["1D", "7D", "1M", "3M", "1Y", "5Y"],
    index=0,
    key="stock_tf"
)

# Time-to-days map
sp_days_map = {
    "1D": 1,
    "7D": 7,
    "1M": 30,
    "3M": 90,
    "1Y": 365,
    "5Y": 365 * 5,
}

lookback_days = sp_days_map[sp_tf]

# S&P 500 sample tickers
tickers = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM",
    "UNH", "XOM", "V", "JNJ", "WMT", "PG", "KO", "HD", "BAC", "CVX",
    "LLY", "PEP", "AVGO"
]

try:
    df = load_stock_heatmap_data(tuple(tickers), lookback_days)
    fig = px.treemap(
        df,
        path=["Ticker"],
        values="Market Cap",
        color="Percent Change",
        color_continuous_scale="RdYlGn",
        hover_data={"Market Cap": ":,.0f", "Percent Change": ":.2f"},
        title=f"📊 S&P 500 Change ({sp_tf}) – Sized by Market Cap"
    )

    fig.update_traces(text=df["Label"])
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error generating heatmap: {e}")
    st.info("Please try a different timeframe or check your network connection.")


# ETF / commodity watchlist requested for broader market context
st.subheader("🧾 ETF and Commodity Watchlist")

watchlist_tickers = ["SPY", "VOO", "GLD", "SLV", "USO"]
watchlist_labels = {
    "SPY": "SPY",
    "VOO": "VOO",
    "GLD": "Gold (GLD)",
    "SLV": "Silver (SLV)",
    "USO": "Oil (USO)",
}

try:
    watch_df = load_watchlist_heatmap_data(tuple(watchlist_tickers), watchlist_labels, lookback_days)
    watch_fig = px.treemap(
        watch_df,
        path=["Label"],
        values="Weight",
        color="Percent Change",
        color_continuous_scale="RdYlGn",
        hover_data={"Percent Change": ":.2f"},
        title=f"SPY, VOO, Gold, Silver, and Oil Change ({sp_tf})",
    )
    watch_fig.update_traces(text=watch_df["Display"])
    st.plotly_chart(watch_fig, use_container_width=True)
except Exception as e:
    st.error(f"Error generating watchlist heatmap: {e}")


# --- Real-Time Crypto Heatmap (Market Cap Weighted + Labels)
# --- Real-Time Crypto Heatmap (Market Cap Weighted + Labels)
st.subheader("🪙 Real-Time Crypto Heatmap")

# Timeframe selector for crypto
crypto_tf = st.selectbox(
    "⏱️ Crypto change timeframe",
    ["24H", "7D", "1M", "3M", "1Y", "5Y"],
    index=0
)

# Map timeframe -> lookback in days
tf_days_map = {
    "24H": 1,
    "7D": 7,
    "1M": 30,
    "3M": 90,
    "1Y": 365,
    "5Y": 365 * 5,
}

lookback_days = tf_days_map[crypto_tf]

crypto_tickers = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD",
    "DOGE-USD", "AVAX-USD", "TON-USD", "DOT-USD"
]

try:
    crypto_df = load_crypto_heatmap_data(tuple(crypto_tickers), lookback_days)
    crypto_fig = px.treemap(
        crypto_df,
        path=["Symbol"],
        values="Market Cap",
        color="Percent Change",
        color_continuous_scale="RdYlGn",
        hover_data={"Market Cap": ":,.0f", "Percent Change": ":.2f"},
        title=f"🪙 Crypto Change ({crypto_tf}) – Sized by Market Cap"
    )

    crypto_fig.update_traces(text=crypto_df["Label"])
    st.plotly_chart(crypto_fig, use_container_width=True)

except Exception as e:
    st.error(f"Error generating crypto heatmap: {e}")


# Owner Key unlock system
owner_key_input = st.sidebar.text_input("🔑 Enter Owner Key", type="password")
OWNER_KEY = get_secret("OWNER_KEY", "")

if owner_key_input == OWNER_KEY and OWNER_KEY != "":
    demo_mode = st.sidebar.checkbox("🎭 Demo Mode", value=False, help="Toggle between live and demo mode")
    if demo_mode:
        st.sidebar.info("Demo Mode active — trades will not be executed.")
    else:
        st.sidebar.success(" Live Mode active — connected to Alpaca paper trading.")
else:
    demo_mode = True
    st.sidebar.info("Demo Mode forced ON for public viewers — safe demo mode.")

# --- Load Alpaca credentials from Streamlit Secrets ---
ALPACA_KEY = get_secret("ALPACA_KEY")
ALPACA_SECRET = get_secret("ALPACA_SECRET")
ALPACA_ENDPOINT = get_secret("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets")

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

st.sidebar.header("💼 Account Summary (Paper Trading)")
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
st.sidebar.header("⚙️ Strategy Settings")
short_window = st.sidebar.number_input("📏 Short-term MA window", min_value=1, max_value=100, value=20, step=1)
long_window = st.sidebar.number_input("📐 Long-term MA window", min_value=1, max_value=200, value=50, step=1)

st.sidebar.header("🔮 ML Forecast Settings")
history_options = {
    "~2 years": 365,
    "~5 years": 913,
    "~10 years": 1825,
    "~15 years": 2738,
}
history_label = st.sidebar.selectbox("Historical data lookback", list(history_options.keys()), index=0)
history_days = history_options[history_label]
forecast_horizon = st.sidebar.slider("Forecast horizon", min_value=5, max_value=60, value=30, step=5)
forecast_lookback = st.sidebar.slider("Lag window", min_value=5, max_value=60, value=20, step=5)
historical_test_points = st.sidebar.slider("Previous forecast test points", min_value=10, max_value=100, value=50, step=10)
optimize_forecast_model = st.sidebar.checkbox("Optimize ML model", value=True)
use_market_context = st.sidebar.checkbox("Use market context features", value=True)
primary_model_choice = st.sidebar.selectbox("Primary forecast model", ["Ensemble", "Best Validation", "Ridge", "XGBoost"], index=0)
forecast_alpha = st.sidebar.number_input(
    "Ridge regularization",
    min_value=0.1,
    max_value=100.0,
    value=10.0,
    step=0.5,
)
run_forecast_rankings = st.sidebar.checkbox("Run forecast rankings", value=True)
selected_forecast_symbols = st.sidebar.multiselect(
    "Forecast ranking tickers",
    options=SYMBOL_OPTIONS,
    default=DEFAULT_FORECAST_SYMBOLS,
    format_func=ticker_label,
)

# --- Symbol input ---
symbol = st.selectbox(
    "🏷️ Symbol",
    options=SYMBOL_OPTIONS,
    index=SYMBOL_OPTIONS.index("AAPL"),
    format_func=ticker_label,
    key="symbol_select",
)

# --- Manual Trade Buttons ---
col1, col2 = st.columns(2)
with col1:
    if st.button(f"🟢 Buy {symbol}"):
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
    if st.button(f"🔴 Sell {symbol}"):
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
    df = load_ohlcv(symbol, history_days)
    
    sig = sma_crossover(df, short_window, long_window)
    bt = simple_vector_backtest(df, sig)
    actual_close = df["close"]
    if isinstance(actual_close, pd.DataFrame):
        actual_close = actual_close.iloc[:, 0]
    actual_close = pd.to_numeric(actual_close, errors="coerce").dropna()
    context_df = load_market_context(history_days) if use_market_context else pd.DataFrame()
    
    st.subheader("📈 Actual Value, ML Based Forecast, and Crossover Strategy")
    try:
        model_results = cached_model_results(
            symbol,
            history_days,
            forecast_horizon,
            forecast_lookback,
            forecast_alpha,
            optimize_forecast_model,
            use_market_context,
        )
        if primary_model_choice == "Best Validation":
            primary_model = best_model_name(model_results, preferred="")
        else:
            primary_model = best_model_name(model_results, preferred=primary_model_choice)
        if not primary_model:
            raise ValueError("No forecast model produced a usable forecast.")

        forecast_result = model_results[primary_model]
        forecast_df = forecast_result.forecast
        historical_result = None
        historical_forecasts = pd.DataFrame()
        forecast_change = forecast_result.metrics.get("forecast_change_pct", 0.0)
        probability_up = forecast_result.metrics.get("probability_up_pct", 0.0)
        probability_down = forecast_result.metrics.get("probability_down_pct", 0.0)
        confidence = forecast_result.metrics.get("confidence_pct", 0.0)
        model_edge = max(confidence - 50.0, 0.0)
        quality_label = signal_quality(confidence)
        expected_error = forecast_result.metrics.get("expected_error_pct", 0.0)
        selected_window = forecast_result.metrics.get("selected_lookback_window", forecast_lookback)
        selected_alpha = forecast_result.metrics.get("selected_ridge_alpha", forecast_alpha)
        shrink_factor = forecast_result.metrics.get("shrink_factor", 1.0)
        raw_forecast = forecast_result.metrics.get("raw_forecast_change_pct", forecast_change)
        try:
            historical_result = cached_historical_forecasts(
                symbol,
                history_days,
                forecast_horizon,
                int(selected_window),
                float(selected_alpha),
                historical_test_points,
                primary_model,
                use_market_context,
            )
            historical_forecasts = historical_result.forecasts
        except Exception as history_error:
            st.info(f"Previous ML forecast test unavailable: {history_error}")

        latest_actual_date = actual_close.index[-1]
        latest_actual_price = float(actual_close.iloc[-1])
        latest_actual_label = f"Actual: ${latest_actual_price:,.2f}"
        final_forecast_date = forecast_df.index[-1]
        final_forecast_price = float(forecast_df["forecast_close"].iloc[-1])
        final_forecast_label = f"{primary_model}: ${final_forecast_price:,.2f} ({forecast_change:+.2f}%)"
        forecast_plot_x = [latest_actual_date, *forecast_df.index.tolist()]
        forecast_plot_y = [latest_actual_price, *forecast_df["forecast_close"].tolist()]

        price_fig = go.Figure()
        price_fig.add_trace(
            go.Scatter(
                x=actual_close.index,
                y=actual_close,
                mode="lines",
                name="Actual Value",
                line=dict(color="#1f77b4", width=2),
                hovertemplate="%{x}<br>Actual value: $%{y:,.2f}<extra></extra>",
            )
        )
        price_fig.add_trace(
            go.Scatter(
                x=forecast_df.index,
                y=forecast_df["upper_estimate"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        price_fig.add_trace(
            go.Scatter(
                x=forecast_df.index,
                y=forecast_df["lower_estimate"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(44, 160, 44, 0.15)",
                name="90% estimate range",
                hoverinfo="skip",
            )
        )
        price_fig.add_trace(
            go.Scatter(
                x=forecast_plot_x,
                y=forecast_plot_y,
                mode="lines",
                name=f"Primary ML Forecast ({primary_model})",
                line=dict(color="#2ca02c", width=2, dash="dash"),
                hovertemplate="%{x}<br>Primary ML forecast: $%{y:,.2f}<extra></extra>",
            )
        )
        comparison_colors = {"Ridge": "#8c564b", "XGBoost": "#d62728", "Ensemble": "#2ca02c"}
        for comparison_name, comparison_result in model_results.items():
            if comparison_name == primary_model or comparison_result.forecast is None or comparison_result.forecast.empty:
                continue

            price_fig.add_trace(
                go.Scatter(
                    x=[latest_actual_date, *comparison_result.forecast.index.tolist()],
                    y=[latest_actual_price, *comparison_result.forecast["forecast_close"].tolist()],
                    mode="lines",
                    name=f"{comparison_name} Forecast",
                    line=dict(
                        color=comparison_colors.get(comparison_name, "#7f7f7f"),
                        width=1.5,
                        dash="dot",
                    ),
                    hovertemplate=f"%{{x}}<br>{comparison_name}: $%{{y:,.2f}}<extra></extra>",
                )
            )
        if not historical_forecasts.empty:
            error_line_x = []
            error_line_y = []
            for _, row in historical_forecasts.iterrows():
                error_line_x.extend([row["forecast_date"], row["forecast_date"], None])
                error_line_y.extend([row["actual_close"], row["forecast_close"], None])

            price_fig.add_trace(
                go.Scatter(
                    x=error_line_x,
                    y=error_line_y,
                    mode="lines",
                    name="Previous Forecast Error",
                    line=dict(color="rgba(255, 127, 14, 0.25)", width=1),
                    hoverinfo="skip",
                )
            )
            price_fig.add_trace(
                go.Scatter(
                    x=historical_forecasts["forecast_date"],
                    y=historical_forecasts["forecast_close"],
                    mode="markers",
                    name="Previous ML Forecasts",
                    marker=dict(color="#ff7f0e", size=7, symbol="diamond"),
                    customdata=historical_forecasts[
                        ["actual_close", "error_pct", "confidence_pct", "expected_error_pct"]
                    ],
                    hovertemplate=(
                        "%{x}<br>"
                        "Previous ML forecast: $%{y:,.2f}<br>"
                        "Actual close: $%{customdata[0]:,.2f}<br>"
                        "Error: %{customdata[1]:+.2f}%<br>"
                        "Confidence then: %{customdata[2]:.1f}%<br>"
                        "Expected error then: %{customdata[3]:.2f}%"
                        "<extra></extra>"
                    ),
                )
            )
        price_fig.add_trace(
            go.Scatter(
                x=[latest_actual_date],
                y=[latest_actual_price],
                mode="markers+text",
                name="Latest Actual Value",
                marker=dict(color="#1f77b4", size=10, line=dict(color="white", width=2)),
                text=[latest_actual_label],
                textposition="bottom left",
                textfont=dict(color="#0d47a1", size=13),
                hovertemplate="%{x}<br>Latest actual value: $%{y:,.2f}<extra></extra>",
            )
        )
        price_fig.add_trace(
            go.Scatter(
                x=[final_forecast_date],
                y=[final_forecast_price],
                mode="markers+text",
                name=f"{forecast_horizon}-Day {primary_model} Forecast",
                marker=dict(color="#2ca02c", size=10, line=dict(color="white", width=2)),
                text=[final_forecast_label],
                textposition="top right",
                textfont=dict(color="#1b5e20", size=13),
                hovertemplate="%{x}<br>ML based forecast value: $%{y:,.2f}<extra></extra>",
            )
        )
        price_fig.add_trace(
            go.Scatter(
                x=bt.index,
                y=bt["curve"],
                mode="lines",
                name="Using the Crossover Strategy",
                line=dict(color="#9467bd", width=2),
                yaxis="y2",
                hovertemplate="%{x}<br>Crossover strategy equity: %{y:.4f}<extra></extra>",
            )
        )
        price_fig.update_layout(
            xaxis_title="Date",
            yaxis=dict(title="Actual / Forecast Price"),
            yaxis2=dict(
                title="Crossover Strategy Equity Curve",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            hovermode="x unified",
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(price_fig, use_container_width=True)

        metric_cols = st.columns(3)
        metric_cols[0].metric(
            f"{forecast_horizon}-day forecast",
            f"${final_forecast_price:,.2f}",
            f"{forecast_change:.2f}%",
        )
        metric_cols[1].metric("Probability Up", f"{probability_up:.1f}%")
        metric_cols[2].metric("Expected Error", f"±{expected_error:.2f}%")

        quality_cols = st.columns(3)
        quality_cols[0].metric("Model Edge", f"{model_edge:.1f}%", quality_label)
        if historical_result is not None:
            quality_cols[1].metric(
                "Previous Forecast MAE",
                f"{historical_result.metrics['historical_mae_pct']:.2f}%",
            )
            quality_cols[2].metric(
                "Previous Direction Hit Rate",
                f"{historical_result.metrics['historical_direction_accuracy']:.1f}%",
            )
        elif "holdout_direction_accuracy" in forecast_result.metrics:
            quality_cols[1].metric(
                "Holdout Direction",
                f"{forecast_result.metrics['holdout_direction_accuracy']:.1f}%",
            )
            quality_cols[2].metric(
                "Holdout MAE",
                f"{forecast_result.metrics['holdout_mae_pct']:.2f}%",
            )
        else:
            quality_cols[1].metric("Training Samples", forecast_result.metrics.get("training_samples", 0))
            quality_cols[2].metric("Probability Down", f"{probability_down:.1f}%")

        st.caption(
            "Model: "
            f"{forecast_result.model_name}. "
            f"Primary model: {primary_model}. "
            f"Market context features: {'on' if use_market_context and not context_df.empty else 'off'}. "
            f"Selected lag window: {selected_window}; "
            f"ridge alpha: {selected_alpha:g}; "
            f"raw forecast: {raw_forecast:+.2f}%; "
            f"calibrated forecast: {forecast_change:+.2f}%; "
            f"shrink factor: {shrink_factor:.2f}."
        )
        if quality_label == "No Edge":
            st.info(
                "The optimized model is effectively neutral here: directional probability is close to 50%, "
                "so the current feature set does not justify a strong forecast."
            )

        with st.expander("Model comparison"):
            comparison_table = model_results_table(model_results)
            st.dataframe(
                comparison_table.style.format(
                    {
                        "Forecast Price": "${:,.2f}",
                        "Forecast Return %": "{:+.2f}%",
                        "Probability Up %": "{:.1f}%",
                        "Model Edge %": "{:.1f}%",
                        "Expected Error %": "{:.2f}%",
                        "Validation MAE %": "{:.2f}%",
                        "Validation Direction %": "{:.1f}%",
                        "Score": "{:+.3f}",
                    },
                    na_rep="",
                ),
                use_container_width=True,
            )

        if historical_result is not None:
            with st.expander("Previous forecast accuracy"):
                st.caption(
                    "Each row shows what the ML model would have forecasted using only data available at that past date."
                )
                display_history = historical_forecasts.rename(
                    columns={
                        "as_of_date": "Forecast Made On",
                        "forecast_date": "Forecast Target Date",
                        "as_of_price": "Price Then",
                        "forecast_close": "Forecast Close",
                        "actual_close": "Actual Close",
                        "predicted_change_pct": "Forecast Return %",
                        "actual_change_pct": "Actual Return %",
                        "error_pct": "Error %",
                        "abs_error_pct": "Absolute Error %",
                        "confidence_pct": "Confidence %",
                        "expected_error_pct": "Expected Error %",
                        "direction_correct": "Direction Correct",
                    }
                )
                st.dataframe(display_history)

        with st.expander("Forecast values"):
            display_forecast = forecast_df.rename(
                columns={
                    "forecast_close": "Forecast Close",
                    "lower_estimate": "Lower Estimate",
                    "upper_estimate": "Upper Estimate",
                    "expected_daily_return_pct": "Expected Daily Return %",
                }
            )
            st.dataframe(display_forecast)
    except Exception as forecast_error:
        st.line_chart(actual_close)
        st.warning(f"ML forecast unavailable: {forecast_error}")
    
    # --- Latest Signal + Timestamp ---
    signal_emoji = "BUY" if sig.iloc[-1] == 1 else "FLAT"
    st.write(f"**✨ Latest Signal:** {signal_emoji}")
    st.caption(f"Last updated {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC}")
    
except Exception as e:
    st.error(f" Error loading market data: {e}")
    st.info("Please check if the symbol is valid and try again.")

st.markdown("---")

# --- Forecast ranking tables ---
if run_forecast_rankings:
    st.subheader("🔎 ML Based Forecast Rankings")
    ranking_symbols = selected_forecast_symbols

    if not ranking_symbols:
        st.info("Add at least one symbol in the forecast ranking symbols box.")
    else:
        with st.spinner("Loading cached forecast rankings..."):
            ranking_table, ranking_errors = cached_forecast_rankings(
                tuple(ranking_symbols),
                history_days,
                forecast_horizon,
                forecast_lookback,
                forecast_alpha,
                optimize_forecast_model,
                use_market_context,
                primary_model_choice,
            )

        if not ranking_table.empty:
            buy_candidates = ranking_table[ranking_table["Forecast Return %"] > 0]
            sell_candidates = ranking_table[ranking_table["Forecast Return %"] < 0]
            strongest_buy = buy_candidates.sort_values("Score", ascending=False).head(10)
            strongest_sell = sell_candidates.sort_values("Score", ascending=True).head(10)

            buy_col, sell_col = st.columns(2)
            with buy_col:
                st.markdown("**Strongest Buy Forecasts**")
                if strongest_buy.empty:
                    st.info("No positive forecast candidates.")
                else:
                    st.dataframe(format_ranking_table(strongest_buy), use_container_width=True)
            with sell_col:
                st.markdown("**Strongest Sell Forecasts**")
                if strongest_sell.empty:
                    st.info("No negative forecast candidates.")
                else:
                    st.dataframe(format_ranking_table(strongest_sell), use_container_width=True)

            with st.expander("All forecast ranking results"):
                st.dataframe(format_ranking_table(ranking_table.sort_values("Score", ascending=False)), use_container_width=True)

            st.caption(
                "Rankings are model outputs based on forecast return, confidence, and expected error. They are not financial advice."
            )

        if ranking_errors:
            with st.expander("Symbols skipped during forecast ranking"):
                st.write("\n".join(ranking_errors))

# Debug info (only show in sidebar if needed)
if st.sidebar.checkbox("🐞 Show Debug Info", value=False):
    try:
        r = requests.get("https://paper-api.alpaca.markets/v2/orders", headers={
            "APCA-API-KEY-ID": ALPACA_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET,
        })
        st.sidebar.json(r.json())
    except Exception as e:
        st.sidebar.error(f"Failed to fetch orders: {e}")
