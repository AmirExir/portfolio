import os
import requests
import json
from typing import Optional, Dict, Any
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import RandomForestRegressor
    HAS_XGBOOST = False
from sklearn.preprocessing import StandardScaler


class ErcotAPI:
    """
    Simple client for interacting with ERCOT Public Reports API (https://api.ercot.com/api/public-reports).
    Requires environment variables or direct API key input.
    """

    def __init__(
            self,
            public_key: Optional[str] = None,
            bearer_token: Optional[str] = None,
            username: Optional[str] = None,
            password: Optional[str] = None,
            client_id: Optional[str] = None,
            subscription_key: Optional[str] = None,
        ):
        self.public_key = public_key or os.getenv("ERCOT_PUBLIC_KEY")
        self.bearer_token = bearer_token or os.getenv("ERCOT_BEARER_TOKEN")
        self.subscription_key = subscription_key or os.getenv("ERCOT_SUBSCRIPTION_KEY")
        username = username or os.getenv("ERCOT_USERNAME")
        password = password or os.getenv("ERCOT_PASSWORD")
        client_id = client_id or os.getenv("ERCOT_CLIENT_ID")

        if not self.bearer_token and username and password and client_id:
            self.get_bearer_token(username, password, client_id, verbose=False)

    def get_bearer_token(self, username: str, password: str, client_id: str, scope: str = "openid fec253ea-0d06-4272-a5e6-b478baeecd70 offline_access", verbose: bool = False) -> None:
        """Retrieve a bearer token from ERCOT's OAuth endpoint and store it."""
        token_url = "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
        data = {
            "grant_type": "password",
            "scope": scope,
            "client_id": client_id,
            "username": username,
            "password": password,
        }
        if verbose:
            print("🔐 Requesting bearer token from ERCOT...")
        
        try:
            response = requests.post(token_url, data=data)
            response.raise_for_status()
            token_info = response.json()
            self.bearer_token = token_info.get("access_token") 
            if self.bearer_token:
                if verbose:
                    print(" Bearer token acquired.")
            else:
                raise ValueError("Failed to obtain bearer token.")
        except requests.exceptions.HTTPError as e:
            # Try to extract error details from response
            error_detail = ""
            try:
                error_json = e.response.json()
                error_detail = f"\nError details: {error_json.get('error_description', error_json)}"
            except:
                error_detail = f"\nResponse: {e.response.text[:200]}"
            
            raise ValueError(f"Authentication failed: {str(e)}{error_detail}\n\nPlease check your username, password, and client ID.")

    def _make_request(self, base_url: str, endpoint: str, key: str, params: Optional[Dict[str, Any]] = None, verbose: bool = False, max_retries: int = 3) -> Dict:
        """Internal method to send a GET request to ERCOT API with retry logic for rate limits."""
        url = f"{base_url}/{endpoint.lstrip('/')}"
        
        # Prepare headers
        headers = {}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        if key or self.subscription_key:
            headers["Ocp-Apim-Subscription-Key"] = key or self.subscription_key
        
        if verbose:
            print(f" Requesting: {url}")
            print(f" Params: {params}")
            print(f" Headers: {list(headers.keys())}")
        
        # Retry logic for rate limiting
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit exceeded
                    if attempt < max_retries - 1:
                        # Exponential backoff: wait 2^attempt seconds
                        wait_time = 2 ** attempt
                        if verbose:
                            print(f"⏳ Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                    else:
                        raise  # Re-raise if max retries reached
                else:
                    raise  # Re-raise for non-429 errors

    def get_public(self, endpoint: str, params: Optional[Dict[str, Any]] = None, verbose: bool = False, max_retries: int = 3) -> Dict:
        """Query ERCOT Public Reports API (base: https://api.ercot.com/api/public-reports)"""
        if not self.bearer_token and not self.subscription_key:
            raise ValueError("Missing ERCOT Bearer token or Subscription Key. You need BOTH authentication methods.")

        base_url = "https://api.ercot.com/api/public-reports"
        if verbose:
            print(f" Using Public Reports API: {base_url}/{endpoint}")

        return self._make_request(base_url, endpoint, self.public_key, params, verbose, max_retries)



# --- Helper Functions ---

# Cache API calls for 5 minutes to avoid rate limits
@st.cache_data(ttl=300, show_spinner=False)
def fetch_ercot_data_cached(endpoint: str, params_str: str, bearer_token: str, subscription_key: str):
    """Cached wrapper for ERCOT API calls. TTL=5 minutes to reduce rate limit issues."""
    import json
    params = json.loads(params_str) if params_str else None
    
    # Create temporary API instance
    api = ErcotAPI(bearer_token=bearer_token, subscription_key=subscription_key)
    return api.get_public(endpoint, params=params, verbose=False, max_retries=3)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_news_file_index(path: str = "ERCOTAPI"):
    """Fetch file index from GitHub for n8n-generated news files (cached for 5 minutes)."""
    contents_url = f"https://api.github.com/repos/AmirExir/portfolio/contents/{path.strip('/')}"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Streamlit-ERCOT-Dashboard",
    }
    response = requests.get(contents_url, headers=headers, timeout=12)
    if response.status_code == 403 and "rate limit" in response.text.lower():
        return None
    response.raise_for_status()
    files = response.json()
    return files if isinstance(files, list) else []


@st.cache_data(ttl=300, show_spinner=False)
def fetch_news_repo_tree(branch: str = "main"):
    """Fetch recursive file tree from GitHub as a fallback when folder listing is insufficient."""
    tree_url = f"https://api.github.com/repos/AmirExir/portfolio/git/trees/{branch}?recursive=1"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Streamlit-ERCOT-Dashboard",
    }
    response = requests.get(tree_url, headers=headers, timeout=12)
    if response.status_code == 403 and "rate limit" in response.text.lower():
        return []
    response.raise_for_status()
    payload = response.json()
    tree = payload.get("tree", []) if isinstance(payload, dict) else []
    return tree if isinstance(tree, list) else []


def _normalize_news_text(raw_text: str) -> str:
    """Parse text payloads that may be plain text or JSON from n8n outputs."""
    try:
        payload = json.loads(raw_text)
        if isinstance(payload, dict):
            return str(payload.get("content") or payload.get("message") or payload)
        if isinstance(payload, list):
            return "\n".join(str(item) for item in payload)
        return str(payload)
    except json.JSONDecodeError:
        return raw_text


def _read_latest_local_news(local_dir: str, prefixes) -> Optional[Dict[str, str]]:
    """Local fallback for Streamlit deployments when GitHub API is unavailable."""
    if not os.path.isdir(local_dir):
        return None

    candidates = []
    for root, _, files in os.walk(local_dir):
        for name in files:
            lower_name = name.lower()
            if not lower_name.endswith((".txt", ".md", ".json")):
                continue
            if any(lower_name.startswith(p.lower()) for p in prefixes):
                rel_path = os.path.relpath(os.path.join(root, name), local_dir)
                candidates.append(rel_path)

    if not candidates:
        return None

    latest_name = sorted(candidates, reverse=True)[0]
    latest_path = os.path.join(local_dir, latest_name)
    try:
        with open(latest_path, "r", encoding="utf-8") as f:
            content = _normalize_news_text(f.read().strip())
        return {"name": os.path.basename(latest_name), "content": content}
    except Exception:
        return None


def get_latest_news_by_prefix(
    prefixes,
    repo_path: str = "ERCOTAPI",
    allow_heuristic_fallback: bool = True,
) -> Optional[Dict[str, str]]:
    """Return the latest matching news item using GitHub first, then local fallback."""
    candidate_paths = [repo_path, f"{repo_path}/market_agent"]

    for candidate_path in candidate_paths:
        try:
            files = fetch_news_file_index(candidate_path)
        except Exception:
            files = None

        if not files:
            continue

        matches = [
            f for f in files
            if f.get("type") == "file"
            and f.get("name", "").lower().endswith((".txt", ".md", ".json"))
            and any(f.get("name", "").lower().startswith(p.lower()) for p in prefixes)
        ]
        if matches:
            latest = sorted(matches, key=lambda x: x.get("name", ""), reverse=True)[0]
            download_url = latest.get("download_url")
            if download_url:
                try:
                    r = requests.get(download_url, timeout=12)
                    r.raise_for_status()
                    return {
                        "name": latest.get("name", ""),
                        "content": _normalize_news_text(r.text.strip()),
                    }
                except Exception:
                    pass

    # Fallback: search full repo tree for older or manually placed summary files.
    try:
        tree_items = fetch_news_repo_tree("main")
    except Exception:
        tree_items = []

    if tree_items:
        tree_matches = []
        heuristic_matches = []
        for item in tree_items:
            if item.get("type") != "blob":
                continue
            rel_path = item.get("path", "")
            if not rel_path.lower().endswith((".txt", ".md", ".json")):
                continue
            base_name = os.path.basename(rel_path).lower()
            if any(base_name.startswith(p.lower()) for p in prefixes):
                tree_matches.append(rel_path)
            elif allow_heuristic_fallback and (
                base_name.startswith("summary_")
                or ("ercot" in base_name and ("news" in base_name or "summary" in base_name))
            ):
                heuristic_matches.append(rel_path)

        if not tree_matches and allow_heuristic_fallback and heuristic_matches:
            tree_matches = heuristic_matches

        if tree_matches:
            latest_path = sorted(tree_matches, reverse=True)[0]
            raw_url = (
                "https://raw.githubusercontent.com/AmirExir/portfolio/main/"
                f"{latest_path}"
            )
            try:
                r = requests.get(raw_url, timeout=12)
                r.raise_for_status()
                return {
                    "name": os.path.basename(latest_path),
                    "content": _normalize_news_text(r.text.strip()),
                }
            except Exception:
                pass

    # Fallback to local files in ERCOTAPI directory
    local_dir = os.path.dirname(os.path.abspath(__file__))
    return _read_latest_local_news(local_dir, prefixes)


def make_arrow_safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert problematic object-dtype datetime values to strings for Streamlit Arrow serialization."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    safe_df = df.copy()
    object_cols = safe_df.select_dtypes(include=['object']).columns

    for col in object_cols:
        series = safe_df[col]
        non_null = series.dropna()
        if non_null.empty:
            continue

        has_datetime_objects = non_null.map(
            lambda v: isinstance(v, (pd.Timestamp, datetime, np.datetime64))
        ).any()

        if has_datetime_objects:
            safe_df[col] = series.map(
                lambda v: v.isoformat()
                if isinstance(v, (pd.Timestamp, datetime))
                else (pd.Timestamp(v).isoformat() if isinstance(v, np.datetime64) else v)
            )

    return safe_df


APP_BUILD = "2026-06-18 professional-dashboard-v2"
ERCOT_BLUE = "#0b2f4f"
ERCOT_CYAN = "#00a3c7"
ERCOT_GREEN = "#1f9d55"
ERCOT_ORANGE = "#f59e0b"
ERCOT_RED = "#dc2626"
CHART_TEMPLATE = "plotly_white"


def inject_dashboard_css() -> None:
    """Apply a polished Streamlit theme without changing global app behavior."""
    st.markdown(
        """
        <style>
        :root {
            --ercot-blue: #0b2f4f;
            --ercot-cyan: #00a3c7;
            --ercot-green: #1f9d55;
            --ercot-orange: #f59e0b;
            --ercot-red: #dc2626;
            --ercot-slate: #334155;
            --ercot-muted: #64748b;
            --ercot-panel: #ffffff;
            --ercot-border: #e2e8f0;
            --ercot-bg: #f8fafc;
        }
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
            max-width: 1420px;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b2f4f 0%, #123b63 100%);
        }
        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] select {
            color: #0f172a !important;
        }
        [data-testid="stSidebar"] .stButton > button {
            border: 1px solid rgba(255,255,255,0.22);
            background: rgba(255,255,255,0.10);
            color: #ffffff;
        }
        .dashboard-hero {
            padding: 1.35rem 1.45rem;
            border-radius: 22px;
            background:
                radial-gradient(circle at top right, rgba(0,163,199,0.25), transparent 32%),
                linear-gradient(135deg, #071f36 0%, #0b2f4f 48%, #123b63 100%);
            color: #ffffff;
            border: 1px solid rgba(255,255,255,0.12);
            box-shadow: 0 20px 45px rgba(15, 23, 42, 0.18);
            margin-bottom: 1rem;
        }
        .dashboard-hero h1 {
            margin: 0;
            font-size: 2.35rem;
            line-height: 1.05;
            letter-spacing: -0.045em;
        }
        .dashboard-hero p {
            margin: 0.55rem 0 0 0;
            color: #dbeafe;
            font-size: 1.02rem;
            max-width: 860px;
        }
        .hero-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }
        .hero-badge {
            border: 1px solid rgba(255,255,255,0.22);
            border-radius: 999px;
            padding: 0.32rem 0.72rem;
            background: rgba(255,255,255,0.10);
            color: #f8fafc;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .section-title {
            margin: 0.25rem 0 0.25rem 0;
            color: var(--ercot-blue);
            letter-spacing: -0.025em;
        }
        .section-subtitle {
            margin-top: -0.2rem;
            margin-bottom: 0.7rem;
            color: var(--ercot-muted);
            font-size: 0.95rem;
        }
        .metric-card {
            border: 1px solid var(--ercot-border);
            border-radius: 18px;
            background: var(--ercot-panel);
            padding: 1rem 1.05rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            min-height: 112px;
        }
        .metric-label {
            color: var(--ercot-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.72rem;
            font-weight: 700;
        }
        .metric-value {
            color: #0f172a;
            font-size: 1.75rem;
            line-height: 1.15;
            font-weight: 800;
            letter-spacing: -0.035em;
            margin-top: 0.25rem;
        }
        .metric-note {
            color: var(--ercot-muted);
            font-size: 0.82rem;
            margin-top: 0.35rem;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.25rem 0.68rem;
            font-size: 0.78rem;
            font-weight: 700;
            border: 1px solid rgba(15,23,42,0.08);
        }
        .status-ok { background: #ecfdf5; color: #047857; }
        .status-warn { background: #fffbeb; color: #b45309; }
        .status-error { background: #fef2f2; color: #b91c1c; }
        .news-panel {
            border: 1px solid var(--ercot-border);
            border-radius: 20px;
            background: var(--ercot-panel);
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
        }
        .small-muted {
            color: var(--ercot-muted);
            font-size: 0.82rem;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--ercot-border);
            border-radius: 16px;
            padding: 0.85rem 0.95rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }
        div[data-testid="stMetric"] label {
            color: var(--ercot-muted) !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
            border-bottom: 1px solid var(--ercot-border);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.5rem 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        f"""
        <div class="dashboard-hero">
            <h1>ERCOT Grid Intelligence Dashboard</h1>
            <p>
                Real-time Texas grid analytics for load, renewable output, nodal pricing,
                outage capacity, AI-generated news, and machine-learning load forecasting.
            </p>
            <div class="hero-badges">
                <span class="hero-badge">Public Reports API</span>
                <span class="hero-badge">5-minute smart cache</span>
                <span class="hero-badge">ML load forecast</span>
                <span class="hero-badge">n8n news pipeline</span>
                <span class="hero-badge">Build {APP_BUILD}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"<h2 class='section-title'>{title}</h2>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='section-subtitle'>{subtitle}</div>", unsafe_allow_html=True)


def render_metric_card(label: str, value: str, note: str = "", accent: str = ERCOT_BLUE) -> None:
    st.markdown(
        f"""
        <div class="metric-card" style="border-top: 4px solid {accent};">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_pill(label: str, status: str = "ok") -> None:
    status_class = {
        "ok": "status-ok",
        "warn": "status-warn",
        "error": "status-error",
    }.get(status, "status-ok")
    st.markdown(
        f"<span class='status-pill {status_class}'>{label}</span>",
        unsafe_allow_html=True,
    )


def format_mw(value: Any) -> str:
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):,.0f} MW"
    except Exception:
        return "N/A"


def format_price(value: Any) -> str:
    try:
        if pd.isna(value):
            return "N/A"
        return f"${float(value):,.2f}/MWh"
    except Exception:
        return "N/A"


def format_percent(value: Any) -> str:
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.1f}%"
    except Exception:
        return "N/A"


def map_ercot_fields(df: pd.DataFrame, payload: Dict[str, Any]) -> pd.DataFrame:
    """Apply ERCOT `fields` metadata to API rows when rows arrive as positional arrays."""
    if df.empty:
        return df
    fields = payload.get("fields") if isinstance(payload, dict) else None
    if isinstance(fields, list):
        column_mapping = {i: field.get("name", f"col_{i}") for i, field in enumerate(fields)}
        df = df.rename(columns=column_mapping)
    return df


def dataframe_from_payload(payload: Dict[str, Any]) -> pd.DataFrame:
    data = payload.get("data", []) if isinstance(payload, dict) else []
    if not data:
        return pd.DataFrame()
    return map_ercot_fields(pd.DataFrame(data), payload)


def find_column(df: pd.DataFrame, exact: Optional[list[str]] = None, contains: Optional[list[str]] = None) -> Optional[str]:
    """Find a column by exact case-insensitive name first, then by required substring terms."""
    if df.empty:
        return None
    exact = exact or []
    contains = contains or []
    normalized = {str(col).lower(): col for col in df.columns}
    for name in exact:
        if name.lower() in normalized:
            return normalized[name.lower()]
    for col in df.columns:
        col_lower = str(col).lower()
        if contains and all(term.lower() in col_lower for term in contains):
            return col
    return None


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def latest_timestamp_label(df: pd.DataFrame) -> str:
    if "timestamp" not in df.columns or df.empty:
        return "Latest available interval"
    value = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
    if value.empty:
        return "Latest available interval"
    return value.max().strftime("%b %d, %Y %H:%M")


def apply_professional_layout(
    fig: go.Figure,
    title: str,
    yaxis_title: str,
    height: int = 470,
    legend_y: float = 1.04,
) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=20, color=ERCOT_BLUE)),
        template=CHART_TEMPLATE,
        height=height,
        hovermode="x unified",
        margin=dict(l=35, r=25, t=70, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=legend_y, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a"),
    )
    fig.update_xaxes(title_text="Time", showgrid=True, gridcolor="#eef2f7")
    fig.update_yaxes(title_text=yaxis_title, showgrid=True, gridcolor="#eef2f7", zerolinecolor="#cbd5e1")
    return fig


def render_dataframe(df: pd.DataFrame, height: int = 320) -> None:
    st.dataframe(make_arrow_safe_dataframe(df), width="stretch", height=height)


from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_load_forecast_model(historical_data, tuning_mode: str = "auto", manual_params: Optional[Dict[str, Any]] = None):
    """Train an ML model (XGBoost or Random Forest) to forecast load based on historical patterns.
       Returns: model, scaler, df, train/test split, predictions, and metrics for plotting/metrics display."""
    if len(historical_data) < 48:  # Need at least 2 days
        return None, None, None, None, None, None

    # Feature engineering: hour of day, day of week, rolling averages
    df = historical_data.copy()
    df['hour'] = pd.to_datetime(df.index).hour
    df['day_of_week'] = pd.to_datetime(df.index).dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Cyclical encoding for hour (23:00 and 00:00 are close)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Lag features
    df['load_lag_1h'] = df['load'].shift(1)
    df['load_lag_24h'] = df['load'].shift(24)

    # Rolling statistics
    df['rolling_mean_3h'] = df['load'].rolling(3, min_periods=1).mean()
    df['rolling_mean_24h'] = df['load'].rolling(24, min_periods=1).mean()
    df['rolling_std_24h'] = df['load'].rolling(24, min_periods=1).std()

    # Prepare features and target
    features = ['hour', 'day_of_week', 'is_weekend', 'hour_sin', 'hour_cos',
                'load_lag_1h', 'load_lag_24h', 'rolling_mean_3h', 'rolling_mean_24h', 'rolling_std_24h']

    df = df.dropna()

    if len(df) < 24:
        return None, None, None, None, None, None

    X = df[features]
    y = df['load']

    # Train/test split (80/20, no shuffle to preserve time order)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Train model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use time-series-aware validation for tuning.
    n_splits = min(5, max(2, len(X_train_scaled) // 10))
    cv = TimeSeriesSplit(n_splits=n_splits)

    best_params = {}
    best_cv_score = None

    if HAS_XGBOOST:
        if tuning_mode == "manual":
            manual_params = manual_params or {}
            best_params = {
                'n_estimators': int(manual_params.get('n_estimators', 100)),
                'max_depth': int(manual_params.get('max_depth', 6)),
                'learning_rate': float(manual_params.get('learning_rate', 0.1)),
                'subsample': float(manual_params.get('subsample', 0.8)),
                'colsample_bytree': float(manual_params.get('colsample_bytree', 0.8)),
                'min_child_weight': int(manual_params.get('min_child_weight', 1))
            }
            model = XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                verbosity=0,
                **best_params
            )
            model.fit(X_train_scaled, y_train)
        else:
            base_model = XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                verbosity=0
            )
            param_distributions = {
                'n_estimators': [100, 150, 200, 250, 300, 400],
                'max_depth': [3, 4, 5, 6, 7, 8, 10],
                'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'min_child_weight': [1, 3, 5]
            }
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_distributions,
                n_iter=20,
                scoring='neg_mean_absolute_error',
                cv=cv,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            search.fit(X_train_scaled, y_train)
            model = search.best_estimator_
            best_params = search.best_params_
            best_cv_score = -search.best_score_
    else:
        if tuning_mode == "manual":
            manual_params = manual_params or {}
            best_params = {
                'n_estimators': int(manual_params.get('n_estimators', 100)),
                'max_depth': manual_params.get('max_depth', 15),
                'min_samples_split': int(manual_params.get('min_samples_split', 5)),
                'min_samples_leaf': int(manual_params.get('min_samples_leaf', 1))
            }
            model = RandomForestRegressor(
                random_state=42,
                **best_params
            )
            model.fit(X_train_scaled, y_train)
        else:
            base_model = RandomForestRegressor(random_state=42)
            param_distributions = {
                'n_estimators': [100, 150, 200, 300, 400],
                'max_depth': [8, 12, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_distributions,
                n_iter=12,
                scoring='neg_mean_absolute_error',
                cv=cv,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            search.fit(X_train_scaled, y_train)
            model = search.best_estimator_
            best_params = search.best_params_
            best_cv_score = -search.best_score_

    # Predict on train and test sets
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Compute metrics
    metrics = {
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "train_r2": r2_score(y_train, y_train_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "test_r2": r2_score(y_test, y_test_pred),
        "best_params": best_params,
        "cv_mae": best_cv_score,
        "tuning_mode": tuning_mode
    }

    # For plotting: return the indices for X_train and X_test (to allow time series plotting)
    return model, scaler, df, (X_train, X_test, y_train, y_test), (y_train_pred, y_test_pred), metrics


# --- Streamlit Dashboard ---
def main():
    st.set_page_config(
        page_title="ERCOT Grid Intelligence Dashboard",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_dashboard_css()
    render_hero()

    action_col_1, action_col_2, action_col_3, action_col_4 = st.columns([1.1, 1.1, 1.3, 5.5])
    with action_col_1:
        if st.button("Refresh data", help="Clear cached API/news data and reload the dashboard"):
            st.cache_data.clear()
            st.rerun()
    with action_col_2:
        st.link_button("Telegram", "https://t.me/ERCOTNEWS", help="Open the ERCOT News Telegram channel")
    with action_col_3:
        st.link_button("ERCOT API Market", "https://apimarket.ercot.com/", help="Open ERCOT API Market")
    with action_col_4:
        st.markdown(
            f"<div class='small-muted' style='text-align:right; padding-top:0.55rem;'>"
            f"Last dashboard render: {datetime.now().strftime('%b %d, %Y %H:%M:%S')} | Build {APP_BUILD}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Unified news prefixes (ERCOT + regulatory updates in one panel)
    all_news_prefixes = [
        "ercot_news_", "ercot_summary_", "summary_ercot_",
        "summary_",
        "datacenter_news_", "data_center_news_", "dc_news_",
        "nogrr_", "nodal_operating_guide_", "nodal_guide_change_",
        "pgrr_", "planning_guide_change_", "planning_guide_update_",
    ]

    item = get_latest_news_by_prefix(all_news_prefixes, repo_path="ERCOTAPI/news_summaries")
    st.markdown("<div class='news-panel'>", unsafe_allow_html=True)
    news_header_col, news_status_col = st.columns([5, 1.4])
    with news_header_col:
        render_section_header(
            "ERCOT Intelligence Brief",
            "Latest n8n-generated Texas grid, market-rule, interconnection, and large-load monitoring summary.",
        )
    with news_status_col:
        render_status_pill("Live pipeline" if item else "Waiting for update", "ok" if item else "warn")
    if item:
        st.caption(f"Latest file: {item['name']}")
        st.markdown(item["content"])
    else:
        st.info("Awaiting n8n workflow updates. The dashboard will display the newest summary when a file lands.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Sidebar for configuration
    st.sidebar.title("Control Center")
    st.sidebar.caption("Credentials, data window, model tuning, and diagnostics.")
    
    # Credentials section
    st.sidebar.subheader("API Credentials")
    
    # Check if credentials are in environment or Streamlit secrets
    # Check for non-empty values
    env_username = os.getenv("ERCOT_USERNAME", "").strip()
    env_password = os.getenv("ERCOT_PASSWORD", "").strip()
    env_client_id = os.getenv("ERCOT_CLIENT_ID", "").strip()
    env_sub_key = os.getenv("ERCOT_SUBSCRIPTION_KEY", "").strip()
    
    has_env_creds = bool(env_username and env_password and env_client_id and env_sub_key)
    has_secrets = False
    credential_source = "Not configured"
    
    # Try Streamlit secrets (from .streamlit/secrets.toml or Streamlit Cloud secrets manager)
    if not has_env_creds:
        try:
            has_secrets = bool(st.secrets.get("ERCOT_USERNAME") and st.secrets.get("ERCOT_PASSWORD") and 
                             st.secrets.get("ERCOT_CLIENT_ID") and st.secrets.get("ERCOT_SUBSCRIPTION_KEY"))
        except Exception:
            pass
    
    # Check if user wants to save credentials to session state (for this session only)
    if 'saved_credentials' not in st.session_state:
        st.session_state.saved_credentials = None
    
    # Check session state for saved credentials
    if st.session_state.saved_credentials:
        credential_source = "Session memory"
        st.sidebar.success("Using saved credentials for this session")
        try:
            api = ErcotAPI(**st.session_state.saved_credentials)
        except Exception as e:
            st.sidebar.error("Failed to authenticate")
            st.error(f"**Authentication Error:** {str(e)}")
            st.session_state.saved_credentials = None
            api = None
    elif has_env_creds:
        credential_source = "Environment variables"
        st.sidebar.success("Using environment credentials")
        try:
            api = ErcotAPI()
        except Exception as e:
            st.sidebar.error("Failed to authenticate with environment credentials")
            st.error(f"""
            **Authentication Error:** {str(e)}
            
            **Common Issues:**
            1. **Wrong password** - Double-check your ERCOT_PASSWORD
            2. **Wrong username** - Should be your email address
            3. **Expired credentials** - You may need to reset your password at https://apimarket.ercot.com/
            4. **Client ID mismatch** - Verify ERCOT_CLIENT_ID is correct
            
            **To fix:**
            - Check your environment variables with: `echo $ERCOT_USERNAME`
            - Or use the manual input below
            """)
            api = None
    elif has_secrets:
        credential_source = "Streamlit secrets"
        st.sidebar.success("Using Streamlit secrets")
        st.sidebar.info("Credentials are loaded from `.streamlit/secrets.toml` or Streamlit Cloud secrets.")
        try:
            api = ErcotAPI(
                username=st.secrets["ERCOT_USERNAME"],
                password=st.secrets["ERCOT_PASSWORD"],
                client_id=st.secrets["ERCOT_CLIENT_ID"],
                subscription_key=st.secrets["ERCOT_SUBSCRIPTION_KEY"]
            )
        except Exception as e:
            st.sidebar.error(f"Failed to authenticate with secrets")
            st.error(f"""
            **Authentication Error:** {str(e)}
            
            **To fix:** Edit `.streamlit/secrets.toml` and verify all credentials are correct
            """)
            api = None
    else:
        st.sidebar.warning("No persistent credentials found. Enter credentials manually:")
        
        with st.sidebar.expander("Enter ERCOT API Credentials", expanded=True):
            st.markdown("**Step 1: Get Subscription Key**")
            st.markdown("1. Go to [ERCOT API Market](https://apimarket.ercot.com/)")
            st.markdown("2. Sign in and navigate to 'Products'")
            st.markdown("3. Subscribe to 'ERCOT Public API' (free tier)")
            st.markdown("4. Copy your **Subscription Key** from your profile")
            st.markdown("---")
            st.markdown("**Step 2: Enter Credentials**")
            
            username = st.text_input("Username (Email)", type="default", help="Your ERCOT portal username/email")
            password = st.text_input("Password", type="password", help="Your ERCOT portal password")
            subscription_key = st.text_input("Subscription Key", type="password", help="From ERCOT API Market portal")
            client_id = st.text_input("Client ID", value="fec253ea-0d06-4272-a5e6-b478baeecd70", help="ERCOT API Client ID (default)")
            
            if username and password and client_id and subscription_key:
                try:
                    api = ErcotAPI(username=username, password=password, client_id=client_id, subscription_key=subscription_key)
                    credential_source = "Manual input"
                    st.sidebar.success("Credentials accepted. Bearer token acquired.")
                    
                    # Ask if user wants to save for this session
                    save_creds = st.sidebar.checkbox(" Remember credentials for this session", value=False,
                                                     help="Credentials will be stored in memory (not saved to disk) for this browser session only")
                    if save_creds:
                        st.session_state.saved_credentials = {
                            'username': username,
                            'password': password,
                            'client_id': client_id,
                            'subscription_key': subscription_key
                        }
                        credential_source = "Session memory"
                        st.sidebar.success("Credentials saved for this session.")
                        st.sidebar.info("Credentials are held in browser session memory only, not written to disk.")
                except Exception as e:
                    st.sidebar.error(f"Authentication failed: {e}")
                    api = None
            else:
                st.sidebar.info("Please enter all credentials above to continue.")
                api = None
    
    # Only show the rest of the app if API is initialized
    if api is None:
        st.warning(" Please configure API credentials to access ERCOT data.")
        
        # Check if running on Streamlit Cloud
        is_cloud = os.getenv("STREAMLIT_SHARING_MODE") or os.getenv("STREAMLIT_CLOUD")
        
        if is_cloud:
            st.error("""
            **🚀 Streamlit Cloud Deployment Detected**
            
            To configure secrets on Streamlit Cloud:
            1. Go to your app dashboard: https://share.streamlit.io/
            2. Click on your app
            3. Click the ** Settings** button (three dots menu)
            4. Select **"Secrets"** from the left sidebar
            5. Paste the following format:
            
            ```toml
            ERCOT_USERNAME = "your_email@example.com"
            ERCOT_PASSWORD = "your_password"
            ERCOT_CLIENT_ID = "fec253ea-0d06-4272-a5e6-b478baeecd70"
            ERCOT_SUBSCRIPTION_KEY = "your_subscription_key"
            ```
            
            6. Click **"Save"**
            7. Your app will automatically restart with the credentials!
            
            🔒 **These credentials are encrypted and never visible in your code or GitHub!**
            """)
        
        st.info("""
        **How to get ERCOT API credentials:**
        
        **Important:** You need TWO things:
        1. **Subscription Key** (Primary requirement):
           - Register at https://apimarket.ercot.com/
           - Sign in and click "Products" → "ERCOT Public API"
           - Click "Subscribe" (free tier available)
           - Go to "Profile" → "Subscriptions" to find your **Primary Key**
           - This is your **Subscription Key**
        
        2. **Login Credentials** (Username & Password):
           - Use the same username/email and password from your ERCOT portal account
           - The Client ID is pre-filled (default value)
        """)
        return

    st.sidebar.divider()
    st.sidebar.subheader("Operations")
    st.sidebar.markdown(f"**Credential source:** {credential_source}")
    st.sidebar.markdown("**API cache:** 5 minutes")
    
    # Date range selector
    date_range = st.sidebar.slider(
        "Historical days to load",
        min_value=1,
        max_value=90,
        value=30,
        help="Number of days of historical data to fetch"
    )
    
    # Debug mode
    debug_mode = st.sidebar.checkbox("Debug mode", value=False, help="Show detailed API request/response info")

    # Load model tuning controls
    if "load_model_auto_tune" not in st.session_state:
        st.session_state.load_model_auto_tune = True

    st.sidebar.subheader("Load Forecast Model")
    mode_col_1, mode_col_2 = st.sidebar.columns(2)
    with mode_col_1:
        if st.button("Auto-optimize"):
            st.session_state.load_model_auto_tune = True
    with mode_col_2:
        if st.button("Manual"):
            st.session_state.load_model_auto_tune = False

    manual_load_params = {}
    if st.session_state.load_model_auto_tune:
        st.sidebar.success("Automatic tuning enabled")
        st.sidebar.caption("Uses time-series cross validation to select model settings from the current data window.")
        tuning_mode = "auto"
    else:
        st.sidebar.info("Manual tuning enabled")
        tuning_mode = "manual"
        if HAS_XGBOOST:
            manual_load_params["n_estimators"] = st.sidebar.slider("n_estimators", 50, 500, 200, 10)
            manual_load_params["max_depth"] = st.sidebar.slider("max_depth", 3, 12, 6, 1)
            manual_load_params["learning_rate"] = st.sidebar.slider("learning_rate", 0.01, 0.3, 0.1, 0.01)
            manual_load_params["subsample"] = st.sidebar.slider("subsample", 0.5, 1.0, 0.8, 0.05)
            manual_load_params["colsample_bytree"] = st.sidebar.slider("colsample_bytree", 0.5, 1.0, 0.8, 0.05)
            manual_load_params["min_child_weight"] = st.sidebar.slider("min_child_weight", 1, 10, 1, 1)
        else:
            manual_load_params["n_estimators"] = st.sidebar.slider("n_estimators", 50, 500, 200, 10)
            manual_load_params["max_depth"] = st.sidebar.slider("max_depth", 5, 30, 15, 1)
            manual_load_params["min_samples_split"] = st.sidebar.slider("min_samples_split", 2, 20, 5, 1)
            manual_load_params["min_samples_leaf"] = st.sidebar.slider("min_samples_leaf", 1, 10, 1, 1)
    
    st.sidebar.info("Smart caching is enabled to reduce ERCOT API rate-limit pressure. Use the top Refresh data button to clear cached API/news calls.")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=date_range)
    
    # Dashboard sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Load & Forecast", 
        "Renewables", 
        "Real-Time Pricing",
        "Resource Outages"
    ])
    
    # TAB 1: LOAD ANALYSIS & FORECAST
    with tab1:
        render_section_header(
            "System Load and Forecast",
            "Actual ERCOT system load, available ERCOT forecast data, and a local machine-learning 24-hour forecast.",
        )
        
        try:
            with st.spinner("📥 Fetching load data..."):
                # Fetch actual load data using correct parameters: operatingDayFrom/To
                load_params = {
                    "operatingDayFrom": start_date.strftime("%Y-%m-%d"),
                    "operatingDayTo": end_date.strftime("%Y-%m-%d"),
                    "page": 1,
                    "size": 5000
                }
                
                import json
                actual_load_data = fetch_ercot_data_cached(
                    "np6-345-cd/act_sys_load_by_wzn",
                    json.dumps(load_params),
                    api.bearer_token,
                    api.subscription_key
                )
                if debug_mode:
                    st.json({"params_used": load_params, "response_keys": list(actual_load_data.keys()) if isinstance(actual_load_data, dict) else "N/A"})
                
                try:
                    forecast_data = fetch_ercot_data_cached(
                        "np3-565-cd/lf_by_model_weather_zone",
                        json.dumps(load_params),
                        api.bearer_token,
                        api.subscription_key
                    )
                except Exception:
                    forecast_data = {"data": []}
                
                if "data" in actual_load_data and len(actual_load_data["data"]) > 0:
                    actual_df = dataframe_from_payload(actual_load_data)
                    forecast_df = dataframe_from_payload(forecast_data)
                    
                    # Debug: Show first few rows
                    if debug_mode:
                        st.write("**Load Data Sample:**")
                        render_dataframe(actual_df.head(), height=180)
                        st.write(f"Columns: {list(actual_df.columns)}")
                        if "fields" in actual_load_data:
                            st.write("**Fields metadata:**")
                            st.json(actual_load_data["fields"])
                    
                    # Parse timestamp: ERCOT uses operatingDay + hourEnding (hour 24 = midnight next day)
                    if 'operatingDay' in actual_df.columns and 'hourEnding' in actual_df.columns:
                        def parse_ercot_timestamp(row):
                            try:
                                date = pd.to_datetime(row['operatingDay'])
                                hour_str = str(row['hourEnding']).strip().replace('.0', '')
                                hour = int(hour_str) if hour_str.isdigit() else 0
                                # ERCOT defines 24 as midnight next day
                                return date + timedelta(days=1) if hour == 24 else date + timedelta(hours=hour)
                            except Exception:
                                return pd.NaT

                        actual_df['timestamp'] = actual_df.apply(parse_ercot_timestamp, axis=1)
                        actual_df = (
                            actual_df.dropna(subset=['timestamp'])
                                    .sort_values('timestamp')
                                    .drop_duplicates(subset=['timestamp'])
                                    .set_index('timestamp')
                        )

                        # Interpolate only numeric columns; string/bool columns are filled by nearest known values.
                        numeric_cols = actual_df.select_dtypes(include=['number']).columns
                        non_numeric_cols = actual_df.columns.difference(numeric_cols)

                        actual_df = actual_df.resample('h').asfreq()

                        if len(numeric_cols) > 0:
                            actual_df[numeric_cols] = actual_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
                            actual_df[numeric_cols] = actual_df[numeric_cols].interpolate(method='linear', limit_direction='both')

                        if len(non_numeric_cols) > 0:
                            actual_df[non_numeric_cols] = actual_df[non_numeric_cols].ffill().bfill()

                        actual_df = actual_df.reset_index()
                        actual_df['timestamp'] = pd.to_datetime(actual_df['timestamp'], errors='coerce')
                        actual_df = actual_df.dropna(subset=['timestamp'])
                                                                
                    # Display metrics - use total/system-wide load when available.
                    load_col = find_column(
                        actual_df,
                        exact=["total", "ercot", "system_wide", "systemwide", "systemTotal"],
                        contains=["total"],
                    )
                    
                    if load_col and not actual_df.empty:
                        load_series = coerce_numeric(actual_df[load_col])
                        latest_load = load_series.iloc[-1]
                        avg_load = load_series.mean()
                        max_load = load_series.max()
                        min_load = load_series.min()
                        load_factor = (avg_load / max_load * 100) if max_load else np.nan

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            render_metric_card("Current Load", format_mw(latest_load), latest_timestamp_label(actual_df), ERCOT_CYAN)
                        with col2:
                            render_metric_card("Average Load", format_mw(avg_load), f"{date_range}-day window", ERCOT_BLUE)
                        with col3:
                            render_metric_card("Peak Load", format_mw(max_load), "Observed in selected window", ERCOT_ORANGE)
                        with col4:
                            render_metric_card("Load Factor", format_percent(load_factor), f"Minimum {format_mw(min_load)}", ERCOT_GREEN)
                    
                    # Plot actual vs forecast
                    fig = make_subplots(specs=[[{"secondary_y": False}]])
                    
                    if load_col and not actual_df.empty:
                        x_axis = actual_df['timestamp'] if 'timestamp' in actual_df.columns else actual_df.index
                        fig.add_trace(
                            go.Scatter(
                                x=x_axis,
                                y=actual_df[load_col],
                                mode='lines',
                                name='Actual Load',
                                line=dict(color=ERCOT_BLUE, width=2.7),
                                fill="tozeroy",
                                fillcolor="rgba(11, 47, 79, 0.08)",
                            )
                        )
                    
                    if not forecast_df.empty:
                        # Parse timestamp for forecast data
                        forecast_time_cols = [col for col in forecast_df.columns if isinstance(col, str) and ('time' in col.lower() or 'date' in col.lower() or 'hour' in col.lower())]
                        if forecast_time_cols:
                            forecast_df['timestamp'] = pd.to_datetime(forecast_df[forecast_time_cols[0]], errors='coerce')
                            forecast_df = forecast_df.dropna(subset=['timestamp'])
                            forecast_df = forecast_df.sort_values('timestamp')
                        
                        x_axis_forecast = forecast_df['timestamp'] if 'timestamp' in forecast_df.columns else forecast_df.index
                        forecast_load_col = find_column(
                            forecast_df,
                            exact=["SystemTotal", "systemTotal", "total", "ercot"],
                            contains=["system"],
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=x_axis_forecast,
                                y=forecast_df[forecast_load_col] if forecast_load_col else forecast_df.iloc[:, -1],
                                mode='lines',
                                name='Forecast Load',
                                line=dict(color=ERCOT_ORANGE, width=2.4, dash='dash')
                            )
                        )
                    
                    fig = apply_professional_layout(
                        fig,
                        "ERCOT System Load: Actual vs Forecast",
                        "Load (MW)",
                        height=520,
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    # ML Forecast Section
                    render_section_header(
                        "Machine Learning Load Forecast",
                        "Next 24-hour forecast trained on the selected historical load window.",
                    )
                    
                    if load_col and not actual_df.empty:
                        # Prepare data for ML
                        if 'timestamp' in actual_df.columns:
                            ml_df = pd.DataFrame({
                                'load': actual_df[load_col].values
                            }, index=actual_df['timestamp'].values)
                        else:
                            ml_df = pd.DataFrame({
                                'load': actual_df[load_col].values
                            }, index=pd.date_range(end=pd.Timestamp.now(), periods=len(actual_df), freq='h'))
                        
                        model, scaler, training_df, split_data, preds, metrics = train_load_forecast_model(
                            ml_df,
                            tuning_mode=tuning_mode,
                            manual_params=manual_load_params
                        )
                        
                        if model is not None:
                            # --- ML Model Performance Metrics and Diagnostics ---
                            if split_data is not None and preds is not None and metrics is not None:
                                X_train, X_test, y_train, y_test = split_data
                                y_train_pred, y_test_pred = preds

                                score_col_1, score_col_2, score_col_3, score_col_4 = st.columns(4)
                                with score_col_1:
                                    render_metric_card("Validation MAE", format_mw(metrics["test_mae"]), "Lower is better", ERCOT_CYAN)
                                with score_col_2:
                                    render_metric_card("Validation RMSE", format_mw(metrics["test_rmse"]), "Error volatility", ERCOT_ORANGE)
                                with score_col_3:
                                    render_metric_card("Validation R2", f"{metrics['test_r2']:.3f}", "Explained variance", ERCOT_GREEN)
                                with score_col_4:
                                    render_metric_card("Training MAE", format_mw(metrics["train_mae"]), "In-sample fit", ERCOT_BLUE)

                                if metrics.get("best_params"):
                                    with st.expander("Selected model settings", expanded=False):
                                        st.write(f"Mode: {'Automatic' if metrics.get('tuning_mode') == 'auto' else 'Manual'}")
                                        if metrics.get("cv_mae") is not None:
                                            st.write(f"Cross-validated MAE: {format_mw(metrics['cv_mae'])}")
                                        st.json(metrics["best_params"])

                                fig_diag = go.Figure()
                                train_idx = y_train.index if hasattr(y_train, "index") else None
                                test_idx = y_test.index if hasattr(y_test, "index") else None
                                fig_diag.add_trace(go.Scatter(
                                    x=train_idx,
                                    y=y_train,
                                    mode='lines',
                                    name='Train Actual',
                                    line=dict(color=ERCOT_BLUE, width=2, dash='solid')
                                ))
                                fig_diag.add_trace(go.Scatter(
                                    x=train_idx,
                                    y=y_train_pred,
                                    mode='lines',
                                    name='Train Predicted',
                                    line=dict(color=ERCOT_CYAN, width=2, dash='dot')
                                ))
                                fig_diag.add_trace(go.Scatter(
                                    x=test_idx,
                                    y=y_test,
                                    mode='lines',
                                    name='Validation Actual',
                                    line=dict(color=ERCOT_RED, width=2, dash='solid')
                                ))
                                fig_diag.add_trace(go.Scatter(
                                    x=test_idx,
                                    y=y_test_pred,
                                    mode='lines',
                                    name='Validation Predicted',
                                    line=dict(color=ERCOT_ORANGE, width=2, dash='dot')
                                ))
                                fig_diag = apply_professional_layout(
                                    fig_diag,
                                    "Training Fit vs Validation",
                                    "Load (MW)",
                                    height=430,
                                )
                                st.plotly_chart(fig_diag, width='stretch')

                            # Generate forecast for next 24 hours
                            last_timestamp = training_df.index[-1] if hasattr(training_df.index[-1], 'hour') else pd.Timestamp.now()
                            future_hours = []
                            future_loads = []
                            
                            # Get last known values for lag features
                            last_load = training_df['load'].iloc[-1]
                            last_load_24h_ago = training_df['load'].iloc[-24] if len(training_df) >= 24 else last_load
                            
                            for h in range(24):
                                future_time = last_timestamp + timedelta(hours=h+1)
                                future_hours.append(future_time)
                                
                                # Create features
                                hour = future_time.hour
                                dow = future_time.dayofweek
                                is_weekend = 1 if dow >= 5 else 0
                                hour_sin = np.sin(2 * np.pi * hour / 24)
                                hour_cos = np.cos(2 * np.pi * hour / 24)
                                
                                # Use predicted values as lag features for future predictions
                                load_lag_1h = future_loads[-1] if future_loads else last_load
                                load_lag_24h = last_load_24h_ago
                                
                                # Rolling statistics from recent data
                                recent_3h = training_df['load'].iloc[-3:].tolist() + future_loads[-2:]
                                rolling_mean_3h = np.mean(recent_3h[-3:]) if len(recent_3h) >= 3 else training_df['rolling_mean_3h'].iloc[-1]
                                rolling_mean_24h = training_df['rolling_mean_24h'].iloc[-1]
                                rolling_std_24h = training_df['rolling_std_24h'].iloc[-1]
                                
                                features = [[hour, dow, is_weekend, hour_sin, hour_cos,
                                           load_lag_1h, load_lag_24h, rolling_mean_3h,
                                           rolling_mean_24h, rolling_std_24h]]
                                
                                # Predict
                                features_scaled = scaler.transform(features)
                                predicted_load = model.predict(features_scaled)[0]
                                future_loads.append(predicted_load)
                            
                            future_load = np.array(future_loads)
                            
                            fig_ml = go.Figure()
                            fig_ml.add_trace(
                                go.Scatter(
                                    x=future_hours,
                                    y=future_load,
                                    mode='lines+markers',
                                    name='ML Forecast',
                                    line=dict(color=ERCOT_GREEN, width=3.2),
                                    marker=dict(size=7, color=ERCOT_GREEN)
                                )
                            )
                            
                            model_name = "XGBoost" if HAS_XGBOOST else "Random Forest"
                            fig_ml = apply_professional_layout(
                                fig_ml,
                                f"{model_name} Load Forecast: Next 24 Hours",
                                "Predicted Load (MW)",
                                height=430,
                            )
                            st.plotly_chart(fig_ml, width='stretch')
                            
                            forecast_table = pd.DataFrame({
                                'Timestamp': [pd.Timestamp(h).strftime("%b %d %H:%M") for h in future_hours],
                                'Predicted Load': [format_mw(load) for load in future_load]
                            })
                            render_dataframe(forecast_table, height=250)
                        else:
                            st.warning("⚠️ Not enough historical data to train ML model. Need at least 48 hours.")
                    
                else:
                    st.warning("📭 No load data available for the selected period.")
                    
        except Exception as e:
            st.error(f"❌ Error fetching load data: {e}")
    
    # TAB 2: RENEWABLE GENERATION
    with tab2:
        render_section_header(
            "Renewable Generation",
            "Hourly wind and solar actuals versus available short-term forecast signals.",
        )
        
        col1, col2 = st.columns(2)
        
        # Wind Generation
        with col1:
            st.subheader("Wind Power")
            try:
                with st.spinner("Fetching wind data..."):
                    wind_params = {
                        "page": 1,
                        "size": 2000
                    }
                    import json
                    wind_data = fetch_ercot_data_cached(
                        "np4-732-cd/wpp_hrly_avrg_actl_fcast",
                        json.dumps(wind_params),
                        api.bearer_token,
                        api.subscription_key
                    )
                    
                    if "data" in wind_data and len(wind_data["data"]) > 0:
                        wind_df = dataframe_from_payload(wind_data)
                        
                        # Parse timestamp
                        wind_time_cols = [col for col in wind_df.columns if isinstance(col, str) and ('time' in col.lower() or 'date' in col.lower() or 'hour' in col.lower())]
                        if wind_time_cols:
                            wind_df['timestamp'] = pd.to_datetime(wind_df[wind_time_cols[0]], errors='coerce')
                            wind_df = wind_df.dropna(subset=['timestamp'])
                            wind_df = wind_df.sort_values('timestamp')
                        
                        if debug_mode:
                            st.write("**Wind Data Sample:**")
                            render_dataframe(wind_df.head(), height=180)
                            st.write(f"Columns: {list(wind_df.columns)}")
                        
                        fig_wind = go.Figure()
                        x_axis_wind = wind_df['timestamp'] if 'timestamp' in wind_df.columns else wind_df.index
                        
                        # Add actual and forecast traces - API uses 'genSystemWide' not 'ACTUAL_SYSTEM_WIDE'
                        # Find actual column
                        actual_col = find_column(wind_df, exact=["genSystemWide"], contains=["gen", "system"])
                        
                        if actual_col:
                            fig_wind.add_trace(
                                go.Scatter(
                                    x=x_axis_wind,
                                    y=wind_df[actual_col],
                                    mode='lines',
                                    name='Actual Wind',
                                    line=dict(color=ERCOT_CYAN, width=2.6),
                                    fill="tozeroy",
                                    fillcolor="rgba(0, 163, 199, 0.10)",
                                )
                            )
                        
                        # Find forecast column
                        forecast_col = find_column(wind_df, contains=["stwpf", "system"])
                        
                        if forecast_col:
                            fig_wind.add_trace(
                                go.Scatter(
                                    x=x_axis_wind,
                                    y=wind_df[forecast_col],
                                    mode='lines',
                                    name='Wind Forecast',
                                    line=dict(color=ERCOT_BLUE, width=2.3, dash='dash')
                                )
                            )

                        if actual_col:
                            latest_wind = coerce_numeric(wind_df[actual_col]).dropna()
                            render_metric_card(
                                "Latest Wind Output",
                                format_mw(latest_wind.iloc[-1] if not latest_wind.empty else np.nan),
                                latest_timestamp_label(wind_df),
                                ERCOT_CYAN,
                            )

                        fig_wind = apply_professional_layout(
                            fig_wind,
                            "Wind Generation: Actual vs Forecast",
                            "Generation (MW)",
                            height=430,
                        )
                        
                        if len(fig_wind.data) > 0:
                            st.plotly_chart(fig_wind, width='stretch')
                        else:
                            st.warning("⚠️ Wind data received but columns not found. Enable Debug Mode to see data structure.")
                    else:
                        st.info("📭 No wind data available.")
            except Exception as e:
                st.error(f"❌ Error fetching wind data: {e}")
        
        # Solar Generation
        with col2:
            st.subheader("Solar Power")
            try:
                with st.spinner("Fetching solar data..."):
                    solar_params = {
                        "page": 1,
                        "size": 2000
                    }
                    solar_data = fetch_ercot_data_cached(
                        "np4-737-cd/spp_hrly_avrg_actl_fcast",
                        json.dumps(solar_params),
                        api.bearer_token,
                        api.subscription_key
                    )
                    
                    if "data" in solar_data and len(solar_data["data"]) > 0:
                        solar_df = dataframe_from_payload(solar_data)
                        
                        # Parse timestamp
                        solar_time_cols = [col for col in solar_df.columns if isinstance(col, str) and ('time' in col.lower() or 'date' in col.lower() or 'hour' in col.lower())]
                        if solar_time_cols:
                            solar_df['timestamp'] = pd.to_datetime(solar_df[solar_time_cols[0]], errors='coerce')
                            solar_df = solar_df.dropna(subset=['timestamp'])
                            solar_df = solar_df.sort_values('timestamp')
                        
                        if debug_mode:
                            st.write("**Solar Data Sample:**")
                            render_dataframe(solar_df.head(), height=180)
                            st.write(f"Columns: {list(solar_df.columns)}")
                        
                        fig_solar = go.Figure()
                        x_axis_solar = solar_df['timestamp'] if 'timestamp' in solar_df.columns else solar_df.index
                        
                        # Find actual solar column - API uses 'genSystemWide'
                        actual_solar_col = find_column(solar_df, exact=["genSystemWide"], contains=["gen", "system"])
                        
                        if actual_solar_col:
                            fig_solar.add_trace(
                                go.Scatter(
                                    x=x_axis_solar,
                                    y=solar_df[actual_solar_col],
                                    mode='lines',
                                    name='Actual Solar',
                                    line=dict(color=ERCOT_ORANGE, width=2.6),
                                    fill="tozeroy",
                                    fillcolor="rgba(245, 158, 11, 0.12)",
                                )
                            )
                        
                        # Find solar forecast column
                        forecast_solar_col = find_column(solar_df, contains=["stppf", "system"])
                        
                        if forecast_solar_col:
                            fig_solar.add_trace(
                                go.Scatter(
                                    x=x_axis_solar,
                                    y=solar_df[forecast_solar_col],
                                    mode='lines',
                                    name='Solar Forecast',
                                    line=dict(color=ERCOT_RED, width=2.3, dash='dash')
                                )
                            )

                        if actual_solar_col:
                            latest_solar = coerce_numeric(solar_df[actual_solar_col]).dropna()
                            render_metric_card(
                                "Latest Solar Output",
                                format_mw(latest_solar.iloc[-1] if not latest_solar.empty else np.nan),
                                latest_timestamp_label(solar_df),
                                ERCOT_ORANGE,
                            )

                        fig_solar = apply_professional_layout(
                            fig_solar,
                            "Solar Generation: Actual vs Forecast",
                            "Generation (MW)",
                            height=430,
                        )
                        
                        if len(fig_solar.data) > 0:
                            st.plotly_chart(fig_solar, width='stretch')
                        else:
                            st.warning("⚠️ Solar data received but columns not found. Enable Debug Mode to see data structure.")
                    else:
                        st.info("📭 No solar data available.")
            except Exception as e:
                st.error(f"❌ Error fetching solar data: {e}")
    
    # TAB 3: REAL-TIME PRICING
    with tab3:
        render_section_header(
            "Real-Time Market Pricing",
            "Latest locational marginal prices for hubs, zones, and settlement points returned by ERCOT.",
        )
        
        try:
            with st.spinner("Fetching pricing data..."):
                lmp_params = {
                    "page": 1,
                    "size": 1000
                }
                lmp_data = fetch_ercot_data_cached(
                    "np6-788-cd/lmp_node_zone_hub",
                    json.dumps(lmp_params),
                    api.bearer_token,
                    api.subscription_key
                )
                
                if "data" in lmp_data and len(lmp_data["data"]) > 0:
                    lmp_df = dataframe_from_payload(lmp_data)
                    
                    # Filter for major hubs
                    point_col = find_column(lmp_df, exact=["SettlementPoint", "settlementPoint"])
                    type_col = find_column(lmp_df, exact=["SettlementPointType", "settlementPointType"])
                    price_col = find_column(lmp_df, exact=["SettlementPointPrice", "settlementPointPrice"], contains=["price"])
                    if point_col:
                        hubs = lmp_df[lmp_df[type_col] == 'HU'] if type_col else lmp_df
                        
                        if not hubs.empty and price_col:
                            hubs = hubs.copy()
                            hubs[price_col] = coerce_numeric(hubs[price_col])
                            hubs = hubs.dropna(subset=[price_col]).sort_values(price_col, ascending=False)
                            price_series = hubs[price_col]
                            price_col_1, price_col_2, price_col_3, price_col_4 = st.columns(4)
                            with price_col_1:
                                render_metric_card("Highest Hub Price", format_price(price_series.max()), "Visible hub set", ERCOT_RED)
                            with price_col_2:
                                render_metric_card("Average Hub Price", format_price(price_series.mean()), "Simple average", ERCOT_BLUE)
                            with price_col_3:
                                render_metric_card("Lowest Hub Price", format_price(price_series.min()), "Visible hub set", ERCOT_GREEN)
                            with price_col_4:
                                render_metric_card("Hub Count", f"{len(hubs):,}", "Returned by endpoint", ERCOT_CYAN)

                            fig_lmp = px.bar(
                                hubs.head(15),
                                x=point_col,
                                y=price_col,
                                labels={price_col: 'Price ($/MWh)', point_col: 'Settlement Point'},
                                color=price_col,
                                color_continuous_scale='RdYlGn_r'
                            )
                            
                            fig_lmp.update_layout(
                                title=dict(text="Latest LMP Prices at Major Hubs", x=0.01, xanchor="left", font=dict(size=20, color=ERCOT_BLUE)),
                                template=CHART_TEMPLATE,
                                height=520,
                                margin=dict(l=35, r=25, t=70, b=70),
                                coloraxis_showscale=False,
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="#ffffff",
                            )
                            fig_lmp.update_xaxes(title_text="Settlement Point", tickangle=-35, showgrid=False)
                            fig_lmp.update_yaxes(title_text="Price ($/MWh)", showgrid=True, gridcolor="#eef2f7", zerolinecolor="#cbd5e1")
                            st.plotly_chart(fig_lmp, width='stretch')
                            
                            # Show data table
                            table_cols = [point_col, price_col]
                            if type_col:
                                table_cols.insert(1, type_col)
                            render_dataframe(hubs[table_cols].head(30), height=360)
                        else:
                            st.info("Price data columns were not found. Enable Debug Mode to inspect the ERCOT payload.")
                    else:
                        render_dataframe(lmp_df.head(30), height=360)
                else:
                    st.warning("No pricing data available.")
        except Exception as e:
            st.error(f"Error fetching pricing data: {e}")
    
    # TAB 4: RESOURCE OUTAGES
    with tab4:
        render_section_header(
            "Resource Outages",
            "Hourly unavailable capacity by category, with stacked trend view and summary statistics.",
        )
        
        try:
            with st.spinner("Fetching outage data..."):
                outage_params = {
                    "page": 1,
                    "size": 2000
                }
                outage_data = fetch_ercot_data_cached(
                    "np3-233-cd/hourly_res_outage_cap",
                    json.dumps(outage_params),
                    api.bearer_token,
                    api.subscription_key
                )
                
                if "data" in outage_data and len(outage_data["data"]) > 0:
                    outage_df = dataframe_from_payload(outage_data)
                    
                    # Parse timestamp
                    outage_time_cols = [col for col in outage_df.columns if isinstance(col, str) and ('time' in col.lower() or 'date' in col.lower() or 'hour' in col.lower())]
                    if outage_time_cols:
                        outage_df['timestamp'] = pd.to_datetime(outage_df[outage_time_cols[0]], errors='coerce')
                        outage_df = outage_df.dropna(subset=['timestamp'])
                        outage_df = outage_df.sort_values('timestamp')
                    
                    if debug_mode:
                        st.write("**Outage Data Sample:**")
                        render_dataframe(outage_df.head(), height=180)
                        st.write(f"Columns: {list(outage_df.columns)}")
                    
                    candidate_numeric_cols = [
                        col for col in outage_df.columns
                        if col != "timestamp" and pd.to_numeric(outage_df[col], errors="coerce").notna().any()
                    ]
                    numeric_cols = []
                    for col in candidate_numeric_cols:
                        outage_df[col] = coerce_numeric(outage_df[col])
                        if outage_df[col].abs().sum(skipna=True) > 0:
                            numeric_cols.append(col)
                    
                    if len(numeric_cols) > 0:
                        latest_row = outage_df.sort_values("timestamp").iloc[-1] if "timestamp" in outage_df.columns else outage_df.iloc[-1]
                        latest_total = sum(float(latest_row[col]) for col in numeric_cols if pd.notna(latest_row[col]))
                        avg_total = outage_df[numeric_cols].sum(axis=1).mean()
                        peak_total = outage_df[numeric_cols].sum(axis=1).max()
                        top_latest = sorted(
                            [(col, float(latest_row[col])) for col in numeric_cols if pd.notna(latest_row[col])],
                            key=lambda item: item[1],
                            reverse=True,
                        )[:1]
                        top_label = top_latest[0][0] if top_latest else "N/A"

                        out_col_1, out_col_2, out_col_3, out_col_4 = st.columns(4)
                        with out_col_1:
                            render_metric_card("Latest Outage Capacity", format_mw(latest_total), latest_timestamp_label(outage_df), ERCOT_RED)
                        with out_col_2:
                            render_metric_card("Average Outage Capacity", format_mw(avg_total), "Selected data returned", ERCOT_BLUE)
                        with out_col_3:
                            render_metric_card("Peak Outage Capacity", format_mw(peak_total), "Selected data returned", ERCOT_ORANGE)
                        with out_col_4:
                            render_metric_card("Top Current Category", str(top_label), "Largest latest value", ERCOT_CYAN)

                        fig_outage = go.Figure()
                        x_axis_outage = outage_df['timestamp'] if 'timestamp' in outage_df.columns else outage_df.index
                        colors = [
                            "rgba(220, 38, 38, 0.62)",
                            "rgba(245, 158, 11, 0.62)",
                            "rgba(0, 163, 199, 0.62)",
                            "rgba(31, 157, 85, 0.62)",
                            "rgba(11, 47, 79, 0.62)",
                            "rgba(100, 116, 139, 0.62)",
                        ]
                        
                        for idx, col in enumerate(numeric_cols[:min(8, len(numeric_cols))]):
                            fig_outage.add_trace(
                                go.Scatter(
                                    x=x_axis_outage,
                                    y=outage_df[col],
                                    mode='lines',
                                    name=str(col),
                                    stackgroup='one',
                                    line=dict(width=1.5),
                                    fillcolor=colors[idx % len(colors)]
                                )
                            )
                        
                        fig_outage = apply_professional_layout(
                            fig_outage,
                            "Resource Outages by Category",
                            "Outage Capacity (MW)",
                            height=520,
                        )
                        st.plotly_chart(fig_outage, width='stretch')
                    else:
                        st.info("No numeric outage capacity columns were found in the returned payload.")
                    
                    st.subheader("Outage Summary Statistics")
                    render_dataframe(outage_df[numeric_cols].describe() if numeric_cols else outage_df.describe(), height=360)
                else:
                    st.warning("No outage data available.")
        except Exception as e:
            st.error(f"Error fetching outage data: {e}")


# Run the Streamlit dashboard
if __name__ == "__main__":
    main()
