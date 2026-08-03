import os
import requests
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import quote_plus
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import time
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import RandomForestRegressor
    HAS_XGBOOST = False
from sklearn.preprocessing import StandardScaler

try:
    from ERCOTAPI.latest_updates import load_latest_updates, revision_request_identity
    from ERCOTAPI.news_pipeline import (
        assess_news_brief,
        format_brief_age,
        news_item_sort_key,
    )
    from ERCOTAPI.grid_atlas import (
        POWER_PLANT_SOURCE_URL,
        SUBSTATION_SOURCE_URL,
        TRANSMISSION_SOURCE_URL,
        grid_atlas_change_summary,
        load_packaged_texas_grid,
        load_public_texas_grid,
    )
    from ERCOTAPI.grid_atlas_store import (
        GridAtlasStoreError,
        grid_atlas_region,
        grid_atlas_regions,
        load_grid_atlas_manifest,
        load_packaged_grid_region,
    )
except ImportError:
    from latest_updates import load_latest_updates, revision_request_identity
    from news_pipeline import (
        assess_news_brief,
        format_brief_age,
        news_item_sort_key,
    )
    from grid_atlas import (
        POWER_PLANT_SOURCE_URL,
        SUBSTATION_SOURCE_URL,
        TRANSMISSION_SOURCE_URL,
        grid_atlas_change_summary,
        load_packaged_texas_grid,
        load_public_texas_grid,
    )
    from grid_atlas_store import (
        GridAtlasStoreError,
        grid_atlas_region,
        grid_atlas_regions,
        load_grid_atlas_manifest,
        load_packaged_grid_region,
    )


LATEST_UPDATES_PATH = Path(__file__).with_name("latest_ercot_updates.json")
OPERATIONS_MESSAGES_URL = "https://www.ercot.com/services/comm/mkt_notices/opsmessages"
PUBLIC_NOTICES_URL = "https://www.ercot.com/services/comm/mkt_notices/notices"
MARKET_NOTICES_URL = "https://www.ercot.com/services/comm/mkt_notices/archives"
ERCOT_ASSISTANT_URL = (
    "https://amirexir-por-chatbot-ercot-all-in-oneercot-assistant-app-ahgre0."
    "streamlit.app/"
)
ERCOT_CONNECT_TIMEOUT_SECONDS = 5
ERCOT_READ_TIMEOUT_SECONDS = 30
ERCOT_REQUEST_TIMEOUT = (
    ERCOT_CONNECT_TIMEOUT_SECONDS,
    ERCOT_READ_TIMEOUT_SECONDS,
)


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
            response = requests.post(
                token_url,
                data=data,
                timeout=ERCOT_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            token_info = response.json()
            self.bearer_token = token_info.get("access_token") 
            if self.bearer_token:
                if verbose:
                    print(" Bearer token acquired.")
            else:
                raise ValueError("Failed to obtain bearer token.")
        except requests.exceptions.Timeout as e:
            raise ValueError(
                "Authentication timed out while contacting ERCOT. Please try again."
            ) from e
        except requests.exceptions.RequestException as e:
            # Try to extract error details from response
            error_detail = ""
            response = e.response
            if response is not None:
                try:
                    error_json = response.json()
                    error_detail = (
                        "\nError details: "
                        f"{error_json.get('error_description', error_json)}"
                    )
                except (ValueError, AttributeError):
                    error_detail = f"\nResponse: {response.text[:200]}"
            
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
                response = requests.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=ERCOT_REQUEST_TIMEOUT,
                )
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
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    if verbose:
                        print(
                            "ERCOT request timed out or lost its connection. "
                            f"Waiting {wait_time}s before retry "
                            f"{attempt + 1}/{max_retries}..."
                        )
                    time.sleep(wait_time)
                else:
                    raise

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


@st.cache_data(show_spinner=False)
def load_packaged_grid_atlas_cached() -> Dict[str, Any]:
    """Open the checked-in reference snapshot; never call ArcGIS or the RAG."""

    return load_packaged_texas_grid()


@st.cache_resource(show_spinner=False)
def load_grid_atlas_manifest_cached() -> Dict[str, Any]:
    """Load the tiny checked-in U.S.–Canada Atlas manifest once per process."""

    return load_grid_atlas_manifest()


@st.cache_resource(max_entries=2, show_spinner=False)
def load_packaged_grid_region_cached(region_id: str) -> Dict[str, Any]:
    """Open only the selected local Atlas shard; no source or AI request."""

    return load_packaged_grid_region(region_id)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_grid_atlas_cached() -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Fetch and validate public ArcGIS layers after an explicit user action."""

    packaged_payload = load_packaged_texas_grid()
    payload = load_public_texas_grid()
    comparison = grid_atlas_change_summary(packaged_payload, payload)
    return payload, comparison


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

    latest_name = max(candidates, key=news_item_sort_key)
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
            latest = max(
                matches,
                key=lambda item: news_item_sort_key(item.get("name", "")),
            )
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
            latest_path = max(tree_matches, key=news_item_sort_key)
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


APP_BUILD = "2026-07-23 us-canada-grid-atlas-v5"
ERCOT_API_MARKET_URL = "https://apimarket.ercot.com/"
LOVABLE_ERCOT_DASHBOARD_URL = "https://ercot-news-watch.lovable.app/"
ERCOT_TIMEZONE = ZoneInfo("America/Chicago")
ERCOT_BLUE = "#0b2f4f"
ERCOT_CYAN = "#00a3c7"
ERCOT_GREEN = "#1f9d55"
ERCOT_ORANGE = "#f59e0b"
ERCOT_RED = "#dc2626"
CHART_TEMPLATE = "plotly_white"
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def inject_dashboard_css() -> None:
    """Apply a polished Streamlit theme without changing global app behavior."""
    st.markdown(
        """
        <style>
        :root {
            --ercot-blue: #0b2f4f;
            --ercot-blue-hover: #123f67;
            --ercot-cyan: #00a3c7;
            --ercot-accent: #0b5e75;
            --ercot-accent-hover: #06485c;
            --ercot-soft-blue: #eaf4f8;
            --ercot-focus: #0891b2;
            --ercot-green: #1f9d55;
            --ercot-orange: #f59e0b;
            --ercot-red: #dc2626;
            --ercot-slate: #334155;
            --ercot-muted: #64748b;
            --ercot-panel: #ffffff;
            --ercot-border: #e2e8f0;
            --ercot-bg: #f8fafc;
            --ercot-alert-text: #172033;
            --ercot-alert-link: #075985;
        }
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > .main {
            background: var(--ercot-bg);
            color: #0f172a;
        }
        [data-testid="stHeader"] {
            background: rgba(248,250,252,0.92);
            border-bottom: 1px solid rgba(226,232,240,0.72);
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
        .stButton > button,
        .stLinkButton > a,
        a[data-testid^="stBaseLinkButton-"] {
            background: var(--ercot-blue) !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            border: 1px solid var(--ercot-blue) !important;
            border-radius: 10px !important;
            font-weight: 750 !important;
            min-height: 2.6rem;
            width: 100% !important;
            white-space: nowrap !important;
            overflow: visible !important;
            opacity: 1 !important;
            box-shadow: 0 5px 14px rgba(11, 47, 79, 0.14) !important;
            transition:
                background-color 150ms ease,
                border-color 150ms ease,
                box-shadow 150ms ease,
                transform 150ms ease !important;
        }
        .stButton > button *,
        .stLinkButton > a *,
        a[data-testid^="stBaseLinkButton-"] * {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            opacity: 1 !important;
        }
        .stButton > button:hover,
        .stLinkButton > a:hover,
        a[data-testid^="stBaseLinkButton-"]:hover {
            background: var(--ercot-blue-hover) !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            border-color: var(--ercot-focus) !important;
            box-shadow: 0 8px 20px rgba(11, 47, 79, 0.24) !important;
            transform: translateY(-1px);
        }
        .stLinkButton > a:visited,
        a[data-testid^="stBaseLinkButton-"]:visited {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }
        .stButton > button:focus-visible,
        .stLinkButton > a:focus-visible,
        a[data-testid^="stBaseLinkButton-"]:focus-visible {
            outline: 3px solid rgba(8, 145, 178, 0.34) !important;
            outline-offset: 2px !important;
            box-shadow: 0 0 0 2px #ffffff, 0 0 0 5px var(--ercot-focus) !important;
        }
        .stButton > button:active,
        .stLinkButton > a:active,
        a[data-testid^="stBaseLinkButton-"]:active {
            background: #082a46 !important;
            transform: translateY(0);
            box-shadow: 0 3px 8px rgba(11, 47, 79, 0.18) !important;
        }
        .stButton > button:disabled {
            background: #e2e8f0 !important;
            border-color: #cbd5e1 !important;
            color: #475569 !important;
            -webkit-text-fill-color: #475569 !important;
            box-shadow: none !important;
            cursor: not-allowed !important;
        }
        .stButton > button:disabled * {
            color: #475569 !important;
            -webkit-text-fill-color: #475569 !important;
        }
        button[data-testid="stBaseButton-primary"],
        .stButton > button[kind="primary"] {
            background: var(--ercot-accent) !important;
            border-color: var(--ercot-accent) !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }
        button[data-testid="stBaseButton-primary"]:hover,
        .stButton > button[kind="primary"]:hover {
            background: var(--ercot-accent-hover) !important;
            border-color: #22d3ee !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
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
        [data-testid="stAlert"],
        [data-testid="stAlert"] *,
        [data-testid="stAlertContainer"],
        [data-testid="stAlertContainer"] * {
            color: var(--ercot-alert-text) !important;
            -webkit-text-fill-color: var(--ercot-alert-text) !important;
            opacity: 1 !important;
        }
        [data-testid="stAlert"] a,
        [data-testid="stAlertContainer"] a {
            color: var(--ercot-alert-link) !important;
            -webkit-text-fill-color: var(--ercot-alert-link) !important;
            text-decoration-color: var(--ercot-alert-link) !important;
        }
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
        [data-testid="stTextInput"] [data-baseweb="input"],
        [data-testid="stTextInput"] [data-baseweb="base-input"],
        [data-testid="stTextInput"] input {
            background: #ffffff !important;
            color: #172033 !important;
            -webkit-text-fill-color: #172033 !important;
            border-color: #8aa5b8 !important;
            caret-color: #172033 !important;
        }
        [data-testid="stTextInput"] input::placeholder {
            color: #64748b !important;
            -webkit-text-fill-color: #64748b !important;
            opacity: 1 !important;
        }
        [data-testid="stTextInput"] [data-baseweb="input"]:focus-within {
            border-color: var(--ercot-focus) !important;
            box-shadow: 0 0 0 2px rgba(8, 145, 178, 0.18) !important;
        }
        [data-testid="stCheckbox"] label,
        [data-testid="stCheckbox"] label *,
        [data-testid="stCheckbox"] label p,
        [data-testid="stCheckbox"] label span {
            color: #172033 !important;
            -webkit-text-fill-color: #172033 !important;
            opacity: 1 !important;
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
        .stTabs [data-baseweb="tab-list"],
        .stTabs [role="tablist"],
        [data-testid="stTabs"] [role="tablist"] {
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 0.5rem !important;
            height: auto !important;
            overflow: visible !important;
            padding: 0.2rem 0 0.65rem !important;
            border-bottom: 1px solid var(--ercot-border) !important;
        }
        .stTabs [data-baseweb="tab"],
        .stTabs button[role="tab"],
        [data-testid="stTabs"] button[role="tab"] {
            flex: 0 0 auto !important;
            min-height: 2.45rem !important;
            border: 1px solid #8aa5b8 !important;
            border-radius: 999px !important;
            padding: 0.48rem 0.9rem !important;
            background: #eaf4f8 !important;
            color: var(--ercot-blue) !important;
            -webkit-text-fill-color: var(--ercot-blue) !important;
            opacity: 1 !important;
            box-shadow: 0 2px 7px rgba(15, 23, 42, 0.05) !important;
            transition:
                background-color 140ms ease,
                border-color 140ms ease,
                box-shadow 140ms ease !important;
        }
        .stTabs [data-baseweb="tab"] *,
        .stTabs button[role="tab"] *,
        [data-testid="stTabs"] button[role="tab"] * {
            color: inherit !important;
            -webkit-text-fill-color: inherit !important;
            opacity: 1 !important;
            font-weight: 700 !important;
        }
        .stTabs [data-baseweb="tab"]:hover,
        .stTabs button[role="tab"]:hover,
        [data-testid="stTabs"] button[role="tab"]:hover {
            background: #d7edf4 !important;
            border-color: var(--ercot-accent) !important;
            color: #073b55 !important;
            -webkit-text-fill-color: #073b55 !important;
            box-shadow: 0 5px 13px rgba(11, 47, 79, 0.10) !important;
        }
        .stTabs [data-baseweb="tab"]:focus-visible,
        .stTabs button[role="tab"]:focus-visible,
        [data-testid="stTabs"] button[role="tab"]:focus-visible {
            outline: 3px solid rgba(8, 145, 178, 0.30) !important;
            outline-offset: 2px !important;
        }
        .stTabs [aria-selected="true"],
        [data-testid="stTabs"] button[aria-selected="true"] {
            background: var(--ercot-blue) !important;
            border-color: var(--ercot-blue) !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            box-shadow: 0 6px 15px rgba(11, 47, 79, 0.18) !important;
        }
        .stTabs [aria-selected="true"] *,
        [data-testid="stTabs"] button[aria-selected="true"] * {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            font-weight: 800 !important;
        }
        .stTabs [data-baseweb="tab-highlight"],
        .stTabs [data-baseweb="tab-border"] {
            display: none !important;
        }
        .st-key-ercot_document_category_filter,
        [class*="st-key-ercot_document_category_filter"],
        [class*="st-key-ercot_revision_family_filter"],
        [class*="st-key-ercot_dashboard_view"],
        [class*="st-key-north_american_grid_atlas_region"] {
            margin: 0.25rem 0 0.9rem;
        }
        .st-key-ercot_document_category_filter [role="radiogroup"],
        .st-key-ercot_document_category_filter [data-baseweb="button-group"],
        [class*="st-key-ercot_document_category_filter"] [role="radiogroup"],
        [class*="st-key-ercot_revision_family_filter"] [role="radiogroup"],
        [class*="st-key-ercot_dashboard_view"] [role="radiogroup"],
        [class*="st-key-north_american_grid_atlas_region"] [role="radiogroup"],
        [class*="st-key-ercot_document_category_filter"] [data-baseweb="button-group"],
        [class*="st-key-ercot_revision_family_filter"] [data-baseweb="button-group"],
        [class*="st-key-ercot_dashboard_view"] [data-baseweb="button-group"],
        [class*="st-key-north_american_grid_atlas_region"] [data-baseweb="button-group"] {
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 0.5rem !important;
            overflow: visible !important;
        }
        .st-key-ercot_document_category_filter button,
        [class*="st-key-ercot_document_category_filter"] button,
        [class*="st-key-ercot_revision_family_filter"] button,
        [class*="st-key-ercot_dashboard_view"] button,
        [class*="st-key-north_american_grid_atlas_region"] button {
            flex: 0 0 auto !important;
            width: auto !important;
            min-height: 2.35rem !important;
            padding: 0.44rem 0.82rem !important;
            border: 1px solid #8aa5b8 !important;
            border-radius: 999px !important;
            background: #eaf4f8 !important;
            color: var(--ercot-blue) !important;
            -webkit-text-fill-color: var(--ercot-blue) !important;
            font-weight: 750 !important;
            opacity: 1 !important;
            box-shadow: 0 2px 7px rgba(15, 23, 42, 0.05) !important;
        }
        .st-key-ercot_document_category_filter button *,
        .st-key-ercot_document_category_filter button p,
        .st-key-ercot_document_category_filter button span,
        [class*="st-key-ercot_document_category_filter"] button *,
        [class*="st-key-ercot_revision_family_filter"] button *,
        [class*="st-key-ercot_dashboard_view"] button *,
        [class*="st-key-north_american_grid_atlas_region"] button * {
            color: inherit !important;
            -webkit-text-fill-color: inherit !important;
            opacity: 1 !important;
        }
        .st-key-ercot_document_category_filter button:hover,
        [class*="st-key-ercot_document_category_filter"] button:hover,
        [class*="st-key-ercot_revision_family_filter"] button:hover,
        [class*="st-key-ercot_dashboard_view"] button:hover,
        [class*="st-key-north_american_grid_atlas_region"] button:hover {
            background: #d7edf4 !important;
            border-color: var(--ercot-accent) !important;
            color: #073b55 !important;
            -webkit-text-fill-color: #073b55 !important;
        }
        .st-key-ercot_document_category_filter button[kind="pillsActive"],
        .st-key-ercot_document_category_filter button[data-testid="stBaseButton-pillsActive"],
        [class*="st-key-ercot_document_category_filter"] button[aria-pressed="true"],
        [class*="st-key-ercot_revision_family_filter"] button[aria-pressed="true"],
        [class*="st-key-ercot_dashboard_view"] button[aria-pressed="true"],
        [class*="st-key-north_american_grid_atlas_region"] button[aria-pressed="true"],
        [class*="st-key-ercot_document_category_filter"] button[aria-selected="true"],
        [class*="st-key-ercot_revision_family_filter"] button[aria-selected="true"],
        [class*="st-key-ercot_dashboard_view"] button[aria-selected="true"],
        [class*="st-key-north_american_grid_atlas_region"] button[aria-selected="true"],
        [class*="st-key-ercot_document_category_filter"] button[data-active="true"],
        [class*="st-key-ercot_revision_family_filter"] button[data-active="true"],
        [class*="st-key-ercot_dashboard_view"] button[data-active="true"],
        [class*="st-key-north_american_grid_atlas_region"] button[data-active="true"],
        [class*="st-key-ercot_document_category_filter"] button[data-testid$="-pillsActive"],
        [class*="st-key-ercot_revision_family_filter"] button[data-testid$="-pillsActive"],
        [class*="st-key-ercot_dashboard_view"] button[data-testid$="-pillsActive"],
        [class*="st-key-north_american_grid_atlas_region"] button[data-testid$="-pillsActive"],
        [class*="st-key-ercot_document_category_filter"] button[kind$="Active"],
        [class*="st-key-ercot_revision_family_filter"] button[kind$="Active"],
        [class*="st-key-ercot_dashboard_view"] button[kind$="Active"],
        [class*="st-key-north_american_grid_atlas_region"] button[kind$="Active"] {
            background: var(--ercot-blue) !important;
            border-color: var(--ercot-blue) !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            box-shadow: 0 6px 15px rgba(11, 47, 79, 0.18) !important;
        }
        .st-key-ercot_document_category_filter button[kind="pillsActive"] *,
        .st-key-ercot_document_category_filter button[data-testid="stBaseButton-pillsActive"] *,
        [class*="st-key-ercot_document_category_filter"] button[aria-pressed="true"] *,
        [class*="st-key-ercot_revision_family_filter"] button[aria-pressed="true"] *,
        [class*="st-key-ercot_dashboard_view"] button[aria-pressed="true"] *,
        [class*="st-key-north_american_grid_atlas_region"] button[aria-pressed="true"] *,
        [class*="st-key-ercot_document_category_filter"] button[aria-selected="true"] *,
        [class*="st-key-ercot_revision_family_filter"] button[aria-selected="true"] *,
        [class*="st-key-ercot_dashboard_view"] button[aria-selected="true"] *,
        [class*="st-key-north_american_grid_atlas_region"] button[aria-selected="true"] *,
        [class*="st-key-ercot_document_category_filter"] button[data-testid$="-pillsActive"] *,
        [class*="st-key-ercot_revision_family_filter"] button[data-testid$="-pillsActive"] *,
        [class*="st-key-ercot_dashboard_view"] button[data-testid$="-pillsActive"] *,
        [class*="st-key-north_american_grid_atlas_region"] button[data-testid$="-pillsActive"] * {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }
        .st-key-ercot_document_category_filter button:focus-visible,
        [class*="st-key-ercot_document_category_filter"] button:focus-visible,
        [class*="st-key-ercot_revision_family_filter"] button:focus-visible,
        [class*="st-key-ercot_dashboard_view"] button:focus-visible,
        [class*="st-key-north_american_grid_atlas_region"] button:focus-visible {
            outline: 3px solid rgba(8, 145, 178, 0.30) !important;
            outline-offset: 2px !important;
        }
        [data-testid="stExpander"] details {
            overflow: hidden;
            border: 1px solid #d7e1ea !important;
            border-radius: 12px !important;
            background: #ffffff !important;
            box-shadow: 0 3px 10px rgba(15, 23, 42, 0.045);
        }
        [data-testid="stExpander"] summary {
            min-height: 2.8rem;
            background: #ffffff !important;
            color: #172033 !important;
            -webkit-text-fill-color: #172033 !important;
            transition: background-color 140ms ease, color 140ms ease;
        }
        [data-testid="stExpander"] summary *,
        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] summary span {
            color: #172033 !important;
            -webkit-text-fill-color: #172033 !important;
            opacity: 1 !important;
            font-weight: 650 !important;
        }
        [data-testid="stExpander"] summary:hover {
            background: var(--ercot-soft-blue) !important;
            color: var(--ercot-blue) !important;
            -webkit-text-fill-color: var(--ercot-blue) !important;
        }
        [data-testid="stExpander"] summary:hover * {
            color: var(--ercot-blue) !important;
            -webkit-text-fill-color: var(--ercot-blue) !important;
        }
        [data-testid="stExpander"] summary:focus-visible {
            outline: 3px solid rgba(8, 145, 178, 0.30) !important;
            outline-offset: -3px !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] details {
            border-color: rgba(255,255,255,0.24) !important;
            background: #f8fafc !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary *,
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"],
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] * {
            color: #172033 !important;
            -webkit-text-fill-color: #172033 !important;
            opacity: 1 !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover * {
            background: #eaf4f8 !important;
            color: var(--ercot-blue) !important;
            -webkit-text-fill-color: var(--ercot-blue) !important;
        }
        [data-testid="stSidebar"] [data-baseweb="input"],
        [data-testid="stSidebar"] [data-baseweb="base-input"],
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] select {
            background: #ffffff !important;
            color: #172033 !important;
            -webkit-text-fill-color: #172033 !important;
            caret-color: #172033 !important;
        }
        [data-testid="stSidebar"] [data-baseweb="input"] *,
        [data-testid="stSidebar"] [data-baseweb="base-input"] * {
            color: #172033 !important;
            -webkit-text-fill-color: #172033 !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] a {
            color: #0369a1 !important;
            -webkit-text-fill-color: #0369a1 !important;
            text-decoration-color: #0369a1 !important;
        }
        h1, h2, h3 {
            color: #0f172a;
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
                <span class="hero-badge">Lovable version</span>
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


def categorize_ercot_update(item: Dict[str, Any]) -> str:
    """Group technical updates for scanning without changing RAG contents."""
    source = str(item.get("source") or "").upper()
    title = str(item.get("title") or "").upper()
    revision = revision_request_identity(
        str(item.get("document_number") or ""),
        str(item.get("title") or ""),
        str(item.get("url") or ""),
    )
    if revision:
        return "Revision Requests (xRRs)"
    combined = f"{source} {title}"
    if any(term in combined for term in ("PROTOCOL", "PLANNING GUIDE", "OPERATING GUIDE", "OTHER BINDING DOCUMENT")):
        return "Protocols, Guides & OBDs"
    if source in {"DWG", "SSWG", "RIWG", "LLWG", "RPG", "TAC", "BOARD OF DIRECTORS"}:
        return "Groups & Governance"
    return "Other Technical Documents"


XRR_FAMILY_ORDER = (
    "NPRR", "PGRR", "NOGRR", "OBDRR", "RRGRR", "VCMRR", "COPMGRR",
    "LPGRR", "RMGRR", "SMOGRR", "CMGRR", "SCR",
)


def ercot_assistant_question_url(question: str) -> str:
    """Create a no-auto-submit assistant link with a review question."""

    return f"{ERCOT_ASSISTANT_URL}?question={quote_plus(question)}"


def _ercot_update_date(item: Dict[str, Any]) -> datetime:
    for value in (item.get("published_date"), item.get("downloaded_at")):
        candidate = str(value or "").strip()
        if not candidate:
            continue
        for pattern in (
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d",
            "%B %d, %Y",
            "%b %d, %Y",
        ):
            try:
                return datetime.strptime(candidate, pattern).replace(tzinfo=None)
            except ValueError:
                continue
    return datetime.min


def _render_update_item(item: Dict[str, Any]) -> None:
    title = str(item.get("title") or item.get("document_number") or "ERCOT document")
    number = str(item.get("document_number") or "").strip()
    label = f"{number} — {title}" if number and number.casefold() not in title.casefold() else title
    with st.expander(label):
        st.write(item.get("explanation") or "New ERCOT technical material.")
        metadata = " · ".join(
            value for value in (
                str(item.get("source") or "").strip(),
                str(item.get("published_date") or "").strip(),
                str(item.get("status") or "").strip(),
            ) if value
        )
        if metadata:
            st.caption(metadata)
        if item.get("url"):
            st.link_button("Open ERCOT source", str(item["url"]))


def _render_update_collection(
    items: list[Dict[str, Any]],
    *,
    empty_message: str,
    key: str,
    initial_limit: int = 20,
) -> None:
    if not items:
        st.info(empty_message)
        return
    ordered = sorted(items, key=_ercot_update_date, reverse=True)
    show_all = len(ordered) <= initial_limit or st.toggle(
        f"Show all {len(ordered)} documents",
        value=False,
        key=f"{key}_show_all",
    )
    visible = ordered if show_all else ordered[:initial_limit]
    if not show_all:
        st.caption(f"Showing the {len(visible)} newest documents.")
    for item in visible:
        _render_update_item(item)


def _revision_groups(
    items: list[Dict[str, Any]],
) -> dict[str, dict[str, list[Dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[Dict[str, Any]]]] = {}
    for item in items:
        revision = revision_request_identity(
            str(item.get("document_number") or ""),
            str(item.get("title") or ""),
            str(item.get("url") or ""),
        )
        if not revision:
            continue
        revision_id, family = revision
        grouped.setdefault(family, {}).setdefault(revision_id, []).append(item)
    return grouped


def _revision_issue_number(revision_id: str) -> int:
    match = re.search(r"(\d+)$", revision_id)
    return int(match.group(1)) if match else -1


def _revision_issue_summary(revision_id: str, materials: list[Dict[str, Any]]) -> str:
    proposal = next(
        (
            item for item in materials
            if re.search(
                rf"\b\d+{re.escape(revision_id.rstrip('0123456789'))}-01\b",
                str(item.get("title") or ""),
                re.IGNORECASE,
            )
        ),
        None,
    )
    if proposal is None:
        return ""
    title = str(proposal.get("title") or "").strip()
    title = re.sub(
        r"^\s*\d+(?:NPRR|PGRR|NOGRR|OBDRR|RRGRR|VCMRR|COPMGRR|LPGRR|"
        r"RMGRR|SMOGRR|CMGRR|SCR)-\d+\s*",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(r"\s+\d{6}\s*$", "", title).strip(" —-")
    return title


def _revision_practical_impact(details: Dict[str, Any]) -> str:
    """Give a conservative engineering/market synthesis without an AI request."""

    combined = " ".join(
        str(details.get(field) or "")
        for field in ("issue_title", "official_description", "reason")
    ).casefold()
    if "load shed" in combined or "essential load" in combined:
        return (
            "This can affect how emergency curtailment is prepared and executed, including "
            "coordination intended to avoid concentrating interruptions on essential loads."
        )
    if any(term in combined for term in ("765-kv", "voltage limit", "reactive power")):
        return (
            "This affects the operating voltage envelope and the procedures used to keep "
            "high-voltage equipment within reliable limits."
        )
    if "stability" in combined:
        return (
            "This can change stability-study timing, inputs, or review milestones for planned "
            "generation and Large Loads, so project teams may need to adjust study schedules "
            "and model-readiness dates."
        )
    if "non-consequential load loss" in combined:
        return (
            "This can change which post-contingency load loss is acceptable in planning "
            "studies and therefore can influence whether a case passes criteria or requires "
            "a mitigation project."
        )
    if any(term in combined for term in ("large load", "interconnection", "generator")):
        return (
            "This can affect interconnection study assumptions, submitted models, project "
            "milestones, or how planned generation and Large Loads are represented in ERCOT "
            "planning cases."
        )
    if any(term in combined for term in ("reserve", "ancillary service", "drss")):
        return (
            "This can affect resource qualification, reserve deployment, operator actions, "
            "and how reliability services participate in real-time operations."
        )
    if any(term in combined for term in ("real-time co-optimization", "rtcb")):
        return (
            "This can affect co-optimized dispatch, Ancillary Service treatment, operating "
            "instructions, and the related settlement implementation."
        )
    if any(term in combined for term in ("ptp", "bid fee", "day-ahead market", "dam")):
        return (
            "This can change Day-Ahead Market bidding incentives, participant charges, or "
            "congestion-hedging behavior."
        )
    if any(term in combined for term in ("mitigated offer", "contract for capacity")):
        return (
            "This can change offer mitigation or pricing treatment for contracted capacity, "
            "with implications for market-power controls and participant offers."
        )
    if any(term in combined for term in ("model", "telemetry", "data requirement")):
        return (
            "This can affect the completeness and timing of data used for planning studies "
            "or real-time visibility, and therefore the quality of resulting engineering "
            "decisions."
        )
    family = str(details.get("revision_family") or "")
    family_impacts = {
        "PGRR": (
            "If incorporated, this would change an ERCOT planning process or criterion and "
            "may affect study scope, case assumptions, schedules, or project conclusions."
        ),
        "NOGRR": (
            "If incorporated, this would change an ERCOT operating-guide process and may "
            "affect operator actions, reliability coordination, or participant procedures."
        ),
        "NPRR": (
            "If incorporated, this would change a Nodal Protocol rule and may affect market "
            "participant obligations, dispatch, qualification, bidding, or settlement."
        ),
        "OBDRR": (
            "If incorporated into the controlling Other Binding Document, this would change "
            "the referenced implementation procedure or operational requirement."
        ),
    }
    return family_impacts.get(
        family,
        "This revision may change the process or requirement identified in the affected sections.",
    )


def _render_revision_requests(
    items: list[Dict[str, Any]],
    revision_issues: Dict[str, Dict[str, Any]] | None = None,
) -> None:
    revision_issues = revision_issues or {}
    grouped = _revision_groups(items)
    families = [family for family in XRR_FAMILY_ORDER if grouped.get(family)]
    issue_count = sum(len(grouped[family]) for family in families)
    material_count = sum(
        len(materials)
        for family in families
        for materials in grouped[family].values()
    )
    if not families:
        st.info("No new revision-request issues are available.")
        return

    st.caption(
        f"{issue_count} unique revision requests across {material_count} proposal, comment, "
        "impact-analysis, ballot, committee, and approval materials."
    )
    family_labels = {
        f"{family} ({len(grouped[family])})": family
        for family in families
    }
    selected_label = st.pills(
        "Revision-request family",
        list(family_labels),
        default=next(iter(family_labels)),
        selection_mode="single",
        label_visibility="collapsed",
        key="ercot_revision_family_filter",
    )
    selected_family = family_labels.get(selected_label or "", families[0])
    search_col, limit_col = st.columns([3, 1])
    with search_col:
        search = st.text_input(
            "Search this xRR family",
            placeholder=f"Search {selected_family} number or title",
            key=f"ercot_xrr_search_{selected_family}",
        ).strip().casefold()
    with limit_col:
        show_all = st.toggle(
            "Show all issues",
            value=False,
            key=f"ercot_xrr_show_all_{selected_family}",
        )

    issues = list(grouped[selected_family].items())
    issues.sort(
        key=lambda pair: (
            max((_ercot_update_date(item) for item in pair[1]), default=datetime.min),
            _revision_issue_number(pair[0]),
        ),
        reverse=True,
    )
    if search:
        issues = [
            pair for pair in issues
            if search in pair[0].casefold()
            or any(search in str(item.get("title") or "").casefold() for item in pair[1])
        ]
    visible = issues if show_all else issues[:12]
    if not visible:
        st.info(f"No {selected_family} issues match that search.")
        return
    if not show_all and len(issues) > len(visible):
        st.caption(f"Showing the 12 newest of {len(issues)} {selected_family} issues.")

    for revision_id, materials in visible:
        ordered_materials = sorted(materials, key=_ercot_update_date, reverse=True)
        details = dict(revision_issues.get(revision_id) or {})
        summary = str(details.get("issue_title") or "").strip()
        if not summary:
            summary = _revision_issue_summary(revision_id, materials)
        latest_date = str(ordered_materials[0].get("published_date") or "").strip()
        detail_parts = [f"{len(materials)} material{'s' if len(materials) != 1 else ''}"]
        if latest_date:
            detail_parts.append(f"latest {latest_date}")
        label = revision_id
        if summary:
            label += f" — {summary}"
        label += f" · {' · '.join(detail_parts)}"
        with st.expander(label):
            preferred = next(
                (item for item in materials if _revision_issue_summary(revision_id, [item])),
                ordered_materials[0],
            )
            official_description = str(
                details.get("official_description")
                or preferred.get("explanation")
                or "ERCOT has not supplied a description in the saved issue record."
            )
            st.markdown("**What ERCOT says this revision does**")
            st.write(official_description)

            effectiveness_note = str(details.get("effectiveness_note") or "").strip()
            if effectiveness_note:
                if details.get("effective_state") == "pending_proposal":
                    st.warning(effectiveness_note)
                else:
                    st.info(effectiveness_note)

            metadata_columns = st.columns(3)
            metadata_values = (
                ("Status", details.get("status") or preferred.get("status") or "Unknown"),
                ("Affected sections", details.get("affected_sections") or "Not listed"),
                ("Sponsor", details.get("sponsor") or "Not listed"),
            )
            for column, (metadata_label, metadata_value) in zip(
                metadata_columns,
                metadata_values,
            ):
                with column:
                    st.caption(metadata_label)
                    st.markdown(f"**{metadata_value}**")

            latest_action = details.get("latest_action")
            if isinstance(latest_action, dict) and latest_action:
                action_text = " · ".join(
                    value
                    for value in (
                        str(latest_action.get("date") or "").strip(),
                        str(latest_action.get("governing_body") or "").strip(),
                        str(latest_action.get("action") or "").strip(),
                        str(latest_action.get("next_step") or "").strip(),
                    )
                    if value
                )
                if action_text:
                    st.markdown("**Latest recorded action**")
                    st.write(action_text)

            st.markdown("**Practical impact (dashboard synthesis)**")
            st.write(_revision_practical_impact(details or {
                "revision_family": revision_id.rstrip("0123456789"),
                "official_description": official_description,
                "issue_title": summary,
            }))

            issue_url = str(details.get("issue_url") or "").strip()
            action_columns = st.columns(2)
            with action_columns[0]:
                st.link_button(
                    "Open official ERCOT issue",
                    issue_url or str(preferred.get("url") or ""),
                    use_container_width=True,
                )
            with action_columns[1]:
                st.link_button(
                    f"Ask assistant about {revision_id}",
                    ercot_assistant_question_url(
                        f"Explain {revision_id}, its current status, affected sections, "
                        "engineering or market impact, and whether it is governing yet."
                    ),
                    use_container_width=True,
                )

            st.markdown(f"**Source materials ({len(ordered_materials)})**")
            for artifact in ordered_materials:
                artifact_title = str(
                    artifact.get("title") or artifact.get("document_number") or revision_id
                )
                artifact_meta = " · ".join(
                    value for value in (
                        str(artifact.get("source") or "").strip(),
                        str(artifact.get("published_date") or "").strip(),
                        str(artifact.get("status") or "").strip(),
                    ) if value
                )
                artifact_text, artifact_action = st.columns([5, 1])
                with artifact_text:
                    st.markdown(f"**{artifact_title}**")
                    if artifact_meta:
                        st.caption(artifact_meta)
                with artifact_action:
                    if artifact.get("url"):
                        st.link_button(
                            "Open",
                            str(artifact["url"]),
                            use_container_width=True,
                        )


def render_latest_ercot_documents() -> None:
    """Show the saved technical-update feed and keep live notices outside the RAG."""
    payload = load_latest_updates(LATEST_UPDATES_PATH)
    items = payload.get("items", []) if isinstance(payload, dict) else []
    revision_issues = (
        {
            str(revision_id): dict(details)
            for revision_id, details in payload.get("revision_issues", {}).items()
            if isinstance(details, dict)
        }
        if isinstance(payload, dict)
        and isinstance(payload.get("revision_issues"), dict)
        else {}
    )
    categories = [
        "Revision Requests (xRRs)",
        "Protocols, Guides & OBDs",
        "Groups & Governance",
        "Other Technical Documents",
    ]
    grouped = {category: [] for category in categories}
    for item in items:
        grouped[categorize_ercot_update(item)].append(item)
    revisions = _revision_groups(grouped["Revision Requests (xRRs)"])
    unique_revision_count = sum(len(issues) for issues in revisions.values())
    render_section_header(
        "Latest ERCOT Documents and Explanations",
        "New 2026+ technical documents are organized by document family and revision-request issue. "
        "Operational and public notices remain separate from the engineering RAG.",
    )
    (
        revision_tab,
        governing_tab,
        groups_tab,
        other_tab,
        operations_tab,
        notices_tab,
    ) = st.tabs(
        [
            f"Revision Requests ({unique_revision_count})",
            f"Protocols, Guides & OBDs ({len(grouped['Protocols, Guides & OBDs'])})",
            f"Groups & Governance ({len(grouped['Groups & Governance'])})",
            f"Other Technical ({len(grouped['Other Technical Documents'])})",
            "Operational messages",
            "Public & market notices",
        ]
    )
    with revision_tab:
        _render_revision_requests(
            grouped["Revision Requests (xRRs)"],
            revision_issues,
        )
    with governing_tab:
        _render_update_collection(
            grouped["Protocols, Guides & OBDs"],
            empty_message="No new Protocol, Guide, or Other Binding Document updates.",
            key="ercot_governing_updates",
        )
    with groups_tab:
        _render_update_collection(
            grouped["Groups & Governance"],
            empty_message="No new working-group or governance documents.",
            key="ercot_group_updates",
        )
    with other_tab:
        _render_update_collection(
            grouped["Other Technical Documents"],
            empty_message="No other new technical documents.",
            key="ercot_other_updates",
        )
    with operations_tab:
        st.info(
            "Operational messages are time-sensitive grid communications. They are intentionally separate from the engineering RAG and are not embedded."
        )
        st.link_button("View live ERCOT operational messages", OPERATIONS_MESSAGES_URL)
    with notices_tab:
        st.info(
            "Public and market notices are displayed as live ERCOT sources, not downloaded into or embedded by the engineering RAG."
        )
        notice_col, market_col = st.columns(2)
        with notice_col:
            st.link_button("View public notices", PUBLIC_NOTICES_URL, use_container_width=True)
        with market_col:
            st.link_button("View market-notice archive", MARKET_NOTICES_URL, use_container_width=True)


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
    value = normalize_timestamps(df["timestamp"]).dropna()
    if value.empty:
        return "Latest available interval"
    return format_timestamp_label(value.max())


def ercot_now() -> pd.Timestamp:
    """Current wall-clock time in ERCOT/Central time, returned timezone-naive for report comparisons."""
    return pd.Timestamp.now(tz=ERCOT_TIMEZONE).tz_localize(None)


def current_interval_cutoff(hours_ahead: int = 1) -> pd.Timestamp:
    """Allow one hourly report interval ahead because ERCOT rows are often labeled by hour-ending."""
    return ercot_now() + pd.Timedelta(hours=hours_ahead)


def normalize_timestamps(values: Any) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if isinstance(parsed, pd.Timestamp):
        parsed = pd.Series([parsed])
    try:
        if parsed.dt.tz is not None:
            parsed = parsed.dt.tz_convert(ERCOT_TIMEZONE).dt.tz_localize(None)
    except AttributeError:
        pass
    return parsed


def format_timestamp_label(value: Any) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return "Latest available interval"
    try:
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert(ERCOT_TIMEZONE).tz_localize(None)
    except AttributeError:
        pass
    return timestamp.strftime("%b %d, %Y %H:%M CT")


def format_timedelta_short(delta: pd.Timedelta) -> str:
    total_minutes = max(0, int(delta.total_seconds() // 60))
    hours, minutes = divmod(total_minutes, 60)
    days, hours = divmod(hours, 24)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def interval_freshness_note(value: Any, prefix: str = "As of", stale_after_hours: float = 4.0) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return "Latest available interval"
    try:
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert(ERCOT_TIMEZONE).tz_localize(None)
    except AttributeError:
        pass
    age = ercot_now() - timestamp
    if age > pd.Timedelta(hours=stale_after_hours):
        return f"Posted through {format_timestamp_label(timestamp)} ({format_timedelta_short(age)} old)"
    return f"{prefix} {format_timestamp_label(timestamp)}"


def latest_non_null_reading(
    df: pd.DataFrame,
    value_col: Optional[str],
    as_of: Optional[pd.Timestamp] = None,
    note_prefix: str = "As of",
    stale_after_hours: Optional[float] = None,
) -> tuple[Optional[float], str]:
    """Return the latest non-empty numeric reading and its own timestamp label."""
    if df.empty or not value_col or value_col not in df.columns:
        return None, "No available interval"

    values = coerce_numeric(df[value_col])
    valid_df = df.loc[values.notna()].copy()
    if valid_df.empty:
        return None, "No available interval"

    valid_df[value_col] = coerce_numeric(valid_df[value_col])
    if "timestamp" in valid_df.columns:
        valid_df["timestamp"] = normalize_timestamps(valid_df["timestamp"])
        valid_df = valid_df.dropna(subset=["timestamp"]).sort_values("timestamp")
        if as_of is not None:
            valid_df = valid_df[valid_df["timestamp"] <= as_of]
        if valid_df.empty:
            return None, "No available interval"
        row = valid_df.iloc[-1]
        if stale_after_hours is None:
            note = f"{note_prefix} {format_timestamp_label(row['timestamp'])}"
        else:
            note = interval_freshness_note(row["timestamp"], prefix=note_prefix, stale_after_hours=stale_after_hours)
        return float(row[value_col]), note

    return float(valid_df[value_col].iloc[-1]), "Latest available interval"


def latest_row_at_or_before(
    df: pd.DataFrame,
    value_cols: list[str],
    as_of: Optional[pd.Timestamp] = None,
) -> Optional[pd.Series]:
    """Return the latest row with any requested value populated, optionally capped by timestamp."""
    value_cols = [col for col in value_cols if col and col in df.columns]
    if df.empty or not value_cols:
        return None

    result = df.copy()
    for col in value_cols:
        result[col] = coerce_numeric(result[col])
    result = result.dropna(subset=value_cols, how="all")

    if "timestamp" in result.columns:
        result["timestamp"] = normalize_timestamps(result["timestamp"])
        result = result.dropna(subset=["timestamp"]).sort_values("timestamp")
        if as_of is not None:
            result = result[result["timestamp"] <= as_of]

    if result.empty:
        return None
    return result.iloc[-1]


def filter_time_window(
    df: pd.DataFrame,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns:
        return df

    result = df.copy()
    result["timestamp"] = normalize_timestamps(result["timestamp"])
    result = result.dropna(subset=["timestamp"])
    if start is not None:
        result = result[result["timestamp"] >= start]
    if end is not None:
        result = result[result["timestamp"] <= end]
    return result.sort_values("timestamp")


def time_window_around_now(df: pd.DataFrame, hours_back: int = 48, hours_forward: int = 72) -> pd.DataFrame:
    now = ercot_now()
    return filter_time_window(
        df,
        start=now - pd.Timedelta(hours=hours_back),
        end=now + pd.Timedelta(hours=hours_forward),
    )


def rows_at_latest_timestamp(df: pd.DataFrame, as_of: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns:
        return df
    result = filter_time_window(df, end=as_of)
    if result.empty:
        return result
    latest_ts = result["timestamp"].max()
    return result[result["timestamp"] == latest_ts]


def has_numeric_values(df: pd.DataFrame, value_col: Optional[str]) -> bool:
    return bool(
        not df.empty
        and value_col
        and value_col in df.columns
        and coerce_numeric(df[value_col]).notna().any()
    )


def humanize_ercot_column(col: Any) -> str:
    raw = str(col)
    raw = re.sub(r"^total", "", raw, flags=re.IGNORECASE)
    raw = raw.replace("IRR", " IRR ")
    raw = raw.replace("MW", " MW ")
    label = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", raw).strip()
    label = label.replace("MW Zone", "MW -")
    label = label.replace("I R R", "IRR")
    return " ".join(label.split())


def is_capacity_column(col: Any) -> bool:
    col_lower = str(col).lower()
    if any(term in col_lower for term in ["timestamp", "date", "time", "hour", "flag", "interval", "posted"]):
        return False
    return "mw" in col_lower or "capacity" in col_lower or "cap" in col_lower


def get_outage_capacity_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if not is_capacity_column(col):
            continue
        numeric = coerce_numeric(df[col])
        if numeric.notna().any() and numeric.abs().sum(skipna=True) > 0:
            cols.append(col)
    return cols


def find_numeric_column_by_terms(df: pd.DataFrame, terms: list[str]) -> Optional[str]:
    for col in df.columns:
        col_lower = str(col).lower()
        if all(term.lower() in col_lower for term in terms) and coerce_numeric(df[col]).notna().any():
            return col
    return None


def find_price_column(df: pd.DataFrame) -> Optional[str]:
    price_col = find_column(
        df,
        exact=[
            "SettlementPointPrice",
            "settlementPointPrice",
            "LMP",
            "lmp",
            "Price",
            "price",
            "lmpPrice",
            "LMPPrice",
        ],
    )
    if price_col and coerce_numeric(df[price_col]).notna().any():
        return price_col

    for terms in (["settlement", "price"], ["lmp"], ["price"]):
        price_col = find_numeric_column_by_terms(df, terms)
        if price_col:
            return price_col

    numeric_candidates = []
    for col in df.columns:
        col_lower = str(col).lower()
        if any(term in col_lower for term in ["date", "time", "hour", "flag", "type", "id"]):
            continue
        numeric = coerce_numeric(df[col])
        if numeric.notna().any():
            numeric_candidates.append(col)
    return numeric_candidates[-1] if numeric_candidates else None


def find_settlement_point_column(df: pd.DataFrame) -> Optional[str]:
    point_col = find_column(
        df,
        exact=[
            "SettlementPoint",
            "settlementPoint",
            "SettlementPointName",
            "settlementPointName",
            "Settlement Point",
            "settlement_point",
        ],
    )
    if point_col:
        return point_col

    candidates = []
    for col in df.columns:
        col_lower = str(col).lower()
        if "settlement" in col_lower and "point" in col_lower and "type" not in col_lower and "price" not in col_lower:
            candidates.append(col)
    return candidates[0] if candidates else None


def find_settlement_type_column(df: pd.DataFrame) -> Optional[str]:
    type_col = find_column(
        df,
        exact=[
            "SettlementPointType",
            "settlementPointType",
            "SettlementPointTypeCode",
            "settlementPointTypeCode",
            "Settlement Point Type",
        ],
    )
    if type_col:
        return type_col

    for col in df.columns:
        col_lower = str(col).lower()
        if "settlement" in col_lower and "type" in col_lower:
            return col
    return None


def add_ercot_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Create a reliable timestamp from ERCOT date/hour fields when available."""
    if df.empty:
        return df

    result = df.copy()
    date_col = find_column(
        result,
        exact=["deliveryDate", "DeliveryDate", "operatingDay", "OperatingDay", "Delivery Date", "Operating Day"],
    )
    if date_col is None:
        for col in result.columns:
            col_lower = str(col).lower()
            if ("delivery" in col_lower and "date" in col_lower) or ("operating" in col_lower and "day" in col_lower):
                date_col = col
                break
    if date_col is None:
        for col in result.columns:
            col_lower = str(col).lower()
            if "date" in col_lower and "post" not in col_lower and "create" not in col_lower:
                date_col = col
                break

    hour_col = find_column(result, exact=["hourEnding", "HourEnding", "hour_ending", "Hour Ending"])
    if hour_col is None:
        for col in result.columns:
            col_lower = str(col).lower()
            if "hour" in col_lower and "end" in col_lower:
                hour_col = col
                break

    if date_col is not None and hour_col is not None:
        def parse_hour_ending(row: pd.Series) -> pd.Timestamp:
            date_value = pd.to_datetime(row[date_col], errors="coerce")
            if pd.isna(date_value):
                return pd.NaT
            match = re.search(r"\d{1,2}", str(row[hour_col]))
            if not match:
                return pd.NaT
            hour = int(match.group(0))
            base = date_value.normalize()
            return base + timedelta(days=1) if hour >= 24 else base + timedelta(hours=hour)

        result["timestamp"] = result.apply(parse_hour_ending, axis=1)
    else:
        candidate_cols = []
        for col in result.columns:
            col_lower = str(col).lower()
            if "hour" in col_lower and "date" not in col_lower and "time" not in col_lower:
                continue
            if any(term in col_lower for term in ["timestamp", "datetime", "interval", "time", "date"]):
                candidate_cols.append(col)
        for col in candidate_cols:
            parsed = pd.to_datetime(result[col], errors="coerce")
            if parsed.notna().any():
                result["timestamp"] = parsed
                break

    if "timestamp" in result.columns:
        result = result.dropna(subset=["timestamp"]).sort_values("timestamp")
    return result


def compact_time_series(df: pd.DataFrame, value_cols: list[str], max_days: Optional[int] = None) -> pd.DataFrame:
    """Deduplicate ERCOT interval rows into a readable one-row-per-timestamp series."""
    value_cols = [col for col in value_cols if col and col in df.columns]
    if df.empty or "timestamp" not in df.columns or not value_cols:
        return df

    result = df[["timestamp"] + value_cols].copy()
    for col in value_cols:
        result[col] = coerce_numeric(result[col])
    result = result.dropna(subset=["timestamp"]).dropna(subset=value_cols, how="all")
    result = result.groupby("timestamp", as_index=False)[value_cols].mean(numeric_only=True)

    if max_days and not result.empty:
        cutoff = result["timestamp"].max() - pd.Timedelta(days=max_days)
        result = result[result["timestamp"] >= cutoff]
    return result.sort_values("timestamp")


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
        margin=dict(l=72, r=28, t=82, b=72),
        legend=dict(orientation="h", yanchor="bottom", y=legend_y, xanchor="right", x=1),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a"),
    )
    fig.update_xaxes(
        title_text="Time",
        showgrid=True,
        gridcolor="#eef2f7",
        linecolor="#cbd5e1",
        tickfont=dict(color="#0f172a", size=12),
        title_font=dict(color="#0f172a", size=14),
        automargin=True,
    )
    fig.update_yaxes(
        title_text=yaxis_title,
        showgrid=True,
        gridcolor="#eef2f7",
        zerolinecolor="#cbd5e1",
        linecolor="#cbd5e1",
        tickfont=dict(color="#0f172a", size=12),
        title_font=dict(color="#0f172a", size=14),
        automargin=True,
    )
    return fig


def render_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)


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


def ercot_atlas_assets() -> pd.DataFrame:
    """Small contextual overlays kept separate from public infrastructure data."""

    records = [
        (
            "DFW data-center cluster",
            "Data-center context",
            32.940,
            -96.820,
            "Large-load cluster",
            "Approximate regional context",
            "Public company reports",
        ),
        (
            "Austin data-center cluster",
            "Data-center context",
            30.400,
            -97.710,
            "Large-load cluster",
            "Approximate regional context",
            "Public company reports",
        ),
        (
            "San Antonio data-center cluster",
            "Data-center context",
            29.530,
            -98.480,
            "Large-load cluster",
            "Approximate regional context",
            "Public company reports",
        ),
        (
            "Houston Hub",
            "Price-hub context",
            29.720,
            -95.500,
            "Settlement hub",
            "Location only; no live price",
            "Illustrative location",
        ),
        (
            "North Hub",
            "Price-hub context",
            32.550,
            -97.050,
            "Settlement hub",
            "Location only; no live price",
            "Illustrative location",
        ),
        (
            "South Hub",
            "Price-hub context",
            28.950,
            -98.250,
            "Settlement hub",
            "Location only; no live price",
            "Illustrative location",
        ),
        (
            "West Hub",
            "Price-hub context",
            31.650,
            -102.050,
            "Settlement hub",
            "Location only; no live price",
            "Illustrative location",
        ),
    ]
    frame = pd.DataFrame(
        records,
        columns=["name", "layer", "lat", "lon", "type", "detail", "source"],
    )
    frame["city"] = ""
    frame["status"] = ""
    frame["voltage"] = np.nan
    frame["capacity_mw"] = np.nan
    frame["period"] = ""
    frame["source_url"] = ""
    return frame


def _atlas_voltage_band(value: Any) -> str:
    try:
        voltage = float(value)
    except (TypeError, ValueError):
        return "Unknown voltage"
    if not np.isfinite(voltage) or voltage <= 0:
        return "Unknown voltage"
    if voltage >= 500:
        return "500–765 kV"
    if voltage >= 345:
        return "345–499 kV"
    if voltage >= 230:
        return "230–344 kV"
    if voltage >= 138:
        return "138–229 kV"
    if voltage >= 69:
        return "69–137 kV"
    return "Below 69 kV"


def _atlas_fuel_group(value: Any) -> str:
    fuel = str(value or "").strip().casefold()
    for term, label in (
        ("solar", "Solar"),
        ("wind", "Wind"),
        ("natural gas", "Natural gas"),
        ("gas", "Natural gas"),
        ("coal", "Coal"),
        ("nuclear", "Nuclear"),
        ("batter", "Battery"),
        ("hydro", "Hydro"),
        ("biomass", "Biomass"),
        ("petroleum", "Petroleum"),
    ):
        if term in fuel:
            return label
    return "Other / unknown"


def _atlas_public_assets(payload: Dict[str, Any]) -> pd.DataFrame:
    records: list[Dict[str, Any]] = []
    for plant in payload.get("power_plants", []):
        capacity = plant.get("capacity_mw")
        capacity_text = (
            f"{float(capacity):,.1f} MW installed"
            if capacity is not None
            else "Capacity unavailable"
        )
        location = ", ".join(
            value
            for value in (
                plant.get("city"),
                plant.get("county"),
                plant.get("province"),
                plant.get("state"),
            )
            if value
        )
        records.append(
            {
                "name": plant.get("name") or "Unnamed power plant",
                "layer": "Power plant",
                "lat": plant.get("lat"),
                "lon": plant.get("lon"),
                "type": _atlas_fuel_group(plant.get("fuel")),
                "detail": " · ".join(
                    value
                    for value in (capacity_text, plant.get("technology"), location)
                    if value
                ),
                "source": plant.get("source") or "U.S. EIA public power-plant layer",
                "city": plant.get("city") or "",
                "status": "",
                "voltage": np.nan,
                "capacity_mw": capacity,
                "period": plant.get("period") or "",
                "source_url": plant.get("source_url") or POWER_PLANT_SOURCE_URL,
                "country": plant.get("country") or "US",
            }
        )
    for substation in payload.get("substations", []):
        voltage = substation.get("max_voltage")
        voltage_text = (
            f"{float(voltage):,.0f} kV maximum"
            if voltage is not None
            else "Voltage unavailable"
        )
        voltage_provenance = (
            " · ".join(
                value
                for value in (
                    str(substation.get("voltage_match_status") or ""),
                    (
                        f"{float(substation['voltage_match_confidence']):.0%} confidence"
                        if substation.get("voltage_match_confidence") is not None
                        else ""
                    ),
                    (
                        f"retrieved {substation['voltage_retrieved_at']}"
                        if substation.get("voltage_retrieved_at")
                        else ""
                    ),
                )
                if value
            )
            if substation.get("voltage_source")
            else ""
        )
        location = ", ".join(
            value
            for value in (
                substation.get("city"),
                substation.get("county"),
                substation.get("province"),
                substation.get("state"),
            )
            if value
        )
        records.append(
            {
                "name": substation.get("name") or "Unnamed substation",
                "layer": (
                    "Autotransformer"
                    if substation.get("autotransformer")
                    else "Substation"
                ),
                "lat": substation.get("lat"),
                "lon": substation.get("lon"),
                "type": _atlas_voltage_band(voltage),
                "detail": " · ".join(
                    value
                    for value in (
                        voltage_text,
                        substation.get("owner"),
                        substation.get("operator"),
                        voltage_provenance,
                        location,
                    )
                    if value
                ),
                "source": substation.get("source") or "Public ArcGIS substation layer",
                "city": substation.get("city") or "",
                "status": substation.get("status") or "",
                "voltage": voltage,
                "capacity_mw": np.nan,
                "period": substation.get("source_date") or "",
                "source_url": substation.get("source_url") or SUBSTATION_SOURCE_URL,
                "country": substation.get("country") or "US",
            }
        )
    if not records:
        return pd.DataFrame(columns=ercot_atlas_assets().columns)
    return pd.DataFrame.from_records(records)


def render_grid_atlas() -> None:
    """Render the packaged Texas grid, with an optional explicit source check."""

    render_section_header(
        "ERCOT Grid Atlas",
        "Explore public Texas generation, transmission, substations, approximate large-load "
        "context, and price-hub locations in one spatial view.",
    )

    try:
        with st.spinner("Opening the packaged Texas infrastructure snapshot…"):
            packaged_payload = load_packaged_grid_atlas_cached()
    except RuntimeError as exc:
        st.error(str(exc))
        packaged_payload = {
            "transmission_lines": [],
            "substations": [],
            "power_plants": [],
            "errors": {"packaged_snapshot": str(exc)},
        }

    check_col, restore_col, explanation_col = st.columns([1.45, 1.25, 3.3])
    with check_col:
        check_for_updates = st.button(
            "Check source for updates",
            key="ercot_atlas_check_sources",
            type="primary",
            use_container_width=True,
            help=(
                "On click, compare the bundled snapshot with the three free public ArcGIS "
                "layers. This does not call OpenAI, create embeddings, or run automatically."
            ),
        )
    live_override = st.session_state.get("ercot_atlas_live_override")
    with restore_col:
        use_packaged_snapshot = False
        if isinstance(live_override, dict):
            use_packaged_snapshot = st.button(
                "Use packaged snapshot",
                key="ercot_atlas_restore_packaged",
                use_container_width=True,
                help="Discard the session-only source check and return to the uploaded snapshot.",
            )
    with explanation_col:
        st.caption(
            "The uploaded snapshot opens immediately. A source check is optional, cached for "
            "one hour, and changes only this browser session."
        )

    if use_packaged_snapshot:
        st.session_state.pop("ercot_atlas_live_override", None)
        st.session_state["ercot_atlas_source_status"] = {
            "level": "info",
            "message": "Using the packaged Texas grid snapshot again.",
        }
        live_override = None

    if check_for_updates:
        try:
            with st.spinner("Checking the three public ArcGIS layers for changes…"):
                candidate_payload, comparison = fetch_live_grid_atlas_cached()
            if comparison["changed"]:
                layer_names = {
                    "transmission_lines": "transmission lines",
                    "substations": "substations",
                    "power_plants": "power plants",
                }
                details = []
                for collection in comparison["changed_collections"]:
                    counts = comparison["counts"][collection]
                    delta = counts["delta"]
                    if delta:
                        details.append(
                            f"{layer_names[collection]} {delta:+,} records "
                            f"({counts['source']:,} at source)"
                        )
                    else:
                        details.append(
                            f"{layer_names[collection]} content or geometry changed "
                            f"({counts['source']:,} records)"
                        )
                st.session_state["ercot_atlas_live_override"] = candidate_payload
                st.session_state["ercot_atlas_source_status"] = {
                    "level": "success",
                    "message": (
                        "New public-source content detected: "
                        + "; ".join(details)
                        + ". Showing the refreshed records for this browser session. "
                        "The uploaded snapshot remains unchanged until the next deployment."
                    ),
                }
                live_override = candidate_payload
            else:
                st.session_state.pop("ercot_atlas_live_override", None)
                st.session_state["ercot_atlas_source_status"] = {
                    "level": "success",
                    "message": (
                        "No source changes found. The packaged snapshot already matches all "
                        "three public ArcGIS layers."
                    ),
                }
                live_override = None
        except (RuntimeError, requests.RequestException) as exc:
            st.session_state.pop("ercot_atlas_live_override", None)
            live_override = None
            st.session_state["ercot_atlas_source_status"] = {
                "level": "warning",
                "message": (
                    "The source check was incomplete, so the map returned to the packaged "
                    "snapshot: "
                    f"{exc}"
                ),
            }

    source_status = st.session_state.get("ercot_atlas_source_status")
    if isinstance(source_status, dict):
        status_message = str(source_status.get("message") or "")
        if source_status.get("level") == "success":
            st.success(status_message)
        elif source_status.get("level") == "warning":
            st.warning(status_message)
        else:
            st.info(status_message)

    payload = live_override if isinstance(live_override, dict) else packaged_payload
    using_live_source_check = isinstance(live_override, dict)

    if not payload.get("errors"):
        latest_line_source = max(
            (
                str(record.get("source_date") or "")
                for record in payload.get("transmission_lines", [])
                if record.get("source_date")
            ),
            default="not supplied",
        )
        latest_substation_source = max(
            (
                str(record.get("source_date") or "")
                for record in payload.get("substations", [])
                if record.get("source_date")
            ),
            default="not supplied",
        )
        latest_plant_period = max(
            (
                str(record.get("period") or "")
                for record in payload.get("power_plants", [])
                if record.get("period")
            ),
            default="not supplied",
        )
        snapshot_time = str(
            payload.get("fetched_at" if using_live_source_check else "generated_at")
            or "not supplied"
        )
        try:
            snapshot_time = (
                datetime.fromisoformat(snapshot_time.replace("Z", "+00:00"))
                .astimezone(ERCOT_TIMEZONE)
                .strftime("%b %d, %Y %H:%M %Z")
            )
        except ValueError:
            pass
        source_label = (
            "Session source check" if using_live_source_check else "Packaged snapshot"
        )
        st.caption(
            f"{source_label}: {snapshot_time}. "
            "Source/reporting dates — transmission: "
            f"{latest_line_source} · substations: {latest_substation_source} · "
            f"plants: {latest_plant_period}. Dashboard retrieval time is not the same as "
            "the source data date."
        )
    else:
        st.warning(
            "The packaged Texas snapshot is unavailable. Source links remain below, "
            "but the dashboard will not make an automatic live ArcGIS request."
        )

    public_assets = _atlas_public_assets(payload)
    context_assets = ercot_atlas_assets()
    assets = pd.DataFrame.from_records(
        [
            *public_assets.to_dict(orient="records"),
            *context_assets.to_dict(orient="records"),
        ],
        columns=context_assets.columns,
    )
    transmission_lines = list(payload.get("transmission_lines") or [])
    colors = {
        "Power plant": "#fde047",
        "Substation": "#f8fafc",
        "Autotransformer": "#f472b6",
        "Data-center context": "#c084fc",
        "Price-hub context": "#fbbf24",
    }
    symbols = {
        "Power plant": "circle",
        "Substation": "square",
        "Autotransformer": "diamond",
        "Data-center context": "diamond",
        "Price-hub context": "circle",
    }

    filter_col, search_col, display_col = st.columns([2.2, 1.6, 1])
    with filter_col:
        selected_layers = st.multiselect(
            "Infrastructure layers",
            options=list(colors),
            default=list(colors),
            help="Public facilities and contextual overlays remain visibly distinct.",
        )
    with search_col:
        asset_query = st.text_input(
            "Find an asset",
            placeholder="Houston, wind, Oncor, 345 kV…",
            key="ercot_atlas_asset_search",
        )
    with display_col:
        show_transmission = st.checkbox(
            "Transmission lines",
            value=True,
            disabled=not bool(transmission_lines),
        )

    voltage_col, unknown_col, capacity_col, fuel_col = st.columns([1.2, 1.2, 1.2, 1.8])
    with voltage_col:
        minimum_voltage = st.selectbox(
            "Minimum voltage",
            [0, 69, 100, 138, 230, 345, 500, 765],
            index=0,
            format_func=lambda value: "No minimum" if value == 0 else f"{value} kV",
            key="ercot_atlas_minimum_voltage_v2",
        )
    with unknown_col:
        include_unknown_voltage = st.checkbox(
            "Include unknown voltage",
            value=True,
            key="ercot_atlas_unknown_voltage_v2",
            help="Missing HIFLD values such as -999999 are treated as unknown, not as kV.",
        )
    with capacity_col:
        minimum_capacity = st.number_input(
            "Minimum plant MW",
            min_value=0,
            max_value=5_000,
            value=0,
            step=25,
        )
    available_fuels = sorted(
        {
            str(value)
            for value in public_assets.loc[
                public_assets.get("layer", pd.Series(dtype=str)) == "Power plant",
                "type",
            ].dropna()
        }
    )
    with fuel_col:
        selected_fuel = st.selectbox(
            "Plant fuel",
            ["All fuels", *available_fuels],
            key="ercot_atlas_fuel",
        )

    filtered = assets[assets["layer"].isin(selected_layers)].copy()
    if {"Substation", "Autotransformer"} & set(selected_layers):
        is_substation = filtered["layer"].isin(["Substation", "Autotransformer"])
        known_voltage = pd.to_numeric(filtered["voltage"], errors="coerce")
        substation_allowed = known_voltage.ge(float(minimum_voltage))
        if include_unknown_voltage:
            substation_allowed = substation_allowed | known_voltage.isna()
        filtered = filtered[~is_substation | substation_allowed]
    is_plant = filtered["layer"] == "Power plant"
    plant_capacity = pd.to_numeric(filtered["capacity_mw"], errors="coerce")
    filtered = filtered[
        ~is_plant
        | plant_capacity.fillna(0).ge(float(minimum_capacity))
    ]
    if selected_fuel != "All fuels":
        filtered = filtered[
            (filtered["layer"] != "Power plant") | (filtered["type"] == selected_fuel)
        ]

    visible_lines = []
    for line in transmission_lines:
        voltage = line.get("voltage")
        if voltage is None:
            if not include_unknown_voltage:
                continue
        elif float(voltage) < float(minimum_voltage):
            continue
        visible_lines.append(line)

    if asset_query.strip():
        query = asset_query.strip().lower()
        searchable = (
            filtered[["name", "type", "detail", "source", "city", "status"]]
            .astype(str)
            .agg(" ".join, axis=1)
            .str.lower()
        )
        filtered = filtered[searchable.str.contains(query, regex=False)]
        visible_lines = [
            line
            for line in visible_lines
            if query
            in " ".join(
                str(line.get(field) or "")
                for field in (
                    "name",
                    "id",
                    "owner",
                    "status",
                    "voltage",
                    "voltage_class",
                    "substation_1",
                    "substation_2",
                )
            ).lower()
        ]

    metric_columns = st.columns(5)
    metric_values = (
        ("Transmission", len(visible_lines), len(transmission_lines)),
        (
            "Power plants",
            int((filtered["layer"] == "Power plant").sum()),
            len(payload.get("power_plants") or []),
        ),
        (
            "Substations",
            int(filtered["layer"].isin(["Substation", "Autotransformer"]).sum()),
            len(payload.get("substations") or []),
        ),
        (
            "Data centers",
            int((filtered["layer"] == "Data-center context").sum()),
            3,
        ),
        (
            "Price hubs",
            int((filtered["layer"] == "Price-hub context").sum()),
            4,
        ),
    )
    for column, (label, visible_count, total_count) in zip(metric_columns, metric_values):
        with column:
            st.metric(label, f"{visible_count:,}", help=f"{total_count:,} loaded before filters")

    map_col, insight_col = st.columns([3.3, 1.15])
    with map_col:
        fig = go.Figure()
        line_colors = {
            "500–765 kV": "#f97316",
            "345–499 kV": "#22c55e",
            "230–344 kV": "#a855f7",
            "138–229 kV": "#0ea5e9",
            "69–137 kV": "#ef4444",
            "Below 69 kV": "#64748b",
            "Unknown voltage": "#94a3b8",
        }
        if show_transmission:
            for voltage_band, line_color in line_colors.items():
                band_lines = [
                    line
                    for line in visible_lines
                    if _atlas_voltage_band(line.get("voltage")) == voltage_band
                ]
                if not band_lines:
                    continue
                latitudes: list[Any] = []
                longitudes: list[Any] = []
                hover_text: list[Any] = []
                for line in band_lines:
                    voltage = line.get("voltage")
                    detail = " · ".join(
                        value
                        for value in (
                            line.get("name"),
                            f"{float(voltage):,.0f} kV" if voltage is not None else "Voltage unknown",
                            line.get("owner"),
                            line.get("operator"),
                            line.get("status"),
                        )
                        if value
                    )
                    for path in line.get("paths", []):
                        longitudes.extend(point[0] for point in path)
                        latitudes.extend(point[1] for point in path)
                        hover_text.extend([detail] * len(path))
                        longitudes.append(None)
                        latitudes.append(None)
                        hover_text.append(None)
                fig.add_trace(go.Scattermap(
                    lat=latitudes,
                    lon=longitudes,
                    mode="lines",
                    line={
                        "width": 2.1 if voltage_band in {"500–765 kV", "345–499 kV"} else 1.3,
                        "color": line_color,
                    },
                    name=f"Lines · {voltage_band}",
                    legendgroup="Transmission",
                    text=hover_text,
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=True,
                ))

        for layer_name, layer_color in colors.items():
            layer_df = filtered[filtered["layer"] == layer_name]
            if layer_df.empty:
                continue
            groups = (
                layer_df.groupby("type", dropna=False)
                if layer_name == "Power plant"
                else [(layer_name, layer_df)]
            )
            for group_name, group_df in groups:
                if layer_name == "Power plant":
                    marker_sizes = (
                        pd.to_numeric(group_df["capacity_mw"], errors="coerce")
                        .fillna(0)
                        .clip(lower=0)
                        .map(lambda value: min(18, 6 + np.sqrt(value) / 5))
                    )
                    marker_color = layer_color
                    trace_name = f"Plant · {group_name}"
                elif layer_name in {"Substation", "Autotransformer"}:
                    marker_sizes = (
                        pd.to_numeric(group_df["voltage"], errors="coerce")
                        .fillna(69)
                        .clip(lower=0)
                        .map(lambda value: min(10, 4 + value / 120))
                    )
                    marker_color = layer_color
                    trace_name = layer_name
                else:
                    marker_sizes = 11
                    marker_color = layer_color
                    trace_name = layer_name
                fig.add_trace(go.Scattermap(
                    lat=group_df["lat"],
                    lon=group_df["lon"],
                    mode="markers",
                    name=trace_name,
                    marker={
                        "size": marker_sizes,
                        "color": marker_color,
                        "symbol": symbols[layer_name],
                        "opacity": 0.84,
                    },
                    customdata=group_df[["type", "detail", "source"]],
                    text=group_df["name"],
                    hovertemplate=(
                        "<b>%{text}</b><br>%{customdata[0]}<br>%{customdata[1]}"
                        "<br><span style='color:#cbd5e1'>%{customdata[2]}</span>"
                        "<extra></extra>"
                    ),
                ))

        fig.update_layout(
            height=690,
            margin={"l": 0, "r": 0, "t": 8, "b": 0},
            paper_bgcolor="#07111f",
            map={
                "style": "carto-darkmatter",
                "center": {"lat": 31.0, "lon": -99.25},
                "zoom": 5.2,
            },
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 0.01,
                "xanchor": "center",
                "x": 0.5,
                "bgcolor": "rgba(7,17,31,.86)",
                "font": {"color": "#e2e8f0", "size": 10},
            },
        )
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "scrollZoom": True})

    with insight_col:
        st.markdown("#### Inspect a facility")
        if filtered.empty:
            st.info("No assets match the current filters.")
        else:
            record_options = filtered.index.tolist()
            selected_index = st.selectbox(
                "Asset",
                record_options,
                format_func=lambda index: (
                    f"{filtered.loc[index, 'name']} · {filtered.loc[index, 'layer']}"
                ),
                label_visibility="collapsed",
            )
            record = filtered.loc[selected_index]
            st.markdown(
                f"**{record['name']}**  \n"
                f"{record['layer']} · {record['type']}  \n"
                f"{record['detail']}"
            )
            if record.get("period"):
                st.caption(f"Source/reporting period: {record['period']}")
            st.caption(f"Source/method: {record['source']}")
            if record.get("source_url"):
                st.link_button(
                    "Open source layer",
                    str(record["source_url"]),
                    use_container_width=True,
                )

        st.markdown("#### Ask the engineering assistant")
        st.caption(
            "Use the assistant for requirements, guides, procedures, and xRR changes. "
            "The public map itself is not embedded."
        )
        st.link_button(
            "Ask about ERCOT planning",
            ercot_assistant_question_url(
                "What ERCOT planning requirements apply to the facilities or voltage level I am reviewing?"
            ),
            use_container_width=True,
        )

    source_col_1, source_col_2, source_col_3 = st.columns(3)
    with source_col_1:
        st.link_button(
            "Transmission source",
            TRANSMISSION_SOURCE_URL,
            use_container_width=True,
        )
    with source_col_2:
        st.link_button(
            "Substation source",
            SUBSTATION_SOURCE_URL,
            use_container_width=True,
        )
    with source_col_3:
        st.link_button(
            "Power-plant source",
            POWER_PLANT_SOURCE_URL,
            use_container_width=True,
        )
    st.warning(
        "Reference infrastructure only—not an ERCOT planning model, operating model, "
        "or real-time topology. HIFLD geometry is approximate; the plant and substation "
        "layers cover Texas, not only the ERCOT footprint. Data-center points and price-hub "
        "locations are contextual, and no illustrative price is shown."
    )


def render_north_american_grid_atlas() -> None:
    """Render one selected packaged U.S.–Canada Atlas shard."""

    render_section_header(
        "NERC U.S. & Canada Grid Atlas",
        "Explore public reference infrastructure across the United States and Canada, "
        "then filter to an approximate ISO/RTO footprint without downloading GIS data.",
    )

    try:
        manifest = load_grid_atlas_manifest_cached()
        region_definitions = grid_atlas_regions(manifest)
    except GridAtlasStoreError as exc:
        st.error(str(exc))
        return

    labels = [str(region["label"]) for region in region_definitions]
    id_by_label = {
        str(region["label"]): str(region["id"])
        for region in region_definitions
    }
    region_ids = set(id_by_label.values())
    requested_region = str(st.query_params.get("grid_region") or "")
    if requested_region not in region_ids:
        requested_region = str(manifest.get("default_region") or "ercot")
    default_label = next(
        (
            label
            for label, region_id in id_by_label.items()
            if region_id == requested_region
        ),
        labels[0],
    )
    selected_label = st.pills(
        "Grid area",
        labels,
        default=default_label,
        selection_mode="single",
        key="north_american_grid_atlas_region",
        help=(
            "All choices open checked-in compressed data. ISO/RTO footprints are "
            "approximate reference boundaries, not proof of asset membership."
        ),
    )
    if selected_label not in id_by_label:
        selected_label = default_label
    region_id = id_by_label[selected_label]
    st.query_params["grid_region"] = region_id
    region = grid_atlas_region(manifest, region_id)

    try:
        with st.spinner(f"Opening packaged {selected_label} infrastructure…"):
            payload = load_packaged_grid_region_cached(region_id)
    except GridAtlasStoreError as exc:
        st.error(str(exc))
        return

    generated_at = str(manifest.get("generated_at") or "not supplied")
    try:
        generated_at = (
            datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
            .astimezone(ERCOT_TIMEZONE)
            .strftime("%b %d, %Y %H:%M %Z")
        )
    except ValueError:
        pass
    artifact_megabytes = float(region.get("gzip_bytes") or 0) / 1_048_576
    st.caption(
        f"{region['label']} loaded from a {artifact_megabytes:.1f} MB packaged shard · "
        f"Atlas built {generated_at} · no bulk source, OpenAI, or embedding request."
    )
    st.info(str(region.get("detail") or manifest.get("disclaimer") or ""))

    public_assets = _atlas_public_assets(payload)
    if region_id == "ercot":
        context_assets = ercot_atlas_assets()
    else:
        context_assets = pd.DataFrame(columns=ercot_atlas_assets().columns)
    asset_frames = [public_assets]
    if not context_assets.empty:
        asset_frames.append(context_assets)
    assets = pd.concat(asset_frames, ignore_index=True, sort=False)
    for column, default in (
        ("name", ""),
        ("layer", ""),
        ("lat", np.nan),
        ("lon", np.nan),
        ("type", ""),
        ("detail", ""),
        ("source", ""),
        ("city", ""),
        ("status", ""),
        ("voltage", np.nan),
        ("capacity_mw", np.nan),
        ("period", ""),
        ("source_url", ""),
        ("country", ""),
    ):
        if column not in assets:
            assets[column] = default

    transmission_lines = list(payload.get("transmission_lines") or [])
    boundaries = list(payload.get("boundaries") or [])
    colors = {
        "Power plant": "#fde047",
        "Substation": "#f8fafc",
        "Autotransformer": "#f472b6",
    }
    if region_id == "ercot":
        colors.update(
            {
                "Data-center context": "#c084fc",
                "Price-hub context": "#fbbf24",
            }
        )
    symbols = {
        "Power plant": "circle",
        "Substation": "square",
        "Autotransformer": "diamond",
        "Data-center context": "diamond",
        "Price-hub context": "circle",
    }

    filter_col, search_col, line_col, boundary_col = st.columns([2.1, 1.6, 1, 1])
    with filter_col:
        selected_layers = st.multiselect(
            "Infrastructure layers",
            options=list(colors),
            default=list(colors),
            key=f"grid_atlas_layers_{region_id}",
            help="Public facilities and contextual ERCOT-only overlays remain distinct.",
        )
    with search_col:
        asset_query = st.text_input(
            "Find an asset",
            placeholder="Houston, wind, 345 kV, station…",
            key=f"grid_atlas_search_{region_id}",
        )
    with line_col:
        show_transmission = st.checkbox(
            "Transmission lines",
            value=True,
            key=f"grid_atlas_lines_{region_id}",
            disabled=not bool(transmission_lines),
        )
    with boundary_col:
        show_boundaries = st.checkbox(
            "Region boundaries",
            value=True,
            key=f"grid_atlas_boundaries_{region_id}",
            disabled=not bool(boundaries),
        )

    voltage_options = [0, 69, 100, 138, 230, 345, 500, 765]
    default_voltage = int(region.get("default_minimum_voltage") or 0)
    default_voltage_index = (
        voltage_options.index(default_voltage)
        if default_voltage in voltage_options
        else 0
    )
    voltage_col, unknown_col, capacity_col, fuel_col = st.columns([1.2, 1.3, 1.2, 1.8])
    with voltage_col:
        minimum_voltage = st.selectbox(
            "Minimum voltage",
            voltage_options,
            index=default_voltage_index,
            format_func=lambda value: "No minimum" if value == 0 else f"{value} kV",
            key=f"grid_atlas_minimum_voltage_v2_{region_id}",
        )
    with unknown_col:
        include_unknown_voltage = st.checkbox(
            "Include unknown voltage",
            value=bool(region.get("include_unknown_voltage", True)),
            key=f"grid_atlas_unknown_voltage_v2_{region_id}",
            help=(
                "Canadian CanVec lines have no voltage attribute. Missing HIFLD sentinel "
                "values are also treated as unknown, never as negative kV."
            ),
        )
    with capacity_col:
        minimum_capacity = st.number_input(
            "Minimum plant MW",
            min_value=0,
            max_value=5_000,
            value=0,
            step=25,
            key=f"grid_atlas_capacity_{region_id}",
        )
    available_fuels = sorted(
        {
            str(value)
            for value in public_assets.loc[
                public_assets.get("layer", pd.Series(dtype=str)) == "Power plant",
                "type",
            ].dropna()
        }
    )
    with fuel_col:
        selected_fuel = st.selectbox(
            "Plant fuel",
            ["All fuels", *available_fuels],
            key=f"grid_atlas_fuel_{region_id}",
        )

    filtered = assets[assets["layer"].isin(selected_layers)].copy()
    if {"Substation", "Autotransformer"} & set(selected_layers):
        is_substation = filtered["layer"].isin(["Substation", "Autotransformer"])
        known_voltage = pd.to_numeric(filtered["voltage"], errors="coerce")
        substation_allowed = known_voltage.ge(float(minimum_voltage))
        if include_unknown_voltage:
            substation_allowed = substation_allowed | known_voltage.isna()
        filtered = filtered[~is_substation | substation_allowed]
    is_plant = filtered["layer"] == "Power plant"
    plant_capacity = pd.to_numeric(filtered["capacity_mw"], errors="coerce")
    filtered = filtered[
        ~is_plant | plant_capacity.fillna(0).ge(float(minimum_capacity))
    ]
    if selected_fuel != "All fuels":
        filtered = filtered[
            (filtered["layer"] != "Power plant") | (filtered["type"] == selected_fuel)
        ]

    visible_lines = []
    for line in transmission_lines:
        voltage = line.get("voltage")
        if voltage is None:
            if not include_unknown_voltage:
                continue
        elif float(voltage) < float(minimum_voltage):
            continue
        visible_lines.append(line)

    if asset_query.strip():
        query = asset_query.strip().lower()
        search_columns = ["name", "type", "detail", "source", "city", "status"]
        searchable = (
            filtered[search_columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.lower()
        )
        filtered = filtered[searchable.str.contains(query, regex=False)]
        visible_lines = [
            line
            for line in visible_lines
            if query
            in " ".join(
                str(line.get(field) or "")
                for field in (
                    "name",
                    "id",
                    "owner",
                    "status",
                    "voltage",
                    "voltage_class",
                    "substation_1",
                    "substation_2",
                    "country",
                )
            ).lower()
        ]

    metric_columns = st.columns(3)
    metric_values = (
        ("Transmission lines", len(visible_lines), len(transmission_lines)),
        (
            "Power plants",
            int((filtered["layer"] == "Power plant").sum()),
            len(payload.get("power_plants") or []),
        ),
        (
            "Substations / transformers",
            int(filtered["layer"].isin(["Substation", "Autotransformer"]).sum()),
            len(payload.get("substations") or []),
        ),
    )
    for column, (label, visible_count, total_count) in zip(metric_columns, metric_values):
        with column:
            st.metric(
                label,
                f"{visible_count:,}",
                help=f"{total_count:,} records loaded in this packaged shard",
            )
    st.caption(
        "Packaged shard totals before display filters — "
        f"{len(transmission_lines):,} lines · "
        f"{len(payload.get('power_plants') or []):,} plants · "
        f"{len(payload.get('substations') or []):,} substations/transformers."
    )

    map_col, insight_col = st.columns([3.3, 1.15])
    with map_col:
        fig = go.Figure()
        if show_boundaries:
            boundary_styles = {
                "market": (
                    "#fbbf24",
                    "Approximate ISO/RTO footprint",
                ),
                "nerc": (
                    "#c084fc",
                    "Approximate U.S. NERC reference region",
                ),
            }
            for boundary_kind, (boundary_color, trace_name) in boundary_styles.items():
                kind_boundaries = [
                    boundary
                    for boundary in boundaries
                    if boundary.get("kind") == boundary_kind
                ]
                if not kind_boundaries:
                    continue
                latitudes: list[Any] = []
                longitudes: list[Any] = []
                hover_text: list[Any] = []
                for boundary in kind_boundaries:
                    label = str(boundary.get("label") or "Region")
                    for polygon in boundary.get("polygons", []):
                        for ring in [
                            polygon.get("outer", []),
                            *polygon.get("holes", []),
                        ]:
                            longitudes.extend(point[0] for point in ring)
                            latitudes.extend(point[1] for point in ring)
                            hover_text.extend([label] * len(ring))
                            longitudes.append(None)
                            latitudes.append(None)
                            hover_text.append(None)
                fig.add_trace(
                    go.Scattermap(
                        lat=latitudes,
                        lon=longitudes,
                        mode="lines",
                        line={"width": 1.8, "color": boundary_color},
                        name=trace_name,
                        text=hover_text,
                        hovertemplate="%{text}<extra></extra>",
                    )
                )

        line_colors = {
            "500–765 kV": "#f97316",
            "345–499 kV": "#22c55e",
            "230–344 kV": "#a855f7",
            "138–229 kV": "#0ea5e9",
            "69–137 kV": "#ef4444",
            "Below 69 kV": "#64748b",
            "Unknown voltage": "#94a3b8",
        }
        if show_transmission:
            for voltage_band, line_color in line_colors.items():
                band_lines = [
                    line
                    for line in visible_lines
                    if _atlas_voltage_band(line.get("voltage")) == voltage_band
                ]
                if not band_lines:
                    continue
                latitudes = []
                longitudes = []
                hover_text = []
                for line in band_lines:
                    voltage = line.get("voltage")
                    detail = " · ".join(
                        value
                        for value in (
                            line.get("name"),
                            (
                                f"{float(voltage):,.0f} kV"
                                if voltage is not None
                                else "Voltage unknown"
                            ),
                            line.get("owner"),
                            line.get("operator"),
                            line.get("status"),
                            line.get("country"),
                            (
                                " · ".join(
                                    value
                                    for value in (
                                        str(line.get("voltage_match_status") or ""),
                                        (
                                            f"{float(line['voltage_match_confidence']):.0%} confidence"
                                            if line.get("voltage_match_confidence")
                                            is not None
                                            else ""
                                        ),
                                    )
                                    if value
                                )
                                if line.get("voltage_source")
                                else ""
                            ),
                        )
                        if value
                    )
                    for path in line.get("paths", []):
                        longitudes.extend(point[0] for point in path)
                        latitudes.extend(point[1] for point in path)
                        hover_text.extend([detail] * len(path))
                        longitudes.append(None)
                        latitudes.append(None)
                        hover_text.append(None)
                fig.add_trace(
                    go.Scattermap(
                        lat=latitudes,
                        lon=longitudes,
                        mode="lines",
                        line={
                            "width": (
                                2.1
                                if voltage_band in {"500–765 kV", "345–499 kV"}
                                else 1.2
                            ),
                            "color": line_color,
                        },
                        name=f"Lines · {voltage_band}",
                        legendgroup="Transmission",
                        text=hover_text,
                        hovertemplate="%{text}<extra></extra>",
                    )
                )

        for layer_name, layer_color in colors.items():
            layer_df = filtered[filtered["layer"] == layer_name]
            if layer_df.empty:
                continue
            groups = (
                layer_df.groupby("type", dropna=False)
                if layer_name == "Power plant"
                else [(layer_name, layer_df)]
            )
            for group_name, group_df in groups:
                if layer_name == "Power plant":
                    marker_sizes = (
                        pd.to_numeric(group_df["capacity_mw"], errors="coerce")
                        .fillna(0)
                        .clip(lower=0)
                        .map(lambda value: min(18, 6 + np.sqrt(value) / 5))
                    )
                    marker_color = layer_color
                    trace_name = f"Plant · {group_name}"
                elif layer_name in {"Substation", "Autotransformer"}:
                    marker_sizes = (
                        pd.to_numeric(group_df["voltage"], errors="coerce")
                        .fillna(69)
                        .clip(lower=0)
                        .map(lambda value: min(10, 4 + value / 120))
                    )
                    marker_color = layer_color
                    trace_name = layer_name
                else:
                    marker_sizes = 11
                    marker_color = layer_color
                    trace_name = layer_name
                fig.add_trace(
                    go.Scattermap(
                        lat=group_df["lat"],
                        lon=group_df["lon"],
                        mode="markers",
                        name=trace_name,
                        marker={
                            "size": marker_sizes,
                            "color": marker_color,
                            "symbol": symbols[layer_name],
                            "opacity": 0.84,
                        },
                        customdata=group_df[["type", "detail", "source"]],
                        text=group_df["name"],
                        hovertemplate=(
                            "<b>%{text}</b><br>%{customdata[0]}<br>%{customdata[1]}"
                            "<br><span style='color:#cbd5e1'>%{customdata[2]}</span>"
                            "<extra></extra>"
                        ),
                    )
                )

        map_center = region.get("center") or {"lat": 46.0, "lon": -101.0}
        fig.update_layout(
            height=700,
            margin={"l": 0, "r": 0, "t": 8, "b": 0},
            paper_bgcolor="#07111f",
            map={
                "style": "carto-darkmatter",
                "center": {
                    "lat": float(map_center.get("lat") or 46.0),
                    "lon": float(map_center.get("lon") or -101.0),
                },
                "zoom": float(region.get("zoom") or 2.0),
            },
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 0.01,
                "xanchor": "center",
                "x": 0.5,
                "bgcolor": "rgba(7,17,31,.86)",
                "font": {"color": "#e2e8f0", "size": 10},
            },
        )
        st.plotly_chart(
            fig,
            width="stretch",
            config={"displaylogo": False, "scrollZoom": True},
        )

    with insight_col:
        st.markdown("#### Inspect a facility")
        if filtered.empty:
            st.info("No assets match the current filters.")
        else:
            inspector_records = filtered.sort_values("name").head(250)
            if len(filtered) > len(inspector_records):
                st.caption(
                    f"Showing 250 of {len(filtered):,} matching facilities. "
                    "Use Find an asset to narrow the list."
                )
            record_options = inspector_records.index.tolist()
            selected_index = st.selectbox(
                "Asset",
                record_options,
                format_func=lambda index: (
                    f"{inspector_records.loc[index, 'name']} · "
                    f"{inspector_records.loc[index, 'layer']}"
                ),
                label_visibility="collapsed",
                key=f"grid_atlas_inspector_{region_id}",
            )
            record = inspector_records.loc[selected_index]
            st.markdown(
                f"**{record['name']}**  \n"
                f"{record['layer']} · {record['type']}  \n"
                f"{record['detail']}"
            )
            if record.get("period"):
                st.caption(f"Source/reporting period: {record['period']}")
            st.caption(f"Source/method: {record['source']}")
            if record.get("source_url"):
                st.link_button(
                    "Open source layer",
                    str(record["source_url"]),
                    use_container_width=True,
                )

        if region_id == "ercot":
            st.markdown("#### Ask the ERCOT assistant")
            st.caption(
                "Use the assistant for ERCOT requirements, guides, procedures, and xRR "
                "changes. Public map geometry is not engineering evidence."
            )
            st.link_button(
                "Ask about ERCOT planning",
                ercot_assistant_question_url(
                    "What ERCOT planning requirements apply to the facilities or voltage "
                    "level I am reviewing?"
                ),
                use_container_width=True,
            )

    source_links = [
        ("U.S. transmission", manifest["sources"]["us_transmission_lines"]),
        ("U.S. substations", manifest["sources"]["us_substations"]),
        ("U.S. power plants", manifest["sources"]["us_power_plants"]),
        (
            "Canada lines & transformers",
            manifest["sources"]["canada_lines_and_transformers"],
        ),
        ("Canada power plants", manifest["sources"]["canada_power_plants"]),
        ("ISO/RTO boundaries", manifest["sources"]["iso_rto_boundaries"]),
    ]
    for row_start in range(0, len(source_links), 3):
        source_columns = st.columns(3)
        for column, (label, url) in zip(
            source_columns,
            source_links[row_start : row_start + 3],
        ):
            with column:
                st.link_button(label, url, use_container_width=True)

    st.warning(
        str(manifest.get("disclaimer") or "")
        + " Canadian CanVec line/transformer attributes extend through 2015 and omit "
        "voltage, owner, and electrical connectivity; NACEI plant data are an August "
        "2017 reference. Displayed NERC regional polygons cover the contiguous U.S. only."
    )


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

    action_col_1, action_col_2, action_col_3, action_col_4, action_col_5 = st.columns(5)
    with action_col_1:
        if st.button(
            "Refresh data",
            help=(
                "Clear cached API/news data and rerun the dashboard. Grid Atlas region "
                "files remain packaged locally and do not trigger a bulk source download."
            ),
            type="primary",
            use_container_width=True,
        ):
            st.cache_data.clear()
            st.rerun()
    with action_col_2:
        st.link_button(
            "ERCOT Assistant",
            ERCOT_ASSISTANT_URL,
            help="Ask cited questions about ERCOT guides, procedures, and revision requests.",
            use_container_width=True,
        )
    with action_col_3:
        st.link_button(
            "Telegram",
            "https://t.me/ERCOTNEWS",
            help="Open the ERCOT News Telegram channel",
            use_container_width=True,
        )
    with action_col_4:
        st.link_button(
            "Lovable App",
            LOVABLE_ERCOT_DASHBOARD_URL,
            help="Open the Lovable version of this ERCOT dashboard",
            use_container_width=True,
        )
    with action_col_5:
        st.link_button(
            "API Credentials",
            ERCOT_API_MARKET_URL,
            help="Open ERCOT API Market to sign in, subscribe to the public API, and copy your subscription key.",
            use_container_width=True,
        )
    st.markdown(
        f"<div class='small-muted' style='text-align:right; padding-top:0.25rem;'>"
        f"Last dashboard render: {datetime.now().strftime('%b %d, %Y %H:%M:%S')} | Build {APP_BUILD}"
        f"</div>",
        unsafe_allow_html=True,
    )
    requested_view = str(st.query_params.get("view", "")).strip().casefold()
    if requested_view in {"documents", "changes", "ercot-documents"}:
        default_view = "ERCOT Documents & Changes"
    elif requested_view in {"atlas", "grid-atlas", "texas-grid"}:
        default_view = "Grid Atlas"
    else:
        default_view = "Grid Operations & Analytics"
    dashboard_view = st.pills(
        "Dashboard view",
        ["Grid Operations & Analytics", "Grid Atlas", "ERCOT Documents & Changes"],
        default=default_view,
        selection_mode="single",
        label_visibility="collapsed",
        key="ercot_dashboard_view",
    )
    if dashboard_view == "ERCOT Documents & Changes":
        render_latest_ercot_documents()
        return
    if dashboard_view == "Grid Atlas":
        render_north_american_grid_atlas()
        return

    # Unified news prefixes (ERCOT + regulatory updates in one panel)
    all_news_prefixes = [
        "ercot_news_", "ercot_summary_", "summary_ercot_",
        "summary_",
        "datacenter_news_", "data_center_news_", "dc_news_",
        "nogrr_", "nodal_operating_guide_", "nodal_guide_change_",
        "pgrr_", "planning_guide_change_", "planning_guide_update_",
    ]

    item = get_latest_news_by_prefix(all_news_prefixes, repo_path="ERCOTAPI/news_summaries")
    brief_state = assess_news_brief(item["name"]) if item else None
    st.markdown("<div class='news-panel'>", unsafe_allow_html=True)
    news_header_col, news_status_col = st.columns([5, 1.4])
    with news_header_col:
        render_section_header(
            "ERCOT Intelligence Brief",
            "Latest n8n-generated Texas grid, market-rule, interconnection, and large-load monitoring summary.",
        )
    with news_status_col:
        render_status_pill(
            brief_state.label if brief_state else "Waiting for update",
            brief_state.status if brief_state else "warn",
        )
    if item:
        caption = f"Latest file: {item['name']}"
        if brief_state and brief_state.published_at:
            published_ct = brief_state.published_at.astimezone(ZoneInfo("America/Chicago"))
            caption += (
                f" · Published {published_ct:%b %d, %Y %H:%M} CT"
                f" · {format_brief_age(brief_state.age_hours)}"
            )
        st.caption(caption)
        if brief_state and not brief_state.is_fresh:
            st.warning(
                "The n8n publisher has not delivered a fresh ERCOT brief. "
                "Refresh data reloads published files, but it does not run the workflow."
            )
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
            3. **Expired credentials** - You may need to reset your password at {ERCOT_API_MARKET_URL}
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
            st.markdown(f"1. Go to [ERCOT API Market]({ERCOT_API_MARKET_URL})")
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
        
        st.info(f"""
        **How to get ERCOT API credentials:**
        
        **Important:** You need TWO things:
        1. **Subscription Key** (Primary requirement):
           - Register at {ERCOT_API_MARKET_URL}
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
        "Resource Outages",
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
                    
                    actual_df = add_ercot_timestamp(actual_df)
                    forecast_df = add_ercot_timestamp(forecast_df)

                    # ERCOT uses operatingDay + hourEnding; resample after parsing one timestamp per hour.
                    if 'timestamp' in actual_df.columns:
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
                        actual_df['timestamp'] = normalize_timestamps(actual_df['timestamp'])
                        actual_df = actual_df.dropna(subset=['timestamp'])
                                                                
                    # Display metrics - use total/system-wide load when available.
                    load_col = find_column(
                        actual_df,
                        exact=["total", "ercot", "system_wide", "systemwide", "systemTotal"],
                        contains=["total"],
                    )
                    
                    if load_col and not actual_df.empty:
                        load_series = coerce_numeric(actual_df[load_col]).dropna()
                        latest_load, latest_load_note = latest_non_null_reading(
                            actual_df,
                            load_col,
                            as_of=current_interval_cutoff(),
                            note_prefix="As of",
                            stale_after_hours=4,
                        )
                        if latest_load is None:
                            latest_load, latest_load_note = latest_non_null_reading(
                                actual_df,
                                load_col,
                                note_prefix="Latest returned",
                            )
                        avg_load = load_series.mean()
                        max_load = load_series.max()
                        min_load = load_series.min()
                        load_factor = (avg_load / max_load * 100) if max_load else np.nan

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            render_metric_card("Latest Actual Load", format_mw(latest_load), latest_load_note, ERCOT_CYAN)
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
                    
                    render_chart(fig)
                    
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
                                render_chart(fig_diag)

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
                            render_chart(fig_ml)
                            
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
                        
                        wind_df = add_ercot_timestamp(wind_df)
                        
                        # Add actual and forecast traces - API uses 'genSystemWide' not 'ACTUAL_SYSTEM_WIDE'
                        # Find actual column
                        actual_col = find_column(wind_df, exact=["genSystemWide"], contains=["gen", "system"])
                        forecast_col = find_column(wind_df, contains=["stwpf", "system"])
                        wind_series_all = compact_time_series(wind_df, [actual_col, forecast_col])
                        wind_series = time_window_around_now(wind_series_all, hours_back=48, hours_forward=72)
                        if wind_series.empty:
                            wind_series = wind_series_all

                        if debug_mode:
                            st.write("**Wind Data Sample:**")
                            render_dataframe(wind_df.head(), height=180)
                            st.write(f"Columns: {list(wind_df.columns)}")
                            st.write("**Cleaned Wind Chart Series:**")
                            render_dataframe(wind_series.head(), height=180)
                        
                        fig_wind = go.Figure()
                        x_axis_wind = wind_series['timestamp'] if 'timestamp' in wind_series.columns else wind_series.index

                        if has_numeric_values(wind_series, actual_col):
                            fig_wind.add_trace(
                                go.Scatter(
                                    x=x_axis_wind,
                                    y=wind_series[actual_col],
                                    mode='lines',
                                    name='Actual Wind',
                                    line=dict(color=ERCOT_CYAN, width=2.6),
                                    fill="tozeroy",
                                    fillcolor="rgba(0, 163, 199, 0.10)",
                                )
                            )
                        
                        if has_numeric_values(wind_series, forecast_col):
                            fig_wind.add_trace(
                                go.Scatter(
                                    x=x_axis_wind,
                                    y=wind_series[forecast_col],
                                    mode='lines',
                                    name='Wind Forecast',
                                    line=dict(color=ERCOT_BLUE, width=2.3, dash='dash')
                                )
                            )

                        wind_value, wind_note = latest_non_null_reading(
                            wind_series_all,
                            actual_col,
                            as_of=current_interval_cutoff(),
                            note_prefix="As of",
                            stale_after_hours=3,
                        )
                        forecast_wind_value, forecast_wind_note = latest_non_null_reading(
                            wind_series_all,
                            forecast_col,
                            note_prefix="Forecast through",
                        )
                        wind_metric_col_1, wind_metric_col_2 = st.columns(2)
                        with wind_metric_col_1:
                            render_metric_card(
                                "Latest Wind Actual",
                                format_mw(wind_value) if wind_value is not None else "N/A",
                                wind_note,
                                ERCOT_CYAN,
                            )
                        with wind_metric_col_2:
                            render_metric_card(
                                "Wind Forecast Horizon",
                                format_mw(forecast_wind_value) if forecast_wind_value is not None else "N/A",
                                forecast_wind_note,
                                ERCOT_BLUE,
                            )

                        fig_wind = apply_professional_layout(
                            fig_wind,
                            "Wind Generation: Actual vs Forecast",
                            "Generation (MW)",
                            height=430,
                        )
                        st.caption("Chart view: past 48 hours and next 72 forecast hours, grouped to one value per ERCOT interval.")
                        
                        if len(fig_wind.data) > 0:
                            render_chart(fig_wind)
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
                        
                        solar_df = add_ercot_timestamp(solar_df)
                        
                        # Find actual solar column - API uses 'genSystemWide'
                        actual_solar_col = find_column(solar_df, exact=["genSystemWide"], contains=["gen", "system"])
                        forecast_solar_col = find_column(solar_df, contains=["stppf", "system"])
                        solar_series_all = compact_time_series(solar_df, [actual_solar_col, forecast_solar_col])
                        solar_series = time_window_around_now(solar_series_all, hours_back=48, hours_forward=72)
                        if solar_series.empty:
                            solar_series = solar_series_all

                        if debug_mode:
                            st.write("**Solar Data Sample:**")
                            render_dataframe(solar_df.head(), height=180)
                            st.write(f"Columns: {list(solar_df.columns)}")
                            st.write("**Cleaned Solar Chart Series:**")
                            render_dataframe(solar_series.head(), height=180)
                        
                        fig_solar = go.Figure()
                        x_axis_solar = solar_series['timestamp'] if 'timestamp' in solar_series.columns else solar_series.index

                        if has_numeric_values(solar_series, actual_solar_col):
                            fig_solar.add_trace(
                                go.Scatter(
                                    x=x_axis_solar,
                                    y=solar_series[actual_solar_col],
                                    mode='lines',
                                    name='Actual Solar',
                                    line=dict(color=ERCOT_ORANGE, width=2.6),
                                    fill="tozeroy",
                                    fillcolor="rgba(245, 158, 11, 0.12)",
                                )
                            )
                        
                        if has_numeric_values(solar_series, forecast_solar_col):
                            fig_solar.add_trace(
                                go.Scatter(
                                    x=x_axis_solar,
                                    y=solar_series[forecast_solar_col],
                                    mode='lines',
                                    name='Solar Forecast',
                                    line=dict(color=ERCOT_RED, width=2.3, dash='dash')
                                )
                            )

                        solar_value, solar_note = latest_non_null_reading(
                            solar_series_all,
                            actual_solar_col,
                            as_of=current_interval_cutoff(),
                            note_prefix="As of",
                            stale_after_hours=3,
                        )
                        forecast_solar_value, forecast_solar_note = latest_non_null_reading(
                            solar_series_all,
                            forecast_solar_col,
                            note_prefix="Forecast through",
                        )
                        solar_metric_col_1, solar_metric_col_2 = st.columns(2)
                        with solar_metric_col_1:
                            render_metric_card(
                                "Latest Solar Actual",
                                format_mw(solar_value) if solar_value is not None else "N/A",
                                solar_note,
                                ERCOT_ORANGE,
                            )
                        with solar_metric_col_2:
                            render_metric_card(
                                "Solar Forecast Horizon",
                                format_mw(forecast_solar_value) if forecast_solar_value is not None else "N/A",
                                forecast_solar_note,
                                ERCOT_RED,
                            )

                        fig_solar = apply_professional_layout(
                            fig_solar,
                            "Solar Generation: Actual vs Forecast",
                            "Generation (MW)",
                            height=430,
                        )
                        st.caption("Chart view: past 48 hours and next 72 forecast hours, grouped to one value per ERCOT interval.")
                        
                        if len(fig_solar.data) > 0:
                            render_chart(fig_solar)
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
                    lmp_df = add_ercot_timestamp(dataframe_from_payload(lmp_data))
                    latest_lmp_df = rows_at_latest_timestamp(
                        lmp_df,
                        as_of=ercot_now() + pd.Timedelta(minutes=10),
                    )
                    if latest_lmp_df.empty:
                        latest_lmp_df = lmp_df
                    if "timestamp" in latest_lmp_df.columns and not latest_lmp_df.empty:
                        lmp_interval_note = interval_freshness_note(
                            latest_lmp_df["timestamp"].max(),
                            prefix="SCED interval",
                            stale_after_hours=1,
                        )
                    else:
                        lmp_interval_note = "Latest returned interval"
                    
                    point_col = find_settlement_point_column(lmp_df)
                    type_col = find_settlement_type_column(lmp_df)
                    price_col = find_price_column(lmp_df)
                    if point_col:
                        hubs = latest_lmp_df
                        if type_col:
                            type_values = hubs[type_col].astype(str).str.upper()
                            hub_mask = type_values.eq("HU") | type_values.str.contains("HUB", na=False)
                            if hub_mask.any():
                                hubs = hubs[hub_mask]
                        
                        if not hubs.empty and price_col:
                            hubs = hubs.copy()
                            hubs[price_col] = coerce_numeric(hubs[price_col])
                            hubs = hubs.dropna(subset=[price_col]).sort_values(price_col, ascending=False)
                            if hubs.empty:
                                st.info("Pricing rows were returned, but no numeric price values were available after parsing.")
                                render_dataframe(lmp_df.head(30), height=360)
                            else:
                                price_series = hubs[price_col]
                                price_col_1, price_col_2, price_col_3, price_col_4 = st.columns(4)
                                with price_col_1:
                                    render_metric_card("Highest Hub Price", format_price(price_series.max()), lmp_interval_note, ERCOT_RED)
                                with price_col_2:
                                    render_metric_card("Average Hub Price", format_price(price_series.mean()), "Simple hub average", ERCOT_BLUE)
                                with price_col_3:
                                    render_metric_card("Lowest Hub Price", format_price(price_series.min()), "Visible hub set", ERCOT_GREEN)
                                with price_col_4:
                                    render_metric_card("Hub Count", f"{len(hubs):,}", "Latest interval rows", ERCOT_CYAN)

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
                                    paper_bgcolor="#ffffff",
                                    plot_bgcolor="#ffffff",
                                    font=dict(color="#0f172a"),
                                )
                                fig_lmp.update_xaxes(
                                    title_text="Settlement Point",
                                    tickangle=-35,
                                    showgrid=False,
                                    tickfont=dict(color="#0f172a", size=12),
                                    title_font=dict(color="#0f172a", size=14),
                                    automargin=True,
                                )
                                fig_lmp.update_yaxes(
                                    title_text="Price ($/MWh)",
                                    showgrid=True,
                                    gridcolor="#eef2f7",
                                    zerolinecolor="#cbd5e1",
                                    tickfont=dict(color="#0f172a", size=12),
                                    title_font=dict(color="#0f172a", size=14),
                                    automargin=True,
                                )
                                render_chart(fig_lmp)
                                
                                table_cols = [point_col, price_col]
                                if type_col:
                                    table_cols.insert(1, type_col)
                                render_dataframe(hubs[table_cols].head(30), height=360)
                        else:
                            st.info("Pricing values were not found in the returned payload. Enable Debug Mode to inspect the ERCOT columns.")
                    else:
                        st.info("Settlement point column was not found. Showing the returned pricing payload.")
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
                    outage_df = add_ercot_timestamp(dataframe_from_payload(outage_data))
                    
                    if debug_mode:
                        st.write("**Outage Data Sample:**")
                        render_dataframe(outage_df.head(), height=180)
                        st.write(f"Columns: {list(outage_df.columns)}")
                    
                    numeric_cols = get_outage_capacity_columns(outage_df)
                    
                    if len(numeric_cols) > 0:
                        outage_series = compact_time_series(outage_df, numeric_cols)
                        outage_chart_df = time_window_around_now(outage_series, hours_back=1, hours_forward=72)
                        if outage_chart_df.empty:
                            outage_chart_df = outage_series
                        
                        if debug_mode:
                            st.write("**Cleaned Outage Chart Series:**")
                            render_dataframe(outage_chart_df.head(), height=180)
                        
                        current_row = latest_row_at_or_before(
                            outage_series,
                            numeric_cols,
                            as_of=current_interval_cutoff(),
                        )
                        current_note_prefix = "Current interval"
                        if current_row is None:
                            current_row = latest_row_at_or_before(outage_series, numeric_cols)
                            current_note_prefix = "Latest returned"

                        horizon_row = latest_row_at_or_before(outage_series, numeric_cols)
                        current_total = sum(float(current_row[col]) for col in numeric_cols if pd.notna(current_row[col])) if current_row is not None else np.nan
                        horizon_total = sum(float(horizon_row[col]) for col in numeric_cols if pd.notna(horizon_row[col])) if horizon_row is not None else np.nan
                        current_note = (
                            f"{current_note_prefix} {format_timestamp_label(current_row['timestamp'])}"
                            if current_row is not None and "timestamp" in current_row.index
                            else "Latest available interval"
                        )
                        horizon_note = (
                            f"Forecast through {format_timestamp_label(horizon_row['timestamp'])}"
                            if horizon_row is not None and "timestamp" in horizon_row.index
                            else latest_timestamp_label(outage_series)
                        )

                        avg_total = outage_chart_df[numeric_cols].sum(axis=1).mean()
                        peak_total = outage_chart_df[numeric_cols].sum(axis=1).max()
                        top_latest = sorted(
                            [(col, float(current_row[col])) for col in numeric_cols if current_row is not None and pd.notna(current_row[col])],
                            key=lambda item: item[1],
                            reverse=True,
                        )[:1]
                        top_label = humanize_ercot_column(top_latest[0][0]) if top_latest else "N/A"

                        out_col_1, out_col_2, out_col_3, out_col_4, out_col_5 = st.columns(5)
                        with out_col_1:
                            render_metric_card("Current Scheduled Outages", format_mw(current_total), current_note, ERCOT_RED)
                        with out_col_2:
                            render_metric_card("Average Next 72h", format_mw(avg_total), "Outage Scheduler window", ERCOT_BLUE)
                        with out_col_3:
                            render_metric_card("Peak Next 72h", format_mw(peak_total), "Outage Scheduler window", ERCOT_ORANGE)
                        with out_col_4:
                            render_metric_card("Forecast Horizon", format_mw(horizon_total), horizon_note, ERCOT_GREEN)
                        with out_col_5:
                            render_metric_card("Top Current Category", str(top_label), "Largest current value", ERCOT_CYAN)

                        fig_outage = go.Figure()
                        x_axis_outage = outage_chart_df['timestamp'] if 'timestamp' in outage_chart_df.columns else outage_chart_df.index
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
                                    y=outage_chart_df[col],
                                    mode='lines',
                                    name=humanize_ercot_column(col),
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
                        st.caption(
                            "Chart view: current interval plus next 72 forecast hours from the Outage Scheduler. "
                            "The report covers the next 168 hours; future dates are forecast horizons, not live actuals."
                        )
                        render_chart(fig_outage)
                    else:
                        st.info("No numeric outage capacity columns were found in the returned payload.")
                    
                    st.subheader("Outage Summary Statistics")
                    render_dataframe(outage_chart_df[numeric_cols].describe() if numeric_cols else outage_df.describe(), height=360)
                else:
                    st.warning("No outage data available.")
        except Exception as e:
            st.error(f"Error fetching outage data: {e}")


# Run the Streamlit dashboard
if __name__ == "__main__":
    main()
