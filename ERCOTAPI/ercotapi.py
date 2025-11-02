import os
import requests
from typing import Optional, Dict, Any
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
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
                    print("✅ Bearer token acquired.")
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

    def _make_request(self, base_url: str, endpoint: str, key: str, params: Optional[Dict[str, Any]] = None, verbose: bool = False) -> Dict:
        """Internal method to send a GET request to ERCOT API."""
        url = f"{base_url}/{endpoint.lstrip('/')}"
        
        # Prepare headers
        headers = {}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        if key or self.subscription_key:
            headers["Ocp-Apim-Subscription-Key"] = key or self.subscription_key
        
        if verbose:
            print(f"🌐 Requesting: {url}")
            print(f"🔍 Params: {params}")
            print(f"🔑 Headers: {list(headers.keys())}")
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    def get_public(self, endpoint: str, params: Optional[Dict[str, Any]] = None, verbose: bool = False) -> Dict:
        """Query ERCOT Public Reports API (base: https://api.ercot.com/api/public-reports)"""
        if not self.bearer_token and not self.subscription_key:
            raise ValueError("Missing ERCOT Bearer token or Subscription Key. You need BOTH authentication methods.")

        base_url = "https://api.ercot.com/api/public-reports"
        if verbose:
            print(f"🔄 Using Public Reports API: {base_url}/{endpoint}")

        return self._make_request(base_url, endpoint, self.public_key, params, verbose)



# --- Helper Functions ---
def train_load_forecast_model(historical_data):
    """Train an ML model (XGBoost or Random Forest) to forecast load based on historical patterns."""
    if len(historical_data) < 48:  # Need at least 2 days
        return None, None, None
    
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
        return None, None, None
    
    X = df[features]
    y = df['load']
    
    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if HAS_XGBOOST:
        # XGBoost with optimized hyperparameters for time series
        model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective='reg:squarederror',
            verbosity=0
        )
    else:
        # Fallback to Random Forest
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=15,
            min_samples_split=5
        )
    
    model.fit(X_scaled, y)
    
    return model, scaler, df


# --- Streamlit Dashboard ---
def main():
    st.set_page_config(page_title="ERCOT Grid Analytics Dashboard", layout="wide")
    st.title("⚡ ERCOT Grid Analytics & Forecasting Dashboard")
    st.markdown("**Real-time grid monitoring, renewable generation tracking, and ML-powered load forecasting**")

    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Credentials section
    st.sidebar.subheader("🔐 API Credentials")
    
    # Check if credentials are in environment or Streamlit secrets
    # Check for non-empty values
    env_username = os.getenv("ERCOT_USERNAME", "").strip()
    env_password = os.getenv("ERCOT_PASSWORD", "").strip()
    env_client_id = os.getenv("ERCOT_CLIENT_ID", "").strip()
    env_sub_key = os.getenv("ERCOT_SUBSCRIPTION_KEY", "").strip()
    
    has_env_creds = bool(env_username and env_password and env_client_id and env_sub_key)
    has_secrets = False
    
    # Try Streamlit secrets (from .streamlit/secrets.toml or Streamlit Cloud secrets manager)
    if not has_env_creds:
        try:
            has_secrets = bool(st.secrets.get("ERCOT_USERNAME") and st.secrets.get("ERCOT_PASSWORD") and 
                             st.secrets.get("ERCOT_CLIENT_ID") and st.secrets.get("ERCOT_SUBSCRIPTION_KEY"))
        except:
            pass
    
    # Check if user wants to save credentials to session state (for this session only)
    if 'saved_credentials' not in st.session_state:
        st.session_state.saved_credentials = None
    
    # Check session state for saved credentials
    if st.session_state.saved_credentials:
        st.sidebar.success("✅ Using saved credentials (this session only)")
        try:
            api = ErcotAPI(**st.session_state.saved_credentials)
        except Exception as e:
            st.sidebar.error(f"❌ Failed to authenticate")
            st.error(f"**Authentication Error:** {str(e)}")
            st.session_state.saved_credentials = None
            api = None
    elif has_env_creds:
        st.sidebar.success("✅ Using credentials from environment variables")
        try:
            api = ErcotAPI()
        except Exception as e:
            st.sidebar.error(f"❌ Failed to authenticate with environment credentials")
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
        st.sidebar.success("✅ Using credentials from Streamlit secrets file")
        st.sidebar.info("📁 Credentials stored in `.streamlit/secrets.toml` (not in code!)")
        try:
            api = ErcotAPI(
                username=st.secrets["ERCOT_USERNAME"],
                password=st.secrets["ERCOT_PASSWORD"],
                client_id=st.secrets["ERCOT_CLIENT_ID"],
                subscription_key=st.secrets["ERCOT_SUBSCRIPTION_KEY"]
            )
        except Exception as e:
            st.sidebar.error(f"❌ Failed to authenticate with secrets")
            st.error(f"""
            **Authentication Error:** {str(e)}
            
            **To fix:** Edit `.streamlit/secrets.toml` and verify all credentials are correct
            """)
            api = None
    else:
        st.sidebar.warning("⚠️ No environment credentials found. Please enter manually:")
        
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
                    st.sidebar.success("✅ Credentials accepted! Bearer token acquired.")
                    
                    # Ask if user wants to save for this session
                    save_creds = st.sidebar.checkbox("💾 Remember credentials for this session", value=False,
                                                     help="Credentials will be stored in memory (not saved to disk) for this browser session only")
                    if save_creds:
                        st.session_state.saved_credentials = {
                            'username': username,
                            'password': password,
                            'client_id': client_id,
                            'subscription_key': subscription_key
                        }
                        st.sidebar.success("✅ Credentials saved for this session!")
                        st.sidebar.info("🔒 Secure: Credentials only in browser memory, not saved to disk")
                except Exception as e:
                    st.sidebar.error(f"❌ Authentication failed: {e}")
                    api = None
            else:
                st.sidebar.info("👆 Please enter all credentials above to continue")
                api = None
    
    # Only show the rest of the app if API is initialized
    if api is None:
        st.warning("⚠️ Please configure API credentials to access ERCOT data.")
        
        # Check if running on Streamlit Cloud
        is_cloud = os.getenv("STREAMLIT_SHARING_MODE") or os.getenv("STREAMLIT_CLOUD")
        
        if is_cloud:
            st.error("""
            **🚀 Streamlit Cloud Deployment Detected**
            
            To configure secrets on Streamlit Cloud:
            1. Go to your app dashboard: https://share.streamlit.io/
            2. Click on your app
            3. Click the **⚙️ Settings** button (three dots menu)
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
    
    # Date range selector
    date_range = st.sidebar.slider(
        "Historical Days to Load",
        min_value=1,
        max_value=7,
        value=3,
        help="Number of days of historical data to fetch"
    )
    
    # Debug mode
    debug_mode = st.sidebar.checkbox("🔍 Debug Mode", value=False, help="Show detailed API request/response info")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=date_range)
    
    # Dashboard sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Load Analysis & Forecast", 
        "🌬️ Renewable Generation", 
        "💰 Real-Time Pricing",
        "⚠️ Resource Outages"
    ])
    
    # TAB 1: LOAD ANALYSIS & FORECAST
    with tab1:
        st.header("System Load: Actual vs Forecast with ML Prediction")
        
        try:
            with st.spinner("Fetching load data..."):
                # Fetch actual load data using correct parameters: operatingDayFrom/To
                load_params = {
                    "operatingDayFrom": start_date.strftime("%Y-%m-%d"),
                    "operatingDayTo": end_date.strftime("%Y-%m-%d"),
                    "page": 1,
                    "size": 5000
                }
                
                actual_load_data = api.get_public("np6-345-cd/act_sys_load_by_wzn", params=load_params, verbose=debug_mode)
                if debug_mode:
                    st.json({"params_used": load_params, "response_keys": list(actual_load_data.keys()) if isinstance(actual_load_data, dict) else "N/A"})
                
                try:
                    forecast_data = api.get_public("np3-565-cd/lf_by_model_weather_zone", params=load_params, verbose=False)
                except:
                    forecast_data = {"data": []}
                
                if "data" in actual_load_data and len(actual_load_data["data"]) > 0:
                    actual_df = pd.DataFrame(actual_load_data["data"])
                    forecast_df = pd.DataFrame(forecast_data["data"]) if "data" in forecast_data else pd.DataFrame()
                    
                    # Debug: Show first few rows
                    if debug_mode:
                        st.write("**Load Data Sample:**")
                        st.dataframe(actual_df.head())
                        st.write(f"Columns: {list(actual_df.columns)}")
                        if "fields" in actual_load_data:
                            st.write("**Fields metadata:**")
                            st.json(actual_load_data["fields"])
                    
                    # Map column indices to names using 'fields' metadata
                    if "fields" in actual_load_data and isinstance(actual_load_data["fields"], list):
                        column_mapping = {i: field.get('name', f'col_{i}') for i, field in enumerate(actual_load_data["fields"])}
                        actual_df.rename(columns=column_mapping, inplace=True)
                        if debug_mode:
                            st.write("**Renamed Columns:**", list(actual_df.columns))
                    
                    # Parse timestamp: ERCOT uses operatingDay + hourEnding (hour 24 = midnight next day)
                    if 'operatingDay' in actual_df.columns and 'hourEnding' in actual_df.columns:
                        def parse_ercot_timestamp(row):
                            try:
                                date = pd.to_datetime(row['operatingDay'])
                                hour = int(row['hourEnding'])
                                if hour == 24:
                                    return date + timedelta(days=1)
                                else:
                                    return date + timedelta(hours=hour)
                            except:
                                return pd.NaT
                        actual_df['timestamp'] = actual_df.apply(parse_ercot_timestamp, axis=1)
                        actual_df = actual_df.sort_values('timestamp')
                    
                    # Display metrics - use 'total' column for system-wide load
                    col1, col2, col3 = st.columns(3)
                    load_col = None
                    for col in actual_df.columns:
                        if isinstance(col, str) and col.lower() in ['total', 'ercot', 'system_wide', 'systemwide']:
                            load_col = col
                            break
                    
                    if load_col and not actual_df.empty:
                        latest_load = actual_df[load_col].iloc[-1]
                        avg_load = actual_df[load_col].mean()
                        max_load = actual_df[load_col].max()
                        
                        col1.metric("Current System Load", f"{latest_load:,.0f} MW")
                        col2.metric("Average Load", f"{avg_load:,.0f} MW")
                        col3.metric("Peak Load", f"{max_load:,.0f} MW")
                    
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
                                line=dict(color='blue', width=2)
                            )
                        )
                    
                    if not forecast_df.empty:
                        # Parse timestamp for forecast data
                        forecast_time_cols = [col for col in forecast_df.columns if 'time' in col.lower() or 'date' in col.lower() or 'hour' in col.lower()]
                        if forecast_time_cols:
                            forecast_df['timestamp'] = pd.to_datetime(forecast_df[forecast_time_cols[0]])
                            forecast_df = forecast_df.sort_values('timestamp')
                        
                        x_axis_forecast = forecast_df['timestamp'] if 'timestamp' in forecast_df.columns else forecast_df.index
                        fig.add_trace(
                            go.Scatter(
                                x=x_axis_forecast,
                                y=forecast_df.get('SystemTotal', forecast_df.iloc[:, -1]),
                                mode='lines',
                                name='Forecast Load',
                                line=dict(color='orange', width=2, dash='dash')
                            )
                        )
                    
                    fig.update_layout(
                        title="ERCOT System Load: Actual vs Forecast",
                        xaxis_title="Time",
                        yaxis_title="Load (MW)",
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # ML Forecast Section
                    st.subheader("🤖 Machine Learning Load Forecast (Next 24 Hours)")
                    
                    if load_col and not actual_df.empty:
                        # Prepare data for ML
<<<<<<< HEAD
                        ml_df = pd.DataFrame({
                            'load': actual_df[load_col].values
                        }, index=range(len(actual_df)))
=======
                        if 'timestamp' in actual_df.columns:
                            ml_df = pd.DataFrame({
                                'load': actual_df[load_col].values
                            }, index=actual_df['timestamp'].values)
                        else:
                            ml_df = pd.DataFrame({
                                'load': actual_df[load_col].values
                            }, index=pd.date_range(end=pd.Timestamp.now(), periods=len(actual_df), freq='H'))
>>>>>>> f56b3538ceb2e0c951a1ebe05e926b73fb68cc72
                        
                        model, scaler, training_df = train_load_forecast_model(ml_df)
                        
                        if model is not None:
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
                            
                            # Plot ML forecast
                            fig_ml = go.Figure()
                            
                            fig_ml.add_trace(
                                go.Scatter(
                                    x=future_hours,
                                    y=future_load,
                                    mode='lines+markers',
                                    name='ML Forecast',
                                    line=dict(color='green', width=3)
                                )
                            )
                            
                            model_name = "XGBoost" if HAS_XGBOOST else "Random Forest"
                            fig_ml.update_layout(
                                title=f"{model_name} Load Forecast (Next 24 Hours)",
                                xaxis_title="Time",
                                yaxis_title="Predicted Load (MW)",
                                height=400
                            )
                            
                            st.plotly_chart(fig_ml, use_container_width=True)
                            
                            # Show forecast table
                            forecast_table = pd.DataFrame({
                                'Hour': [f"{h.hour}:00" for h in future_hours],
                                'Predicted Load (MW)': [f"{load:,.0f}" for load in future_load]
                            })
                            st.dataframe(forecast_table, use_container_width=True)
                        else:
                            st.warning("Not enough historical data to train ML model. Need at least 48 hours.")
                    
                else:
                    st.warning("No load data available for the selected period.")
                    
        except Exception as e:
            st.error(f"Error fetching load data: {e}")
    
    # TAB 2: RENEWABLE GENERATION
    with tab2:
        st.header("Renewable Energy Generation (Wind & Solar)")
        
        col1, col2 = st.columns(2)
        
        # Wind Generation
        with col1:
            st.subheader("🌬️ Wind Power Production")
            try:
                with st.spinner("Fetching wind data..."):
                    wind_params = {
                        "page": 1,
                        "size": 2000
                    }
                    wind_data = api.get_public("np4-732-cd/wpp_hrly_avrg_actl_fcast", params=wind_params)
                    
                    if "data" in wind_data and len(wind_data["data"]) > 0:
                        wind_df = pd.DataFrame(wind_data["data"])
                        
                        # Parse timestamp
                        wind_time_cols = [col for col in wind_df.columns if 'time' in col.lower() or 'date' in col.lower() or 'hour' in col.lower()]
                        if wind_time_cols:
                            wind_df['timestamp'] = pd.to_datetime(wind_df[wind_time_cols[0]])
                            wind_df = wind_df.sort_values('timestamp')
                        
                        if debug_mode:
                            st.write("**Wind Data Sample:**", wind_df.head())
                            st.write(f"Columns: {list(wind_df.columns)}")
                        
                        fig_wind = go.Figure()
                        x_axis_wind = wind_df['timestamp'] if 'timestamp' in wind_df.columns else wind_df.index
                        
<<<<<<< HEAD
                        # Add actual and forecast traces - API uses 'genSystemWide'
                        actual_col = None
                        for col in wind_df.columns:
                            if isinstance(col, str) and ('gensystemwide' in col.lower() or ('actual' in col.lower() and 'system' in col.lower())):
=======
                        # Add actual and forecast traces - API uses 'genSystemWide' not 'ACTUAL_SYSTEM_WIDE'
                        # Find actual column
                        actual_col = None
                        for col in wind_df.columns:
                            if isinstance(col, str) and ('gensystemwide' in col.lower() or 'actual' in col.lower() and 'system' in col.lower()):
>>>>>>> f56b3538ceb2e0c951a1ebe05e926b73fb68cc72
                                actual_col = col
                                break
                        
                        if actual_col:
                            fig_wind.add_trace(
                                go.Scatter(
                                    x=x_axis_wind,
                                    y=wind_df[actual_col],
                                    mode='lines',
                                    name='Actual Wind',
                                    line=dict(color='teal', width=2)
                                )
                            )
                        
<<<<<<< HEAD
=======
                        # Find forecast column
>>>>>>> f56b3538ceb2e0c951a1ebe05e926b73fb68cc72
                        forecast_col = None
                        for col in wind_df.columns:
                            if isinstance(col, str) and ('stwpf' in col.lower() and 'system' in col.lower()):
                                forecast_col = col
                                break
                        
                        if forecast_col:
                            fig_wind.add_trace(
                                go.Scatter(
                                    x=x_axis_wind,
                                    y=wind_df[forecast_col],
                                    mode='lines',
                                    name='Wind Forecast',
                                    line=dict(color='lightblue', width=2, dash='dash')
                                )
                            )
                        
                        fig_wind.update_layout(
                            title="Wind Generation (Actual vs Forecast)",
                            xaxis_title="Time",
                            yaxis_title="Generation (MW)",
                            height=400
                        )
                        
                        if len(fig_wind.data) > 0:
                            st.plotly_chart(fig_wind, use_container_width=True)
                        else:
                            st.warning("Wind data received but columns not found. Enable Debug Mode to see data structure.")
                    else:
                        st.info("No wind data available.")
            except Exception as e:
                st.error(f"Error fetching wind data: {e}")
        
        # Solar Generation
        with col2:
            st.subheader("☀️ Solar Power Production")
            try:
                with st.spinner("Fetching solar data..."):
                    solar_params = {
                        "page": 1,
                        "size": 2000
                    }
                    solar_data = api.get_public("np4-737-cd/spp_hrly_avrg_actl_fcast", params=solar_params)
                    
                    if "data" in solar_data and len(solar_data["data"]) > 0:
                        solar_df = pd.DataFrame(solar_data["data"])
                        
                        # Parse timestamp
                        solar_time_cols = [col for col in solar_df.columns if 'time' in col.lower() or 'date' in col.lower() or 'hour' in col.lower()]
                        if solar_time_cols:
                            solar_df['timestamp'] = pd.to_datetime(solar_df[solar_time_cols[0]])
                            solar_df = solar_df.sort_values('timestamp')
                        
                        if debug_mode:
                            st.write("**Solar Data Sample:**", solar_df.head())
                            st.write(f"Columns: {list(solar_df.columns)}")
                        
                        fig_solar = go.Figure()
                        x_axis_solar = solar_df['timestamp'] if 'timestamp' in solar_df.columns else solar_df.index
                        
                        # Find actual solar column - API uses 'genSystemWide'
                        actual_solar_col = None
                        for col in solar_df.columns:
<<<<<<< HEAD
                            if isinstance(col, str) and ('gensystemwide' in col.lower() or ('actual' in col.lower() and 'system' in col.lower())):
=======
                            if isinstance(col, str) and ('gensystemwide' in col.lower() or 'actual' in col.lower() and 'system' in col.lower()):
>>>>>>> f56b3538ceb2e0c951a1ebe05e926b73fb68cc72
                                actual_solar_col = col
                                break
                        
                        if actual_solar_col:
                            fig_solar.add_trace(
                                go.Scatter(
                                    x=x_axis_solar,
                                    y=solar_df[actual_solar_col],
                                    mode='lines',
                                    name='Actual Solar',
                                    line=dict(color='gold', width=2)
                                )
                            )
                        
<<<<<<< HEAD
=======
                        # Find solar forecast column
>>>>>>> f56b3538ceb2e0c951a1ebe05e926b73fb68cc72
                        forecast_solar_col = None
                        for col in solar_df.columns:
                            if isinstance(col, str) and ('stppf' in col.lower() and 'system' in col.lower()):
                                forecast_solar_col = col
                                break
                        
                        if forecast_solar_col:
                            fig_solar.add_trace(
                                go.Scatter(
                                    x=x_axis_solar,
                                    y=solar_df[forecast_solar_col],
                                    mode='lines',
                                    name='Solar Forecast',
                                    line=dict(color='orange', width=2, dash='dash')
                                )
                            )
                        
                        fig_solar.update_layout(
                            title="Solar Generation (Actual vs Forecast)",
                            xaxis_title="Time",
                            yaxis_title="Generation (MW)",
                            height=400
                        )
                        
                        if len(fig_solar.data) > 0:
                            st.plotly_chart(fig_solar, use_container_width=True)
                        else:
                            st.warning("Solar data received but columns not found. Enable Debug Mode to see data structure.")
                    else:
                        st.info("No solar data available.")
            except Exception as e:
                st.error(f"Error fetching solar data: {e}")
    
    # TAB 3: REAL-TIME PRICING
    with tab3:
        st.header("💰 Real-Time Market Pricing (LMPs)")
        
        try:
            with st.spinner("Fetching pricing data..."):
                lmp_params = {
                    "page": 1,
                    "size": 1000
                }
                lmp_data = api.get_public("np6-788-cd/lmp_node_zone_hub", params=lmp_params)
                
                if "data" in lmp_data and len(lmp_data["data"]) > 0:
                    lmp_df = pd.DataFrame(lmp_data["data"])
                    
                    # Filter for major hubs
                    if 'SettlementPoint' in lmp_df.columns:
                        hubs = lmp_df[lmp_df['SettlementPointType'] == 'HU'] if 'SettlementPointType' in lmp_df.columns else lmp_df
                        
                        if not hubs.empty and 'SettlementPointPrice' in hubs.columns:
                            fig_lmp = px.bar(
                                hubs.head(10),
                                x='SettlementPoint',
                                y='SettlementPointPrice',
                                title="Latest LMP Prices at Major Hubs",
                                labels={'SettlementPointPrice': 'Price ($/MWh)', 'SettlementPoint': 'Hub'},
                                color='SettlementPointPrice',
                                color_continuous_scale='RdYlGn_r'
                            )
                            
                            fig_lmp.update_layout(height=500)
                            st.plotly_chart(fig_lmp, use_container_width=True)
                            
                            # Show data table
                            st.dataframe(hubs[['SettlementPoint', 'SettlementPointPrice']].head(20), use_container_width=True)
                        else:
                            st.info("Price data columns not found.")
                    else:
                        st.dataframe(lmp_df.head(20), use_container_width=True)
                else:
                    st.warning("No pricing data available.")
        except Exception as e:
            st.error(f"Error fetching pricing data: {e}")
    
    # TAB 4: RESOURCE OUTAGES
    with tab4:
        st.header("⚠️ Resource Outages by Fuel Type")
        
        try:
            with st.spinner("Fetching outage data..."):
                outage_params = {
                    "page": 1,
                    "size": 2000
                }
                outage_data = api.get_public("np3-233-cd/hourly_res_outage_cap", params=outage_params)
                
                if "data" in outage_data and len(outage_data["data"]) > 0:
                    outage_df = pd.DataFrame(outage_data["data"])
                    
                    # Parse timestamp
                    outage_time_cols = [col for col in outage_df.columns if 'time' in col.lower() or 'date' in col.lower() or 'hour' in col.lower()]
                    if outage_time_cols:
                        outage_df['timestamp'] = pd.to_datetime(outage_df[outage_time_cols[0]])
                        outage_df = outage_df.sort_values('timestamp')
                    
                    if debug_mode:
                        st.write("**Outage Data Sample:**", outage_df.head())
                        st.write(f"Columns: {list(outage_df.columns)}")
                    
                    # Create stacked area chart for outages
                    numeric_cols = outage_df.select_dtypes(include=[np.number]).columns
                    
                    if len(numeric_cols) > 3:
                        fig_outage = go.Figure()
                        x_axis_outage = outage_df['timestamp'] if 'timestamp' in outage_df.columns else outage_df.index
                        
                        # Define proper colors
                        colors = ['rgba(255,99,132,0.6)', 'rgba(54,162,235,0.6)', 'rgba(255,206,86,0.6)', 
                                  'rgba(75,192,192,0.6)', 'rgba(153,102,255,0.6)', 'rgba(255,159,64,0.6)']
                        
                        for idx, col in enumerate(numeric_cols[2:min(8, len(numeric_cols))]):  # Show first few MW columns
                            fig_outage.add_trace(
                                go.Scatter(
                                    x=x_axis_outage,
                                    y=outage_df[col],
                                    mode='lines',
                                    name=str(col),
                                    stackgroup='one',
                                    fillcolor=colors[idx % len(colors)]
                                )
                            )
                        
                        fig_outage.update_layout(
                            title="Resource Outages by Category Over Time",
                            xaxis_title="Time",
                            yaxis_title="Outage Capacity (MW)",
                            height=500,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_outage, use_container_width=True)
                    
                    # Show summary statistics
                    st.subheader("📊 Outage Summary Statistics")
                    st.dataframe(outage_df.describe(), use_container_width=True)
                else:
                    st.warning("No outage data available.")
        except Exception as e:
            st.error(f"Error fetching outage data: {e}")


# Run the Streamlit dashboard
if __name__ == "__main__":
    main()