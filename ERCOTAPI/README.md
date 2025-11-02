# ERCOT Grid Analytics Dashboard

A comprehensive Streamlit dashboard for monitoring ERCOT grid operations, featuring:
- Real-time load analysis
- ML-powered load forecasting (Random Forest)
- Renewable generation tracking (Wind & Solar)
- Real-time LMP pricing
- Resource outage monitoring

## 🔐 Setup Credentials (Choose One Method)

### Method 1: Streamlit Cloud Secrets (RECOMMENDED for Production)

**Perfect for deployed apps - credentials are encrypted and never in your code!**

1. Deploy to Streamlit Cloud: https://share.streamlit.io/
2. Go to App Settings → **Secrets**
3. Paste your credentials:
   ```toml
   ERCOT_USERNAME = "your_email@example.com"
   ERCOT_PASSWORD = "your_password"
   ERCOT_CLIENT_ID = "fec253ea-0d06-4272-a5e6-b478baeecd70"
   ERCOT_SUBSCRIPTION_KEY = "your_subscription_key"
   ```
4. Click "Save" - Done! 🎉

**See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions.**

### Method 2: Local Secrets File (For Local Development)

1. Create the secrets file:
   ```bash
   mkdir -p .streamlit
   nano .streamlit/secrets.toml
   ```

2. Add your credentials (same format as above)

3. Save and restart Streamlit

### Method 2: Environment Variables

**macOS/Linux:**
```bash
export ERCOT_USERNAME="your_email@example.com"
export ERCOT_PASSWORD="your_password"
export ERCOT_CLIENT_ID="fec253ea-0d06-4272-a5e6-b478baeecd70"
export ERCOT_SUBSCRIPTION_KEY="your_subscription_key"
```

**Windows (PowerShell):**
```powershell
$env:ERCOT_USERNAME="your_email@example.com"
$env:ERCOT_PASSWORD="your_password"
$env:ERCOT_CLIENT_ID="fec253ea-0d06-4272-a5e6-b478baeecd70"
$env:ERCOT_SUBSCRIPTION_KEY="your_subscription_key"
```

**Permanent (add to ~/.zshrc or ~/.bashrc):**
```bash
echo 'export ERCOT_USERNAME="your_email@example.com"' >> ~/.zshrc
echo 'export ERCOT_PASSWORD="your_password"' >> ~/.zshrc
echo 'export ERCOT_CLIENT_ID="fec253ea-0d06-4272-a5e6-b478baeecd70"' >> ~/.zshrc
echo 'export ERCOT_SUBSCRIPTION_KEY="your_subscription_key"' >> ~/.zshrc
source ~/.zshrc
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Set the following environment variables:

```bash
export ERCOT_USERNAME="your_username"
export ERCOT_PASSWORD="your_password"
export ERCOT_CLIENT_ID="your_client_id"
export ERCOT_PUBLIC_KEY="your_api_key"  # Optional
```

## Usage

```bash
streamlit run ercotapi.py
```

Then open your browser to `http://localhost:8501`

## API Endpoints Used

- **np6-345-cd/act_sys_load_by_wzn**: Actual system load by weather zone
- **np3-565-cd/lf_by_model_weather_zone**: 7-day load forecast
- **np4-732-cd/wpp_hrly_avrg_actl_fcast**: Wind power production (actual & forecast)
- **np4-737-cd/spp_hrly_avrg_actl_fcast**: Solar power production (actual & forecast)
- **np6-788-cd/lmp_node_zone_hub**: Real-time LMPs at hubs and nodes
- **np3-233-cd/hourly_res_outage_cap**: Hourly resource outage capacity

## Machine Learning Model

The dashboard includes a **Random Forest Regressor** that:
- Trains on historical load data (minimum 48 hours required)
- Uses features: hour of day, day of week, 24-hour rolling mean/std
- Generates 24-hour ahead load forecasts
- Updates predictions as new data arrives

## Dashboard Controls

- **Historical Days Slider**: Select 1-7 days of historical data to load
- **Tabs**: Navigate between Load Analysis, Renewables, Pricing, and Outages
- **Interactive Charts**: Hover for details, zoom, pan, and download plots

## Data Refresh

- Real-time data updates based on ERCOT's publication schedule
- Load data: hourly
- Wind/Solar: hourly averages
- LMPs: every 5 minutes (SCED interval)
- Outages: hourly

## Performance Tips

- Start with 1-2 days of historical data for faster initial load
- ML model trains automatically when sufficient data is available
- Large date ranges may take longer to fetch and process

## Future Enhancements

- [ ] Add DAM (Day-Ahead Market) price forecasting
- [ ] Implement anomaly detection for outages
- [ ] Add ancillary services analysis
- [ ] Include weather correlation analysis
- [ ] Deploy alerts for high price events

## License

MIT License - See LICENSE file for details

## Author

Amir Exir, P.E., NCSO  
Power Systems Engineer | AI Researcher  
[amirexirpe.com](https://amirexirpe.com)
