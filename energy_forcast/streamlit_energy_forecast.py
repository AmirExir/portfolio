
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

@st.cache_data
def load_data():
    df = pd.read_csv("AEP_hourly.csv", parse_dates=["Datetime"])
    df.columns = ["Datetime", "MW"]
    df = df.sort_values("Datetime")
    df.set_index("Datetime", inplace=True)
    df["MW"] = df["MW"].interpolate(method='time')
    return df

@st.cache_resource
def train_model(df):
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["lag_1"] = df["MW"].shift(1)
    df["lag_24"] = df["MW"].shift(24)
    df["lag_168"] = df["MW"].shift(168)
    df["rolling_24h_mean"] = df["MW"].rolling(24).mean()
    df["rolling_168h_mean"] = df["MW"].rolling(168).mean()
    df = df.dropna()

    train = df.loc[:"2017-12-31"]
    test = df.loc["2018-01-01":]
    features = [
        "hour", "dayofweek", "month", "is_weekend",
        "lag_1", "lag_24", "lag_168",
        "rolling_24h_mean", "rolling_168h_mean"
    ]
    X_train, y_train = train[features], train["MW"]
    X_test, y_test = test[features], test["MW"]

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return model, df, rmse

def forecast_n_hours(model, df, n_steps):
    last_row = df.iloc[-1].copy()
    future_preds, future_index = [], []

    for _ in range(n_steps):
        next_time = last_row.name + pd.Timedelta(hours=1)
        features_row = {
            "hour": next_time.hour,
            "dayofweek": next_time.dayofweek,
            "month": next_time.month,
            "is_weekend": int(next_time.dayofweek >= 5),
            "lag_1": last_row["MW"],
            "lag_24": df.loc[next_time - pd.Timedelta(hours=24), "MW"] if (next_time - pd.Timedelta(hours=24)) in df.index else last_row["MW"],
            "lag_168": df.loc[next_time - pd.Timedelta(hours=168), "MW"] if (next_time - pd.Timedelta(hours=168)) in df.index else last_row["MW"],
            "rolling_24h_mean": df["MW"].iloc[-24:].mean(),
            "rolling_168h_mean": df["MW"].iloc[-168:].mean(),
        }
        next_pred = model.predict(pd.DataFrame([features_row]))[0]
        future_preds.append(next_pred)
        future_index.append(next_time)
        last_row["MW"] = next_pred
        last_row.name = next_time
        df.loc[next_time] = last_row

    return pd.DataFrame({"Datetime": future_index, "Predicted_MW": future_preds})

# Streamlit UI
st.title("🔮 AEP Hourly Load Forecasting App")

df = load_data()
model, df, rmse = train_model(df)
st.success(f"✅ Model trained with RMSE: {rmse:.2f} MW")

n_steps = st.slider("Select forecast horizon (hours):", min_value=1, max_value=168, value=12, step=1)
forecast_df = forecast_n_hours(model, df.copy(), n_steps)

st.subheader("Forecast Plot")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["MW"].iloc[-48:], label="Historical", linewidth=2)
ax.plot(forecast_df["Datetime"], forecast_df["Predicted_MW"], label=f"{n_steps}h Forecast", linewidth=2, linestyle="--", marker="o")
ax.set_xlabel("Datetime")
ax.set_ylabel("MW")
ax.set_title(f"Forecast of Next {n_steps} Hours")
ax.legend()
ax.grid(True)
st.pyplot(fig)
