# agent/broker.py
import os
import requests
import streamlit as st

# Define BASE for consistent API base URL
BASE = st.secrets.get("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets")


def _headers():
    return {
        "APCA-API-KEY-ID": st.secrets["ALPACA_KEY"],
        "APCA-API-SECRET-KEY": st.secrets["ALPACA_SECRET"],
        "Content-Type": "application/json",
    }

def get_account():
    """Fetch account info safely."""
    try:
        endpoint = st.secrets.get("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets")
        r = requests.get(f"{endpoint}/v2/account", headers=_headers())
        if r.status_code == 200:
            return r.json()
        else:
            st.warning(f"⚠️ Alpaca API returned {r.status_code}: {r.text[:200]}")
            return {}
    except Exception as e:
        st.error(f"⚠️ Error fetching account: {e}")
        return {}

def submit_order(symbol, qty, side, type="market", tif="day", stop_price=None):
    """Send a buy or sell order to Alpaca. Cancels open orders for the same symbol before submitting."""
    cancel_open_orders(symbol)
    payload = {
        "symbol": symbol,
        "qty": qty,
        "side": side,            # ← this decides buy or sell
        "type": type,
        "time_in_force": tif,
    }
    if stop_price:
        payload["stop_price"] = str(stop_price)

    r = requests.post(f"{BASE}/v2/orders", headers=_headers(), json=payload)
    return r.json()

def cancel_open_orders(symbol=None):
    """Cancel any open orders (optionally filtered by symbol)."""
    r = requests.get(f"{BASE}/v2/orders", headers=_headers())
    orders = r.json()
    for o in orders:
        if (symbol is None) or (o["symbol"].upper() == symbol.upper()):
            requests.delete(f"{BASE}/v2/orders/{o['id']}", headers=_headers())