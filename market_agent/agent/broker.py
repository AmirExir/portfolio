# agent/broker.py
import os
import requests
import streamlit as st

# Define consistent API base URL
BASE = st.secrets.get("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets").rstrip("/")

def _headers():
    return {
        "APCA-API-KEY-ID": st.secrets["ALPACA_KEY"],
        "APCA-API-SECRET-KEY": st.secrets["ALPACA_SECRET"],
        "Content-Type": "application/json",
    }

def get_account():
    """Fetch account info safely."""
    try:
        url = f"{BASE}/v2/account"
        r = requests.get(url, headers=_headers())
        if r.status_code == 200:
            return r.json()
        else:
            st.warning(f"⚠️ Alpaca API returned {r.status_code} for {url}: {r.text[:200]}")
            return {}
    except Exception as e:
        st.error(f"⚠️ Error fetching account from {BASE}: {e}")
        return {}

def submit_order(symbol, qty, side, type="market", tif="day", stop_price=None):
    """Send a buy or sell order to Alpaca. Cancels open orders for the same symbol before submitting."""
    cancel_open_orders(symbol)
    payload = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": type,
        "time_in_force": tif,
    }
    if stop_price:
        payload["stop_price"] = str(stop_price)

    try:
        url = f"{BASE}/v2/orders"
        r = requests.post(url, headers=_headers(), json=payload)
        if r.status_code == 200 or r.status_code == 201:
            return r.json()
        else:
            st.warning(f"⚠️ Order rejected ({r.status_code}): {r.text[:200]}")
            return {"error": r.text}
    except Exception as e:
        st.error(f"⚠️ Error submitting order: {e}")
        return {"error": str(e)}

def cancel_open_orders(symbol=None):
    """Cancel any open orders (optionally filtered by symbol)."""
    try:
        url = f"{BASE}/v2/orders"
        r = requests.get(url, headers=_headers())
        if r.status_code != 200:
            st.warning(f"⚠️ Failed to fetch open orders: {r.status_code} - {r.text[:200]}")
            return
        orders = r.json()
        for o in orders:
            if (symbol is None) or (o["symbol"].upper() == symbol.upper()):
                requests.delete(f"{BASE}/v2/orders/{o['id']}", headers=_headers())
    except Exception as e:
        st.error(f"⚠️ Error cancelling orders: {e}")