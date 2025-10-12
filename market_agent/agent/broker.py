import os
import requests

# Load credentials from environment
ALPACA_KEY = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")
BASE = os.getenv("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets")

def _headers():
    return {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
        "Content-Type": "application/json",
    }

def get_account():
    """Fetch account info"""
    r = requests.get(f"{BASE}/v2/account", headers=_headers())
    return r.json()

def submit_order(symbol, qty, side, type="market", tif="day", stop_price=None):
    """Send order to Alpaca (paper or live depending on BASE)"""
    payload = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": type,
        "time_in_force": tif,
    }
    if stop_price:
        payload["stop_price"] = str(stop_price)

    r = requests.post(f"{BASE}/v2/orders", headers=_headers(), json=payload)
    return r.json()