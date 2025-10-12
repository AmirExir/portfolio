# agent/broker.py
import os
import requests

ALPACA_KEY = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")
BASE = "https://paper-api.alpaca.markets"

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
    """Send a buy or sell order to Alpaca"""
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