# agent/broker.py (Alpaca paper)
import os, requests, time
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
    return requests.get(f"{BASE}/v2/account", headers=_headers()).json()

def submit_order(symbol, qty, side, type="market", tif="day", stop_price=None):
    payload = {"symbol": symbol, "qty": qty, "side": side, "type": type, "time_in_force": tif}
    if stop_price: payload["stop_price"] = str(stop_price)
    r = requests.post(f"{BASE}/v2/orders", headers=_headers(), json=payload)
    return r.json()