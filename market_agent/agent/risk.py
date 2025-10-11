# agent/risk.py
def target_position_qty(equity: float, price: float, risk_frac=0.1) -> int:
    # allocate 10% of equity to the position as a simple starter
    dollar_alloc = equity * risk_frac
    return max(int(dollar_alloc // price), 0)

def stop_loss(price_entry: float, pct=0.03) -> float:
    return round(price_entry * (1 - pct), 2)