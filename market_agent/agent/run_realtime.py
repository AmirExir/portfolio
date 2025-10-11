# agent/run_realtime.py
import os, time
from loguru import logger
from agent.data import get_ohlcv
from agent.strategy import sma_crossover
from agent.risk import target_position_qty, stop_loss
from agent.broker import get_account, submit_order

SYMBOL = os.getenv("SYMBOL", "SPY")

def main():
    acct = get_account()
    equity = float(acct.get("equity", "100000"))
    df = get_ohlcv(SYMBOL, 200)
    sig = sma_crossover(df, 20, 50)
    if len(sig.dropna()) == 0: 
        logger.info("Not enough data for signals yet.")
        return
    last_sig, prev_sig = sig.iloc[-1], sig.iloc[-2]

    if prev_sig == 0 and last_sig == 1:
        price = df["close"].iloc[-1]
        qty = target_position_qty(equity, price, 0.1)
        sl = stop_loss(price, 0.03)
        logger.info(f"BUY {SYMBOL} x{qty} @~{price} SL {sl}")
        print(submit_order(SYMBOL, qty, "buy"))
        # optional protective stop as separate stop order:
        # print(submit_order(SYMBOL, qty, "sell", type="stop", stop_price=sl))
    elif prev_sig == 1 and last_sig == 0:
        # flatten (sell market) — assumes only one open long
        logger.info(f"SELL {SYMBOL} (flatten)")
        print(submit_order(SYMBOL, 1, "sell"))  # replace 1 with current qty tracking

if __name__ == "__main__":
    main()