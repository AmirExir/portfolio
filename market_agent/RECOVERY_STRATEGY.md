# Oversold recovery study

An oversold reading can coexist with a new closing low. The daily report shows how two paper entry rules respond to that combination. They each track a separate hypothetical position. A 25% starter uses a quarter of the budget for one symbol. The existing portfolio sizing, forecast qualification and execution policy still govern actual decisions.

## Rules evaluated after each completed session

| Condition | Paper position for the next open |
| --- | --- |
| Close below the previous 20-session closing low, or a one-day loss at or above the shock threshold | Zero, including when RSI is oversold. The threshold is the greater of 2% or twice the sample standard deviation of the previous 20 daily percentage returns. |
| Close above the 20-session simple moving average, with that average above the 50-session average | Full research position. Hold while the close and 20-session average both exceed the 50-session average, subject to the risk rule. |
| Early entry outside that trend | Take a 25% starter on a close with 14-session RSI below 35 once both risk warnings are clear. |
| Confirmed entry outside that trend | After an oversold close within the preceding 20 sessions, take a 25% starter when RSI reaches 35 or higher and the close exceeds each of the previous five closes. Both risk warnings must be clear. |
| An existing starter | Promote when the trend qualifies. Otherwise exit on a risk warning, a close below the starter's 20-session closing floor, or 30 sessions after entry. |

Each strategy evaluates the risk condition first and the trend condition second. A dropped trend position can qualify for the early starter at the same close. A renewed oversold reading can open another 20-session confirmation window. The closing-price hurdle rolls each session. RSI uses simple averages of gains and losses over 14 changes. The chart assigns 100 to an all-gain window, 0 to an all-loss window, and 50 to a completely flat window. Above 70 is labeled overbought. The strategy can hold an overbought uptrend.

For NVDA on November 20, 2026, RSI was 13.75 and the close was $125.53. The 14.01% loss exceeded the 6.28% shock threshold and the close broke $145.98. Both paper positions were zero. For the next session, the five-close hurdle was $156.14. A Monday close at $156.15 would produce RSI 30.79. The other 13 price changes imply a closing price of at least $164.68 on a one-cent grid for RSI to reach 35. Both hurdles recalculate after each session.

## Historical comparison

Each cell shows net annualized return followed by maximum peak-to-trough loss, measured at session opens. All instruments share equal initial budgets, with targets restored at subsequent opens. The 50% invested comparison holds the same mix with the remainder in cash.

| Period | Fully invested | 50% invested | Trend only | Early starter | Confirmed starter |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2017–2020 | 20.26% / 36.76% | 10.36% / 20.02% | 12.01% / 7.13% | 13.16% / 8.96% | 13.71% / 7.20% |
| 2021–2023 | 21.18% / 30.00% | 10.71% / 15.94% | 10.11% / 10.34% | 10.01% / 13.82% | 10.08% / 12.25% |
| 2024–November 20, 2026 | 30.44% / 22.10% | 14.96% / 11.56% | 15.49% / 11.32% | 17.12% / 11.93% | 16.58% / 13.26% |

In the last period, mean exposure was 51.84% for early entry and 51.20% for confirmed entry. Doubling one-way costs from 5 to 10 basis points reduced their annualized returns to 16.03% and 15.60%. For NVDA alone, the early rule returned 26.52% annualized with a 25.79% drawdown. The confirmed rule returned 21.55% with a 24.09% drawdown. Holding NVDA returned 45.16% with a 42.84% drawdown.

All three periods are retrospective. The 16 surviving instruments were selected from the current watchlist and several have overlapping exposures. This favors companies still in the universe. Lower average exposure accounts for part of the drawdown reduction. Fully investing produced the highest return in each period. The preferred starter depended on the period and the investor's objective. The study makes no estimate of the probability that any particular close is the bottom.

## Reproduce

From the repository root with the market-agent dependencies installed, run:

```sh
python -m market_agent.evaluate_recovery --start 2017-01-01 --end 2020-12-31
python -m market_agent.evaluate_recovery --start 2021-01-01 --end 2023-12-31
python -m market_agent.evaluate_recovery --start 2024-01-01 --end 2026-11-20
python -m market_agent.evaluate_recovery --start 2024-01-01 --end 2026-11-20 --cost-bps 10
python -m market_agent.evaluate_recovery --symbols NVDA --start 2024-01-01 --end 2026-11-20
```

Data were Yahoo Finance adjusted daily opens and closes for NVDA, AMD, MU, MSFT, META, INTC, AVGO, SPY, QQQ, XLK, GLD, SLV, TLT, USO, OXY and SLB. For the recorded run, all 16 had 2,738 aligned prices from January 4, 2016 through November 20, 2026. To use a saved data snapshot, pass `--prices-dir DIRECTORY` with one `SYMBOL.csv` per instrument and columns `date,open,close`. Adjustments and vendor corrections can change a later replay.

Each period begins with cash and a signal computed on previous closes. Trades occur at the next recorded open and incur 5 basis points per dollar traded, including initial entry and rebalancing. The last valuation is the last open in the period with positions marked to market. Intraday drawdowns, taxes, cash interest and market impact beyond the assumed trading cost are outside this study. The study trains no model and uses no future forecasts. It compares price rules in isolation. Forward paper outcomes can be compared with the three historical windows before changing execution policy.
