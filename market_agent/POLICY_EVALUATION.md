# Market Policy Evaluation

The market agent now has an evaluation foundation that is independent of the
live forecasting and order paths:

- `agent/ledger.py` records point-in-time predictions and matured outcomes.
- `agent/evaluation.py` creates purged walk-forward folds, evaluates matured
  forecasts, calculates daily execution performance, and applies promotion
  gates.

These utilities do not promote or execute a policy. A policy remains in shadow
mode until an explicit caller supplies sufficient evidence and every promotion
gate passes.

## Units and session assumptions

- Returns and weights are decimal fractions. `0.05` means 5% and `0.25` means a
  25% target allocation.
- `as_of_session` and `target_session` are exchange-session dates, not elapsed
  calendar days.
- Fold construction must receive a complete, ordered exchange-session series.
- A 1-session dataset and a 30-session dataset must be evaluated separately.
- Daily portfolio returns must be non-overlapping. Do not pass overlapping
  30-session target returns into the Sharpe, drawdown, or CVaR calculation.

## Append-only ledger

Create a `PredictionRecord` at decision time, including:

- UTC creation time and data cutoff
- decision and target sessions
- model, policy, feature-set, and benchmark versions
- forecast, probability/interval when available, and target weight

Append an `OutcomeRecord` only after the target session matures. The ledger
rejects unknown predictions, duplicate outcomes, and pre-maturity outcomes. Its
JSONL events form a SHA-256 hash chain, so a read fails if a prior event was
edited or reordered.

Actual ledger files are generated evaluation data and should be stored outside
Git or in an ignored runtime-data directory.

## Purged walk-forward evaluation

`WalkForwardConfig` defaults both the purge and embargo to the forecast horizon
and rejects shorter values. `run_frozen_walk_forward`:

1. rejects mixed horizons and outcome columns presented as features;
2. gives the trainer only chronological training rows;
3. verifies all training labels mature before the test starts;
4. fits one versioned `FittedPolicy` per fold;
5. invokes inference once for the complete test block; and
6. skips the whole fold until all of its target dates have matured.

The fit and prediction callbacks should use deterministic seeds and place model
parameters, preprocessing versions, and dataset hashes in
`FittedPolicy.metadata`.

## Execution metrics

`evaluate_policy_returns` consumes daily target weights and daily asset and
benchmark returns. It applies transaction cost only to absolute target-weight
changes, including exits. It reports:

- net and benchmark-relative returns
- turnover and total transaction cost
- annualized return, volatility, and Sharpe
- maximum drawdown and lower-tail CVaR

It enforces long-only weights and a configurable gross-exposure ceiling.
Portfolio sector, name, cash, and correlation limits should be applied before
these target weights are evaluated.

## Promotion gates

`evaluate_promotion_gates` requires all of the following by default:

- at least 60 shadow sessions
- net Sharpe at least 0.25 above the champion baseline
- higher net return under at least doubled transaction costs
- no worse maximum drawdown or CVaR
- improvement in at least 70% of frozen walk-forward folds
- at least 60 probability observations
- expected calibration error no greater than 0.10
- Brier score no greater than 0.25

A failed or unavailable metric fails its gate. Promotion still requires an
explicit human-controlled change to the live policy configuration.
