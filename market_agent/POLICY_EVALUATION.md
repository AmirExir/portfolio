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
- A 1-session dataset and a 30-session dataset must be evaluated and promoted
  separately. Passing the gates for one horizon does not authorize the other.
- The contextual allocator refreshes daily in both cases. “1” and “30” identify
  the calibrated forecast context and required purge, while realized policy
  P&L is measured on non-overlapping one-session execution returns.
- Daily portfolio returns must be non-overlapping. Do not pass overlapping
  30-session target returns into the Sharpe, drawdown, or CVaR calculation.

## Append-only ledger

Create a `PredictionRecord` at decision time, including:

- UTC creation time and data cutoff
- decision and target sessions
- model, policy, feature-set, and benchmark versions
- forecast, probability/interval when available, and target weight
- `return_start_session` when execution begins after the decision session

Append an `OutcomeRecord` only after the target session matures. The ledger
rejects unknown predictions, duplicate outcomes, and pre-maturity outcomes. Its
JSONL events form a SHA-256 hash chain, so a read fails if a prior event was
edited or reordered.

Maturity is timestamp-specific, not merely date-specific. Every prediction
stores `target_maturity_utc`; when no override is supplied, the ledger resolves
it to 4:00 p.m. `America/New_York` on `target_session` and converts it to UTC.
The target date must be exactly `horizon_sessions` valid US-equity sessions
after `return_start_session` (which defaults to `as_of_session`); a
weekday-only offset is rejected around exchange holidays. The prediction must
also be created before the next session opens,
and its `data_cutoff_utc` must satisfy the same bound. This prevents a caller
from observing any part of the outcome window and then backfilling a
prediction. Its outcome must:

- reference the immutable `prediction_id`;
- repeat the exact `target_session` and `target_maturity_utc`; and
- have `recorded_at_utc` on or after that maturity timestamp.

An outcome recorded earlier on the target date is therefore still rejected.
Use an explicitly verified maturity timestamp for shortened, extended, closed,
or non-US trading sessions rather than assuming the default regular close.

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

Inference is both frozen and keyed. The evaluator fingerprints the fitted model
immediately before and after prediction and rejects a callback that mutates its
state. Prediction output must contain unique `(as_of_session, symbol)` keys that
match the fold's test rows exactly. Results are joined one-to-one by those keys,
not by DataFrame row order, preventing a reorder from attaching a forecast to
the wrong symbol or session. The fitted policy version is retained with every
fold output.

The fit and prediction callbacks should use deterministic seeds and place model
parameters, preprocessing versions, and dataset hashes in
`FittedPolicy.metadata`.

Purging cannot repair revised or backfilled features. The input frame must be
built from the feature values and event publications available at each
`as_of_session`; point-in-time source snapshots and their data cutoffs belong in
the prediction ledger.

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

The target weight must be known and executable before its paired return period.
For a decision generated after the close, shift that weight to the next
session; pairing it with the just-completed session would be look-ahead leakage.
The ledger therefore stores a close-t decision with
`return_start_session=t+1` and evaluates the t+1 to t+2 close return.

For the contextual RL policy, ledger the two evidence roles separately:

- `RL Forecast Context` stores the calibrated 1- or 30-session non-RL forecast
  and its matured probability outcome.
- `RL Policy` stores the delayed one-session target-weight execution outcome.

Do not compute forecast MAE or Brier score from the execution placeholder.
Promotion calibration must filter the former model name and exact
horizon-specific policy version; portfolio return/drawdown evidence comes from
the latter execution path.

## Promotion gates

Promotion evidence is rejected before scoring unless its provenance agrees:

- `evaluation_id`, forecast horizon, candidate model/version, policy version,
  forecast cutoff, and baseline version must be explicit;
- forecast observations must be exact matured ledger matches for that
  candidate model, policy version, horizon, and cutoff;
- prediction IDs and forecast metrics must be derived by
  `build_ledger_backed_promotion_evidence`, not supplied independently;
- the authenticated ledger-head snapshot must match when gates are evaluated;
- candidate, baseline, and both cost-stress runs must cover identical sessions;
- each fold must use the same horizon and the declared candidate and baseline
  policy versions;
- fold sessions must be non-overlapping and exactly partition the full
  evaluation session set; and
- `shadow_sessions` must equal the candidate's evaluated session count.

The champion must be the strongest registered **non-RL** baseline. Both Ridge
and the fixed non-RL ensemble are mandatory candidates; supply any additional
eligible non-RL alternatives in `baseline_candidates`. The evaluator selects
strength by finite net Sharpe, then cumulative net return as the tie-breaker,
and rejects evidence naming a weaker baseline. Do not place an RL policy in
that baseline registry.

The doubled-cost evidence must be an exact replay of the ordinary candidate and
baseline paths. It requires exactly twice the configured basis-point cost, the
same sessions, gross returns, benchmark components, exposures, and turnover,
and exactly twice each period's transaction cost. A separately optimized or
otherwise changed stress path is not valid doubled-cost evidence.

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
`evaluate_promotion_gates` also requires the matching `PredictionLedger`;
omitting it or passing a different/tampered ledger fails provenance even if a
caller recomputes the public evaluation hash.

Run the complete process independently for 1-session and 30-session policies:
use separate horizon-filtered ledger observations, separate
`WalkForwardConfig` instances, separate policy versions, separate
`PromotionEvidence`, and separate promotion decisions. Never combine their
samples, folds, calibration statistics, or shadow-session counts.
