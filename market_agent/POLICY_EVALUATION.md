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
- `as_of_session` and `target_session` follow the prediction's explicit
  `session_calendar`: regular US-equity sessions for `us_equity`, and labeled
  UTC calendar days for `utc_daily_24_7`.
- Fold construction must receive a complete, ordered exchange-session series.
- A 1-session dataset and a 30-session dataset must be evaluated and promoted
  separately. Passing the gates for one horizon does not authorize the other.
- The contextual allocator refreshes daily in both cases. “1” and “30” identify
  the calibrated forecast context and required purge, while realized policy
  P&L is measured on non-overlapping one-session execution returns.
- Daily portfolio returns must be non-overlapping. Do not pass overlapping
  30-session target returns into the Sharpe, drawdown, or CVaR calculation.

## Forecast qualification invariants

- A published reliability grade and a new allocation require
  `validation_is_oos` to be the literal boolean `true`. Missing, malformed, or
  false values fail closed.
- Validation must show strictly positive MAE skill over the zero-return
  baseline, direction skill over the training-only majority-direction
  baseline, and Brier skill over the training-only base-rate probability.
  Zero realized or predicted returns belong to the binary `not_up` class, the
  same definition used by the probability target.
- Live direct-model fits are cold fits, matching the outer-holdout evaluation
  regime. Weight artifacts remain research/cache infrastructure and cannot
  warm-start a live forecast until equivalent warm-start OOS validation is
  implemented.
- LSTM and Transformer forecasts are component-shadow diagnostics. They remain
  in prospective evaluation records but are excluded from the registered live
  ensemble and champion selection until an explicit, ledger-backed promotion.

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

Each completed run also records every available non-RL component forecast,
including the selected model, as a zero-allocation `component-shadow-v1`
prediction. A component marked ineligible for the live ensemble is still
recorded with its actual `live_eligible` flag. These records cannot place or
size an order. Their purpose is to build matched prospective Ridge, XGBoost,
MLP, LSTM, Transformer, and ensemble evidence so a later champion or component
promotion can be based on matured outcomes rather than repeated report
holdouts.

Maturity is timestamp-specific, not merely date-specific. Every prediction
stores `target_maturity_utc` and `session_calendar`. Schema-v2 records that do
not contain `session_calendar` retain the original `us_equity` behavior. A US
equity target matures at 4:00 p.m. `America/New_York` on `target_session`; a
`utc_daily_24_7` target labeled with date D matures at D+1 00:00 UTC. The target
date must be exactly `horizon_sessions` sessions after `return_start_session`
(which defaults to `as_of_session`) under that calendar. The US-equity path
rejects weekday-only offsets around exchange holidays.

A US-equity prediction must be created strictly before the next session opens.
A qualifying equity record uses the `prospective_equity` timing class and
retains the existing next-open evidence rule.
A 24/7 daily prediction must be created strictly before the first future UTC
daily bar completes, and its metadata records the publication lag from the
completed input bar. The named strict close-t tolerance is 30 minutes
(`UTC_DAILY_STRICT_CLOSE_T_MAX_LAG_SECONDS`). Crypto predictions published no
later than that inclusive boundary set `strict_close_t_eligible=true`; later
predictions remain visible as `delayed_research_only` but are excluded from
ledger-backed promotion evidence by default. Crypto records retain the raw
provider ticker and use `BTC-USD` as their benchmark so weekend outcomes have
aligned closes. In both calendars, `data_cutoff_utc` must satisfy the same
strict publication bound. Each outcome must:

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
- calendar, benchmark, model version, feature-set version, postprocessor
  version, and strict-versus-delayed timing are separate cohorts; optional
  exact filters are persisted in `PromotionEvidence`, and an omitted dimension
  fails closed when the remaining records contain mixed values;
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
