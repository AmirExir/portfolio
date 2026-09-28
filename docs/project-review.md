# Portfolio project review — September 2026

## Scope and selection

This was a source-code review of the portfolio site, its linked ML projects,
the ERCOT dashboard/central retrieval architecture, and the existing market
evaluation documentation. It was not an audit of every historical script,
hosted deployment, regulatory corpus, or saved model. No live trading, messaging,
paid API requests, deployment, or source-data replacement was performed.

Four projects offered especially useful improvements because their evaluation
or prediction paths contained concrete defects. Priority followed engineering
correctness, reproducibility, and usefulness to a reviewer.

| Project | Original problem | Implemented improvement |
| --- | --- | --- |
| Hourly load forecast | Rolling means included the target hour; interpolation could use future observations; positional lags crossed missing/duplicate hours; a single RMSE obscured the prediction horizon | Validated hourly data with explicit duplicate handling; causal features; chronological training/validation/test; same-row baselines and recursive backtesting; cached experiments and downloadable evidence |
| Power fault classifier | The scaler saw every row before splitting; model selection reused the test population; accuracy hid class-specific behavior; float labels and subset confusion matrices were handled incorrectly | Scaling inside stratified CV pipelines; untouched held-out test; macro F1/per-class metrics and majority baseline; strict input validation; consistent class ordering; new reproducible run artifacts |
| Power grid GNN | A builder applied physical voltage thresholds to already-preprocessed values; predictor inputs included target-derived measurements; preprocessing preceded splitting | Preserve supplied labels; exclude target-derived predictors; validate scenario graphs; split by scenario; fit new preprocessing on training graphs; evaluate the selected checkpoint on held-out scenarios |
| ERCOT dashboard forecast | Rolling statistics included the target; scaling preceded CV; recursive forecasting reused one previous-day value and frozen daily windows; absent timestamps were invented | A separate tested forecasting module; fold-local scaling; chronological holdout and persistence/day baselines; advancing recursive features; strict timestamp/data checks; explicit one-hour versus 24-hour evaluation scope |

## Architecture and compatibility

The changes retain project directories and application launch paths. Small
domain modules now own data validation, feature construction, and evaluation;
Streamlit code handles interaction and presentation. This separation is needed
to regression-test the calculations without starting a UI or contacting a
service. Existing source CSVs, saved model binaries, and screenshots are not
replaced by synthetic examples or newly claimed results.

The ERCOT forecast's six-value training return interface remains available via
`ERCOTAPI.ercotapi`. Its returned model is refitted on all eligible observations
after holdout metrics have been computed; the downloaded metadata distinguishes
those stages. Insufficient history returns no model. Invalid data now produces
an explicit error instead of silently inventing or filling study inputs.

The GNN changes affect model input dimensions and topology representation, so
old trained weights/graph files are not interchangeable with new runs. Active
branches are represented in both message-passing directions. Explicitly offline
equipment is retained in the source but excluded from the active graph under
the documented policy; physical source topology and identifiers are not edited.
See the [GNN guide](../GNN/README.md) for exact feature and label assumptions.

## Remaining limitations

- **Hourly forecast:** AEP/PJM source timestamps lack offsets. Explicit duplicate
  averaging is a data-preparation choice, not a reconstruction of daylight-saving
  history. Performance must be read from a reproducible run; correcting leakage
  may lower previously displayed scores.
- **Fault classifier:** Row-level stratification cannot establish independence
  across faults/events without event identifiers. Existing model files and old
  accuracy images were not retrained or revalidated by this change. Only load
  trusted model artifacts.
- **GNN:** The bundled CSV features were already preprocessed and their original
  transformations and class thresholds are unavailable. New train-only scaling
  cannot undo unknown upstream preprocessing. Supplied labels are preserved,
  not certified as physical violation criteria. Thermal inputs lack operating
  dispatch/load information, limiting what the model can infer.
- **ERCOT forecast:** The selectable short history is an exploratory sample.
  One-hour holdout metrics are not 24-hour forecast metrics. Missing weather,
  independent seasonal evaluation, and calibrated uncertainty remain open work.
- **Other projects:** The central ERCOT retrieval and market evaluation paths
  already have dedicated modules and tests. They remain candidates for separate
  end-to-end evidence reviews, especially corpus effectiveness and prospective
  model performance. AELab's screenshots/PDF are not enough to audit its complete
  engineering implementation here.

## Reproduce the checks

Use the relevant project's environment rather than installing all dependency
sets together. From the repository root:

```sh
python -m unittest discover -s energy_forcast/tests -v
python -m unittest discover -s fault_classifier/tests -v
python -m unittest discover -s GNN/tests -v
python -m unittest ERCOTAPI.tests.test_load_forecast -v
python -m unittest ERCOTAPI.tests.test_ercot_api_client ERCOTAPI.tests.test_dashboard_styles -v
python -m unittest tests.test_portfolio_site -v
```

These checks verify data/feature contracts, split isolation, deterministic
calculations, prediction integration, and local site addresses. They do not
certify out-of-sample performance on a new operating region, production event
stream, physical study, or live hosted deployment. Each project README includes
its own runnable experiment and narrower validation instructions.

## Validation performed

The combined run passed **68 tests** in the existing Python 3.13 environment:
18 hourly-forecast tests, 14 fault-classifier tests, 15 GNN tests (including two
Streamlit application tests), nine ERCOT forecasting tests, five existing ERCOT
API/style regressions, and seven portfolio-address tests.

Additional execution checks passed: the full AEP/PJM experiment and 48-hour
forecast, forecasting-app startup and horizon change, every forecasting notebook
code cell, one-epoch voltage/thermal GNN CLI runs, a bundled-data fault-classifier
training/prediction run in a temporary directory, and an XGBoost dashboard
forecast on deterministic synthetic hourly data. These are execution checks,
not claims of improved predictive accuracy. Syntax and whitespace checks passed.
No original CSV, saved model, media, or secret file was changed.
