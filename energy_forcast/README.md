# Hourly load forecasting

An inspectable AEP / PJM hourly-load forecasting example with a Streamlit app and a notebook. The app compares XGBoost with simple baselines, separates model selection from test evaluation, and distinguishes one-hour-ahead predictions from recursive forecasts. All load values and errors use MW.

The bundled CSV ends on **2018-08-03 at 00:00**. Forecasts start after the last supplied observation; the bundled example is not a live grid forecast.

## Run

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r energy_forcast/requirements.txt
python -m streamlit run energy_forcast/streamlit_energy_forecast.py
```

The default run uses `AEP_hourly.csv` beside the app. You can upload another CSV with `Datetime` and `MW` columns (`AEP_MW` is also accepted). Values must be nonnegative numeric loads or empty cells for missing measurements. Timestamps must be timezone-naive, unique hourly boundaries unless duplicate averaging is explicitly selected. Roughly eight continuous weeks are needed for the default split; every partition must contain at least 168 eligible hourly rows.

The notebook uses the same core and can run from the repository root or this directory in your existing Jupyter environment. Saved outputs were cleared because the previous notebook used target-leaking rolling features; rerun it to obtain valid results.

## Data handling and assumptions

The source CSV is never edited. The bundled example contains four extra rows with duplicate clock labels and 27 absent hourly slots. Its duplicate measurements are explicitly averaged, with a visible warning and processing report. For uploads, duplicates are rejected unless you opt into averaging. A duplicate group containing any missing measurement remains missing.

The loader sorts timestamps and creates an hourly grid. Absent hours remain `NaN`; it never interpolates or fills using future observations. Training/evaluation exclude missing targets and predictors that require missing history. Because the longest rolling window is one week, a missing observation makes the following 168 feature rows ineligible. Forecasting from the last observation requires a complete, measured preceding week.

These timestamps contain no timezone offsets. Averaging repeated labels and leaving missing labels unknown is an explicit demonstration policy, **not a reconstruction of daylight-saving transitions**. Resolve timezone/DST ambiguity upstream for operational use. Timestamp ranges above 1,000,000 hourly slots are rejected before allocating the regular grid.

## Features and validation

The target is hourly load `MW(t)`. Features are hour, weekday, month, weekend flag, load at `t−1`, `t−24`, and `t−168`, plus mean load over the preceding 24 and 168 hours. Rolling means are computed from `load.shift(1)`, excluding the target itself. The regular hourly grid makes lags represent elapsed clock hours instead of adjacent CSV rows. Recursive inference uses the same feature definitions.

1. Split eligible rows chronologically into 70% training, 15% validation, and 15% test. There is no random shuffling.
2. Compare XGBoost tree depths 4 and 6 on validation **one-step MAE**, using 100 estimators, learning rate 0.1, histogram trees, one worker, and random seed 42. The first candidate wins a tie. `ForecastConfig` exposes these settings for programmatic use.
3. Refit the selected configuration on training + validation. Compute test MAE and RMSE against persistence, previous-day, and previous-week baselines on identical rows. Validation scores were used for selection and are not unbiased test estimates.
4. Evaluate that same fitted model on up to 30 complete, nonoverlapping 24-hour windows sampled evenly across the test period. Every method sees only observations before each window. XGBoost feeds predictions back into its own subsequent inputs; persistence repeats the latest load, and seasonal baselines repeat the corresponding historical pattern.
5. Only after evaluation, fit a separate deployment model on all eligible observations for forecasting beyond the end of the supplied dataset.

One-step evaluation assumes actual measurements arrive each hour, so earlier holdout observations are available as history. Recursive evaluation excludes actual observations throughout each forecast horizon. The model is not retrained between test origins. These two scores measure different uses and should not be compared as though they were the same experiment. Test metrics evaluate the training + validation model, not the deployment model fitted on all history.

## Architecture and outputs

| File | Responsibility |
| --- | --- |
| `forecasting.py` | CSV validation, causal features, chronological splits, model selection, baselines, recursive backtest, forecast, experiment metadata |
| `streamlit_energy_forecast.py` | Upload controls, cached training, plots, evaluation explanations, and downloads |
| `load_forecasting.ipynb` | Exploratory plots and use of the same forecasting core |
| `tests/test_forecasting.py` | Deterministic input, causality, split, baseline, inference, and training-boundary regression tests |

The Streamlit app caches fitted models by input content, duplicate policy, and core source fingerprint. Changing the display horizon reuses training results. Cache entries are bounded and are not persisted across process restarts. No model files or pickle inputs are loaded.

Downloadable artifacts:

- `experiment_metadata.json`: source filename/hash, core source hash, creation time, software versions, preprocessing report, features, seed/configuration, candidate scores, selected parameters, train/validation/test and deployment-fit periods, evaluated origins, metrics, and requested forecast length.
- `test_predictions.csv`: actual load and all four methods' one-step predictions.
- `recursive_backtest.csv`: actual load, all four methods' predictions, forecast origin, and lead time in hours.
- `forecast.csv`: final model predictions after the last input timestamp, when the recent history is complete.

Keep the metadata alongside CSV downloads to preserve context. Model outputs are estimates, not independently verified engineering conclusions. No source data, predictions, or models are transmitted to an external service.

## Verification

From the repository root after installing runtime dependencies:

```bash
python -m unittest discover -s energy_forcast/tests -v
python -m py_compile energy_forcast/forecasting.py energy_forcast/streamlit_energy_forecast.py
```

The regression suite independently checks known feature values, changes to current/future targets, missing hours and duplicates, strict split order, recursive feature parity, baseline patterns, input immutability, metric arithmetic, invalid outputs, and which timestamps each model fit may see. It uses lightweight deterministic predictors instead of expensive training.

A full-data Streamlit smoke check, including a horizon change, can be run without starting a web server:

```bash
python - <<'PY'
from streamlit.testing.v1 import AppTest
app = AppTest.from_file("energy_forcast/streamlit_energy_forecast.py", default_timeout=30).run()
assert not app.exception and not app.error
app.slider[0].set_value(48).run()
assert not app.exception and not app.error
PY
```

The full-data smoke validates execution and output availability, not future forecast quality. Runtime and model scores can vary with dependency versions; the metadata records the installed versions.

## Limitations and next steps

The model uses load history and calendar fields only. It has no weather, holiday calendar, uncertainty intervals, live telemetry, automatic drift monitoring, or timezone reconstruction. Uploaded data are assumed to describe a single aggregate load series with consistent MW units and sampling meaning; the app cannot verify those engineering assumptions. Duplicate averaging may distort loads near repeated clock labels.

The default chronological holdout is one historical split. The recursive evaluation samples at most 30 origins and validates a 24-hour horizon, while the interactive forecast supports 1–48 hours. It does not establish accuracy at every horizon or operating condition, nor guarantee improvement over a baseline on other data. Depth selection optimizes one-step performance; it is not tuned specifically for multistep forecasts.

Useful follow-up work is timezone-aware source data, archived weather available at each forecast origin, seasonal walk-forward evaluations, and prediction intervals checked for empirical coverage. Each requires suitable data and further validation.
