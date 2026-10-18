# Hourly load forecasting

An inspectable AEP / PJM hourly-load forecasting example with a Streamlit app and a notebook. The app compares XGBoost with simple baselines, separates model selection, interval calibration, and test evaluation, and distinguishes one-hour-ahead predictions from 24/48-hour recursive forecasts. It supports optional archived forecast weather and independent expanding-history seasonal evaluations. All load values and errors use MW.

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
2. Reserve the first half of validation for model selection and its second half for interval calibration. Compare XGBoost depths 4 and 6 on selection **one-step MAE**, using 100 estimators, learning rate 0.1, histogram trees, one worker, and seed 42. The first candidate wins a tie. `ForecastConfig` exposes these settings.
3. Refit on training + selection rows, excluding calibration and test targets. Compute test MAE/RMSE against persistence, previous-day, and previous-week baselines on identical rows. Selection scores are not unbiased test estimates.
4. Run the frozen evaluation model recursively on calibration windows to estimate an absolute-error quantile for each lead through 48 hours. Then evaluate up to 30 complete, nonoverlapping windows **per horizon**, at 24 and 48 hours, sampled across the untouched test period. XGBoost feeds predictions back; persistence repeats the latest load, and seasonal baselines repeat the historical pattern. All methods use identical windows within each horizon.
5. Evaluate the last four complete eligible meteorological seasons independently. Each fold splits only its earlier history into 70% training, 15% selection, and 15% calibration, chooses its own depth, and freezes that model for the season. Report errors, interval coverage, sample counts, and width by season/horizon, with downloadable lead-level predictions. No configuration is selected using later seasons. December belongs to the following year's winter. Insufficient history or weather coverage is reported rather than filled.
6. Only after evaluation, fit a separate deployment model on all eligible observations for forecasting beyond the dataset. No weather archive is required for the load/calendar feature mode.

One-step evaluation assumes actual measurements arrive each hour. Recursive evaluation excludes actual observations throughout the forecast horizon. Models are not retrained between origins within a partition or seasonal fold. These scores measure different uses. Test metrics evaluate the model fitted before calibration, not the deployment model fitted on all history. Seasonal folds may overlap the main holdout in calendar time: keep these evaluations separate instead of pooling their errors.

## Archived forecast weather

No archived weather is bundled or downloaded. The optional upload expects an independently sourced **forecast-vintage archive**, with this CSV header:

```csv
valid_time,issued_at,available_at,record_type,temperature_c,source
```

- `valid_time`: the target hourly boundary, using the same normalized clock as load.
- `issued_at`: timestamp the forecast was issued, strictly before its target.
- `available_at`: when that vintage first became available to the forecasting system, at or after issuance. Do not substitute a present-day archive download time or backdate actual weather observations.
- `record_type`: exactly `forecast`; observed weather and reanalysis are rejected.
- `temperature_c`: finite predicted temperature in degrees Celsius, representing the modeled load region. Select or aggregate locations upstream; the app cannot verify geographic suitability.
- `source`: nonempty provider/product identifier. The file hash and listed sources are recorded.

For each target, the feature join selects the latest-issued vintage whose issue **and availability** timestamps are no later than the origin. A delayed older vintage cannot replace a newer available one. One-step origins are `t−1`; all 48 recursive steps use the **same** origin. Newer weather releases within a forecast window are excluded. Duplicate target/issue pairs, missing provenance, invalid timestamps, and nonfinite temperatures fail validation.

Missing eligible weather remains missing. Such training rows and recursive windows are excluded with counts in metadata. A deployment forecast fails clearly if the uploaded archive lacks weather for its entire horizon available at the final observed-load timestamp. Include both historical vintages and the vintage covering the desired future horizon in the upload. The app never falls back silently to observed weather or a load-only model. Without a weather upload it runs the load/calendar model.

All timestamps remain timezone-naive: normalize load and weather to the same unambiguous clock upstream. Schema checks cannot prove that provider timestamps or forecast labels are authentic. The archive's provenance remains a user-supplied assumption.

Programmatic use:

```python
from pathlib import Path
from energy_forcast.weather import read_weather_csv
from energy_forcast.forecasting import read_load_csv, run_experiment, forecast_with_intervals

load, report = read_load_csv(Path("load.csv").read_bytes())
weather = read_weather_csv(Path("weather_forecast_archive.csv").read_bytes())
experiment = run_experiment(load, report, weather=weather)
forecast = forecast_with_intervals(experiment, load, 48, weather)
```

## Empirical prediction intervals

For each lead, sort absolute calibration errors and select rank `ceil((n+1) × nominal_coverage)` (one-based). Default nominal coverage is 90%, with at least ten distinct calibration origins. If the rank exceeds the sample count or fewer than ten origins exist, that lead's bounds remain missing; the app reports this. Calibration uses nonoverlapping windows at the longest configured horizon, so short datasets can support point forecasts while lacking interval estimates.

Bounds are point forecast ± calibrated radius, with the lower bound clipped to zero MW. Held-out reports show empirical coverage, mean width, and supported interval-hour counts alongside MAE/RMSE. Calibration targets are excluded from fitting and depth selection, and test errors never set interval widths. Each seasonal fold has independent calibration preceding its test season.

These are empirical intervals. Serial dependence, nonstationarity, limited samples, and weather coverage gaps prevent distribution-free coverage claims. **Deployment refits on all history and transfers widths from the earlier evaluation model. Its coverage is not independently established.** The UI and metadata identify that distinction; held-out coverage assesses the frozen evaluation model only.

## Architecture and outputs

| File | Responsibility |
| --- | --- |
| `forecasting.py` | CSV validation, causal features, chronological splits, model selection, baselines, recursive backtest, forecast, experiment metadata |
| `weather.py` | Forecast-vintage schema/provenance checks and causal availability joins |
| `evaluation.py` | Lead-specific calibration, bounds, seasonal labels, coverage/error summaries |
| `streamlit_energy_forecast.py` | Upload controls, cached training, plots, evaluation explanations, and downloads |
| `load_forecasting.ipynb` | Exploratory plots and use of the same forecasting core |
| `tests/test_forecasting.py` | Deterministic input, causality, split, baseline, inference, and training-boundary regression tests |
| `tests/test_weather_evaluation.py` | Weather-availability boundaries, interval arithmetic, 48-hour forecasts, and seasonal-fit boundaries |

The Streamlit app caches fitted models by load/weather input content, duplicate policy, and fingerprints of all three core modules. Changing the display horizon reuses training results. Cache entries are bounded and are not persisted across process restarts. No model files or pickle inputs are loaded.

Downloadable artifacts:

- `experiment_metadata.json`: source filename/hash, module hashes, creation time, versions, preprocessing, features, seed/configuration, candidate scores, selected parameters, selection/calibration/test/deployment periods, calibration origins/quantiles, weather provenance/coverage, seasonal-fold settings, metrics, and requested forecast length.
- `test_predictions.csv`: actual load and all four methods' one-step predictions.
- `recursive_backtest.csv`: actual load, all four methods' predictions, forecast origin, horizon/lead time, and interval bounds.
- `seasonal_backtests.csv`: independent seasonal-fold predictions and bounds, with fold labels.
- `forecast.csv`: final model predictions and supported empirical bounds after the last input timestamp, when recent load history and optional weather coverage are complete.

Keep the metadata alongside CSV downloads to preserve context. Model outputs are estimates, not independently verified engineering conclusions. No source data, predictions, or models are transmitted to an external service.

## Verification

From the repository root after installing runtime dependencies:

```bash
python -m unittest discover -s energy_forcast/tests -v
python -m py_compile energy_forcast/forecasting.py energy_forcast/weather.py energy_forcast/evaluation.py energy_forcast/streamlit_energy_forecast.py
```

The regression suite independently checks known feature values, changes to current/future targets, missing hours and duplicates, strict split order, recursive feature parity, baseline patterns, input immutability, metric arithmetic, invalid outputs, and which timestamps each model fit may see. Additional tests cover forecast issue/availability boundaries, delayed older vintages, 48-hour weather cutoffs, exact interval quantiles, missing calibration, held-out coverage arithmetic, and independent seasonal selection/calibration boundaries. It uses lightweight deterministic predictors instead of expensive training.

A full-data Streamlit smoke check, including a horizon change, can be run without starting a web server:

```bash
python - <<'PY'
from streamlit.testing.v1 import AppTest
app = AppTest.from_file("energy_forcast/streamlit_energy_forecast.py", default_timeout=120).run()
assert not app.exception and not app.error
app.slider[0].set_value(48).run()
assert not app.exception and not app.error
PY
```

The full-data smoke validates execution and output availability, not future forecast quality. Runtime and model scores can vary with dependency versions; the metadata records the installed versions.

## Limitations

No archived weather is bundled, so the default run remains a load/calendar model. The optional feature is regional forecast temperature only; it has no humidity, wind, holiday calendar, live telemetry, automatic drift monitoring, or timezone reconstruction. Uploaded series are assumed to use consistent MW units and sampling meaning. Duplicate averaging may distort repeated clock labels.

Default evaluations sample at most 30 origins per horizon and four complete seasonal folds; they do not establish accuracy at every operating condition or guarantee improvement over a baseline. Depth selection optimizes one-step rather than multistep performance. Forecast/interval availability is conditional on complete recent load, weather coverage when enabled, and enough separate calibration windows. Missing bounds are not zero uncertainty.

Weather provenance and geographic suitability need independent checking. Availability filters prevent the code from selecting future-issued or future-available weather, but cannot detect incorrectly labeled observations in a submitted archive. Future performance and deployment-model interval coverage remain unverified.
