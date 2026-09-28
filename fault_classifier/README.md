# Power-system fault classifier

Classifies six current/voltage measurements into four-bit fault labels in **G-C-B-A** order. The Streamlit app accepts CSV uploads, preserves encoded prediction columns, and evaluates optional ground truth with per-class precision, recall, F1, support, and an explicitly ordered confusion matrix.

## Run

Use Python 3.10+ in an isolated environment, then from the repository root:

```sh
python -m pip install -r fault_classifier/requirements.txt
streamlit run fault_classifier/fault_classifier_app.py
```

Without configuration, the app uses the existing repository artifacts and displays a warning about their unverified independent performance. Artifacts resolve relative to the project, so launching from another directory also works. Both original `*_local.py` entry points delegate to the shared implementation.

## Train and evaluate a new run

```sh
python fault_classifier/fault_classification_v2.py \
  --output-dir fault_classifier/artifacts/run-001 \
  --models "Logistic Regression" "Random Forest" \
  --predict fault_classifier/detect_dataset.csv

FAULT_CLASSIFIER_ARTIFACT_DIR="$PWD/fault_classifier/artifacts/run-001" \
  streamlit run fault_classifier/fault_classifier_app.py
```

`--data` defaults to the included `classData.csv`. The output directory must be new: existing runs, source data, and committed models are never overwritten. Omitting `--models` compares Logistic Regression, Random Forest, SVM, and MLP. `--include-xgboost` adds the optional XGBoost candidate. `--folds`, `--test-fraction`, and `--seed` default to 5, 0.2, and 42.

The workflow:

1. Validates finite numeric features and strictly binary ground-truth flags. Numeric `0.0`/`1.0` flags normalize to four-bit strings. Partial labels, missing values, nonbinary flags, and insufficient class support are rejected.
2. Makes one stratified row-level train/test split. Duplicate feature rows are rejected for explicit resolution, because copying observations across partitions would leak data.
3. Fits a `StandardScaler` inside each training-only stratified CV fold, selects the candidate with the highest mean macro F1, and fits that pipeline on the training partition.
4. Evaluates the selected model and a training-majority baseline once on held-out test rows. The saved model is **not refitted on test rows**, so its held-out evaluation remains applicable.

Outputs include `model_bundle.joblib`, `evaluation.json`, and optional `predictions.csv`. The JSON records dataset SHA-256, source path, execution timestamp, source commit and dirty status, feature/label order, candidate parameters, versions, seed, train/test row positions, CV scores, class metrics, baseline, and assumptions. Cross-validation fold assignment is reproducible from the recorded train-row order, seed, fold count, and library version. A tie in CV macro F1 selects the first candidate in configured order. Warnings such as nonconvergence remain visible.

`modeling.py` contains validation, fitting and metrics; `artifacts.py` adapts new bundles and the older model/scaler/encoder format; the CLI handles files/provenance; the Streamlit app handles presentation. Model selection never uses held-out test metrics. Repeatedly tuning against that same test result would invalidate its independence.

## Input assumptions and limitations

- Features are `Ia`, `Ib`, `Ic`, `Va`, `Vb`, `Vc`. Labels are `G`, `C`, `B`, `A`; all four labels or none are required for uploads. Extra columns are ignored for prediction.
- Units, measurement acquisition context, event IDs, and timestamps are not documented in these CSVs. The tool performs no unit conversion and makes no physical protection-performance claim.
- Row splitting assumes independent observations. Correlation among samples from the same simulated or physical event could still inflate scores. Event-grouped or chronological validation needs metadata that the source dataset does not supply.
- The binary `Output (S)` column in `detect_dataset.csv` is not treated as four-bit multiclass ground truth.
- Uploaded-data evaluation is diagnostic unless the operator establishes independence from training data. True labels absent from the model remain in the confusion matrix and metrics and are explicitly flagged.
- Macro F1 in displayed metrics averages all listed classes; unsupported/absent classes contribute zero. Always inspect per-class support.
- Existing historical JSON accuracies and plots came from globally scaled data and are retained as historical artifacts, not presented as valid held-out performance.
- Joblib/pickle artifacts can execute code. Load only repository artifacts you trust or runs you trained in a trusted environment. The app accepts CSV uploads, never model uploads. `FAULT_CLASSIFIER_ARTIFACT_DIR` is an operator-controlled local path; a missing configured bundle fails clearly rather than falling back to a legacy model. Keep runtime versions aligned with the recorded versions.

## Validation

```sh
python -m unittest discover -s fault_classifier/tests -v
python -m compileall -q fault_classifier
```

Tests use deterministic synthetic observations, prove that every scaler fit excludes held-out rows and that CV fits exclude validation rows, exercise float-label and confusion-matrix regressions, verify invalid-input handling, check reproducibility, run the CLI from a separate directory, confirm no-overwrite behavior, and smoke-test the app with a freshly trained temporary model. Committed pickle artifacts are never loaded by the tests.

Next validation work should add event/time metadata and independently sourced labeled events before comparing field performance.
