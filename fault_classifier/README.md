# Power Fault Classifier

Classifies six current/voltage measurements into four-bit fault labels in **G-C-B-A** order. The Streamlit app accepts CSV uploads, preserves encoded prediction columns, and evaluates optional ground truth with per-class precision, recall, F1, support, and an explicitly ordered confusion matrix.

The app opens on **Analyze measurements**: upload a CSV, or try 30 reproducibly
sampled observations from the bundled dataset. The sample may include training
rows and is explicitly a workflow demonstration, not an independent evaluation.
You can download a header-only CSV template, the sample measurements, and results
with source row numbers, input measurements, and the existing `Fault Code` and
`Fault String` columns. Optional ground truth adds `True Fault` and diagnostic
metrics. The **Model evaluation** tab holds the held-out evaluation, majority
baseline, per-class metrics, cross-validation and downloadable model report.
The **Input guide** explains units, validation rules, and label encoding.

## Run

Use Python 3.10+ in an isolated environment, then from the repository root:

```sh
python -m pip install -r fault_classifier/requirements.txt
streamlit run fault_classifier/fault_classifier_app.py
```

Without configuration, the app trains a fixed Random Forest (100 trees, seed 42,
one worker) from the bundled `classData.csv` using the **installed runtime**. It
uses the same stratified 80/20 train/test split and five training-only CV folds
as the CLI. CV is diagnostic for this fixed configuration; no model selection
is claimed. The held-out test is never fitted. The app displays the resulting
evaluation and caches the model in memory across uploads and reruns. Cache keys
include CSV contents, training implementation, and scikit-learn version.

The initial startup performs training with a visible progress message. A process
restart rebuilds the in-memory model; no pickle is read or written on the default
path. Source data and existing model files remain unchanged. Files resolve
relative to the project, so launching from another directory also works. Both
original `*_local.py` entry points delegate to the shared implementation.

### Saved-model version mismatch

The repository's historical model was saved with scikit-learn 1.6.1. Loading it
under newer releases can produce one warning per tree in its random forest.
Scikit-learn does not support cross-version model loading; see its
[model-persistence guidance](https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations).
The default app now avoids that mismatch by training in the serving runtime.

If `FAULT_CLASSIFIER_ARTIFACT_DIR` selects a saved bundle, the loader stops on the
first `InconsistentVersionWarning` or mismatched/missing version metadata. It
shows one actionable error and does not perform inference or switch models.
Rebuild the bundle in the same environment as the deployed app, or remove that
environment setting to use the built-in model. Keep all saved-model dependencies
aligned with the training environment. Existing legacy files are retained for
explicit, trusted use in matching environments, and are not used by the default UI.

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

`modeling.py` contains validation, fitting and metrics; `artifacts.py` builds the
runtime default and loads version-compatible bundles (or the older
model/scaler/encoder format for explicit legacy callers); the CLI handles
files/provenance; the Streamlit app handles presentation and caching. Model
selection never uses held-out test metrics. Repeatedly tuning against that same
test result would invalidate its independence.

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

Compatibility regressions cover default startup without deserialization, cache
reuse/invalidation, saved-estimator version warnings, version-metadata mismatches,
and a single actionable UI error without fallback for incompatible configured
bundles.

UI regressions additionally exercise the upload-first layout, bundled sample
provenance, input/output row alignment, unlabeled uploads, invalid measurements,
partial labels, and cache reuse while switching to the sample.

Next validation work should add event/time metadata and independently sourced labeled events before comparing field performance.
