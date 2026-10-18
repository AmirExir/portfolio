"""Validated fault data and reproducible, leakage-safe model evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from typing import Any, Mapping

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

FEATURE_COLUMNS = ("Ia", "Ib", "Ic", "Va", "Vb", "Vc")
LABEL_COLUMNS = ("G", "C", "B", "A")


def validate_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return finite numeric features in model order without changing the source."""
    if data.empty:
        raise ValueError("The CSV must contain at least one data row.")
    if not data.columns.is_unique:
        raise ValueError("CSV column names must be unique.")
    missing = sorted(set(FEATURE_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}.")
    try:
        features = data.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("All current and voltage features must be numeric.") from exc
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Current and voltage features must not contain missing or infinite values.")
    return features


def fault_labels(data: pd.DataFrame, *, required: bool = True) -> pd.Series | None:
    """Build G-C-B-A strings from strictly binary flags, including numeric 0.0/1.0."""
    present = set(LABEL_COLUMNS) & set(data.columns)
    if not present and not required:
        return None
    missing = sorted(set(LABEL_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Provide all four ground-truth flags; missing: {', '.join(missing)}.")
    try:
        flags = data.loc[:, LABEL_COLUMNS].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("Ground-truth flags G, C, B, A must be binary 0 or 1.") from exc
    if not flags.isin([0, 1]).to_numpy().all():
        raise ValueError("Ground-truth flags G, C, B, A must be binary 0 or 1 without missing values.")
    return flags.astype(int).astype(str).agg("".join, axis=1).rename("fault_type")


def classification_metrics(
    truth: Any, predictions: Any, classes: list[str]
) -> dict[str, Any]:
    """Evaluate every supplied class with a consistently ordered confusion matrix."""
    actual = np.asarray(truth)
    predicted = np.asarray(predictions)
    if actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("Truth and predictions must have the same nonzero length.")
    if len(set(classes)) != len(classes) or not (set(actual) | set(predicted)) <= set(classes):
        raise ValueError("Metric classes must be unique and include every observed label.")
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=classes, average="macro", zero_division=0)),
        "per_class": classification_report(
            actual, predicted, labels=classes, output_dict=True, zero_division=0
        ),
        "classes": classes,
        "confusion_matrix": confusion_matrix(actual, predicted, labels=classes).tolist(),
        "samples": int(actual.size),
    }


def candidate_models(seed: int = 42, *, include_xgboost: bool = False) -> dict[str, BaseEstimator]:
    """Construct the portfolio's candidate models with reproducible random seeds."""
    models: dict[str, BaseEstimator] = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1),
        "SVM (RBF Kernel)": SVC(),
        "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=seed),
    }
    if include_xgboost:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(eval_metric="mlogloss", random_state=seed, n_jobs=1)
    return models


@dataclass
class TrainingResult:
    """An evaluated pipeline retained on its training partition and its provenance."""

    pipeline: Pipeline
    label_encoder: LabelEncoder
    report: dict[str, Any]

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Validate raw measurements and return four-bit G-C-B-A labels."""
        codes = self.pipeline.predict(validate_features(data))
        return self.label_encoder.inverse_transform(np.asarray(codes, dtype=int))


def train_and_evaluate(
    data: pd.DataFrame,
    *,
    models: Mapping[str, BaseEstimator] | None = None,
    seed: int = 42,
    test_fraction: float = 0.2,
    folds: int = 5,
    group_column: str | None = None,
    data_metadata: Mapping[str, Any] | None = None,
) -> TrainingResult:
    """Select by training-only stratified CV macro F1, then evaluate one held-out test.

    The final pipeline is fitted only on the training partition. An event_id column
    enables event-disjoint holdout and CV automatically; group_column overrides it.
    """
    features = validate_features(data)
    labels = fault_labels(data)
    metadata = validate_data_metadata(data_metadata) if data_metadata is not None else None
    group_column = group_column or ("event_id" if "event_id" in data else None)
    groups = event_groups(data, group_column) if group_column else None
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be strictly between 0 and 1.")
    if not isinstance(folds, int) or folds < 2:
        raise ValueError("folds must be an integer of at least 2.")
    counts = labels.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError("Training requires at least two classes and at least two rows per class.")
    if features.duplicated().any():
        raise ValueError("Duplicate feature rows found. Resolve duplicates or split by event before training.")
    try:
        if groups is None:
            train_rows, test_rows = train_test_split(
                np.arange(len(data)), test_size=test_fraction, random_state=seed, stratify=labels
            )
        else:
            train_rows, test_rows = next(GroupShuffleSplit(
                n_splits=1, test_size=test_fraction, random_state=seed
            ).split(features, labels, groups))
    except ValueError as exc:
        raise ValueError(f"Unable to form a stratified train/test split: {exc}") from exc
    train_labels = labels.iloc[train_rows]
    test_labels = labels.iloc[test_rows]
    if set(train_labels) != set(labels) or set(test_labels) != set(labels):
        raise ValueError("Every class must appear in both train and test partitions; adjust the split or add rows.")
    if train_labels.value_counts().min() < folds:
        raise ValueError(f"Each training class needs at least {folds} rows for stratified CV; reduce --folds or add rows.")

    encoder = LabelEncoder().fit(train_labels)
    encoded_train = encoder.transform(train_labels)
    x_train = features.iloc[train_rows]
    x_test = features.iloc[test_rows]
    candidates = dict(candidate_models(seed) if models is None else models)
    if not candidates:
        raise ValueError("Provide at least one candidate model.")
    if groups is None:
        cv = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(x_train, encoded_train))
    else:
        train_groups = groups.iloc[train_rows]
        if train_groups.nunique() < folds:
            raise ValueError(f"Need at least {folds} distinct training events for grouped CV.")
        cv = list(StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(
            x_train, encoded_train, train_groups
        ))
        for fit, valid in cv:
            if set(encoded_train[fit]) != set(encoded_train) or set(encoded_train[valid]) != set(encoded_train):
                raise ValueError("Each grouped CV partition must contain all classes; add independent events or reduce folds.")
    scores: dict[str, Any] = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", clone(estimator))])
        result = cross_validate(
            pipeline, x_train, encoded_train, cv=cv,
            scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
            error_score="raise", n_jobs=1,
        )
        scores[name] = {
            metric: {
                "mean": float(np.mean(result[f"test_{metric}"])),
                "std": float(np.std(result[f"test_{metric}"])),
                "folds": result[f"test_{metric}"].tolist(),
            }
            for metric in ("accuracy", "macro_f1")
        }
    selected_name = max(scores, key=lambda name: scores[name]["macro_f1"]["mean"])
    selected = Pipeline([("scaler", StandardScaler()), ("classifier", clone(candidates[selected_name]))])
    selected.fit(x_train, encoded_train)
    prediction = encoder.inverse_transform(selected.predict(x_test).astype(int))
    baseline = DummyClassifier(strategy="most_frequent").fit(x_train, train_labels)
    classes = encoder.classes_.tolist()
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "features": list(FEATURE_COLUMNS),
        "label_order": list(LABEL_COLUMNS),
        "classes": classes,
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "split": {
            "method": "event-disjoint random groups" if groups is not None else "stratified random rows",
            "test_fraction": test_fraction,
            "train_row_positions": train_rows.tolist(),
            "test_row_positions": test_rows.tolist(),
            "cv_folds": folds,
            "cv_partition": "training rows only",
            "time_periods": None,
            "group_column": group_column,
            "train_groups": sorted(set(groups.iloc[train_rows])) if groups is not None else None,
            "test_groups": sorted(set(groups.iloc[test_rows])) if groups is not None else None,
            "cv_row_positions": [{"train": train_rows[fit].tolist(), "validation": train_rows[valid].tolist()} for fit, valid in cv],
        },
        "preprocessing": "StandardScaler fitted independently inside each CV fold and on training rows for final fit",
        "selection_metric": "mean CV macro F1",
        "selected_model": selected_name,
        "candidate_parameters": {name: estimator.get_params(deep=True) for name, estimator in candidates.items()},
        "cross_validation": scores,
        "held_out_test": classification_metrics(test_labels, prediction, classes),
        "majority_baseline": {"class": str(baseline.predict(x_test.iloc[:1])[0]),
                              **classification_metrics(test_labels, baseline.predict(x_test), classes)},
        "saved_model_training_partition": "training rows only; held-out rows never fitted",
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "data_metadata": metadata,
        "development_measurement_sha256": sorted(set(measurement_hashes(data))) if groups is not None else None,
        "limitations": [
            ("Events are kept disjoint; event independence and field representativeness still require engineering review."
             if groups is not None else "Rows are assumed independent. Event IDs and timestamps are absent; correlated measurements may make row-level scores optimistic."),
            ("Units and acquisition context are operator-supplied metadata, not independently verified. No unit conversion is applied."
             if metadata else "Input current/voltage units and acquisition context are not documented by the source CSV; match training conventions without unit conversion."),
            "Dataset evaluation does not establish field protection performance or calibration.",
        ],
    }
    return TrainingResult(selected, encoder, report)


def event_groups(data: pd.DataFrame, column: str) -> pd.Series:
    """Validate explicit event identifiers without treating absent values as events."""
    if column not in data or data[column].isna().any():
        raise ValueError(f"Event column {column!r} must exist without missing values.")
    groups = data[column].astype(str).str.strip()
    if groups.eq("").any():
        raise ValueError("Event identifiers must not be empty.")
    return groups


def measurement_hashes(data: pd.DataFrame) -> list[str]:
    """Fingerprint canonical numeric measurements to detect exact development reuse."""
    values = validate_features(data).to_numpy(dtype="<f8", copy=True)
    values[values == 0] = 0  # Normalize signed zero; preserve all other numeric values.
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in values]


def validate_data_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Require a traceable measurement contract; never infer physical units."""
    if not isinstance(metadata, Mapping) or metadata.get("schema_version") != 1:
        raise ValueError("Data metadata must use schema_version 1.")
    for key in ("dataset_id", "source", "acquisition_context", "label_definition", "measurement_convention"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"Data metadata requires nonempty {key}.")
    units = metadata.get("units")
    if not isinstance(units, dict) or set(units) != set(FEATURE_COLUMNS) or any(
        not isinstance(value, str) or not value.strip() for value in units.values()
    ):
        raise ValueError("Metadata units must explicitly specify all six measurement columns.")
    return dict(metadata)


def evaluate_independent_events(
    result: TrainingResult, data: pd.DataFrame, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """Score a separate labeled event set without fitting or changing the model.

    This checks event-ID separation and declared measurement compatibility. It
    cannot establish truthful labeling, common event naming, or field validity.
    """
    declared = validate_data_metadata(metadata)
    trained = result.report.get("data_metadata")
    if not trained:
        raise ValueError("Independent evaluation requires training data metadata with known units.")
    for key in ("units", "measurement_convention", "label_definition"):
        if declared[key] != trained[key]:
            raise ValueError(f"External data {key} does not match the training contract; no conversion is inferred.")
    split = result.report["split"]
    if not split.get("group_column"):
        raise ValueError("Independent event evaluation requires an event-grouped training run.")
    groups = event_groups(data, split["group_column"])
    used = set(split["train_groups"]) | set(split["test_groups"])
    if set(groups) & used:
        raise ValueError("External events overlap the training/development dataset.")
    fingerprints = result.report.get("development_measurement_sha256")
    if not fingerprints:
        raise ValueError("Rebuild this training report to check external measurements against development records.")
    if set(measurement_hashes(data)) & set(fingerprints):
        raise ValueError("External measurements exactly repeat development records; renamed events do not establish independence.")
    labels = fault_labels(data)
    evaluated_classes = sorted(set(result.report["classes"]) | set(labels))
    return {
        "data_metadata": declared,
        "events": sorted(set(groups)),
        "metrics": classification_metrics(labels, result.predict(data), evaluated_classes),
        "majority_baseline": classification_metrics(labels, np.repeat(result.report["majority_baseline"]["class"], len(data)), evaluated_classes),
        "labels_absent_from_training": sorted(set(labels) - set(result.report["classes"])),
        "limitation": "Event-ID disjointness and declared units checked; source independence and field validity require review.",
    }
