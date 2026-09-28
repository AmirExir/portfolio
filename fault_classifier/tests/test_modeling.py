"""Regression tests use synthetic data and never load committed pickle files."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from fault_classifier import modeling
from fault_classifier.artifacts import PROJECT_DIR, artifact_paths, load_predictor


def sample_data(rows_per_class: int = 24) -> pd.DataFrame:
    """Create unique observations with repeatable binary G-C-B-A labels."""
    rng = np.random.default_rng(11)
    rows = rows_per_class * 3
    features = pd.DataFrame(rng.normal(size=(rows, 6)), columns=modeling.FEATURE_COLUMNS)
    labels = ["0000", "1001", "1111"]
    for index, column in enumerate(modeling.LABEL_COLUMNS):
        features[column] = np.repeat([int(label[index]) for label in labels], rows_per_class)
    features["Ia"] += np.repeat([0, 4, 8], rows_per_class)
    return features


class RecordingScaler(StandardScaler):
    """Record fitted row positions to prove CV and holdout isolation."""

    fitted_rows: list[set[int]] = []

    def fit(self, X, y=None, sample_weight=None):
        self.fitted_rows.append(set(X.index))
        return super().fit(X, y, sample_weight=sample_weight)


class ValidationTests(unittest.TestCase):
    def test_float_binary_flags_preserve_four_bit_labels(self):
        data = sample_data(2).astype({column: float for column in modeling.LABEL_COLUMNS})
        self.assertEqual(modeling.fault_labels(data).tolist(), ["0000"] * 2 + ["1001"] * 2 + ["1111"] * 2)

    def test_binary_label_validation_rejects_fractional_missing_and_nonnumeric(self):
        for invalid in (0.5, 2, np.nan, np.inf, "yes"):
            with self.subTest(invalid=invalid):
                data = sample_data(2).astype({"G": object})
                data.loc[0, "G"] = invalid
                with self.assertRaisesRegex(ValueError, "binary"):
                    modeling.fault_labels(data)

    def test_optional_labels_require_all_or_none(self):
        data = sample_data()
        self.assertIsNone(modeling.fault_labels(data.drop(columns=list(modeling.LABEL_COLUMNS)), required=False))
        with self.assertRaisesRegex(ValueError, "missing: A"):
            modeling.fault_labels(data.drop(columns="A"), required=False)

    def test_features_are_ordered_numeric_and_source_unchanged(self):
        data = sample_data()[list(reversed(sample_data().columns))].astype({"Ia": str})
        original = data.copy(deep=True)
        features = modeling.validate_features(data)
        self.assertEqual(features.columns.tolist(), list(modeling.FEATURE_COLUMNS))
        self.assertTrue(pd.api.types.is_numeric_dtype(features["Ia"]))
        pd.testing.assert_frame_equal(data, original)

    def test_feature_errors_are_actionable(self):
        for invalid in (np.nan, np.inf, -np.inf, "bad"):
            with self.subTest(invalid=invalid):
                data = sample_data().astype({"Va": object})
                data.loc[0, "Va"] = invalid
                with self.assertRaises(ValueError):
                    modeling.validate_features(data)
        with self.assertRaisesRegex(ValueError, "Missing required feature columns: Va"):
            modeling.validate_features(sample_data().drop(columns="Va"))
        with self.assertRaisesRegex(ValueError, "at least one"):
            modeling.validate_features(sample_data().iloc[:0])

    def test_confusion_matrix_retains_full_class_order_for_subset(self):
        metrics = modeling.classification_metrics(["1001", "1001"], ["1001", "0000"], ["0000", "1001", "1111"])
        self.assertEqual(metrics["confusion_matrix"], [[0, 0, 0], [1, 1, 0], [0, 0, 0]])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["per_class"]["1001"]["support"], 2)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3) / 3)

    def test_metrics_never_silently_drop_unknown_labels(self):
        with self.assertRaisesRegex(ValueError, "every observed label"):
            modeling.classification_metrics(["0011"], ["1001"], ["1001"])


class TrainingTests(unittest.TestCase):
    def test_scaler_fits_only_each_cv_train_fold_and_final_train_rows(self):
        data = sample_data()
        RecordingScaler.fitted_rows = []
        with patch.object(modeling, "StandardScaler", RecordingScaler):
            result = modeling.train_and_evaluate(data, models={"logistic": LogisticRegression()}, folds=3)
        split = result.report["split"]
        train_rows, test_rows = set(split["train_row_positions"]), set(split["test_row_positions"])
        self.assertFalse(train_rows & test_rows)
        self.assertEqual(train_rows | test_rows, set(range(len(data))))
        self.assertEqual(len(RecordingScaler.fitted_rows), 4)
        for rows in RecordingScaler.fitted_rows[:3]:
            self.assertLess(rows, train_rows)
            self.assertFalse(rows & test_rows)
        self.assertEqual(RecordingScaler.fitted_rows[-1], train_rows)
        for row in train_rows:
            self.assertEqual(sum(row in rows for rows in RecordingScaler.fitted_rows[:3]), 2)
        np.testing.assert_allclose(result.pipeline.named_steps["scaler"].mean_, modeling.validate_features(data).iloc[split["train_row_positions"]].mean())
        self.assertEqual(result.report["held_out_test"]["samples"], len(test_rows))
        self.assertIn("majority_baseline", result.report)
        self.assertEqual(result.report["saved_model_training_partition"], "training rows only; held-out rows never fitted")

    def test_repeated_seed_reproduces_split_predictions_and_scores(self):
        data = sample_data()
        first = modeling.train_and_evaluate(data, models={"logistic": LogisticRegression()}, folds=3)
        second = modeling.train_and_evaluate(data, models={"logistic": LogisticRegression()}, folds=3)
        self.assertEqual(first.report["split"], second.report["split"])
        self.assertEqual(first.report["cross_validation"], second.report["cross_validation"])
        self.assertEqual(first.report["held_out_test"], second.report["held_out_test"])
        np.testing.assert_array_equal(first.predict(data), second.predict(data))

    def test_insufficient_classes_and_duplicate_measurements_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least two classes"):
            modeling.train_and_evaluate(sample_data().iloc[:24], folds=3)
        with self.assertRaisesRegex(ValueError, "at least 5 rows"):
            modeling.train_and_evaluate(sample_data(4), folds=5)
        data = sample_data()
        with self.assertRaisesRegex(ValueError, "Duplicate feature rows"):
            modeling.train_and_evaluate(pd.concat([data, data.iloc[[0]]]), folds=3)


class ArtifactTests(unittest.TestCase):
    def test_cli_roundtrip_runs_outside_repository_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data.csv"
            output = root / "run"
            data = sample_data()
            data.to_csv(source, index=False)
            command = [sys.executable, str(PROJECT_DIR / "fault_classification_v2.py"), "--data", str(source), "--output-dir", str(output), "--models", "Logistic Regression", "--folds", "3", "--predict", str(source)]
            completed = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=40)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads((output / "evaluation.json").read_text())
            self.assertEqual(report["dataset"]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            predictor = load_predictor(str(output))
            predictions = predictor.predict(data)
            self.assertEqual(predictions.columns.tolist(), ["Fault Code", "Fault String"])
            self.assertEqual(len(predictions), len(data))
            saved = pd.read_csv(output / "predictions.csv", dtype={"Predicted": str})
            self.assertEqual(saved["Predicted"].tolist(), predictions["Fault String"].tolist())
            original_bundle = (output / "model_bundle.joblib").read_bytes()
            repeated = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=40)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("already exists", repeated.stderr)
            self.assertEqual((output / "model_bundle.joblib").read_bytes(), original_bundle)

    def test_configured_missing_bundle_does_not_fall_back_to_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_predictor(directory)

    def test_legacy_adapter_preserves_predictions_with_mocked_trusted_artifacts(self):
        data = sample_data()
        trained = modeling.train_and_evaluate(data, models={"logistic": LogisticRegression()}, folds=3)
        artifacts = [trained.pipeline.named_steps["classifier"], trained.pipeline.named_steps["scaler"], trained.label_encoder]
        with patch("fault_classifier.artifacts.joblib.load", side_effect=artifacts) as loader:
            predictor = load_predictor()
        self.assertIsNone(predictor.report)
        np.testing.assert_array_equal(predictor.predict(data)["Fault String"], trained.predict(data))
        self.assertEqual([call.args[0] for call in loader.call_args_list], list(artifact_paths()))
        self.assertTrue(all(path.is_absolute() for path in artifact_paths()))

    def test_app_smoke_with_newly_trained_trusted_bundle(self):
        from streamlit.testing.v1 import AppTest

        trained = modeling.train_and_evaluate(sample_data(), models={"logistic": LogisticRegression()}, folds=3)
        with tempfile.TemporaryDirectory() as directory:
            joblib.dump({"schema_version": 1, "pipeline": trained.pipeline, "label_encoder": trained.label_encoder, "report": trained.report}, Path(directory) / "model_bundle.joblib")
            with patch.dict(os.environ, {"FAULT_CLASSIFIER_ARTIFACT_DIR": directory}):
                app = AppTest.from_file(str(PROJECT_DIR / "fault_classifier_app.py")).run(timeout=20)
            self.assertEqual(len(app.exception), 0, str(app.exception))
            self.assertEqual(len(app.error), 0, str(app.error))
            self.assertTrue(any("Held-out" in element.value for element in app.subheader))


if __name__ == "__main__":
    unittest.main()
