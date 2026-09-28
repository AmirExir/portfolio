"""Runtime compatibility regressions use synthetic data and trusted test bundles."""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import sklearn
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.linear_model import LogisticRegression
import streamlit as st
from streamlit.testing.v1 import AppTest

from fault_classifier import artifacts, modeling
from fault_classifier.tests.test_modeling import sample_data

SAVED_SKLEARN_VERSION = "1.6.1" if sklearn.__version__ != "1.6.1" else "1.5.2"


def incompatible_estimator(*args: object, **kwargs: object) -> None:
    """Simulate a trusted pickle emitting the deployed application's warning."""
    warnings.warn(
        InconsistentVersionWarning(
            estimator_name="DecisionTreeClassifier",
            current_sklearn_version=sklearn.__version__,
            original_sklearn_version=SAVED_SKLEARN_VERSION,
        ),
        stacklevel=2,
    )
    raise AssertionError("Deserialization continued after a scikit-learn version mismatch.")


class RuntimeCompatibilityTests(unittest.TestCase):
    """Reject incompatible estimators before predictions or automatic fallback."""

    def test_legacy_version_warning_aborts_at_first_estimator(self) -> None:
        with patch.object(artifacts.joblib, "load", side_effect=incompatible_estimator) as loader:
            with self.assertRaises(artifacts.ModelCompatibilityError) as caught:
                artifacts.load_predictor()
        self.assertIn(SAVED_SKLEARN_VERSION, str(caught.exception))
        self.assertIn(sklearn.__version__, str(caught.exception))
        self.assertEqual(loader.call_count, 1)

    def test_configured_bundle_warning_never_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(artifacts.joblib, "load", side_effect=incompatible_estimator) as loader,
                patch.object(artifacts, "train_default_predictor") as train_default,
            ):
                with self.assertRaises(artifacts.ModelCompatibilityError):
                    artifacts.load_predictor(directory)
            loader.assert_called_once_with(Path(directory).resolve() / "model_bundle.joblib")
            train_default.assert_not_called()

    def test_bundle_version_metadata_is_required_even_without_a_pickle_warning(self) -> None:
        for recorded_version in (None, SAVED_SKLEARN_VERSION):
            with self.subTest(recorded_version=recorded_version):
                report = {"features": list(modeling.FEATURE_COLUMNS)}
                if recorded_version is not None:
                    report["versions"] = {"scikit_learn": recorded_version}
                bundle = {
                    "schema_version": 1,
                    "pipeline": object(),
                    "label_encoder": object(),
                    "report": report,
                }
                with patch.object(artifacts.joblib, "load", return_value=bundle):
                    with self.assertRaises(artifacts.ModelCompatibilityError) as caught:
                        artifacts.load_predictor("/trusted/operator-selected-test-directory")
                self.assertIn(recorded_version or "unknown", str(caught.exception))
                self.assertIn(sklearn.__version__, str(caught.exception))

    def test_default_training_uses_current_runtime_and_independent_test_rows(self) -> None:
        data = sample_data()
        content = data.to_csv(index=False).encode("utf-8")
        with patch.object(artifacts.joblib, "load") as loader:
            predictor = artifacts.train_default_predictor(content)
        loader.assert_not_called()
        report = predictor.report
        self.assertIsNotNone(report)
        self.assertEqual(report["versions"]["scikit_learn"], sklearn.__version__)
        self.assertEqual(report["dataset"]["sha256"], sha256(content).hexdigest())
        self.assertEqual(report["selected_model"], "Random Forest")
        split = report["split"]
        train_rows, test_rows = set(split["train_row_positions"]), set(split["test_row_positions"])
        self.assertFalse(train_rows & test_rows)
        self.assertEqual(train_rows | test_rows, set(range(len(data))))
        self.assertEqual(report["held_out_test"]["samples"], len(test_rows))
        np.testing.assert_allclose(
            predictor.model.named_steps["scaler"].mean_,
            modeling.validate_features(data).iloc[split["train_row_positions"]].mean(),
        )
        self.assertEqual(predictor.predict(data).shape, (len(data), 2))


class RuntimeAppTests(unittest.TestCase):
    """Exercise actual Streamlit startup while keeping training small and local."""

    @classmethod
    def setUpClass(cls) -> None:
        """Prepare a valid predictor without loading repository pickle artifacts."""
        cls.data = sample_data()
        result = modeling.train_and_evaluate(
            cls.data, models={"Logistic Regression": LogisticRegression()}, folds=3
        )
        cls.predictor = artifacts.FaultPredictor(result.pipeline, result.label_encoder, result.report)

    def setUp(self) -> None:
        st.cache_resource.clear()

    def tearDown(self) -> None:
        st.cache_resource.clear()

    def test_default_startup_avoids_pickles_and_retrains_only_when_source_changes(self) -> None:
        source_content = [self.data.to_csv(index=False).encode("utf-8")]
        source_path = artifacts.PROJECT_DIR / "classData.csv"
        original_read_bytes = Path.read_bytes

        def read_source(path: Path) -> bytes:
            if path.resolve() == source_path.resolve():
                return source_content[0]
            return original_read_bytes(path)

        with (
            patch.dict(os.environ, {"FAULT_CLASSIFIER_ARTIFACT_DIR": ""}),
            patch.object(Path, "read_bytes", autospec=True, side_effect=read_source),
            patch.object(artifacts, "train_default_predictor", return_value=self.predictor) as train_default,
            patch.object(artifacts.joblib, "load") as loader,
        ):
            app = AppTest.from_file(str(artifacts.PROJECT_DIR / "fault_classifier_app.py"))
            app.run(timeout=20)
            self.assertEqual(len(app.exception), 0, str(app.exception))
            self.assertEqual(len(app.error), 0, str(app.error))
            self.assertFalse(any("compatibility" in notice.value.lower() for notice in app.warning))
            self.assertTrue(any("Held-out" in heading.value for heading in app.subheader))
            train_default.assert_called_once_with(source_content[0])

            app.run(timeout=20)
            self.assertEqual(len(app.exception), 0, str(app.exception))
            self.assertEqual(train_default.call_count, 1)

            changed_data = self.data.copy()
            changed_data.loc[0, "Ia"] += 1.0
            source_content[0] = changed_data.to_csv(index=False).encode("utf-8")
            app.run(timeout=20)
            self.assertEqual(len(app.exception), 0, str(app.exception))
            self.assertEqual(len(app.error), 0, str(app.error))
            self.assertEqual(train_default.call_count, 2)
            self.assertEqual(train_default.call_args.args, (source_content[0],))
            loader.assert_not_called()

    def test_configured_incompatible_bundle_stops_without_default_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "model_bundle.joblib").touch()
            with (
                patch.dict(os.environ, {"FAULT_CLASSIFIER_ARTIFACT_DIR": directory}),
                patch.object(artifacts.joblib, "load", side_effect=incompatible_estimator) as loader,
                patch.object(artifacts, "train_default_predictor") as train_default,
            ):
                app = AppTest.from_file(str(artifacts.PROJECT_DIR / "fault_classifier_app.py")).run(timeout=20)
            self.assertEqual(len(app.exception), 0, str(app.exception))
            self.assertEqual(len(app.error), 1, str(app.error))
            self.assertEqual(len(app.warning), 0, str(app.warning))
            self.assertIn(SAVED_SKLEARN_VERSION, app.error[0].value)
            self.assertEqual(len(app.get("file_uploader")), 0)
            self.assertEqual(loader.call_count, 1)
            train_default.assert_not_called()


if __name__ == "__main__":
    unittest.main()
