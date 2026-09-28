"""Exercise user-visible CSV and sample workflows without deserializing models."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from io import BytesIO
import os
from pathlib import Path
from typing import Iterator
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from sklearn.linear_model import LogisticRegression
import streamlit as st
from streamlit.testing.v1 import AppTest

from fault_classifier import artifacts, modeling
from fault_classifier.tests.test_modeling import sample_data


class AppWorkflowTests(unittest.TestCase):
    """Preserve sample provenance, row alignment and actionable CSV errors."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build one trusted deterministic predictor for presentation checks."""
        cls.source = sample_data()
        result = modeling.train_and_evaluate(
            cls.source, models={"Logistic Regression": LogisticRegression()}, folds=3,
        )
        cls.predictor = artifacts.FaultPredictor(result.pipeline, result.label_encoder, result.report)

    def setUp(self) -> None:
        st.cache_resource.clear()

    def tearDown(self) -> None:
        st.cache_resource.clear()

    @contextmanager
    def running_app(self, upload: bytes | None = None) -> Iterator[tuple[AppTest, MagicMock]]:
        """Use synthetic CSV bytes and avoid legacy estimator loading entirely."""
        original_read_bytes = Path.read_bytes

        def read_source(path: Path) -> bytes:
            if path.resolve() == artifacts.PROJECT_DIR / "classData.csv":
                return self.source.to_csv(index=False).encode("utf-8")
            return original_read_bytes(path)

        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"FAULT_CLASSIFIER_ARTIFACT_DIR": ""}))
            stack.enter_context(patch.object(Path, "read_bytes", autospec=True, side_effect=read_source))
            prepare = stack.enter_context(patch.object(artifacts, "train_default_predictor", return_value=self.predictor))
            loader = stack.enter_context(patch.object(artifacts.joblib, "load"))
            if upload is not None:
                stack.enter_context(patch.object(st, "file_uploader", side_effect=lambda *args, **kwargs: BytesIO(upload)))
            app = AppTest.from_file(str(artifacts.PROJECT_DIR / "fault_classifier_app.py")).run(timeout=20)
            yield app, prepare
            loader.assert_not_called()

    def test_startup_prioritizes_upload_and_keeps_evaluation_in_its_tab(self) -> None:
        with self.running_app() as (app, _):
            self.assertFalse(app.exception, str(app.exception))
            self.assertFalse(app.error, str(app.error))
            self.assertEqual(app.title[0].value, "Power Fault Classifier")
            self.assertEqual([tab.label for tab in app.tabs], ["Analyze measurements", "Model evaluation", "Input guide"])
            self.assertEqual(len(app.tabs[0].get("file_uploader")), 1)
            self.assertEqual(len(app.tabs[0].metric), 0)
            self.assertTrue(any("Held-out" in element.value for element in app.tabs[1].subheader))

    def test_sample_preserves_rows_and_warns_about_training_overlap_without_retraining(self) -> None:
        with self.running_app() as (app, prepare):
            app.radio[0].set_value("Bundled sample").run(timeout=20)
            self.assertFalse(app.exception, str(app.exception))
            self.assertFalse(app.error, str(app.error))
            self.assertTrue(any("Some may be training rows" in element.value for element in app.tabs[0].info))
            result = app.tabs[0].dataframe[0].value
            expected = self.source.sample(n=30, random_state=42).reset_index(drop=True)
            self.assertEqual(result["Data row"].tolist(), list(range(1, 31)))
            # CSV serialization can alter the last floating-point bit.
            np.testing.assert_allclose(result[list(modeling.FEATURE_COLUMNS)], expected[list(modeling.FEATURE_COLUMNS)], rtol=1e-13, atol=1e-14)
            self.assertEqual(result["True Fault"].tolist(), modeling.fault_labels(expected).tolist())
            self.assertEqual(result["Fault String"].tolist(), self.predictor.predict(expected)["Fault String"].tolist())
            self.assertEqual(prepare.call_count, 1)

    def test_unlabeled_upload_generates_predictions_without_diagnostic_truth(self) -> None:
        uploaded = self.source.iloc[:5].drop(columns=list(modeling.LABEL_COLUMNS))
        with self.running_app(uploaded.to_csv(index=False).encode("utf-8")) as (app, _):
            self.assertFalse(app.exception, str(app.exception))
            self.assertFalse(app.error, str(app.error))
            result = app.tabs[0].dataframe[0].value
            self.assertEqual(len(result), len(uploaded))
            self.assertNotIn("True Fault", result)
            self.assertNotIn("Compare with supplied labels", [element.label for element in app.tabs[0].expander])

    def test_invalid_measurements_and_partial_labels_fail_without_predictions(self) -> None:
        invalid_inputs = [
            (b"Ia,Ib\n1,2\n", "Missing required feature columns"),
            (self.source.drop(columns="A").to_csv(index=False).encode("utf-8"), "Provide all four ground-truth flags"),
            (b"Ia,Ib,Ic,Va,Vb,Vc\n1,2,3,4,5,inf\n", "missing or infinite"),
        ]
        for content, message in invalid_inputs:
            with self.subTest(message=message), self.running_app(content) as (app, _):
                self.assertFalse(app.exception, str(app.exception))
                self.assertEqual(len(app.error), 1)
                self.assertIn(message, app.error[0].value)
                self.assertEqual(len(app.tabs[0].dataframe), 0)
                self.assertEqual(len(app.tabs[0].metric), 0)


if __name__ == "__main__":
    unittest.main()
