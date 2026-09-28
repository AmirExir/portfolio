"""Streamlit interface for validated fault inference and traceable evaluation."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import pickle
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import streamlit as st

if __package__:
    from .artifacts import FaultPredictor, artifact_paths, load_predictor
    from .modeling import classification_metrics, fault_labels
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from fault_classifier.artifacts import FaultPredictor, artifact_paths, load_predictor
    from fault_classifier.modeling import classification_metrics, fault_labels

LOGGER = logging.getLogger(__name__)


@st.cache_resource
def cached_predictor(directory: str | None, file_versions: tuple[tuple[int, int], ...]) -> tuple[FaultPredictor, list[str]]:
    """Cache immutable local artifacts, invalidating when file size or mtime changes."""
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always")
        predictor = load_predictor(directory)
    return predictor, [str(notice.message) for notice in notices]


def show_metrics(metrics: dict) -> None:
    """Show class-sensitive scores and a matrix with the explicit label order."""
    left, right = st.columns(2)
    left.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    right.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
    st.caption("Macro F1 averages all listed classes; absent classes contribute zero. See support counts below.")
    st.dataframe(pd.DataFrame({label: metrics["per_class"][label] for label in metrics["classes"]}).T)
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay(
        np.asarray(metrics["confusion_matrix"]), display_labels=metrics["classes"]
    ).plot(ax=ax, cmap="viridis", values_format="d", colorbar=False)
    ax.set_xlabel("Predicted G-C-B-A label")
    ax.set_ylabel("True G-C-B-A label")
    st.pyplot(fig)
    plt.close(fig)


def main() -> None:
    """Render the app using an optional operator-configured, trusted model run."""
    st.set_page_config(page_title="Power Fault Classifier", layout="centered")
    st.title("Power System Fault Classifier by Amir Exir")
    st.write("Upload a CSV with `Ia`, `Ib`, `Ic`, `Va`, `Vb`, `Vc`. Optional true labels use all four flags: `G`, `C`, `B`, `A`.")
    st.caption("Fault strings preserve G-C-B-A bit order. Use the same measurement units and acquisition conventions as the training dataset; those units are not documented in the source CSV.")
    directory = os.environ.get("FAULT_CLASSIFIER_ARTIFACT_DIR") or None
    # Model locations are controlled by the deployment operator, never CSV uploads.
    try:
        versions = tuple((path.stat().st_mtime_ns, path.stat().st_size) for path in artifact_paths(directory))
        predictor, notices = cached_predictor(directory, versions)
    except (OSError, ValueError, TypeError, AttributeError, ImportError, EOFError, pickle.UnpicklingError):
        LOGGER.exception("Unable to load fault-classifier artifacts")
        st.error("Unable to load model artifacts. Train a new run and set FAULT_CLASSIFIER_ARTIFACT_DIR to its trusted local directory; check the application log for details.")
        st.stop()
    for notice in notices:
        st.warning(f"Model compatibility warning: {notice}")
    if predictor.report is None:
        st.warning("Using the repository's legacy model. Its original evaluation fitted scaling before splitting and refitted on all rows. The historical accuracy files do not establish independent test performance. Train a new run for a held-out evaluation.")
    else:
        report = predictor.report
        st.subheader("Held-out training-run evaluation")
        st.write(f"Selected model: {report['selected_model']} (selected by training-only CV macro F1)")
        st.caption(f"Test rows: {report['held_out_test']['samples']}; seed: {report['random_seed']}. The saved model was fitted only on training rows.")
        st.dataframe(pd.DataFrame({
            "Selected model": {key: report["held_out_test"][key] for key in ("accuracy", "macro_f1")},
            "Majority baseline": {key: report["majority_baseline"][key] for key in ("accuracy", "macro_f1")},
        }).T)
        with st.expander("Cross-validation scores and held-out class metrics"):
            st.dataframe(pd.DataFrame({
                name: {"CV macro F1": values["macro_f1"]["mean"], "CV accuracy": values["accuracy"]["mean"]}
                for name, values in report["cross_validation"].items()
            }).T)
            show_metrics(report["held_out_test"])
            st.json({key: report[key] for key in ("dataset", "source_revision", "versions") if key in report})
        for limitation in report["limitations"]:
            st.caption(limitation)

    uploaded_file = st.file_uploader("Upload test CSV", type="csv")
    if uploaded_file is None:
        return
    try:
        data = pd.read_csv(uploaded_file)
        truth = fault_labels(data, required=False)
        predictions = predictor.predict(data)
        if truth is not None:
            predictions["True Fault"] = truth
        st.subheader("Predicted fault types")
        st.dataframe(predictions)
        st.download_button("Download Results", predictions.to_csv(index=False), "predictions.csv", "text/csv")
        if truth is not None:
            st.subheader("Uploaded-data evaluation")
            st.info("Uploaded rows may overlap training data. These scores are diagnostic and are not an independent test unless you verify the data provenance.")
            unknown = sorted(set(truth) - set(predictor.classes))
            if unknown:
                st.warning(f"True labels absent from model training: {', '.join(unknown)}. They remain included in evaluation.")
            classes = predictor.classes + unknown
            show_metrics(classification_metrics(truth, predictions["Fault String"], classes))
    except (ValueError, TypeError, pd.errors.ParserError, UnicodeError) as exc:
        st.error(f"Unable to process CSV: {exc}")


if __name__ == "__main__":
    main()
