"""Streamlit interface for validated fault inference and traceable evaluation."""

from __future__ import annotations

import logging
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import pickle
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import ConfusionMatrixDisplay
import streamlit as st

if __package__:
    from .artifacts import FaultPredictor, ModelCompatibilityError, artifact_paths, load_predictor, train_default_predictor
    from .modeling import FEATURE_COLUMNS, LABEL_COLUMNS, classification_metrics, fault_labels
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from fault_classifier.artifacts import FaultPredictor, ModelCompatibilityError, artifact_paths, load_predictor, train_default_predictor
    from fault_classifier.modeling import FEATURE_COLUMNS, LABEL_COLUMNS, classification_metrics, fault_labels

LOGGER = logging.getLogger(__name__)


@st.cache_resource(max_entries=2)
def cached_predictor(
    directory: str, file_versions: tuple[tuple[int, int], ...], core_fingerprint: str,
    sklearn_version: str,
) -> tuple[FaultPredictor, list[str]]:
    """Cache immutable local artifacts, invalidating when file size or mtime changes."""
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always")
        predictor = load_predictor(directory)
    return predictor, list(dict.fromkeys(str(notice.message) for notice in notices))


@st.cache_resource(show_spinner="Preparing and evaluating the fault model...", max_entries=1)
def cached_default_predictor(
    content: bytes, core_fingerprint: str, sklearn_version: str,
) -> tuple[FaultPredictor, list[str]]:
    """Reuse the runtime fit until source data, implementation, or version changes."""
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always")
        predictor = train_default_predictor(content)
    return predictor, list(dict.fromkeys(str(notice.message) for notice in notices))


def show_metrics(metrics: dict) -> None:
    """Show class-sensitive scores and a matrix with the explicit label order."""
    left, right = st.columns(2)
    left.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    right.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
    st.caption("Macro F1 averages all listed classes; absent classes contribute zero. See support counts below.")
    st.dataframe(pd.DataFrame({label: metrics["per_class"][label] for label in metrics["classes"]}).T, use_container_width=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay(
        np.asarray(metrics["confusion_matrix"]), display_labels=metrics["classes"]
    ).plot(ax=ax, cmap="viridis", values_format="d", colorbar=False)
    ax.set_xlabel("Predicted G-C-B-A label")
    ax.set_ylabel("True G-C-B-A label")
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


def show_model_evaluation(predictor: FaultPredictor, configured: bool) -> None:
    """Keep model evidence available without placing it ahead of the upload task."""
    report = predictor.report
    if report is None:
        st.warning("This legacy model has no independent held-out evaluation. Train a new run to evaluate it.")
        return
    st.subheader("Held-out model evaluation")
    st.write("Results from the training run's test partition. These scores describe this dataset, not field protection performance.")
    if configured:
        st.caption(f"{report['selected_model']} · selected by training-only cross-validation macro F1")
    else:
        st.caption(f"{report['selected_model']} · fixed configuration · trained from the bundled dataset")
    st.caption(f"{report['held_out_test']['samples']:,} test rows · seed {report['random_seed']} · model fitted only on training rows")
    st.dataframe(pd.DataFrame({
        "Model": {key: report["held_out_test"][key] for key in ("accuracy", "macro_f1")},
        "Majority baseline": {key: report["majority_baseline"][key] for key in ("accuracy", "macro_f1")},
    }).T.rename(columns={"accuracy": "Accuracy", "macro_f1": "Macro F1"}), use_container_width=True)
    with st.expander("Per-class results and confusion matrix"):
        show_metrics(report["held_out_test"])
    with st.expander("Cross-validation and model provenance"):
        st.dataframe(pd.DataFrame({
            name: {"CV macro F1": values["macro_f1"]["mean"], "CV accuracy": values["accuracy"]["mean"]}
            for name, values in report["cross_validation"].items()
        }).T, use_container_width=True)
        if not configured:
            st.caption(f"Built in scikit-learn {sklearn.__version__} and cached for reuse. No saved estimator is loaded on the default path.")
        st.json({key: report[key] for key in ("dataset", "source_revision", "versions") if key in report})
        st.download_button("Download evaluation report", json.dumps(report, indent=2), "fault_model_evaluation.json", "application/json")
    with st.expander("Evaluation assumptions and limitations"):
        for limitation in report["limitations"]:
            st.write(limitation)


def show_input_guide() -> None:
    """Explain the source-data contract and label encoding without inventing units."""
    st.subheader("Prepare your measurements")
    st.write("Provide one observation per row with these six numeric columns:")
    st.dataframe(pd.DataFrame({
        "Columns": ["Ia, Ib, Ic", "Va, Vb, Vc", "G, C, B, A"],
        "Content": ["Phase current measurements", "Phase voltage measurements", "Optional ground-truth flags (all four or none)"],
        "Requirement": ["Finite numeric values", "Finite numeric values", "Binary 0 or 1"],
    }), hide_index=True, use_container_width=True)
    st.write("The source CSV does not document measurement units or acquisition conventions. No unit conversion is applied; use measurements compatible with the training source. Predictions are exploratory and are not a verified protection setting or study result.")
    st.markdown("**Reading the output**")
    st.write("Fault String preserves the dataset's four binary flags in G-C-B-A order: ground, phase C, phase B, phase A. For example, 1001 encodes G=1, C=0, B=0, A=1. Fault Code is the model's numeric class identifier, not a severity ranking.")
    st.write("Missing, nonnumeric or infinite measurements are rejected. Extra columns are ignored by the model. To evaluate your results, include all four ground-truth flags; a single detection column such as Output (S) is not multiclass ground truth.")
    st.caption("Uploaded-data scores require independently sourced labels to support an independent test. Correlated observations from the same event can inflate row-level test scores.")


def show_predictions(data: pd.DataFrame, predictor: FaultPredictor, *, sample: bool) -> None:
    """Validate measurements, display row-aligned predictions and optional diagnostics."""
    truth = fault_labels(data, required=False)
    predictions = predictor.predict(data)
    if truth is not None:
        predictions["True Fault"] = truth
    results = data.loc[:, list(FEATURE_COLUMNS)].join(predictions)
    results.insert(0, "Data row", np.arange(1, len(data) + 1))

    st.subheader("Classification results")
    rows, classes, labels = st.columns(3)
    rows.metric("Observations", f"{len(data):,}")
    classes.metric("Predicted classes", predictions["Fault String"].nunique())
    labels.metric("Labeled observations", f"{len(data) if truth is not None else 0:,}")
    st.caption("Fault String uses G-C-B-A order. Data row refers to the input row, excluding the CSV header. See Input guide for label details.")
    st.dataframe(results, hide_index=True, use_container_width=True)
    st.download_button("Download predictions", results.to_csv(index=False), "predictions.csv", "text/csv", type="primary")
    with st.expander("Predicted class distribution"):
        counts = predictions["Fault String"].value_counts().sort_index().rename_axis("Fault String").rename("Observations")
        st.bar_chart(counts, color="#0f766e")
    if truth is not None:
        with st.expander("Compare with supplied labels"):
            st.info(
                "This sample comes from the bundled training source and may include training rows. These scores only demonstrate the workflow; they are not an independent test."
                if sample else
                "Uploaded rows may overlap training data. These scores are diagnostic unless you verify that the source is independent of model training."
            )
            unknown = sorted(set(truth) - set(predictor.classes))
            if unknown:
                st.warning(f"True labels absent from model training: {', '.join(unknown)}. They remain included in evaluation.")
            show_metrics(classification_metrics(truth, predictions["Fault String"], predictor.classes + unknown))


def show_analysis(predictor: FaultPredictor, source_path: Path) -> None:
    """Offer an upload first and a clearly labeled bundled demonstration."""
    st.subheader("Classify your measurements")
    st.write("Upload phase currents and voltages to inspect predicted fault labels, then export the results.")
    mode = st.radio("Data source", ["Upload CSV", "Bundled sample"], horizontal=True)
    if mode == "Upload CSV":
        uploaded_file = st.file_uploader("Measurement CSV", type="csv", help="Required: Ia, Ib, Ic, Va, Vb, Vc. Optional ground truth: G, C, B, A.")
        st.caption("Required columns: Ia, Ib, Ic, Va, Vb, Vc. Optional labels: G, C, B, A.")
        st.download_button("Download CSV template", ",".join(FEATURE_COLUMNS) + "\n", "fault_measurements_template.csv", "text/csv")
        if uploaded_file is None:
            st.info("Choose a CSV above, or select Bundled sample to try the workflow.")
            return
        try:
            data = pd.read_csv(uploaded_file)
        except (ValueError, TypeError, pd.errors.ParserError, UnicodeError) as exc:
            st.error(f"Unable to read CSV: {exc}")
            return
    else:
        st.info("Demo: 30 observations from the bundled dataset. Some may be training rows; this sample does not measure independent model performance.")
        try:
            source = pd.read_csv(BytesIO(source_path.read_bytes()))
            data = source.sample(n=min(30, len(source)), random_state=42).reset_index(drop=True)
            data = data.loc[:, [*FEATURE_COLUMNS, *LABEL_COLUMNS]]
        except (OSError, ValueError, KeyError, UnicodeError) as exc:
            st.error(f"Unable to load the bundled sample: {exc}")
            return
        st.download_button("Download sample CSV", data.to_csv(index=False), "fault_sample.csv", "text/csv")
    try:
        show_predictions(data, predictor, sample=mode == "Bundled sample")
    except (ValueError, TypeError) as exc:
        st.error(f"Unable to process CSV: {exc}")


def main() -> None:
    """Render a runtime-trained default or an operator-configured compatible run."""
    st.set_page_config(page_title="Power Fault Classifier", page_icon="⚡", layout="wide")
    st.markdown("""
        <style>
        .fault-eyebrow { color: #0f766e; font-weight: 700; letter-spacing: .12em;
            font-size: .78rem; margin-bottom: .5rem; }
        </style>
        <div class="fault-eyebrow">POWER SYSTEMS · MACHINE LEARNING</div>
        """, unsafe_allow_html=True)
    st.title("Power Fault Classifier")
    st.write("Explore fault patterns in three-phase current and voltage measurements.")
    st.caption("By Amir Exir · Exploratory model · Source measurement units are undocumented")
    directory = os.environ.get("FAULT_CLASSIFIER_ARTIFACT_DIR") or None
    # Model locations are controlled by the deployment operator, never CSV uploads.
    try:
        project_dir = Path(__file__).resolve().parent
        core_fingerprint = sha256(b"".join(
            project_dir.joinpath(name).read_bytes() for name in ("artifacts.py", "modeling.py")
        )).hexdigest()
        if directory:
            versions = tuple((path.stat().st_mtime_ns, path.stat().st_size) for path in artifact_paths(directory))
            predictor, notices = cached_predictor(directory, versions, core_fingerprint, sklearn.__version__)
        else:
            predictor, notices = cached_default_predictor(
                project_dir.joinpath("classData.csv").read_bytes(), core_fingerprint, sklearn.__version__,
            )
    except ModelCompatibilityError as exc:
        st.error(str(exc))
        st.stop()
    except (OSError, ValueError, TypeError, AttributeError, ImportError, EOFError, pickle.UnpicklingError):
        LOGGER.exception("Unable to prepare fault-classifier model")
        st.error(
            "Unable to prepare the fault model. Check the configured model directory "
            "or bundled classData.csv and inspect the application log for details."
        )
        st.stop()
    for notice in notices:
        st.warning(f"Model preparation warning: {notice}")
    analyze, evaluation, guide = st.tabs(["Analyze measurements", "Model evaluation", "Input guide"])
    with analyze:
        show_analysis(predictor, project_dir / "classData.csv")
    with evaluation:
        show_model_evaluation(predictor, configured=directory is not None)
    with guide:
        show_input_guide()


if __name__ == "__main__":
    main()
