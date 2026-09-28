"""Runtime-compatible training and inference using locally trusted artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import InconsistentVersionWarning

if __package__:
    from .modeling import FEATURE_COLUMNS, train_and_evaluate, validate_features
else:
    from modeling import FEATURE_COLUMNS, train_and_evaluate, validate_features

PROJECT_DIR = Path(__file__).resolve().parent


class ModelCompatibilityError(ValueError):
    """A saved estimator cannot be used with the active scikit-learn version."""


@dataclass
class FaultPredictor:
    """Prediction adapter preserving legacy encoded outputs and new pipelines."""

    model: Any
    label_encoder: Any
    report: dict[str, Any] | None
    scaler: Any = None

    @property
    def classes(self) -> list[str]:
        """Return model labels in encoder order."""
        return self.label_encoder.classes_.tolist()

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate input and return the existing Fault Code/Fault String columns."""
        features = validate_features(data)
        values = self.scaler.transform(features) if self.scaler is not None else features
        raw_codes = np.asarray(self.model.predict(values))
        if not np.isfinite(raw_codes).all() or not np.equal(raw_codes, raw_codes.astype(int)).all():
            raise ValueError("The model returned invalid encoded class predictions.")
        codes = raw_codes.astype(int)
        labels = self.label_encoder.inverse_transform(codes)
        return pd.DataFrame({"Fault Code": codes, "Fault String": labels}, index=data.index)


def artifact_paths(directory: str | None = None) -> tuple[Path, ...]:
    """Resolve the configured bundle or repository-relative legacy artifacts."""
    if directory:
        return (Path(directory).expanduser().resolve() / "model_bundle.joblib",)
    return tuple(PROJECT_DIR / name for name in ("fault_model.pkl", "scaler.pkl", "label_encoder.pkl"))


def train_default_predictor(content: bytes) -> FaultPredictor:
    """Build an evaluated default in the serving runtime without loading pickle.

    Use one fixed Random Forest candidate to bound startup work. Reuse the same
    train-only CV/held-out evaluation as the CLI; never fit held-out test rows.
    The UI caches this in memory, and source CSV/model files remain unchanged.
    """
    result = train_and_evaluate(
        pd.read_csv(BytesIO(content)),
        models={"Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)},
        seed=42, test_fraction=0.2, folds=5,
    )
    result.report.update({
        "model_origin": "trained in the serving runtime from bundled classData.csv",
        "selection_metric": "fixed Random Forest configuration; CV is diagnostic only",
        "dataset": {"name": "classData.csv", "sha256": sha256(content).hexdigest()},
        "source_revision": {
            "artifacts_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "modeling_sha256": sha256(PROJECT_DIR.joinpath("modeling.py").read_bytes()).hexdigest(),
        },
    })
    return FaultPredictor(result.pipeline, result.label_encoder, result.report)


def _load_compatible_artifact(path: Path) -> Any:
    # Stop at the first incompatible estimator (a forest may contain hundreds).
    # Suppressing this warning would allow unsupported predictions to continue.
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        try:
            return joblib.load(path)
        except InconsistentVersionWarning as exc:
            raise ModelCompatibilityError(
                f"Saved model uses scikit-learn {exc.original_sklearn_version}; "
                f"this app uses {exc.current_sklearn_version}. "
                "Retrain the configured bundle in the app's environment, or remove "
                "FAULT_CLASSIFIER_ARTIFACT_DIR to use the built-in runtime-trained model."
            ) from exc


def load_predictor(directory: str | None = None) -> FaultPredictor:
    """Load only operator-trusted local files; joblib/pickle can execute Python code.

    A configured directory must contain a new bundle. Failure never silently falls
    back to another model. Version mismatches abort loading before inference.
    This function must not receive upload paths. The default UI uses training
    from CSV instead of this function's legacy no-directory compatibility path.
    """
    paths = artifact_paths(directory)
    if directory:
        bundle = _load_compatible_artifact(paths[0])
        if not isinstance(bundle, dict) or bundle.get("schema_version") != 1:
            raise ValueError("Unsupported model bundle schema. Retrain using the current CLI.")
        if not {"pipeline", "label_encoder", "report"} <= bundle.keys():
            raise ValueError("Incomplete model bundle. Retrain using the current CLI.")
        report = bundle["report"]
        if not isinstance(report, dict) or report.get("features") != list(FEATURE_COLUMNS):
            raise ValueError("Model bundle feature metadata does not match this application.")
        trained_version = report.get("versions", {}).get("scikit_learn")
        if trained_version != sklearn.__version__:
            raise ModelCompatibilityError(
                f"Bundle metadata records scikit-learn {trained_version or 'unknown'}; "
                f"this app uses {sklearn.__version__}. Retrain the configured bundle "
                "in the app's environment, or remove FAULT_CLASSIFIER_ARTIFACT_DIR "
                "to use the built-in runtime-trained model."
            )
        predictor = FaultPredictor(bundle["pipeline"], bundle["label_encoder"], report)
        if report.get("classes") != predictor.classes:
            raise ValueError("Model bundle class metadata does not match its label encoder.")
        return predictor
    model, scaler, encoder = (_load_compatible_artifact(path) for path in paths)
    return FaultPredictor(model, encoder, None, scaler)
