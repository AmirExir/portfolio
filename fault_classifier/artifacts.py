"""Inference using locally trusted bundles or the existing repository artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

if __package__:
    from .modeling import FEATURE_COLUMNS, validate_features
else:
    from modeling import FEATURE_COLUMNS, validate_features

PROJECT_DIR = Path(__file__).resolve().parent


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


def load_predictor(directory: str | None = None) -> FaultPredictor:
    """Load only operator-trusted local files; joblib/pickle can execute Python code.

    A configured directory must contain a new bundle. Failure never silently falls
    back to the older repository model. This function must not receive upload paths.
    """
    paths = artifact_paths(directory)
    if directory:
        bundle = joblib.load(paths[0])
        if not isinstance(bundle, dict) or bundle.get("schema_version") != 1:
            raise ValueError("Unsupported model bundle schema. Retrain using the current CLI.")
        if not {"pipeline", "label_encoder", "report"} <= bundle.keys():
            raise ValueError("Incomplete model bundle. Retrain using the current CLI.")
        report = bundle["report"]
        if not isinstance(report, dict) or report.get("features") != list(FEATURE_COLUMNS):
            raise ValueError("Model bundle feature metadata does not match this application.")
        predictor = FaultPredictor(bundle["pipeline"], bundle["label_encoder"], report)
        if report.get("classes") != predictor.classes:
            raise ValueError("Model bundle class metadata does not match its label encoder.")
        return predictor
    model, scaler, encoder = (joblib.load(path) for path in paths)
    return FaultPredictor(model, encoder, None, scaler)
