"""Train fault classifiers without leaking test data or overwriting prior artifacts."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
from typing import Sequence

import joblib
import pandas as pd

if __package__:
    from .modeling import candidate_models, train_and_evaluate
else:
    from modeling import candidate_models, train_and_evaluate

PROJECT_DIR = Path(__file__).resolve().parent


def main(argv: Sequence[str] | None = None) -> int:
    """Run reproducible training and save a new, operator-trusted model directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_DIR / "classData.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing paths are never overwritten")
    parser.add_argument("--predict", type=Path, help="Optional feature CSV to predict after training")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--include-xgboost", action="store_true")
    parser.add_argument("--models", nargs="+", help="Subset of model names, quoted when containing spaces")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("--output-dir already exists; choose a new run directory.")
    try:
        models = candidate_models(args.seed, include_xgboost=args.include_xgboost)
        if args.models:
            unknown = set(args.models) - set(models)
            if unknown:
                parser.error(f"Unknown model names: {sorted(unknown)}. Available: {list(models)}")
            models = {name: models[name] for name in args.models}
        source_bytes = args.data.read_bytes()
        result = train_and_evaluate(
            pd.read_csv(BytesIO(source_bytes)), models=models, seed=args.seed,
            test_fraction=args.test_fraction, folds=args.folds,
        )
        prediction_data = None
        if args.predict:
            prediction_data = pd.read_csv(args.predict)
            prediction_data["Predicted"] = result.predict(prediction_data)
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, capture_output=True,
                text=True, check=True, timeout=5,
            ).stdout.strip()
            dirty = bool(subprocess.run(
                ["git", "status", "--porcelain", "--", str(PROJECT_DIR)],
                cwd=PROJECT_DIR, capture_output=True, text=True, check=True, timeout=5,
            ).stdout.strip())
        except (OSError, subprocess.SubprocessError):
            commit, dirty = None, None
        result.report.update({
            "dataset": {"path": str(args.data.resolve()), "sha256": hashlib.sha256(source_bytes).hexdigest()},
            "source_revision": {"git_commit": commit, "project_worktree_modified": dirty},
            "output_directory": str(args.output_dir.resolve()),
        })
        args.output_dir.mkdir(parents=True, exist_ok=False)
        joblib.dump({
            "schema_version": 1, "pipeline": result.pipeline,
            "label_encoder": result.label_encoder, "report": result.report,
        }, args.output_dir / "model_bundle.joblib")
        (args.output_dir / "evaluation.json").write_text(
            json.dumps(result.report, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8"
        )
        if prediction_data is not None:
            prediction_data.to_csv(args.output_dir / "predictions.csv", index=False)
    except (OSError, ValueError, ImportError, pd.errors.ParserError) as exc:
        parser.exit(1, f"Training failed: {exc}\n")
    print(f"Selected by training CV macro F1: {result.report['selected_model']}")
    print(f"Held-out test macro F1: {result.report['held_out_test']['macro_f1']:.4f}")
    print(f"Majority baseline macro F1: {result.report['majority_baseline']['macro_f1']:.4f}")
    print(f"Saved model and reproducibility report to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
