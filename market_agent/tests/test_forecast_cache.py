from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from market_agent.agent.forecast import _ensemble_result, forecast_close_prices
from market_agent.agent.policy import smart_policy_report
from market_agent.forecast_cache import (
    snapshot_from_model_results,
    snapshots_to_ranking_frame,
)


def _limited_history_snapshot() -> dict:
    """Build the short-history, missing-holdout case seen in SPCX reports."""
    steps = np.arange(82, dtype=float)
    close = 100.0 * np.exp(0.0006 * steps + 0.02 * np.sin(steps / 13.0))
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(len(close), 1_000_000.0),
        },
        index=pd.bdate_range("2026-06-12", periods=len(close)),
    )
    ridge = forecast_close_prices(
        frame,
        horizon_days=30,
        lookback_window=20,
        optimize_model=False,
        symbol="SPCX",
    )
    models = {"Ridge": ridge}
    models["Ensemble"] = _ensemble_result(models)
    policy = smart_policy_report(
        frame,
        forecast_metrics=models["Ensemble"].metrics,
        model_results=models,
    )
    snapshot = snapshot_from_model_results(
        "SPCX",
        float(close[-1]),
        models,
        smart_policy=policy,
        as_of_session=frame.index[-1],
    )
    return json.loads(json.dumps(snapshot, allow_nan=False))


class ForecastCacheTests(unittest.TestCase):
    def test_short_history_null_metrics_preserve_research_only_row(self) -> None:
        snapshot = _limited_history_snapshot()
        metrics = snapshot["models"]["Ensemble"]["metrics"]
        self.assertIsNone(metrics["holdout_mae_pct"])
        self.assertIsNone(metrics["holdout_direction_accuracy"])

        frame, errors = snapshots_to_ranking_frame([snapshot], "Ensemble")

        self.assertEqual(errors, [])
        self.assertEqual(len(frame), 1)
        row = frame.iloc[0]
        self.assertEqual(row["Symbol"], "SPCX")
        self.assertTrue(np.isnan(row["Validation MAE %"]))
        self.assertTrue(np.isnan(row["Direction Hit Rate %"]))
        self.assertFalse(row["Validation Is OOS"])
        self.assertEqual(row["Validation Samples"], 0)
        self.assertFalse(row["Policy Allocation Eligible"])
        self.assertEqual(row["Policy Target %"], 0.0)
        self.assertIn(
            "insufficient_validation_samples",
            row["Policy Allocation Blockers"],
        )

    def test_unavailable_optional_component_return_does_not_drop_row(self) -> None:
        snapshot = _limited_history_snapshot()
        snapshot["models"]["XGBoost"] = {
            "model_name": "unavailable",
            "metrics": {"forecast_change_pct": None},
            "forecast": [],
        }

        frame, errors = snapshots_to_ranking_frame([snapshot], "Ensemble")

        self.assertEqual(errors, [])
        self.assertEqual(len(frame), 1)
        self.assertTrue(np.isnan(frame.iloc[0]["XGBoost Return %"]))


if __name__ == "__main__":
    unittest.main()
