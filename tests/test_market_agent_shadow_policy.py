from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from market_agent.agent import forecast as forecast_module
from market_agent.agent.forecast import (
    ForecastResult,
    _ensemble_result,
    reinforcement_policy_forecast,
)
from market_agent.agent.policy import _rl_component, smart_policy_decision
from market_agent.forecast_cache import build_cache_key, select_model_name


def _forecast_result(
    name: str,
    *,
    forecast_close: float = 100.0,
    forecast_change_pct: float = 0.0,
    holdout_mae_pct: float = 5.0,
    horizon_days: int = 30,
    **extra_metrics,
) -> ForecastResult:
    index = pd.bdate_range("2026-01-02", periods=2)
    forecast = pd.DataFrame(
        {
            "forecast_close": [forecast_close, forecast_close],
            "lower_estimate": [forecast_close * 0.95, forecast_close * 0.95],
            "upper_estimate": [forecast_close * 1.05, forecast_close * 1.05],
            "expected_daily_return_pct": [0.0, 0.0],
        },
        index=index,
    )
    metrics = {
        "forecast_change_pct": forecast_change_pct,
        "probability_up_pct": 50.0,
        "confidence_pct": 50.0,
        "expected_error_pct": 5.0,
        "holdout_mae_pct": holdout_mae_pct,
        "holdout_rmse_pct": holdout_mae_pct,
        "holdout_direction_accuracy": 50.0,
        "horizon_days": horizon_days,
    }
    metrics.update(extra_metrics)
    return ForecastResult(forecast=forecast, metrics=metrics, model_name=name)


def _price_frame(*, rising: bool = False) -> pd.DataFrame:
    index = pd.bdate_range("2025-01-02", periods=260)
    close = np.linspace(100.0, 200.0, len(index)) if rising else np.full(len(index), 100.0)
    return pd.DataFrame({"close": close}, index=index)


class ShadowModelSelectionTests(unittest.TestCase):
    def test_best_validation_never_selects_rl_policy(self):
        results = {
            "RL Policy": _forecast_result("RL", holdout_mae_pct=0.01),
            "Ridge": _forecast_result("Ridge", holdout_mae_pct=4.0),
        }

        self.assertEqual(
            select_model_name(results, preferred="Best Validation"),
            "Ridge",
        )

    def test_live_ensemble_excludes_rl_policy(self):
        ridge = _forecast_result(
            "Ridge",
            forecast_close=100.0,
            forecast_change_pct=0.0,
            holdout_mae_pct=10.0,
        )
        rl = _forecast_result(
            "RL",
            forecast_close=1000.0,
            forecast_change_pct=900.0,
            holdout_mae_pct=0.01,
        )

        ensemble = _ensemble_result({"Ridge": ridge, "RL Policy": rl})

        self.assertEqual(ensemble.metrics["component_weights"], {"Ridge": 1.0})
        pd.testing.assert_series_equal(
            ensemble.forecast["forecast_close"],
            ridge.forecast["forecast_close"],
            check_names=False,
        )
        self.assertEqual(ensemble.metrics["forecast_change_pct"], 0.0)


class ShadowRlContributionTests(unittest.TestCase):
    def test_unseen_rl_forecast_state_defaults_to_hold(self):
        states = np.zeros((40, 5), dtype=float)
        targets = np.linspace(-0.02, 0.02, len(states))

        with (
            patch.object(
                forecast_module,
                "_rl_training_data",
                return_value=(states, targets),
            ),
            patch.object(forecast_module, "_update_rl_q_table", return_value=None),
            patch.object(
                forecast_module,
                "_rl_state_from_window",
                return_value=np.ones(5, dtype=float),
            ),
        ):
            result = reinforcement_policy_forecast(
                _price_frame(),
                horizon_days=30,
                lookback_window=20,
                warm_start=False,
            )

        self.assertEqual(result.metrics["rl_action"], "hold")
        self.assertEqual(result.metrics["forecast_change_pct"], 0.0)

    def test_missing_rl_state_defaults_to_hold_without_contribution(self):
        component, diagnostics = _rl_component({})

        self.assertEqual(component, 0.0)
        self.assertEqual(diagnostics["action"], "hold")

    def test_tied_rl_values_default_to_hold_without_contribution(self):
        component, diagnostics = _rl_component(
            {
                "rl_action": "long",
                "rl_q_short": 0.0,
                "rl_q_hold": 0.0,
                "rl_q_long": 0.0,
            }
        )

        self.assertEqual(component, 0.0)
        self.assertEqual(diagnostics["action"], "hold")

    def test_low_reliability_signal_has_zero_allocation(self):
        decision = smart_policy_decision(
            df=_price_frame(rising=True),
            equity=100_000.0,
            risk_fraction=0.10,
            forecast_metrics={
                "forecast_change_pct": 20.0,
                "expected_error_pct": 2.0,
                "confidence_pct": 80.0,
                "reliability": "Low",
                "horizon_days": 30,
            },
        )

        self.assertEqual(decision.action, "HOLD")
        self.assertEqual(decision.target_qty, 0)
        self.assertEqual(decision.target_position_fraction, 0.0)

    def test_short_horizon_auxiliary_results_do_not_change_long_horizon_policy(self):
        primary_metrics = {
            "forecast_change_pct": 0.0,
            "expected_error_pct": 5.0,
            "confidence_pct": 50.0,
            "horizon_days": 30,
        }
        baseline = smart_policy_decision(
            df=_price_frame(),
            equity=100_000.0,
            risk_fraction=0.10,
            forecast_metrics=primary_metrics,
        )
        with_one_day_results = smart_policy_decision(
            df=_price_frame(),
            equity=100_000.0,
            risk_fraction=0.10,
            forecast_metrics=primary_metrics,
            model_results={
                "Ensemble": _forecast_result(
                    "one-day ensemble",
                    forecast_change_pct=10.0,
                    holdout_mae_pct=1.0,
                    horizon_days=1,
                    probability_up_pct=90.0,
                    confidence_pct=90.0,
                    expected_error_pct=1.0,
                ),
                "RL Policy": _forecast_result(
                    "one-day RL",
                    horizon_days=1,
                    rl_action="long",
                    rl_q_short=-0.02,
                    rl_q_hold=0.0,
                    rl_q_long=0.08,
                ),
            },
        )

        self.assertEqual(with_one_day_results.score, baseline.score)
        self.assertEqual(with_one_day_results.action, baseline.action)
        self.assertEqual(with_one_day_results.target_position_fraction, 0.0)


class HorizonIsolationTests(unittest.TestCase):
    def test_model_cache_key_separates_one_and_thirty_day_policies(self):
        common = {
            "symbols": ["SNDK"],
            "history_days": 913,
            "lookback_window": 20,
            "ridge_alpha": 10.0,
            "optimize_model": True,
            "use_market_context": True,
            "sequence_model": "off",
            "include_rl_policy": True,
        }

        one_day_key = build_cache_key(horizon_days=1, **common)
        thirty_day_key = build_cache_key(horizon_days=30, **common)

        self.assertNotEqual(one_day_key, thirty_day_key)


class ShadowReliabilityTests(unittest.TestCase):
    @staticmethod
    def _report_module():
        # yfinance is an execution dependency of the report script, but this
        # pure reporting test must also run in lightweight test environments.
        if "yfinance" not in sys.modules:
            yfinance_stub = types.ModuleType("yfinance")
            yfinance_stub.download = lambda *_args, **_kwargs: pd.DataFrame()
            sys.modules["yfinance"] = yfinance_stub

        from market_agent import daily_ml_forecast_report

        return daily_ml_forecast_report

    def test_rl_only_metrics_never_receive_actionable_reliability(self):
        grade = self._report_module().reliability_grade(
            {
                "Selected Model": "RL Policy",
                "Forecast Return %": 50.0,
                "Expected Error %": 1.0,
                "Model Edge %": 40.0,
                "Direction Hit Rate %": 99.0,
                "Validation MAE %": 0.5,
            }
        )

        self.assertEqual(grade, "Low")

    def test_low_reliability_report_row_has_zero_policy_target(self):
        rows = self._report_module().enrich_rows_with_signal_metadata(
            [
                {
                    "Selected Model": "Ridge",
                    "Model Call": "Neutral / No Edge",
                    "Forecast Return %": 1.0,
                    "Expected Error %": 10.0,
                    "Model Edge %": 1.0,
                    "Direction Hit Rate %": 40.0,
                    "Validation MAE %": 10.0,
                    "Smart Policy": "Buy",
                    "Policy Score": 1.0,
                    "Policy Target %": 10.0,
                }
            ],
            SimpleNamespace(min_signal_return_pct=2.0),
        )

        self.assertEqual(rows[0]["Reliability"], "Low")
        self.assertEqual(rows[0]["Policy Target %"], 0.0)


if __name__ == "__main__":
    unittest.main()
