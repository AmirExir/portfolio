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
    _build_policy_forecast_context,
    _ensemble_result,
    reinforcement_policy_forecast,
)
from market_agent.agent.policy import _rl_component, smart_policy_decision
from market_agent.agent.shadow_policy import ShadowDecision
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


def _policy_forecast_context(index: pd.Index) -> pd.DataFrame:
    forecast_return = np.full(len(index), 0.08)
    uncertainty = np.full(len(index), 0.02)
    return pd.DataFrame(
        {
            "forecast_return": forecast_return,
            "forecast_probability_up": np.full(len(index), 0.70),
            "forecast_lower_bound": (
                forecast_return - 1.645 * uncertainty
            ),
            "forecast_model_agreement": np.full(len(index), 1.0),
            "forecast_uncertainty": uncertainty,
        },
        index=index,
    )


class ShadowModelSelectionTests(unittest.TestCase):
    def test_best_validation_never_selects_rl_policy(self):
        results = {
            "RL Policy": _forecast_result("RL", holdout_mae_pct=0.01),
            "Ridge": _forecast_result("Ridge", holdout_mae_pct=4.0),
            "Ensemble": _forecast_result("Ensemble", holdout_mae_pct=8.0),
            "XGBoost": _forecast_result("XGBoost", holdout_mae_pct=0.5),
        }

        self.assertEqual(
            select_model_name(results, preferred="Best Validation"),
            "Ensemble",
        )

    def test_best_validation_uses_ridge_when_fixed_ensemble_is_unavailable(self):
        results = {
            "Ridge": _forecast_result("Ridge", holdout_mae_pct=4.0),
            "XGBoost": _forecast_result("XGBoost", holdout_mae_pct=0.5),
        }

        self.assertEqual(
            select_model_name(results, preferred="Best Validation"),
            "Ridge",
        )

    def test_unusable_preferred_model_falls_back_to_live_candidate(self):
        results = {
            "XGBoost": ForecastResult(
                forecast=pd.DataFrame(),
                metrics={"error": "unavailable", "live_eligible": True},
                model_name="XGBoost",
            ),
            "Ridge": _forecast_result(
                "Ridge",
                forecast_change_pct=1.0,
                holdout_mae_pct=2.0,
                live_eligible=True,
            ),
        }

        self.assertEqual(select_model_name(results, preferred="XGBoost"), "Ridge")

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

    def test_policy_context_never_reads_holdout_actual_returns(self):
        close = _price_frame(rising=True)["close"]
        metrics = {
            "forecast_change_pct": 8.0,
            "probability_up_pct": 70.0,
            "expected_error_pct": 2.0,
            "holdout_samples": 3,
            "holdout_predicted_return_pct": [1.0, 2.0, 3.0],
            "holdout_actual_return_pct": [-50.0, 50.0, -50.0],
            "holdout_probability_up": [0.55, 0.60, 0.65],
            "holdout_expected_error_pct": [2.0, 2.5, 3.0],
            "holdout_model_agreement": [0.5, 0.75, 1.0],
            "validation_is_oos": True,
            "validation_scheme": "nested_purged_outer_holdout",
            "horizon_days": 30,
            "live_eligible": True,
        }
        original = _forecast_result("Ensemble", **metrics)
        changed = _forecast_result(
            "Ensemble",
            **{
                **metrics,
                "holdout_actual_return_pct": [999.0, 999.0, 999.0],
            },
        )

        first = _build_policy_forecast_context(
            close=close,
            source_name="Ensemble",
            result=original,
            components={"Ridge": original},
        )
        second = _build_policy_forecast_context(
            close=close,
            source_name="Ensemble",
            result=changed,
            components={"Ridge": changed},
        )

        pd.testing.assert_frame_equal(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertEqual(
            list(first[0].index),
            list(close.index[-33:-30]),
        )
        self.assertFalse(first[2]["actual_outcomes_used"])

    def test_comparison_builds_ensemble_before_passing_one_fixed_context_to_rl(
        self,
    ):
        component = _forecast_result(
            "direct model",
            forecast_change_pct=5.0,
            probability_up_pct=65.0,
            expected_error_pct=2.0,
            holdout_samples=3,
            holdout_predicted_return_pct=[1.0, 2.0, 3.0],
            holdout_actual_return_pct=[0.5, 1.5, 2.5],
            holdout_probability_up=[0.55, 0.60, 0.65],
            holdout_expected_error_pct=[2.0, 2.0, 2.0],
            holdout_model_agreement=[1.0, 1.0, 1.0],
            validation_is_oos=True,
            validation_scheme="nested_purged_outer_holdout",
            live_eligible=True,
        )
        shadow = _forecast_result(
            "shadow",
            shadow_mode=True,
            live_eligible=False,
        )

        with (
            patch.object(
                forecast_module,
                "forecast_close_prices",
                return_value=component,
            ),
            patch.object(
                forecast_module,
                "reinforcement_policy_forecast",
                return_value=shadow,
            ) as rl_forecast,
        ):
            results = forecast_module.compare_forecast_models(
                _price_frame(rising=True),
                horizon_days=30,
                include_rl=True,
                warm_start=False,
            )

        passed = rl_forecast.call_args.kwargs
        self.assertEqual(
            passed["forecast_context_metadata"]["source_model"],
            "Ensemble",
        )
        self.assertNotIn(
            "RL Policy",
            passed["forecast_context_metadata"]["component_models"],
        )
        self.assertIn("forecast_context_df", passed)
        self.assertIn("latest_forecast_context", passed)
        self.assertLess(
            list(results).index("Ensemble"),
            list(results).index("RL Policy"),
        )


class ShadowRlContributionTests(unittest.TestCase):
    def test_unverified_live_position_uses_auditable_shadow_tail(self):
        prices = _price_frame(rising=True)
        forecast_context = _policy_forecast_context(prices.index)
        forced_long = ShadowDecision(
            action_fraction=1.0,
            target_exposure=0.05,
            state=tuple(),
            state_visits=10,
            action_visits=10,
            q_values=(0.0, 0.01, 0.02, 0.03),
            abstained=False,
            reason="test shadow decision",
        )
        with patch.object(
            forecast_module.ShadowTargetWeightPolicy,
            "decide",
            return_value=forced_long,
        ):
            result = reinforcement_policy_forecast(
                prices,
                horizon_days=30,
                lookback_window=20,
                warm_start=False,
                forecast_context_df=forecast_context,
                latest_forecast_context=(
                    forecast_context.iloc[-1].to_dict()
                ),
                forecast_context_metadata={
                    "source_model": "Ensemble",
                    "context_version": "fixed-non-rl-oos-v1-h30",
                    "validation_scheme": "test_oos",
                    "component_models": ["Ridge", "XGBoost"],
                    "horizon_sessions": 30,
                },
            )

        self.assertEqual(result.metrics["rl_action"], "long")
        self.assertEqual(result.metrics["rl_target_weight"], 0.05)
        self.assertEqual(result.metrics["forecast_change_pct"], 0.0)
        self.assertFalse(result.metrics["rl_position_state_verified"])
        self.assertEqual(
            result.metrics["rl_position_state_source"],
            "frozen_oos_shadow_tail",
        )
        self.assertTrue(result.metrics["rl_position_state_auditable"])
        self.assertFalse(result.metrics["rl_live_allocation_enabled"])
        self.assertFalse(result.metrics["live_eligible"])
        self.assertEqual(
            result.metrics["policy_version"],
            "rl-shadow-contextual-v3-h30",
        )
        self.assertEqual(
            result.metrics["policy_execution_horizon_sessions"],
            1,
        )
        self.assertEqual(
            result.metrics["policy_execution_start_session"],
            str(
                forecast_module._future_index(
                    prices.index,
                    2,
                )[0].date()
            ),
        )
        self.assertEqual(
            result.metrics["policy_execution_target_session"],
            str(
                forecast_module._future_index(
                    prices.index,
                    2,
                )[1].date()
            ),
        )
        self.assertEqual(
            result.metrics["forecast_context_horizon_sessions"],
            30,
        )

        with patch.object(
            forecast_module.ShadowTargetWeightPolicy,
            "decide",
            return_value=forced_long,
        ):
            one_day = reinforcement_policy_forecast(
                prices,
                horizon_days=1,
                lookback_window=20,
                warm_start=False,
                forecast_context_df=forecast_context,
                latest_forecast_context=(
                    forecast_context.iloc[-1].to_dict()
                ),
                forecast_context_metadata={
                    "source_model": "Ensemble",
                    "context_version": "fixed-non-rl-oos-v1-h1",
                    "validation_scheme": "test_oos",
                    "component_models": ["Ridge", "XGBoost"],
                    "horizon_sessions": 1,
                },
            )
        self.assertEqual(
            one_day.metrics["policy_version"],
            "rl-shadow-contextual-v3-h1",
        )
        self.assertNotEqual(
            one_day.metrics["policy_version"],
            result.metrics["policy_version"],
        )
        self.assertEqual(
            one_day.metrics["policy_execution_horizon_sessions"],
            1,
        )

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

    def test_fresh_earnings_is_used_once_and_adverse_event_blocks_buy(self):
        primary_metrics = {
            "forecast_change_pct": 10.0,
            "expected_error_pct": 2.0,
            "confidence_pct": 70.0,
            "holdout_mae_pct": 2.0,
            "holdout_samples": 20,
            "holdout_direction_accuracy": 60.0,
            "calibration_error_pct": 10.0,
            "brier_score": 0.20,
            "validation_is_oos": True,
            "horizon_days": 30,
        }
        model_results = {
            name: _forecast_result(
                name,
                forecast_change_pct=return_pct,
                holdout_mae_pct=2.0,
                holdout_samples=20,
                holdout_direction_accuracy=60.0,
                calibration_error_pct=10.0,
                brier_score=0.20,
                validation_is_oos=True,
                horizon_days=30,
            )
            for name, return_pct in (("Ridge", 10.0), ("XGBoost", 8.0))
        }
        baseline = smart_policy_decision(
            df=_price_frame(rising=True),
            equity=100_000.0,
            risk_fraction=0.05,
            forecast_metrics=primary_metrics,
            model_results=model_results,
        )
        positive = smart_policy_decision(
            df=_price_frame(rising=True),
            equity=100_000.0,
            risk_fraction=0.05,
            forecast_metrics={
                **primary_metrics,
                "earnings_event_flag": True,
                "earnings_event_score": 0.8,
                "earnings_confidence": 0.75,
                "earnings_policy_eligible": True,
            },
            model_results=model_results,
        )
        adverse = smart_policy_decision(
            df=_price_frame(rising=True),
            equity=100_000.0,
            risk_fraction=0.05,
            forecast_metrics={
                **primary_metrics,
                "earnings_event_flag": True,
                "earnings_event_score": -0.8,
                "earnings_confidence": 0.75,
                "earnings_policy_eligible": True,
            },
            model_results=model_results,
        )

        self.assertGreater(baseline.target_position_fraction, 0.0)
        self.assertGreater(positive.score, baseline.score)
        self.assertEqual(
            positive.diagnostics["earnings"]["component"],
            0.5,
        )
        self.assertEqual(adverse.target_position_fraction, 0.0)
        self.assertIn(
            "adverse_earnings_event",
            adverse.diagnostics["allocation_blockers"],
        )

    def test_unqualified_earnings_context_cannot_increase_live_policy_score(self):
        primary_metrics = {
            "forecast_change_pct": 10.0,
            "expected_error_pct": 2.0,
            "confidence_pct": 70.0,
            "holdout_mae_pct": 2.0,
            "holdout_samples": 20,
            "holdout_direction_accuracy": 60.0,
            "calibration_error_pct": 10.0,
            "brier_score": 0.20,
            "validation_is_oos": True,
            "horizon_days": 30,
        }
        model_results = {
            name: _forecast_result(
                name,
                forecast_change_pct=return_pct,
                holdout_mae_pct=2.0,
                holdout_samples=20,
                holdout_direction_accuracy=60.0,
                calibration_error_pct=10.0,
                brier_score=0.20,
                validation_is_oos=True,
                horizon_days=30,
            )
            for name, return_pct in (("Ridge", 10.0), ("XGBoost", 8.0))
        }
        baseline = smart_policy_decision(
            df=_price_frame(rising=True),
            equity=100_000.0,
            risk_fraction=0.05,
            forecast_metrics=primary_metrics,
            model_results=model_results,
        )
        unqualified = smart_policy_decision(
            df=_price_frame(rising=True),
            equity=100_000.0,
            risk_fraction=0.05,
            forecast_metrics={
                **primary_metrics,
                "earnings_event_flag": True,
                "earnings_event_score": 1.0,
                "earnings_confidence": 1.0,
            },
            model_results=model_results,
        )

        self.assertEqual(unqualified.score, baseline.score)
        self.assertEqual(
            unqualified.diagnostics["earnings"]["component"],
            0.0,
        )


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

    def test_published_buy_requires_allocation_eligibility_and_nonzero_target(self):
        report = self._report_module()
        args = SimpleNamespace(min_signal_return_pct=2.0)
        base = {
            "Selected Model": "Ridge",
            "Model Call": "Buy",
            "Forecast Return %": 8.0,
            "Policy Target %": 5.0,
            "Policy Allocation Eligible": True,
            "Reliability": "High",
        }

        self.assertTrue(report.is_threshold_buy(base, args))
        self.assertFalse(
            report.is_threshold_buy(
                {**base, "Policy Allocation Eligible": False},
                args,
            )
        )
        self.assertFalse(
            report.is_threshold_buy(
                {**base, "Policy Target %": 0.0},
                args,
            )
        )

    def test_report_row_surfaces_immediate_earnings_interpretation(self):
        text = self._report_module().format_row(
            {
                "Symbol": "MU",
                "Forecast Return %": 8.0,
                "Model Call": "Buy",
                "Selected Model": "Ridge",
                "Probability Up %": 70.0,
                "Model Edge %": 20.0,
                "Expected Error %": 2.0,
                "Earnings Event": True,
                "Earnings Score": 0.8,
                "Earnings Confidence %": 75.0,
                "Earnings Summary": "EPS and revenue beat expectations.",
            }
        )

        self.assertIn("earnings +0.80 @ 75%", text)
        self.assertIn("EPS and revenue beat expectations.", text)


if __name__ == "__main__":
    unittest.main()
