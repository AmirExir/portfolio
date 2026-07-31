from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from market_agent.daily_ml_forecast_report import (
    _load_external_earnings_payload,
    append_prediction_records,
    apply_portfolio_constraints,
)
from market_agent.agent.earnings import us_equity_trading_sessions
from market_agent.agent.ledger import PredictionLedger


UTC = timezone.utc


def _args(output_dir: Path, current_weights_json: str) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=output_dir,
        portfolio_current_weights_json=current_weights_json,
        portfolio_cash_reserve_pct=15.0,
        portfolio_max_name_pct=5.0,
        portfolio_max_sector_pct=20.0,
        portfolio_max_cluster_pct=15.0,
        portfolio_max_volatility_pct=15.0,
        portfolio_max_turnover_pct=20.0,
        portfolio_drawdown_breaker_pct=10.0,
        portfolio_drawdown_pct=0.0,
        min_signal_return_pct=2.0,
    )


def _row() -> dict[str, object]:
    return {
        "Symbol": "SNDK",
        "Policy Target %": 5.0,
        "Policy Reason": "qualified research signal",
        "Smart Policy": "Buy",
        "Policy Score": 4.0,
        "Forecast Return %": 8.0,
        "Model Call": "Buy",
        "Reliability": "Moderate",
    }


def _history() -> dict[str, pd.Series]:
    index = pd.bdate_range("2025-01-02", periods=100)
    prices = 100.0 * np.exp(np.linspace(0.0, 0.08, len(index)))
    return {"SNDK": pd.Series(prices, index=index)}


class DailyPortfolioAuthorizationTests(unittest.TestCase):
    def test_rl_ledger_record_uses_delayed_daily_execution_window(self) -> None:
        as_of_session = date(2026, 1, 2)
        calendar = us_equity_trading_sessions(
            as_of=datetime(2026, 1, 2, 17, tzinfo=UTC),
            observed_sessions=(as_of_session,),
            future_session_count=35,
        )
        future_sessions = [
            session for session in calendar if session > as_of_session
        ]
        forecast_target = future_sessions[29]
        execution_start = future_sessions[0]
        execution_target = future_sessions[1]
        rl_metrics = {
            "shadow_mode": True,
            "policy_version": "rl-shadow-contextual-v3-h30",
            "model_version": "execution-aligned-contextual-shadow-v3-h30",
            "policy_feature_set_version": (
                "shadow-market-position-event-forecast-v3-h30"
            ),
            "policy_execution_start_session": (
                execution_start.isoformat()
            ),
            "policy_execution_target_session": (
                execution_target.isoformat()
            ),
            "policy_execution_horizon_sessions": 1,
            "policy_decision_refresh_sessions": 1,
            "forecast_context_horizon_sessions": 30,
            "forecast_context_source": "Ensemble",
            "forecast_context_version": "fixed-non-rl-oos-v1-h30",
            "rl_position_state_source": "frozen_oos_shadow_tail",
            "rl_position_state_as_of": as_of_session.isoformat(),
            "rl_position_state_auditable": True,
            "rl_live_allocation_enabled": False,
            "rl_target_weight": 0.05,
            "rl_action": "long",
            "rl_action_fraction": 1.0,
            "rl_state_visits": 12,
            "rl_abstained": False,
            "forecast_context_return": 0.12,
            "forecast_context_probability_up": 0.70,
            "forecast_context_lower_bound": 0.03,
            "forecast_context_model_agreement": 0.75,
            "forecast_context_uncertainty": 0.05,
        }
        snapshot = {
            "symbol": "SNDK",
            "as_of_session": as_of_session.isoformat(),
            "data_cutoff_utc": "2026-01-02T21:00:00Z",
            "models": {
                "Ensemble": {
                    "metrics": {
                        "validation_is_oos": True,
                    }
                },
                "RL Policy": {"metrics": rl_metrics},
            },
        }
        row = {
            "Symbol": "SNDK",
            "As Of Session": as_of_session.isoformat(),
            "Target Session": forecast_target.isoformat(),
            "Selected Model": "Ensemble",
            "Forecast Return %": 12.0,
            "Expected Error %": 5.0,
            "Probability Up %": 70.0,
            "Policy Target %": 5.0,
            "Policy Allocation Eligible": True,
        }

        with tempfile.TemporaryDirectory() as directory:
            result = append_prediction_records(
                output_dir=Path(directory),
                rows=[row],
                snapshots=[snapshot],
                horizon_days=30,
                created_at_utc=datetime(
                    2026,
                    1,
                    2,
                    22,
                    tzinfo=UTC,
                ),
            )
            predictions = PredictionLedger(
                Path(directory) / "prediction_ledger.jsonl"
            ).predictions()

        self.assertEqual(result["appended"], 3)
        rl_record = next(
            record
            for record in predictions
            if record.model_name == "RL Policy"
        )
        self.assertEqual(rl_record.as_of_session, as_of_session)
        self.assertEqual(
            rl_record.return_start_session,
            execution_start,
        )
        self.assertEqual(rl_record.target_session, execution_target)
        self.assertEqual(rl_record.horizon_sessions, 1)
        self.assertEqual(
            rl_record.metadata[
                "forecast_context_horizon_sessions"
            ],
            30,
        )
        context_record = next(
            record
            for record in predictions
            if record.model_name == "RL Forecast Context"
        )
        self.assertEqual(context_record.horizon_sessions, 30)
        self.assertEqual(
            context_record.target_session,
            forecast_target,
        )
        self.assertAlmostEqual(
            context_record.forecast_return,
            0.12,
        )
        self.assertAlmostEqual(
            context_record.probability_positive or 0.0,
            0.70,
        )

    def test_verified_external_earnings_payload_is_loaded_by_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = {
                "symbol": "MU",
                "reported_at": "2026-07-29T16:05:00-04:00",
                "timestamp_quality": "provider_reported",
                "eps": {"actual": 1.10, "estimate": 1.00},
                "revenue": {
                    "actual": 9_000_000_000,
                    "estimate": 8_700_000_000,
                },
                "guidance": {"direction": "raised"},
            }
            Path(directory, "MU.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            loaded, error = _load_external_earnings_payload(
                "MU",
                SimpleNamespace(earnings_payload_dir=directory),
            )

        self.assertIsNone(error)
        self.assertEqual(loaded, payload)

    def test_external_earnings_symbol_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "MU.json").write_text(
                json.dumps(
                    {
                        "symbol": "SNDK",
                        "reported_at": "2026-07-29T16:05:00-04:00",
                    }
                ),
                encoding="utf-8",
            )
            loaded, error = _load_external_earnings_payload(
                "MU",
                SimpleNamespace(earnings_payload_dir=directory),
            )

        self.assertIsNone(loaded)
        self.assertEqual(error, "external_earnings_symbol_mismatch")

    def test_previous_recommendations_do_not_authorize_a_new_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows, diagnostics = apply_portfolio_constraints(
                [_row()],
                _history(),
                _args(Path(directory), ""),
            )

        self.assertEqual(rows[0]["Pre-Portfolio Target %"], 5.0)
        self.assertEqual(rows[0]["Policy Target %"], 0.0)
        self.assertFalse(rows[0]["Policy Allocation Eligible"])
        self.assertFalse(diagnostics["portfolio_state_verified"])
        self.assertIn(
            "unverified_portfolio_state",
            diagnostics["binding_constraints"],
        )

    def test_verified_weights_and_covariance_authorize_constrained_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows, diagnostics = apply_portfolio_constraints(
                [_row()],
                _history(),
                _args(Path(directory), "{}"),
            )

        self.assertTrue(rows[0]["Portfolio State Verified"])
        self.assertTrue(rows[0]["Portfolio Covariance Verified"])
        self.assertTrue(rows[0]["Policy Allocation Eligible"])
        self.assertGreater(rows[0]["Policy Target %"], 0.0)
        self.assertLessEqual(rows[0]["Policy Target %"], 5.0)
        self.assertTrue(diagnostics["allocation_eligible"])

    def test_missing_covariance_blocks_positive_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows, diagnostics = apply_portfolio_constraints(
                [_row()],
                {},
                _args(Path(directory), "{}"),
            )

        self.assertEqual(rows[0]["Policy Target %"], 0.0)
        self.assertFalse(rows[0]["Policy Allocation Eligible"])
        self.assertFalse(diagnostics["covariance_verified"])
        self.assertIn(
            "covariance_unavailable",
            diagnostics["binding_constraints"],
        )

    def test_missing_classification_blocks_positive_allocation(self) -> None:
        index = pd.bdate_range("2025-01-02", periods=100)
        history = {
            "UNKNOWN": pd.Series(
                np.linspace(100.0, 110.0, len(index)),
                index=index,
            )
        }
        row = {
            **_row(),
            "Symbol": "UNKNOWN",
        }
        with tempfile.TemporaryDirectory() as directory:
            rows, diagnostics = apply_portfolio_constraints(
                [row],
                history,
                _args(Path(directory), "{}"),
            )

        self.assertEqual(rows[0]["Policy Target %"], 0.0)
        self.assertFalse(rows[0]["Policy Allocation Eligible"])
        self.assertFalse(
            rows[0]["Portfolio Classification Verified"]
        )
        self.assertIn(
            "classification_unavailable",
            diagnostics["binding_constraints"],
        )


if __name__ == "__main__":
    unittest.main()
