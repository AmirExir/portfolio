from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from market_agent.agent.shadow_policy import (
    ACTION_FRACTIONS,
    CONTEXT_FEATURES,
    STATE_COMPONENTS,
    PolicyPosition,
    ShadowPolicyConfig,
    ShadowTargetWeightPolicy,
    StateDiscretizer,
    execution_aligned_reward,
)


def _policy_frame(rows: int = 100, asset_return: float = 0.01) -> pd.DataFrame:
    index = pd.date_range("2025-01-02", periods=rows, freq="B")
    return pd.DataFrame(
        {
            "decision_horizon": np.full(rows, 30.0),
            "asset_momentum": np.full(rows, 0.04),
            "asset_volatility": np.full(rows, 0.30),
            "spy_trend": np.full(rows, 0.01),
            "qqq_trend": np.full(rows, 0.015),
            "vix_level": np.full(rows, 18.0),
            "tlt_trend": np.full(rows, -0.005),
            "gap": np.zeros(rows),
            "volume_shock": np.full(rows, 0.20),
            "event_flag": np.zeros(rows),
            "event_score": np.zeros(rows),
            "event_confidence": np.zeros(rows),
            "sector_relative_strength": np.full(rows, 0.02),
            "sector_breadth": np.full(rows, 0.60),
            "forecast_return": np.full(rows, 0.08),
            "forecast_probability_up": np.full(rows, 0.65),
            "forecast_lower_bound": np.full(rows, 0.03),
            "forecast_model_agreement": np.full(rows, 0.75),
            "forecast_uncertainty": np.full(rows, 0.03),
            "portfolio_correlation": np.full(rows, 0.30),
            "forward_asset_return": np.full(rows, asset_return),
            "forward_benchmark_return": np.zeros(rows),
        },
        index=index,
    )


class RewardTests(unittest.TestCase):
    def test_reward_matches_execution_aligned_formula(self) -> None:
        result = execution_aligned_reward(
            exposure=0.05,
            previous_exposure=0.02,
            asset_return=0.02,
            benchmark_return=0.005,
            drawdown_increment=0.003,
            portfolio_risk=0.02,
            transaction_cost_bps=10.0,
            drawdown_penalty=0.10,
            portfolio_risk_penalty=0.01,
        )

        expected_active = 0.05 * (0.02 - 0.005)
        expected_cost = 10.0 / 10_000.0 * abs(0.05 - 0.02)
        expected_drawdown = 0.10 * 0.003
        expected_risk = 0.01 * 0.05 * 0.02
        self.assertAlmostEqual(result.active_return, expected_active)
        self.assertAlmostEqual(result.turnover_cost, expected_cost)
        self.assertAlmostEqual(result.drawdown_cost, expected_drawdown)
        self.assertAlmostEqual(result.portfolio_risk_cost, expected_risk)
        self.assertAlmostEqual(
            result.reward,
            expected_active - expected_cost - expected_drawdown - expected_risk,
        )

    def test_cost_applies_only_when_exposure_changes(self) -> None:
        unchanged = execution_aligned_reward(
            exposure=0.05,
            previous_exposure=0.05,
            asset_return=0.0,
            benchmark_return=0.0,
            drawdown_increment=0.0,
            portfolio_risk=0.0,
            transaction_cost_bps=10.0,
            drawdown_penalty=0.0,
            portfolio_risk_penalty=0.0,
        )
        changed = execution_aligned_reward(
            exposure=0.05,
            previous_exposure=0.0,
            asset_return=0.0,
            benchmark_return=0.0,
            drawdown_increment=0.0,
            portfolio_risk=0.0,
            transaction_cost_bps=10.0,
            drawdown_penalty=0.0,
            portfolio_risk_penalty=0.0,
        )

        self.assertEqual(unchanged.turnover_cost, 0.0)
        self.assertGreater(changed.turnover_cost, 0.0)


class StateTests(unittest.TestCase):
    def test_state_declares_all_required_market_and_position_inputs(self) -> None:
        required = {
            "current_exposure",
            "entry_drawdown",
            "holding_period",
            "previous_action",
            "decision_horizon",
            "asset_momentum",
            "asset_volatility",
            "spy_trend",
            "qqq_trend",
            "vix_level",
            "tlt_trend",
            "gap",
            "volume_shock",
            "event_flag",
            "event_score",
            "event_confidence",
            "sector_relative_strength",
            "sector_breadth",
            "forecast_return",
            "forecast_probability_up",
            "forecast_lower_bound",
            "forecast_model_agreement",
            "forecast_uncertainty",
            "portfolio_correlation",
        }
        self.assertEqual(set(STATE_COMPONENTS), required)
        self.assertTrue(set(CONTEXT_FEATURES).issubset(required))

    def test_each_state_input_can_change_its_corresponding_bin(self) -> None:
        discretizer = StateDiscretizer()
        context = _policy_frame(1).iloc[0].to_dict()
        base_position = PolicyPosition()
        base = discretizer.encode(
            context,
            base_position,
            risk_budget_fraction=0.05,
        )

        position_variants = {
            "current_exposure": PolicyPosition(exposure=0.05),
            "entry_drawdown": PolicyPosition(entry_drawdown=-0.06),
            "holding_period": PolicyPosition(holding_period=15),
            "previous_action": PolicyPosition(previous_action_fraction=1.0),
        }
        for name, position in position_variants.items():
            changed = discretizer.encode(
                context,
                position,
                risk_budget_fraction=0.05,
            )
            component = STATE_COMPONENTS.index(name)
            self.assertNotEqual(base[component], changed[component], name)

        context_variants = {
            "decision_horizon": 1.0,
            "asset_momentum": -0.20,
            "asset_volatility": 0.80,
            "spy_trend": -0.05,
            "qqq_trend": -0.05,
            "vix_level": 45.0,
            "tlt_trend": 0.05,
            "gap": 0.05,
            "volume_shock": 1.20,
            "event_flag": 1.0,
            "event_score": -0.80,
            "event_confidence": 0.90,
            "sector_relative_strength": -0.10,
            "sector_breadth": 0.10,
            "forecast_return": 0.20,
            "forecast_probability_up": 0.20,
            "forecast_lower_bound": -0.20,
            "forecast_model_agreement": 0.20,
            "forecast_uncertainty": 0.20,
            "portfolio_correlation": 0.90,
        }
        for name, value in context_variants.items():
            changed_context = dict(context)
            changed_context[name] = value
            changed = discretizer.encode(
                changed_context,
                base_position,
                risk_budget_fraction=0.05,
            )
            component = STATE_COMPONENTS.index(name)
            self.assertNotEqual(base[component], changed[component], name)

    def test_forward_returns_are_not_state_components(self) -> None:
        discretizer = StateDiscretizer()
        first = _policy_frame(1, asset_return=0.50).iloc[0].to_dict()
        second = dict(first)
        second["forward_asset_return"] = -0.50
        second["forward_benchmark_return"] = 0.20

        first_state = discretizer.encode(
            first,
            PolicyPosition(),
            risk_budget_fraction=0.05,
        )
        second_state = discretizer.encode(
            second,
            PolicyPosition(),
            risk_budget_fraction=0.05,
        )

        self.assertEqual(first_state, second_state)


class ShadowPolicyTests(unittest.TestCase):
    def _config(self, **overrides: object) -> ShadowPolicyConfig:
        values: dict[str, object] = {
            "training_epochs": 20,
            "exploration_rate": 0.30,
            "min_state_visits": 1,
            "min_action_visits": 1,
            "random_seed": 42,
        }
        values.update(overrides)
        return ShadowPolicyConfig(**values)

    def test_training_is_reproducible_with_fixed_seed(self) -> None:
        frame = _policy_frame(60)
        first = ShadowTargetWeightPolicy(self._config()).fit(frame)
        second = ShadowTargetWeightPolicy(self._config()).fit(frame)

        self.assertEqual(first.q_table_snapshot(), second.q_table_snapshot())
        self.assertEqual(first.visit_snapshot(), second.visit_snapshot())
        state_visits, action_visits = first.visit_snapshot()
        self.assertTrue(all(count <= len(frame) for count in state_visits.values()))
        self.assertTrue(all(count <= len(frame) for count in action_visits.values()))

    def test_evaluation_is_frozen_and_actions_are_target_budget_fractions(self) -> None:
        frame = _policy_frame(100)
        policy = ShadowTargetWeightPolicy(self._config()).fit(frame.iloc[:60])
        before_q = policy.q_table_snapshot()
        before_visits = policy.visit_snapshot()

        result = policy.evaluate(frame.iloc[60:])

        self.assertTrue(policy.is_frozen)
        self.assertEqual(before_q, policy.q_table_snapshot())
        self.assertEqual(before_visits, policy.visit_snapshot())
        self.assertTrue(set(result.decisions["action_fraction"]).issubset(ACTION_FRACTIONS))
        self.assertLessEqual(
            float(result.decisions["target_exposure"].max()),
            policy.config.risk_budget_fraction,
        )

    def test_unknown_and_low_visit_states_abstain_to_zero(self) -> None:
        frame = _policy_frame(20)
        low_visit_policy = ShadowTargetWeightPolicy(
            self._config(training_epochs=1, min_state_visits=100)
        ).fit(frame)
        low_visit = low_visit_policy.decide(frame.iloc[0].to_dict())
        self.assertTrue(low_visit.abstained)
        self.assertEqual(low_visit.target_exposure, 0.0)
        self.assertIn("below", low_visit.reason)

        changed_context = frame.iloc[0].to_dict()
        changed_context["vix_level"] = 50.0
        unseen = low_visit_policy.decide(changed_context)
        self.assertTrue(unseen.abstained)
        self.assertEqual(unseen.target_exposure, 0.0)
        self.assertEqual(unseen.reason, "unseen state")

    def test_tied_q_values_abstain_to_zero(self) -> None:
        zero_return_frame = _policy_frame(10, asset_return=0.0)
        config = self._config(
            training_epochs=10,
            exploration_rate=1.0,
            transaction_cost_bps=0.0,
            drawdown_penalty=0.0,
            portfolio_risk_penalty=0.0,
        )
        policy = ShadowTargetWeightPolicy(config).fit(zero_return_frame)
        decision = policy.decide(zero_return_frame.iloc[0].to_dict())

        self.assertTrue(decision.abstained)
        self.assertEqual(decision.target_exposure, 0.0)
        self.assertEqual(decision.reason, "top Q-values are tied")

    def test_rejects_nonchronological_or_unmatured_data(self) -> None:
        frame = _policy_frame(10)
        shuffled = frame.iloc[[1, 0, *range(2, len(frame))]]
        with self.assertRaisesRegex(ValueError, "chronological"):
            ShadowTargetWeightPolicy(self._config()).fit(shuffled)

        unmatured = frame.copy()
        unmatured.iloc[-1, unmatured.columns.get_loc("forward_asset_return")] = np.nan
        with self.assertRaisesRegex(ValueError, "matured"):
            ShadowTargetWeightPolicy(self._config()).fit(unmatured)

    def test_purged_holdout_requires_minimum_gap_and_stays_frozen(self) -> None:
        frame = _policy_frame(110)
        policy = ShadowTargetWeightPolicy(self._config(minimum_purge_sessions=30))

        with self.assertRaisesRegex(ValueError, "purged sessions"):
            policy.fit_purged_holdout(
                frame,
                train_end=frame.index[39],
                test_start=frame.index[69],
            )
        with self.assertRaisesRegex(ValueError, "integer"):
            policy.fit_purged_holdout(
                frame,
                train_end=frame.index[39],
                test_start=frame.index[70],
                purge_sessions=30.5,
            )

        result = policy.fit_purged_holdout(
            frame,
            train_end=frame.index[39],
            test_start=frame.index[70],
        )

        self.assertEqual(result.purge_sessions, 30)
        self.assertEqual(result.training_end, frame.index[39])
        self.assertEqual(result.evaluation_start, frame.index[70])
        self.assertTrue(policy.is_frozen)
        self.assertEqual(len(result.decisions), 40)


if __name__ == "__main__":
    unittest.main()
