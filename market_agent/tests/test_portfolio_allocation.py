from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from market_agent.agent.portfolio import (
    PortfolioConstraints,
    allocate_target_weights,
    constrain_incremental_target,
)


class PortfolioAllocationTests(unittest.TestCase):
    def test_applies_name_sector_cluster_gross_and_cash_limits(self) -> None:
        symbols = [f"T{i:02d}" for i in range(24)]
        proposed = {symbol: 0.20 for symbol in symbols}
        sectors = {
            symbol: f"Sector{index // 4}"
            for index, symbol in enumerate(symbols)
        }
        clusters = {
            symbol: f"Cluster{index // 3}"
            for index, symbol in enumerate(symbols)
        }
        constraints = PortfolioConstraints(
            cash_reserve=0.15,
            max_name_weight=0.05,
            max_sector_weight=0.20,
            max_cluster_weight=0.12,
            max_annual_volatility=None,
            max_turnover=None,
        )

        result = allocate_target_weights(
            proposed,
            sectors=sectors,
            correlation_clusters=clusters,
            constraints=constraints,
        )

        self.assertLessEqual(result.gross_exposure, 0.85 + 1e-12)
        self.assertGreaterEqual(result.cash_weight, 0.15 - 1e-12)
        self.assertTrue(
            all(weight <= 0.05 + 1e-12 for weight in result.target_weights.values())
        )
        self.assertTrue(
            all(exposure <= 0.20 + 1e-12 for exposure in result.sector_exposures.values())
        )
        self.assertTrue(
            all(exposure <= 0.12 + 1e-12 for exposure in result.cluster_exposures.values())
        )
        self.assertIn("name", result.binding_constraints)
        self.assertIn("gross_and_cash_reserve", result.binding_constraints)

    def test_does_not_scale_small_requests_up_to_fill_budget(self) -> None:
        result = allocate_target_weights(
            {"AAPL": 0.02, "MSFT": 0.01},
            sectors={"AAPL": "Technology", "MSFT": "Technology"},
            correlation_clusters={"AAPL": "MegaCap", "MSFT": "MegaCap"},
            constraints=PortfolioConstraints(
                max_annual_volatility=None,
                max_turnover=None,
            ),
        )

        self.assertAlmostEqual(result.target_weights["AAPL"], 0.02)
        self.assertAlmostEqual(result.target_weights["MSFT"], 0.01)
        self.assertAlmostEqual(result.gross_exposure, 0.03)
        self.assertAlmostEqual(result.cash_weight, 0.97)

    def test_scales_to_annual_volatility_limit(self) -> None:
        covariance = pd.DataFrame(
            [[0.04, 0.0], [0.0, 0.04]],
            index=["AAPL", "MSFT"],
            columns=["AAPL", "MSFT"],
        )
        max_volatility = 0.01

        result = allocate_target_weights(
            {"AAPL": 0.05, "MSFT": 0.05},
            sectors={"AAPL": "Hardware", "MSFT": "Software"},
            correlation_clusters={"AAPL": "AAPL", "MSFT": "MSFT"},
            annual_covariance=covariance,
            constraints=PortfolioConstraints(
                max_annual_volatility=max_volatility,
                max_turnover=None,
            ),
        )

        self.assertAlmostEqual(result.annualized_volatility or 0.0, max_volatility)
        self.assertIn("annual_volatility", result.binding_constraints)
        expected_weight = 0.05 / np.sqrt(2.0)
        self.assertAlmostEqual(result.target_weights["AAPL"], expected_weight)
        self.assertAlmostEqual(result.target_weights["MSFT"], expected_weight)

    def test_limits_turnover_by_interpolating_from_current_weights(self) -> None:
        result = allocate_target_weights(
            {"MSFT": 0.05},
            current_weights={"AAPL": 0.02},
            sectors={"AAPL": "Hardware", "MSFT": "Software"},
            correlation_clusters={"AAPL": "AAPL", "MSFT": "MSFT"},
            constraints=PortfolioConstraints(
                max_annual_volatility=None,
                max_turnover=0.035,
            ),
        )

        self.assertAlmostEqual(result.turnover, 0.035)
        self.assertAlmostEqual(result.target_weights["AAPL"], 0.01)
        self.assertAlmostEqual(result.target_weights["MSFT"], 0.025)
        self.assertFalse(result.turnover_cap_overridden)
        self.assertIn("turnover", result.binding_constraints)

    def test_drawdown_breaker_moves_to_cash_and_overrides_turnover(self) -> None:
        result = allocate_target_weights(
            {"AAPL": 0.05},
            current_weights={"AAPL": 0.04},
            sectors={"AAPL": "Technology"},
            correlation_clusters={"AAPL": "MegaCap"},
            current_drawdown=-0.10,
            constraints=PortfolioConstraints(
                max_annual_volatility=None,
                max_turnover=0.01,
                drawdown_circuit_breaker=0.10,
            ),
        )

        self.assertTrue(result.circuit_breaker_triggered)
        self.assertTrue(result.turnover_cap_overridden)
        self.assertEqual(result.target_weights, {"AAPL": 0.0})
        self.assertAlmostEqual(result.cash_weight, 1.0)
        self.assertAlmostEqual(result.turnover, 0.04)
        self.assertIn("drawdown_circuit_breaker", result.binding_constraints)
        self.assertTrue(any("overrides max_turnover" in item for item in result.warnings))

    def test_hard_cap_overrides_turnover_for_unsafe_current_position(self) -> None:
        result = allocate_target_weights(
            {"AAPL": 0.0},
            current_weights={"AAPL": 0.20},
            sectors={"AAPL": "Technology"},
            correlation_clusters={"AAPL": "MegaCap"},
            constraints=PortfolioConstraints(
                max_annual_volatility=None,
                max_turnover=0.01,
            ),
        )

        self.assertLessEqual(result.target_weights["AAPL"], 0.05)
        self.assertGreater(result.turnover, 0.01)
        self.assertTrue(result.turnover_cap_overridden)
        self.assertTrue(any("Hard exposure" in item for item in result.warnings))

    def test_missing_classification_is_reported_without_guessing(self) -> None:
        result = allocate_target_weights(
            {"AAPL": 0.04, "UNKNOWN": 0.04},
            sectors={"AAPL": "Technology"},
            correlation_clusters={"AAPL": "MegaCap"},
            constraints=PortfolioConstraints(
                max_annual_volatility=None,
                max_turnover=None,
            ),
        )

        self.assertIn("Unclassified:UNKNOWN", result.sector_exposures)
        self.assertIn("Unclassified:UNKNOWN", result.cluster_exposures)
        self.assertTrue(
            any("Missing sector classification" in item for item in result.warnings)
        )
        self.assertTrue(
            any(
                "Missing correlation cluster classification" in item
                for item in result.warnings
            )
        )

    def test_rejects_invalid_constraint_ranges_and_covariance(self) -> None:
        with self.assertRaisesRegex(ValueError, "cash_reserve"):
            PortfolioConstraints(cash_reserve=0.25)
        with self.assertRaisesRegex(ValueError, "max_name_weight"):
            PortfolioConstraints(max_name_weight=0.10)

        nonsymmetric = pd.DataFrame(
            [[0.04, 0.01], [0.0, 0.04]],
            index=["AAPL", "MSFT"],
            columns=["AAPL", "MSFT"],
        )
        with self.assertRaisesRegex(ValueError, "symmetric"):
            allocate_target_weights(
                {"AAPL": 0.03, "MSFT": 0.03},
                annual_covariance=nonsymmetric,
            )

    def test_rejects_negative_and_nonfinite_weights(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            allocate_target_weights({"AAPL": -0.01})
        with self.assertRaisesRegex(ValueError, "finite"):
            allocate_target_weights({"AAPL": np.nan})
        with self.assertRaisesRegex(ValueError, "signed fraction"):
            allocate_target_weights({"AAPL": 0.01}, current_drawdown=0.10)

    def test_empty_portfolio_is_valid(self) -> None:
        covariance = pd.DataFrame(dtype=float)

        result = allocate_target_weights(
            {},
            annual_covariance=covariance,
            constraints=PortfolioConstraints(max_turnover=None),
        )

        self.assertEqual(result.target_weights, {})
        self.assertEqual(result.gross_exposure, 0.0)
        self.assertEqual(result.cash_weight, 1.0)
        self.assertEqual(result.annualized_volatility, 0.0)
        self.assertEqual(result.warnings, ())

    def test_incremental_target_respects_unchanged_portfolio_capacity(self) -> None:
        current = {
            **{f"T{i:02d}": 0.05 for i in range(16)},
            "SNDK": 0.01,
        }
        sectors = {
            **{f"T{i:02d}": f"Sector{i // 4}" for i in range(16)},
            "SNDK": "Storage",
        }
        clusters = {
            **{f"T{i:02d}": f"Cluster{i // 3}" for i in range(16)},
            "SNDK": "Memory",
        }

        result = constrain_incremental_target(
            "SNDK",
            0.05,
            current_weights=current,
            sectors=sectors,
            correlation_clusters=clusters,
            annual_covariance=None,
            constraints=PortfolioConstraints(
                max_annual_volatility=None,
                max_turnover=0.20,
            ),
        )

        self.assertAlmostEqual(result.gross_before, 0.81)
        self.assertAlmostEqual(result.allowed_target, 0.05)
        self.assertAlmostEqual(result.gross_after, 0.85)
        self.assertGreaterEqual(1.0 - result.gross_after, 0.15 - 1e-12)

    def test_incremental_target_cannot_assume_other_sector_sales(self) -> None:
        result = constrain_incremental_target(
            "SNDK",
            0.05,
            current_weights={"MU": 0.15},
            sectors={"SNDK": "Semiconductors", "MU": "Semiconductors"},
            correlation_clusters={"SNDK": "Memory", "MU": "Memory"},
            annual_covariance=None,
            constraints=PortfolioConstraints(
                max_annual_volatility=None,
                max_turnover=None,
            ),
        )

        self.assertEqual(result.allowed_target, 0.0)
        self.assertAlmostEqual(result.gross_after, 0.15)

    def test_incremental_target_respects_volatility_and_drawdown(self) -> None:
        covariance = pd.DataFrame(
            [[1.0]],
            index=["SNDK"],
            columns=["SNDK"],
        )
        constrained = constrain_incremental_target(
            "SNDK",
            0.05,
            current_weights={},
            sectors={"SNDK": "Semiconductors"},
            correlation_clusters={"SNDK": "Memory"},
            annual_covariance=covariance,
            constraints=PortfolioConstraints(
                max_annual_volatility=0.01,
                max_turnover=None,
            ),
        )
        stopped = constrain_incremental_target(
            "SNDK",
            0.05,
            current_weights={},
            sectors={"SNDK": "Semiconductors"},
            correlation_clusters={"SNDK": "Memory"},
            annual_covariance=covariance,
            current_drawdown=-0.10,
            constraints=PortfolioConstraints(max_turnover=None),
        )

        self.assertAlmostEqual(constrained.allowed_target, 0.01)
        self.assertAlmostEqual(constrained.annualized_volatility or 0.0, 0.01)
        self.assertEqual(stopped.allowed_target, 0.0)


if __name__ == "__main__":
    unittest.main()
