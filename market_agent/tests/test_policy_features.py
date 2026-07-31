from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from market_agent.agent.policy_features import build_shadow_policy_data


def _prices(rows: int = 180) -> pd.DataFrame:
    index = pd.date_range("2025-01-02", periods=rows, freq="B")
    close = 100.0 * np.exp(np.linspace(0.0, 0.30, rows))
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.linspace(1_000_000, 1_400_000, rows),
        },
        index=index,
    )


def _context(index: pd.Index) -> pd.DataFrame:
    base = np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            "context_SPY": 400.0 + base,
            "context_QQQ": 350.0 + 1.2 * base,
            "context_^VIX": 18.0 + np.sin(base / 10.0),
            "context_TLT": 90.0 - 0.02 * base,
            "context_XLK": 180.0 + 0.8 * base,
            "context_XLF": 40.0 + 0.05 * base,
            "context_XLE": 80.0 - 0.03 * base,
            "context_XLV": 120.0 + 0.02 * base,
            "context_XLY": 150.0 + 0.1 * base,
        },
        index=index,
    )


def _forecast_context(index: pd.Index) -> pd.DataFrame:
    forecast_return = np.linspace(-0.02, 0.04, len(index))
    uncertainty = np.full(len(index), 0.015)
    return pd.DataFrame(
        {
            "forecast_return": forecast_return,
            "forecast_probability_up": np.linspace(0.40, 0.70, len(index)),
            "forecast_lower_bound": (
                forecast_return - 1.645 * uncertainty
            ),
            "forecast_model_agreement": np.full(len(index), 0.75),
            "forecast_uncertainty": uncertainty,
        },
        index=index,
    )


def _latest_forecast_context(frame: pd.DataFrame) -> dict[str, float]:
    return {
        column: float(value)
        for column, value in frame.iloc[-1].items()
    }


class PolicyFeatureTests(unittest.TestCase):
    def test_horizon_policies_are_separate_but_use_daily_execution_labels(self) -> None:
        prices = _prices()
        context = _context(prices.index)
        forecast_context = _forecast_context(prices.index)
        one_day = build_shadow_policy_data(
            prices,
            context_df=context,
            horizon_days=1,
            symbol="MU",
            forecast_context_df=forecast_context,
            latest_forecast_context=_latest_forecast_context(
                forecast_context
            ),
        )
        thirty_day = build_shadow_policy_data(
            prices,
            context_df=context,
            horizon_days=30,
            symbol="MU",
            forecast_context_df=forecast_context,
            latest_forecast_context=_latest_forecast_context(
                forecast_context
            ),
        )

        self.assertEqual(one_day.horizon_days, 1)
        self.assertEqual(thirty_day.horizon_days, 30)
        self.assertEqual(len(one_day.training_frame), len(thirty_day.training_frame))
        self.assertEqual(one_day.as_of, prices.index[-1])
        self.assertEqual(thirty_day.as_of, prices.index[-1])
        self.assertEqual(thirty_day.training_frame.index[-1], prices.index[-3])
        self.assertTrue(
            one_day.training_frame["forward_asset_return"].equals(
                thirty_day.training_frame["forward_asset_return"]
            )
        )
        self.assertEqual(one_day.latest_context["decision_horizon"], 1.0)
        self.assertEqual(thirty_day.latest_context["decision_horizon"], 30.0)
        self.assertEqual(thirty_day.execution_lag_sessions, 1)

    def test_decision_features_do_not_change_when_only_future_prices_change(self) -> None:
        prices = _prices()
        context = _context(prices.index)
        forecast_context = _forecast_context(prices.index)
        cutoff = prices.index[120]
        original = build_shadow_policy_data(
            prices.loc[:cutoff],
            context_df=context.loc[:cutoff],
            horizon_days=30,
            symbol="MU",
            forecast_context_df=forecast_context.loc[:cutoff],
            latest_forecast_context=_latest_forecast_context(
                forecast_context.loc[:cutoff]
            ),
        )

        changed = prices.copy()
        changed.loc[changed.index > cutoff, "close"] *= 3.0
        replay = build_shadow_policy_data(
            changed.loc[:cutoff],
            context_df=context.loc[:cutoff],
            horizon_days=30,
            symbol="MU",
            forecast_context_df=forecast_context.loc[:cutoff],
            latest_forecast_context=_latest_forecast_context(
                forecast_context.loc[:cutoff]
            ),
        )

        self.assertEqual(original.latest_context, replay.latest_context)

    def test_market_position_and_event_context_is_present(self) -> None:
        prices = _prices()
        context = _context(prices.index)
        forecast_context = _forecast_context(prices.index)
        events = pd.Series(0.0, index=prices.index)
        events.iloc[-1] = 1.0
        portfolio_returns = pd.Series(
            np.sin(np.arange(len(prices)) / 7.0) / 100.0,
            index=prices.index,
        )
        result = build_shadow_policy_data(
            prices,
            context_df=context,
            horizon_days=30,
            symbol="MU",
            event_flags=events,
            latest_event_context={
                "event_flag": 1,
                "event_score": 0.75,
                "event_confidence": 0.80,
            },
            portfolio_returns=portfolio_returns,
            forecast_context_df=forecast_context,
            latest_forecast_context=_latest_forecast_context(
                forecast_context
            ),
            forecast_context_source="Ensemble",
            forecast_context_version="fixed-non-rl-oos-v1-h30",
        )

        self.assertEqual(result.latest_context["event_flag"], 1.0)
        self.assertEqual(result.latest_context["event_score"], 0.75)
        self.assertEqual(result.latest_context["event_confidence"], 0.80)
        self.assertIn("sector_breadth", result.latest_context)
        self.assertIn("sector_relative_strength", result.latest_context)
        self.assertTrue(np.isfinite(result.latest_context["portfolio_correlation"]))
        self.assertEqual(result.correlation_source, "supplied_portfolio")
        self.assertEqual(result.sector_proxy, "XLK")
        self.assertEqual(result.forecast_context_source, "Ensemble")
        self.assertEqual(
            result.latest_context["forecast_probability_up"],
            forecast_context["forecast_probability_up"].iloc[-1],
        )

    def test_forecast_context_is_not_carried_across_missing_dates(self) -> None:
        prices = _prices()
        context = _context(prices.index)
        complete = _forecast_context(prices.index)
        supplied_dates = prices.index[[70, 105, 140]]
        sparse = complete.loc[supplied_dates]

        result = build_shadow_policy_data(
            prices,
            context_df=context,
            horizon_days=30,
            symbol="MU",
            forecast_context_df=sparse,
            latest_forecast_context=_latest_forecast_context(complete),
            forecast_context_source="Ridge",
            forecast_context_version="test-oos",
        )

        self.assertEqual(
            list(result.training_frame.index),
            list(supplied_dates),
        )
        self.assertEqual(result.forecast_context_samples, len(supplied_dates))
        self.assertEqual(
            result.latest_context["forecast_return"],
            float(complete["forecast_return"].iloc[-1]),
        )
        self.assertNotEqual(
            result.latest_context["forecast_return"],
            float(sparse["forecast_return"].iloc[-1]),
        )


if __name__ == "__main__":
    unittest.main()
