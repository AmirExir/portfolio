"""Check that default run inputs can satisfy unchanged evidence-count gates."""

from __future__ import annotations

import datetime as dt
import unittest

import numpy as np
import pandas as pd

from market_agent import daily_ml_forecast_report as report
from market_agent.agent.earnings import us_equity_trading_sessions
from market_agent.agent.forecast import forecast_close_prices


def _history_frame(calendar_days: int) -> pd.DataFrame:
    """Build reproducible synthetic closes on actual exchange-session labels."""

    as_of = dt.datetime(2026, 11, 5, 22, tzinfo=dt.timezone.utc)
    sessions = us_equity_trading_sessions(
        as_of=as_of,
        lookback_days=calendar_days,
        future_session_count=1,
    )
    sessions = [session for session in sessions if session <= as_of.date()]
    rng = np.random.default_rng(721)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, len(sessions))))
    return pd.DataFrame({"close": close}, index=pd.DatetimeIndex(sessions))


class ProfileValidationHistoryTests(unittest.TestCase):
    def test_default_profiles_can_meet_nonoverlap_count_without_lowering_gates(
        self,
    ) -> None:
        minimum = report.RELIABILITY_POLICY_CONFIG.min_nonoverlapping_validation_samples
        self.assertEqual(minimum, 8)
        for profile in ("quick", "scheduled", "quality", "overnight"):
            with self.subTest(profile=profile):
                raw_args = ["--run-profile", profile, "--no-rl-policy"]
                args = report.apply_run_profile(
                    report.build_parser().parse_args(raw_args), raw_args
                )
                result = forecast_close_prices(
                    _history_frame(args.history_days),
                    horizon_days=args.horizon,
                    lookback_window=args.lookback,
                    ridge_alpha=args.ridge_alpha,
                    optimize_model=not args.no_optimize,
                    model_type="ridge",
                    symbol="TEST",
                )
                self.assertTrue(result.metrics["validation_is_oos"])
                self.assertGreaterEqual(
                    result.metrics["holdout_nonoverlapping_samples"], minimum
                )

    def test_explicit_short_history_is_preserved_and_exposes_evidence_shortfall(
        self,
    ) -> None:
        raw_args = ["--run-profile", "quick", "--history-days", "913"]
        args = report.apply_run_profile(
            report.build_parser().parse_args(raw_args), raw_args
        )
        self.assertEqual(args.history_days, 913)
        result = forecast_close_prices(
            _history_frame(args.history_days),
            horizon_days=args.horizon,
            lookback_window=args.lookback,
            ridge_alpha=args.ridge_alpha,
            optimize_model=not args.no_optimize,
            model_type="ridge",
            symbol="TEST",
        )
        self.assertEqual(result.metrics["holdout_nonoverlapping_samples"], 5)
        self.assertLess(
            result.metrics["holdout_nonoverlapping_samples"],
            report.RELIABILITY_POLICY_CONFIG.min_nonoverlapping_validation_samples,
        )


if __name__ == "__main__":
    unittest.main()
