from __future__ import annotations

from datetime import date, datetime
import unittest
from zoneinfo import ZoneInfo

import pandas as pd

from market_agent.agent.earnings import (
    EarningsInterpretationConfig,
    effective_decision_session,
    fetch_yfinance_earnings_payload,
    interpret_earnings_payload,
    parse_earnings_payload,
    us_equity_trading_sessions,
)


EASTERN = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
SESSIONS = (
    date(2026, 7, 27),
    date(2026, 7, 28),
    date(2026, 7, 29),
    date(2026, 7, 30),
    date(2026, 7, 31),
    date(2026, 8, 3),
    date(2026, 8, 4),
)


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "symbol": "XYZ",
        "reported_at": "2026-07-29T16:05:00-04:00",
        "fiscal_period": "Q2 2026",
        "eps": {"actual": 1.10, "estimate": 1.00},
        "revenue": {"actual": 5_100.0, "estimate": 5_000.0},
        "guidance": {"direction": "raised"},
        "source": "test-provider",
    }
    payload.update(overrides)
    return payload


class EarningsInterpretationTests(unittest.TestCase):
    def test_positive_release_produces_bounded_policy_context(self) -> None:
        signal = interpret_earnings_payload(
            _payload(),
            as_of=datetime(2026, 7, 30, 8, 30, tzinfo=EASTERN),
            decision_session=date(2026, 7, 30),
            trading_sessions=SESSIONS,
        )

        self.assertEqual(signal.event_flag, 1)
        self.assertTrue(signal.policy_eligible)
        self.assertEqual(signal.outcome, "beat")
        self.assertGreater(signal.event_score, 0.0)
        self.assertLessEqual(signal.event_score, 1.0)
        self.assertGreater(signal.confidence, 0.0)
        self.assertLessEqual(signal.confidence, 1.0)
        self.assertAlmostEqual(signal.eps_surprise_pct or 0.0, 10.0)
        self.assertAlmostEqual(signal.revenue_surprise_pct or 0.0, 2.0)
        self.assertIn("EPS beat by 10.0%", signal.summary)
        self.assertEqual(
            signal.policy_context(),
            {
                "event_flag": 1.0,
                "event_score": signal.event_score,
                "event_confidence": signal.confidence,
                "earnings_event_flag": 1.0,
                "earnings_event_score": signal.event_score,
                "earnings_confidence": signal.confidence,
                "earnings_policy_eligible": True,
            },
        )

    def test_conflicting_eps_and_revenue_are_mixed(self) -> None:
        signal = interpret_earnings_payload(
            _payload(
                eps={"actual": 1.20, "estimate": 1.00},
                revenue={"actual": 4_500.0, "estimate": 5_000.0},
                guidance=None,
            ),
            as_of=datetime(2026, 7, 30, 8, 0, tzinfo=EASTERN),
            decision_session="2026-07-30",
            trading_sessions=SESSIONS,
        )

        self.assertEqual(signal.outcome, "mixed")
        self.assertLess(signal.confidence, 0.80)
        self.assertIn("revenue missed by 10.0%", signal.summary)

    def test_missing_one_estimate_uses_only_comparable_evidence(self) -> None:
        signal = interpret_earnings_payload(
            _payload(
                eps={"actual": 1.20},
                revenue={"actual": 5_100.0, "estimate": 5_000.0},
                guidance=None,
            ),
            as_of=datetime(2026, 7, 30, 8, 0, tzinfo=EASTERN),
            decision_session="2026-07-30",
            trading_sessions=SESSIONS,
        )

        self.assertEqual(signal.event_flag, 1)
        self.assertIsNone(signal.eps_surprise_pct)
        self.assertIn("eps_estimate_missing", signal.data_quality_flags)
        self.assertAlmostEqual(signal.confidence, 0.35)

    def test_missing_all_estimates_blocks_policy_use(self) -> None:
        signal = interpret_earnings_payload(
            _payload(
                eps={"actual": 1.20},
                revenue={"actual": 5_100.0},
                guidance=None,
            ),
            as_of=datetime(2026, 7, 30, 8, 0, tzinfo=EASTERN),
            decision_session="2026-07-30",
            trading_sessions=SESSIONS,
        )

        self.assertEqual(signal.event_flag, 0)
        self.assertFalse(signal.policy_eligible)
        self.assertEqual(signal.event_score, 0.0)
        self.assertEqual(signal.confidence, 0.0)
        self.assertIn("no_comparable_metrics", signal.blockers)
        self.assertIn("eps_estimate_missing", signal.data_quality_flags)
        self.assertIn("revenue_estimate_missing", signal.data_quality_flags)

    def test_zero_estimate_is_not_divided_and_is_flagged(self) -> None:
        signal = interpret_earnings_payload(
            _payload(
                eps={"actual": 0.10, "estimate": 0.0},
                revenue={"actual": 5_100.0, "estimate": 5_000.0},
                guidance=None,
            ),
            as_of=datetime(2026, 7, 30, 8, 0, tzinfo=EASTERN),
            decision_session="2026-07-30",
            trading_sessions=SESSIONS,
        )

        self.assertEqual(signal.event_flag, 1)
        self.assertIsNone(signal.eps_surprise_pct)
        self.assertIn("eps_estimate_zero", signal.data_quality_flags)
        self.assertTrue(-1.0 <= signal.event_score <= 1.0)

    def test_stale_event_is_zeroed(self) -> None:
        signal = interpret_earnings_payload(
            _payload(reported_at="2026-07-27T08:00:00-04:00"),
            as_of=datetime(2026, 7, 30, 8, 0, tzinfo=EASTERN),
            decision_session="2026-07-30",
            config=EarningsInterpretationConfig(
                max_event_age_sessions=2
            ),
            trading_sessions=SESSIONS,
        )

        self.assertTrue(signal.is_stale)
        self.assertEqual(signal.age_sessions, 3)
        self.assertEqual(signal.event_flag, 0)
        self.assertEqual(signal.event_score, 0.0)
        self.assertIn("stale_event", signal.blockers)
        self.assertEqual(
            signal.policy_context(),
            {
                "event_flag": 0.0,
                "event_score": 0.0,
                "event_confidence": 0.0,
                "earnings_event_flag": 0.0,
                "earnings_event_score": 0.0,
                "earnings_confidence": 0.0,
                "earnings_policy_eligible": False,
            },
        )

    def test_premarket_release_is_effective_same_session(self) -> None:
        event = parse_earnings_payload(
            _payload(reported_at="2026-07-29T08:00:00-04:00")
        )

        self.assertEqual(
            effective_decision_session(
                event.reported_at,
                trading_sessions=SESSIONS,
            ),
            date(2026, 7, 29),
        )

    def test_after_hours_and_regular_hours_use_next_session(self) -> None:
        after_hours = parse_earnings_payload(_payload())
        during_market = parse_earnings_payload(
            _payload(reported_at="2026-07-29T12:00:00-04:00")
        )

        self.assertEqual(
            effective_decision_session(
                after_hours.reported_at,
                trading_sessions=SESSIONS,
            ),
            date(2026, 7, 30),
        )
        self.assertEqual(
            effective_decision_session(
                during_market.reported_at,
                trading_sessions=SESSIONS,
            ),
            date(2026, 7, 30),
        )

    def test_utc_timestamp_is_converted_before_session_assignment(self) -> None:
        event = parse_earnings_payload(
            _payload(reported_at="2026-07-29T20:05:00Z")
        )

        self.assertEqual(
            effective_decision_session(
                event.reported_at,
                trading_sessions=SESSIONS,
            ),
            date(2026, 7, 30),
        )

    def test_explicit_sessions_skip_a_weekday_exchange_holiday(self) -> None:
        sessions = (
            date(2026, 7, 2),
            date(2026, 7, 6),
            date(2026, 7, 7),
        )
        event = parse_earnings_payload(
            _payload(reported_at="2026-07-02T16:05:00-04:00")
        )

        self.assertEqual(
            effective_decision_session(
                event.reported_at,
                trading_sessions=sessions,
            ),
            date(2026, 7, 6),
        )

    def test_us_equity_rule_sessions_skip_observed_independence_day(self) -> None:
        sessions = us_equity_trading_sessions(
            as_of=datetime(2026, 7, 2, 17, 0, tzinfo=EASTERN),
            observed_sessions=(date(2026, 7, 1), date(2026, 7, 2)),
        )
        event = parse_earnings_payload(
            _payload(reported_at="2026-07-02T16:05:00-04:00")
        )

        self.assertNotIn(date(2026, 7, 3), sessions)
        self.assertEqual(
            effective_decision_session(
                event.reported_at,
                trading_sessions=sessions,
            ),
            date(2026, 7, 6),
        )

    def test_future_publication_cannot_leak_actuals_or_summary(self) -> None:
        signal = interpret_earnings_payload(
            _payload(),
            as_of=datetime(2026, 7, 29, 15, 59, tzinfo=EASTERN),
            decision_session="2026-07-30",
            trading_sessions=SESSIONS,
        )

        self.assertEqual(signal.event_flag, 0)
        self.assertEqual(signal.event_score, 0.0)
        self.assertEqual(signal.confidence, 0.0)
        self.assertIsNone(signal.eps_surprise_pct)
        self.assertIsNone(signal.revenue_surprise_pct)
        self.assertNotIn("10.0%", signal.summary)
        self.assertIn("event_not_yet_published", signal.blockers)

    def test_known_after_hours_event_cannot_enter_prior_session(self) -> None:
        signal = interpret_earnings_payload(
            _payload(),
            as_of=datetime(2026, 7, 29, 17, 0, tzinfo=EASTERN),
            decision_session="2026-07-29",
            trading_sessions=SESSIONS,
        )

        self.assertEqual(signal.event_flag, 0)
        self.assertIsNone(signal.eps_surprise_pct)
        self.assertNotIn("10.0%", signal.summary)
        self.assertIn(
            "event_not_effective_for_decision_session",
            signal.blockers,
        )

    def test_naive_timestamp_requires_an_explicit_timezone(self) -> None:
        blocked = interpret_earnings_payload(
            _payload(reported_at="2026-07-29T16:05:00"),
            as_of=datetime(2026, 7, 30, 8, 0, tzinfo=EASTERN),
            decision_session="2026-07-30",
            trading_sessions=SESSIONS,
        )
        parsed = parse_earnings_payload(
            _payload(reported_at="2026-07-29T16:05:00"),
            default_timezone="America/New_York",
        )

        self.assertIn("invalid_payload", blocked.blockers)
        self.assertEqual(parsed.reported_at.tzinfo, EASTERN)

    def test_date_only_timestamp_is_rejected_to_avoid_session_guessing(self) -> None:
        signal = interpret_earnings_payload(
            _payload(
                reported_at="2026-07-29",
                timezone="America/New_York",
            ),
            as_of=datetime(2026, 7, 30, 8, 0, tzinfo=EASTERN),
            decision_session="2026-07-30",
            trading_sessions=SESSIONS,
        )

        self.assertEqual(signal.event_flag, 0)
        self.assertIn("invalid_payload", signal.blockers)

    def test_guidance_only_event_is_supported_with_lower_confidence(self) -> None:
        signal = interpret_earnings_payload(
            _payload(
                eps=None,
                revenue=None,
                guidance={"direction": "lowered"},
            ),
            as_of=datetime(2026, 7, 30, 8, 0, tzinfo=EASTERN),
            decision_session="2026-07-30",
            trading_sessions=SESSIONS,
        )

        self.assertEqual(signal.event_flag, 1)
        self.assertEqual(signal.outcome, "miss")
        self.assertLess(signal.event_score, 0.0)
        self.assertAlmostEqual(signal.confidence, 0.20)


class YFinanceAdapterTests(unittest.TestCase):
    def test_provider_failure_returns_unavailable(self) -> None:
        def failing_factory(symbol: str) -> object:
            raise RuntimeError(f"unavailable for {symbol}")

        result = fetch_yfinance_earnings_payload(
            "XYZ",
            as_of=datetime(2026, 7, 30, 8, 0, tzinfo=EASTERN),
            ticker_factory=failing_factory,
        )

        self.assertFalse(result.available)
        self.assertEqual(result.error_code, "provider_error:RuntimeError")
        self.assertIsNone(result.payload)

    def test_injected_provider_filters_future_and_unreported_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "EPS Estimate": [1.30, 1.00, 0.90],
                "Reported EPS": [float("nan"), 1.10, 0.95],
            },
            index=pd.DatetimeIndex(
                [
                    datetime(2026, 10, 29, 16, 0, tzinfo=EASTERN),
                    datetime(2026, 7, 29, 16, 0, tzinfo=EASTERN),
                    datetime(2026, 4, 29, 16, 0, tzinfo=EASTERN),
                ]
            ),
        )

        class FakeTicker:
            def get_earnings_dates(self, *, limit: int) -> pd.DataFrame:
                self.limit = limit
                return frame

        result = fetch_yfinance_earnings_payload(
            "xyz",
            as_of=datetime(2026, 7, 30, 8, 0, tzinfo=EASTERN),
            ticker_factory=lambda symbol: FakeTicker(),
        )

        self.assertTrue(result.available)
        self.assertIsNotNone(result.payload)
        assert result.payload is not None
        self.assertEqual(result.payload["symbol"], "XYZ")
        self.assertEqual(
            result.payload["eps"],
            {"actual": 1.10, "estimate": 1.00},
        )
        self.assertEqual(result.payload["timestamp_quality"], "scheduled")

        signal = interpret_earnings_payload(
            result.payload,
            as_of=datetime(2026, 7, 30, 8, 0, tzinfo=EASTERN),
            decision_session="2026-07-30",
            trading_sessions=SESSIONS,
        )
        self.assertIn(
            "imprecise_report_timestamp",
            signal.data_quality_flags,
        )
        self.assertLess(signal.confidence, 0.45)
        self.assertFalse(signal.policy_eligible)
        self.assertIn(
            "unverified_report_timestamp",
            signal.blockers,
        )
        self.assertEqual(
            signal.policy_context()["earnings_event_score"],
            0.0,
        )

    def test_adapter_does_not_return_future_results(self) -> None:
        frame = pd.DataFrame(
            {"EPS Estimate": [1.0], "Reported EPS": [1.2]},
            index=pd.DatetimeIndex(
                [datetime(2026, 8, 1, 16, 0, tzinfo=UTC)]
            ),
        )

        class FakeTicker:
            def get_earnings_dates(self, *, limit: int) -> pd.DataFrame:
                return frame

        result = fetch_yfinance_earnings_payload(
            "XYZ",
            as_of=datetime(2026, 7, 30, 12, 0, tzinfo=UTC),
            ticker_factory=lambda symbol: FakeTicker(),
        )

        self.assertFalse(result.available)
        self.assertEqual(
            result.error_code,
            "no_reported_earnings_as_of",
        )


if __name__ == "__main__":
    unittest.main()
