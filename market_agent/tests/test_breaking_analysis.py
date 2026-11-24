"""Breaking-event policy and process recovery on isolated forecast directories."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from market_agent import breaking_analysis as subject
from market_agent.publisher_headlines import HeadlineCheck, PublisherHeadline


NOW = datetime(2026, 11, 23, 15, 4, tzinfo=timezone.utc)


def article(title: str, url: str = "https://source.example/release", source: str = "NVIDIA Newsroom",
            published_at: datetime = NOW - timedelta(minutes=30)) -> PublisherHeadline:
    """Construct source records with deterministic publication times."""
    return PublisherHeadline(source, title, url, published_at)


def check(*items: PublisherHeadline, failures: tuple[str, ...] = ()) -> HeadlineCheck:
    return HeadlineCheck(NOW, items, failures, ())


def write_report(symbols: tuple[str, ...], directory: Path, _lock: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": "2026-11-23T15-04-15Z", "run_profile": "quick", "run_complete": True,
        "symbols": list(symbols), "ranked_count": len(symbols), "as_of_sessions": ["2026-11-20"],
        "rows": [{"Symbol": symbol, "As Of Session": "2026-11-20", "Last Price": 50.0} for symbol in symbols],
        "telegram_recommendations": "Fixture market analysis.", "errors": [],
    }
    (directory / "ml_forecast_rankings_latest.json").write_text(json.dumps(report))


class BreakingAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.event = article("NVIDIA Board Reinstates Jensen Huang as President, CEO and Director")

    def test_concrete_covered_events_only(self) -> None:
        cases = (
            ("NVIDIA Board Reinstates Jensen Huang as President, CEO and Director", "ceo_appointment"),
            ("Nvidia CEO steps down", "ceo_departure"),
            ("Nvidia shares jump as Jensen Huang returns as CEO", "ceo_appointment"),
            ("Walmart slashes its annual sales outlook", "outlook_cut"),
            ("Fed cuts interest rates by a quarter point", "fed_rate_cut"),
            ("Federal Reserve raises benchmark interest rate", "fed_rate_hike"),
            ("Walmart agrees to buy a specialty retailer", "acquisition"),
            ("Nvidia CEO may return on Monday", None),
            ("Microsoft CEO names his five favorite books", None),
            ("Nvidia CEO appoints new chief financial officer", None),
            ("Trump says the Fed must cut rates now", None),
            ("Fed leaves rates unchanged after investors bet on a cut in rates", None),
            ("Nvidia reinstates CEO. What now?", "ceo_appointment"),
            ("In May, Nvidia appoints its next CEO", "ceo_appointment"),
            ("Shareholder urges Nvidia to reinstate its CEO", None),
            ("Why did the Nvidia board reinstate its CEO?", None),
            ("Tomorrow's earnings preview for Nvidia", None),
            ("Small-town business appoints CEO", None),
        )
        for title, expected in cases:
            with self.subTest(title=title):
                selected = subject.classify_headline(article(title), NOW)
                self.assertEqual(selected["category"] if selected else None, expected)
        for headline in (article(self.event.title, source="Unconfigured"), article(self.event.title, published_at=NOW + timedelta(seconds=1)),
                         article(self.event.title, published_at=NOW - subject.MAX_AGE - timedelta(seconds=1))):
            self.assertIsNone(subject.classify_headline(headline, NOW))

    def test_event_batch_dedupes_across_runs_publishers_and_preserves_daily_sessions(self) -> None:
        duplicate = article("Nvidia shares jump as Huang returns as CEO", "https://other.example/copy",
                            "MarketWatch", NOW - timedelta(minutes=5))
        macro = article("Fed cuts rates", "https://other.example/fed", "BBC Business")
        with patch.object(subject.subprocess, "run") as unused:
            result = subject.poll_breaking_analysis(self.root, now=NOW,
                check_feeds=lambda: check(self.event, duplicate, macro), run_forecast=write_report)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["symbols"], ["NVDA", "SPY", "QQQ"])
            self.assertTrue((self.root / result["report_path"]).is_file())
            latest = subject.load_latest_analysis(self.root)
            self.assertEqual(len(latest["events"]), 2)
            self.assertEqual(latest["as_of_sessions"], ["2026-11-20"])
            self.assertEqual(latest["summary"], "Fixture market analysis.")
            delayed_copy = subject.poll_breaking_analysis(self.root, now=NOW + timedelta(minutes=40),
                check_feeds=lambda: check(duplicate), run_forecast=write_report)
            self.assertEqual(delayed_copy["status"], "no_event")
            unused.assert_not_called()

    def test_cooldown_queues_distinct_opposing_news_until_next_run(self) -> None:
        subject.poll_breaking_analysis(self.root, now=NOW, check_feeds=lambda: check(self.event), run_forecast=write_report)
        # Rolling coverage can carry a new decision at the same publisher URL.
        exit_news = article("Nvidia CEO resigns", self.event.url, self.event.publisher)
        result = subject.poll_breaking_analysis(self.root, now=NOW + timedelta(minutes=10),
            check_feeds=lambda: check(exit_news), run_forecast=write_report)
        self.assertEqual(result["status"], "cooldown")
        # The new story can rotate out of this feed before the cooldown ends.
        result = subject.poll_breaking_analysis(self.root, now=NOW + timedelta(minutes=40),
            check_feeds=lambda: check(), run_forecast=write_report)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(subject.load_latest_analysis(self.root)["events"][0]["category"], "ceo_departure")

    def test_failures_retry_with_backoff_and_interrupted_completion_recovers_without_refitting(self) -> None:
        def crash_after_output(symbols: tuple[str, ...], directory: Path, lock_fd: int) -> None:
            write_report(symbols, directory, lock_fd)
            raise RuntimeError("interrupted after child committed its artifacts")
        def fail(*_args: object) -> None:
            raise ValueError("no usable forecast")
        with self.assertRaisesRegex(ValueError, "no usable"):
            subject.poll_breaking_analysis(self.root, now=NOW, check_feeds=lambda: check(self.event), run_forecast=fail)
        self.assertIsNone(subject.load_latest_analysis(self.root))
        result = subject.poll_breaking_analysis(self.root, now=NOW + timedelta(minutes=5), check_feeds=lambda: check(), run_forecast=fail)
        self.assertEqual(result["status"], "retry_wait")
        self.assertIn("no usable forecast", result["last_error"])
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            subject.poll_breaking_analysis(self.root, now=NOW + timedelta(minutes=16), check_feeds=lambda: check(),
                                           run_forecast=crash_after_output)
        result = subject.poll_breaking_analysis(self.root, now=NOW + timedelta(minutes=17), check_feeds=lambda: check(),
                                                run_forecast=fail)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(subject.load_latest_analysis(self.root)["events"]), 1)

    def test_benchmark_only_and_corrupt_artifacts_cannot_complete_company_event(self) -> None:
        expected = {"run_id": "fixture", "events": [], "symbols": ["NVDA", "SPY", "QQQ"]}
        write_report(tuple(expected["symbols"]), self.root, 0)
        path = self.root / "ml_forecast_rankings_latest.json"
        fixture = json.loads(path.read_text())
        for payload, reason in (([], "malformed"), ({k: v for k, v in fixture.items() if k != "ranked_count"}, "completed report"),
                                ({**fixture, "ranked_count": 2, "rows": fixture["rows"][1:]}, "NVDA")):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                path.write_text(json.dumps(payload))
                subject._validated_result(self.root, self.root, expected)

    def test_overlapping_poll_cannot_start_a_second_analysis(self) -> None:
        def nested(symbols: tuple[str, ...], directory: Path, lock_fd: int) -> None:
            nested_status = subject.poll_breaking_analysis(self.root, now=NOW, check_feeds=lambda: self.fail("feed called"))
            self.assertEqual(nested_status["status"], "busy")
            write_report(symbols, directory, lock_fd)
        result = subject.poll_breaking_analysis(self.root, now=NOW, check_feeds=lambda: check(self.event), run_forecast=nested)
        self.assertEqual(result["status"], "completed")

    def test_all_provider_outage_is_visible_and_incomplete_report_is_not_published(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "No publisher"):
            subject.poll_breaking_analysis(self.root, now=NOW, check_feeds=lambda: check(failures=("a", "b", "c")))
        def incomplete(symbols: tuple[str, ...], directory: Path, lock_fd: int) -> None:
            write_report(symbols, directory, lock_fd)
            path = directory / "ml_forecast_rankings_latest.json"
            report = json.loads(path.read_text())
            report.update(run_complete=False, as_of_sessions=[])
            path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "completed report"):
            subject.poll_breaking_analysis(self.root, now=NOW, check_feeds=lambda: check(self.event), run_forecast=incomplete)
        self.assertIsNone(subject.load_latest_analysis(self.root))
        recovered = subject.poll_breaking_analysis(self.root, now=NOW + timedelta(minutes=20),
            check_feeds=lambda: check(), run_forecast=write_report)
        self.assertEqual(recovered["status"], "completed")


if __name__ == "__main__":
    unittest.main()
