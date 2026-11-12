"""Regressions for readable dates, complete candidate lists, and delivery."""

import datetime as dt
import unittest
from unittest.mock import patch

from market_agent import daily_ml_forecast_report as report
from market_agent.report_dates import format_report_timestamp, format_session_date
from market_agent.report_freshness import newest_text_report, report_text_timestamp
from market_agent.telegram_delivery import split_telegram_text
from market_agent.tests.test_daily_report_output import _research_buy_row


class ReportPresentationTests(unittest.TestCase):
    def test_session_dates_do_not_shift_to_the_prior_day(self) -> None:
        self.assertEqual(format_session_date("2026-12-18"), "Dec 18, 2026")
        self.assertEqual(format_session_date("2026-02-30"), "unavailable")
        self.assertEqual(format_session_date(None), "unavailable")

    def test_local_dates_round_trip_through_freshness_across_dst(self) -> None:
        for utc, expected in (
            (dt.datetime(2026, 11, 12, 5, 8, tzinfo=dt.timezone.utc),
             "Nov 11, 2026 at 11:08 PM CST (UTC-06:00)"),
            (dt.datetime(2026, 7, 28, 1, 47, tzinfo=dt.timezone.utc),
             "Jul 27, 2026 at 8:47 PM CDT (UTC-05:00)"),
            (dt.datetime(2026, 11, 1, 6, 30, tzinfo=dt.timezone.utc),
             "Nov 1, 2026 at 1:30 AM CDT (UTC-05:00)"),
            (dt.datetime(2026, 11, 1, 7, 30, tzinfo=dt.timezone.utc),
             "Nov 1, 2026 at 1:30 AM CST (UTC-06:00)"),
        ):
            with self.subTest(utc=utc), patch.dict(
                "os.environ", {"MARKET_REPORT_TIMEZONE": "America/Chicago"},
            ):
                generated = format_report_timestamp(utc)
                self.assertEqual(generated, expected)
                self.assertEqual(report_text_timestamp(f"Generated: {generated}"), utc)

    def test_new_local_header_beats_old_iso_timestamp(self) -> None:
        current = "Generated: Nov 11, 2026 at 11:08 PM CST (UTC-06:00)"
        old = "Generated: 2026-11-11 05:08 UTC"
        self.assertEqual(newest_text_report([
            ("ml_forecast_rankings_latest.txt", current),
            ("ml_forecast_rankings_2026-11-11T05-08-00Z.txt", old),
        ]), current)

    def test_invalid_timezone_fails_with_configuration_name(self) -> None:
        with patch.dict("os.environ", {"MARKET_REPORT_TIMEZONE": "bad/zone"}):
            with self.assertRaisesRegex(ValueError, "MARKET_REPORT_TIMEZONE"):
                format_report_timestamp(dt.datetime.now(dt.timezone.utc))

    def test_existing_qualified_buy_does_not_hide_other_buy_candidates(self) -> None:
        args = report.build_parser().parse_args(["--run-profile", "quick"])
        qualified = _research_buy_row()
        candidate = _research_buy_row(**{
            "Symbol": "MSFT", "Nonoverlapping Validation Samples": 5,
        })
        rendered = report.build_telegram_text([qualified, candidate], [], args)
        self.assertIn("Passed model checks: 1 buy, 0 sell/avoid.", rendered)
        self.assertIn("Buy candidates: 1 of 1", rendered)
        self.assertIn("MSFT: forecast", rendered)
        self.assertIn("(5/8 required)", rendered)
        self.assertIn("General news is not included in these forecast scores.", rendered)
        self.assertLess(rendered.index("BUY SIGNALS"), rendered.index("RESEARCH WATCHLIST"))

    def test_capped_display_counts_all_qualified_signals(self) -> None:
        args = report.build_parser().parse_args(["--max-signal-rows", "1"])
        rows = [_research_buy_row(**{"Symbol": symbol}) for symbol in ("AAA", "BBB")]
        rendered = report.build_telegram_text(rows, [], args)
        self.assertIn("Passed model checks: 2 buy, 0 sell/avoid.", rendered)
        self.assertIn("Showing 1 of 2", rendered)
        self.assertEqual(rendered.count(": forecast"), 1)

    def test_sell_probability_is_the_complement_not_another_up_probability(self) -> None:
        row = _research_buy_row(**{
            "Model Call": "Sell", "Forecast Return %": -10.0, "Probability Up %": 18.0,
        })
        rendered = report.format_telegram_signal_row(row)
        self.assertIn("chance of no rise: 82%", rendered)
        self.assertIn("estimated error: 8.0 percentage points", rendered)
        self.assertNotIn("chance of a rise", rendered)

    def test_long_report_preserves_later_sell_sections_and_unicode(self) -> None:
        text = "BUY SIGNALS\n" + "AAPL rise\n" * 600 + "SELL / AVOID\n" + "NVDA 🧪\n" * 600
        chunks = split_telegram_text(text)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(len(chunk.encode("utf-16-le")) // 2 <= 3900 for chunk in chunks))
        self.assertIn("SELL / AVOID", "".join(chunks))
        self.assertEqual("".join(split_telegram_text("🧪" * 4000)), "🧪" * 4000)

    def test_sender_sends_every_chunk_in_order_without_a_real_request(self) -> None:
        text = "Buy section\n" + "x" * 6000 + "\nSell section"
        with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "test", "TELEGRAM_CHAT_ID": "test"}), patch.object(
            report.requests, "post",
        ) as post:
            report.send_telegram(text)
        self.assertGreater(post.call_count, 1)
        sent = [call.kwargs["json"]["text"] for call in post.call_args_list]
        self.assertEqual("".join(sent), text)


if __name__ == "__main__":
    unittest.main()
