from __future__ import annotations

from datetime import datetime, timezone
import unittest

from market_agent.report_freshness import (
    newest_json_payload,
    newest_text_report,
    newest_timestamped_path,
    parse_report_timestamp,
    report_path_is_newer,
)


class ReportFreshnessTests(unittest.TestCase):
    def test_timestamp_parser_accepts_report_and_filename_formats(self) -> None:
        expected = datetime(2026, 8, 8, 2, 21, 21, tzinfo=timezone.utc)

        self.assertEqual(
            parse_report_timestamp("Generated: 2026-08-08T02-21-21Z"),
            expected,
        )
        self.assertEqual(
            parse_report_timestamp(
                "ml_forecast_rankings_2026-08-08T02-21-21Z.txt"
            ),
            expected,
        )

    def test_newer_timestamped_text_beats_stale_latest_alias(self) -> None:
        stale = "Market Optimization\nGenerated: 2026-08-01 01:58 UTC"
        current = "Market Optimization\nGenerated: 2026-08-08 02:21 UTC"

        selected = newest_text_report(
            [
                ("ml_forecast_rankings_latest.txt", stale),
                ("ml_forecast_rankings_2026-08-08T02-21-21Z.txt", current),
            ]
        )

        self.assertEqual(selected, current)

    def test_publication_filename_can_be_newer_than_embedded_model_time(self) -> None:
        previous = "Market Optimization\nGenerated: 2026-08-08 02:10 UTC"
        current = "Market Optimization\nGenerated: 2026-08-08 02:00 UTC"

        selected = newest_text_report(
            [
                ("ml_forecast_rankings_2026-08-08T02-10-00Z.txt", previous),
                ("ml_forecast_rankings_2026-08-08T02-21-21Z.txt", current),
            ]
        )

        self.assertEqual(selected, current)

    def test_newer_json_payload_beats_stale_latest_alias(self) -> None:
        stale = {"generated_at": "2026-08-01T01-58-00Z", "rows": [{}]}
        current = {"generated_at": "2026-08-08T02-21-21Z", "rows": [{}]}

        selected = newest_json_payload(
            [
                ("ml_forecast_rankings_latest.json", stale),
                ("ml_forecast_rankings_2026-08-08T02-21-21Z.json", current),
            ]
        )

        self.assertIs(selected, current)

    def test_newest_path_is_selected_without_loading_historical_payloads(self) -> None:
        selected = newest_timestamped_path(
            [
                "ml_forecast_rankings_2026-08-01T01-58-00Z.json",
                "ml_forecast_rankings_latest.json",
                "ml_forecast_rankings_2026-08-08T02-21-21Z.json",
            ]
        )

        self.assertEqual(
            selected,
            "ml_forecast_rankings_2026-08-08T02-21-21Z.json",
        )

    def test_newer_remote_path_beats_stale_local_timestamp(self) -> None:
        local_timestamp = parse_report_timestamp("2026-08-01T01-58-00Z")

        self.assertTrue(
            report_path_is_newer(
                "ml_forecast_rankings_2026-08-08T02-21-21Z.txt",
                local_timestamp,
            )
        )
        self.assertFalse(
            report_path_is_newer(
                "ml_forecast_rankings_2026-07-30T17-39-00Z.txt",
                local_timestamp,
            )
        )


if __name__ == "__main__":
    unittest.main()
