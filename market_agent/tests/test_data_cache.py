from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from market_agent.agent import data


def _ohlcv_frame(dates: list[str], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": [1_000.0] * len(closes),
        },
        index=pd.to_datetime(dates),
    )


class OhlcvCacheHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.cache_root = Path(self.temporary_directory.name)
        self.now = datetime(2026, 8, 28, 12, tzinfo=timezone.utc)

    def test_existing_short_cache_is_backfilled_to_requested_calendar_start(
        self,
    ) -> None:
        cache_path = self.cache_root / "AAPL_1d.csv"
        cached = _ohlcv_frame(
            ["2026-01-02", "2026-08-27"],
            [100.0, 110.0],
        )
        data._normalize_ohlcv(cached).to_csv(cache_path, index_label="date")
        requested_start = (self.now - timedelta(days=1_825)).date()
        downloaded = _ohlcv_frame(
            [requested_start.isoformat(), "2026-08-27"],
            [60.0, 111.0],
        )

        with (
            patch.object(data, "_cache_root", return_value=self.cache_root),
            patch.object(
                data,
                "_download_ohlcv",
                return_value=downloaded,
            ) as download,
        ):
            result = data.get_ohlcv(
                "AAPL",
                1_825,
                now=self.now,
            )

        self.assertEqual(download.call_args.args[1], requested_start.isoformat())
        self.assertEqual(result.index.min().date(), requested_start)
        self.assertTrue(result.attrs["history_coverage"]["coverage_complete"])
        self.assertTrue(result.attrs["history_coverage"]["backfill_requested"])
        stored = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        self.assertEqual(stored.index.min().date(), requested_start)

    def test_short_request_returns_short_window_without_erasing_long_cache(
        self,
    ) -> None:
        long_start = (self.now - timedelta(days=1_825)).date()
        short_start = (self.now - timedelta(days=913)).date()
        initial = _ohlcv_frame(
            [long_start.isoformat(), short_start.isoformat(), "2026-08-27"],
            [50.0, 75.0, 100.0],
        )
        incremental = _ohlcv_frame(["2026-08-27"], [101.0])

        with (
            patch.object(data, "_cache_root", return_value=self.cache_root),
            patch.object(
                data,
                "_download_ohlcv",
                side_effect=[initial, incremental],
            ) as download,
        ):
            long_result = data.get_ohlcv(
                "MSFT",
                1_825,
                now=self.now,
            )
            short_result = data.get_ohlcv(
                "MSFT",
                913,
                now=self.now,
            )

        self.assertEqual(long_result.index.min().date(), long_start)
        self.assertEqual(short_result.index.min().date(), short_start)
        self.assertEqual(
            download.call_args_list[1].args[1],
            (pd.Timestamp("2026-08-27") - pd.Timedelta(days=7)).strftime(
                "%Y-%m-%d"
            ),
        )
        stored = pd.read_csv(
            self.cache_root / "MSFT_1d.csv",
            index_col=0,
            parse_dates=True,
        )
        self.assertEqual(stored.index.min().date(), long_start)

    def test_provider_limited_history_is_reported_explicitly(self) -> None:
        requested_start = (self.now - timedelta(days=1_825)).date()
        listed_later = _ohlcv_frame(
            ["2025-01-02", "2026-08-27"],
            [10.0, 20.0],
        )

        with (
            patch.object(data, "_cache_root", return_value=self.cache_root),
            patch.object(
                data,
                "_download_ohlcv",
                return_value=listed_later,
            ),
        ):
            result = data.get_ohlcv(
                "NEWIPO",
                1_825,
                now=self.now,
            )

        coverage = result.attrs["history_coverage"]
        self.assertEqual(coverage["requested_start"], requested_start.isoformat())
        self.assertFalse(coverage["coverage_complete"])
        self.assertEqual(coverage["coverage_status"], "provider_history_limited")
        self.assertEqual(coverage["available_rows"], 2)


if __name__ == "__main__":
    unittest.main()
