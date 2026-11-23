"""Regression checks for an explicit, read-only dashboard news refresh."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

from market_agent.publisher_headlines import (
    HeadlineCheck, PublisherHeadline, check_publisher_headlines, parse_publisher_feed,
)


CHECKED = datetime(2026, 11, 23, 15, 5, tzinfo=timezone.utc)
PUBLISHED = CHECKED - timedelta(minutes=15)


def rss_item(title: str, date: str, url: str) -> str:
    """Build a synthetic RSS article."""
    return f"<item><title>{title}</title><pubDate>{date}</pubDate><link>{url}</link></item>"


class PublisherHeadlinesTests(unittest.TestCase):
    def test_merges_current_rss_and_atom_with_incomplete_source_coverage(self) -> None:
        url = "https://example.org/story"
        dated = "Mon, 23 Nov 2026 08:50:00 -0600"
        items = [
            rss_item("Board announcement", dated, url),
            rss_item("Duplicate", dated, url + "#section"),
            rss_item("Stale", "Sat, 21 Nov 2026 00:00:00 GMT", url + "/old"),
            rss_item("Future", "Fri, 27 Nov 2026 10:30:00 GMT", url + "/future"),
            rss_item("Unknown zone", "Mon, 23 Nov 2026 08:55:00", url + "/naive"),
            rss_item("Bad link", dated, "javascript:alert(1)"),
        ]
        responses = {
            "rss": ("<rss><channel>" + "".join(items) + "</channel></rss>").encode(),
            "atom": b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry>
                <title>Second report</title><published>2026-11-23T15:01:00Z</published>
                <link rel="self" href="https://example.org/api"/>
                <link href="https://example.org/second"/></entry></feed>""",
            "empty": b"<rss><channel/></rss>",
            "invalid": b"<html>Unavailable</html>",
        }

        def fetch(url: str) -> bytes:
            if url == "offline":
                raise requests.Timeout("fixture timeout")
            return responses[url]

        feeds = tuple((name, name.lower()) for name in ("RSS", "Atom", "Empty", "Invalid", "Offline"))
        with patch("market_agent.publisher_headlines._read_feed", side_effect=fetch):
            check = check_publisher_headlines(feeds, now=CHECKED)
        self.assertEqual(check.checked_at, CHECKED)
        self.assertEqual([item.title for item in check.headlines], ["Second report", "Board announcement"])
        self.assertEqual(check.headlines[1], PublisherHeadline("RSS", "Board announcement", url, PUBLISHED))
        self.assertEqual(check.empty_sources, ("Empty",))
        self.assertEqual(check.failures, ("Invalid (invalid or unsupported feed)", "Offline (connection failed)"))

    def test_rejects_unsafe_or_oversized_documents(self) -> None:
        with self.assertRaisesRegex(ValueError, "document type"):
            parse_publisher_feed(b"<!DOCTYPE rss><rss><channel/></rss>", "Fixture", CHECKED)
        response = MagicMock()
        response.__enter__.return_value = response
        response.iter_content.return_value = [b"1" * (2 * 1024 * 1024), b"2"]
        with patch("market_agent.publisher_headlines.requests.get", return_value=response) as get:
            check = check_publisher_headlines((("Fixture", "https://example.org/rss"),), now=CHECKED)
        self.assertEqual(check.headlines, ())
        self.assertEqual(len(check.failures), 1)
        self.assertTrue(get.call_args.kwargs["stream"])
        self.assertIsNotNone(get.call_args.kwargs["timeout"])

    def test_dashboard_checks_on_each_click_and_keeps_the_brief_after_an_outage(self) -> None:
        # Streamlit caches script-level functions across AppTest instances in this process.
        st.cache_data.clear()
        self.addCleanup(st.cache_data.clear)
        source = Path(__file__).resolve().parents[1] / "market_agent_app.py"
        body = "Saved market brief"
        archive_url = "https://example.org/archive"
        headline = PublisherHeadline("Publisher", "New company filing", "https://example.org/filing", PUBLISHED)
        first = HeadlineCheck(CHECKED, (headline,), ("Other publisher (connection failed)",), ())
        second = HeadlineCheck(CHECKED + timedelta(minutes=5), (), ("Publisher (connection failed)",), ())
        delayed = HeadlineCheck(PUBLISHED + timedelta(hours=24), (headline,), (), ())

        def github(url: str, **kwargs: object) -> Mock:
            if url == archive_url:
                return Mock(status_code=200, text=body)
            if url.endswith("/news_summaries"):
                return Mock(status_code=200, **{"json.return_value": [{
                    "name": "news_summary_2026-11-23T14-00-00-000Z.txt",
                    "type": "file", "download_url": archive_url,
                }]})
            return Mock(status_code=404)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(sys, "path", [str(source.parent), *sys.path]),
            patch("market_agent.publisher_headlines.check_publisher_headlines", side_effect=[first, second, delayed]) as check,
            patch("requests.get", side_effect=github) as get,
            patch("agent.data.get_ohlcv", return_value=pd.DataFrame()),
            patch("agent.earnings.load_earnings_payload_file", return_value=Mock(available=False, error_code="fixture")),
            patch("yfinance.download", return_value=pd.DataFrame()),
        ):
            script = Path(directory, source.name)
            script.write_text(source.read_text())
            app = AppTest.from_file(str(script)).run(timeout=60)
            check.assert_not_called()
            app.button(key="refresh_market_news").click().run(timeout=60)
            self.assertFalse(app.exception)
            news = next(tab for tab in app.tabs if tab.label == "AI-Generated Market Summary")
            self.assertIn("New company filing", news.dataframe[0].value["Headline"].tolist())
            self.assertTrue(any("Other publisher" in warning.value for warning in news.warning))
            self.assertTrue(any(caption.value.startswith("Brief published") for caption in news.caption))
            reads = sum(call.args == (archive_url,) for call in get.call_args_list)

            app.run(timeout=60)
            self.assertEqual(check.call_count, 1)
            self.assertEqual(sum(call.args == (archive_url,) for call in get.call_args_list), reads)
            app.button(key="refresh_market_news").click().run(timeout=60)
            self.assertFalse(app.exception)
            news = next(tab for tab in app.tabs if tab.label == "AI-Generated Market Summary")
            self.assertEqual(check.call_count, 2)
            self.assertFalse(news.dataframe)
            self.assertTrue(any("All publisher feeds" in error.value for error in news.error))
            self.assertIn(body, [info.value for info in news.info])
            app.button(key="refresh_market_news").click().run(timeout=60)
            news = next(tab for tab in app.tabs if tab.label == "AI-Generated Market Summary")
            self.assertTrue(any("at least 24 hours old" in warning.value for warning in news.warning))
            self.assertTrue(any(caption.value.startswith("Newest article published") for caption in news.caption))
