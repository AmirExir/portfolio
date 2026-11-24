"""Present the most recent completed event-driven forecast on the dashboard."""

from __future__ import annotations

from pathlib import Path
from subprocess import SubprocessError

import streamlit as st

from market_agent.breaking_analysis import load_latest_analysis, poll_breaking_analysis
from market_agent.publisher_headlines import HeadlineCheck
from market_agent.report_dates import format_report_timestamp, format_session_date
from market_agent.report_freshness import parse_report_timestamp


def analyze_refreshed_headlines(directory: Path, check: HeadlineCheck) -> None:
    """Use the button's existing feed response and the same lock and receipts as the timer."""
    try:
        with st.spinner("Checking for a new market-moving event and running its forecast if needed…"):
            result = poll_breaking_analysis(directory, now=check.checked_at, check_feeds=lambda: check)
        if result["status"] == "completed":
            st.success("A new forecast is available under Market Analysis → Scheduled outlook.")
        elif result["status"] == "busy":
            st.caption("Another breaking-news check or forecast is already running.")
        elif result["status"] in {"cooldown", "retry_wait"}:
            st.caption("A qualifying event is waiting for the next eligible analysis.")
    except (ImportError, OSError, RuntimeError, SubprocessError, ValueError) as exc:
        st.warning(f"The news-triggered analysis could not finish: {exc}")


def render_breaking_analysis(directory: Path) -> None:
    """Show the triggering articles, daily data sessions and separate event report."""
    try:
        latest = load_latest_analysis(directory)
        if latest is None:
            return
        generated = parse_report_timestamp(latest["generated_at"])
        if generated is None:
            raise ValueError("Missing generation time")
        with st.expander("Latest news-triggered market analysis"):
            st.caption(f"Generated {format_report_timestamp(generated)}")
            for event in latest["events"]:
                published = parse_report_timestamp(event["published_at"])
                if published is None:
                    raise ValueError("Missing article publication time")
                st.link_button(event["title"], event["url"])
                st.caption(f"{event['source']} · {format_report_timestamp(published)}")
            sessions = ", ".join(format_session_date(value) for value in latest["as_of_sessions"])
            st.caption(f"Forecasts use daily prices through {sessions}.")
            st.text(latest["summary"])
            if latest["errors"]:
                st.warning("Some requested symbols or inputs were unavailable in this analysis.")
    except (KeyError, TypeError, ValueError, OSError) as exc:
        st.warning(f"Could not read the saved news-triggered analysis: {exc}")
