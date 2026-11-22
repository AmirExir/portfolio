"""Streamlit display for the saved, completed-session price watch."""

from __future__ import annotations

import pandas as pd
import streamlit as st

if __package__:
    from .price_watch_report import (
        format_next_watch_prices, format_price_watch, format_watch_prices, is_usable_observation,
    )
else:
    from price_watch_report import (
        format_next_watch_prices, format_price_watch, format_watch_prices, is_usable_observation,
    )


def _watch_row(item: dict) -> dict:
    if not is_usable_observation(item):
        raise ValueError("Invalid saved price observation")
    close, low, high = format_watch_prices(item)
    next_low, next_high = format_next_watch_prices(item) or ("Unavailable", "Unavailable")
    conditions = []
    if item["shock"]:
        conditions.append("Loss threshold reached")
    if item["breakdown"]:
        conditions.append("Below prior closing low")
    if item["entry_watch"]:
        conditions.append("Above prior closing high" if item["entry_triggered"] else "Near breakout level")
    return {
        "Symbol": item["symbol"],
        "Session": item["as_of_session"],
        "Price condition": " · ".join(conditions) or "No trigger",
        "Close": close,
        "1-session change": item["change_pct"],
        "Next-close lower level": next_low,
        "Next-close upper level": next_high,
        "Loss threshold": item["shock_threshold_pct"],
        "Prior 20-session closing low": low,
        "Prior 20-session closing high": high,
    }


def render_scheduled_price_watch(plan: dict | None) -> None:
    """Show closing-price alerts and searchable per-symbol levels from one saved run."""
    if not isinstance(plan, dict) or not plan:
        st.caption("Price-watch data is unavailable in this saved report.")
        return
    try:
        watch_lines = format_price_watch(plan)
        rows = pd.DataFrame([_watch_row(item) for item in plan.get("observations", [])])
    except (KeyError, IndexError, TypeError, ValueError, OverflowError):
        st.warning("The saved price watch contains incomplete or invalid data.")
        return
    st.subheader(watch_lines[0])
    st.text("\n".join(watch_lines[1:]).strip())
    if rows.empty:
        return
    with st.expander(f"Price conditions by symbol ({len(rows)})"):
        query = st.text_input("Find a symbol", key="scheduled_price_watch_symbol").strip().upper()
        if query:
            rows = rows[rows["Symbol"].str.contains(query, regex=False)]
        if not rows.empty:
            st.caption(
                "Next-close levels use the 20 sessions ending on the displayed date. "
                "Prior levels use the 20 sessions preceding that date."
            )
            st.dataframe(
                rows.style.format({"1-session change": "{:+.2f}%", "Loss threshold": "{:.2f}%"}),
                use_container_width=True, hide_index=True,
            )
        else:
            st.caption("No observed symbol matches this search.")
