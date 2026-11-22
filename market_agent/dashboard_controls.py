"""Shared Streamlit controls for the market analysis views."""

from collections.abc import Callable, Sequence
from time import time_ns

import streamlit as st


def _store_universe(widget_key: str) -> None:
    st.session_state["market_universe"] = list(st.session_state[widget_key])


def choose_universe(
    view: str, options: Sequence[str], defaults: Sequence[str], label: Callable[[str], str],
) -> list[str]:
    """Edit the same symbol universe from rankings, patterns or the stock heatmap."""
    selected = st.session_state.setdefault("market_universe", list(defaults))
    key = f"market_universe_{view}"
    if st.session_state.get(key) != selected:
        st.session_state[key] = list(selected)
    with st.expander(f"Choose symbols ({len(selected)})"):
        return st.multiselect(
            "Symbols to include", options, key=key, format_func=label,
            on_change=_store_universe, args=(key,),
            help="Rankings, the pattern scan and the stock heatmap share this selection.",
        )


def forecast_run_buttons(run_label: str, rebuild_label: str, key: str, *, disabled: bool) -> tuple[bool, bool]:
    """Request one run and invalidate this view's cache generation only on a rebuild."""
    generation = f"{key}_refresh_nonce"
    st.session_state.setdefault(generation, 0)
    with st.container(horizontal=True):
        run = st.button(run_label, key=f"{key}_run", type="primary", disabled=disabled)
        rebuild = st.button(
            rebuild_label, key=f"{key}_rebuild", disabled=disabled,
            help="Refit models with the current settings, even when a matching forecast is saved.",
        )
    if rebuild:
        # Streamlit shares the calculation cache across browser sessions.
        st.session_state[generation] = time_ns()
    return run or rebuild, rebuild
