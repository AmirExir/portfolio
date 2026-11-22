"""Daily price comparisons and a zero-centered color scale for return charts."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from math import isfinite

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale, unlabel_rgb

if __package__:
    from .agent.earnings import us_equity_trading_sessions
else:
    from agent.earnings import us_equity_trading_sessions


# Percent magnitudes are fixed by horizon so changing the visible universe
# cannot recolor a symbol. The narrow center gives small moves a distinct tint.
_RETURN_SCALE = [
    [0.0, "#851c32"], [0.25, "#cb434e"], [0.43, "#ef9290"],
    [0.499999999999, "#ffe1dd"], [0.5, "#e9e9e9"], [0.500000000001, "#d7efe1"],
    [0.57, "#82c6a1"], [0.75, "#238456"], [1.0, "#075139"],
]


def format_signed_return(value: object) -> str:
    """Format a percentage-point move with its sign and preserve sub-cent-percent moves."""
    if isinstance(value, (bool, np.bool_)):
        return "Unavailable"
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return "Unavailable"
    if not isfinite(value):
        return "Unavailable"
    if value == 0:
        return "0.00%"
    if abs(value) < 0.01:
        return f"{value:+.2g}%"
    return f"{value:+.2f}%"


def return_color_axis(lookback_days: int) -> dict:
    """Return fixed symmetric percentage-point limits and a signed, labeled colorbar."""
    if lookback_days < 1:
        raise ValueError("The return horizon must be positive.")
    limit = next((cap for days, cap in ((1, 15), (7, 30), (30, 40), (90, 75), (365, 150))
                  if lookback_days <= days), 300)
    ticks = [-limit, -limit / 2, 0, limit / 2, limit]
    return {
        "cauto": False, "cmin": -limit, "cmax": limit, "colorscale": _RETURN_SCALE,
        "colorbar": dict(
            title=dict(text="Change", font=dict(color="#17212d", size=14)),
            tickvals=ticks,
            ticktext=[f"≤−{limit}%", f"−{limit / 2:g}%", "0%", f"+{limit / 2:g}%", f"≥+{limit}%"],
            tickfont=dict(color="#17212d", size=13), bgcolor="#ffffff", thickness=16, len=0.82,
        ),
    }


def heatmap_returns(close: pd.DataFrame, tickers: tuple[str, ...], lookback_days: int, *, crypto: bool) -> pd.DataFrame:
    """Measure moves between observed daily closes at the requested calendar horizon.

    For equities, use the last exchange session on or before the comparison
    date; one day compares to the preceding exchange session. Crypto trades
    daily. Missing endpoints stay unavailable even if an older price exists.
    """
    if lookback_days < 1:
        raise ValueError("The return horizon must be positive.")
    result = pd.DataFrame({"Ticker": tickers, "Percent Change": np.nan})
    if close.empty:
        return result
    prices = close.reindex(columns=tickers).copy()
    dates = pd.DatetimeIndex(prices.index)
    prices.index = dates.tz_localize(None).normalize()
    if prices.index.hasnans or prices.index.has_duplicates:
        raise ValueError("Heatmap history requires distinct, dated daily prices.")
    end = prices.index.max().date()
    start = end - timedelta(days=lookback_days)
    if not crypto:
        sessions = us_equity_trading_sessions(
            as_of=datetime.combine(start, time(12), tzinfo=timezone.utc), lookback_days=14, future_session_count=1,
        )
        start = max(session for session in sessions if session <= start)
    result.attrs.update(start=start.isoformat(), end=end.isoformat())
    if pd.Timestamp(start) in prices.index:
        endpoints = prices.loc[[pd.Timestamp(start), pd.Timestamp(end)]].apply(pd.to_numeric, errors="coerce")
        valid = (np.isfinite(endpoints) & (endpoints > 0)).all()
        returns = (endpoints.iloc[1] - endpoints.iloc[0]) / endpoints.iloc[0] * 100
        result["Percent Change"] = returns.where(valid & np.isfinite(returns)).to_numpy()
    return result


def _label_color(background: str) -> str:
    channels = [val / 255 for val in unlabel_rgb(background)]
    linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in channels]
    luminance = sum(c * w for c, w in zip(linear, (0.2126, 0.7152, 0.0722)))
    return "#000000" if luminance > 0.179 else "#ffffff"


def build_return_heatmap(data: pd.DataFrame, lookback_days: int) -> tuple[go.Figure, str]:
    """Build a readable flat treemap and describe its area weighting and coverage.

    Supply Ticker, Label and Percent Change columns. An optional Market Cap
    column sizes measured symbols by market cap when available.
    """
    measured = data.loc[np.isfinite(data["Percent Change"])]
    if measured.empty:
        return go.Figure(), "No symbols have prices at both ends of this timeframe."
    tiles = measured
    sizes = [1] * len(tiles)
    hover = "%{label}<br>Change %{text}"
    sizing = "Tiles have equal area."
    if "Market Cap" in measured:
        has_cap = np.isfinite(measured["Market Cap"]) & (measured["Market Cap"] > 0)
        if has_cap.any():
            tiles = measured.loc[has_cap]
            sizes = tiles["Market Cap"].to_list()
            sizing = "Tile area shows market capitalization."
            hover += "<br>Market cap $%{value:,.0f}"
            if not has_cap.all():
                sizing += f" {len(measured) - len(tiles)} without market cap appear only in the table."
        else:
            sizing = "Tiles have equal area because market cap is unavailable."
    if len(measured) < len(data):
        sizing += f" Change unavailable for {len(data) - len(measured)} of {len(data)} symbols."
    axis = return_color_axis(lookback_days)
    values = tiles["Percent Change"].to_numpy(dtype=float)
    positions = 0.5 + 0.5 * np.clip(values / axis["cmax"], -1, 1)
    label_colors = [_label_color(color) for color in sample_colorscale(_RETURN_SCALE, positions)]
    figure = go.Figure(go.Treemap(
        ids=tiles["Ticker"].tolist(), labels=tiles["Label"].tolist(), parents=[""] * len(tiles),
        values=sizes, marker=dict(colors=values.tolist(), coloraxis="coloraxis", line=dict(width=1, color="white")),
        text=[format_signed_return(value) for value in values],
        texttemplate="<b>%{label}</b><br>%{text}", textfont=dict(size=15, color=label_colors),
        hovertemplate=hover + "<extra></extra>",
        tiling=dict(pad=2), pathbar=dict(visible=False), root=dict(color="#ffffff"),
    ))
    figure.update_layout(coloraxis=axis, uniformtext=dict(minsize=12, mode="hide"))
    return figure, sizing
