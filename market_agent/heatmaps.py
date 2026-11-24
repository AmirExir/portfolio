"""Daily price comparisons and a zero-centered color scale for return charts."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from math import floor, isfinite, log10

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


def format_heatmap_price(value: object) -> str:
    """Format a USD quote with cents, retaining four significant digits below a dollar."""
    if isinstance(value, (bool, np.bool_)):
        return "Unavailable"
    try:
        price = float(value)
    except (TypeError, ValueError, OverflowError):
        return "Unavailable"
    if not isfinite(price) or price <= 0:
        return "Unavailable"
    decimals = 2 if price >= 1 else 3 - floor(log10(price))
    return f"${price:,.{decimals}f}"


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
    result = pd.DataFrame({"Ticker": tickers, "Price (USD)": np.nan, "Percent Change": np.nan})
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
    latest = pd.to_numeric(prices.loc[pd.Timestamp(end)], errors="coerce")
    latest = latest.where(np.isfinite(latest) & (latest > 0))
    result["Price (USD)"] = latest.to_numpy()
    if pd.Timestamp(start) in prices.index:
        baseline = pd.to_numeric(prices.loc[pd.Timestamp(start)], errors="coerce")
        baseline = baseline.where(np.isfinite(baseline) & (baseline > 0))
        returns = (latest - baseline) / baseline * 100
        result["Percent Change"] = returns.where(np.isfinite(returns)).to_numpy()
    return result


def _label_color(background: str) -> str:
    channels = [val / 255 for val in unlabel_rgb(background)]
    linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in channels]
    luminance = sum(c * w for c, w in zip(linear, (0.2126, 0.7152, 0.0722)))
    return "#000000" if luminance > 0.179 else "#ffffff"


def build_return_heatmap(data: pd.DataFrame, lookback_days: int) -> tuple[go.Figure, str]:
    """Build a readable flat treemap and describe its area weighting and coverage.

    Supply Ticker, Label and Percent Change columns. Price (USD) adds the ending
    quote to each tile. Market Cap sizes measured symbols when available.
    """
    measured = data.loc[np.isfinite(data["Percent Change"])]
    if measured.empty:
        return go.Figure(), "No symbols have prices at both ends of this timeframe."
    tiles = measured
    sizes = [1] * len(tiles)
    sizing = "Tiles have equal area."
    cap_hover = ""
    if "Market Cap" in measured:
        has_cap = np.isfinite(measured["Market Cap"]) & (measured["Market Cap"] > 0)
        if not has_cap.any():
            return go.Figure(), "Market-cap data is unavailable. Prices and returns are shown in the by-symbol table."
        tiles = measured.loc[has_cap]
        sizes = tiles["Market Cap"].to_list()
        sizing = f"Tile area is proportional to USD market cap. Showing {len(tiles)} of {len(data)} symbols."
        cap_hover = "<br>Market cap $%{value:,.0f}<br>Share of displayed area %{percentRoot:.3~p}"
        if not has_cap.all():
            missing_caps = ", ".join(measured.loc[~has_cap, "Ticker"])
            sizing += f" Market cap unavailable for {missing_caps}. See the by-symbol table."
    if len(measured) < len(data):
        missing = ", ".join(data.loc[~np.isfinite(data["Percent Change"]), "Ticker"])
        sizing += f" Change unavailable for {len(data) - len(measured)} of {len(data)} symbols ({missing})."
    axis = return_color_axis(lookback_days)
    values = tiles["Percent Change"].to_numpy(dtype=float)
    positions = 0.5 + 0.5 * np.clip(values / axis["cmax"], -1, 1)
    label_colors = [_label_color(color) for color in sample_colorscale(_RETURN_SCALE, positions)]
    quote_details = None
    price_line = ""
    quote_hover = ""
    if "Price (USD)" in tiles:
        quote_details = tiles[["Price (USD)"]].copy()
        quote_details["Price (USD)"] = quote_details["Price (USD)"].map(format_heatmap_price)
        price_line = "<br>%{customdata[0]}"
        quote_hover = "<br>Price %{customdata[0]}"
        if "Price as of" in tiles:
            quote_details["Price as of"] = tiles["Price as of"].replace("", "Unavailable").fillna("Unavailable")
            quote_hover += "<br>As of %{customdata[1]}"
        if cap_hover and "Cap source" in tiles:
            cap_hover += f"<br>Source %{{customdata[{len(quote_details.columns)}]}}"
            quote_details["Cap source"] = tiles["Cap source"].fillna("Unavailable")
            if "Size input as of" in tiles and tiles["Size input as of"].notna().any():
                cap_hover += f"<br>Size input as of %{{customdata[{len(quote_details.columns)}]}}"
                quote_details["Size input as of"] = tiles["Size input as of"].fillna("Date unavailable")
    full_label = "<b>%{label}</b>" + price_line + "<br>%{text}"
    text_template = full_label
    if cap_hover:
        # Reserve the space in small tiles for the ticker. Hover retains the full quote.
        one_percent = sum(sizes) * 0.01
        text_template = ["<b>%{label}</b>" if size < one_percent else full_label for size in sizes]
    figure = go.Figure(go.Treemap(
        ids=tiles["Ticker"].tolist(), labels=tiles["Label"].tolist(), parents=[""] * len(tiles),
        values=sizes, marker=dict(colors=values.tolist(), coloraxis="coloraxis", line=dict(width=1, color="white")),
        text=[format_signed_return(value) for value in values],
        customdata=quote_details, texttemplate=text_template,
        textfont=dict(size=15, color=label_colors),
        hovertemplate="%{label}" + quote_hover + "<br>Change %{text}" + cap_hover + "<extra></extra>",
        tiling=dict(pad=0 if cap_hover else 2), pathbar=dict(visible=False), root=dict(color="#ffffff"),
    ))
    figure.update_layout(coloraxis=axis, uniformtext=None if cap_hover else dict(minsize=12, mode="hide"))
    return figure, sizing
