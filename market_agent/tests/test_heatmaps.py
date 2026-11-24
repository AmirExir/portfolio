"""Return signs, area weighting and observation windows of the dashboard heatmaps."""

import unittest

import numpy as np
import pandas as pd
from plotly.colors import sample_colorscale, unlabel_rgb

from market_agent.heatmaps import (
    build_return_heatmap, format_heatmap_price, format_signed_return, heatmap_returns, return_color_axis,
)


class HeatmapTests(unittest.TestCase):
    def test_color_sign_and_intensity_stay_anchored_as_the_universe_changes(self) -> None:
        for moves in ((-14, -6, -1.9), (0.5, 1, 7), (-25, -0.001, 0, 0.001, 21), (0,), (-6,)):
            symbols = [f"T{i}" for i in range(len(moves))]
            data = pd.DataFrame({"Ticker": symbols, "Label": symbols, "Percent Change": moves})
            fig, _ = build_return_heatmap(data, 1)
            axis = fig.layout.coloraxis
            self.assertEqual((axis.cmin, axis.cmax, axis.cauto), (-15, 15, False))
            self.assertIn("0%", axis.colorbar.ticktext)
            self.assertEqual(fig.data[0].marker.coloraxis, "coloraxis")
            self.assertEqual(tuple(fig.data[0].labels), tuple(symbols))
            self.assertEqual(tuple(fig.data[0].marker.colors), moves)
            self.assertEqual(tuple(fig.data[0].text), tuple(map(format_signed_return, moves)))
            positions = np.clip((np.array(moves) + 15) / 30, 0, 1)
            colors = sample_colorscale([list(stop) for stop in axis.colorscale], positions)
            for value, color, font in zip(moves, colors, fig.data[0].textfont.color):
                r, g, b = unlabel_rgb(color)
                self.assertEqual(np.sign(g - r), np.sign(value))
                if value == 0:
                    self.assertEqual((r, g), (b, b))
                linear = [(v / 255 / 12.92 if v <= 10 else ((v / 255 + .055) / 1.055) ** 2.4)
                          for v in (r, g, b)]
                luminance = sum(c * w for c, w in zip(linear, (.2126, .7152, .0722)))
                contrast = (luminance + .05) / .05 if font == "#000000" else 1.05 / (luminance + .05)
                self.assertGreater(contrast, 4.5)
        axis = return_color_axis(1)
        self.assertEqual(return_color_axis(20), return_color_axis(30))
        self.assertGreater(return_color_axis(365)["cmax"], axis["cmax"])
        rgb = [unlabel_rgb(c) for c in sample_colorscale(axis["colorscale"], [0.01, 0.43, 0.57, 0.99])]
        self.assertLess(sum(rgb[0]), sum(rgb[1]))
        self.assertLess(sum(rgb[3]), sum(rgb[2]))

    def test_ticker_alignment_cap_sizes_and_missing_observations(self) -> None:
        data = pd.DataFrame({
            "Ticker": ["Z", "C", "A", "E"], "Label": ["Zulu", "Charlie", "Alpha", "Echo"],
            "Percent Change": [-2, 0, 3.5, np.nan], "Market Cap": [30, np.nan, 90, 100],
        })
        fig, sizing = build_return_heatmap(data, 7)
        trace = fig.data[0]
        self.assertEqual((tuple(trace.ids), tuple(trace.values), tuple(trace.text)), (("Z", "A"), (30, 90), ("-2.00%", "+3.50%")))
        self.assertIn("1 without market cap", sizing)
        self.assertIn("Change unavailable for 1", sizing)
        self.assertEqual(len(data), 4)  # table retains every requested symbol
        data["Market Cap"] = np.nan
        fig, sizing = build_return_heatmap(data, 7)
        self.assertIn("equal area", sizing)
        self.assertEqual(tuple(fig.data[0].values), (1, 1, 1))
        self.assertNotIn("Market cap", fig.data[0].hovertemplate)
        data["Percent Change"] = np.nan
        self.assertFalse(build_return_heatmap(data, 7)[0].data)

    def test_price_above_return_and_hover_time_follow_visible_symbols(self) -> None:
        data = pd.DataFrame({
            "Ticker": ["A", "B", "C"], "Label": ["A", "B", "C"],
            "Price (USD)": [12345.67, 101.0, 0.000012345],
            "Percent Change": [2, 0, -1], "Market Cap": [200, np.nan, 100],
            "Price as of": ["2026-11-23 15:02 UTC", "2026-11-23 close", "2026-11-23 15:01 UTC"],
        })
        trace = build_return_heatmap(data, 1)[0].data[0]
        self.assertEqual(tuple(trace.ids), ("A", "C"))
        self.assertEqual(trace.customdata.tolist(), [
            ["$12,345.67", "2026-11-23 15:02 UTC"],
            ["$0.00001234", "2026-11-23 15:01 UTC"],
        ])
        self.assertEqual(trace.texttemplate, "<b>%{label}</b><br>%{customdata[0]}<br>%{text}")
        self.assertIn("As of %{customdata[1]}", trace.hovertemplate)
        self.assertEqual(tuple(trace.text), ("+2.00%", "-1.00%"))
        for invalid in (None, np.nan, np.inf, pd.NA, False, 0, -2, "bad"):
            self.assertEqual(format_heatmap_price(invalid), "Unavailable")

    def test_missing_comparison_preserves_today_price_and_missing_quote_is_never_filled(self) -> None:
        dates = pd.to_datetime(["2026-11-20", "2026-11-23"])
        quotes = pd.DataFrame({
            "LATEST": [None, 15.25], "OLDER": [90, None], "ZERO": [10, 0], "INF": [10, np.inf],
        }, index=dates)
        symbols = ("INF", "LATEST", "MISSING", "OLDER", "ZERO")
        for horizon in (1, 30):
            observations = heatmap_returns(quotes, symbols, horizon, crypto=False).set_index("Ticker")
            self.assertEqual(observations.loc["LATEST", "Price (USD)"], 15.25)
            self.assertTrue(observations.drop("LATEST")["Price (USD)"].isna().all())
            self.assertTrue(observations["Percent Change"].isna().all())
        self.assertTrue(heatmap_returns(pd.DataFrame(), symbols, 1, crypto=True)["Price (USD)"].isna().all())

    def test_calendar_horizon_weekends_and_no_filled_or_shortened_returns(self) -> None:
        days = pd.to_datetime(["2026-11-13", "2026-11-20", "2026-11-22", "2026-11-23"])
        prices = pd.DataFrame({
            "A": [90, 100, 104, 105], "B": [80, 90, 91, np.nan], "C": [50, np.nan, 50, 50],
            "D": [55, 0, 10, 15], "E": [10, np.inf, 10, 15], "F": [7, 8, 8, 8],
        }, index=days)
        tickers = ("A", "B", "C", "D", "E", "F", "MISSING")
        stocks = heatmap_returns(prices.drop(days[2]), tickers, 1, crypto=False)
        self.assertEqual(stocks.attrs, {"start": "2026-11-20", "end": "2026-11-23"})
        returns = stocks.set_index("Ticker")["Percent Change"]
        self.assertEqual(returns["A"], 5)
        self.assertEqual(returns["F"], 0)
        self.assertTrue(returns[["B", "C", "D", "E", "MISSING"]].isna().all())
        crypto = heatmap_returns(prices, tickers, 1, crypto=True)
        self.assertEqual(crypto.attrs["start"], "2026-11-22")
        self.assertAlmostEqual(crypto.loc[0, "Percent Change"], 100 / 104)
        # The needed comparison day is absent from the provider response.
        weekly = heatmap_returns(prices, tickers, 7, crypto=False)
        self.assertEqual(weekly.attrs["start"], "2026-11-16")
        self.assertTrue(weekly["Percent Change"].isna().all())
        single = heatmap_returns(prices.tail(1), tickers, 1, crypto=False)
        self.assertTrue(single["Percent Change"].isna().all())

    def test_rounded_and_missing_values_cannot_look_like_a_zero_move(self) -> None:
        self.assertEqual(format_signed_return(-0.001), "-0.001%")
        self.assertEqual(format_signed_return(0.0002), "+0.0002%")
        self.assertEqual(format_signed_return(-0.0), "0.00%")
        for invalid in (None, np.nan, np.inf, pd.NA, False, "bad"):
            self.assertEqual(format_signed_return(invalid), "Unavailable")


if __name__ == "__main__":
    unittest.main()
