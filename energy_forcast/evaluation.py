"""Empirical lead-specific intervals and calendar-season evaluation summaries."""

from math import ceil

import numpy as np
import pandas as pd


def season_labels(index: pd.DatetimeIndex) -> np.ndarray:
    """Label meteorological seasons, assigning December to the following winter."""
    names = np.array(["Winter", "Spring", "Summer", "Autumn"])[(index.month % 12) // 3]
    years = index.year + (index.month == 12)
    return np.array([f"{year} {name}" for year, name in zip(years, names)])


def calibrate_intervals(predictions: pd.DataFrame, coverage: float = 0.9,
                        min_origins: int = 10) -> pd.DataFrame:
    """Estimate absolute-error quantiles using separate chronological calibration data.

    One error per origin and lead is required. A finite-sample order statistic
    ceil((n+1)*coverage) is used; insufficient samples give no interval. Temporal
    dependence prevents claiming distribution-free coverage guarantees.
    """
    if not 0 < coverage < 1 or min_origins < 1:
        raise ValueError("Interval coverage must be between zero and one and min_origins positive.")
    if predictions.empty:
        return pd.DataFrame(columns=["Radius (MW)", "Calibration origins", "Nominal coverage"])
    if predictions.duplicated(["Origin", "Lead time (hours)"]).any():
        raise ValueError("Calibration requires unique origin/lead errors, without repeated horizons.")
    errors = (predictions["Actual (MW)"] - predictions.XGBoost).abs()
    if not np.isfinite(errors).all():
        raise ValueError("Calibration residuals must be finite.")
    rows = []
    for lead, group in predictions.assign(error=errors).groupby("Lead time (hours)"):
        count = len(group)
        rank = ceil((count + 1) * coverage)
        radius = float(np.sort(group.error)[rank - 1]) if count >= min_origins and rank <= count else np.nan
        rows.append({"Lead time (hours)": lead, "Radius (MW)": radius,
                     "Calibration origins": count, "Nominal coverage": coverage})
    return pd.DataFrame(rows).set_index("Lead time (hours)")


def add_intervals(predictions: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    """Attach nonnegative MW bounds; leave unsupported lead times visibly missing."""
    frame = predictions.copy()
    radius = frame["Lead time (hours)"].map(calibration["Radius (MW)"])
    frame["Lower (MW)"] = (frame.XGBoost - radius).clip(lower=0)
    frame["Upper (MW)"] = frame.XGBoost + radius
    return frame


def evaluation_summary(predictions: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Report errors for identical windows and empirical interval coverage where present."""
    if predictions.empty:
        return pd.DataFrame()
    frame = predictions.copy()
    frame["Season"] = season_labels(pd.DatetimeIndex(frame.index))
    rows = []
    for key, group in frame.groupby(by, sort=True):
        keys = key if isinstance(key, tuple) else (key,)
        for name in ("XGBoost", "Persistence", "Previous day", "Previous week"):
            error = group["Actual (MW)"] - group[name]
            row = {**dict(zip(by, keys)), "Model": name, "Hours": len(group),
                   "Origins": group.Origin.nunique(), "MAE (MW)": float(error.abs().mean()),
                   "RMSE (MW)": float(np.sqrt((error ** 2).mean()))}
            if name == "XGBoost" and "Lower (MW)" in group:
                supported = group.dropna(subset=["Lower (MW)", "Upper (MW)"])
                row["Interval hours"] = len(supported)
                if len(supported):
                    row["Empirical coverage"] = float(supported["Actual (MW)"].between(
                        supported["Lower (MW)"], supported["Upper (MW)"]).mean())
                    row["Mean width (MW)"] = float((supported["Upper (MW)"] - supported["Lower (MW)"]).mean())
            rows.append(row)
    return pd.DataFrame(rows)
