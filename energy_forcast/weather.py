"""Validate archived forecast weather and reconstruct information available at origins."""

from dataclasses import dataclass
from functools import cached_property
from hashlib import sha256
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd


WEATHER_FEATURES = ("temperature_c",)


@dataclass(frozen=True)
class WeatherArchive:
    """Forecast vintages, with provenance supplied by the archive producer."""

    records: pd.DataFrame
    report: dict[str, Any]

    @cached_property
    def _available_records(self) -> pd.DataFrame:
        records = self.records.sort_values(["valid_time", "available_at", "issued_at"])
        latest_issue = records.groupby("valid_time")["issued_at"].cummax()
        return records.loc[records.issued_at == latest_issue].sort_values(["available_at", "issued_at"])

    def at_origins(self, targets: pd.DatetimeIndex, origins: pd.DatetimeIndex) -> pd.DataFrame:
        """Return latest-issued forecasts actually available by each origin.

        Unknown values remain NaN. A later archive retrieval timestamp must never
        be substituted for the timestamp the forecast first became available.
        """
        if len(targets) != len(origins) or (origins >= targets).any():
            raise ValueError("Each weather target needs one strictly earlier forecast origin.")
        if targets.tz is not None or origins.tz is not None or targets.hasnans or origins.hasnans:
            raise ValueError("Weather queries require nonempty timezone-naive clock labels.")
        # Discard late arrivals of older vintages, retaining the latest issuance
        # available at every arrival event. This makes an as-of join causal.
        records = self._available_records
        records = records.loc[records.valid_time.isin(targets)]
        requests = pd.DataFrame({"valid_time": targets, "origin": origins, "row": np.arange(len(targets))})
        joined = pd.merge_asof(
            requests.sort_values("origin"), records, left_on="origin", right_on="available_at",
            by="valid_time", direction="backward", allow_exact_matches=True,
        ).sort_values("row")
        return pd.DataFrame(joined[list(WEATHER_FEATURES)].to_numpy(), index=targets,
                            columns=WEATHER_FEATURES)


def read_weather_csv(content: bytes) -> WeatherArchive:
    """Read forecast-only weather vintages without downloading or inventing data.

    Required columns: valid_time, issued_at, available_at, record_type,
    temperature_c, source. All timestamps use the same normalized clock as load.
    """
    try:
        records = pd.read_csv(BytesIO(content))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError("Provide a readable weather forecast archive CSV.") from exc
    required = {"valid_time", "issued_at", "available_at", "record_type", "temperature_c", "source"}
    if records.empty or not required.issubset(records.columns):
        raise ValueError(f"Weather CSV requires rows and columns: {', '.join(sorted(required))}.")
    records = records.loc[:, sorted(required)].copy()
    if not records.record_type.eq("forecast").all():
        raise ValueError("Weather must contain forecast vintages only; observed/reanalysis weather is not accepted.")
    if records.source.isna().any() or records.source.astype(str).str.strip().eq("").any():
        raise ValueError("Every weather forecast needs a nonempty source identifier.")
    for column in ("valid_time", "issued_at", "available_at"):
        try:
            stamps = pd.DatetimeIndex(pd.to_datetime(records[column], format="mixed", errors="raise"))
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"Weather {column} must contain valid timestamps.") from exc
        if stamps.hasnans or stamps.tz is not None:
            raise ValueError("Weather timestamps must be nonempty and timezone-naive, aligned with load upstream.")
        if column == "valid_time" and not stamps.equals(stamps.floor("h")):
            raise ValueError("Weather valid_time must lie on hourly boundaries.")
        records[column] = stamps
    if (records.issued_at > records.available_at).any():
        raise ValueError("Weather available_at cannot precede issued_at.")
    if (records.issued_at >= records.valid_time).any():
        raise ValueError("Weather forecasts must be issued strictly before their valid_time.")
    if records.duplicated(["valid_time", "issued_at"]).any():
        raise ValueError("Duplicate weather valid_time/issued_at vintages are ambiguous; aggregate locations upstream.")
    try:
        records["temperature_c"] = pd.to_numeric(records.temperature_c, errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Weather temperature_c must be numeric degrees Celsius.") from exc
    if not np.isfinite(records.temperature_c).all():
        raise ValueError("Weather temperature_c must be finite; omit unavailable vintages instead.")
    return WeatherArchive(records, {
        "source_sha256": sha256(content).hexdigest(), "rows": len(records),
        "sources": sorted(records.source.astype(str).unique().tolist()), "features": list(WEATHER_FEATURES),
        "availability_policy": "Latest issued_at with issued_at <= available_at <= forecast origin",
        "timestamp_assumption": "Same timezone-naive clock as load; normalization is the data provider's responsibility",
        "provenance_limit": "Timestamps and forecast designation are validated, but archive authenticity is not independently verified",
    })
