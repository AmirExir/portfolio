"""Prospective comparisons of local hourly MW forecasts with ERCOT vintages.

All input indexes denote interval START, must carry an explicit timezone, and
are normalized to UTC. Differences between forecasts are disagreement, never
accuracy. An insert-only SQLite store captures forecasts before their target
hours and later evaluates both against the same first-recorded actuals.

Scores are per saved run; overlapping runs and different forecast leads are
not pooled. The store protects against accidental edits through SQL triggers
and content hashes, not against an administrator replacing the database.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Callable, Iterator, Mapping

import pandas as pd


UTC = timezone.utc
SCHEMA_VERSION = 1


class ComparisonError(ValueError):
    """Invalid, ambiguous, or non-prospective comparison evidence."""


def _now() -> datetime:
    return datetime.now(UTC)


def _instant(value: datetime | str, label: str) -> datetime:
    try:
        if not isinstance(value, (str, datetime)):
            raise ValueError("not a timestamp")
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is None:
            raise ValueError("missing offset")
        if stamp.nanosecond:
            raise ValueError("sub-microsecond precision is unsupported")
        return stamp.to_pydatetime().astimezone(UTC)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ComparisonError(f"{label} must be a valid timezone-aware timestamp.") from exc


def _mw(value: Any) -> float:
    if isinstance(value, bool):
        raise ComparisonError("Loads must be finite nonnegative MW, not boolean values.")
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ComparisonError("Loads must be numeric MW.") from exc
    if not math.isfinite(number) or number < 0:
        raise ComparisonError("Loads must be finite nonnegative MW.")
    return number


def _json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ComparisonError("Comparison metadata must contain finite JSON values.") from exc


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _metadata(value: Mapping[str, Any], keys: tuple[str, ...], hash_key: str) -> dict[str, Any]:
    result = json.loads(_json(dict(value)))
    for key in keys:
        if not isinstance(result.get(key), str) or not result[key].strip():
            raise ComparisonError(f"Metadata requires a nonempty {key}.")
    digest = result[hash_key]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ComparisonError(f"{hash_key} must be a lowercase SHA-256 digest.")
    return result


def _hour_start(value: datetime | str, label: str) -> datetime:
    stamp = _instant(value, label)
    if stamp.minute or stamp.second or stamp.microsecond:
        raise ComparisonError(f"{label} must be an exact hourly interval start.")
    return stamp


def _frame_values(frame: pd.DataFrame, column: str) -> dict[datetime, float]:
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ComparisonError("Forecast/actual indexes must be timezone-aware interval starts.")
    if frame.columns.duplicated().any() or column not in frame.columns or frame.empty:
        raise ComparisonError(f"A nonempty frame with one {column} column is required.")
    if pd.api.types.is_bool_dtype(frame[column]):
        raise ComparisonError("Boolean values are not MW observations.")
    result = {}
    for stamp, value in frame[column].items():
        start = _hour_start(stamp, "interval start")
        if start in result:
            raise ComparisonError("Duplicate hourly intervals are not averaged or discarded.")
        result[start] = _mw(value)
    return result


def compare_forecasts(
    model_forecast: pd.DataFrame,
    official_report: Mapping[str, Any],
    *,
    model_issued_at: datetime | str,
    model_data_cutoff_utc: datetime | str,
    model_metadata: Mapping[str, Any],
    now: datetime | str | None = None,
) -> dict[str, Any]:
    """Align future hourly forecasts and quantify disagreement in MW.

    ``model_forecast`` uses the existing model's ``predicted_load_mw`` column.
    Its index must explicitly denote interval START; end-labeled data must be
    converted by the caller. ``model_data_cutoff_utc`` is the END of the latest
    actual interval used in fitting/forecasting. The official vintage must have
    been retrieved by the local model's issue time. Metadata requires
    ``model_type``, ``method_version`` and ``dataset_sha256``.
    """

    captured = _instant(now or _now(), "capture time")
    issued = _instant(model_issued_at, "model issue time")
    cutoff = _instant(model_data_cutoff_utc, "model data cutoff")
    if cutoff > issued or issued > captured:
        raise ComparisonError("Model data cutoff must be at or before issue time, which cannot be after capture.")
    metadata = _metadata(model_metadata, ("model_type", "method_version", "dataset_sha256"), "dataset_sha256")
    if official_report.get("ok") is not True or official_report.get("units") != "MW" or official_report.get("forecast_kind") != "official_ercot_np3_561_cd":
        raise ComparisonError("A successful official NP3-561-CD report in MW is required.")
    published = _instant(official_report.get("published_at"), "ERCOT publication time")
    retrieved = _instant(official_report.get("generated_at"), "ERCOT retrieval time")
    if not published <= retrieved <= issued:
        raise ComparisonError("ERCOT publication and retrieval must precede the local model's issue time.")
    provenance = _metadata(
        {key: official_report.get(key) for key in ("source_url", "source_sha256", "report_name")},
        ("source_url", "source_sha256", "report_name"), "source_sha256",
    )
    official = {}
    for row in official_report.get("hourly_series", []):
        start = _hour_start(row.get("interval_start_utc"), "ERCOT interval start")
        end = _instant(row.get("interval_end_utc"), "ERCOT interval end")
        if end - start != timedelta(hours=1) or start in official:
            raise ComparisonError("ERCOT intervals must be distinct, exactly one elapsed hour each.")
        official[start] = _mw(row.get("load_mw"))
    if not official:
        raise ComparisonError("ERCOT supplied no hourly forecast intervals.")
    model = _frame_values(model_forecast, "predicted_load_mw")
    future = {start for start in model if start > captured}
    matched = sorted(future & official.keys())
    rows = [{
        "interval_start_utc": start.isoformat(),
        "interval_end_utc": (start + timedelta(hours=1)).isoformat(),
        "model_mw": model[start], "ercot_mw": official[start],
        "difference_mw": model[start] - official[start],
        "model_lead_hours": (start - issued).total_seconds() / 3600,
        "ercot_lead_hours": (start - published).total_seconds() / 3600,
    } for start in matched]
    differences = [row["difference_mw"] for row in rows]
    return {
        "schema_version": SCHEMA_VERSION, "comparison_kind": "forecast_disagreement", "units": "MW",
        "captured_at": captured.isoformat(), "model_issued_at": issued.isoformat(),
        "model_data_cutoff_utc": cutoff.isoformat(), "model_metadata": metadata,
        "ercot_published_at": published.isoformat(), "ercot_retrieved_at": retrieved.isoformat(),
        "ercot_source": provenance, "rows": rows,
        "summary": {
            "matched_hours": len(rows), "excluded_elapsed_model_hours": len(model) - len(future),
            "model_hours_without_ercot": len(future - official.keys()),
            "mean_difference_mw": sum(differences) / len(rows) if rows else None,
            "mean_absolute_difference_mw": sum(abs(value) for value in differences) / len(rows) if rows else None,
            "maximum_absolute_difference_mw": max(map(abs, differences)) if rows else None,
            "interpretation": "Forecast disagreement, not accuracy. Accuracy requires later actual loads on the same saved intervals.",
        },
    }


def _validate_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild derived fields, preventing forged differences or stale metadata."""

    payload = json.loads(_json(dict(snapshot)))
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("comparison_kind") != "forecast_disagreement" or payload.get("units") != "MW":
        raise ComparisonError("Unsupported comparison snapshot schema or units.")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ComparisonError("Only nonempty prospective comparisons can be saved.")
    try:
        frame = pd.DataFrame({"predicted_load_mw": [row["model_mw"] for row in rows]}, index=pd.DatetimeIndex([_instant(row["interval_start_utc"], "snapshot interval") for row in rows]))
        official = {
            "ok": True, "units": "MW", "forecast_kind": "official_ercot_np3_561_cd",
            "published_at": payload["ercot_published_at"], "generated_at": payload["ercot_retrieved_at"],
            **payload["ercot_source"],
            "hourly_series": [{"interval_start_utc": row["interval_start_utc"], "interval_end_utc": row["interval_end_utc"], "load_mw": row["ercot_mw"]} for row in rows],
        }
        rebuilt = compare_forecasts(frame, official, model_issued_at=payload["model_issued_at"], model_data_cutoff_utc=payload["model_data_cutoff_utc"], model_metadata=payload["model_metadata"], now=payload["captured_at"])
    except (KeyError, TypeError) as exc:
        raise ComparisonError("Comparison snapshot is incomplete.") from exc
    if rebuilt["rows"] != rows:
        raise ComparisonError("Snapshot interval values, lead times or differences are inconsistent.")
    summary = payload.get("summary", {})
    for key in ("matched_hours", "mean_difference_mw", "mean_absolute_difference_mw", "maximum_absolute_difference_mw"):
        if summary.get(key) != rebuilt["summary"][key]:
            raise ComparisonError("Snapshot disagreement summary is inconsistent with its hourly rows.")
    for key in ("excluded_elapsed_model_hours", "model_hours_without_ercot"):
        if isinstance(summary.get(key), bool) or not isinstance(summary.get(key), int) or summary[key] < 0:
            raise ComparisonError("Snapshot coverage counts must be nonnegative integers.")
    return payload


def _error_metrics(errors: list[float]) -> dict[str, Any] | None:
    if not errors:
        return None
    return {"sample_count": len(errors), "mae_mw": sum(map(abs, errors)) / len(errors), "rmse_mw": math.sqrt(sum(value * value for value in errors) / len(errors)), "bias_mw": sum(errors) / len(errors)}


def validate_comparison_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Return an independent validated copy for report and chart consumers.

    Validation reconstructs every interval, lead time and disagreement metric.
    It checks prospective capture semantics, not current forecast freshness;
    consumers presenting a current report must also verify its source vintage.
    """
    if not isinstance(snapshot, Mapping):
        raise ComparisonError("Comparison snapshot must be a mapping.")
    if not isinstance(snapshot.get("summary"), Mapping):
        raise ComparisonError("Comparison snapshot requires a disagreement summary.")
    return _validate_snapshot(snapshot)


class ForecastComparisonStore:
    """Insert-only prospective snapshots and observed actual vintages.

    Supply a persistent, non-Git path. SQLite transactions serialize writers;
    repeated identical snapshots/actual vintages are idempotent. ``clock`` is
    injectable for deterministic tests; production callers should omit it.
    Actuals retain their first-recorded vintage for scoring, so later revisions
    do not silently change previously published accuracy figures.
    """

    def __init__(self, path: str | Path, *, clock: Callable[[], datetime] = _now) -> None:
        self.path = Path(path)
        self.clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS snapshots (id TEXT PRIMARY KEY, payload TEXT NOT NULL, recorded_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS actuals (id TEXT PRIMARY KEY, interval_start TEXT NOT NULL, source_hash TEXT NOT NULL, payload TEXT NOT NULL, recorded_at TEXT NOT NULL, UNIQUE(interval_start, source_hash));
                CREATE INDEX IF NOT EXISTS actual_interval_idx ON actuals(interval_start);
            """)
            for table in ("snapshots", "actuals"):
                for operation in ("UPDATE", "DELETE"):
                    connection.execute(f"CREATE TRIGGER IF NOT EXISTS {table}_no_{operation.lower()} BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT, 'forecast evidence is append-only'); END")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def append_snapshot(self, snapshot: Mapping[str, Any]) -> str:
        """Persist a comparison before every included interval begins."""

        payload = _validate_snapshot(snapshot)
        identifier = _hash(payload)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute("SELECT payload FROM snapshots WHERE id=?", (identifier,)).fetchone()
            if existing:
                if _hash(json.loads(existing["payload"])) != identifier:
                    raise ComparisonError("Saved snapshot integrity check failed.")
                return identifier
            recorded = _instant(self.clock(), "recording time")
            if _instant(payload["captured_at"], "capture time") > recorded or any(recorded >= _instant(row["interval_start_utc"], "target start") for row in payload["rows"]):
                raise ComparisonError("A forecast must be captured and saved strictly before every target interval starts.")
            connection.execute("INSERT INTO snapshots VALUES (?,?,?)", (identifier, _json(payload), recorded.isoformat()))
        return identifier

    def list_snapshots(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return newest verified snapshots; no runs are pooled or rescored."""

        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ComparisonError("Snapshot limit must be an integer from 1 to 1000.")
        with self._connect() as connection:
            records = connection.execute("SELECT * FROM snapshots ORDER BY recorded_at DESC, rowid DESC LIMIT ?", (limit,)).fetchall()
        results = []
        for record in records:
            payload = _validate_snapshot(json.loads(record["payload"]))
            if _hash(payload) != record["id"]:
                raise ComparisonError("Saved snapshot integrity check failed.")
            results.append({"snapshot_id": record["id"], "recorded_at": record["recorded_at"], **payload})
        return results

    def append_actuals(self, actuals: pd.DataFrame, *, observed_at: datetime | str, source_metadata: Mapping[str, Any]) -> int:
        """Append measured ``load`` MW at aware interval starts, after maturity."""

        values = _frame_values(actuals, "load")
        observed = _instant(observed_at, "actual observation time")
        metadata = _metadata(source_metadata, ("source_url", "source_sha256"), "source_sha256")
        added = 0
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            recorded = _instant(self.clock(), "recording time")
            if observed > recorded or any(start + timedelta(hours=1) > observed for start in values):
                raise ComparisonError("Actual loads cannot be recorded before their complete hourly intervals have elapsed.")
            for start, value in sorted(values.items()):
                existing = connection.execute("SELECT payload FROM actuals WHERE interval_start=? AND source_hash=?", (start.isoformat(), metadata["source_sha256"])).fetchone()
                if existing:
                    prior = json.loads(existing["payload"])
                    if prior["load_mw"] != value:
                        raise ComparisonError("The same actual-data source hash cannot describe different MW values.")
                    continue
                latest = connection.execute("SELECT payload FROM actuals WHERE interval_start=? ORDER BY recorded_at DESC, rowid DESC LIMIT 1", (start.isoformat(),)).fetchone()
                if latest and json.loads(latest["payload"])["load_mw"] == value:
                    # A new bundle hash often reflects one newly completed
                    # hour. Keep prior unchanged hours' original provenance.
                    continue
                payload = {"interval_start_utc": start.isoformat(), "interval_end_utc": (start + timedelta(hours=1)).isoformat(), "load_mw": value, "observed_at": observed.isoformat(), "source_metadata": metadata}
                connection.execute("INSERT INTO actuals VALUES (?,?,?,?,?)", (_hash(payload), start.isoformat(), metadata["source_sha256"], _json(payload), recorded.isoformat()))
                added += 1
        return added

    def evaluate_snapshot(self, snapshot_id: str, *, as_of: datetime | str | None = None) -> dict[str, Any]:
        """Score both frozen forecasts on identical matured, observed intervals.

        MAE, RMSE and signed bias are MW; bias is forecast minus actual. Rows
        retain both distinct source lead times. Missing actuals are pending
        evidence, never zero load. Different runs/vintages are not combined.
        """

        cutoff = _instant(as_of or self.clock(), "evaluation time")
        if cutoff > _instant(self.clock(), "current time"):
            raise ComparisonError("Evaluation time cannot be in the future.")
        with self._connect() as connection:
            record = connection.execute("SELECT * FROM snapshots WHERE id=?", (snapshot_id,)).fetchone()
            if record is None:
                raise ComparisonError("Unknown forecast snapshot.")
            snapshot = _validate_snapshot(json.loads(record["payload"]))
            if _hash(snapshot) != snapshot_id:
                raise ComparisonError("Saved snapshot integrity check failed.")
            if _instant(record["recorded_at"], "snapshot recording time") > cutoff:
                raise ComparisonError("The snapshot was not yet recorded at the evaluation time.")
            rows, pending, missing = [], 0, 0
            for forecast in snapshot["rows"]:
                if _instant(forecast["interval_end_utc"], "target end") > cutoff:
                    pending += 1
                    continue
                observations = connection.execute("SELECT * FROM actuals WHERE interval_start=? ORDER BY recorded_at, rowid", (forecast["interval_start_utc"],)).fetchall()
                eligible = [item for item in observations if _instant(item["recorded_at"], "actual recording time") <= cutoff]
                if not eligible:
                    missing += 1
                    continue
                first = eligible[0]
                actual = json.loads(first["payload"])
                if _hash(actual) != first["id"] or actual["interval_end_utc"] != forecast["interval_end_utc"] or _instant(actual["observed_at"], "actual observation time") > cutoff:
                    raise ComparisonError("Saved actual-load integrity check failed.")
                actual_mw = _mw(actual["load_mw"])
                rows.append({**forecast, "actual_mw": actual_mw, "model_error_mw": forecast["model_mw"] - actual_mw, "ercot_error_mw": forecast["ercot_mw"] - actual_mw, "actual_observed_at": actual["observed_at"], "actual_recorded_at": first["recorded_at"], "actual_source": actual["source_metadata"]})
        return {
            "snapshot_id": snapshot_id, "status": "complete" if len(rows) == len(snapshot["rows"]) else "partial" if rows else "awaiting_actuals",
            "units": "MW", "as_of": cutoff.isoformat(), "matched_count": len(rows), "pending_count": pending, "missing_actual_count": missing,
            "model_issued_at": snapshot["model_issued_at"], "ercot_published_at": snapshot["ercot_published_at"], "model_metadata": snapshot["model_metadata"],
            "model_metrics": _error_metrics([row["model_error_mw"] for row in rows]), "ercot_metrics": _error_metrics([row["ercot_error_mw"] for row in rows]), "rows": rows,
            "actual_revision_policy": "first recorded source observation per interval",
            "interpretation": "Per-run accuracy on the same observed hours. Forecast leads differ and are retained per row; overlapping runs are not pooled. Actuals may be revised by the source.",
        }
