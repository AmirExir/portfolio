import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
from pathlib import Path
from functools import lru_cache
import contextlib
import io
import json
import logging
import os
import re


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _provider_aliases() -> dict[str, str]:
    path = Path(__file__).with_name("market_data_symbols.json")
    payload = json.loads(path.read_text())
    aliases = payload.get("aliases")
    if payload.get("provider") != "yahoo_finance" or not isinstance(aliases, dict):
        raise ValueError(f"Invalid market data symbol configuration: {path.name}")
    if any(
        not isinstance(key, str)
        or not isinstance(value, str)
        or not key.strip()
        or not value.strip()
        for key, value in aliases.items()
    ):
        raise ValueError(f"Invalid market data symbol aliases: {path.name}")
    return {key.strip().upper(): value.strip().upper() for key, value in aliases.items()}


def _cache_matches_provider(metadata: dict, symbol: str, provider_symbol: str) -> bool:
    if metadata.get("provider", "yahoo_finance") != "yahoo_finance":
        return False
    recorded_symbol = metadata.get("provider_symbol")
    if recorded_symbol is not None:
        return recorded_symbol == provider_symbol
    # Preserve legacy caches for unchanged symbols. Aliased instruments require
    # explicit provenance; an unverified cache must never cross asset identities.
    return symbol == provider_symbol and metadata.get("symbol", symbol) == symbol


def _with_source_metadata(
    frame: pd.DataFrame,
    symbol: str,
    provider_symbol: str,
    *,
    cache_fallback: bool,
) -> pd.DataFrame:
    source = {
        "provider": "yahoo_finance",
        "requested_symbol": symbol,
        "provider_symbol": provider_symbol,
        "cache_fallback": cache_fallback,
    }
    frame.attrs["market_data_source"] = source
    frame.attrs["history_coverage"]["data_source"] = dict(source)
    return frame


def _cache_root() -> Path:
    configured = os.getenv("MARKET_AGENT_DATA_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[1] / "reports" / "ohlcv_cache"


def _cache_path(symbol: str, interval: str) -> Path:
    safe_symbol = re.sub(r"[^A-Za-z0-9_.-]+", "_", symbol)
    safe_interval = re.sub(r"[^A-Za-z0-9_.-]+", "_", interval)
    return _cache_root() / f"{safe_symbol}_{safe_interval}.csv"


def _cache_metadata_path(symbol: str, interval: str) -> Path:
    return _cache_path(symbol, interval).with_suffix(".metadata.json")


def _normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(column[0]).lower() for column in frame.columns]
    else:
        frame = frame.rename(columns=str.lower)

    available = [column for column in ["open", "high", "low", "close", "volume"] if column in frame.columns]
    if "close" not in available:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    frame = frame[available].copy()
    for column in ["open", "high", "low", "close", "volume"]:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame.index = pd.to_datetime(frame.index, errors="coerce")
    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_convert(None)
    frame = frame[~frame.index.isna()]
    frame = frame[["open", "high", "low", "close", "volume"]]
    frame = frame.dropna(subset=["close"]).sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def _read_cached_ohlcv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    try:
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
        return _normalize_ohlcv(cached)
    except Exception:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def _write_cached_ohlcv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index_label="date")


def _normalized_utc_now(now: datetime | None = None) -> datetime:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _requested_start(lookback_days: int, now: datetime | None = None) -> pd.Timestamp:
    if int(lookback_days) <= 0:
        raise ValueError("lookback_days must be a positive calendar-day count")
    reference = _normalized_utc_now(now)
    return pd.Timestamp(reference.date()) - pd.Timedelta(days=int(lookback_days))


def _requested_window(
    frame: pd.DataFrame,
    lookback_days: int,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Return the deterministic requested calendar window without trimming cache."""

    if frame.empty:
        return frame.copy()
    requested_start = _requested_start(lookback_days, now)
    reference_date = pd.Timestamp(_normalized_utc_now(now).date())
    return frame.loc[
        (frame.index >= requested_start) & (frame.index <= reference_date)
    ].copy()


def _read_cache_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_cache_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def _metadata_timestamp(value) -> pd.Timestamp | None:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert(None)
    return parsed.normalize()


def _history_coverage_metadata(
    frame: pd.DataFrame,
    cached_frame: pd.DataFrame,
    *,
    lookback_days: int,
    requested_start: pd.Timestamp,
    backfill_requested: bool,
    backfill_succeeded: bool,
) -> dict:
    available_start = frame.index.min() if not frame.empty else None
    cache_start = cached_frame.index.min() if not cached_frame.empty else None
    tolerance = pd.Timedelta(days=7)
    coverage_complete = bool(
        available_start is not None
        and pd.Timestamp(available_start).normalize() <= requested_start + tolerance
    )
    if coverage_complete:
        status = "requested_window_available"
    elif backfill_requested and backfill_succeeded:
        status = "provider_history_limited"
    elif backfill_requested:
        status = "backfill_unavailable"
    else:
        status = "cached_history_limited"
    return {
        "requested_calendar_days": int(lookback_days),
        "requested_start": requested_start.date().isoformat(),
        "available_start": (
            pd.Timestamp(available_start).date().isoformat()
            if available_start is not None
            else None
        ),
        "available_rows": int(len(frame)),
        "cache_start": (
            pd.Timestamp(cache_start).date().isoformat()
            if cache_start is not None
            else None
        ),
        "cache_rows": int(len(cached_frame)),
        "backfill_requested": bool(backfill_requested),
        "coverage_complete": coverage_complete,
        "coverage_status": status,
    }


@contextlib.contextmanager
def _quiet_yfinance_output():
    sink = io.StringIO()
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield
    finally:
        logging.disable(previous_disable_level)


def _download_ohlcv(symbol: str, fetch_start: str, interval: str) -> pd.DataFrame:
    try:
        with _quiet_yfinance_output():
            return yf.download(
                symbol,
                start=fetch_start,
                interval=interval,
                progress=False,
                threads=False,
            )
    except Exception:
        return pd.DataFrame()


def get_ohlcv(
    symbol: str,
    lookback_days: int = 200,
    interval: str = "1d",
    *,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Return a deterministic calendar lookback while retaining shared cache.

    Existing caches are incrementally refreshed. When their earliest row is too
    recent for the requested window, one explicit backfill is requested. A
    sidecar records the earliest start already requested so newly listed assets
    do not repeatedly download unavailable pre-listing history. Short requests
    never truncate older rows retained for later overnight runs. Configured
    provider aliases use separate caches and retain their source identity in
    returned history metadata; legacy caches from another symbol are not reused.
    """

    symbol = str(symbol).strip().upper()
    if not symbol:
        raise ValueError("symbol must not be empty")
    provider_symbol = _provider_aliases().get(symbol, symbol)
    requested_start = _requested_start(lookback_days, now)
    path = _cache_path(provider_symbol, interval)
    metadata_path = _cache_metadata_path(provider_symbol, interval)
    use_cache = os.getenv("MARKET_AGENT_DISABLE_OHLCV_CACHE", "").strip().lower() not in {"1", "true", "yes"}
    cached = _read_cached_ohlcv(path) if use_cache else pd.DataFrame()
    cache_metadata = _read_cache_metadata(metadata_path) if use_cache else {}
    if use_cache and not _cache_matches_provider(cache_metadata, symbol, provider_symbol):
        if not cached.empty:
            logger.warning(
                "Ignoring OHLCV cache with unverified provider identity: %s (%s)",
                symbol,
                provider_symbol,
            )
        cached = pd.DataFrame()
        cache_metadata = {}

    recorded_backfill_start = _metadata_timestamp(
        cache_metadata.get("earliest_backfill_requested_start")
    )
    cached_start = cached.index.min().normalize() if not cached.empty else None
    coverage_tolerance = pd.Timedelta(days=7)
    cache_needs_backfill = bool(
        not cached.empty
        and cached_start is not None
        and cached_start > requested_start + coverage_tolerance
        and (
            recorded_backfill_start is None
            or recorded_backfill_start > requested_start
        )
    )

    if use_cache and not cached.empty:
        if cache_needs_backfill:
            fetch_start = requested_start.strftime("%Y-%m-%d")
        else:
            fetch_start = (
                cached.index.max().to_pydatetime() - timedelta(days=7)
            ).strftime("%Y-%m-%d")
    else:
        fetch_start = requested_start.strftime("%Y-%m-%d")

    downloaded = _normalize_ohlcv(_download_ohlcv(provider_symbol, fetch_start, interval))

    if use_cache:
        if not downloaded.empty:
            if cached.empty:
                merged = downloaded.copy()
            else:
                merged = pd.concat([cached, downloaded]).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
            _write_cached_ohlcv(path, merged)
            earliest_requested = requested_start if cache_needs_backfill or cached.empty else recorded_backfill_start
            if recorded_backfill_start is not None and earliest_requested is not None:
                earliest_requested = min(recorded_backfill_start, earliest_requested)
            metadata_payload = {
                "schema_version": 2,
                "symbol": str(symbol).upper(),
                "provider": "yahoo_finance",
                "provider_symbol": provider_symbol,
                "interval": str(interval),
                "earliest_backfill_requested_start": (
                    earliest_requested.date().isoformat()
                    if earliest_requested is not None
                    else None
                ),
                "available_start": merged.index.min().date().isoformat(),
                "available_end": merged.index.max().date().isoformat(),
                "updated_at_utc": _normalized_utc_now(now).isoformat(),
            }
            _write_cache_metadata(metadata_path, metadata_payload)
            requested = _requested_window(merged, lookback_days, now)
            requested.attrs["history_coverage"] = _history_coverage_metadata(
                requested,
                merged,
                lookback_days=lookback_days,
                requested_start=requested_start,
                backfill_requested=bool(cache_needs_backfill or cached.empty),
                backfill_succeeded=True,
            )
            return _with_source_metadata(
                requested, symbol, provider_symbol, cache_fallback=False
            )
        if not cached.empty:
            logger.warning(
                "OHLCV refresh returned no data; using cache subject to freshness checks: %s (%s)",
                symbol,
                provider_symbol,
            )
            requested = _requested_window(cached, lookback_days, now)
            requested.attrs["history_coverage"] = _history_coverage_metadata(
                requested,
                cached,
                lookback_days=lookback_days,
                requested_start=requested_start,
                backfill_requested=cache_needs_backfill,
                backfill_succeeded=False,
            )
            return _with_source_metadata(
                requested, symbol, provider_symbol, cache_fallback=True
            )

    if downloaded.empty:
        provider_detail = f" (Yahoo symbol {provider_symbol})" if symbol != provider_symbol else ""
        raise ValueError(f"No OHLCV data returned for {symbol}{provider_detail}.")
    requested = _requested_window(downloaded, lookback_days, now)
    requested.attrs["history_coverage"] = _history_coverage_metadata(
        requested,
        downloaded,
        lookback_days=lookback_days,
        requested_start=requested_start,
        backfill_requested=True,
        backfill_succeeded=True,
    )
    return _with_source_metadata(
        requested, symbol, provider_symbol, cache_fallback=False
    )
