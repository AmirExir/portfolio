"""Explicit startup bootstrap for deployments without a persistent generation store."""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

from .config import IngestionConfig, SourceRoot, default_config
from .embeddings import EmbeddingProvider
from .pipeline import IngestionPipeline
from .retrieval import LoadedIndex, index_state, load_index
from .store import current_generation_id, load_manifest


class CentralIndexUnavailable(RuntimeError):
    """Raised when an active application cannot load a complete central index."""


_STARTUP_LOCK = threading.RLock()
_VALIDATED_GENERATIONS: dict[
    Path,
    tuple[str, tuple[tuple[str, str, str, str, tuple[str, ...]], ...]],
] = {}


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def startup_source_roots(config: IngestionConfig) -> tuple[SourceRoot, ...]:
    """Return bounded checked-in roots suitable for a one-time UI bootstrap.

    The downloaded official archive is intentionally excluded: scheduled
    ingestion owns that potentially large corpus. A fresh hosted checkout can
    still build the checked-in manuals and versioned official uploads without
    crawling or embedding a machine-local archive during UI startup. Every
    configured checked-in root is required so a packaging omission cannot
    silently fall back to whichever older corpus remains available.
    """

    roots: list[SourceRoot] = []
    seen: set[Path] = set()
    missing: list[Path] = []
    for source in config.source_roots:
        resolved = source.path.resolve(strict=False)
        if source.is_generated or source.name == "official_downloads":
            continue
        if not source.path.is_dir():
            missing.append(source.path)
            continue
        if resolved in seen:
            continue
        roots.append(source)
        seen.add(resolved)
    if missing:
        paths = ", ".join(str(path) for path in missing)
        raise CentralIndexUnavailable(
            f"Configured checked-in ERCOT source roots are missing: {paths}"
        )
    return tuple(roots)


def startup_index_state(
    config: IngestionConfig | None = None,
) -> tuple[str | None, int]:
    """Return a UI cache token with deployment errors normalized for display."""

    try:
        return index_state(config=config or default_config())
    except Exception as exc:
        raise CentralIndexUnavailable(
            f"Unable to read the central ERCOT index state: {type(exc).__name__}: {exc}"
        ) from exc


def _source_root_signature(
    roots: Iterable[SourceRoot],
) -> tuple[tuple[str, str, str, str, tuple[str, ...]], ...]:
    """Return the source configuration identity bound to process validation."""

    return tuple(
        sorted(
            (
                source.name,
                str(source.path.resolve(strict=False)),
                source.source_authority,
                source.default_source_kind,
                tuple(source.default_collections),
            )
            for source in roots
        )
    )


def _critical_errors(
    manifest: Mapping[str, Any],
    source_names: set[str],
) -> list[tuple[str, str]]:
    documents = manifest.get("documents", {})
    if not isinstance(documents, dict):
        return [("manifest", "documents registry is invalid")]
    return sorted(
        (
            str(path),
            str(record.get("error") or "unknown ingestion error"),
        )
        for path, record in documents.items()
        if isinstance(record, dict)
        and record.get("source_category") in source_names
        and record.get("status") == "error"
    )


def _manifest_chunk_count(manifest: Mapping[str, Any]) -> int:
    summary = manifest.get("summary", {})
    if isinstance(summary, Mapping) and "chunks" in summary:
        try:
            return max(0, int(summary.get("chunks", 0)))
        except (TypeError, ValueError):
            pass
    content = manifest.get("content", {})
    if not isinstance(content, Mapping):
        return 0
    return sum(
        len(record.get("chunk_ids", []))
        for record in content.values()
        if isinstance(record, Mapping)
    )


def _unrepresented_source_roots(
    manifest: Mapping[str, Any],
    roots: Iterable[SourceRoot],
) -> list[str]:
    """Return required roots without an active, retrievable document record."""

    documents = manifest.get("documents", {})
    if not isinstance(documents, Mapping):
        return sorted(source.name for source in roots)
    represented = {
        str(record.get("source_category") or "")
        for record in documents.values()
        if isinstance(record, Mapping)
        and record.get("status") in {"ingested", "duplicate"}
        and record.get("indexed_sha256")
    }
    return sorted(source.name for source in roots if source.name not in represented)


def ensure_central_generation(
    config: IngestionConfig | None = None,
    *,
    embedder: EmbeddingProvider | None = None,
    bootstrap_on_missing: bool | None = None,
    refresh: bool | None = None,
) -> dict[str, Any]:
    """Ensure checked-in authoritative sources have a complete central generation.

    This function is called explicitly by application startup code, never at
    module import. By default each process performs one bounded incremental
    refresh of checked-in sources and bootstraps when ``CURRENT`` is absent.
    Set ``ERCOT_RAG_STARTUP_REFRESH=false`` to skip that refresh when a complete
    generation already exists, or
    ``ERCOT_RAG_BOOTSTRAP_ON_MISSING=false`` to require a pre-provisioned store.
    """

    selected = config or default_config()
    should_bootstrap = (
        _env_flag("ERCOT_RAG_BOOTSTRAP_ON_MISSING", True)
        if bootstrap_on_missing is None
        else bootstrap_on_missing
    )
    should_refresh = (
        _env_flag("ERCOT_RAG_STARTUP_REFRESH", True) if refresh is None else refresh
    )
    force_refresh = refresh is True
    index_key = selected.index_dir.resolve(strict=False)

    with _STARTUP_LOCK:
        roots = startup_source_roots(selected)
        if not roots:
            raise CentralIndexUnavailable(
                "No checked-in authoritative ERCOT source roots are available for startup sync"
            )
        root_signature = _source_root_signature(roots)
        current = current_generation_id(selected.index_dir)
        if (
            current
            and _VALIDATED_GENERATIONS.get(index_key) == (current, root_signature)
            and not force_refresh
        ):
            return {"status": "current", "generation": current, "changed": False}

        source_names = {source.name for source in roots}

        existing = load_manifest(selected.index_dir, current) if current else None
        existing_manifest = existing[1] if existing else None
        existing_errors = (
            _critical_errors(existing_manifest, source_names)
            if existing_manifest is not None
            else []
        )
        existing_missing = (
            _unrepresented_source_roots(existing_manifest, roots)
            if existing_manifest is not None
            else sorted(source_names)
        )
        needs_update = (
            not current
            or should_refresh
            or bool(existing_errors)
            or bool(existing_missing)
        )
        if not current and not should_bootstrap:
            raise CentralIndexUnavailable(
                "No central ERCOT RAG generation is mounted and startup bootstrap is disabled. "
                "Provision ERCOT_RAG_STORE or enable ERCOT_RAG_BOOTSTRAP_ON_MISSING."
            )
        if existing_errors and not should_bootstrap and not should_refresh:
            preview = "; ".join(f"{path}: {error}" for path, error in existing_errors[:3])
            raise CentralIndexUnavailable(
                "The mounted central ERCOT generation has incomplete checked-in sources and "
                f"startup repair is disabled: {preview}"
            )

        result: dict[str, Any] = {
            "status": "current",
            "generation": current,
            "changed": False,
        }
        if needs_update:
            try:
                pipeline = IngestionPipeline(selected, embedder=embedder)
                result = pipeline.update([source.path for source in roots])
            except Exception as exc:
                raise CentralIndexUnavailable(
                    f"Central ERCOT startup sync failed: {type(exc).__name__}: {exc}"
                ) from exc

        active = load_manifest(selected.index_dir)
        if active is None:
            raise CentralIndexUnavailable(
                "Startup sync did not publish a usable central ERCOT index"
            )
        generation_id, manifest = active
        if _manifest_chunk_count(manifest) < 1:
            raise CentralIndexUnavailable(
                "Startup sync did not publish a usable central ERCOT index"
            )

        errors = _critical_errors(manifest, source_names)
        if errors:
            preview = "; ".join(f"{path}: {error}" for path, error in errors[:3])
            suffix = f"; and {len(errors) - 3} more" if len(errors) > 3 else ""
            raise CentralIndexUnavailable(
                "Central ERCOT startup sync is incomplete; refusing legacy or partial retrieval. "
                f"{preview}{suffix}"
            )
        missing_sources = _unrepresented_source_roots(manifest, roots)
        if missing_sources:
            raise CentralIndexUnavailable(
                "Central ERCOT startup sync is incomplete; required checked-in source roots "
                f"have no retrievable documents: {', '.join(missing_sources)}"
            )
        _VALIDATED_GENERATIONS[index_key] = (generation_id, root_signature)
        return result


def load_startup_index(
    collections: str | Iterable[str] | None = None,
    *,
    config: IngestionConfig | None = None,
    embedder: EmbeddingProvider | None = None,
    bootstrap_on_missing: bool | None = None,
    refresh: bool | None = None,
) -> LoadedIndex:
    """Load a central-only index, bootstrapping checked-in sources when needed."""

    selected = config or default_config()
    try:
        ensure_central_generation(
            selected,
            embedder=embedder,
            bootstrap_on_missing=bootstrap_on_missing,
            refresh=refresh,
        )
    except CentralIndexUnavailable:
        raise
    except Exception as exc:
        raise CentralIndexUnavailable(
            f"Unable to validate the central ERCOT index: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        loaded = load_index(collections, config=selected, allow_legacy=False)
    except Exception as exc:
        raise CentralIndexUnavailable(
            f"Unable to load the validated central ERCOT index: {type(exc).__name__}: {exc}"
        ) from exc
    if loaded.source != "central" or not loaded.ready:
        raise CentralIndexUnavailable(
            f"Central ERCOT collection {collections!r} is unavailable after startup sync"
        )
    return loaded
